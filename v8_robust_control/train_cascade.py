"""PPO curriculum for two-ball exchange and complete three-ball tasks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import shutil
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from .cascade_env import CascadeEnv, runtime_sources
from .compare_residual import read_json, digest
from .residual_env import UNSAFE_REASONS
from jugglebot.humanoid import cascade as C
from .train_transfer import CASES, MaskedTransferPolicy, TransferFeatures, Fixed, counts, dominates
from .train_residual import TrainingProgress, isolated_evaluation_rng


def sources():
    return dict(runtime_sources(), **{
        'v8_robust_control/train_transfer.py': digest(Path(__file__).with_name('train_transfer.py')),
        'v8_robust_control/train_cascade.py': digest(Path(__file__)),
        'v8_robust_control/train_cascade.py:CascadeFeatures':
            hashlib.sha256(inspect.getsource(CascadeFeatures).encode('utf-8')).hexdigest()})


class CascadeFeatures(TransferFeatures):
    """Scale measured reference errors without changing shared single-ball inputs."""
    def __init__(self, observation_space):
        super().__init__(observation_space)
        for offset in (0, 14, 28):
            self.scale[offset+6:offset+9] = 200.
            self.scale[offset+9:offset+12] = 50.


def runtime_compatible(recorded, current=None, *, subset=False):
    """Weights require unchanged environment and policy code, not a trainer CLI."""
    driver = 'v8_robust_control/train_cascade.py'
    expected = {name: value for name, value in recorded.items() if name != driver}
    actual = {name: value for name, value in (sources() if current is None else current).items() if name != driver}
    return all(actual.get(name) == value for name, value in expected.items()) if subset else actual == expected


def qualifies(value, nominal):
    return (value['clean'] >= nominal['clean'] and value['sensor_only'] >= nominal['sensor_only']
            and value['unsafe'] <= nominal['unsafe'] and value['perturbed'] >= nominal['perturbed'])


def score(trials, nominal):
    value = counts(trials)
    return (int(qualifies(value, nominal)),
            min(value['clean']-nominal['clean'], value['sensor_only']-nominal['sensor_only'], 0),
            -max(0, value['unsafe']-nominal['unsafe']), value['perturbed'], -value['unsafe'])


def selection_score(trials, nominal, manifest):
    if manifest.get('selection_protocol') != 'nominal_task_then_robustness':
        return score(trials, nominal)
    value = counts(trials)
    return (int(value['clean'] == nominal['clean']), value['clean'], int(qualifies(value, nominal)),
            -max(0, value['unsafe']-nominal['unsafe']), value['sensor_only']+value['perturbed'],
            value['perturbed'], -value['unsafe'])


class PhaseTimed:
    def __init__(self, actions, horizon):
        self.actions, self.horizon = np.asarray(actions), horizon

    def predict(self, obs, deterministic=True):
        index = min(len(self.actions)-1, max(0, round(float(obs[168])*CascadeEnv.TIME_SCALE_S/.02)))
        return self.actions[index].copy(), None


class Frozen:
    def __init__(self, model):
        self.model, self.action = model, None
        self.ball_count = model.juggling_ball_count

    def reset(self):
        self.action = None

    def predict(self, obs, deterministic=True):
        if self.action is None and np.all(obs[[14*ball+12 for ball in range(self.ball_count)]] > .5):
            self.action = self.model.predict(obs, deterministic=True)[0]
        return np.zeros(6) if self.action is None else self.action.copy(), None


class Randomized(gym.Wrapper):
    def __init__(self, env, cases):
        super().__init__(env)
        self.base_config = env.config
        self.cases = list(cases)

    def reset(self, *, seed=None, options=None):
        if seed is not None or not hasattr(self, 'conditions'):
            self.conditions = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1703,)))
        case = self.cases[int(self.conditions.integers(len(self.cases)))]
        self.env.config = replace(self.base_config, **CASES[case])
        observation, info = self.env.reset(seed=seed, options=options)
        return observation, dict(info, perception_case=case)


def evaluate(actor, reference_dir, seeds, fit=False, clean_only=False):
    env = CascadeEnv(reference_dir)
    base_config = env.config
    bins = int(np.ceil(base_config.horizon_s/.02))+1
    sums, samples = np.zeros((bins, 6)), np.zeros(bins, dtype=int)
    trials = []
    try:
        for case, settings in CASES.items():
            if clean_only and case != 'clean':
                continue
            env.config = replace(base_config, **settings)
            for seed in seeds:
                obs, _ = env.reset(seed=seed)
                if hasattr(actor, 'reset'):
                    actor.reset()
                while True:
                    action, _ = actor.predict(obs, deterministic=True)
                    if fit:
                        index = min(bins-1, max(0, round(float(obs[168])*CascadeEnv.TIME_SCALE_S/.02)))
                        sums[index] += np.clip(action, -1., 1.)
                        samples[index] += 1
                    obs, _, done, truncated, info = env.step(action)
                    if done or truncated:
                        break
                trials.append(dict(case=case, seed=seed, **info))
    finally:
        env.close()
    if fit:
        return trials, dict(actions=(sums/np.maximum(samples[:, None], 1)).tolist(),
                            sums=sums.tolist(), counts=samples.tolist(), horizon_s=base_config.horizon_s)
    return trials


def make_model(env, seed, warm=None, warm_dir=None):
    model = PPO(MaskedTransferPolicy, env, seed=seed, device='cpu', verbose=0,
                learning_rate=.0001, n_steps=256, batch_size=64, n_epochs=5,
                gamma=.99, ent_coef=0., target_kl=.01, use_sde=True, sde_sample_freq=4,
                policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh,
                    squash_output=True, log_std_init=-4.5, features_extractor_class=CascadeFeatures))
    model.juggling_observation_layout = CascadeEnv.OBSERVATION_LAYOUT
    model.juggling_action_layout = CascadeEnv.ACTION_LAYOUT
    provenance, retain_actor = None, False
    if warm is not None:
        row = next(row for row in warm['models'] if row['seed'] == seed)
        path = warm_dir/row['checkpoint']
        if digest(path) != row['checkpoint_sha256']:
            raise ValueError('Warm-start checkpoint checksum differs')
        with isolated_evaluation_rng():
            previous = PPO.load(path, device='cpu')
        if previous.observation_space.shape != (191,) or previous.action_space.shape != (6,):
            raise ValueError('Warm-start policy interface differs')
        if warm['manifest'].get('observation_layout') is not None and (
                getattr(previous, 'juggling_observation_layout', None) != warm['manifest']['observation_layout']
                or getattr(previous, 'juggling_action_layout', None) != warm['manifest']['action_layout']):
            raise ValueError('Warm-start policy layout differs from its archive')
        if previous.num_timesteps != row['selected_step'] or previous.seed != seed:
            raise ValueError('Warm-start checkpoint budget or training seed differs')
        feature_scale = model.policy.features_extractor.scale.clone()
        model.policy.load_state_dict(previous.policy.state_dict())
        model.policy.features_extractor.scale.copy_(feature_scale)
        previous_balls = warm['manifest'].get('ball_count', 1)
        same_layout = (warm['manifest'].get('observation_layout') == CascadeEnv.OBSERVATION_LAYOUT
                       and warm['manifest'].get('action_layout') == CascadeEnv.ACTION_LAYOUT)
        retain_actor = same_layout and previous_balls <= env.unwrapped.count
        with torch.no_grad():
            for network in (model.policy.mlp_extractor.policy_net, model.policy.mlp_extractor.value_net):
                linear = next(layer for layer in network if isinstance(layer, torch.nn.Linear))
                for missing in range(previous_balls, 3):
                    # An unused ball slot has age=2; keep its old constant
                    # contribution while making the newly active inputs neutral.
                    linear.bias.add_(2.*linear.weight[:, 14*missing+13])
                linear.weight[:, 14*previous_balls:42] = 0.
                if not same_layout:
                    changed = [index for offset in (0, 14, 28) for index in range(offset+6, offset+12)]+[168, 183, 184]
                    linear.weight[:, changed] = 0.
        provenance = dict(checkpoint_sha256=digest(path), selected_step=row['selected_step'],
                          source_sha256=warm['manifest']['source_sha256'], previous_ball_count=previous_balls,
                          actor_retained=retain_actor,
                          reference_error_features_initialized=not same_layout,
                          previous_robustness_qualified=warm['summary']['all_candidates_passed'] if 'summary' in warm else None,
                          transfer=('policy encoder, actor and exploration retained; value head reset' if retain_actor
                                    else 'network initialization; mean/value heads and exploration scale reset'))
    if not retain_actor:
        torch.nn.init.zeros_(model.policy.action_net.weight)
        torch.nn.init.zeros_(model.policy.action_net.bias)
    torch.nn.init.zeros_(model.policy.value_net.weight)
    torch.nn.init.zeros_(model.policy.value_net.bias)
    if not retain_actor:
        with torch.no_grad():
            model.policy.log_std.fill_(-4.5)
    return model, provenance


def train_one(task):
    seed, manifest, folder, warm, warm_dir, nominal = task
    torch.set_num_threads(1)
    env = Randomized(CascadeEnv(manifest['reference_dir']), manifest['training_cases'])
    curve, elapsed = [], 0.
    try:
        model, initialization = make_model(env, seed, warm, warm_dir)
        obs, _ = env.env.reset(seed=0)
        if initialization is None or not initialization['actor_retained']:
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], 0.)
        model.juggling_task = manifest['task']
        model.juggling_throws = manifest['requested_throws']
        model.juggling_ball_count = manifest['ball_count']
        model.juggling_source_sha256 = manifest['source_sha256']
        for step in manifest['checkpoints']:
            started = time.perf_counter()
            model.learn(step-model.num_timesteps, reset_num_timesteps=False,
                        callback=TrainingProgress(f"{manifest['task']}{manifest['requested_throws']}_seed{seed}"))
            elapsed += time.perf_counter()-started
            path = folder/f'ppo_seed{seed}_step{step}.zip'
            last_obs = np.asarray(model._last_obs).copy()
            expected = model.predict(last_obs, deterministic=True)[0]
            model.save(path)
            with isolated_evaluation_rng():
                loaded = PPO.load(path, device='cpu')
                np.testing.assert_array_equal(expected, loaded.predict(last_obs, deterministic=True)[0])
                trials = evaluate(loaded, manifest['reference_dir'], manifest['validation_seeds'])
            point = dict(step=step, checkpoint=path.name, checkpoint_sha256=digest(path),
                         trials=trials, score=list(selection_score(trials, nominal, manifest)))
            curve.append(point)
            print('[Cascade validation]', manifest['task'], manifest['requested_throws'], seed, step, counts(trials), flush=True)
        selected = max(curve, key=lambda row: (tuple(row['score']), -row['step']))
        path = folder/f'ppo_seed{seed}.zip'
        shutil.copyfile(folder/selected['checkpoint'], path)
        with isolated_evaluation_rng():
            actor = PPO.load(path, device='cpu')
            replayed, fitted = evaluate(actor, manifest['reference_dir'], manifest['validation_seeds'], fit=True)
            if replayed != selected['trials']:
                raise ValueError('Selected policy replay differs from validation')
            frozen = evaluate(Frozen(actor), manifest['reference_dir'], manifest['validation_seeds'])
            timed = evaluate(PhaseTimed(fitted['actions'], fitted['horizon_s']), manifest['reference_dir'], manifest['validation_seeds'])
        return dict(seed=seed, training_steps=model.num_timesteps, selected_step=selected['step'],
                    checkpoint=path.name, checkpoint_sha256=digest(path), save_load_verified=True,
                    initialization=initialization, training_wall_time_s=elapsed, curve=curve,
                    trials=selected['trials'], frozen_trials=frozen, timed_trials=timed, timed_fit=fitted)
    finally:
        env.close()


def validate_trials(trials, manifest, seeds):
    expected = {(case, seed) for case in CASES for seed in seeds}
    if len(trials) != len(expected) or {(row['case'], row['seed']) for row in trials} != expected:
        raise ValueError('Missing or duplicate cascade evaluation scenes')
    for row in trials:
        config = dict(manifest['task_config'], **CASES[row['case']])
        event_reward = 10. if row['is_success'] else -20. if row['reason'] in UNSAFE_REASONS else -10.
        if (row['task'] != manifest['task'] or row['environment_version'] != CascadeEnv.ENVIRONMENT_VERSION
                or row['config'] != config or row['requested_throws'] != manifest['requested_throws']
                or row['is_success'] != (row['reason'] == 'SUCCESS')
                or row['max_joint_offset_rad'] > config['max_joint_offset_rad']+1e-12
                or row['max_grip_offset_rad'] > config['max_grip_offset_rad']+1e-12
                or row['max_wrapping_offset_rad'] > CascadeEnv.MAX_WRAPPING_OFFSET_RAD+1e-12
                or row['wrapping_limit_rad'] != CascadeEnv.MAX_WRAPPING_OFFSET_RAD
                or row['wrapping_rate_limit_rad_s'] != CascadeEnv.MAX_WRAPPING_RATE_RAD_S
                or row['reward_sums']['event'] != event_reward or row['reward_sums']['shaping'] != 0.
                or abs(row['reward_sums']['progress']-CascadeEnv.CATCH_PROGRESS_REWARD*row['catches']/row['requested_throws']) > 1e-10
                or abs(sum(row['reward_sums'].values())-row['episode_return']) > 1e-8):
            raise ValueError('Different cascade task, configuration or motor/reward bounds')
        if row['is_success']:
            required = manifest['requested_throws']
            catches = [event for event in row['events'] if event['kind'] == 'catch']
            releases = [event for event in row['events'] if event['kind'] == 'release']
            if (row['releases'] != required or row['catches'] != required or row['initial_hold_s'] < .05-1e-10
                    or row['final_hold_s'] < (.8 if manifest['task'] == 'complete' else .2)-1e-10
                    or sorted(event['id'] for event in catches) != list(range(1, required+1))
                    or [event['id'] for event in releases] != list(range(1, required+1))
                    or row['max_ball_height_m'] >= 1.3 or row['failure_event'] is not None):
                raise ValueError('Unconfirmed complete cascade task')
            for event in releases:
                if event['vertical_speed_m_s'] <= 1.:
                    raise ValueError('Release did not have the required upward motion')
            for event in catches:
                if event.get('ending'):
                    if manifest['task'] != 'complete' or event['id'] != required or event['continuous_support_s'] < .8-1e-10:
                        raise ValueError('Invalid ending catch exception')
                elif ('th' not in event['supporting_digits'] or len(event['supporting_digits']) < 2
                      or event['time']-event['palm_touch_time'] < .070
                      or event['settled_relative_speed'] >= C.CAPTURE_RELATIVE_SPEED_RATIO*event['initial_relative_speed']
                      or event['settled_ball_speed'] >= event['incoming_ball_speed']):
                    raise ValueError('Catch did not preserve palm/thumb support and cushioning')


def summarize(record):
    manifest, models = record['manifest'], record['models']
    if (manifest['training_seeds'] != [0, 1, 2] or manifest['observation_size'] != 191
            or manifest['action_size'] != 6 or sorted(row['seed'] for row in models) != [0, 1, 2]
            or set(manifest['training_seeds']) & set(manifest['validation_seeds'])):
        raise ValueError('Incomplete or overlapping cascade training protocol')
    if (not manifest['checkpoints'] or manifest['checkpoints'] != sorted(set(manifest['checkpoints']))
            or any(step < 256 or step % 256 for step in manifest['checkpoints'])):
        raise ValueError('Invalid cascade training budgets')
    validate_trials(record['baseline'], manifest, manifest['validation_seeds'])
    nominal = counts(record['baseline'])
    for model in models:
        fit = model['timed_fit']
        actions, sums, fitted_counts = (np.asarray(fit[name]) for name in ('actions', 'sums', 'counts'))
        bins = int(np.ceil(manifest['task_config']['horizon_s']/.02))+1
        if (actions.shape != (bins, 6) or sums.shape != (bins, 6) or fitted_counts.shape != (bins,)
                or fit['horizon_s'] != manifest['task_config']['horizon_s']
                or not np.all(np.isfinite(np.r_[actions.ravel(), sums.ravel(), fitted_counts]))
                or np.any(fitted_counts < 0) or np.any(fitted_counts != np.floor(fitted_counts))
                or not np.allclose(actions, sums/np.maximum(fitted_counts[:, None], 1), rtol=0., atol=1e-12)):
            raise ValueError('Cascade time-sequence fit differs')
        for field in ('trials', 'frozen_trials', 'timed_trials'):
            validate_trials(model[field], manifest, manifest['validation_seeds'])
        if ([point['step'] for point in model['curve']] != manifest['checkpoints']
                or model['training_steps'] != manifest['checkpoints'][-1] or not model['save_load_verified']):
            raise ValueError('Cascade checkpoint budget or save/load evidence differs')
        for point in model['curve']:
            validate_trials(point['trials'], manifest, manifest['validation_seeds'])
            if point['score'] != list(selection_score(point['trials'], nominal, manifest)):
                raise ValueError('Wrong cascade checkpoint score')
        selected = max(model['curve'], key=lambda row: (tuple(row['score']), -row['step']))
        if (model['selected_step'] != selected['step'] or model['trials'] != selected['trials']
                or model['checkpoint_sha256'] != selected['checkpoint_sha256']):
            raise ValueError('Cascade policy not selected on validation only')
    best_timed = max(models, key=lambda row: selection_score(row['timed_trials'], nominal, manifest))
    gates = {str(row['seed']): qualifies(counts(row['trials']), nominal) for row in models}
    advantage = {str(row['seed']): all(dominates(counts(row['trials']), counts(reference)) for reference in
                 (record['baseline'], best_timed['timed_trials'], row['frozen_trials'])) for row in models}
    result = dict(baseline=nominal, learned={str(row['seed']): counts(row['trials']) for row in models},
                best_timed_seed=best_timed['seed'], timed=counts(best_timed['timed_trials']),
                task_and_safety_gates=gates, all_candidates_passed=all(gates.values()),
                feedback_advantage=advantage)
    if manifest.get('selection_protocol') == 'nominal_task_then_robustness':
        nominal_gates = {str(row['seed']): counts(row['trials'])['clean'] == len(manifest['validation_seeds']) for row in models}
        result.update(nominal_task_gates=nominal_gates, all_nominal_tasks_completed=all(nominal_gates.values()))
    return result


def test_one(task):
    candidate, training, folder, seeds = task
    torch.set_num_threads(1)
    if candidate['kind'] == 'zero':
        actor = Fixed()
    elif candidate['kind'] == 'timed':
        actor = PhaseTimed(candidate['actions'], candidate['horizon_s'])
    else:
        row = next(row for row in training['models'] if row['seed'] == candidate['seed'])
        path = folder/row['checkpoint']
        if digest(path) != row['checkpoint_sha256']:
            raise ValueError('Independent policy checkpoint differs')
        actor = PPO.load(path, device='cpu')
        manifest = training['manifest']
        if (actor.juggling_task != manifest['task'] or actor.juggling_throws != manifest['requested_throws']
                or actor.juggling_ball_count != manifest['ball_count']
                or actor.juggling_observation_layout != manifest['observation_layout']
                or actor.juggling_action_layout != manifest['action_layout']
                or actor.juggling_source_sha256 != manifest['source_sha256']
                or actor.num_timesteps != row['selected_step']):
            raise ValueError('Policy task, training source or budget differs')
        if candidate['kind'] == 'frozen':
            actor = Frozen(actor)
    trials = evaluate(actor, training['manifest']['reference_dir'], seeds)
    print('[Cascade independent]', candidate['id'], counts(trials), flush=True)
    return dict(**candidate, trials=trials)


def report(record):
    summary = summarize(record)
    if summary != record['summary']:
        raise ValueError('Cascade summary differs')
    manifest = record['manifest']
    n = len(manifest['validation_seeds'])
    lines = [f"# {manifest['task']}：{manifest['requested_throws']} 次完整抛接验证", '',
             '任务验收与 RL 相对对照的收益分别报告。策略必须保留无扰动完成情况、安全性及扰动完成数；额外反馈收益由时间序列和冻结对照判定。', '',
             '零误差时反馈增益保持名义轨迹；无扰动行用于兼容性检查，扰动行与对照用于检验学习收益。', '',
             '| 控制器 | 理想初态 | 仅感知噪声 | 物理初态扰动 | 安全失败 |', '| --- | --- | --- | --- | --- |']
    rows = [('zero', summary['baseline']), ('timed', summary['timed'])]
    rows += [(f'PPO seed {seed}', value) for seed, value in summary['learned'].items()]
    for name, value in rows:
        lines.append(f"| {name} | {value['clean']}/{n} | {value['sensor_only']}/{n} | {value['perturbed']}/{6*n} | {value['unsafe']}/{8*n} |")
    if 'nominal_task_gates' in summary:
        lines += ['', f"无扰动完整任务验收：{summary['nominal_task_gates']}。"]
    lines += ['', f"扰动非退化验收：{summary['task_and_safety_gates']}。",
              f"严格超过名义、时间序列和自身冻结对照：{summary['feedback_advantage']}。", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--reference-dir', type=Path, required=True)
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--warm-start-study', type=Path)
    run.add_argument('--warm-start-dir', type=Path)
    run.add_argument('--checkpoints', type=int, nargs='+', default=[1024, 4096, 8192])
    run.add_argument('--seed-start', type=int, default=380000)
    run.add_argument('--episodes', type=int, default=3)
    run.add_argument('--jobs', type=int, default=3)
    run.add_argument('--training-cases', nargs='+', choices=list(CASES),
                     default=['clean', 'sensor_only_3mm', 'seat_2mm', 'seat_2mm_noise_3mm'])
    export = commands.add_parser('report')
    export.add_argument('--input', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output')
    if args.command == 'report':
        text = report(read_json(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
        print(text)
        return
    if (args.output_dir.exists() or args.checkpoints != sorted(set(args.checkpoints))
            or not args.checkpoints or any(step < 256 or step % 256 for step in args.checkpoints)
            or args.seed_start < 380000 or args.episodes < 1 or args.jobs < 1
            or bool(args.warm_start_study) != bool(args.warm_start_dir)):
        parser.error('Use a new model directory, distinct 256-step budgets and paired warm-start arguments')
    fingerprints = sources()
    warm = None if args.warm_start_study is None else read_json(args.warm_start_study)
    if warm is not None:
        if not runtime_compatible(warm['manifest']['source_sha256'], fingerprints, subset=True):
            raise ValueError('Warm-start source no longer matches')
        if warm['manifest'].get('scope') == 'cascade_residual_validation':
            if (summarize(warm) != warm['summary'] or any(counts(row['trials'])['clean'] != len(warm['manifest']['validation_seeds'])
                    for row in warm['models'])):
                raise ValueError('Previous cascade stage did not complete every nominal task')
        elif warm['manifest'].get('scope') == 'complete_single_ball_residual_validation':
            from .train_transfer import summarize as summarize_single
            if summarize_single(warm) != warm['summary']:
                raise ValueError('Single-ball initialization archive differs')
        else:
            raise ValueError('Unsupported warm-start study')
    env = CascadeEnv(args.reference_dir)
    try:
        config = asdict(env.config)
        task, requested, ball_count = env.task, env.target_throws, env.count
        env.reset(seed=0)
        while True:
            _, _, done, _, zero = env.step(np.zeros(6))
            if done:
                break
        if not zero['is_success'] or zero['reference_state_max_error'] != 0.:
            raise ValueError('Nominal cascade does not reproduce a completed task')
    finally:
        env.close()
    manifest = dict(scope='cascade_residual_validation', task=task, requested_throws=requested, ball_count=ball_count,
        reference_dir=args.reference_dir.as_posix(), reference_sha256=digest(args.reference_dir/'reference.npz'),
        reference_manifest_sha256=digest(args.reference_dir/'manifest.json'), task_config=config,
        source_sha256=fingerprints, training_seeds=[0, 1, 2], validation_seeds=list(range(args.seed_start, args.seed_start+args.episodes)),
        checkpoints=args.checkpoints, training_cases=args.training_cases, observation_size=191, action_size=6,
        observation_layout=CascadeEnv.OBSERVATION_LAYOUT, action_layout=CascadeEnv.ACTION_LAYOUT,
        selection_protocol='nominal_task_then_robustness',
        warm_start_sha256=None if warm is None else digest(args.warm_start_study),
        optimizer=dict(algorithm='PPO', learning_rate=.0001, n_steps=256, batch_size=64, epochs=5,
                       target_kl=.01, fresh_log_std_init=-4.5), zero_residual_evidence=zero)
    manifest['versions'] = {name: importlib.metadata.version(name) for name in
                           ('mujoco', 'numpy', 'torch', 'gymnasium', 'stable-baselines3')}
    baseline = evaluate(Fixed(), args.reference_dir, manifest['validation_seeds'])
    print('[Cascade baseline]', task, requested, counts(baseline), flush=True)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    with ProcessPoolExecutor(args.jobs) as pool:
        models = list(pool.map(train_one, [(seed, manifest, args.output_dir, warm, args.warm_start_dir, counts(baseline))
                                          for seed in manifest['training_seeds']]))
    if fingerprints != sources():
        raise ValueError('Cascade source changed during training')
    record = dict(manifest=manifest, baseline=baseline, models=models)
    record['summary'] = summarize(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n', encoding='utf-8')
    print('[Cascade qualification]', record['summary'], flush=True)


if __name__ == '__main__':
    from v8_robust_control.train_cascade import main as module_main
    module_main()

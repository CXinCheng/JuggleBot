"""Train/evaluate bounded residual PPO on a complete physical ball transfer."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import hashlib
import importlib.metadata
from pathlib import Path
import shutil
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .compare_residual import digest, read_json
from .residual_env import UNSAFE_REASONS
from .train_residual import isolated_evaluation_rng, TrainingProgress
from .transfer_env import TransferEnv, TransferConfig, runtime_sources


CASES = {
    'clean': {},
    'sensor_only_3mm': dict(noise_std_m=.003),
    'seat_2mm': dict(seat_jitter_m=.002),
    'seat_2mm_noise_3mm': dict(seat_jitter_m=.002, noise_std_m=.003),
    'seat_5mm': dict(seat_jitter_m=.005),
    'seat_5mm_noise_3mm': dict(seat_jitter_m=.005, noise_std_m=.003),
    'seat_2mm_delay20': dict(seat_jitter_m=.002, delay_s=.02),
    'combined': dict(seat_jitter_m=.002, noise_std_m=.003, delay_s=.02, dropout=.1),
}
TIME_INDEX = 168


def sources():
    return dict(runtime_sources(), **{'v8_robust_control/train_transfer.py': digest(Path(__file__))})


def counts(trials):
    return dict(clean=sum(t['is_success'] for t in trials if t['case'] == 'clean'),
                sensor_only=sum(t['is_success'] for t in trials if t['case'] == 'sensor_only_3mm'),
                perturbed=sum(t['is_success'] for t in trials if t['case'] not in ('clean', 'sensor_only_3mm')),
                unsafe=sum(t['reason'] in UNSAFE_REASONS for t in trials))


def score(trials, baseline):
    value = counts(trials)
    return (int(value['clean'] >= baseline['clean'] and value['sensor_only'] >= baseline['sensor_only']
                and value['unsafe'] <= baseline['unsafe']),
            min(value['clean']-baseline['clean'], value['sensor_only']-baseline['sensor_only'], 0),
            -max(0, value['unsafe']-baseline['unsafe']), value['perturbed'], -value['unsafe'])


def dominates(value, reference):
    return (value['clean'] >= reference['clean'] and value['sensor_only'] >= reference['sensor_only']
            and value['unsafe'] <= reference['unsafe'] and value['perturbed'] > reference['perturbed'])


class TransferFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=TransferEnv.OBSERVATION_SIZE)
        if observation_space.shape != (191,):
            raise ValueError('Transfer observation layout differs')
        scale = torch.ones(191)
        for offset in (0, 14, 28):
            scale[offset:offset+3] = 20.
            scale[offset+6:offset+9] = 20.
        self.register_buffer('scale', scale)

    def forward(self, observation):
        return torch.clamp(observation*self.scale, -10., 10.)


class MaskedTransferPolicy(ActorCriticPolicy):
    def _latents(self, observation):
        return self.mlp_extractor(self.extract_features(observation))

    @staticmethod
    def log_prob(distribution, actions, mask):
        gaussian = distribution.bijector.inverse(actions)
        terms = distribution.distribution.log_prob(gaussian)-distribution.bijector.log_prob_correction(gaussian)
        return (terms*mask).sum(-1)

    def forward(self, observation, deterministic=False):
        latent, critic = self._latents(observation)
        distribution = self._get_action_dist_from_latent(latent)
        mask = (observation[..., -6:] > .5).to(observation.dtype)
        actions = distribution.get_actions(deterministic=deterministic)*mask
        return actions, self.value_net(critic), self.log_prob(distribution, actions, mask)

    def evaluate_actions(self, observation, actions):
        latent, critic = self._latents(observation)
        distribution = self._get_action_dist_from_latent(latent)
        mask = (observation[..., -6:] > .5).to(observation.dtype)
        return self.value_net(critic), self.log_prob(distribution, actions, mask), None

    def _predict(self, observation, deterministic=False):
        return self.forward(observation, deterministic)[0]


class RandomizedTransfer(gym.Wrapper):
    def reset(self, *, seed=None, options=None):
        if seed is not None or not hasattr(self, 'conditions'):
            self.conditions = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1703,)))
        case = list(CASES)[int(self.conditions.integers(len(CASES)))]
        self.env.config = TransferConfig(**CASES[case])
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, dict(info, perception_case=case)


class Fixed:
    def __init__(self, action=None):
        self.action = np.zeros(6) if action is None else np.asarray(action, dtype=float)

    def reset(self):
        pass

    def predict(self, observation, deterministic=True):
        return self.action.copy(), None


class Frozen(Fixed):
    def __init__(self, model):
        self.model = model
        self.action = None

    def reset(self):
        self.action = None

    def predict(self, observation, deterministic=True):
        if self.action is None and observation[12] > .5:
            self.action = self.model.predict(observation, deterministic=True)[0]
        return np.zeros(6) if self.action is None else self.action.copy(), None


class Timed(Fixed):
    def __init__(self, actions):
        self.actions = np.asarray(actions)

    def predict(self, observation, deterministic=True):
        index = min(len(self.actions)-1, max(0, round(float(observation[TIME_INDEX])*1.6/.02)))
        return self.actions[index].copy(), None


def evaluate(actor, reference_dir, seeds, fit_schedule=False):
    trials = []
    sums = np.zeros((80, 6))
    samples = np.zeros(80, dtype=int)
    env = TransferEnv(reference_dir)
    try:
        for case, settings in CASES.items():
            env.config = TransferConfig(**settings)
            for seed in seeds:
                obs, _ = env.reset(seed=seed)
                if hasattr(actor, 'reset'):
                    actor.reset()
                while True:
                    action, _ = actor.predict(obs, deterministic=True)
                    index = min(79, max(0, round(float(obs[TIME_INDEX])*1.6/.02)))
                    if fit_schedule:
                        sums[index] += np.clip(action, -1., 1.)
                        samples[index] += 1
                    obs, _, done, truncated, info = env.step(action)
                    if done or truncated:
                        break
                trials.append(dict(case=case, seed=seed, **info))
    finally:
        env.close()
    if fit_schedule:
        fitted = sums/np.maximum(samples[:, None], 1)
        return trials, dict(actions=fitted.tolist(), sums=sums.tolist(), counts=samples.tolist())
    return trials


def fixed_candidates():
    values = [dict(id='zero', action=[0.]*6)]
    for axis in range(6):
        for amplitude in (-.1, .1):
            action = np.zeros(6)
            action[axis] = amplitude
            values.append(dict(id=f'axis{axis}_{amplitude:g}', action=action.tolist()))
    return values


def reference_task(task):
    candidate, manifest = task
    torch.set_num_threads(1)
    trials = evaluate(Fixed(candidate['action']), manifest['reference_dir'], manifest['validation_seeds'])
    print('[Transfer fixed]', candidate['id'], counts(trials), flush=True)
    return dict(**candidate, trials=trials)


def create_model(env, seed):
    model = PPO(MaskedTransferPolicy, env, learning_rate=.0003, seed=seed, device='cpu', verbose=0,
                n_steps=128, batch_size=64, n_epochs=5, gamma=.99, ent_coef=0., target_kl=.03,
                use_sde=True, sde_sample_freq=4,
                policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh, squash_output=True,
                                   log_std_init=-3., features_extractor_class=TransferFeatures))
    torch.nn.init.zeros_(model.policy.action_net.weight)
    torch.nn.init.zeros_(model.policy.action_net.bias)
    return model


def train_one(task):
    seed, manifest, folder, nominal = task
    torch.set_num_threads(1)
    env = RandomizedTransfer(TransferEnv(manifest['reference_dir']))
    points = []
    elapsed = 0.
    try:
        model = create_model(env, seed)
        progress = TrainingProgress(f'transfer_ppo_seed{seed}')
        for step in manifest['checkpoints']:
            started = time.perf_counter()
            model.learn(step-model.num_timesteps, reset_num_timesteps=False, callback=progress)
            elapsed += time.perf_counter()-started
            path = folder/f'ppo_seed{seed}_step{step}.zip'
            observation = np.asarray(model._last_obs).copy()
            expected = model.predict(observation, deterministic=True)[0]
            model.save(path)
            with isolated_evaluation_rng():
                loaded = PPO.load(path, device='cpu')
                np.testing.assert_array_equal(expected, loaded.predict(observation, deterministic=True)[0])
                trials = evaluate(loaded, manifest['reference_dir'], manifest['validation_seeds'])
            points.append(dict(step=step, checkpoint=path.name, checkpoint_sha256=digest(path), trials=trials,
                               score=list(score(trials, nominal))))
            print('[Transfer PPO validation]', seed, step, counts(trials), flush=True)
        selected = max(points, key=lambda point: (tuple(point['score']), -point['step']))
        path = folder/f'ppo_seed{seed}.zip'
        shutil.copyfile(folder/selected['checkpoint'], path)
        actor = PPO.load(path, device='cpu')
        with isolated_evaluation_rng():
            replayed, fitted = evaluate(actor, manifest['reference_dir'], manifest['validation_seeds'], fit_schedule=True)
            assert [(t['case'], t['seed'], t['reason']) for t in replayed] == [(t['case'], t['seed'], t['reason']) for t in selected['trials']]
            frozen = evaluate(Frozen(actor), manifest['reference_dir'], manifest['validation_seeds'])
            timed = evaluate(Timed(fitted['actions']), manifest['reference_dir'], manifest['validation_seeds'])
        return dict(seed=seed, training_steps=model.num_timesteps, selected_step=selected['step'],
                    checkpoint=path.name, checkpoint_sha256=digest(path), save_load_verified=True,
                    training_wall_time_s=elapsed, curve=points, trials=selected['trials'],
                    frozen_trials=frozen, timed_trials=timed, timed_actions=fitted['actions'], timed_fit=fitted)
    finally:
        env.close()


def validate_trials(trials, seeds):
    expected = {(case, seed) for case in CASES for seed in seeds}
    if len(trials) != len(expected) or {(t['case'], t['seed']) for t in trials} != expected:
        raise ValueError('Incomplete or duplicate transfer scenes')
    for trial in trials:
        if trial['environment_version'] != TransferEnv.ENVIRONMENT_VERSION or trial['task'] != 'single_ball_transfer':
            raise ValueError('Different transfer runtime')
        if trial['config'] != asdict(TransferConfig(**CASES[trial['case']])):
            raise ValueError('Transfer case uses different physical or sensor settings')
        if trial['is_success'] != (trial['reason'] == 'SUCCESS'):
            raise ValueError('Success/reason mismatch')
        if trial['is_success'] and (trial['initial_hold_max_s'] < .05-1e-10 or trial['final_hold_s'] < .2-1e-10
                                    or trial['airborne_s'] < .3 or trial['release_time_s'] is None
                                    or trial['palm_touch_time_s'] is None
                                    or not .4 <= trial['palm_touch_time_s']-trial['release_time_s'] <= .75):
            raise ValueError('Unverified full transfer success')
        event = 10. if trial['is_success'] else -20. if trial['reason'] in UNSAFE_REASONS else -10.
        if (trial['reward_sums']['event'] != event
                or abs(sum(trial['reward_sums'].values())-trial['episode_return']) > 1e-8
                or np.max(np.abs(trial['joint_offset_rad'])) > .04+1e-12
                or np.max(np.abs(trial['grip_offset_rad'])) > .15+1e-12
                or trial['max_joint_offset_rad'] > .04+1e-12 or trial['max_grip_offset_rad'] > .15+1e-12
                or np.max(np.abs(trial['initial_displacement_m'])) > trial['config']['seat_jitter_m']+1e-12
                or trial['is_success'] and trial['max_ball_height_m'] >= 1.3):
            raise ValueError('Invalid transfer reward or motor bound')


def summarize(record):
    manifest, refs, models = (record[key] for key in ('manifest', 'references', 'models'))
    if (manifest['cases'] != CASES or manifest['training_seeds'] != [0, 1, 2]
            or not manifest['validation_seeds'] or len(set(manifest['validation_seeds'])) != len(manifest['validation_seeds'])
            or set(manifest['validation_seeds']) & set(manifest['training_seeds'])
            or manifest['observation_size'] != 191 or manifest['action_size'] != 6
            or manifest['task_config'] != asdict(TransferConfig())):
        raise ValueError('Different transfer study protocol')
    if [{key: row[key] for key in ('id', 'action')} for row in refs] != fixed_candidates():
        raise ValueError('Missing fixed transfer control')
    if sorted(row['seed'] for row in models) != manifest['training_seeds']:
        raise ValueError('Incomplete transfer policies')
    for row in refs:
        validate_trials(row['trials'], manifest['validation_seeds'])
    nominal = counts(refs[0]['trials'])
    fixed = max(refs, key=lambda row: score(row['trials'], nominal))
    for row in models:
        fit = row['timed_fit']
        actions, sums, fitted_counts = (np.asarray(fit[name]) for name in ('actions', 'sums', 'counts'))
        if (actions.shape != (80, 6) or sums.shape != (80, 6) or fitted_counts.shape != (80,)
                or not np.all(np.isfinite(np.r_[actions.ravel(), sums.ravel(), fitted_counts]))
                or np.any(fitted_counts < 0) or np.any(fitted_counts != np.floor(fitted_counts))
                or not np.allclose(actions, sums/np.maximum(fitted_counts[:, None], 1), rtol=0., atol=1e-12)
                or row['timed_actions'] != fit['actions']):
            raise ValueError('Timed control was not fitted to validation requests')
        for field in ('trials', 'frozen_trials', 'timed_trials'):
            validate_trials(row[field], manifest['validation_seeds'])
        if ([point['step'] for point in row['curve']] != manifest['checkpoints'] or not row['save_load_verified']
                or row['training_steps'] != manifest['checkpoints'][-1]):
            raise ValueError('Missing checkpoint or save/load evidence')
        for point in row['curve']:
            validate_trials(point['trials'], manifest['validation_seeds'])
            if point['score'] != list(score(point['trials'], nominal)):
                raise ValueError('Wrong checkpoint score')
        selected = max(row['curve'], key=lambda point: (tuple(point['score']), -point['step']))
        if row['selected_step'] != selected['step'] or row['trials'] != selected['trials'] or row['checkpoint_sha256'] != selected['checkpoint_sha256']:
            raise ValueError('Policy not selected on validation only')
    timed = max(models, key=lambda row: score(row['timed_trials'], nominal))
    benchmarks = [nominal, counts(fixed['trials']), counts(timed['timed_trials'])]
    gates = {str(row['seed']): all(dominates(counts(row['trials']), ref) for ref in
                                  [*benchmarks, counts(row['frozen_trials'])]) for row in models}
    return dict(best_fixed=fixed['id'], best_timed_seed=timed['seed'], candidate_gates=gates,
                all_candidates_passed=all(gates.values()), baseline=nominal,
                fixed=counts(fixed['trials']), timed=counts(timed['timed_trials']),
                learned={str(row['seed']): counts(row['trials']) for row in models})


def test_task(task):
    candidate, training, folder, seeds = task
    torch.set_num_threads(1)
    if candidate['kind'] == 'fixed':
        actor = Fixed(candidate['action'])
    elif candidate['kind'] == 'timed':
        actor = Timed(candidate['actions'])
    else:
        row = next(row for row in training['models'] if row['seed'] == candidate['seed'])
        path = folder/row['checkpoint']
        if digest(path) != row['checkpoint_sha256']:
            raise ValueError('Policy checkpoint differs from validation')
        actor = PPO.load(path, device='cpu')
        if (actor.observation_space.shape != (191,) or actor.action_space.shape != (6,)
                or actor.num_timesteps != row['selected_step']):
            raise ValueError('Policy interface or training budget differs')
        if candidate['kind'] == 'frozen':
            actor = Frozen(actor)
    trials = evaluate(actor, training['manifest']['reference_dir'], seeds)
    print('[Transfer independent]', candidate['id'], counts(trials), flush=True)
    return dict(**candidate, trials=trials)


def summarize_test(record, training):
    selected = summarize(training)
    if not selected['all_candidates_passed']:
        raise ValueError('Independent test requires passing validation candidates')
    seeds = record['seeds']
    if (not seeds or len(set(seeds)) != len(seeds)
            or set(seeds) & set(training['manifest']['training_seeds']+training['manifest']['validation_seeds'])
            or record['source_sha256'] != training['manifest']['source_sha256']):
        raise ValueError('Different runtime or overlapping independent seeds')
    expected = ['zero', 'fixed', 'timed']+[f'{kind}_seed{seed}' for seed in (0, 1, 2) for kind in ('live', 'frozen')]
    if [row['id'] for row in record['records']] != expected:
        raise ValueError('Incomplete independent controls or policies')
    values = {}
    for row in record['records']:
        validate_trials(row['trials'], seeds)
        values[row['id']] = counts(row['trials'])
    gates = {str(seed): all(dominates(values[f'live_seed{seed}'], values[ref]) for ref in
                           ['zero', 'fixed', 'timed', f'frozen_seed{seed}']) for seed in (0, 1, 2)}
    return dict(counts=values, candidate_gates=gates, all_candidates_passed=all(gates.values()),
                scope='single ball only; two/three-ball and complete startup/shutdown not evaluated')


def report(record, test=None):
    summary = summarize(record)
    if summary != record['summary']:
        raise ValueError('Transfer summary differs')
    n = len(record['manifest']['validation_seeds'])
    lines = ['# Full Single-Ball Residual PPO Validation', '',
             'The task starts with the ball held in the right hand, executes a physical release, and holds it stably in the left hand for 0.2 s. Physical initial-state offsets are applied only at reset; perception disturbances affect position measurements independently.',
             'Nominal motor references are generated offline, and training integrates physics continuously without online planning rollouts. Zero residual reproduces the nominal trajectory under clean initial conditions.',
             'Fixed and per-step mean controls are selected on validation only; frozen output waits for the first valid position observation. This report is not an independent test conclusion.', '',
             '| Controller | Clean | Perception noise only | Physical seat jitter & combined | Safety failures |', '| --- | --- | --- | --- | --- |']
    rows = [('zero', summary['baseline']), ('fixed', summary['fixed']), ('timed', summary['timed'])]
    rows += [(f'PPO seed {seed}', value) for seed, value in summary['learned'].items()]
    for name, value in rows:
        lines.append(f"| {name} | {value['clean']}/{n} | {value['sensor_only']}/{n} | {value['perturbed']}/{6*n} | {value['unsafe']}/{8*n} |")
    lines += ['', f"Candidate gates: {summary['candidate_gates']}.",
              'Each training seed must preserve clean, perception-only, and safety metrics while strictly exceeding zero-residual, fixed, per-step mean, and own-frozen controls on disturbed completions.',
              'Only after all seeds pass are fresh independent seeds evaluated; failure does not support scaling to two- or three-ball policies.', '']
    if test is not None:
        raw = (json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n').encode('utf-8')
        if test['training_sha256'] not in {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest()}:
            raise ValueError('Independent test refers to a different training archive')
        checked = summarize_test(test, record)
        if checked != test['summary']:
            raise ValueError('Independent transfer summary differs')
        n = len(test['seeds'])
        lines += ['## Independent Test', '', '| Controller | Clean | Perception noise only | Physical seat jitter & combined | Safety failures |',
                  '| --- | --- | --- | --- | --- |']
        for name, value in checked['counts'].items():
            lines.append(f"| {name} | {value['clean']}/{n} | {value['sensor_only']}/{n} | {value['perturbed']}/{6*n} | {value['unsafe']}/{8*n} |")
        lines += ['', f"Independent acceptance: {checked['candidate_gates']}.", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--reference-dir', type=Path, required=True)
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--checkpoints', type=int, nargs='+', default=[2048, 8192])
    run.add_argument('--seed-start', type=int, default=380000)
    run.add_argument('--episodes', type=int, default=5)
    run.add_argument('--jobs', type=int, default=3)
    export = commands.add_parser('report')
    export.add_argument('--input', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--test', type=Path)
    test = commands.add_parser('test')
    test.add_argument('--training', type=Path, required=True)
    test.add_argument('--input-dir', type=Path, required=True)
    test.add_argument('--seed-start', type=int, default=390000)
    test.add_argument('--episodes', type=int, default=20)
    test.add_argument('--jobs', type=int, default=3)
    test.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output')
    if args.command == 'report':
        text = report(read_json(args.input), None if args.test is None else read_json(args.test))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8', newline='\n')
        print(text)
        return
    if args.command == 'test':
        training = read_json(args.training)
        selection = summarize(training)
        if (not selection['all_candidates_passed'] or selection != training['summary']
                or training['manifest']['source_sha256'] != sources()
                or args.seed_start < 390000 or args.episodes < 1 or args.jobs < 1):
            raise ValueError('Require passing current-source candidates and fresh independent seeds')
        fixed = next(row for row in training['references'] if row['id'] == selection['best_fixed'])
        timed = next(row for row in training['models'] if row['seed'] == selection['best_timed_seed'])
        candidates = [dict(id='zero', kind='fixed', action=[0.]*6), dict(id='fixed', kind='fixed', action=fixed['action']),
                      dict(id='timed', kind='timed', actions=timed['timed_actions'])]
        candidates += [dict(id=f'{kind}_seed{seed}', kind=kind, seed=seed) for seed in (0, 1, 2) for kind in ('live', 'frozen')]
        seeds = list(range(args.seed_start, args.seed_start+args.episodes))
        if set(seeds) & set(training['manifest']['validation_seeds']):
            raise ValueError('Independent seeds overlap validation')
        with ProcessPoolExecutor(args.jobs) as pool:
            rows = list(pool.map(test_task, [(candidate, training, args.input_dir, seeds) for candidate in candidates]))
        if sources() != training['manifest']['source_sha256']:
            raise ValueError('Transfer source changed during independent testing')
        result = dict(training_sha256=digest(args.training), source_sha256=sources(), seeds=seeds, records=rows)
        result['summary'] = summarize_test(result, training)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, allow_nan=False, separators=(',', ':'))+'\n', encoding='utf-8', newline='\n')
        return
    if (args.output_dir.exists() or args.seed_start < 380000 or args.episodes < 1 or args.jobs < 1
            or not args.checkpoints or args.checkpoints != sorted(set(args.checkpoints))
            or any(step < 128 or step % 128 for step in args.checkpoints)):
        parser.error('Use a new model directory, distinct 128-step budgets and positive development counts')
    args.output_dir.mkdir(parents=True)
    nominal_env = TransferEnv(args.reference_dir)
    try:
        nominal_env.reset(seed=0)
        while True:
            _, _, done, truncated, zero_info = nominal_env.step(np.zeros(6))
            if done or truncated:
                break
        if not zero_info['is_success'] or zero_info['reference_state_max_error'] != 0.:
            raise ValueError('Zero residual does not reproduce and complete the nominal reference')
    finally:
        nominal_env.close()
    manifest = dict(scope='complete_single_ball_residual_validation', cases=CASES, training_seeds=[0, 1, 2],
                    validation_seeds=list(range(args.seed_start, args.seed_start+args.episodes)), checkpoints=args.checkpoints,
                    reference_dir=args.reference_dir.as_posix(), source_sha256=sources(),
                    reference_manifest_sha256=digest(args.reference_dir/'manifest.json'),
                    observation_size=191, action_size=6, task_config=asdict(TransferConfig()),
                    zero_residual_evidence=zero_info,
                    versions={name: importlib.metadata.version(name) for name in ('mujoco', 'numpy', 'torch', 'gymnasium', 'stable-baselines3')},
                    action='two XY palm requests and two grip synergies; observed permissions and physical rate limits',
                    physical_variation='initial seated ball XY only, up to +/-5mm; never changes a live flight')
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
    with ProcessPoolExecutor(args.jobs) as pool:
        refs = list(pool.map(reference_task, [(candidate, manifest) for candidate in fixed_candidates()]))
    (args.output_dir/'references.json').write_text(json.dumps(refs, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    nominal = counts(refs[0]['trials'])
    with ProcessPoolExecutor(args.jobs) as pool:
        models = list(pool.map(train_one, [(seed, manifest, args.output_dir, nominal) for seed in manifest['training_seeds']]))
    if sources() != manifest['source_sha256']:
        raise ValueError('Transfer runtime changed during training')
    record = dict(manifest=manifest, references=refs, models=models)
    record['summary'] = summarize(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
    print('[Transfer screening]', record['summary'], flush=True)


if __name__ == '__main__':
    # Workers and saved policies must reference the importable module even
    # when this file is launched with python -m.
    from v8_robust_control.train_transfer import main as module_main
    module_main()

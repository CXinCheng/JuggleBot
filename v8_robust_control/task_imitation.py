"""Initialize a bounded PPO actor from verified multi-recovery demonstrations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .compare_residual import digest
from .task_env import TaskConfig, TaskPolicy, TaskResidualEnv, runtime_sources, action_authority, context_key
from .task_demonstrations import (EventStudent, fit_event_tables,
                                   encode_command, TaskRequest, TaskCommand, replay_trial, VERIFIED_DEMONSTRATION_MODES)
from .train_task import TaskCurriculum, INITIAL_ACTION_STD_BOUNDS, REWARD_DISCOUNT, write_new


class NormalizedMeasurements(BaseFeaturesExtractor):
    """Frozen training-only statistics, serialized with the PPO policy."""
    def __init__(self, observation_space, center, scale):
        self.keys = tuple(observation_space.spaces)
        size = sum(int(np.prod(space.shape)) for space in observation_space.spaces.values())
        super().__init__(observation_space, features_dim=size)
        center, scale = np.asarray(center, np.float32), np.asarray(scale, np.float32)
        if (center.shape != (size,) or scale.shape != (size,) or np.any(scale <= 0.)
                or not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale))):
            raise ValueError('Require finite per-feature training statistics and positive scales')
        self.register_buffer('center', torch.as_tensor(center))
        self.register_buffer('scale', torch.as_tensor(scale))

    def forward(self, observations):
        values = torch.cat([observations[name].flatten(1) for name in self.keys], dim=1)
        return torch.clamp((values-self.center)/self.scale, -10., 10.)


class ImitationPolicy(TaskPolicy):
    """Keep untrained later events inactive during this initialization pilot."""
    def __init__(self, *args, control_releases=6, **kwargs):
        if type(control_releases) is not int or not 1 <= control_releases <= TaskConfig().throws:
            raise ValueError('Invalid imitation event window')
        self.control_releases = control_releases
        super().__init__(*args, **kwargs)

    def active(self, obs):
        inside = torch.round(obs['context'][..., 5]*TaskConfig().throws) < self.control_releases
        return super().active(obs)*inside.to(obs['context'].dtype)


def load_demonstrations(folders, reference, goal, control_releases):
    rows, inputs, seen, successful_seeds = [], {}, set(), set()
    for folder in folders:
        report_path, protocol_path = folder/'report.json', folder/'protocol.json'
        record = json.loads(report_path.read_text(encoding='utf-8'))
        protocol = record['protocol']
        archive = folder/'driver_source.py'
        source = archive if archive.exists() else Path(__file__).with_name('task_demonstrations.py')
        if (protocol != json.loads(protocol_path.read_text(encoding='utf-8'))
                or protocol['source_sha256'] != runtime_sources()
                or protocol['reference_sha256'] != digest(reference)
                or protocol['driver_sha256'] != digest(source)
                or protocol['task_config'] != asdict(TaskConfig())
                or protocol['goal'] != goal or protocol['control_releases'] != control_releases):
            raise ValueError('Require distinct, compatible demonstration episodes for the requested goal/window')
        seen.update(protocol['seeds'])
        for row in record['records']:
            if row['mode'] not in VERIFIED_DEMONSTRATION_MODES or not row['result'].get('training_prefix'):
                continue
            if row['seed'] in successful_seeds:
                raise ValueError('Do not count repeated successful initial states as independent demonstrations')
            if (row['goal'] != goal or row['control_releases'] != control_releases
                    or row['result']['is_success'] or row['result']['catches'] < goal
                    or row['jitter_m'] != protocol['jitter_mm']/1000.
                    or not np.isfinite(row['jitter_m']) or not 0. <= row['jitter_m'] <= .01
                    or not set(range(1, goal+1)).issubset(row['result']['recovered_ids'])):
                raise ValueError('Missing continuous recovered-flight evidence')
            for item in row['demonstrations']:
                if item['status'] != 'accepted':
                    raise ValueError('Successful episodes may contain only executed labels')
                np.testing.assert_allclose(encode_command(TaskRequest(**item['request']),
                    TaskCommand(**item['command'])), item['action'], rtol=0., atol=1e-12)
            rows.append(row)
            successful_seeds.add(row['seed'])
        for path in (report_path, protocol_path, source):
            inputs[path.as_posix()] = digest(path)
    return rows, inputs, seen


def validate_episode_splits(validation_seeds, test_seeds, all_source_seeds):
    if (not validation_seeds or not test_seeds or len(set(validation_seeds)) != len(validation_seeds)
            or len(set(test_seeds)) != len(test_seeds) or min(validation_seeds+test_seeds) < 0
            or set(test_seeds) & set(all_source_seeds) or set(validation_seeds)&set(all_source_seeds)
            or set(validation_seeds)&set(test_seeds)):
        raise ValueError('Require whole-episode validation and fresh independent test seeds')


def tensors(rows, observation_space):
    keys = tuple(observation_space.spaces)
    observations, targets, weights = {key:[] for key in keys}, [], []
    for row in rows:
        items = row['demonstrations']
        if not items:
            raise ValueError('An episode has no demonstration actions')
        for item in items:
            obs = item['observation']
            if set(obs) != set(keys):
                raise ValueError('Only the declared actor observations may enter imitation')
            for key, space in observation_space.spaces.items():
                value = np.asarray(obs[key], np.float32)
                if value.shape != space.shape or not np.all(np.isfinite(value)):
                    raise ValueError('Invalid sampled actor observation')
                observations[key].append(value)
            target = np.asarray(item['action'], np.float32)
            if target.shape != (6,) or not np.all(np.isfinite(target)) or np.any(np.abs(target)>1.):
                raise ValueError('Invalid bounded expert action')
            targets.append(target)
            weights.append(1./len(items))
    return ({key:torch.as_tensor(np.asarray(values)) for key,values in observations.items()},
            torch.as_tensor(np.asarray(targets)), torch.as_tensor(weights, dtype=torch.float32))


def observation_statistics(observations, keys):
    values = torch.cat([observations[key].flatten(1) for key in keys], dim=1)
    return values.mean(0).tolist(), torch.clamp(values.std(0, unbiased=False), min=.05).tolist()


def initialize_model(env, seed, center, scale, control_releases):
    policy_kwargs = dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh, squash_output=True,
        log_std_init=math.log(.05/math.sqrt(64)), control_releases=control_releases,
        features_extractor_class=NormalizedMeasurements,
        features_extractor_kwargs=dict(center=center, scale=scale))
    model = PPO(ImitationPolicy, env, seed=seed, device='cpu', verbose=0, learning_rate=1e-4,
        n_steps=64, batch_size=32, n_epochs=5, gamma=REWARD_DISCOUNT, ent_coef=0.,
        target_kl=.01, use_sde=True, sde_sample_freq=4, policy_kwargs=policy_kwargs)
    torch.nn.init.zeros_(model.policy.action_net.weight)
    torch.nn.init.zeros_(model.policy.action_net.bias)
    torch.nn.init.zeros_(model.policy.value_net.weight)
    torch.nn.init.zeros_(model.policy.value_net.bias)
    with torch.no_grad():
        std = torch.as_tensor(INITIAL_ACTION_STD_BOUNDS, dtype=model.policy.log_std.dtype)
        model.policy.log_std.copy_(torch.log(std/math.sqrt(64)).expand_as(model.policy.log_std))
    model.policy.reset_noise(1)
    for name, value in dict(task_algorithm='ppo', task_environment_version=TaskResidualEnv.ENVIRONMENT_VERSION,
        task_observation_layout=TaskResidualEnv.OBSERVATION_LAYOUT, task_action_layout=TaskResidualEnv.ACTION_LAYOUT,
        task_runtime_sha256=runtime_sources(), task_reference_sha256=digest(env.unwrapped.reference_path),
        task_imitation_sha256=digest(Path(__file__)), task_control_releases=control_releases).items():
        setattr(model, name, value)
    return model


def loss_value(policy, batch):
    obs, target, weight = batch
    predicted = policy._predict(obs, deterministic=True)
    error = torch.mean((predicted-target)**2, dim=1)
    return torch.sum(error*weight)/torch.sum(weight)


def fit_actor(model, train, epochs, on_checkpoint=None):
    """Fit the fixed epoch budget; select checkpoints in separate physical rollouts."""
    policy = model.policy
    optimizer = torch.optim.Adam(list(policy.mlp_extractor.policy_net.parameters())+
                                 list(policy.action_net.parameters()), lr=1e-3, weight_decay=1e-5)
    history = []
    checkpoints = {0, max(1, epochs//4), max(1, epochs//2), epochs}
    policy.set_training_mode(True)
    for epoch in range(epochs+1):
        if epoch % 25 == 0 or epoch in checkpoints:
            with torch.no_grad():
                training_loss = float(loss_value(policy, train))
            history.append(dict(epoch=epoch, training_mse=training_loss))
            if epoch in checkpoints and on_checkpoint is not None:
                on_checkpoint(epoch, model, training_loss)
        if epoch == epochs:
            break
        optimizer.zero_grad()
        loss = loss_value(policy, train)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite imitation loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        optimizer.step()
    policy.set_training_mode(False)
    return dict(history=history, final_training_mse=float(loss_value(policy, train).detach()))


def rollout(env, actor, seed, mode):
    obs, _ = env.reset(seed=seed)
    while True:
        obs, _, done, truncated, info = env.step(actor.predict(obs, deterministic=True)[0])
        if done or truncated:
            break
    if info.get('training_prefix') and not set(range(1, env.schedule[0]['catch_goal']+1)).issubset(info['recovered_ids']):
        raise ValueError('Missing physical recovery evidence')
    print('[Imitation rollout]', mode, seed, info['reason'], flush=True)
    return dict(mode=mode, seed=seed, jitter_m=info['config']['seat_jitter_m'], result=info)


def rollout_metrics(rows):
    return dict(episodes=len(rows), recovered=sum(bool(row['result'].get('training_prefix')) for row in rows),
                unsafe=sum(row['result']['reason'] in TaskResidualEnv.UNSAFE for row in rows))


def evaluate_training_cases(env, actor, cases):
    original = env.schedule[0]['seat_jitter_m']
    rows = []
    try:
        for case in cases:
            env.schedule[0]['seat_jitter_m'] = case['jitter_m']
            row = rollout(env, actor, case['seed'], 'bc_training')
            if row['result']['config']['seat_jitter_m'] != case['jitter_m']:
                raise ValueError('Training rollout did not reproduce its demonstrated perturbation')
            rows.append(row)
        return rows
    finally:
        env.schedule[0]['seat_jitter_m'] = original


def evaluate_checkpoint(job):
    checkpoint, reference, jitter, goal, control_releases, train_cases, eval_seeds, split = job
    torch.set_num_threads(1)
    model = PPO.load(checkpoint, device='cpu')
    if (model.task_runtime_sha256 != runtime_sources()
            or model.task_reference_sha256 != digest(Path(reference))
            or model.task_imitation_sha256 != digest(Path(__file__))
            or model.policy.control_releases != control_releases):
        raise ValueError('Imitation checkpoint source, reference or scope differs')
    env = TaskCurriculum(TaskResidualEnv(TaskConfig(), reference),
                         [dict(catch_goal=goal, seat_jitter_m=jitter)])
    try:
        return dict(checkpoint=Path(checkpoint).name,
            training_rollouts=evaluate_training_cases(env, model, train_cases),
            **{split+'_rollouts':[rollout(env, model, seed, 'bc_'+split) for seed in eval_seeds]})
    finally:
        env.close()


def evaluate_baselines(job):
    reference, seed, jitter, goal, control_releases, tables = job
    torch.set_num_threads(1)
    env = TaskCurriculum(TaskResidualEnv(TaskConfig(), reference),
                         [dict(catch_goal=goal, seat_jitter_m=jitter)])
    try:
        return [rollout(env, EventStudent(tables, mode, control_releases), seed, mode)
                for mode in ('zero', 'fixed_first', 'fixed_event', 'nearest_event')]
    finally:
        env.close()


def replay_float32(job):
    """Check the actor's action precision separately from approximation error."""
    reference, source = job
    torch.set_num_threads(1)
    env = TaskCurriculum(TaskResidualEnv(TaskConfig(), reference),
                         [dict(catch_goal=source['goal'], seat_jitter_m=source['jitter_m'])])
    index, timing_errors = 0, []
    try:
        obs, _ = env.reset(seed=source['seed'])
        while True:
            current, action = env.unwrapped._request, np.zeros(6, np.float32)
            if current is not None and current.released < source['control_releases'] and action_authority(obs):
                if index >= len(source['demonstrations']) or source['demonstrations'][index]['key'] != context_key(current):
                    return dict(seed=source['seed'], status='callback_diverged', executed=index)
                item = source['demonstrations'][index]
                action = np.asarray(item['action'], np.float32)
                timing_errors.append(current.time_s-item['request']['time_s'])
                index += 1
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        print('[Float32 action replay]', source['seed'], info['reason'], flush=True)
        return dict(seed=source['seed'], jitter_m=source['jitter_m'], status=info['reason'],
                    executed=index, callback_timing_errors_s=timing_errors, result=info,
                    scope='recorded float32 action diagnostic; not a causal policy')
    finally:
        env.close()


def select_checkpoints(models, validated):
    """Safety first, then target completion and confirmed physical progress."""
    by_path = {row['checkpoint']:row['validation_rollouts'] for row in validated}
    selected = []
    for seed in sorted({row['seed'] for row in models}):
        candidates = [row for row in models if row['seed']==seed]
        def key(model):
            rows = by_path[model['checkpoint']]
            metrics = rollout_metrics(rows)
            progress = sum(len(row['result']['recovered_ids']) for row in rows)
            return (-metrics['unsafe'], metrics['recovered'], progress, -model['epoch'])
        selected.append(max(candidates, key=key))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demonstrations', type=Path, nargs='+', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validation-seeds', type=int, nargs='+', required=True)
    parser.add_argument('--test-seeds', type=int, nargs='+', required=True)
    parser.add_argument('--training-seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--goal', type=int, default=3)
    parser.add_argument('--control-releases', type=int, default=6)
    parser.add_argument('--jitter-mm', type=float, default=.5)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if (args.output_dir.exists() or args.epochs < 25 or args.jobs < 1 or not 1 <= args.goal < 30
            or not 1 <= args.control_releases <= 30 or not 0. < args.jitter_mm <= 10.
            or len(set(args.training_seeds)) != len(args.training_seeds) or min(args.training_seeds)<0):
        parser.error('Use new outputs, positive bounded settings and distinct nonnegative training seeds')
    torch.set_num_threads(1)
    rows, inputs, all_source_seeds = load_demonstrations(args.demonstrations, args.reference,
                                                       args.goal, args.control_releases)
    validate_episode_splits(args.validation_seeds, args.test_seeds, all_source_seeds)
    if not rows:
        parser.error('No demonstration has physically reached the requested recovery goal')
    train_rows = rows
    for path in (Path(__file__), args.reference, Path(__file__).with_name('task_demonstrations.py'),
                 Path(__file__).with_name('train_task.py')):
        inputs[path.as_posix()] = digest(path)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'driver_source.py').open('xb').write(Path(__file__).read_bytes())
    protocol = dict(scope='PPO behavior-cloning initialization, not trained RL or full-task acceptance',
        source_sha256=runtime_sources(), input_sha256=inputs, goal=args.goal, control_releases=args.control_releases,
        train_episode_seeds=[row['seed'] for row in train_rows], validation_episode_seeds=args.validation_seeds,
        demonstration_attempt_seeds=sorted(all_source_seeds),
        demonstration_samples=sum(len(row['demonstrations']) for row in train_rows),
        demonstration_initial_jitter_m={str(row['seed']):row['jitter_m'] for row in train_rows},
        test_episode_seeds=args.test_seeds, network_seeds=args.training_seeds,
        epochs=args.epochs, optimizer='Adam, lr=0.001, weight_decay=0.00001',
        loss='episode-balanced normalized action MSE', normalization='training samples only; scale floor 0.05',
        checkpoint_epochs=sorted({0, max(1,args.epochs//4), max(1,args.epochs//2), args.epochs}),
        checkpoint_selection='independent physical validation: minimize unsafe, maximize recovery goals, then recovered IDs',
        task_config=asdict(TaskConfig()), evaluation_jitter_mm=args.jitter_mm)
    write_new(args.output_dir/'protocol.json', protocol)
    env = TaskCurriculum(TaskResidualEnv(TaskConfig(), args.reference),
                         [dict(catch_goal=args.goal, seat_jitter_m=args.jitter_mm/1000.)])
    models = []
    fitting = []
    try:
        train_batch = tensors(train_rows, env.observation_space)
        center, scale = observation_statistics(train_batch[0], tuple(env.observation_space.spaces))
        for seed in args.training_seeds:
            model = initialize_model(env, seed, center, scale, args.control_releases)
            started = time.perf_counter()
            def checkpoint(epoch, current, mse):
                path = args.output_dir/f'ppo_bc_seed{seed}_epoch{epoch}.zip'
                current.save(path)
                restored = PPO.load(path, device='cpu')
                expected = current.policy._predict(train_batch[0], deterministic=True).detach().numpy()
                np.testing.assert_array_equal(restored.policy._predict(train_batch[0], deterministic=True).detach().numpy(), expected)
                models.append(dict(seed=seed, epoch=epoch, checkpoint=path.name,
                    checkpoint_sha256=digest(path), save_load_verified=True, training_mse=mse))
            fit = fit_actor(model, train_batch, args.epochs, checkpoint)
            fitting.append(dict(seed=seed, fitting_wall_s=time.perf_counter()-started, **fit))
            print('[BC initialization]', seed, fit['final_training_mse'], flush=True)
    finally:
        env.close()
    # Freeze checkpoints before running any independent test scene.
    write_new(args.output_dir/'training.json', dict(protocol=protocol, models=models, fitting=fitting))
    training_hash = digest(args.output_dir/'training.json')
    tables = fit_event_tables(train_rows)
    with ProcessPoolExecutor(args.jobs) as pool:
        replay = list(pool.map(replay_trial, [(args.reference.as_posix(), row) for row in rows]))
        precision = list(pool.map(replay_float32, [(args.reference.as_posix(), row) for row in rows]))
        jobs = [(str(args.output_dir/model['checkpoint']), args.reference.as_posix(), args.jitter_mm/1000.,
                 args.goal, args.control_releases, [], args.validation_seeds, 'validation')
                for model in models]
        validated = list(pool.map(evaluate_checkpoint, jobs))
        selected = select_checkpoints(models, validated)
        write_new(args.output_dir/'selection.json', dict(selected=selected, validation=validated,
            training_sha256=training_hash, test_seeds=args.test_seeds))
        selection_hash = digest(args.output_dir/'selection.json')
        train_cases = [dict(seed=row['seed'], jitter_m=row['jitter_m']) for row in train_rows]
        fit_jobs = [(str(args.output_dir/model['checkpoint']), args.reference.as_posix(), args.jitter_mm/1000.,
                     args.goal, args.control_releases, train_cases, [], 'fit')
                    for model in models if model['epoch']==args.epochs]
        fit_rollouts = list(pool.map(evaluate_checkpoint, fit_jobs))
        jobs = [(str(args.output_dir/model['checkpoint']), args.reference.as_posix(), args.jitter_mm/1000.,
                 args.goal, args.control_releases, train_cases, args.test_seeds, 'test')
                for model in selected]
        evaluated = list(pool.map(evaluate_checkpoint, jobs))
        baseline_jobs = [(args.reference.as_posix(), seed, args.jitter_mm/1000., args.goal,
                          args.control_releases, tables) for seed in args.test_seeds]
        baseline_blocks = list(pool.map(evaluate_baselines, baseline_jobs))
    baseline = [row for block in baseline_blocks for row in block]
    if (any(digest(Path(path)) != value for path,value in inputs.items())
            or digest(args.output_dir/'training.json') != training_hash
            or digest(args.output_dir/'selection.json') != selection_hash
            or any(digest(args.output_dir/model['checkpoint']) != model['checkpoint_sha256'] for model in models)
            or runtime_sources() != protocol['source_sha256']):
        raise ValueError('Initialization or evaluation inputs changed')
    write_new(args.output_dir/'report.json', dict(protocol=protocol, models=models, teacher_replay=replay,
        training_sha256=training_hash, selection_sha256=selection_hash,
        fitting=fitting, validation=validated, selected=selected, fit_rollouts=fit_rollouts,
        float32_replay=precision,
        evaluated=evaluated, baseline=baseline,
        summary=dict(models={row['checkpoint']:dict(training=rollout_metrics(row['training_rollouts']),
            test=rollout_metrics(row['test_rollouts'])) for row in evaluated},
            baseline={mode:rollout_metrics([row for row in baseline if row['mode']==mode])
                      for mode in ('zero', 'fixed_first', 'fixed_event', 'nearest_event')}),
        complete_task_verified=False, rl_gain_verified=False))


if __name__ == '__main__':
    main()

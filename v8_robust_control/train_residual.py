"""Train bounded reception residuals and evaluate against paired feedback controls."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

from .residual_env import CatchConfig, ResidualCatchEnv


EVALUATION_CASES = {
    'baseline': {},
    'noise_1mm': {'noise_std_m': .001},
    'noise_3mm': {'noise_std_m': .003},
    'combined': {'noise_std_m': .003, 'delay_s': .02, 'dropout': .1},
}
EVALUATION_SEED_START = 10000
TRAINING_CASE_PROBABILITIES = (.25, .35, .25, .15)


class RandomizedReception(gym.Wrapper):
    """Episode-level sensor randomization, independent of initial-state RNG."""
    def __init__(self, env):
        super().__init__(env)
        self.base_config = env.config
        self.condition_rng = np.random.default_rng(0)
        self.case = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.condition_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(817,)))
        names = list(EVALUATION_CASES)
        self.case = names[int(self.condition_rng.choice(len(names), p=TRAINING_CASE_PROBABILITIES))]
        changes = dict(noise_std_m=0., delay_s=0., dropout=0.)
        changes.update(EVALUATION_CASES[self.case])
        self.env.config = replace(self.base_config, **changes)
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, dict(info, perception_case=self.case)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, dict(info, perception_case=self.case)


class TrainingProgress(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodes = self.successes = 0
        self.reasons = Counter()

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done:
                self.episodes += 1
                self.successes += int(info['is_success'])
                self.reasons[info['reason']] += 1
        return True


def create_model(algorithm, env, seed, learning_rate):
    common = dict(env=env, seed=seed, learning_rate=learning_rate,
                  gamma=env.unwrapped.config.gamma, device='cpu', verbose=0)
    if algorithm == 'sac':
        # gSDE avoids full-range uniform warmup. Environment limits remain active.
        model = SAC('MlpPolicy', **common, buffer_size=20000, learning_starts=128,
                    batch_size=64, train_freq=4, gradient_steps=1,
                    use_sde=True, use_sde_at_warmup=True, sde_sample_freq=4,
                    ent_coef='auto_0.01',
                    policy_kwargs=dict(net_arch=[64, 64], log_std_init=-3., activation_fn=torch.nn.Tanh))
        mean_layer = next(layer for layer in model.actor.mu.modules() if isinstance(layer, torch.nn.Linear))
        torch.nn.init.zeros_(mean_layer.weight)
        torch.nn.init.zeros_(mean_layer.bias)
    elif algorithm == 'ppo':
        model = PPO('MlpPolicy', **common, n_steps=128, batch_size=64, n_epochs=5,
                    ent_coef=0., target_kl=.03,
                    use_sde=True, sde_sample_freq=4,
                    policy_kwargs=dict(net_arch=[64, 64], log_std_init=-3.))
        torch.nn.init.zeros_(model.policy.action_net.weight)
        torch.nn.init.zeros_(model.policy.action_net.bias)
    else:
        raise ValueError(algorithm)
    return model


def evaluate(model, config, seeds, policies=('zero', 'static', 'random', 'learned'), cases=None):
    records = []
    for case, changes in (cases or EVALUATION_CASES).items():
        env = ResidualCatchEnv(replace(config, **dict(
            {'noise_std_m': 0., 'delay_s': 0., 'dropout': 0.}, **changes)))
        try:
            for seed in seeds:
                for policy in policies:
                    env.config = replace(env.config, feedback_enabled=policy != 'static')
                    obs, _ = env.reset(seed=seed)
                    rng = np.random.default_rng(seed+100000)
                    actions = []
                    start = time.perf_counter()
                    steps = 0
                    while True:
                        if policy in ('zero', 'static'):
                            action = np.zeros(4)
                        elif policy == 'random':
                            action = rng.uniform(-.15, .15, 4)
                        else:
                            action, _ = model.predict(obs, deterministic=True)
                        obs, _, done, truncated, info = env.step(action)
                        actions.append(env.action.copy())
                        steps += 1
                        if done or truncated:
                            break
                    sequence = np.asarray(actions)
                    info.update(case=case, seed=seed, policy=policy, control_steps=steps,
                                wall_time_s=time.perf_counter()-start,
                                residual_rms=float(np.sqrt(np.mean(sequence**2))),
                                residual_change_rms=float(np.sqrt(np.mean(np.diff(sequence, axis=0)**2)))
                                    if len(sequence) > 1 else 0.)
                    records.append(info)
        finally:
            env.close()
    return records


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), Path(__file__).with_name('residual_env.py'),
             Path(__file__).with_name('reception_control.py')]
    paths += list((root/'src/jugglebot/humanoid').glob('*.py'))
    paths += list((root/'src/jugglebot/robustness').glob('*.py'))
    paths += [root/'pyproject.toml', root/'uv.lock']
    paths += [root/'assets/leap_hand/left_hand.xml', root/'assets/leap_hand/right_hand.xml',
              root/'assets/leap_hand/UPSTREAM.json', root/'assets/unitree_g1/g1.xml',
              root/'assets/ur5e_leap_hands/world.xml']
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.exists()}


def summary(records):
    values = {}
    for policy in sorted({r['policy'] for r in records}):
        values[policy] = {}
        for case in dict.fromkeys(row['case'] for row in records):
            rows = [r for r in records if r['policy'] == policy and r['case'] == case]
            values[policy][case] = dict(episodes=len(rows), successes=sum(r['is_success'] for r in rows),
                                        failures=dict(Counter(r['reason'] for r in rows if not r['is_success'])))
    return values


def train_run(arguments):
    algorithm, seed, lr, weight, steps, config, seeds, output_dir = arguments
    torch.set_num_threads(1)
    name = f'{algorithm}_seed{seed}_lr{lr:g}_shape{weight:g}'
    env = RandomizedReception(ResidualCatchEnv(replace(config, shaping_weight=weight)))
    try:
        callback = TrainingProgress()
        model = create_model(algorithm, env, seed, lr)
        start = time.perf_counter()
        print('[Train]', name, steps, flush=True)
        model.learn(total_timesteps=steps, callback=callback)
        duration = time.perf_counter()-start
        checkpoint = output_dir/(name+'.zip')
        obs, _ = env.reset(seed=100)
        before, _ = model.predict(obs, deterministic=True)
        model.save(checkpoint)
        loaded = type(model).load(checkpoint, device='cpu')
        after, _ = loaded.predict(obs, deterministic=True)
        np.testing.assert_array_equal(before, after)
        records = evaluate(loaded, config, seeds, policies=('learned',))
        row = dict(kind='trained', algorithm=algorithm, seed=seed,
                   learning_rate=lr, shaping_weight=weight,
                   training_steps=model.num_timesteps, training_wall_time_s=duration,
                   steps_per_second=model.num_timesteps/duration,
                   training_episodes=callback.episodes, training_successes=callback.successes,
                   training_outcomes=dict(callback.reasons), checkpoint=checkpoint.name,
                   checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   save_load_verified=True, trials=records)
        print('[Evaluated]', name, json.dumps(summary(records)), flush=True)
        return row
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2048)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--algorithms', nargs='+', choices=['sac', 'ppo'], default=['ppo'])
    parser.add_argument('--learning-rates', type=float, nargs='+', default=[.0003])
    parser.add_argument('--shaping-weights', type=float, nargs='+', default=[0.])
    parser.add_argument('--eval-episodes', type=int, default=10, help='Validation episodes per perception case')
    parser.add_argument('--jobs', type=int, default=1, help='Concurrent independent training runs')
    args = parser.parse_args()
    if (args.steps < 128 or args.steps % 128 or args.eval_episodes < 1 or args.jobs < 1
            or len(set(args.seeds)) != len(args.seeds)
            or any(seed < 0 or seed >= EVALUATION_SEED_START for seed in args.seeds)
            or any(not np.isfinite(x) or x <= 0 for x in args.learning_rates)
            or any(not np.isfinite(x) or x < 0 for x in args.shaping_weights)):
        parser.error('Require positive budgets, steps divisible by 128, distinct training seeds < 10000')
    if args.output_dir.exists():
        parser.error('Output directory must be new')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    config = CatchConfig()
    seeds = list(range(EVALUATION_SEED_START, EVALUATION_SEED_START+args.eval_episodes))
    hashes = source_hashes()
    manifest = dict(task='single_ball_reception', config=asdict(config),
                    environment_version=ResidualCatchEnv.ENVIRONMENT_VERSION,
                    algorithms=args.algorithms, steps=args.steps, training_seeds=args.seeds,
                    jobs=args.jobs,
                    evaluation_seeds=seeds, learning_rates=args.learning_rates,
                    shaping_weights=args.shaping_weights, evaluation_cases=EVALUATION_CASES,
                    training_perception_cases=EVALUATION_CASES,
                    training_case_probabilities=TRAINING_CASE_PROBABILITIES,
                    policy_settings=dict(hidden_layers=[64, 64], activation='tanh',
                                         exploration='gSDE', initial_log_std=-3., noise_resample_steps=4),
                    source_sha256=hashes,
                    versions={n: importlib.metadata.version(n) for n in
                              ('gymnasium', 'stable-baselines3', 'torch', 'mujoco', 'numpy')},
                    comparison='equal_environment_steps_per_run; record_training_wall_time',
                    selection='validation only; evaluate_residual uses disjoint confirmation seeds',
                    scope='reception only; no throw policy or complete-task RL')
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    with (args.output_dir/'trials.jsonl').open('x', encoding='utf-8') as stream:
        controls = evaluate(None, config, seeds, policies=('zero', 'static', 'random'))
        stream.write(json.dumps(dict(kind='controls', trials=controls))+'\n')
        stream.flush()
        print('[Controls]', json.dumps(summary(controls)), flush=True)
        tasks = [(a, s, lr, w, args.steps, config, seeds, args.output_dir)
                 for w in args.shaping_weights for lr in args.learning_rates
                 for a in args.algorithms for s in args.seeds]
        pool = ProcessPoolExecutor(args.jobs) if args.jobs > 1 else None
        try:
            results = pool.map(train_run, tasks) if pool else map(train_run, tasks)
            for row in results:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
        finally:
            if pool:
                pool.shutdown()
    if hashes != source_hashes():
        raise RuntimeError('Experiment source changed during the pilot')
    (args.output_dir/'complete.json').write_text(json.dumps(dict(complete=True))+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()

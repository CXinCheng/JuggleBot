"""Prepare measured task references and run bounded policy pilots."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import gymnasium as gym
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO, SAC

from .compare_residual import digest
from .task_control import TaskCommand, TaskSpaceCascade, decode_action
from .task_env import (TaskConfig, TaskResidualEnv, TaskPolicy,
                       runtime_sources, measured_sources, result_reason, action_authority)


def cases():
    from .train_transfer import CASES
    return dict(CASES, seat_0p5mm=dict(seat_jitter_m=.0005))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')


def prepare(output, throws=30, estimator='difference'):
    output = Path(output)
    if output.exists():
        raise ValueError('Measured reference output must be new')
    sources = measured_sources()
    requests = []
    def decision(request):
        requests.append(asdict(request))
        return TaskCommand(request.nominal_parameters)
    controller = TaskSpaceCascade(decision, throws=throws,estimator=estimator)
    started = time.perf_counter()
    result = controller.run_complete(timeout=12. if throws <= 30 else 20.)
    if result_reason(result) != 'SUCCESS' or measured_sources() != sources:
        raise ValueError('Reference requires a complete clean task and unchanged runtime')
    record = dict(schema_version=1, source_sha256=sources, mujoco_version=mujoco.__version__,
        requested_throws=throws, event_estimator=estimator,requests=requests, result=result,
        run_wall_s=time.perf_counter()-started, generator_sha256=digest(Path(__file__)),
        observation_source='sampled positions and causal estimates; robot proprioception and tactile forces')
    write_new(output, record)
    return record


def observation_trial(job):
    estimator,name,settings,seed,throws = job
    torch.set_num_threads(1)
    env = TaskResidualEnv(TaskConfig(throws=throws,horizon_s=12. if throws<=30 else 20.,**settings),
                          estimator=estimator)
    errors = []
    try:
        _,_ = env.reset(seed=seed)
        while True:
            # Evaluation truth is read only while the worker is paused, and is
            # never included in the action or observation fed to the controller.
            request = env._request
            if request is not None:
                for ball in env.controller.airborne:
                    state = request.balls[ball]
                    if state is not None and state.valid and state.velocity_m_s is not None:
                        q,v = env.controller.balls[ball]
                        errors.append([float(np.linalg.norm(np.asarray(state.position_m)-env.controller.data.qpos[q:q+3])),
                                       float(np.linalg.norm(np.asarray(state.velocity_m_s)-env.controller.data.qvel[v:v+3]))])
            _,_,done,truncated,info = env.step(np.zeros(6))
            if done or truncated:
                break
        error = np.asarray(errors).reshape(-1,2)
        print('[Observation baseline]',estimator,name,seed,info['reason'],info['catches'],flush=True)
        return dict(estimator=estimator,case=name,settings=settings,seed=seed,result=info,
            observation=env.controller.conditions_summary()['ball_observation'],
            estimation=dict(scope='valid airborne estimates at paused decision callbacks',samples=len(error),
                position_rmse_m=float(np.sqrt(np.mean(error[:,0]**2))) if len(error) else None,
                velocity_rmse_m_s=float(np.sqrt(np.mean(error[:,1]**2))) if len(error) else None))
    finally:
        env.close()


def observation_baseline(args):
    if (args.output.exists() or args.jobs<1 or len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0
            or args.throws<4 or args.throws%2 or len(set(args.estimators))!=len(args.estimators)
            or len(set(args.cases))!=len(args.cases)):
        raise ValueError('Use new output, distinct nonnegative seeds, unique cases/estimators and even throws >= 4')
    settings = dict(cases(),sensor_only_1mm=dict(noise_std_m=.001),sensor_only_5mm=dict(noise_std_m=.005))
    protocol = dict(source_sha256=runtime_sources(),driver_sha256=digest(Path(__file__)),
        throws=args.throws,estimators=args.estimators,seeds=args.seeds,
        cases={name:settings[name] for name in args.cases},
        scope='Fixed zero event residual; full task acceptance, no policy training or selection')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_new(args.output.with_suffix('.protocol.json'),protocol)
    jobs = [(estimator,name,settings[name],seed,args.throws)
            for estimator in args.estimators for name in args.cases for seed in args.seeds]
    if args.jobs==1:
        rows = [observation_trial(job) for job in jobs]
    else:
        with ProcessPoolExecutor(args.jobs) as pool:
            rows = list(pool.map(observation_trial,jobs))
    if runtime_sources()!=protocol['source_sha256'] or digest(Path(__file__))!=protocol['driver_sha256']:
        raise ValueError('Observation baseline sources changed during execution')
    write_new(args.output,dict(protocol=protocol,trials=rows))


class RandomizedTask(gym.Wrapper):
    def __init__(self, env, training_cases):
        super().__init__(env)
        self.base_config, self.training_cases = env.config, training_cases
        self.rng = None

    def reset(self, *, seed=None, options=None):
        if seed is not None or self.rng is None:
            self.rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1703,)))
        name = self.training_cases[int(self.rng.integers(len(self.training_cases)))]
        self.env.config = replace(self.base_config, **cases()[name])
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, dict(info, case=name)


def curriculum_schedule():
    return [dict(catch_goal=1, seat_jitter_m=.0005),
            dict(catch_goal=1, seat_jitter_m=.001),
            dict(catch_goal=1, seat_jitter_m=.002),
            dict(catch_goal=3, seat_jitter_m=.002),
            dict(catch_goal=6, seat_jitter_m=.002),
            dict(catch_goal=30, seat_jitter_m=.002)]


REWARD_DISCOUNT = 1.
POTENTIAL_BOUND = .25


def measurement_potential(obs):
    """Bounded causal flight-error potential relative to measured calibration."""
    if obs['context'][-1] < .5:
        return 0.
    ball = int(np.clip(round(float(obs['context'][8])*2), 0, 2))
    if obs['errors'][ball, 6] < .5 or obs['balls'][ball, 9] < .5:
        return -POTENTIAL_BOUND
    error = obs['errors'][ball]
    squared = np.sum((error[:3]/4.)**2)+np.sum((error[3:6]/2.5)**2)
    return -POTENTIAL_BOUND*min(1., float(squared))


class TaskCurriculum(gym.Wrapper):
    """Train on recovered flight prefixes; evaluation uses the unwrapped task."""
    WINDOW = 32
    ADVANCE_RATE = .75
    PREFIX_BONUS = 2.

    def __init__(self, env, schedule=None, *, shaping=False):
        super().__init__(env)
        self.schedule = curriculum_schedule() if schedule is None else schedule
        if (not self.schedule or any(type(stage['catch_goal']) is not int
                or not 1 <= stage['catch_goal'] <= env.config.throws for stage in self.schedule)):
            raise ValueError('Curriculum goals must fit the complete task')
        self.base_config = env.config
        self.stage = 0
        self.outcomes = deque(maxlen=self.WINDOW)
        self.stage_history = []
        self.prefixes = self.episodes = self.steps = 0
        self.shaping = shaping
        self._potential = self._shaping_sum = 0.

    def reset(self, *, seed=None, options=None):
        if (self.stage+1 < len(self.schedule) and len(self.outcomes) == self.WINDOW
                and np.mean(self.outcomes) >= self.ADVANCE_RATE):
            self.stage_history.append(dict(stage=self.stage, successes=sum(self.outcomes),
                                           episodes=len(self.outcomes), decisions=self.steps))
            self.stage += 1
            self.outcomes.clear()
        self.env.config = replace(self.base_config, seat_jitter_m=self.schedule[self.stage]['seat_jitter_m'])
        obs, info = self.env.reset(seed=seed, options=options)
        self._potential = measurement_potential(obs) if self.shaping else 0.
        self._shaping_sum = 0.
        return obs, dict(info, curriculum_stage=self.stage, catch_goal=self.schedule[self.stage]['catch_goal'])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        goal = self.schedule[self.stage]['catch_goal']
        prefix = (not terminated and not truncated and goal < self.env.config.throws
                  and set(range(1, goal+1)).issubset(info['recovered_ids']))
        if prefix:
            self.env.close()
            terminated = True
            reward += self.PREFIX_BONUS
            info = dict(info, reason='TRAINING_PREFIX', is_success=False,
                reward_sums=dict(info['reward_sums'], curriculum=self.PREFIX_BONUS),
                episode_return=info['episode_return']+self.PREFIX_BONUS)
            self.prefixes += 1
        if self.shaping:
            potential = 0. if terminated else measurement_potential(obs)
            shaping = REWARD_DISCOUNT*potential-self._potential
            self._potential = potential
            self._shaping_sum += shaping
            reward += shaping
            info = dict(info, reward_sums=dict(info['reward_sums'], potential=self._shaping_sum),
                        episode_return=info['episode_return']+self._shaping_sum)
        if terminated or truncated:
            self.outcomes.append(bool(prefix or info['is_success']))
            self.episodes += 1
        return obs, reward, terminated, truncated, dict(info, training_prefix=bool(prefix),
            curriculum_stage=self.stage, catch_goal=goal)

    def progress_summary(self):
        return dict(stage=self.stage, catch_goal=self.schedule[self.stage]['catch_goal'],
            episodes=self.episodes, prefixes=self.prefixes, decisions=self.steps,
            window_episodes=len(self.outcomes), window_successes=sum(self.outcomes),
            history=self.stage_history)


class TrainingRewardScale(gym.RewardWrapper):
    """Scale optimizer targets while retaining raw physical event accounting."""
    def __init__(self, env, scale):
        if not np.isfinite(scale) or not 0. < scale <= 1.:
            raise ValueError('Training reward scale must be in (0, 1]')
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward):
        return self.scale*reward


ACTION_PROBES = {
    'zero': {},
    'lower_third_throw': dict(throw=[0., 0., -.1, 0., 0., 0.]),
    'forward_third_throw': dict(throw=[.3, 0., 0., 0., 0., 0.]),
    'forward_lower_third_throw': dict(throw=[.3, 0., -.1, 0., 0., 0.]),
    'earlier_third_release': dict(throw=[0., 0., 0., 0., .3, 0.]),
    'higher_first_reception': dict(reception=[0., 0., .25, 0., 0., 0.]),
    'lower_third_throw_medium': dict(throw=[0., 0., -.3, 0., 0., 0.]),
    'lower_third_throw_large': dict(throw=[0., 0., -.6, 0., 0., 0.]),
    'forward_third_throw_large': dict(throw=[.8, 0., 0., 0., 0., 0.]),
    'backward_third_throw': dict(throw=[-.5, 0., 0., 0., 0., 0.]),
    'lateral_third_throw_positive': dict(throw=[0., .5, 0., 0., 0., 0.]),
    'lateral_third_throw_negative': dict(throw=[0., -.5, 0., 0., 0., 0.]),
    'earlier_third_release_large': dict(throw=[0., 0., 0., 0., 1., 0.]),
    'lower_first_throw': dict(released=0, throw=[0., 0., -.3, 0., 0., 0.]),
    'forward_first_throw': dict(released=0, throw=[.3, 0., 0., 0., 0., 0.]),
    'earlier_first_release': dict(released=0, throw=[0., 0., 0., 0., .6, 0.]),
    'forward_first_throw_small': dict(released=0, throw=[.1, 0., 0., 0., 0., 0.]),
    'backward_first_throw_small': dict(released=0, throw=[-.1, 0., 0., 0., 0., 0.]),
    'lateral_first_throw_positive_small': dict(released=0, throw=[0., .1, 0., 0., 0., 0.]),
    'lateral_first_throw_negative_small': dict(released=0, throw=[0., -.1, 0., 0., 0., 0.]),
    'earlier_first_release_small': dict(released=0, throw=[0., 0., 0., 0., .2, 0.]),
    'lower_second_throw': dict(released=1, throw=[0., 0., -.3, 0., 0., 0.]),
    'lateral_second_throw_positive': dict(released=1, throw=[0., .3, 0., 0., 0., 0.]),
    'lateral_second_throw_negative': dict(released=1, throw=[0., -.3, 0., 0., 0., 0.]),
}


class ActionProbe:
    def __init__(self, profile):
        self.profile = ACTION_PROBES[profile]

    def predict(self, obs, deterministic=True):
        if round(float(obs['context'][5])*30) == self.profile.get('released', 2):
            kind = 'throw' if obs['context'][0] > .5 else 'reception'
            return np.asarray(self.profile.get(kind, np.zeros(6))), None
        return np.zeros(6), None


def audit_physics(controller):
    """Read-only diagnostics, queried only while the physics worker is paused."""
    balls = []
    for ball,(q,v) in enumerate(controller.balls):
        balls.append(dict(ball_id=ball,position_m=controller.data.qpos[q:q+3].tolist(),
            velocity_m_s=controller.data.qvel[v:v+3].tolist(),
            hand_contact={side:bool(controller.ball_contact(ball,side)) for side in controller.hands}))
    palms = {side:dict(position_m=controller.data.site_xpos[hand['seat']].tolist(),
        velocity_m_s=controller.spatial(hand)[3:].tolist(),mode=hand['mode'],held=hand['held'],
        capture_confirmed=bool(hand.get('capture_confirmed',False)))
        for side,hand in controller.hands.items()}
    return dict(time_s=float(controller.data.time),balls=balls,palms=palms)


def audit_decision(env, obs, action):
    """Record authority and estimation errors without feeding truth to a policy."""
    request = env._request
    if request is None:
        return None
    authority = action_authority(obs)
    executed = np.asarray(action)*authority
    command = decode_action(request,executed)
    physical = audit_physics(env.controller)
    estimates = []
    for ball,state in enumerate(request.balls):
        usable = state is not None and state.valid and state.velocity_m_s is not None
        truth = physical['balls'][ball]
        estimates.append(dict(ball_id=ball,usable=bool(usable),
            measurement=asdict(state) if state is not None else None,
            position_error_m=float(np.linalg.norm(np.asarray(state.position_m)-truth['position_m'])) if usable else None,
            velocity_error_m_s=float(np.linalg.norm(np.asarray(state.velocity_m_s)-truth['velocity_m_s'])) if usable else None))
    return dict(time_s=request.time_s,kind=request.kind,side=request.side,ball_id=request.ball_id,
        released=request.released,attempt=request.attempt,authority=authority,
        requested_action=np.asarray(action).tolist(),executed_action=executed.tolist(),
        command=asdict(command),nominal_parameters=list(request.nominal_parameters),
        estimates=estimates,physics_diagnostic=physical)


def audit_one(job):
    reference, seed, profile, horizon = job
    torch.set_num_threads(1)
    env = TaskResidualEnv(TaskConfig(seat_jitter_m=.002, horizon_s=horizon), reference)
    try:
        obs, _ = env.reset(seed=seed)
        actor = ActionProbe(profile)
        decisions = []
        while True:
            action = actor.predict(obs)[0]
            diagnostic = audit_decision(env,obs,action)
            if diagnostic is not None:
                decisions.append(diagnostic)
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        validate_trial(info)
        print('[Action audit]', profile, seed, info['catches'], info['reason'], flush=True)
        return dict(seed=seed, profile=profile, safe_window=info['reason'] == 'TIMEOUT', **info,
            decision_diagnostics=decisions,events=env.controller.events,
            initial_ball_states=env.controller.initial_ball_states,
            observation=env.controller.conditions_summary()['ball_observation'],
            terminal_physics=audit_physics(env.controller))
    finally:
        env.close()


def audit_pairs(rows):
    """Compare each bounded probe with the same initial-state zero action."""
    baseline = {row['seed']:row for row in rows if row['profile']=='zero'}
    pairs = []
    for row in rows:
        if row['profile']=='zero' or row['seed'] not in baseline:
            continue
        zero = baseline[row['seed']]
        if row['initial_ball_states']!=zero['initial_ball_states']:
            raise ValueError('Paired action probes must have identical physical initial states')
        changed = [d for d in row['decision_diagnostics'] if np.any(np.asarray(d['executed_action'])!=0.)]
        pairs.append(dict(seed=row['seed'],profile=row['profile'],
            executed_nonzero_decisions=len(changed),first_changed_decision_s=changed[0]['time_s'] if changed else None,
            catches_delta=row['catches']-zero['catches'],
            recovered_delta=len(row['recovered_ids'])-len(zero['recovered_ids']),
            baseline_reason=zero['reason'],probe_reason=row['reason'],
            safety_regression=row['reason'] in TaskResidualEnv.UNSAFE and zero['reason'] not in TaskResidualEnv.UNSAFE,
            scope='finite-window diagnostic, not complete-task success or RL gain'))
    return pairs


def audit(reference, output, seeds, jobs=1, horizon=1.8, profiles=None):
    if Path(output).exists() or not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0 or jobs < 1:
        raise ValueError('Use a new audit output and unique nonnegative development seeds')
    sources, reference_hash = runtime_sources(), digest(Path(reference))
    driver_hash = digest(Path(__file__))
    profiles = list(ACTION_PROBES) if profiles is None else profiles
    if not profiles or len(set(profiles)) != len(profiles) or any(name not in ACTION_PROBES for name in profiles):
        raise ValueError('Require unique known action profiles')
    work = [(reference, seed, name, horizon) for name in profiles for seed in seeds]
    if jobs == 1:
        rows = [audit_one(job) for job in work]
    else:
        with ProcessPoolExecutor(jobs) as pool:
            rows = list(pool.map(audit_one, work))
    if (sources != runtime_sources() or digest(Path(reference)) != reference_hash
            or driver_hash != digest(Path(__file__))):
        raise ValueError('Action audit runtime or reference changed')
    record = dict(schema_version=1, source_sha256=sources, reference_sha256=reference_hash,
        trainer_sha256=driver_hash,
        scope='development-seed action authority; finite window, not complete-task acceptance',
        diagnostic_scope='simulation truth is read-only audit output; excluded from actor input, actions and rewards',
        horizon_s=horizon, profiles={name: ACTION_PROBES[name] for name in profiles}, seeds=seeds,
        paired_probes=audit_pairs(rows),records=rows)
    write_new(output, record)
    return record


INITIAL_ACTION_STD_BOUNDS = (.20, .05, .05, .05, .05, .05)


def make_model(env, seed, algorithm='ppo', reward_scale=1.):
    common = dict(env=env, seed=seed, device='cpu', verbose=0, learning_rate=1e-4,
                  gamma=REWARD_DISCOUNT, use_sde=True, sde_sample_freq=4)
    policy = dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh,
                  log_std_init=math.log(min(INITIAL_ACTION_STD_BOUNDS)/math.sqrt(64)))
    if algorithm == 'ppo':
        model = PPO(TaskPolicy, **common, n_steps=64, batch_size=32, n_epochs=5,
            ent_coef=0., target_kl=.01, policy_kwargs=dict(policy, squash_output=True))
        torch.nn.init.zeros_(model.policy.action_net.weight)
        torch.nn.init.zeros_(model.policy.action_net.bias)
        torch.nn.init.zeros_(model.policy.value_net.weight)
        torch.nn.init.zeros_(model.policy.value_net.bias)
    elif algorithm == 'sac':
        model = SAC('MultiInputPolicy', **common, buffer_size=20000, learning_starts=128,
            batch_size=64, train_freq=4, gradient_steps=1, ent_coef=.01*reward_scale,
            use_sde_at_warmup=True, policy_kwargs=policy)
        mean = next(layer for layer in model.actor.mu.modules() if isinstance(layer, torch.nn.Linear))
        torch.nn.init.zeros_(mean.weight)
        torch.nn.init.zeros_(mean.bias)
    else:
        raise ValueError('Unknown task training algorithm')
    std = model.policy.log_std if algorithm == 'ppo' else model.actor.log_std
    with torch.no_grad():
        bounds = torch.as_tensor(INITIAL_ACTION_STD_BOUNDS, dtype=std.dtype, device=std.device)
        std.copy_(torch.log(bounds/math.sqrt(64)).expand_as(std))
    (model.policy if algorithm == 'ppo' else model.actor).reset_noise(1)
    model.task_algorithm = algorithm
    model.task_observation_layout = TaskResidualEnv.OBSERVATION_LAYOUT
    model.task_action_layout = TaskResidualEnv.ACTION_LAYOUT
    model.task_runtime_sha256 = runtime_sources()
    model.task_environment_version = TaskResidualEnv.ENVIRONMENT_VERSION
    reference = env.unwrapped.reference_path
    model.task_reference_sha256 = digest(reference) if reference else None
    return model


class Zero:
    def predict(self, obs, deterministic=True):
        return np.zeros(6), None


class Feedback:
    """Small ballistic launch correction and earlier catch for faster descent."""
    def predict(self, obs, deterministic=True):
        action = np.zeros(6)
        index = int(round(float(obs['context'][8])*2))
        error = obs['errors'][index]
        if error[6] > .5:
            position, velocity = error[:3]/200., error[3:6]/5.
            if obs['context'][0] > .5:
                action[:3] = -(position/.53+.25*velocity)/np.array([.75, .75, 1.25])
            else:
                action[2] = (.5*max(0., position[2])+.05*max(0., -velocity[2]))/.06
        return np.clip(action, -.15, .15), None


def action_key(obs, horizon, timed):
    context = obs['context']
    key = (int(context[1] > .5), int(context[3] > .5))
    return key+(round(float(context[4])*horizon/.02),) if timed else key


class Table:
    def __init__(self, samples, horizon, timed):
        self.horizon, self.timed = horizon, timed
        buckets = {}
        for obs, action in samples:
            buckets.setdefault(action_key(obs, horizon, timed), []).append(action)
        self.actions = {key: np.mean(values, axis=0) for key, values in buckets.items()}

    def predict(self, obs, deterministic=True):
        key = action_key(obs, self.horizon, self.timed)
        if key not in self.actions and self.timed:
            available = [item for item in self.actions if item[:2] == key[:2]]
            key = min(available, key=lambda item: abs(item[2]-key[2])) if available else key
        return self.actions.get(key, np.zeros(6)).copy(), None


def validate_policy(actor, reference):
    if isinstance(actor, (PPO, SAC)) and (
            getattr(actor, 'task_environment_version', None) != TaskResidualEnv.ENVIRONMENT_VERSION
            or getattr(actor, 'task_observation_layout', None) != TaskResidualEnv.OBSERVATION_LAYOUT
            or getattr(actor, 'task_action_layout', None) != TaskResidualEnv.ACTION_LAYOUT
            or getattr(actor, 'task_runtime_sha256', None) != runtime_sources()
            or getattr(actor, 'task_reference_sha256', None) != digest(Path(reference))):
        raise ValueError('Policy runtime, action semantics or measured reference differs')


def evaluate(actor, reference, seeds, case_names, collect=False):
    validate_policy(actor, reference)
    env = TaskResidualEnv(reference=reference)
    base = env.config
    rows, samples = [], []
    try:
        for name in case_names:
            env.config = replace(base, **cases()[name])
            for seed in seeds:
                started = time.perf_counter()
                obs, _ = env.reset(seed=seed)
                while True:
                    action, _ = actor.predict(obs, deterministic=True)
                    if collect and np.any(obs['context'][:2]) and action_authority(obs):
                        samples.append((obs, np.asarray(action).copy()))
                    obs, _, done, truncated, info = env.step(action)
                    if done or truncated:
                        break
                validate_trial(info)
                rows.append(dict(case=name, seed=seed, wall_s=time.perf_counter()-started, **info))
                print('[Task evaluation]', type(actor).__name__, name, seed, info['catches'],
                      info['reason'], f'{time.perf_counter()-started:.1f}s', flush=True)
    finally:
        env.close()
    return rows, samples


def evaluate_prefix(actor, reference, seeds, jitter=.0005):
    """Development set only: the original task stopped after one ready recovery."""
    validate_policy(actor, reference)
    base = TaskResidualEnv(TaskConfig(), reference)
    env = TaskCurriculum(base, [dict(catch_goal=1, seat_jitter_m=jitter)])
    rows = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            while True:
                obs, _, done, truncated, info = env.step(actor.predict(obs, deterministic=True)[0])
                if done or truncated:
                    break
            if info['training_prefix'] and (info['is_success'] or 1 not in info['recovered_ids']):
                raise ValueError('Prefix must have physical recovery evidence without complete success')
            print('[Prefix evaluation]', type(actor).__name__, seed, info['reason'], flush=True)
            rows.append(dict(seed=seed, **info))
        return rows
    finally:
        env.close()


def validate_trial(row):
    sums = row['reward_sums']
    reason = row['reason']
    expected_event = TaskResidualEnv.terminal_reward(reason)
    if (row['environment_version'] != TaskResidualEnv.ENVIRONMENT_VERSION
            or row['is_success'] != (reason == 'SUCCESS')
            or not 0 <= row['catches'] <= row['releases'] <= row['requested_throws']
            or sums['progress'] != row['catches'] or sums['event'] != expected_event
            or abs(sum(sums.values())-row['episode_return']) > 1e-8
            or not -TaskResidualEnv.ACTION_COST*row['decisions']-1e-10 <= sums['action'] <= 0.
            or not -4*TaskResidualEnv.RATE_COST*row['decisions']-1e-10 <= sums['rate'] <= 0.):
        raise ValueError('Invalid task counts or reward ledger')
    if row['is_success'] and (row['catches'] != row['requested_throws'] or row['failure_event'] is not None
            or row['initial_hold_s'] < .05-1e-10 or row['final_hold_s'] < .8-1e-10
            or row['max_ball_height_m'] >= 1.3 or row['ball_contacts'] or row['grasp_assistance']
            or row['grasp_constraints'] or row['completed_ids'] != list(range(1, row['requested_throws']+1))):
        raise ValueError('Unconfirmed complete-task outcome')


def metrics(rows):
    return dict(clean=sum(row['is_success'] for row in rows if row['case'] == 'clean'),
        perturbed=sum(row['is_success'] for row in rows if row['case'] != 'clean'),
        unsafe=sum(row['reason'] in TaskResidualEnv.UNSAFE for row in rows),
        catches=sum(row['catches'] for row in rows))


def screening(policy, controls):
    grid = {(row['case'], row['seed']) for row in policy}
    if (not controls or len(grid) != len(policy)
            or any(len(rows) != len(grid) or {(row['case'], row['seed']) for row in rows} != grid
                   for rows in controls.values())):
        raise ValueError('Compare identical, unique case/seed grids')
    value = metrics(policy)
    comparisons = {name: dict(clean_nonregression=value['clean'] >= metrics(rows)['clean'],
        safety_nonregression=value['unsafe'] <= metrics(rows)['unsafe'],
        complete_task_gain=value['perturbed'] > metrics(rows)['perturbed']) for name, rows in controls.items()}
    return dict(passed=all(all(checks.values()) for checks in comparisons.values()), comparisons=comparisons,
                scope='small fixed-test pilot screening; not final robustness acceptance')


def assess_policy(actor, args, baseline):
    if args.get('development_only'):
        rows = evaluate_prefix(actor, args['reference'], args['validation_seeds'])
        return dict(development_trials=rows, recovered_prefixes=sum(row['training_prefix'] for row in rows),
                    screening=None)
    validation, samples = evaluate(actor, args['reference'], args['validation_seeds'], args['cases'], collect=True)
    fixed, timed = Table(samples, TaskConfig().horizon_s, False), Table(samples, TaskConfig().horizon_s, True)
    policy_rows, _ = evaluate(actor, args['reference'], args['test_seeds'], args['cases'])
    fixed_rows, _ = evaluate(fixed, args['reference'], args['test_seeds'], args['cases'])
    timed_rows, _ = evaluate(timed, args['reference'], args['test_seeds'], args['cases'])
    controls = dict(baseline, phase_fixed=fixed_rows, time_table=timed_rows)
    return dict(validation=validation, trials=policy_rows, fixed_trials=fixed_rows, timed_trials=timed_rows,
        fitted_tables=dict(source='validation episodes with nonzero action authority',
            fixed={':'.join(map(str,k)):v.tolist() for k,v in fixed.actions.items()},
            timed={':'.join(map(str,k)):v.tolist() for k,v in timed.actions.items()}),
        metrics=metrics(policy_rows), screening=screening(policy_rows, controls))


def train_one(job):
    seed, args, baseline = job
    torch.set_num_threads(1)
    config = TaskConfig()
    base = TaskResidualEnv(config, args['reference'])
    task = (TaskCurriculum(base, shaping=args.get('curriculum_shaping', False))
            if args.get('curriculum') else RandomizedTask(base, args['train_cases']))
    scale = args.get('reward_scale', 1.)
    env = TrainingRewardScale(task, scale)
    try:
        algorithm = args.get('algorithm', 'ppo')
        model = make_model(env, seed, algorithm, reward_scale=scale)
        obs, _ = env.reset(seed=seed)
        np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], np.zeros(6))
        started = time.perf_counter()
        from .train_residual import TrainingProgress
        progress = TrainingProgress(f'task_seed{seed}')
        progress.next_progress = 32
        for _ in range(args['steps']//64):
            model.learn(64, reset_num_timesteps=False, callback=progress)
            print('[Task training]', seed, model.num_timesteps, f'{time.perf_counter()-started:.1f}s', flush=True)
        wall_s = time.perf_counter()-started
        path = Path(args['output_dir'])/f'{algorithm}_seed{seed}.zip'
        # Sampled observations are dictionaries; save/load must preserve actions.
        current = {key: value.copy() for key, value in model._last_obs.items()}
        expected = model.predict(current, deterministic=True)[0]
        model.save(path)
        loaded = type(model).load(path, device='cpu')
        np.testing.assert_array_equal(loaded.predict(current, deterministic=True)[0], expected)
        return dict(seed=seed, steps=model.num_timesteps, training_wall_s=wall_s,
            curriculum=task.progress_summary() if isinstance(task, TaskCurriculum) else None,
            checkpoint=path.name, checkpoint_sha256=digest(path), save_load_verified=True,
            **assess_policy(loaded, args, baseline))
    finally:
        env.close()


def evaluate_saved_one(job):
    row, folder, algorithm, settings, baseline = job
    torch.set_num_threads(1)
    path = Path(folder)/row['checkpoint']
    if digest(path) != row['checkpoint_sha256']:
        raise ValueError('Checkpoint changed before evaluation')
    model = (PPO if algorithm == 'ppo' else SAC).load(path, device='cpu')
    validate_policy(model, settings['reference'])
    result = assess_policy(model, settings, baseline)
    if digest(path) != row['checkpoint_sha256']:
        raise ValueError('Checkpoint changed during evaluation')
    return dict(seed=row['seed'], checkpoint=row['checkpoint'], checkpoint_sha256=row['checkpoint_sha256'],
                training_steps=row['steps'], **result)


def evaluate_saved(args):
    if args.output.exists() or not args.training.is_file() or args.jobs < 1:
        raise ValueError('Use an existing training report and a new evaluation output')
    training_hash, driver_hash = digest(args.training), digest(Path(__file__))
    record = json.loads(args.training.read_text(encoding='utf-8'))
    sources = runtime_sources()
    if (record['source_sha256'] != sources or record['environment_version'] != TaskResidualEnv.ENVIRONMENT_VERSION
            or record['mujoco_version'] != mujoco.__version__ or record['task_config'] != asdict(TaskConfig())):
        raise ValueError('Training report task runtime or observation configuration differs')
    reference = Path(record['settings']['reference'])
    if not reference.is_file() or digest(reference) != record['reference_sha256']:
        raise ValueError('Measured training reference is missing or changed')
    folder = args.input_dir or Path(record['settings']['output_dir'])
    selected = [row for row in record['models'] if args.training_seed is None or row['seed'] == args.training_seed]
    if not selected:
        raise ValueError('Training seed is absent from the report')
    for row in selected:
        path = folder/row['checkpoint']
        if not path.is_file() or digest(path) != row['checkpoint_sha256']:
            raise ValueError(f'Missing or changed checkpoint: {path}')
    groups = [args.seeds, args.validation_seeds]
    if (any(not group or len(set(group)) != len(group) or min(group) < 0 for group in groups)
            or set(args.seeds)&set(record['settings']['train_seeds'])
            or not args.development_only and (set(args.seeds)&set(record['settings']['validation_seeds']+args.validation_seeds)
                or 'clean' not in args.cases or len(set(args.cases)) != len(args.cases))):
        raise ValueError('Use distinct nonnegative evaluation seeds and retain the clean comparator')
    algorithm = record['optimizer']['algorithm'].lower()
    if algorithm not in ('ppo', 'sac'):
        raise ValueError('Unsupported checkpoint algorithm')
    settings = dict(reference=reference.as_posix(), development_only=args.development_only,
        validation_seeds=args.seeds if args.development_only else args.validation_seeds,
        test_seeds=[] if args.development_only else args.seeds,
        cases=[] if args.development_only else args.cases)
    baseline = {name: (evaluate_prefix(actor, reference, args.seeds) if args.development_only
        else evaluate(actor, reference, args.seeds, args.cases)[0])
        for name, actor in (('zero', Zero()), ('feedback', Feedback()))}
    jobs = [(row, folder.as_posix(), algorithm, settings, baseline) for row in selected]
    if args.jobs == 1:
        models = [evaluate_saved_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(args.jobs) as pool:
            models = list(pool.map(evaluate_saved_one, jobs))
    if (runtime_sources() != sources or digest(args.training) != training_hash
            or digest(reference) != record['reference_sha256'] or digest(Path(__file__)) != driver_hash):
        raise ValueError('Evaluation inputs or sources changed')
    result = dict(schema_version=1, scope='development recovery' if args.development_only else 'complete 30-throw comparison',
        source_sha256=sources, evaluator_sha256=driver_hash, training_report_sha256=training_hash,
        training_report=args.training.as_posix(), training_driver_sha256=record['trainer_sha256'],
        reference_sha256=record['reference_sha256'], environment_version=TaskResidualEnv.ENVIRONMENT_VERSION,
        observation_layout=TaskResidualEnv.OBSERVATION_LAYOUT, action_layout=TaskResidualEnv.ACTION_LAYOUT,
        algorithm=algorithm, settings=settings, task_config=asdict(TaskConfig()), baseline=baseline, models=models,
        all_screened=None if args.development_only else all(row['screening']['passed'] for row in models))
    write_new(args.output, result)
    print('Frozen evaluation:', args.output, 'screened:', result['all_screened'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    reference = commands.add_parser('prepare')
    reference.add_argument('--output', type=Path, required=True)
    reference.add_argument('--throws',type=int,default=30)
    reference.add_argument('--estimator',choices=('difference','filtered'),default='difference')
    observation = commands.add_parser('observation-baseline')
    observation.add_argument('--estimators',nargs='+',choices=('difference','filtered'),default=['difference','filtered'])
    observation.add_argument('--cases',nargs='+',choices=list(cases())+['sensor_only_1mm','sensor_only_5mm'],
                             default=['clean','sensor_only_1mm','sensor_only_3mm','sensor_only_5mm','seat_2mm','combined'])
    observation.add_argument('--seeds',type=int,nargs='+',default=[416000,416001])
    observation.add_argument('--throws',type=int,default=30)
    observation.add_argument('--jobs',type=int,default=1)
    observation.add_argument('--output',type=Path,required=True)
    probe = commands.add_parser('audit')
    probe.add_argument('--reference', type=Path, required=True)
    probe.add_argument('--seeds', type=int, nargs='+', default=[410200, 410201, 410202])
    probe.add_argument('--jobs', type=int, default=1)
    probe.add_argument('--profiles', nargs='+', choices=ACTION_PROBES)
    probe.add_argument('--output', type=Path, required=True)
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--training', type=Path, required=True)
    evaluation.add_argument('--input-dir', type=Path)
    evaluation.add_argument('--training-seed', type=int)
    evaluation.add_argument('--seeds', type=int, nargs='+', required=True)
    evaluation.add_argument('--validation-seeds', type=int, nargs='+', default=[890000])
    evaluation.add_argument('--cases', nargs='+', choices=cases(), default=['clean', 'seat_2mm', 'combined'])
    evaluation.add_argument('--development-only', action='store_true')
    evaluation.add_argument('--jobs', type=int, default=1)
    evaluation.add_argument('--output', type=Path, required=True)
    pilot = commands.add_parser('pilot')
    pilot.add_argument('--algorithm', choices=['ppo', 'sac'], default='ppo')
    pilot.add_argument('--reward-scale', type=float, default=.01)
    pilot.add_argument('--reference', type=Path, required=True)
    pilot.add_argument('--steps', type=int, default=128)
    pilot.add_argument('--train-seeds', type=int, nargs='+', default=[0])
    pilot.add_argument('--validation-seeds', type=int, nargs='+', default=[610000])
    pilot.add_argument('--test-seeds', type=int, nargs='+', default=[620000, 620001])
    training = pilot.add_mutually_exclusive_group()
    training.add_argument('--train-cases', nargs='+', choices=cases())
    training.add_argument('--curriculum', action='store_true')
    pilot.add_argument('--curriculum-shaping', action='store_true')
    pilot.add_argument('--development-only', action='store_true')
    pilot.add_argument('--cases', nargs='+', choices=cases(), default=['clean', 'seat_2mm', 'combined'])
    pilot.add_argument('--jobs', type=int, default=1)
    pilot.add_argument('--output-dir', type=Path, required=True)
    pilot.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'prepare':
        prepare(args.output,args.throws,args.estimator)
        print('Measured task reference:', args.output, flush=True)
        return
    if args.command == 'observation-baseline':
        observation_baseline(args)
        return
    if args.command == 'audit':
        audit(args.reference, args.output, args.seeds, args.jobs, profiles=args.profiles)
        return
    if args.command == 'evaluate':
        evaluate_saved(args)
        return
    if not args.curriculum and args.train_cases is None:
        args.train_cases = ['seat_0p5mm', 'seat_2mm']
    if args.curriculum_shaping and not args.curriculum:
        parser.error('--curriculum-shaping requires --curriculum')
    if args.development_only and not args.curriculum:
        parser.error('--development-only requires --curriculum')
    groups = [args.train_seeds, args.validation_seeds, args.test_seeds]
    if (args.steps < 64 or args.steps % 64 or args.jobs < 1 or args.output_dir.exists() or args.output.exists()
            or not np.isfinite(args.reward_scale) or not 0. < args.reward_scale <= 1.
            or not args.reference.is_file() or any(len(set(group)) != len(group) or min(group) < 0 for group in groups)
            or set(args.validation_seeds)&set(args.test_seeds) or 'clean' not in args.cases
            or set(args.train_seeds)&(set(args.validation_seeds)|set(args.test_seeds))):
        parser.error('Use new outputs, distinct seed sets, a clean comparator and a positive multiple of 64 steps')
    sources, reference_hash = runtime_sources(), digest(args.reference)
    driver_hash = digest(Path(__file__))
    settings = vars(args).copy()
    settings = {key: value.as_posix() if isinstance(value, Path) else value for key, value in settings.items()}
    args.output_dir.mkdir(parents=True)
    baseline = {}
    for name, actor in (('zero', Zero()), ('feedback', Feedback())):
        baseline[name] = (evaluate_prefix(actor, args.reference, args.validation_seeds)
            if args.development_only else evaluate(actor, args.reference, args.test_seeds, args.cases)[0])
    jobs = [(seed, settings, baseline) for seed in args.train_seeds]
    if args.jobs == 1:
        models = [train_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(args.jobs) as pool:
            models = list(pool.map(train_one, jobs))
    if runtime_sources() != sources or digest(args.reference) != reference_hash or digest(Path(__file__)) != driver_hash:
        raise ValueError('Pilot sources or reference changed during execution')
    record = dict(schema_version=1,
        scope=('development recovery prefixes; no independent complete-task test'
               if args.development_only else 'bounded event-action policy pilot'), source_sha256=sources,
        environment_version=TaskResidualEnv.ENVIRONMENT_VERSION,
        trainer_sha256=driver_hash, reference_sha256=reference_hash, mujoco_version=mujoco.__version__,
        observation_layout=TaskResidualEnv.OBSERVATION_LAYOUT, action_layout=TaskResidualEnv.ACTION_LAYOUT,
        settings=settings, task_config=asdict(TaskConfig()),
        training_cases={name:cases()[name] for name in (args.train_cases or [])},
        training_curriculum=curriculum_schedule() if args.curriculum else None,
        curriculum_protocol=(dict(window=TaskCurriculum.WINDOW, advance_rate=TaskCurriculum.ADVANCE_RATE,
            prefix_bonus=TaskCurriculum.PREFIX_BONUS, potential_enabled=args.curriculum_shaping,
            potential_bound=POTENTIAL_BOUND, potential_discount=REWARD_DISCOUNT,
            potential_terminal=0., evaluation=('development recovery prefix' if args.development_only
                else 'unwrapped complete 30 throws')) if args.curriculum else None),
        evaluation_cases={name:cases()[name] for name in args.cases},
        reward=dict(progress_per_confirmed_catch=1., success=TaskResidualEnv.SUCCESS_REWARD,
            failure=TaskResidualEnv.FAILURE_REWARD, unsafe=TaskResidualEnv.FAILURE_REWARD,
            action_cost=TaskResidualEnv.ACTION_COST, rate_cost=TaskResidualEnv.RATE_COST,
            discount_unit='finite episode, no decision discount', survival_reward=0.,
            optimizer_scale=args.reward_scale, evaluation_scale=1.),
        residual_authority='measured ball and robot reference innovation; zero at nominal measurements',
        optimizer=(dict(algorithm='PPO', n_steps=64, batch_size=32, n_epochs=5, learning_rate=1e-4,
            gamma=REWARD_DISCOUNT, target_kl=.01, initial_action_std_bounds=list(INITIAL_ACTION_STD_BOUNDS))
            if args.algorithm == 'ppo' else dict(algorithm='SAC', buffer_size=20000, learning_starts=128,
                batch_size=64, train_freq=4, gradient_steps=1, ent_coef=.01*args.reward_scale, learning_rate=1e-4,
                gamma=REWARD_DISCOUNT, use_sde_at_warmup=True, initial_action_std_bounds=list(INITIAL_ACTION_STD_BOUNDS))),
        baseline=baseline, models=models,
        all_screened=None if args.development_only else all(row['screening']['passed'] for row in models))
    write_new(args.output, record)
    print('Pilot:', args.output, 'screened:', record['all_screened'], flush=True)


if __name__ == '__main__':
    main()

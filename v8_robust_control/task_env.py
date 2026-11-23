"""Event-driven residual actions at real throw and reception callbacks."""
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from queue import Queue
from threading import Event, Thread

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import torch
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from jugglebot.robustness.perception import PerceptionConfig
from .cascade_env import reference_sources
from .compare_residual import digest
from .task_control import TaskSpaceCascade, TaskCommand, decode_action, MAX_TOTAL_DEFER_S

MODES = ('hold', 'throw', 'return', 'wait', 'receive', 'carry', 'ending')
ROBOT_SCALE = np.r_[np.full(14, .5), np.full(14, .1), np.full(32, .5),
                    np.full(32, .1), np.full(6, 2.), np.full(6, .5)]
ERROR_SCALE = np.r_[np.full(14, 100.), np.full(14, 2.), np.full(32, 20.),
                    np.full(32, 2.), np.full(6, 100.), np.full(6, 2.)]
_STOP = object()
INNOVATION_DEADBAND = 1e-4


def measured_sources():
    root = Path(__file__).resolve().parents[1]
    names = ['v8_robust_control/'+name+'.py' for name in
             ('task_control','event_observation','reception_control','transfer_observation')]
    names.append('v8_robust_control/event_seed_plans.json')
    return dict(reference_sources(), **{name: digest(root/name) for name in names})


def runtime_sources():
    return dict(measured_sources(), **{'v8_robust_control/task_env.py': digest(Path(__file__))})


@dataclass(frozen=True)
class TaskConfig:
    throws: int = 30
    horizon_s: float = 12.
    seat_jitter_m: float = 0.
    noise_std_m: float = 0.
    delay_s: float = 0.
    dropout: float = 0.
    max_decisions: int = 512

    def __post_init__(self):
        if (type(self.throws) is not int or self.throws < 4 or self.throws % 2
                or type(self.max_decisions) is not int or self.max_decisions < 1
                or not np.isfinite(self.horizon_s) or self.horizon_s <= 0.
                or not np.isfinite(self.seat_jitter_m) or not 0. <= self.seat_jitter_m <= .01):
            raise ValueError('Require even throws >= 4, a positive deadline/budget and reset jitter <= 10 mm')
        PerceptionConfig(self.noise_std_m, self.delay_s, self.dropout)


def context_key(request):
    def field(name):
        return request[name] if isinstance(request, dict) else getattr(request, name)
    return ':'.join(str(field(name)) for name in ('kind', 'side', 'released', 'ball_id'))


def robot_vector(request):
    def field(name):
        return request[name] if isinstance(request, dict) else getattr(request, name)
    names = ('arm_position_rad', 'arm_velocity_rad_s', 'finger_position_rad',
             'finger_velocity_rad_s', 'palm_position_m', 'palm_velocity_m_s')
    return np.concatenate([np.asarray(field(name)).ravel() for name in names])


def observation(request, config, nominal=None):
    """Only detached callback measurements and offline sampled references."""
    balls, errors = np.zeros((3, 12)), np.zeros((3, 7))
    palm = np.asarray(request.palm_position_m)[int(request.side == 'left')]
    for index, state in enumerate(request.balls):
        balls[index, 10] = 10.
        if state is None or not state.valid or state.velocity_m_s is None:
            continue
        balls[index, :3] = 4.*(np.asarray(state.position_m)-palm)
        balls[index, 3:6] = .5*np.asarray(state.velocity_m_s)
        if state.acceleration_m_s2 is not None:
            balls[index, 6:9] = .025*np.asarray(state.acceleration_m_s2)
        balls[index, 9:] = [1., 20.*state.age_s, 20.*(state.received_time_s-state.sample_time_s)]
        reference = nominal['balls'][index] if nominal else None
        if reference and reference['valid'] and reference['velocity_m_s'] is not None:
            errors[index, :3] = 200.*(np.asarray(state.position_m)-reference['position_m'])
            errors[index, 3:6] = 5.*(np.asarray(state.velocity_m_s)-reference['velocity_m_s'])
            errors[index, 6] = 1.
    robot = robot_vector(request)
    robot_error = (robot-robot_vector(nominal))*ERROR_SCALE if nominal else np.zeros(104)
    phase = request.kind == 'throw'
    context = [float(phase), float(not phase), float(request.side == 'right'), float(request.side == 'left'),
        request.time_s/config.horizon_s, request.released/config.throws,
        request.remaining_defer_s/MAX_TOTAL_DEFER_S, request.attempt/8., request.ball_id/2.,
        20.*max(0., request.time_s-request.contact_time_s),
        10.*(request.time_s-nominal['time_s']) if nominal else 0.]
    context += [float(mode == value) for mode in request.hand_modes for value in MODES]
    context.append(float(nominal is not None))
    parameters = np.asarray(request.nominal_parameters)*([.5, .5, .5, 2., 40.] if phase else [20., 20., 20., 2., 2.])
    values = dict(balls=balls, errors=errors, robot=robot*ROBOT_SCALE, robot_error=robot_error,
                  nominal=parameters, tactile=np.asarray(request.contact_normal_force_n)/20., context=np.asarray(context))
    return {name: np.clip(value, -10., 10.).astype(np.float32) for name, value in values.items()}


def action_authority(obs):
    if obs['context'][-1] < .5:
        return 1.
    side = int(obs['context'][3] > .5)
    robot = obs['robot_error']
    innovation = max(float(np.max(np.abs(obs['errors'][:, :6]))),
        *(float(np.max(np.abs(robot[start+side*width:start+(side+1)*width])))
          for start, width in ((0, 7), (14, 7), (28, 16), (60, 16), (92, 3), (98, 3))))
    return float(innovation > INNOVATION_DEADBAND)


class TaskPolicy(MultiInputActorCriticPolicy):
    """Actor gradients are inactive at zero measured innovation or no callback."""
    @staticmethod
    def active(obs):
        context = obs['context']
        side = (context[..., 3] > .5).long()
        innovation = obs['errors'][..., :6].abs().amax(dim=(-2, -1))
        for start, width in ((0, 7), (14, 7), (28, 16), (60, 16), (92, 3), (98, 3)):
            values = obs['robot_error'][..., start:start+2*width].reshape(*side.shape, 2, width).abs().amax(-1)
            innovation = torch.maximum(innovation, values.gather(-1, side[..., None]).squeeze(-1))
        has_authority = (context[..., -1] < .5) | (innovation > INNOVATION_DEADBAND)
        return ((context[..., :2].sum(-1) > .5) & has_authority).to(context.dtype)

    def forward(self, obs, deterministic=False):
        actions, values, probability = super().forward(obs, deterministic)
        mask = self.active(obs)
        return actions*mask[..., None], values, probability*mask

    def evaluate_actions(self, obs, actions):
        values, probability, entropy = super().evaluate_actions(obs, actions)
        mask = self.active(obs)
        return values, probability*mask, None if entropy is None else entropy*mask

    def _predict(self, obs, deterministic=False):
        return super()._predict(obs, deterministic)*self.active(obs)[..., None]


class _Cancelled(Exception):
    pass


class _EpisodeCascade(TaskSpaceCascade):
    cancel = None

    def step(self):
        if self.cancel is not None and self.cancel.is_set():
            raise _Cancelled()
        super().step()
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            self.failure_event = dict(reason='NONFINITE_STATE', time=float(self.data.time))
            raise RuntimeError('NONFINITE_STATE')


def clone_controller(template, perception):
    # Model parameters are constant within this environment. Data, hand FSMs,
    # kinematics, paths and observer history must be independent each reset.
    controller = copy.deepcopy(template, {id(template.model): template.model, id(template.ball_observer): None})
    from .event_observation import make_event_observer
    controller.ball_observer = make_event_observer(controller,controller.estimator,perception=perception,
        sample_period_s=template.ball_observer.sample_period_s,max_age_s=template.ball_observer.max_age_s)
    return controller


def result_reason(result):
    if result['success']:
        if (result['throws'] != result['requested_throws'] or result['catches'] != result['requested_throws']
                or not result['initial_hold'] or result['initial_hold']['continuous_support_s'] < .05-1e-10
                or not result['final_hold'] or result['final_hold']['continuous_support_s'] < .8-1e-10
                or result['airborne_at_end'] or result['failure_event'] or result['drops']
                or result['ball_contacts'] or result['max_height'] >= 1.3
                or result['grasp_assistance'] or result['grasp_constraints']):
            raise ValueError('Unconfirmed complete-task success')
        return 'SUCCESS'
    error = result['error'] or 'INCOMPLETE_TASK'
    reason = (result['failure_event'] or {}).get('reason', error.split(':', 1)[0])
    if reason in ('BALL_COLLISION', 'HIT_ARM', 'ARM_COLLISION') or 'ball-ball contact' in error:
        return 'FORBIDDEN_CONTACT'
    if 'artificial force' in error or 'modified a live ball' in error:
        return 'EXTERNAL_BALL_FORCE'
    return reason


class TaskResidualEnv(gym.Env):
    """Six bounded task parameters per native callback, not a 50 Hz motor action."""
    ENVIRONMENT_VERSION = 5
    OBSERVATION_LAYOUT = 'task_callback_sampled_v1'
    ACTION_LAYOUT = 'throw_velocity_normal_lead_wait_or_reception_xyz_normal_tilt_v1'
    ACTION_COST = .002
    RATE_COST = .002
    SUCCESS_REWARD = 10.
    FAILURE_REWARD = -20.
    UNSAFE = frozenset(('FORBIDDEN_CONTACT', 'POSE_LIMIT', 'NONFINITE_STATE', 'EXTERNAL_BALL_FORCE'))

    @classmethod
    def terminal_reward(cls, reason):
        if reason == 'BUDGET':
            return 0.
        return cls.SUCCESS_REWARD if reason == 'SUCCESS' else cls.FAILURE_REWARD

    def __init__(self, config=None, reference=None, *, estimator=None):
        self.config = config or TaskConfig()
        self.reference = {}
        self.reference_path = None
        if reference is not None:
            self.reference_path = Path(reference)
            record = json.loads(self.reference_path.read_text(encoding='utf-8'))
            reference_estimator = record.get('event_estimator','difference')
            if estimator is not None and estimator!=reference_estimator:
                raise ValueError('Reference and environment estimators differ')
            estimator = reference_estimator
            if (any(record['source_sha256'].get(name) != value for name, value in measured_sources().items())
                    or record['mujoco_version'] != mujoco.__version__
                    or record['requested_throws'] != self.config.throws or result_reason(record['result']) != 'SUCCESS'):
                raise ValueError('Prepare a current sampled reference for this task')
            self.reference = {context_key(row): row for row in record['requests']}
            if len(self.reference) != len(record['requests']):
                raise ValueError('Duplicate reference decision contexts')
        self.estimator = estimator or 'difference'
        self._template = _EpisodeCascade(throws=self.config.throws,estimator=self.estimator)
        shapes = dict(balls=(3, 12), errors=(3, 7), robot=(104,), robot_error=(104,),
                      nominal=(5,), tactile=(2, 5), context=(26,))
        self.observation_space = spaces.Dict({name: spaces.Box(-10., 10., shape, np.float32) for name, shape in shapes.items()})
        self.action_space = spaces.Box(-1., 1., (6,), np.float32)
        self._thread = None
        self._active = False

    def _empty_observation(self):
        return {name: np.zeros(space.shape, np.float32) for name, space in self.observation_space.spaces.items()}

    def _status(self):
        c = self.controller
        return dict(catches=c.catch_count, releases=c.throw_count, elapsed_s=float(c.data.time),
                    completed_ids=sorted(c.caught_events-{0}), recovered_ids=sorted(self._recovered_ids))

    def _decision(self, request):
        if request.kind == 'throw':
            hand = self.controller.hands[request.side]
            flight = hand.get('candidate_flight')
            index = int(request.side == 'left')
            if (flight is not None and flight > 0 and flight in self.controller.caught_events
                    and hand.get('capture_confirmed', False) and hand['held'] == request.ball_id
                    and np.linalg.norm(np.asarray(request.palm_position_m[index])-hand['p']) <= self.controller.LAUNCH_POSITION_TOLERANCE_M
                    and np.linalg.norm(request.palm_velocity_m_s[index]) <= self.controller.LAUNCH_SPEED_LIMIT_M_S):
                self._recovered_ids.add(flight)
        self._messages.put(('request', request, self._status()))
        command = self._commands.get()
        if command is _STOP or self._cancel.is_set():
            raise _Cancelled()
        return command

    def _run(self):
        try:
            result = self.controller.run_complete(timeout=self.config.horizon_s)
            self._messages.put(('result', result, self._status()))
        except _Cancelled:
            pass
        except Exception as error:
            self._messages.put(('error', error, None))

    def _receive(self):
        try:
            message = self._messages.get(timeout=60.)
        except Exception:
            self.close()
            raise
        if message[0] == 'error':
            self.close()
            raise RuntimeError('Task physics worker failed') from message[1]
        self._last_status = message[2]
        self._request = message[1] if message[0] == 'request' else None
        self._result = message[1] if message[0] == 'result' else None
        if self._request is None:
            return self._empty_observation()
        return observation(self._request, self.config, self.reference.get(context_key(self._request)))

    def reset(self, *, seed=None, options=None):
        self.close()
        super().reset(seed=seed)
        if options:
            raise ValueError('Set task perturbations through TaskConfig')
        if self.config.throws != self._template.stop_after:
            raise ValueError('Changing task length requires a new environment')
        self._cancel, self._commands, self._messages = Event(), Queue(), Queue()
        shifts = self.np_random.uniform(-self.config.seat_jitter_m, self.config.seat_jitter_m, (2, 2))[[0, 1, 0]]
        perception = PerceptionConfig(self.config.noise_std_m, self.config.delay_s, self.config.dropout,
                                      int(self.np_random.integers(0, 2**31)))
        self.controller = clone_controller(self._template, perception)
        self.controller.cancel, self.controller.decision = self._cancel, self._decision
        for (q, _), shift in zip(self.controller.balls, shifts):
            self.controller.data.qpos[q:q+2] += shift
        mujoco.mj_forward(self.controller.model, self.controller.data)
        self.controller.ball_observer.update(self.controller.data)
        self.controller.initial_ball_states = [dict(position_m=self.controller.data.qpos[q:q+3].tolist(),
            velocity_m_s=self.controller.data.qvel[v:v+3].tolist()) for q, v in self.controller.balls]
        self._actions = np.zeros((2, 2, 6))
        self._executed_action_peak = 0.
        self._decisions = self._reward_catches = 0
        self._recovered_ids = set()
        self._sums = dict(progress=0., action=0., rate=0., event=0.)
        self._active = True
        self._thread = Thread(target=self._run, name='task-physics', daemon=True)
        self._thread.start()
        obs = self._receive()
        self._observation = obs
        return obs, dict(self._info('RUNNING'), initial_terminal=self._result is not None)

    def _info(self, reason):
        result = self._result
        return dict(self._last_status, reason=reason, is_success=reason == 'SUCCESS',
            requested_throws=self.config.throws, environment_version=self.ENVIRONMENT_VERSION,
            config=asdict(self.config), decisions=self._decisions, episode_return=sum(self._sums.values()),
            max_executed_action=self._executed_action_peak,
            reward_sums=self._sums.copy(), physics_steps=round(self._last_status['elapsed_s']/self.controller.model.opt.timestep),
            failure_event=result['failure_event'] if result else None,
            max_ball_height_m=result['max_height'] if result else None,
            ball_contacts=result['ball_contacts'] if result else None,
            grasp_assistance=result['grasp_assistance'] if result else None,
            grasp_constraints=result['grasp_constraints'] if result else None,
            initial_hold_s=(result['initial_hold'] or {}).get('continuous_support_s', 0.) if result else 0.,
            final_hold_s=(result['final_hold'] or {}).get('continuous_support_s', 0.) if result else 0.)

    def step(self, action):
        if not self._active:
            raise RuntimeError('Reset the task before stepping')
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.):
            raise ValueError('Require six finite actions in [-1, 1]')
        increment = dict(progress=0., action=0., rate=0., event=0.)
        if self._request is not None:
            executed = action*action_authority(self._observation)
            self._executed_action_peak = max(self._executed_action_peak, float(np.max(np.abs(executed))))
            command = decode_action(self._request, executed)
            key = (int(self._request.kind == 'reception'), int(self._request.side == 'left'))
            increment['action'] = -self.ACTION_COST*float(np.mean(executed**2))
            increment['rate'] = -self.RATE_COST*float(np.mean((executed-self._actions[key])**2))
            self._actions[key] = executed
            self._request = None
            self._decisions += 1
            self._commands.put(command)
            obs = self._receive()
        else:
            obs = self._empty_observation()
        self._observation = obs
        increment['progress'] = float(self._last_status['catches']-self._reward_catches)
        self._reward_catches = self._last_status['catches']
        terminated = self._result is not None
        truncated = not terminated and self._decisions >= self.config.max_decisions
        reason = result_reason(self._result) if terminated else 'BUDGET' if truncated else 'RUNNING'
        if terminated:
            increment['event'] = self.terminal_reward(reason)
        for name, value in increment.items():
            self._sums[name] += value
        info = self._info(reason)
        if terminated or truncated:
            self.close()
        return obs, sum(increment.values()), terminated, truncated, info

    def close(self):
        if self._thread is not None:
            self._cancel.set()
            self._commands.put(_STOP)
            self._thread.join(timeout=5.)
            if self._thread.is_alive():
                raise RuntimeError('Task physics worker did not stop')
            self._thread = None
        self._active = False

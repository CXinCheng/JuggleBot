"""Bounded decisions at native throw/reception callbacks, retaining contact feedback."""
import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Callable, Literal

import mujoco
import numpy as np

from jugglebot.humanoid import cascade as C
from jugglebot.humanoid.contact_control import ThrowPath
from jugglebot.humanoid.planning import Quintic
from jugglebot.robustness.observation import BallEstimate
from jugglebot.robustness.perception import PerceptionConfig
from .cascade_env import reference_sources
from .compare_residual import digest
from .complete_task import CompleteCascade


@dataclass(frozen=True)
class TaskRequest:
    kind: Literal['throw', 'reception']
    side: str
    ball_id: int
    time_s: float
    released: int
    attempt: int
    remaining_defer_s: float
    nominal_parameters: tuple[float, ...]
    balls: tuple[BallEstimate | None, ...]
    arm_position_rad: tuple[tuple[float, ...], ...]
    arm_velocity_rad_s: tuple[tuple[float, ...], ...]
    finger_position_rad: tuple[tuple[float, ...], ...]
    finger_velocity_rad_s: tuple[tuple[float, ...], ...]
    palm_position_m: tuple[tuple[float, ...], ...]
    palm_velocity_m_s: tuple[tuple[float, ...], ...]
    contact_normal_force_n: tuple[tuple[float, ...], ...]
    hand_modes: tuple[str, ...]
    physics_step_s: float = .001
    contact_time_s: float = 0.


@dataclass(frozen=True)
class TaskCommand:
    parameters: tuple[float, ...]
    defer_s: float = 0.
    tilt_scale: float = 1.

    def __post_init__(self):
        object.__setattr__(self, 'parameters', tuple(float(x) for x in self.parameters))
        object.__setattr__(self, 'defer_s', float(self.defer_s))
        object.__setattr__(self, 'tilt_scale', float(self.tilt_scale))


THROW_RESIDUAL_LIMITS = np.array([.75, .75, 1.25, .25, .015])
RECEPTION_RESIDUAL_LIMITS = np.array([.06, .06, .06, .25, .25])
MAX_DEFER_S = .025
MAX_TOTAL_DEFER_S = .035
DEFAULT_EVENT_SEED_PLANS = Path(__file__).with_name('event_seed_plans.json')


class DeferralBudgetError(ValueError):
    """A valid individual delay cannot fit in the remaining task budget."""


class RecordedReceptionRejected(RuntimeError):
    """The recorded planner rejected its final reception candidate."""


def decode_action(request, action):
    """Convert six normalized policy outputs to bounded physical parameters."""
    action = np.asarray(action, dtype=float)
    if action.shape != (6,) or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.):
        raise ValueError('Require six finite normalized actions in [-1, 1]')
    if not np.isfinite(request.physics_step_s) or request.physics_step_s <= 0.:
        raise ValueError('Require a finite positive physics timestep')
    nominal = np.asarray(request.nominal_parameters)
    if request.kind == 'throw':
        parameters = nominal+THROW_RESIDUAL_LIMITS*action[:5]
        parameters[4] = np.clip(parameters[4], -C.MAX_FINGER_RELEASE_LEAD_S, C.MAX_FINGER_RELEASE_LEAD_S)
        delay = 0.
        if (action[5] > 0. and request.remaining_defer_s >= C.THROW_REPLAN_INTERVAL_S-1e-12
                and request.attempt+1 < CompleteCascade.MAX_THROW_PLAN_ATTEMPTS):
            tick = request.physics_step_s
            available = int(np.floor((min(request.remaining_defer_s, MAX_DEFER_S)+1e-12)/tick))
            wanted = min(available, round(MAX_DEFER_S*action[5]/tick))
            if wanted*tick >= C.THROW_REPLAN_INTERVAL_S-1e-12:
                delay = wanted*tick
        command = TaskCommand(tuple(float(x) for x in parameters), defer_s=delay)
        validate_command(request, command)
        return command
    bounds = CompleteCascade.reception_bounds(request.side)
    parameters = np.clip(nominal+RECEPTION_RESIDUAL_LIMITS*action[:5], *bounds)
    command = TaskCommand(tuple(float(x) for x in parameters), tilt_scale=1.+.5*max(0., action[5]))
    validate_command(request, command, bounds)
    return command


def validate_command(request, command, reception_bounds=None):
    if request.kind not in ('throw', 'reception'):
        raise ValueError('Unknown task decision kind')
    if not np.isfinite(request.physics_step_s) or request.physics_step_s <= 0.:
        raise ValueError('Require a finite positive physics timestep')
    if not isinstance(command, TaskCommand):
        raise TypeError('A decision callback must return TaskCommand')
    parameters = np.asarray(command.parameters, dtype=float)
    nominal = np.asarray(request.nominal_parameters, dtype=float)
    if (parameters.shape != (5,) or nominal.shape != (5,)
            or not np.all(np.isfinite(parameters)) or not np.all(np.isfinite(nominal)) or not np.isfinite(command.defer_s)
            or not np.isfinite(command.tilt_scale)):
        raise ValueError('Task parameters must be five finite values')
    limits = THROW_RESIDUAL_LIMITS if request.kind == 'throw' else RECEPTION_RESIDUAL_LIMITS
    if np.any(np.abs(parameters-nominal) > limits+1e-12):
        raise ValueError('Task residual exceeds its declared parameter bounds')
    if request.kind == 'throw':
        if (not 0. <= command.defer_s <= MAX_DEFER_S
                or 0. < command.defer_s < C.THROW_REPLAN_INTERVAL_S-1e-12
                or abs(command.defer_s/request.physics_step_s-round(command.defer_s/request.physics_step_s)) > 1e-8
                or command.tilt_scale != 1. or abs(parameters[4]) > C.MAX_FINGER_RELEASE_LEAD_S):
            raise ValueError('Invalid throw delay or finger release lead')
        if command.defer_s > request.remaining_defer_s+1e-12:
            raise DeferralBudgetError('Cumulative throw deferral budget exhausted')
    elif (command.defer_s != 0. or not 1. <= command.tilt_scale <= C.MAX_RECEPTION_TILT_SCALE
          or reception_bounds is None or np.any(parameters < reception_bounds[0]-1e-12)
          or np.any(parameters > reception_bounds[1]+1e-12)):
        raise ValueError('Invalid reception parameters or tilt scale')
    return parameters.copy()


class TaskSpaceCascade(CompleteCascade):
    """Native contact FSM with bounded callback decisions and no launch rollouts.

    Ball inputs to the callback are sampled estimates. The inherited classical
    contact controller still uses simulator contact geometry and ball state.
    This interface is not an end-to-end policy or a robustness certificate.
    """
    STAND_LOOKAHEAD_S = .05
    STAND_ACTIVATION_DISTANCE_M = -.003
    STAND_TARGET_DISTANCE_M = -.001
    STAND_ACCELERATION_GAIN = 400.
    STAND_DAMPING_GAIN = 40.
    LAUNCH_POSITION_TOLERANCE_M = .03
    LAUNCH_SPEED_LIMIT_M_S = .30
    FAST_CAPTURE_SPEED_M_S = 1.8
    FAST_CAPTURE_BRAKING_S = .045
    POSTURE_CLEARANCE_ACTIVATION_M = .14
    POSTURE_CLEARANCE_TARGET_M = .12
    POSTURE_OUTWARD_ACTIVATION_M = .086
    POSTURE_OUTWARD_TARGET_M = .09
    UNCAUGHT_LOOKAHEAD_S = .08
    UNCAUGHT_CLEARANCE_M = .008
    UNCAUGHT_BRAKING_RATE_S = 40.

    def __init__(self, decision: Callable[[TaskRequest], TaskCommand] | None = None,
                 *, ending_guard=True, stand_guard=True, recovery_gate=False,
                 adaptive_braking=True, posture_guard=True, uncaught_braking=False, estimator='difference', **kwargs):
        if estimator not in ('difference','filtered'):
            raise ValueError('Estimator must be difference or filtered')
        self.estimator = estimator
        if kwargs.get('observation_mode', 'sampled') != 'sampled':
            raise ValueError('Task decision observations require sampled ball measurements')
        self.decision = decision
        self.ending_guard = ending_guard
        self.stand_guard = stand_guard
        self.recovery_gate = recovery_gate
        self.recovery_rejections = 0
        self.last_recovery_rejection = None
        self.adaptive_braking = adaptive_braking
        self.fast_capture_brakes = 0
        self.posture_guard = posture_guard
        self.uncaught_braking = uncaught_braking
        self.uncaught_braking_steps = 0
        self.uncaught_braking_first = None
        self.posture_guard_steps = 0
        self.decisions = []
        self._deferrals, self._defer_until = {}, {}
        self.ending_guard_steps = 0
        self.ending_guard_peak_delta_nm = 0.
        self.stand_guard_steps = 0
        self.stand_guard_peak_delta_nm = 0.
        kwargs['observation_mode'] = 'sampled'
        # Optimizer guesses are not fixed commands. Use commands exported from
        # an accepted native run and independently replayed by this controller.
        if kwargs.get('optimizer_inputs') is None:
            kwargs['optimizer_inputs'] = json.loads(DEFAULT_EVENT_SEED_PLANS.read_text(encoding='utf-8'))
        super().__init__(**kwargs)
        self._nominal_inputs = dict(self.warm, **self.execution_warm)
        self._contact_force = self._read_contact_force()
        self._contact_time = float(self.data.time)
        self._stand_geom = self.model.geom('torso_stand').id
        self._wrist_geoms = {}
        for side in self.hands:
            bodies = {self.model.body(side+name).id for name in
                      ('_wrist_roll_link', '_wrist_pitch_link', '_wrist_yaw_link')}
            self._wrist_geoms[side] = tuple(g for g in range(self.model.ngeom)
                if self.model.geom_bodyid[g] in bodies
                and (self.model.geom_contype[g] or self.model.geom_conaffinity[g]))
        self._receive_clearance_geoms = {side:tuple(sorted(set(self._wrist_geoms[side])
            |set(self.palm_geoms[side]))) for side in self.hands}
        if estimator=='filtered':
            from .event_observation import make_event_observer
            previous = self.ball_observer
            self.ball_observer = make_event_observer(self,estimator,perception=previous.perception,
                sample_period_s=previous.sample_period_s,max_age_s=previous.max_age_s)
            self.ball_observer.update(self.data)

    def conditions_summary(self):
        result = super().conditions_summary()
        result.pop('additional_throw_guesses', None)
        result.pop('reception_forecast', None)
        result.pop('release_timing_search', None)
        result.pop('reception_tilt_search', None)
        result.pop('reception_refinement', None)
        result['ball_observation']['rollout_input'] = 'unused; online physics rollouts disabled'
        result['ball_observation']['scope'] = 'current task decisions and reception prediction'
        result['task_decisions'] = dict(online_physics_rollouts=False,
            callback_ball_input='sampled position estimates',
            low_level_contact_input='classical controller simulator contact and ball state',
            throw_residual_limits=THROW_RESIDUAL_LIMITS.tolist(),
            reception_residual_limits=RECEPTION_RESIDUAL_LIMITS.tolist(),
            max_single_defer_s=MAX_DEFER_S, max_total_defer_s=MAX_TOTAL_DEFER_S,
            ending_elbow_guard=self.ending_guard,
            posture_clearance_guard=self.posture_guard,
            launch_recovery=dict(enabled=self.recovery_gate,
                position_tolerance_m=self.LAUNCH_POSITION_TOLERANCE_M,
                speed_limit_m_s=self.LAUNCH_SPEED_LIMIT_M_S),
            capture_braking=dict(adaptive=self.adaptive_braking,
                fast_downward_speed_m_s=self.FAST_CAPTURE_SPEED_M_S,
                fast_braking_s=self.FAST_CAPTURE_BRAKING_S),
            stand_clearance_guard=dict(enabled=self.stand_guard, lookahead_s=self.STAND_LOOKAHEAD_S,
                activation_distance_m=self.STAND_ACTIVATION_DISTANCE_M,
                target_distance_m=self.STAND_TARGET_DISTANCE_M),
            uncaught_reception_braking=dict(enabled=self.stand_guard and self.uncaught_braking,
                lookahead_s=self.UNCAUGHT_LOOKAHEAD_S,clearance_m=self.UNCAUGHT_CLEARANCE_M,
                velocity_decay_rate_s=self.UNCAUGHT_BRAKING_RATE_S),
            acceptance='actual complete-task physical checks')
        return result

    def ready_for_throw(self, side):
        if not super().ready_for_throw(side):
            return False
        # Native readiness checks supported, centered contact and relative
        # speed. Extra world-frame settling is an optional control experiment;
        # it must not delay a supported launch past the next reception window.
        hand = self.hands[side]
        if not self.recovery_gate or hand.get('contact_time', -1.) < 0.:
            return True
        distance = float(np.linalg.norm(self.data.site_xpos[hand['seat']]-hand['p']))
        speed = float(np.linalg.norm(self.spatial(hand)[3:]))
        ready = distance <= self.LAUNCH_POSITION_TOLERANCE_M and speed <= self.LAUNCH_SPEED_LIMIT_M_S
        if not ready and not self.silent:
            self.recovery_rejections += 1
            self.last_recovery_rejection = dict(side=side, time_s=float(self.data.time),
                position_error_m=distance, speed_m_s=speed)
        return bool(ready)

    def _capture_recovery_reference(self, hand, data, motion):
        segments = getattr(motion, 'capture_cushion', None)
        if (not self.adaptive_braking or hand.get('mode') != 'carry' or segments is None
                or getattr(motion, 'capture_braking_adjusted', False)
                or abs(data.time-motion.t0) > 1.1*self.model.opt.timestep
                or motion.initial_linear_velocity[2] >= -self.FAST_CAPTURE_SPEED_M_S):
            return motion
        old_stop, old_recover = segments
        duration = min(self.FAST_CAPTURE_BRAKING_S, .4*motion.duration)
        if duration >= old_stop.duration-1e-12:
            return motion
        start, velocity, initial_acceleration = old_stop.sample(motion.t0)
        goal, goal_velocity, final_acceleration = old_recover.sample(motion.t1)
        stopped = start+.5*duration*velocity
        # Forecast hand dictionaries share motion objects with the live state.
        # Replace the reference rather than mutating the shared object.
        adjusted = copy.copy(motion)
        adjusted.capture_cushion = (
            Quintic(motion.t0, duration, start, velocity, initial_acceleration, stopped, np.zeros(3), np.zeros(3)),
            Quintic(motion.t0+duration, motion.duration-duration,
                stopped, np.zeros(3), np.zeros(3), goal, goal_velocity, final_acceleration))
        adjusted.capture_braking_adjusted = True
        hand['carry'] = adjusted
        if not self.silent:
            self.fast_capture_brakes += 1
        return adjusted

    def follow_palm(self, hand, data, motion, time, target):
        motion = self._capture_recovery_reference(hand, data, motion)
        return super().follow_palm(hand, data, motion, time, target)

    def _read_contact_force(self):
        hands = [self.hands[name] for name in ('right', 'left')]
        forces = np.zeros((2, 5))
        for index, contact in enumerate(self.data.contact):
            if contact.efc_address < 0 or contact.dist > 1e-5:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            other = b if a in self.ball_geom_ids else a if b in self.ball_geom_ids else None
            if other is None:
                continue
            body = self.model.body(self.model.geom_bodyid[other]).name
            for hand_index, hand in enumerate(hands):
                group = 0 if other in self.palm_geoms[('right', 'left')[hand_index]] else None
                for digit_index, digit in enumerate(hand['fingers'].digits):
                    if body.startswith(hand['fingers'].prefix+digit+'_'):
                        group = digit_index+1
                if group is not None:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, index, force)
                    forces[hand_index, group] += max(0., force[0])
        return forces

    def _request(self, kind, side, ball):
        if self.silent:
            raise RuntimeError('Task decisions cannot read private forecast states')
        key = f'{"throw" if kind == "throw" else "return"}:{side}:{self.throw_count}'
        default = self.commands[side] if kind == 'throw' else np.zeros(5)
        nominal = np.asarray(self._nominal_inputs.get(key, default), dtype=float)
        if len(nominal) == 4:
            nominal = np.r_[nominal, 0.]
        hands = [self.hands[name] for name in ('right', 'left')]
        attempt, delayed = self._deferrals.get((side, self.throw_count), (0, 0.))
        vector = lambda value: tuple(float(x) for x in value)
        return TaskRequest(kind, side, ball, float(self.data.time), self.throw_count,
            attempt, max(0., MAX_TOTAL_DEFER_S-delayed), vector(nominal),
            tuple(self.observed_ball(i) for i in range(len(self.balls))),
            tuple(vector(self.data.qpos[h['kin'].qpos_adr]) for h in hands),
            tuple(vector(self.data.qvel[h['kin'].dof_adr]) for h in hands),
            tuple(vector(self.data.qpos[self.model.jnt_qposadr[h['fingers'].joint_ids]]) for h in hands),
            tuple(vector(self.data.qvel[self.model.jnt_dofadr[h['fingers'].joint_ids]]) for h in hands),
            tuple(vector(self.data.site_xpos[h['seat']]) for h in hands),
            tuple(vector(self.spatial(h)[3:]) for h in hands),
            tuple(vector(row) for row in self._contact_force), tuple(h['mode'] for h in hands),
            float(self.model.opt.timestep), self._contact_time)

    def _command(self, request):
        command = TaskCommand(request.nominal_parameters) if self.decision is None else self.decision(request)
        bounds = self.reception_bounds(request.side) if request.kind == 'reception' else None
        parameters = validate_command(request, command, bounds)
        self.decisions.append(dict(kind=request.kind, side=request.side, ball_id=request.ball_id,
            time_s=request.time_s, released=request.released, attempt=request.attempt,
            nominal_parameters=list(request.nominal_parameters), **asdict(command)))
        return command, parameters

    def plan_throw(self, side):
        hand = self.hands[side]
        self._defer_until.pop(side, None)
        request = self._request('throw', side, hand['held'])
        try:
            command, parameters = self._command(request)
        except DeferralBudgetError as error:
            self.failure_event = dict(reason='NO_FEASIBLE_THROW', side=side,
                ball=hand['held'], time=float(self.data.time), attempts=request.attempt+1)
            raise RuntimeError('NO_FEASIBLE_THROW: bounded decision retries exhausted') from error
        if command.defer_s:
            if request.attempt+1 >= self.MAX_THROW_PLAN_ATTEMPTS:
                self.failure_event = dict(reason='NO_FEASIBLE_THROW', side=side,
                    ball=hand['held'], time=float(self.data.time), attempts=request.attempt+1)
                raise RuntimeError('NO_FEASIBLE_THROW: bounded decision retries exhausted')
            delay = max(command.defer_s, C.THROW_REPLAN_INTERVAL_S)
            delayed = MAX_TOTAL_DEFER_S-request.remaining_defer_s+delay
            self._deferrals[(side, self.throw_count)] = (request.attempt+1, delayed)
            self._defer_until[side] = self.data.time+delay
            # The native FSM uses this marker to stay in hold and retry. It is
            # a requested deferral, not a physics-forecast objective value.
            self.last_plan_error = float('inf')
            return None
        mujoco.mj_copyData(hand['kin'].data, self.model, self.data)
        path = ThrowPath(hand['kin'], self.data.site_xpos[hand['seat']].copy(), parameters[:3],
            self.data.qpos[hand['kin'].qpos_adr].copy(), hand['catch_p'], bank=parameters[3], catch_q=hand['catch_q'])
        path.planned_time = self.data.time
        path.finger_release_lead_s = float(parameters[4])
        self.commands[side] = parameters[:4].copy()
        self.last_plan_error = 0.
        return path

    def plan_reception(self, side):
        hand = self.hands[side]
        try:
            command, parameters = self._command(self._request('reception', side, hand['incoming']))
        except RecordedReceptionRejected:
            self.failure_event = dict(reason='NO_FEASIBLE_RECEPTION', time=float(self.data.time),
                                      side=side, ball=hand['incoming'])
            raise
        reference = self.reception_reference(side, parameters, tilt_scale=command.tilt_scale)
        self.apply_reception_reference(hand, *reference)
        hand['return_parameters'] = parameters

    def step(self):
        super().step()
        # Read solved contact forces after step2, not the unsolved contact
        # layout at the planner callback inside the next split step.
        self._contact_force = self._read_contact_force()
        self._contact_time = float(self.data.time)
        for side, deadline in self._defer_until.items():
            if self.hands[side]['mode'] == 'hold':
                self.hands[side]['throw_time'] = max(self.hands[side]['throw_time'], deadline)

    def protect_arm(self, hand, data, torque, response=None, free_acceleration=None):
        torque = super().protect_arm(hand, data, torque, response, free_acceleration)
        torque = self._protect_ending_elbow(hand, data, torque, response, free_acceleration)
        if self.posture_guard and response is not None:
            torque = self._protect_posture_clearance(hand, data, torque, response, free_acceleration)
        if self.stand_guard and response is not None:
            torque = self._protect_stand_clearance(hand, data, torque, response, free_acceleration)
            torque = self._brake_uncaught_reception(hand, data, torque, response, free_acceleration)
        return torque

    def _brake_uncaught_reception(self, hand, data, torque, response, free_acceleration):
        """Stop chasing an uncaught ball when the cushioning tail approaches the stand.

        This only commands bounded arm actuator torques. It neither registers
        a catch nor ends the episode; ordinary physical acceptance still applies.
        """
        reception = hand.get('reception')
        if (not self.uncaught_braking or hand.get('mode')!='receive' or hand.get('held') is not None
                or reception is None or data.time<reception[1].t0):
            return torque
        side,ball = hand['kin'].side,self.ball_geom_ids[hand['incoming']]
        for contact in data.contact:
            other = contact.geom2 if contact.geom1==ball else contact.geom1 if contact.geom2==ball else -1
            if other in self.hand_geoms[side] and contact.dist<=1e-5 and contact.efc_address>=0:
                return torque
        key = (hand['incoming'],reception[1].t0)
        if hand.get('uncaught_braking_key')!=key:
            for geom in self._receive_clearance_geoms[side]:
                separation = self._stand_separation(data,geom,max_distance=.15)
                if separation is None:
                    continue
                distance,gradient,*_ = separation
                speed = float(gradient@data.qvel)
                if distance+self.UNCAUGHT_LOOKAHEAD_S*min(speed,0.)<=self.UNCAUGHT_CLEARANCE_M:
                    hand['uncaught_braking_key'] = key
                    if not self.silent and self.uncaught_braking_first is None:
                        self.uncaught_braking_first = dict(time_s=float(data.time),side=side,distance_m=float(distance))
                    break
        if hand.get('uncaught_braking_key')!=key:
            return torque
        dof = hand['kin'].dof_adr
        acceleration = -self.UNCAUGHT_BRAKING_RATE_S*data.qvel[dof]
        # Abandon the unreachable Cartesian tail. A joint already at its
        # torque limit must not prevent the other joints from braking.
        braking = np.linalg.solve(response[dof],acceleration-free_acceleration[dof])
        if not self.silent:
            self.uncaught_braking_steps += 1
        return np.clip(braking,-hand['force_limits'],hand['force_limits'])

    def _palm_torque_nullspace(self, hand, data, response):
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jp, jr, hand['seat'])
        normal = data.site_xmat[hand['seat']].reshape(3, 3)[:, 2]
        task = np.vstack([jp@response, (np.eye(3)-np.outer(normal, normal))@jr@response])
        return np.eye(7)-np.linalg.pinv(task, rcond=.001)@task

    def _stand_separation(self, data, geom, max_distance=.03):
        points = np.zeros(6)
        distance = mujoco.mj_geomDistance(self.model, data, self._stand_geom, geom, max_distance, points)
        if abs(distance) >= max_distance or abs(distance) < 1e-12:
            return None
        normal = (points[3:]-points[:3])/distance
        point = (points[:3]+points[3:])/2
        bodies = [int(self.model.geom_bodyid[g]) for g in (self._stand_geom, geom)]
        jacobians = [np.zeros((3, self.model.nv)) for _ in bodies]
        for body, jacobian in zip(bodies, jacobians):
            mujoco.mj_jac(self.model, data, jacobian, None, point, body)
        return distance, normal@(jacobians[1]-jacobians[0]), normal, point, bodies

    @staticmethod
    def _bounded_torque_change(torque, correction, limits):
        # Scale the whole null-space direction. Per-joint clipping would
        # change its direction and could disturb the palm task.
        moving = np.abs(correction) > 1e-12
        if not np.any(moving):
            return torque.copy()
        available = np.where(correction > 0., limits-torque, limits+torque)
        fraction = np.clip(np.min(available[moving]/np.abs(correction[moving])), 0., 1.)
        return np.clip(torque+fraction*correction, -limits, limits)

    def _protect_stand_clearance(self, hand, data, torque, response, free_acceleration):
        nullspace = None
        for geom in self._wrist_geoms[hand['kin'].side]:
            separation = self._stand_separation(data, geom)
            if separation is None:
                continue
            distance, gradient, normal, point, bodies = separation
            speed = gradient@data.qvel
            if distance > .015 or distance+self.STAND_LOOKAHEAD_S*min(speed, 0.) > self.STAND_ACTIVATION_DISTANCE_M:
                continue
            dots = [np.zeros((3, self.model.nv)) for _ in bodies]
            for body, dot in zip(bodies, dots):
                mujoco.mj_jacDot(self.model, data, dot, None, point, body)
            row = gradient@response
            bound = (self.STAND_ACCELERATION_GAIN*(self.STAND_TARGET_DISTANCE_M-distance)
                -self.STAND_DAMPING_GAIN*speed-gradient@free_acceleration-normal@(dots[1]-dots[0])@data.qvel)
            if row@torque >= bound:
                continue
            if nullspace is None:
                nullspace = self._palm_torque_nullspace(hand, data, response)
            direction = nullspace@row
            denominator = row@direction
            if denominator <= 1e-12:
                continue
            corrected = self._bounded_torque_change(torque, direction*((bound-row@torque)/denominator), hand['force_limits'])
            self.stand_guard_steps += 1
            self.stand_guard_peak_delta_nm = max(self.stand_guard_peak_delta_nm, float(np.max(np.abs(corrected-torque))))
            torque = corrected
        return torque

    def _protect_ending_elbow(self, hand, data, torque, response, free_acceleration):
        if not self.ending_guard or hand.get('mode') != 'ending' or response is None:
            return torque
        m = self.model
        elbow, shoulder = (m.body(hand['kin'].side+name).id for name in ('_elbow_link', '_shoulder_pitch_link'))
        sign = np.sign(data.xpos[elbow, 1])
        outward = abs(data.xpos[elbow, 1])-abs(data.xpos[shoulder, 1])
        je, js, jde, jds = (np.zeros((3, m.nv)) for _ in range(4))
        mujoco.mj_jacBody(m, data, je, None, elbow)
        mujoco.mj_jacBody(m, data, js, None, shoulder)
        gradient = sign*(je[1]-js[1])
        speed = gradient@data.qvel
        if outward+.08*max(0., speed) < .085:
            return torque
        mujoco.mj_jacDot(m, data, jde, None, data.xpos[elbow], elbow)
        mujoco.mj_jacDot(m, data, jds, None, data.xpos[shoulder], shoulder)
        row = gradient@response
        bound = 400*(.09-outward)-40*speed-gradient@free_acceleration-sign*(jde[1]-jds[1])@data.qvel
        if row@torque <= bound:
            return torque
        direction = self._palm_torque_nullspace(hand, data, response)@row
        denominator = row@direction
        if denominator <= 1e-12:
            return torque
        corrected = np.clip(torque-direction*((row@torque-bound)/denominator), -hand['force_limits'], hand['force_limits'])
        self.ending_guard_steps += 1
        self.ending_guard_peak_delta_nm = max(self.ending_guard_peak_delta_nm, float(np.max(np.abs(corrected-torque))))
        return corrected

    def _protect_posture_clearance(self, hand, data, torque, response, free_acceleration):
        """Brake near elbow limits through available palm-task null space."""
        m = self.model
        elbow, shoulder = (m.body(hand['kin'].side+name).id for name in ('_elbow_link', '_shoulder_pitch_link'))
        outward = abs(data.xpos[elbow, 1])-abs(data.xpos[shoulder, 1])
        clearance = data.xpos[shoulder, 2]-data.xpos[elbow, 2]
        if (outward <= self.POSTURE_OUTWARD_ACTIVATION_M
                and clearance >= self.POSTURE_CLEARANCE_ACTIVATION_M):
            return torque
        je, js, jde, jds = (np.zeros((3, m.nv)) for _ in range(4))
        mujoco.mj_jacBody(m, data, je, None, elbow)
        mujoco.mj_jacBody(m, data, js, None, shoulder)
        mujoco.mj_jacDot(m, data, jde, None, data.xpos[elbow], elbow)
        mujoco.mj_jacDot(m, data, jds, None, data.xpos[shoulder], shoulder)
        sign = np.sign(data.xpos[elbow, 1])
        constraints = []
        if outward > self.POSTURE_OUTWARD_ACTIVATION_M:
            constraints.append((sign*(js[1]-je[1]), sign*(jds[1]-jde[1]),
                                self.POSTURE_OUTWARD_TARGET_M-outward))
        if clearance < self.POSTURE_CLEARANCE_ACTIVATION_M:
            constraints.append((js[2]-je[2], jds[2]-jde[2],
                                clearance-self.POSTURE_CLEARANCE_TARGET_M))
        nullspace = None
        for gradient, derivative, margin in constraints:
            row = gradient@response
            bound = -400.*margin-40.*(gradient@data.qvel)-gradient@free_acceleration-derivative@data.qvel
            if row@torque >= bound:
                continue
            if nullspace is None:
                nullspace = self._palm_torque_nullspace(hand, data, response)
            direction = nullspace@row
            denominator = row@direction
            if denominator > 1e-12:
                torque = self._bounded_torque_change(torque,
                    direction*((bound-row@torque)/denominator), hand['force_limits'])
                self.posture_guard_steps += 1
        return torque


class OracleDecisions:
    """Recorded planner actions for capability tests; never a causal policy."""
    def __init__(self, events, timestep_s=.001):
        if not np.isfinite(timestep_s) or timestep_s <= 0.:
            raise ValueError('Oracle timestamps require a finite positive timestep')
        self.timestep_s = timestep_s
        self.coverage_end_s = max((row['time'] for row in events), default=0.)
        self.events = {(row['kind'], row['side'], round(row['time']/timestep_s)): row
            for row in events if row['kind'] in ('plan', 'return_plan')}

    def __call__(self, request):
        key = ('plan' if request.kind == 'throw' else 'return_plan', request.side,
               round(request.time_s/self.timestep_s))
        row = self.events.get(key)
        if row is None or row['ball'] != request.ball_id:
            reason = 'ORACLE_EXHAUSTED' if request.time_s > self.coverage_end_s+self.timestep_s else 'MISSING_ORACLE_DECISION'
            raise RuntimeError(f'{reason}: no recorded {request.kind} for {request.side} at {request.time_s:.3f} s')
        if request.kind == 'throw':
            if row.get('score', 0.) > .02:
                return TaskCommand(request.nominal_parameters, defer_s=C.THROW_REPLAN_INTERVAL_S)
            command = tuple(row['command'])
            return TaskCommand(command+(0.,) if len(command) == 4 else command)
        diagnostics = row.get('diagnostics', {})
        if not all(diagnostics.get(key, False) for key in ('captured', 'collision_free', 'ready_for_next_throw')):
            raise RecordedReceptionRejected('NO_FEASIBLE_RECEPTION: recorded planner rejected the reception')
        return TaskCommand(tuple(row['parameters']), tilt_scale=row.get('tilt_scale', 1.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle-input', type=Path)
    parser.add_argument('--case', default='seat_2mm')
    parser.add_argument('--seed', type=int, default=410200)
    parser.add_argument('--throws', type=int, default=30)
    parser.add_argument('--timeout', type=float, default=12.)
    parser.add_argument('--disable-ending-guard', action='store_true')
    parser.add_argument('--disable-stand-guard', action='store_true')
    readiness = parser.add_mutually_exclusive_group()
    readiness.add_argument('--recovery-gate', action='store_true',
                           help='Additionally require world-frame palm settling before launch')
    readiness.add_argument('--disable-recovery-gate', dest='recovery_gate', action='store_false',
                           help=argparse.SUPPRESS)
    parser.set_defaults(recovery_gate=False)
    parser.add_argument('--disable-adaptive-braking', action='store_true')
    parser.add_argument('--uncaught-braking', action='store_true')
    parser.add_argument('--estimator',choices=('difference','filtered'),default='difference')
    parser.add_argument('--seed-plans',type=Path,default=DEFAULT_EVENT_SEED_PLANS,
                        help='Fixed event commands, independently validated after native export')
    parser.add_argument('--seat-jitter-mm', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.seat_jitter_mm is not None and (args.oracle_input or not np.isfinite(args.seat_jitter_mm)
                                           or not 0. <= args.seat_jitter_mm <= 10.):
        parser.error('Use reset seat jitter in [0, 10] mm without oracle input')
    if (args.output.exists() or args.seed < 0 or not np.isfinite(args.timeout) or args.timeout <= 0
            or args.throws < 4 or args.throws % 2):
        parser.error('Use a new output, nonnegative seed, even throws >= 4 and finite positive timeout')
    if args.oracle_input and not args.oracle_input.is_file():
        parser.error('Oracle input is missing; generate a native audit with audit_feedback --native-timeout first')
    if not args.seed_plans.is_file():
        parser.error('Fixed event seed plans file is missing')
    seed_plans_hash = digest(args.seed_plans)
    seed_plans = json.loads(args.seed_plans.read_text(encoding='utf-8'))
    from .task_env import measured_sources
    fingerprints = measured_sources()
    settings, policy = ({} if args.seat_jitter_mm is None else dict(seat_jitter_m=args.seat_jitter_mm/1000.)), None
    oracle_hash, oracle_row = None, None
    if args.oracle_input:
        oracle_hash = digest(args.oracle_input)
        record = json.loads(args.oracle_input.read_text(encoding='utf-8'))
        if record['manifest'].get('controller') != 'native' or record['manifest']['requested_throws'] != args.throws:
            parser.error('Oracle input must be a native planning audit for the same task')
        if any(record['manifest']['source_sha256'].get(key) != value for key, value in reference_sources().items()):
            parser.error('Oracle planning sources differ from the current task')
        rows = [row for row in record['records'] if row['case'] == args.case and row['seed'] == args.seed]
        if len(rows) != 1:
            parser.error('Require exactly one matching oracle case and seed')
        settings = rows[0]['settings']
        oracle_row = rows[0]
        if rows[0]['result']['conditions']['mujoco_version'] != mujoco.__version__:
            parser.error('Oracle MuJoCo version differs from the current simulator')
        policy = OracleDecisions(rows[0]['events'], timestep_s=rows[0]['result']['conditions']['timestep_s'])
    if (set(settings)-{'seat_jitter_m', 'noise_std_m', 'delay_s', 'dropout'}
            or not np.isfinite(settings.get('seat_jitter_m', 0.))
            or not 0. <= settings.get('seat_jitter_m', 0.) <= .01):
        parser.error('Unsupported oracle perturbation settings')
    rng = np.random.default_rng(args.seed)
    shifts = rng.uniform(-settings.get('seat_jitter_m', 0.), settings.get('seat_jitter_m', 0.), (2, 2))
    if oracle_row is not None and not np.array_equal(np.asarray(oracle_row['initial_displacement_m']), shifts[[0, 1, 0]]):
        parser.error('Oracle initial displacement differs from the selected seed and settings')
    perception = PerceptionConfig(settings.get('noise_std_m', 0.), settings.get('delay_s', 0.),
                                  settings.get('dropout', 0.), int(rng.integers(0, 2**31)))
    setup_started = time.perf_counter()
    controller = TaskSpaceCascade(policy, throws=args.throws, perception=perception,
        optimizer_inputs=seed_plans,
        ending_guard=not args.disable_ending_guard, stand_guard=not args.disable_stand_guard,
        recovery_gate=args.recovery_gate, adaptive_braking=not args.disable_adaptive_braking,
        uncaught_braking=args.uncaught_braking,estimator=args.estimator)
    for ball, (q, _) in enumerate(controller.balls):
        controller.data.qpos[q:q+2] += shifts[0 if ball in (0, 2) else 1]
    mujoco.mj_forward(controller.model, controller.data)
    controller.ball_observer.reset()
    controller.ball_observer.update(controller.data)
    controller.initial_ball_states = [dict(position_m=controller.data.qpos[q:q+3].tolist(),
        velocity_m_s=controller.data.qvel[v:v+3].tolist()) for q, v in controller.balls]
    setup_wall_s = time.perf_counter()-setup_started
    run_started = time.perf_counter()
    result = controller.run_complete(timeout=args.timeout)
    run_wall_s = time.perf_counter()-run_started
    if fingerprints != measured_sources():
        raise ValueError('Controller source changed during execution')
    if args.oracle_input and digest(args.oracle_input) != oracle_hash:
        raise ValueError('Oracle input changed during execution')
    if digest(args.seed_plans) != seed_plans_hash:
        raise ValueError('Fixed event seed plans changed during execution')
    output = dict(schema_version=1, scope='oracle capability test' if policy else 'zero task residual',
        source_sha256=fingerprints, oracle_sha256=oracle_hash,
        nominal_seed_plans_file=args.seed_plans.as_posix(),nominal_seed_plans_sha256=seed_plans_hash,
        settings=settings, seed=args.seed, initial_displacement_m=shifts[[0, 1, 0]].tolist(),
        setup_wall_s=setup_wall_s, run_wall_s=run_wall_s,
        physics_steps=round(controller.data.time/controller.model.opt.timestep),
        ending_guard=not args.disable_ending_guard, guard_steps=controller.ending_guard_steps,
        stand_guard=not args.disable_stand_guard, stand_guard_steps=controller.stand_guard_steps,
        stand_guard_peak_delta_nm=controller.stand_guard_peak_delta_nm,
        recovery_rejections=controller.recovery_rejections,
        last_recovery_rejection=controller.last_recovery_rejection,
        fast_capture_brakes=controller.fast_capture_brakes,
        uncaught_braking_steps=controller.uncaught_braking_steps,
        uncaught_braking_first=controller.uncaught_braking_first,
        guard_peak_delta_nm=controller.ending_guard_peak_delta_nm, decisions=controller.decisions, result=result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(output, allow_nan=False, separators=(',', ':'))+'\n')
    print(json.dumps(dict(scope=output['scope'], success=result['success'],
        completed_flights=result['catches'], requested_flights=args.throws, error=result['error'])), flush=True)
    if not result['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

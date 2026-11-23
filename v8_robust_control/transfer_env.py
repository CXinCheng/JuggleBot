"""Continuous physical single-ball transfer with bounded motor residuals."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from jugglebot.robustness.observation import BallObserver
from jugglebot.robustness.perception import PerceptionConfig
from .compare_residual import digest, read_json
from .transfer_observation import SupportPositionTracker
from .residual_env import CatchConfig, transition_reward
from .single_ball import SingleBallTransfer, hand_evidence, stable, INITIAL_HOLD_S, FINAL_HOLD_S
from .train_residual import source_hashes


def runtime_sources():
    root = Path(__file__).resolve().parents[1]
    names = ('v8_robust_control/single_ball.py', 'v8_robust_control/transfer_env.py',
             'v8_robust_control/transfer_observation.py',
             'src/jugglebot/humanoid/leap_cascade_seed_plans.json')
    return dict(source_hashes(), **{name: digest(root/name) for name in names})


def reference_sources():
    root = Path(__file__).resolve().parents[1]
    names = ('v8_robust_control/single_ball.py', 'src/jugglebot/humanoid/leap_cascade_seed_plans.json')
    return dict(source_hashes(), **{name: digest(root/name) for name in names},
                **{'v8_robust_control/transfer_env.py:prepare_reference':
                   hashlib.sha256(inspect.getsource(prepare_reference).encode('utf-8')).hexdigest()})


def prepare_reference(folder):
    folder = Path(folder)
    if folder.exists():
        raise ValueError('Reference directory must be new')
    fingerprints = reference_sources()
    generator_sources = runtime_sources()
    controller = SingleBallTransfer('sampled')
    model, data = controller.model, controller.data
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    initial = np.zeros(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, initial, spec)
    controls, states = [], []
    initial_hold = maximum_hold = final_hold = 0.
    dt = float(model.opt.timestep)
    while data.time < 5.:
        controller.step()
        controls.append(data.ctrl.copy())
        states.append(np.r_[data.qpos, data.qvel])
        if controller.throw_count == 0:
            initial_hold = initial_hold+dt if stable(hand_evidence(controller, 'right')) else 0.
            maximum_hold = max(maximum_hold, initial_hold)
        final_hold = final_hold+dt if 1 in controller.caught_events and stable(hand_evidence(controller, 'left')) else 0.
        if final_hold+1e-10 >= FINAL_HOLD_S:
            break
    releases = [event for event in controller.events if event['kind'] == 'release']
    catches = [event for event in controller.events if event['kind'] == 'catch']
    if (maximum_hold+1e-10 < INITIAL_HOLD_S or final_hold+1e-10 < FINAL_HOLD_S
            or len(releases) != 1 or len(catches) != 1 or controller.throw_count != 1):
        raise RuntimeError('Nominal transfer did not pass; do not prepare training data')
    if fingerprints != reference_sources() or generator_sources != runtime_sources():
        raise RuntimeError('Reference source changed during preparation')
    folder.mkdir(parents=True)
    np.savez_compressed(folder/'reference.npz', initial_state=initial, controls=np.asarray(controls), states=np.asarray(states))
    manifest = dict(schema_version=2, task='single_ball_transfer', source_sha256=fingerprints,
                    generator_source_sha256=generator_sources,
                    reference_sha256=digest(folder/'reference.npz'), state_spec=int(spec), timestep_s=dt,
                    release_time_s=releases[0]['separation_time'], reference_duration_s=float(data.time),
                    initial_hold_s=maximum_hold, final_hold_s=final_hold, releases=releases, catches=catches,
                    note='Controls are the offline nominal prior. Recorded ball states are used only for zero-residual verification.')
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


@dataclass(frozen=True)
class TransferConfig:
    control_period_s: float = .02
    horizon_s: float = 1.6
    seat_jitter_m: float = 0.
    noise_std_m: float = 0.
    delay_s: float = 0.
    dropout: float = 0.
    max_cartesian_offset_m: float = .005
    max_joint_offset_rad: float = .02
    max_joint_rate_rad_s: float = .15
    max_grip_offset_rad: float = .1
    max_grip_rate_rad_s: float = .5

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError('Transfer configuration must be finite')
        if (self.control_period_s != .02 or self.horizon_s < 1.1 or not 0 <= self.seat_jitter_m <= .01
                or not 0 < self.max_cartesian_offset_m <= .015 or not 0 < self.max_joint_offset_rad <= .04
                or not 0 < self.max_joint_rate_rad_s <= .3 or not 0 < self.max_grip_offset_rad <= .15
                or not 0 < self.max_grip_rate_rad_s <= .75):
            raise ValueError('Transfer time or action bounds exceed the declared task')
        PerceptionConfig(self.noise_std_m, self.delay_s, self.dropout)


class TransferEnv(gym.Env):
    """191 causal observations; right/left XY and two grip synergies.

    Nominal commands are prepared once. reset() alone may perturb the initial
    seated ball. step() changes motor targets, never ball state or constraints.
    Sensor noise alone cannot disturb an exactly repeated open-loop trajectory;
    seat_jitter_m explicitly supplies physical initial-state uncertainty.
    """
    ENVIRONMENT_VERSION = 2
    OBSERVATION_SIZE = 191
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 50}

    def __init__(self, reference_dir, config=None, render_mode=None):
        self.config = config or TransferConfig()
        self.reference_dir = Path(reference_dir)
        self.reference = read_json(self.reference_dir/'manifest.json')
        if self.reference['schema_version'] != 2 or self.reference['source_sha256'] != reference_sources():
            raise ValueError('Prepare a reference using the current physical runtime')
        if digest(self.reference_dir/'reference.npz') != self.reference['reference_sha256']:
            raise ValueError('Nominal reference checksum differs')
        with np.load(self.reference_dir/'reference.npz', allow_pickle=False) as artifact:
            self.initial_state = artifact['initial_state'].copy()
            self.controls = artifact['controls'].copy()
            self.expected_states = artifact['states'].copy()
        self.controller = SingleBallTransfer('sampled')
        self.model, self.data = self.controller.model, self.controller.data
        self.initial_data = mujoco.MjData(self.model)
        mujoco.mj_copyData(self.initial_data, self.model, self.data)
        self.dt = float(self.model.opt.timestep)
        self.substeps = round(self.config.control_period_s/self.dt)
        self.action_space = spaces.Box(-1., 1., (6,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.OBSERVATION_SIZE,), np.float32)
        self.hands = [self.controller.hands[side] for side in ('right', 'left')]
        self.q, self.v = self.controller.balls[0]
        self.geom = self.controller.ball_geom_ids[0]
        self.force_groups = {}
        for side, prefix in enumerate(('rh_', 'lh_')):
            for geom in self.controller.hand_geoms[('right', 'left')[side]]:
                name = self.model.body(self.model.geom_bodyid[geom]).name
                if name == prefix+'palm':
                    self.force_groups[geom] = (side, 0)
                for group, digit in enumerate(('if_', 'mf_', 'rf_', 'th_'), 1):
                    if name.startswith(prefix+digit):
                        self.force_groups[geom] = (side, group)
        if (self.controls.ndim != 2 or self.controls.shape[1] != self.model.nu
                or self.expected_states.shape != (len(self.controls), self.model.nq+self.model.nv)):
            raise ValueError('Reference dimensions differ from the single-ball model')
        self.render_mode = render_mode
        self._renderer = None
        self.active = False

    def _loads(self):
        loads = np.zeros((2, 10))
        for index, contact in enumerate(self.data.contact):
            other = contact.geom2 if contact.geom1 == self.geom else contact.geom1 if contact.geom2 == self.geom else -1
            group = self.force_groups.get(other)
            if group is None or contact.efc_address < 0 or contact.dist > 1e-5:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            side, digit = group
            loads[side, digit] += max(0., force[0])
            loads[side, digit+5] += np.linalg.norm(force[1:3])
        return loads

    def _estimate(self):
        return self.tracker.estimate(float(self.data.time))

    def permissions(self):
        estimate = self._estimate()
        sensed = (estimate is not None and estimate.valid
                  and self.tracker.uncertainty(float(self.data.time))[0] <= .015)
        return np.array([sensed and not self.released]*2+[sensed]*2
                        +[self.contacts[0] and not self.released, self.contacts[1]], dtype=bool)

    def _observation(self):
        estimate = self._estimate()
        valid = estimate is not None and estimate.valid
        palm_v = [self.controller.spatial(hand)[3:] for hand in self.hands]
        position = np.zeros(3) if not valid else np.asarray(estimate.position_m)-self.data.site_xpos[self.hands[1]['seat']]
        velocity = np.zeros(3) if not valid else np.asarray(estimate.velocity_m_s)-palm_v[1]
        source_position = np.zeros(3) if not valid else np.asarray(estimate.position_m)-self.data.site_xpos[self.hands[0]['seat']]
        source_velocity = np.zeros(3) if not valid else np.asarray(estimate.velocity_m_s)-palm_v[0]
        age = .1 if estimate is None else min(.1, estimate.age_s)
        # Three 14-field ball slots keep the policy interface extensible to
        # two/three balls. Each active slot is relative to both palms; unused
        # slots contain zeros, including their validity flags.
        values = np.r_[np.clip(position, -2., 2.), np.clip(velocity/5., -10., 10.),
                       np.clip(source_position, -2., 2.), np.clip(source_velocity/5., -10., 10.),
                       float(valid), age/.05, np.zeros(28),
                       *[self.data.qpos[hand['kin'].qpos_adr] for hand in self.hands],
                       *[self.data.qvel[hand['kin'].dof_adr]/10. for hand in self.hands],
                       *[self.data.qpos[self.model.jnt_qposadr[hand['fingers'].joint_ids]] for hand in self.hands],
                       *[self.data.qvel[self.model.jnt_dofadr[hand['fingers'].joint_ids]]/10. for hand in self.hands],
                       *[self.data.site_xpos[hand['seat']] for hand in self.hands], *palm_v,
                       np.clip(self.loads.ravel()/10., 0., 10.), self.contacts,
                       self.data.time/self.config.horizon_s, self.joint_offset.ravel()/.04,
                       self.grip_offset/.15, self.permissions()]
        return values.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('Reset options are not supported')
        mujoco.mj_copyData(self.data, self.model, self.initial_data)
        mujoco.mj_setState(self.model, self.data, self.initial_state, self.reference['state_spec'])
        self.controller.failure_event = None
        self.initial_displacement = self.np_random.uniform(-self.config.seat_jitter_m, self.config.seat_jitter_m, 2)
        if self.config.seat_jitter_m:
            self.data.qpos[self.q:self.q+2] += self.initial_displacement
            mujoco.mj_forward(self.model, self.data)
        self.loads = self._loads()
        self.contacts = np.sum(self.loads[:, :5], axis=1) > .02
        perception = PerceptionConfig(self.config.noise_std_m, self.config.delay_s, self.config.dropout,
                                      int(self.np_random.integers(0, 2**31)))
        self.observer = BallObserver([(self.q, self.v)], mode='sampled', position_step_s=self.dt, perception=perception)
        self.tracker = SupportPositionTracker(self.model.opt.gravity, self.dt)
        self.observer.update(self.data)
        if self.observer.observations:
            self.tracker.update(self.observer.observations[0])
        self._notify_support()
        self.joint_offset = np.zeros((2, 7))
        self.grip_offset = np.zeros(2)
        self.last_action = np.zeros(6)
        self.index = 0
        self.released = False
        self.release_time = self.touch_time = None
        self.initial_hold = self.initial_hold_max = self.final_hold = self.absent_source = self.airborne = 0.
        self.initial_validated = False
        self.ever_nonzero = False
        self.reference_state_max_error = 0.
        self.max_ball_height = float(self.data.qpos[self.q+2])
        self.max_joint_offset = self.max_grip_offset = 0.
        self.return_sum = 0.
        self.reward_sums = dict(event=0., shaping=0., action=0., rate=0.)
        self.estimation_errors = {phase: np.zeros(3) for phase in ('held', 'flight', 'contact')}
        self.active = True
        return self._observation(), self._info(None)

    def _notify_support(self):
        side = 0 if self.contacts[0] else 1 if self.contacts[1] else None
        self.tracker.notify_support(float(self.data.time),
            [self.data.site_xpos[hand['seat']] for hand in self.hands],
            [self.controller.spatial(hand)[3:] for hand in self.hands], side)

    def _record_estimation_error(self):
        # Evaluation telemetry only; neither the reward nor policy consumes it.
        estimate = self._estimate()
        if estimate is None or not estimate.valid:
            return
        phase = 'held' if not self.released else 'flight' if self.touch_time is None else 'contact'
        error_p = np.asarray(estimate.position_m)-self.data.qpos[self.q:self.q+3]
        error_v = np.asarray(estimate.velocity_m_s)-self.data.qvel[self.v:self.v+3]
        if not np.all(np.isfinite(np.r_[error_p, error_v])):
            return
        self.estimation_errors[phase] += [1., float(error_p@error_p), float(error_v@error_v)]

    def _targets(self, action):
        for side, hand in enumerate(self.hands):
            desired = np.zeros(7)
            if np.any(action[2*side:2*side+2]):
                jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jp, jr, hand['seat'])
                dof = hand['kin'].dof_adr
                jacobian = np.vstack([jp[:, dof], .1*jr[:, dof]])
                displacement = np.r_[action[2*side:2*side+2]*self.config.max_cartesian_offset_m, np.zeros(4)]
                desired = np.clip(np.linalg.lstsq(jacobian, displacement, rcond=.001)[0],
                                  -self.config.max_joint_offset_rad, self.config.max_joint_offset_rad)
            maximum = self.config.max_joint_rate_rad_s*self.config.control_period_s
            self.joint_offset[side] += np.clip(desired-self.joint_offset[side], -maximum, maximum)
            grip = action[4+side]*self.config.max_grip_offset_rad
            if self.loads[side, 0] <= .02:
                grip = min(grip, 0.)
            maximum = self.config.max_grip_rate_rad_s*self.config.control_period_s
            self.grip_offset[side] += np.clip(grip-self.grip_offset[side], -maximum, maximum)

    def _monitor(self):
        self.max_ball_height = max(self.max_ball_height, float(self.data.qpos[self.q+2]))
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            return 'NONFINITE_STATE'
        self.controller.check_forbidden_contacts()
        right, left = hand_evidence(self.controller, 'right'), hand_evidence(self.controller, 'left')
        if left.external_ball_force:
            return 'EXTERNAL_BALL_FORCE'
        if left.forbidden_contact:
            return 'FORBIDDEN_CONTACT'
        if left.ball_height_m < .4:
            return 'BALL_DROPPED'
        if left.ball_height_m >= 1.3:
            return 'BALL_TOO_HIGH'
        if not self.released:
            self.initial_hold = self.initial_hold+self.dt if stable(right) else 0.
            self.initial_hold_max = max(self.initial_hold_max, self.initial_hold)
            self.initial_validated |= self.initial_hold+1e-10 >= INITIAL_HOLD_S
            self.absent_source = self.absent_source+self.dt if not self.contacts[0] else 0.
            if self.absent_source+1e-10 >= .012 and self.initial_validated:
                self.released = True
                self.release_time = float(self.data.time)-self.absent_source+self.dt
                if abs(self.release_time-self.reference['release_time_s']) > .05:
                    return 'RELEASE_WINDOW'
            if not self.released and self.data.time > self.reference['release_time_s']+.05:
                return 'NO_RELEASE'
        if self.released:
            # A second source-hand impulse must not be counted as the original
            # flight. Catch cushioning remains allowed at the target hand.
            if self.contacts[0]:
                return 'SOURCE_RECONTACT'
            if self.touch_time is None:
                if not np.any(self.contacts):
                    self.airborne += self.dt
                if self.loads[1, 0] > .02:
                    self.touch_time = float(self.data.time)
                    flight = self.touch_time-self.release_time
                    if not .4 <= flight <= .75 or self.airborne < .3:
                        return 'INVALID_FLIGHT'
                elif self.data.time-self.release_time > .75:
                    return 'FLIGHT_TIMEOUT'
            self.final_hold = self.final_hold+self.dt if self.touch_time is not None and stable(left) else 0.
            if self.final_hold+1e-10 >= FINAL_HOLD_S:
                return 'SUCCESS'
            if self.touch_time is not None and not self.contacts[1] and left.normal_m > .12:
                return 'MISSED_CATCH_WINDOW'
        return None

    def step(self, action):
        if not self.active:
            raise RuntimeError('Reset before stepping or after termination')
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.all(np.isfinite(action)):
            raise ValueError('Expected six finite motor residuals')
        effective = np.clip(action, -1., 1.)*self.permissions()
        self.ever_nonzero |= bool(np.any(effective))
        self._targets(effective)
        self.max_joint_offset = max(self.max_joint_offset, float(np.max(np.abs(self.joint_offset))))
        self.max_grip_offset = max(self.max_grip_offset, float(np.max(np.abs(self.grip_offset))))
        start = float(self.data.time)
        loads = np.zeros((2, 10))
        samples = 0
        reason = None
        for _ in range(self.substeps):
            mujoco.mj_step1(self.model, self.data)
            self.data.ctrl[:] = self.controls[min(self.index, len(self.controls)-1)]
            for side, hand in enumerate(self.hands):
                self.data.ctrl[hand['acts']] += self.joint_offset[side]
                fingers = hand['fingers']
                target = self.data.ctrl[fingers.act_indices]+self.grip_offset[side]*fingers.grip_map.sum(axis=1)
                bounds = self.model.actuator_ctrlrange[fingers.act_indices]
                self.data.ctrl[fingers.act_indices] = np.clip(target, bounds[:, 0], bounds[:, 1])
            mujoco.mj_step2(self.model, self.data)
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_comVel(self.model, self.data)
            self.loads = self._loads()
            loads += self.loads
            samples += 1
            self.contacts = np.sum(self.loads[:, :5], axis=1) > .02
            self.observer.update(self.data)
            self._notify_support()
            if self.observer.observations:
                self.tracker.update(self.observer.observations[0])
            if not self.ever_nonzero and not self.config.seat_jitter_m and self.index < len(self.expected_states):
                error = np.max(np.abs(np.r_[self.data.qpos, self.data.qvel]-self.expected_states[self.index]))
                self.reference_state_max_error = max(self.reference_state_max_error, float(error))
            self.index += 1
            try:
                reason = self._monitor()
            except RuntimeError as failure:
                event = self.controller.failure_event
                code = event['reason'] if event is not None else str(failure)
                reason = {'HIT_ARM': 'FORBIDDEN_CONTACT', 'ARM_COLLISION': 'FORBIDDEN_CONTACT',
                          'BALL_COLLISION': 'FORBIDDEN_CONTACT', 'DROPPED': 'BALL_DROPPED'}.get(code, code)
            if reason is None and self.data.time >= self.config.horizon_s-1e-10:
                reason = 'TASK_DEADLINE'
            if reason is not None:
                break
        self.loads = loads/max(1, samples)
        reward, terms = transition_reward(CatchConfig(), 0., 0., self.data.time-start,
                                           effective, self.last_action, reason)
        self.last_action = effective
        self.return_sum += reward
        for key, value in terms.items():
            self.reward_sums[key] += value
        self.active = reason is None
        self._record_estimation_error()
        info = self._info(reason)
        info['reward_terms'] = terms
        return self._observation(), reward, not self.active, False, info

    def _info(self, reason):
        return dict(task='single_ball_transfer', environment_version=self.ENVIRONMENT_VERSION,
                    config=asdict(self.config),
                    physics_steps=self.index,
                    is_success=reason == 'SUCCESS', reason=reason, simulated_time_s=float(self.data.time),
                    failure_event=None if self.controller.failure_event is None else dict(self.controller.failure_event),
                    release_time_s=self.release_time, palm_touch_time_s=self.touch_time, airborne_s=self.airborne,
                    initial_hold_max_s=self.initial_hold_max, final_hold_s=self.final_hold,
                    episode_return=self.return_sum, reward_sums=dict(self.reward_sums),
                    initial_displacement_m=self.initial_displacement.tolist(),
                    zero_reference_verified=not self.ever_nonzero and not self.config.seat_jitter_m,
                    reference_state_max_error=self.reference_state_max_error,
                    max_ball_height_m=self.max_ball_height, max_joint_offset_rad=self.max_joint_offset,
                    max_grip_offset_rad=self.max_grip_offset,
                    estimation={phase: dict(control_samples=int(values[0]),
                        position_rmse_m=float(np.sqrt(values[1]/(3*values[0]))) if values[0] else None,
                        velocity_rmse_m_s=float(np.sqrt(values[2]/(3*values[0]))) if values[0] else None)
                        for phase, values in self.estimation_errors.items()},
                    joint_offset_rad=self.joint_offset.tolist(), grip_offset_rad=self.grip_offset.tolist())

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera='front')
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--output-dir', type=Path, required=True)
    check = commands.add_parser('check')
    check.add_argument('--reference-dir', type=Path, required=True)
    check.add_argument('--seat-jitter-mm', type=float, default=0.)
    check.add_argument('--episodes', type=int, default=5)
    check.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare_reference(args.output_dir), indent=2))
        return
    if args.output.exists() or args.episodes < 1:
        parser.error('Use a new output and a positive episode count')
    env = TransferEnv(args.reference_dir, TransferConfig(seat_jitter_m=args.seat_jitter_mm/1000.))
    records = []
    started = time.perf_counter()
    controls = 0
    try:
        for seed in range(args.episodes):
            env.reset(seed=seed)
            while True:
                _, _, done, truncated, info = env.step(np.zeros(6))
                controls += 1
                if done or truncated:
                    break
            records.append(dict(seed=seed, **info))
    finally:
        env.close()
    result = dict(scope='full_single_ball_zero_residual', config=asdict(env.config), source_sha256=runtime_sources(),
                  reference_sha256=env.reference['reference_sha256'], records=records,
                  wall_time_s=time.perf_counter()-started, control_steps=controls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('Successes', sum(row['is_success'] for row in records), '/', len(records),
          'control steps/s', controls/result['wall_time_s'], flush=True)
    if args.seat_jitter_mm == 0. and any(not row['is_success'] or row['reference_state_max_error'] != 0. for row in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

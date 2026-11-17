"""Two-ball exchange and complete three-ball residual-control environments."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import inspect
import json
import time
import copy
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from jugglebot.robustness.observation import BallObserver
from jugglebot.robustness.perception import PerceptionConfig
from jugglebot.robustness.controller import ObservedCascade
from jugglebot.humanoid import cascade as C
from .transfer_env import TransferEnv, TransferConfig, runtime_sources as transfer_sources
from .transfer_observation import SupportPositionTracker
from .complete_task import CompleteCascade, complete_record
from .compare_residual import digest


class TwoBallTransfer(ObservedCascade):
    def __init__(self):
        super().__init__(n_balls=2, observation_mode='sampled', obs_noise=0.)

    def ready_for_throw(self, side):
        if not self.silent and self.throw_count >= 2:
            return False
        return super().ready_for_throw(side)


def reference_sources():
    record = complete_record(None, 30, 40.)
    return dict(record['source_sha256'], **record['task_inputs_sha256'], **transfer_sources(),
                **{'v8_robust_control/cascade_env.py:TwoBallTransfer':
                   hashlib.sha256(inspect.getsource(TwoBallTransfer).encode('utf-8')).hexdigest(),
                   'v8_robust_control/cascade_env.py:prepare_reference':
                   hashlib.sha256(inspect.getsource(prepare_reference).encode('utf-8')).hexdigest(),
                   'v8_robust_control/cascade_env.py:nominal_support':
                    hashlib.sha256(inspect.getsource(nominal_support).encode('utf-8')).hexdigest(),
                    'v8_robust_control/cascade_env.py:CascadeEnv':
                    hashlib.sha256(inspect.getsource(CascadeEnv).encode('utf-8')).hexdigest()})


def runtime_sources():
    return dict(reference_sources(), **transfer_sources(),
                **{'v8_robust_control/cascade_env.py': digest(Path(__file__))})


class CascadeEnv(TransferEnv):
    ENVIRONMENT_VERSION = 5
    OBSERVATION_LAYOUT = 'left-relative state and nominal-estimate innovation'
    ACTION_LAYOUT = 'XY feedback gains and finger feedback gains'
    TIME_SCALE_S = 40.
    CATCH_PROGRESS_REWARD = 1.
    MAX_WRAPPING_OFFSET_RAD = .55
    MAX_WRAPPING_RATE_RAD_S = 5.

    def __init__(self, reference_dir, config=None, render_mode=None, *, _preparing=False):
        self.reference_dir = Path(reference_dir)
        self.reference = json.loads((self.reference_dir/'manifest.json').read_text(encoding='utf-8'))
        if (self.reference['schema_version'] != 3 or self.reference['source_sha256'] != reference_sources()
                or self.reference['reference_sha256'] != digest(self.reference_dir/'reference.npz')
                or self.reference['mujoco_version'] != mujoco.__version__
                or not _preparing and self.reference['phase'] != 'ready'):
            raise ValueError('Prepare the cascade reference with the current runtime')
        self.task = self.reference['task']
        if self.task not in ('two', 'complete'):
            raise ValueError('Unknown cascade task')
        self.controller = TwoBallTransfer() if self.task == 'two' else CompleteCascade(
            throws=self.reference['requested_throws'], observation_mode='sampled')
        self.model, self.data = self.controller.model, self.controller.data
        self.count = len(self.controller.balls)
        self.release_reference = [event for event in self.reference.get('events', self.reference['result'].get('events'))
                                  if event['kind'] == 'release']
        self.config = config or TransferConfig(horizon_s=self.reference['duration_s']+.5)
        self.target_throws = len(self.release_reference)
        self.initial_sides = [0, 1] if self.count == 2 else [0, 1, 0]
        self.final_sides = self.initial_sides.copy()
        for event in self.release_reference:
            self.final_sides[event['ball']] = 1-int(event['side'] == 'left')
        with np.load(self.reference_dir/'reference.npz', allow_pickle=False) as bundle:
            self.initial_state, self.controls, self.expected_states = (
                bundle[key].copy() for key in ('initial_state', 'controls', 'states'))
            self.finger_reference = bundle['finger_corrections'].copy() if 'finger_corrections' in bundle else None
            self.estimate_reference = bundle['estimate_reference'].copy() if 'estimate_reference' in bundle else None
            self.estimate_valid = bundle['estimate_valid'].copy() if 'estimate_valid' in bundle else None
            self.palm_reference = bundle['palm_reference'].copy() if 'palm_reference' in bundle else None
        if any(value is None for value in (self.finger_reference, self.estimate_reference, self.estimate_valid, self.palm_reference)) and not _preparing:
            raise ValueError('Prepare the measured nominal feedback references')
        self.initial_data = mujoco.MjData(self.model)
        mujoco.mj_copyData(self.initial_data, self.model, self.data)
        self.dt = float(self.model.opt.timestep)
        self.substeps = round(self.config.control_period_s/self.dt)
        self.action_space = spaces.Box(-1., 1., (6,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (191,), np.float32)
        if (self.controls.ndim != 2 or self.controls.shape[1] != self.model.nu or not len(self.controls)
                or self.expected_states.shape[1] != self.model.nq+self.model.nv
                or self.finger_reference is not None and self.finger_reference.shape != (len(self.controls), self.count, 2, 16)
                or self.estimate_reference is not None and self.estimate_reference.shape != (len(self.controls), self.count, 6)
                or self.estimate_valid is not None and self.estimate_valid.shape != (len(self.controls), self.count)
                or self.palm_reference is not None and self.palm_reference.shape != (len(self.controls), 2, 6)
                or not len(self.release_reference) == self.reference['requested_throws']):
            raise ValueError('Reference dimensions or release count differ')
        self.hands = [self.controller.hands[side] for side in ('right', 'left')]
        self.q, self.v = self.controller.balls[0]
        self.geom = self.controller.ball_geom_ids[0]
        self.geom_ball = {geom: index for index, geom in enumerate(self.controller.ball_geom_ids)}
        self.force_groups = {}
        for side, prefix in enumerate(('rh_', 'lh_')):
            for geom in self.controller.hand_geoms[('right', 'left')[side]]:
                name = self.model.body(self.model.geom_bodyid[geom]).name
                if name == prefix+'palm':
                    self.force_groups[geom] = (side, 0)
                for group, digit in enumerate(('if_', 'mf_', 'rf_', 'th_'), 1):
                    if name.startswith(prefix+digit):
                        self.force_groups[geom] = (side, group)
        self.render_mode, self._renderer, self.active = render_mode, None, False
        self.hand_defaults = [{key: copy.deepcopy(value) for key, value in hand.items()
                               if key not in ('kin', 'fingers')} for hand in self.hands]
        if self.task == 'complete':
            self.controller.reception_ball_state = self._ending_ball_state
        self.nominal_controls = self.controls.copy()
        self.reference_touch_times = {event['id']: event['palm_touch_time']
            for event in self.reference.get('nominal_zero_result', {}).get('events', [])
            if event['kind'] == 'catch' and not event.get('ending')}
        self.opening_intervals = [[], []]
        events = self.reference['events']
        for event in events:
            if event['side'] not in ('right', 'left'):
                continue
            side = int(event['side'] == 'left')
            if event['kind'] == 'stroke':
                release = next(candidate for candidate in events if candidate['kind'] == 'release'
                    and candidate['ball'] == event['ball'] and candidate['side'] == event['side']
                    and candidate['time'] >= event['time'])
                self.opening_intervals[side].append((event['time'], release['time']+.02))
            elif event['kind'] == 'end_start':
                self.opening_intervals[side].append((event['time'], float('inf')))

    def _releasing(self, side):
        return any(start-1e-12 <= self.data.time <= stop for start, stop in self.opening_intervals[side])

    def _wrapping_ball(self, side):
        if self.task == 'complete' and side == 0 and self.controller.end is not None:
            return None
        touching = np.flatnonzero(self.ball_contacts[:, side])
        if len(touching) != 1 or self._releasing(side):
            return None
        ball = int(touching[0])
        if self.touch_times[ball] is None or self.ball_loads[ball, side, 0] <= .02:
            return None
        estimate = self.trackers[ball].estimate(float(self.data.time))
        if estimate is None or not estimate.valid or self.trackers[ball].uncertainty(float(self.data.time))[0] > .015:
            return None
        return ball, estimate

    def _targets(self, action):
        self.feedback_gains = .5*(1.+np.clip(action[:4], -1., 1.)).reshape(2, 2)
        self.xy_innovation = np.zeros((2, 2))
        for side in (0, 1):
            innovation = self._xy_innovation(side)
            if (innovation is not None and not self._releasing(side)
                    and not (self.task == 'complete' and side == 0 and self.controller.end is not None)):
                self.xy_innovation[side] = innovation
        self.cartesian_request = self.feedback_gains*np.clip(self.xy_innovation,
            -self.config.max_cartesian_offset_m, self.config.max_cartesian_offset_m)
        # Motor corrections vanish at zero measured reference error for every
        # policy output. The reference never replaces the live ball state.
        super()._targets(np.r_[self.cartesian_request.ravel()/self.config.max_cartesian_offset_m, 0., 0.])
        self.wrapping_gain = 1.+np.clip(action[4:], -1., 1.)

    def _reference_index(self):
        return max(0, min(len(self.controls)-1, round(float(self.data.time)/self.dt)-1))

    def _xy_innovation(self, side):
        if self.estimate_reference is None:
            return None
        index = self._reference_index()
        held, incoming = [], []
        for ball, tracker in enumerate(self.trackers):
            estimate = tracker.estimate(float(self.data.time))
            if (estimate is None or not estimate.valid or not self.estimate_valid[index, ball]
                    or tracker.uncertainty(float(self.data.time))[0] > .015):
                continue
            nominal = self.estimate_reference[index, ball]
            if self.ball_contacts[ball, side]:
                relative = np.asarray(estimate.position_m)-self.data.site_xpos[self.hands[side]['seat']]
                nominal_relative = nominal[:3]-self.palm_reference[index, side, :3]
                held.append((relative-nominal_relative)[:2])
            elif self.flight_ids[ball] is not None and 1-self.flight_sources[ball] == side:
                arrival = self.reference_touch_times.get(self.flight_ids[ball])
                if arrival is not None:
                    remaining = np.clip(arrival-float(self.data.time), 0., .75)
                    error = (np.asarray(estimate.position_m)-nominal[:3]
                             +remaining*(np.asarray(estimate.velocity_m_s)-nominal[3:]))[:2]
                    incoming.append((arrival, error))
        if held:
            return np.mean(held, axis=0)
        return min(incoming, key=lambda item: item[0])[1] if incoming else None

    def _finger_correction(self, ball, side):
        estimate = self.trackers[ball].estimate(float(self.data.time))
        if estimate is None or not estimate.valid:
            return None
        hand = self.hands[side]
        rotation = self.data.site_xmat[hand['seat']].reshape(3, 3)
        local = rotation.T@(np.asarray(estimate.position_m)-self.data.site_xpos[hand['seat']])
        return np.clip(hand['fingers'].wrapping_correction(local-[0., 0., .022]),
                       -self.MAX_WRAPPING_OFFSET_RAD, self.MAX_WRAPPING_OFFSET_RAD)

    def _wrapping_target(self):
        target = np.zeros((2, 16))
        if self.finger_reference is None:
            return target
        index = max(0, min(len(self.finger_reference)-1, round(float(self.data.time)/self.dt)-1))
        for side in (0, 1):
            selected = self._wrapping_ball(side)
            if selected is not None:
                ball, _ = selected
                innovation = self._finger_correction(ball, side)-self.finger_reference[index, ball, side]
                target[side] = np.clip(self.wrapping_gain[side]*innovation,
                                      -self.MAX_WRAPPING_OFFSET_RAD, self.MAX_WRAPPING_OFFSET_RAD)
        return target

    def _ending_ball_state(self, ball):
        estimate = self.trackers[ball].estimate(float(self.data.time))
        if estimate is None or not estimate.valid or self.release_times[ball] is None:
            return None
        # The contact filter carries a support-velocity prior across release.
        # Fit only delivered samples from this flight for a long interception.
        frames = [frame for frame in self.trackers[ball].history if frame.sample_time_s > self.release_times[ball]]
        if len(frames) < 3:
            return None
        latest = frames[-1].sample_time_s
        times = np.array([frame.sample_time_s-latest for frame in frames])
        gravity = self.model.opt.gravity
        positions = np.array([frame.position_m for frame in frames])
        positions -= .5*(times**2+self.dt*times)[:, None]*gravity
        position, velocity = np.linalg.lstsq(np.column_stack([np.ones(len(times)), times]), positions, rcond=None)[0]
        age = float(self.data.time)-latest
        return (position+age*velocity+.5*(age**2+self.dt*age)*gravity, velocity+age*gravity)

    def _ending_control(self):
        if self.task != 'complete' or self.next_release != self.target_throws:
            return
        controller = self.controller
        if controller.end is None:
            retained = []
            hand = self.hands[0]
            for ball, tracker in enumerate(self.trackers):
                if self.flight_ids[ball] == self.target_throws or self.touch_times[ball] is None:
                    continue
                estimate = tracker.estimate(float(self.data.time))
                if estimate is None or not estimate.valid or self.data.time-self.touch_times[ball] < .07:
                    continue
                local = self.data.site_xmat[hand['seat']].reshape(3, 3).T@(
                    np.asarray(estimate.position_m)-self.data.site_xpos[hand['seat']])
                relative = np.asarray(estimate.velocity_m_s)-controller.spatial(hand)[3:]
                digits = controller.supporting_digits(ball, 'right')
                if (self.ball_loads[ball, 0, 0] > .02 and 'th' in digits and len(digits) >= 2
                        and np.linalg.norm(relative) < .15 and np.linalg.norm(local[:2]) < .04
                        and .01 < local[2] < .06):
                    retained.append(ball)
            if len(retained) != 1:
                return
            controller.targets = {ball: ('right', 'left')[1-self.flight_sources[ball]]
                for ball in range(self.count) if self.flight_ids[ball] == self.target_throws}
            hand['mode'], hand['held'], hand['two_phase'] = 'hold', retained[0], False
            hand['q'] = self.data.qpos[hand['kin'].qpos_adr].copy()
            hand['p'] = self.data.site_xpos[hand['seat']].copy()
            controller._maybe_start_end()
        if controller.end is None:
            return
        # This is an analytic interception and servo update, with no forward
        # physics trial. Only next-step motor commands are replaced.
        previous = self.data.ctrl.copy()
        try:
            controller._end_control()
            index = min(len(self.controls)-1, round(float(self.data.time)/self.dt))
            hand = self.hands[0]
            self.controls[index, hand['acts']] = self.data.ctrl[hand['acts']]
            self.controls[index, hand['fingers'].act_indices] = controller.holding_fingers(hand, float(self.data.time))
        finally:
            self.data.ctrl[:] = previous

    def _loads(self):
        self.ball_loads = np.zeros((self.count, 2, 10))
        self.pair_support = False
        for index, contact in enumerate(self.data.contact):
            if contact.efc_address < 0 or contact.dist > 1e-5:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            ball = self.geom_ball.get(a, self.geom_ball.get(b))
            if ball is None:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            if self.count == 3 and {a, b} == {self.controller.ball_geom_ids[0], self.controller.ball_geom_ids[2]}:
                self.pair_support |= force[0] > .02
            group = self.force_groups.get(b if a in self.geom_ball else a)
            if group is not None:
                side, digit = group
                self.ball_loads[ball, side, digit] += max(0., force[0])
                self.ball_loads[ball, side, digit+5] += np.linalg.norm(force[1:3])
        self.ball_contacts = self.ball_loads[:, :, :5].sum(axis=2) > .02
        if (self.count == 3 and self.controller.throw_count == 0 and self.pair_support
                and self.controller._pair_exempt()):
            self.ball_contacts[0, 0] |= self.ball_contacts[2, 0]
            self.ball_contacts[2, 0] |= self.ball_contacts[0, 0]
        return self.ball_loads.sum(axis=0)

    def _notify_support(self):
        positions = [self.data.site_xpos[hand['seat']] for hand in self.hands]
        velocities = [self.controller.spatial(hand)[3:] for hand in self.hands]
        for ball, tracker in enumerate(self.trackers):
            contacts = self.ball_contacts[ball]
            side = 0 if contacts[0] else 1 if contacts[1] else None
            tracker.notify_support(float(self.data.time), positions, velocities, side)
            if len(self.observer.observations) > ball:
                tracker.update(self.observer.observations[ball])
        target = self._wrapping_target()
        maximum = self.MAX_WRAPPING_RATE_RAD_S*self.dt
        self.wrapping_offset += np.clip(target-self.wrapping_offset, -maximum, maximum)
        self.max_wrapping_offset = max(self.max_wrapping_offset, float(np.max(np.abs(self.wrapping_offset))))
        index = min(len(self.controls)-1, round(float(self.data.time)/self.dt))
        self.controls[index] = self.nominal_controls[index]
        for side, hand in enumerate(self.hands):
            self.controls[index, hand['fingers'].act_indices] += self.wrapping_offset[side]
        self._ending_control()

    def permissions(self):
        sensed = [self._xy_innovation(side) is not None for side in (0, 1)]
        masks = np.array([sensed[0]]*2+[sensed[1]]*2+[False, False], dtype=bool)
        for side in (0, 1):
            if self._releasing(side):
                masks[2*side:2*side+2] = False
            if self.task == 'complete' and side == 0 and self.controller.end is not None:
                masks[:2] = False
            masks[4+side] = self._wrapping_ball(side) is not None
        return masks

    def _observation(self):
        values = super()._observation()
        velocities = [self.controller.spatial(hand)[3:] for hand in self.hands]
        index = self._reference_index()
        for ball in range(self.count):
            estimate = self.trackers[ball].estimate(float(self.data.time))
            if estimate is None or not estimate.valid:
                values[ball*14+13] = 2.
                continue
            position, velocity = np.asarray(estimate.position_m), np.asarray(estimate.velocity_m_s)
            innovation = np.zeros(6)
            if self.estimate_reference is not None and self.estimate_valid[index, ball]:
                innovation = np.r_[position, velocity]-self.estimate_reference[index, ball]
            values[ball*14:ball*14+14] = np.r_[
                np.clip(position-self.data.site_xpos[self.hands[1]['seat']], -2., 2.),
                np.clip((velocity-velocities[1])/5., -10., 10.),
                np.clip(innovation[:3], -2., 2.),
                np.clip(innovation[3:]/5., -10., 10.), 1., min(.1, estimate.age_s)/.05]
        # Physical time and cadence keep the same meaning when the requested
        # number of throws changes. These are commanded phase, not ball truth.
        values[168] = self.data.time/self.TIME_SCALE_S
        phase = 2*np.pi*(self.data.time-self.release_reference[0]['separation_time'])/.64
        values[183:185] = [np.sin(phase), np.cos(phase)]
        return values

    def step(self, action):
        previous = len(self.caught_ids)
        observation, reward, done, truncated, info = super().step(action)
        progress = self.CATCH_PROGRESS_REWARD*(len(self.caught_ids)-previous)/self.target_throws
        self.return_sum += progress
        self.reward_sums['progress'] += progress
        terms = dict(info['reward_terms'], progress=progress)
        return observation, reward+progress, done, truncated, dict(self._info(info['reason']), reward_terms=terms)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if options:
            raise ValueError('No reset options')
        self.wrapping_gain = np.ones(2)
        self.feedback_gains = np.full((2, 2), .5)
        self.xy_innovation = self.cartesian_request = np.zeros((2, 2))
        self.wrapping_offset = np.zeros((2, 16))
        self.max_wrapping_offset = 0.
        self.controls = self.nominal_controls.copy()
        self.touch_times = [None]*self.count
        self.next_release = 0
        if self.task == 'complete':
            self.controller.end = self.controller.end_result = None
            self.controller.stopped = False
        for hand, saved in zip(self.hands, self.hand_defaults):
            for key in list(hand):
                if key not in ('kin', 'fingers'):
                    del hand[key]
            hand.update(copy.deepcopy(saved))
        mujoco.mj_copyData(self.data, self.model, self.initial_data)
        mujoco.mj_setState(self.model, self.data, self.initial_state, self.reference['state_spec'])
        self.controller.throw_count = 0
        self.controller.failure_event = None
        shifts = self.np_random.uniform(-self.config.seat_jitter_m, self.config.seat_jitter_m, (2, 2))
        self.initial_displacement = np.array([shifts[side] for side in self.initial_sides])
        if self.config.seat_jitter_m:
            for ball, (q, _) in enumerate(self.controller.balls):
                self.data.qpos[q:q+2] += self.initial_displacement[ball]
            mujoco.mj_forward(self.model, self.data)
        self.loads = self._loads()
        self.contacts = self.loads[:, :5].sum(axis=1) > .02
        perception = PerceptionConfig(self.config.noise_std_m, self.config.delay_s, self.config.dropout,
                                     int(self.np_random.integers(0, 2**31)))
        self.observer = BallObserver(self.controller.balls, mode='sampled', position_step_s=self.dt, perception=perception)
        self.trackers = [SupportPositionTracker(self.model.opt.gravity, self.dt) for _ in range(self.count)]
        self.tracker = self.trackers[0]
        self.observer.update(self.data)
        if self.observer.observations:
            self.tracker.update(self.observer.observations[0])
        self._notify_support()
        self.joint_offset, self.grip_offset, self.last_action = np.zeros((2, 7)), np.zeros(2), np.zeros(6)
        self.index, self.next_release = 0, 0
        self.released, self.ever_nonzero = False, False
        self.reference_state_max_error = 0.
        self.max_joint_offset = self.max_grip_offset = 0.
        self.max_ball_height = max(float(self.data.qpos[q+2]) for q, _ in self.controller.balls)
        self.return_sum = 0.
        self.reward_sums = dict(event=0., shaping=0., action=0., rate=0., progress=0.)
        self.modes = ['held']*self.count
        self.held_sides = self.initial_sides.copy()
        self.absent = np.zeros(self.count)
        self.stable_age = np.zeros(self.count)
        self.airborne_age = np.zeros(self.count)
        self.flight_ids = [None]*self.count
        self.flight_sources = [None]*self.count
        self.release_times = [None]*self.count
        self.touch_times = [None]*self.count
        self.touch_ball_velocities = [None]*self.count
        self.touch_palm_velocities = [None]*self.count
        self.previous_ball_velocities = [self.data.qvel[v:v+3].copy() for _, v in self.controller.balls]
        self.previous_palm_velocities = [self.controller.spatial(hand)[3:].copy() for hand in self.hands]
        self.recent_digits = [{} for _ in range(self.count)]
        self.events, self.caught_ids = [], set()
        self.initial_age = self.initial_hold_max = self.final_age = 0.
        self.initial_validated = False
        self.estimation_errors = np.zeros((self.count, 3))
        self.active = True
        return self._observation(), self._info(None)

    def _supported(self, ball, side, broad=False):
        hand = self.hands[side]
        q, v = self.controller.balls[ball]
        local = self.data.site_xmat[hand['seat']].reshape(3, 3).T @ (self.data.qpos[q:q+3]-self.data.site_xpos[hand['seat']])
        speed = np.linalg.norm(self.data.qvel[v:v+3]-self.controller.spatial(hand)[3:])
        centered = np.linalg.norm(local) < .16 if broad else np.linalg.norm(local[:2]) < .04 and .01 < local[2] < .06
        return bool(self.ball_contacts[ball, side] and speed < .15 and centered)

    def _monitor(self):
        incoming_ball_velocities, incoming_palm_velocities = self.previous_ball_velocities, self.previous_palm_velocities
        self.previous_ball_velocities = [self.data.qvel[v:v+3].copy() for _, v in self.controller.balls]
        self.previous_palm_velocities = [self.controller.spatial(hand)[3:].copy() for hand in self.hands]
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            return 'NONFINITE_STATE'
        heights = [float(self.data.qpos[q+2]) for q, _ in self.controller.balls]
        self.max_ball_height = max(self.max_ball_height, *heights)
        if min(heights) < .4:
            return 'BALL_DROPPED'
        if max(heights) >= 1.3:
            return 'BALL_TOO_HIGH'
        if (np.any(self.data.eq_active) or np.any(self.data.xfrc_applied[self.controller.ball_body_ids])
                or any(np.any(self.data.qfrc_applied[v:v+6]) for _, v in self.controller.balls)):
            return 'EXTERNAL_BALL_FORCE'
        if not self.initial_validated:
            valid = all(self._supported(ball, side, self.count == 3) for ball, side in enumerate(self.initial_sides))
            self.initial_age = self.initial_age+self.dt if valid else 0.
            self.initial_hold_max = max(self.initial_hold_max, self.initial_age)
            self.initial_validated |= self.initial_age >= .05-1e-10
        for ball in range(self.count):
            if self.modes[ball] == 'held':
                side = self.held_sides[ball]
                self.absent[ball] = self.absent[ball]+self.dt if not self.ball_contacts[ball, side] else 0.
                if self.initial_validated and self.absent[ball] >= .012-1e-10:
                    if self.next_release >= self.target_throws:
                        return 'EXTRA_RELEASE'
                    planned = self.release_reference[self.next_release]
                    separation = float(self.data.time)-self.absent[ball]+self.dt
                    if (planned['ball'] != ball or side != int(planned['side'] == 'left')
                            or abs(separation-planned['separation_time']) > .05):
                        return 'RELEASE_WINDOW'
                    _, dof = self.controller.balls[ball]
                    if self.data.qvel[dof+2] <= 1.:
                        return 'INVALID_RELEASE'
                    self.next_release += 1
                    self.controller.throw_count = self.next_release
                    self.flight_ids[ball], self.flight_sources[ball] = self.next_release, side
                    self.modes[ball], self.release_times[ball], self.touch_times[ball] = 'flight', separation, None
                    self.airborne_age[ball] = self.stable_age[ball] = 0.
                    self.events.append(dict(kind='release', id=self.next_release, ball=ball, side=side,
                        time=separation, confirmed_time=float(self.data.time), vertical_speed_m_s=float(self.data.qvel[dof+2])))
            if self.modes[ball] == 'flight':
                source = self.flight_sources[ball]
                target = 1-source
                if self.ball_contacts[ball, source]:
                    return 'SOURCE_RECONTACT'
                if self.touch_times[ball] is None:
                    if not np.any(self.ball_contacts[ball]):
                        self.airborne_age[ball] += self.dt
                    ending = self.count == 3 and self.flight_ids[ball] == self.target_throws
                    touching = self.ball_contacts[ball, target] if ending else self.ball_loads[ball, target, 0] > .02
                    if touching:
                        self.touch_times[ball] = float(self.data.time)
                        _, dof = self.controller.balls[ball]
                        self.touch_ball_velocities[ball] = incoming_ball_velocities[ball].copy()
                        self.touch_palm_velocities[ball] = incoming_palm_velocities[target].copy()
                        flight = self.touch_times[ball]-self.release_times[ball]
                        if not .4 <= flight <= .75 or self.airborne_age[ball] < .3:
                            return 'INVALID_FLIGHT'
                    elif self.data.time-self.release_times[ball] > .75:
                        return 'FLIGHT_TIMEOUT'
                broad = self.count == 3 and self.flight_ids[ball] == self.target_throws
                stable = self.touch_times[ball] is not None and self._supported(ball, target, broad)
                digits = self.controller.supporting_digits(ball, ('right', 'left')[target])
                recent = {digit: stamp for digit, stamp in self.recent_digits[ball].items()
                          if self.data.time-stamp <= C.DIGIT_CONTACT_MEMORY_S}
                recent.update({digit: float(self.data.time) for digit in digits})
                self.recent_digits[ball] = recent
                if not broad:
                    stable &= 'th' in recent and len(recent) >= 2
                self.stable_age[ball] = self.stable_age[ball]+self.dt if stable else 0.
                absorbed = False
                if self.touch_times[ball] is not None:
                    _, dof = self.controller.balls[ball]
                    speed = self.data.qvel[dof:dof+3]
                    relative = speed-self.controller.spatial(self.hands[target])[3:]
                    incoming = self.touch_ball_velocities[ball]
                    initial_relative = incoming-self.touch_palm_velocities[ball]
                    absorbed = (np.linalg.norm(relative) < C.CAPTURE_RELATIVE_SPEED_RATIO*np.linalg.norm(initial_relative)
                                and np.linalg.norm(speed) < np.linalg.norm(incoming))
                if (not broad and self.stable_age[ball] >= .02-1e-10
                        and self.data.time-self.touch_times[ball] >= .070
                        and 'th' in digits and len(digits) >= 2
                        and self.ball_loads[ball, target, 0] > .02 and absorbed):
                    self.caught_ids.add(self.flight_ids[ball])
                    self.held_sides[ball], self.modes[ball] = target, 'held'
                    self.absent[ball] = 0.
                    self.events.append(dict(kind='catch', id=self.flight_ids[ball], ball=ball, side=target,
                        time=float(self.data.time), palm_touch_time=self.touch_times[ball],
                        supporting_digits=digits, initial_relative_speed=float(np.linalg.norm(initial_relative)),
                        settled_relative_speed=float(np.linalg.norm(relative)),
                        incoming_ball_speed=float(np.linalg.norm(incoming)), settled_ball_speed=float(np.linalg.norm(speed))))
        self.controller.check_forbidden_contacts()
        if self.next_release < self.target_throws and self.data.time > self.release_reference[self.next_release]['separation_time']+.05:
            return 'NO_RELEASE'
        required = self.target_throws-1 if self.count == 3 else self.target_throws
        completed = self.next_release == self.target_throws and len(self.caught_ids) == required
        if self.count == 3:
            completed &= any(self.flight_ids[ball] == self.target_throws and self.touch_times[ball] is not None
                             for ball in range(self.count))
        supported = completed and all(self._supported(ball, side, self.count == 3) for ball, side in enumerate(self.final_sides))
        self.final_age = self.final_age+self.dt if supported else 0.
        if self.final_age >= (.8 if self.count == 3 else .2)-1e-10:
            if self.count == 3:
                ball = self.flight_ids.index(self.target_throws)
                self.caught_ids.add(self.target_throws)
                self.held_sides[ball], self.modes[ball] = self.final_sides[ball], 'held'
                self.events.append(dict(kind='catch', id=self.target_throws, ball=ball, side=self.final_sides[ball],
                    time=float(self.data.time), ending=True, continuous_support_s=self.final_age))
            return 'SUCCESS'
        return None

    def _record_estimation_error(self):
        for ball, tracker in enumerate(self.trackers):
            estimate = tracker.estimate(float(self.data.time))
            if estimate is None or not estimate.valid:
                continue
            q, v = self.controller.balls[ball]
            error_p = np.asarray(estimate.position_m)-self.data.qpos[q:q+3]
            error_v = np.asarray(estimate.velocity_m_s)-self.data.qvel[v:v+3]
            if np.all(np.isfinite(np.r_[error_p, error_v])):
                self.estimation_errors[ball] += [1., error_p@error_p, error_v@error_v]

    def _info(self, reason):
        return dict(task=self.task, environment_version=self.ENVIRONMENT_VERSION, config=asdict(self.config),
                    is_success=reason == 'SUCCESS', reason=reason, simulated_time_s=float(self.data.time),
                    releases=self.next_release, catches=len(self.caught_ids), final_hold_s=self.final_age,
                    initial_hold_s=self.initial_hold_max, reference_state_max_error=self.reference_state_max_error,
                    events=self.events.copy(), failure_event=self.controller.failure_event,
                    physics_steps=self.index, requested_throws=self.target_throws,
                    initial_displacement_m=self.initial_displacement.tolist(),
                    episode_return=self.return_sum, reward_sums=dict(self.reward_sums),
                    zero_reference_verified=not self.ever_nonzero and not self.config.seat_jitter_m,
                    max_ball_height_m=self.max_ball_height, max_joint_offset_rad=self.max_joint_offset,
                    max_grip_offset_rad=self.max_grip_offset,
                     max_wrapping_offset_rad=self.max_wrapping_offset,
                     wrapping_limit_rad=self.MAX_WRAPPING_OFFSET_RAD,
                     wrapping_rate_limit_rad_s=self.MAX_WRAPPING_RATE_RAD_S,
                     wrapping_gain=self.wrapping_gain.tolist(), wrapping_offset_rad=self.wrapping_offset.tolist(),
                     adaptive_ending=self.task == 'complete' and self.controller.end is not None,
                     feedback_gains=self.feedback_gains.tolist(), xy_innovation_m=self.xy_innovation.tolist(),
                     cartesian_request_m=self.cartesian_request.tolist(),
                    joint_offset_rad=self.joint_offset.tolist(), grip_offset_rad=self.grip_offset.tolist(),
                    estimation=[dict(control_samples=int(values[0]),
                        position_rmse_m=float(np.sqrt(values[1]/(3*values[0]))) if values[0] else None,
                        velocity_rmse_m_s=float(np.sqrt(values[2]/(3*values[0]))) if values[0] else None)
                        for values in self.estimation_errors])

    def render(self):
        from v6_humanoid_hands.juggle import cascade_camera
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
        self._renderer.update_scene(self.data, camera=cascade_camera(self.controller.torso_stand_height))
        return self._renderer.render()


def nominal_support(controller, assignments):
    for ball, side in assignments.items():
        hand = controller.hands[side]
        q, v = controller.balls[ball]
        local = controller.data.site_xmat[hand['seat']].reshape(3, 3).T @ (
            controller.data.qpos[q:q+3]-controller.data.site_xpos[hand['seat']])
        relative = controller.data.qvel[v:v+3]-controller.spatial(hand)[3:]
        if (not controller.ball_contact(ball, side) or np.linalg.norm(relative) >= .15
                or np.linalg.norm(local[:2]) >= .04 or not .01 < local[2] < .06):
            return False
    return True


def prepare_reference(folder, task, throws=None, timeout=40.):
    """Record a successful nominal task; silent forecasts are never recorded."""
    folder = Path(folder)
    throws = (2 if task == 'two' else 30) if throws is None else throws
    if (folder.exists() or task not in ('two', 'complete') or not np.isfinite(timeout) or timeout <= 0
            or task == 'two' and throws != 2):
        raise ValueError('Use a new directory, two-ball exchange or an even complete-task throw count')
    fingerprints, generator = reference_sources(), runtime_sources()

    class Trace:
        def __init__(self, *args, **kwargs):
            self.tracing = False
            self.controls, self.states = [], []
            super().__init__(*args, **kwargs)
            self.tracing = True

        def step(self):
            record = self.tracing and not self.silent
            super().step()
            if record:
                self.controls.append(self.data.ctrl.copy())
                self.states.append(np.r_[self.data.qpos, self.data.qvel])

    class TracedTwo(Trace, TwoBallTransfer):
        pass

    class TracedComplete(Trace, CompleteCascade):
        pass

    controller = TracedTwo() if task == 'two' else TracedComplete(throws=throws, observation_mode='sampled')
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    initial = np.zeros(mujoco.mj_stateSize(controller.model, spec))
    mujoco.mj_getState(controller.model, controller.data, initial, spec)
    if task == 'complete':
        result = controller.run_complete(timeout)
    else:
        initial_age = initial_max = final_age = 0.
        dt = float(controller.model.opt.timestep)
        result = dict(success=False)
        while controller.data.time < timeout:
            controller.step()
            if not controller.throw_count:
                initial_age = initial_age+dt if nominal_support(controller, {0: 'right', 1: 'left'}) else 0.
                initial_max = max(initial_max, initial_age)
            supported = {1, 2} <= controller.caught_events and nominal_support(controller, {0: 'left', 1: 'right'})
            final_age = final_age+dt if supported else 0.
            if final_age >= .2-1e-10:
                result = dict(success=initial_max >= .05, initial_hold_s=initial_max,
                              final_hold_s=final_age, throws=controller.throw_count,
                              caught_release_ids=sorted(controller.caught_events))
                break
    if not result['success']:
        raise RuntimeError(f'Nominal {task} reference failed: {result}')
    if fingerprints != reference_sources() or generator != runtime_sources():
        raise RuntimeError('Cascade reference source changed during generation')
    folder.mkdir(parents=True)
    np.savez_compressed(folder/'reference.npz', initial_state=initial,
                        controls=np.asarray(controller.controls), states=np.asarray(controller.states))
    manifest = dict(schema_version=3, phase='preparing', task=task, requested_throws=throws, timeout_s=timeout,
                    source_sha256=fingerprints, generator_source_sha256=generator,
                    reference_sha256=digest(folder/'reference.npz'), mujoco_version=mujoco.__version__,
                    state_spec=int(spec), timestep_s=float(controller.model.opt.timestep),
                    duration_s=float(controller.data.time), physics_steps=len(controller.controls),
                    events=controller.events, result=result)
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    class Recorder(CascadeEnv):
        def __init__(self):
            self.recording, self.recorded_states = False, []
            self.recorded_controls, self.recorded_fingers = [], []
            self.recorded_estimates, self.recorded_valid, self.recorded_palms = [], [], []
            super().__init__(folder, _preparing=True)

        def _notify_support(self):
            super()._notify_support()
            if self.recording:
                self.recorded_states.append(np.r_[self.data.qpos, self.data.qvel])
                self.recorded_controls.append(self.data.ctrl.copy())
                estimates, valid = np.zeros((self.count, 6)), np.zeros(self.count, dtype=bool)
                fingers = np.zeros((self.count, 2, 16))
                for ball, tracker in enumerate(self.trackers):
                    estimate = tracker.estimate(float(self.data.time))
                    if estimate is not None and estimate.valid:
                        estimates[ball] = np.r_[estimate.position_m, estimate.velocity_m_s]
                        valid[ball] = True
                    for side in (0, 1):
                        correction = self._finger_correction(ball, side)
                        if correction is not None:
                            fingers[ball, side] = correction
                self.recorded_estimates.append(estimates)
                self.recorded_valid.append(valid)
                self.recorded_fingers.append(fingers)
                self.recorded_palms.append(np.array([np.r_[self.data.site_xpos[hand['seat']], self.controller.spatial(hand)[3:]]
                                                     for hand in self.hands]))

    class FingerTrace(Recorder):
        def __init__(self):
            self.corrections = []
            super().__init__()

        def _wrapping_target(self):
            return np.zeros((2, 16))

        def _ending_control(self):
            pass

        def _notify_support(self):
            super()._notify_support()
            if self.recording:
                self.corrections.append(self.recorded_fingers[-1])

    def replay(env):
        try:
            env.reset(seed=0)
            env.recording = True
            while True:
                _, _, done, _, info = env.step(np.zeros(6))
                if done:
                    break
            if not info['is_success']:
                raise RuntimeError(f'Nominal feedback calibration failed: {info["reason"]}')
            return info
        finally:
            env.close()

    trace = FingerTrace()
    replay(trace)
    controls, original_states = np.asarray(controller.controls), np.asarray(controller.states)
    correction_array = np.asarray(trace.corrections)
    if len(correction_array) < len(controls):
        correction_array = np.concatenate([correction_array, np.repeat(correction_array[-1:],
            len(controls)-len(correction_array), axis=0)])
    correction_array = correction_array[:len(controls)]
    np.savez_compressed(folder/'reference.npz', initial_state=initial, controls=controls,
                        states=original_states, finger_corrections=correction_array)
    manifest['reference_sha256'] = digest(folder/'reference.npz')
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    nominal = Recorder()
    feedback_result = replay(nominal)
    # Compile the executed closed-loop commands and their measured reference.
    # Feedback then tracks deviations from this trajectory, not an old teacher.
    np.savez_compressed(folder/'reference.npz', initial_state=initial,
                        controls=np.asarray(nominal.recorded_controls), states=np.asarray(nominal.recorded_states),
                        finger_corrections=np.asarray(nominal.recorded_fingers),
                        estimate_reference=np.asarray(nominal.recorded_estimates),
                        estimate_valid=np.asarray(nominal.recorded_valid), palm_reference=np.asarray(nominal.recorded_palms))
    manifest.update(reference_sha256=digest(folder/'reference.npz'),
                    duration_s=float(nominal.data.time), physics_steps=len(nominal.recorded_states),
                    nominal_feedback='measured-reference feedback gains and analytic ending; no forward trials',
                    nominal_zero_result=feedback_result)
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    checked = replay(CascadeEnv(folder, _preparing=True))
    if checked['reference_state_max_error'] != 0.:
        raise RuntimeError('Compiled feedback reference does not reproduce the nominal trajectory')
    manifest.update(phase='ready', nominal_zero_result=checked)
    if fingerprints != reference_sources() or generator != runtime_sources():
        raise RuntimeError('Feedback reference source changed during generation')
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--task', choices=('two', 'complete'), required=True)
    prepare.add_argument('--throws', type=int)
    prepare.add_argument('--timeout', type=float, default=40.)
    prepare.add_argument('--output-dir', type=Path, required=True)
    check = commands.add_parser('check')
    check.add_argument('--reference-dir', type=Path, required=True)
    check.add_argument('--episodes', type=int, default=1)
    check.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        manifest = prepare_reference(args.output_dir, args.task, args.throws, args.timeout)
        print('Prepared', manifest['task'], manifest['requested_throws'], 'throws;', manifest['physics_steps'], 'physical steps')
        return
    if args.output.exists() or args.episodes < 1:
        parser.error('Use a new output and a positive episode count')
    fingerprints = runtime_sources()
    env = CascadeEnv(args.reference_dir)
    records, controls = [], 0
    started = time.perf_counter()
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
    if fingerprints != runtime_sources():
        raise RuntimeError('Cascade runtime changed during verification')
    result = dict(source_sha256=fingerprints, reference_sha256=env.reference['reference_sha256'],
                  records=records, control_steps=controls, wall_time_s=time.perf_counter()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('Successes', sum(row['is_success'] for row in records), '/', len(records),
          '; control steps/s', controls/result['wall_time_s'])
    if any(not row['is_success'] or row['reference_state_max_error'] != 0. for row in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Contact-only three-ball juggling with an elliptical underhand scoop.

Only initialization assigns ball positions or velocities. The model contains
no hand-ball equalities. Every physics step verifies that control did not write
ball state or apply external ball forces. A catch requires sustained physical
support after downward cushioning, rather than a first-touch event.
"""
from .model import ARM_JOINTS_RIGHT, ARM_JOINTS_LEFT
from .model import DEFAULT_STAND_HEIGHT_M, BALL_SEAT_CENTER_OFFSET_M, BALL_RADIUS

import json
from pathlib import Path
import mujoco
import numpy as np
from .contact_control import DualArmDynamics, PREFERRED_THROW_RIGHT, SmoothStop, ThrowPath, CartesianPalmMotion
from .contact_control import SOURCE_POSITION_RIGHT_M
from .planning import Quintic

DEFAULT_BALL_COUNT = 3
DEFAULT_THROW_INTERVAL_S = .32
DEFAULT_FLIGHT_TIME_S = .53  # Physical hand separation to palm interception, as v5's F.
NOMINAL_RELEASE_DELAY_S = .17  # Bootstrap phase only, not part of flight time.
INITIAL_AIRBORNE_LANDING_VELOCITY_M_S = np.array([-.12, -.77, -2.65])
INITIAL_AIRBORNE_FORWARD_OFFSET_M = .10
RELEASE_CONFIRMATION_S = .012
MAX_FINGER_RELEASE_LEAD_S = .025  # Bounded actuator timing; positive opens earlier.
DEFAULT_OBSERVATION_NOISE_STD_M = 0.0
# Contact/motion calibration for the default G1/LEAP Hand, 85 mm / 120 g ball.
# These are reference-controller parameters, not measured ball state or replay data.
SUPPORTED_BALL_CENTER_OFFSET_M = .0223  # Dynamic support correction relative to the seat site.
CATCH_CUSHION_DURATION_S = .12
CAPTURE_BRAKING_DURATION_S = .08
MIN_CATCH_RECOVERY_DURATION_S = .24
MAX_RECEPTION_SETTLE_FORECAST_S = .30
LOADED_RECOVERY_TILT_DEG = -8.0
LOADED_SOURCE_TILT_DEG = 10.0
INITIAL_HOLD_TILT_DURATION_S = .08
THROW_READY_MARGIN_S = .005
# A modest forward palm pitch avoids excessive wrist extension after a catch.
LOADED_SOURCE_AXIS = np.array([np.sin(np.deg2rad(LOADED_SOURCE_TILT_DEG)), 0., np.cos(np.deg2rad(LOADED_SOURCE_TILT_DEG))])
LOADED_RECOVERY_PEAK_S = .075
BALL_DROP_HEIGHT_M = .4
RECEIVE_VELOCITY_RATIO = .55
RECEIVE_TANGENTIAL_VELOCITY_RATIO = .90
RECEIVE_PALM_BACK_TILT = .04
RECEIVE_INWARD_BANK = .08
RECEIVE_PREPARE_LEAD_S = .11
RECEIVE_PREPARE_DURATION_S = .07
MAX_PLANNED_APEX_M = 1.29
APEX_PENALTY_WEIGHT = 4.
BALLISTIC_POSITION_CORRECTION_GAIN = 2.
THROW_TIME_ERROR_WEIGHT = 2.
BALLISTIC_TIME_CORRECTION_GAIN = 5.
# Contact release is nonsmooth: probe nearby release branches before local IK.
COARSE_THROW_VERTICAL_OFFSETS_M_S = (-.4, 0., .4)
COARSE_THROW_BANK_OFFSETS_RAD = (-.02, 0., .02)
COARSE_THROW_FORWARD_OFFSETS_M_S = (-.4, 0., .4)
COARSE_THROW_WIDE_BANK_OFFSETS_RAD = (-.08, 0., .08)
DELAYED_OPPOSITION_ONSET_S = .08  # After closure, only while the grip remains unconfirmed.
CAPTURE_FORCE_FEEDBACK_DELAY_S = .070  # Wait for impact and geometric closure.
CAPTURE_RELATIVE_SPEED_RATIO = .4
DIGIT_CONTACT_MEMORY_S = .004  # Bridge solver-scale force chatter, not loss of support.
MIN_PLANNED_UPSTROKE_M = .121
MIN_PLANNED_VERTICAL_RATIO = 1.42
MIN_PLANNED_FINGERS_FORWARD = .851
THROW_POSTURE_PENALTY_WEIGHT = 20.
THROW_REPLAN_INTERVAL_S = .005
THROW_BRAKING_PREVIEW_S = .08
MIN_PLANNED_BALL_SPACING_M = 2*BALL_RADIUS + .004
BALL_SPACING_PENALTY_WEIGHT = 20.
# Reference search limits. Physical rollouts still reject missed/colliding
# catches, and the original pose regressions remain unchanged. The central
# full-size LEAP grasp needs over 30 mm of fore/aft reference adjustment.
RECEPTION_PARAMETER_LIMITS = np.array([.05, .055, .04, .12, .12])
RECEPTION_FINITE_DIFFERENCE = np.array([.006, .006, .006, .03, .03])
RECEPTION_FINE_DIFFERENCE = np.array([.001, .001, .001, .006, .006])
MAX_PALM_HEIGHT_M = 1.15
MAX_BALL_HEIGHT_M = 1.30
MAX_ELBOW_OUT_M = .10
MIN_SHOULDER_ELBOW_CLEARANCE_M = .08
MAX_ARM_JOINT_VIOLATION_RAD = .004
MAX_WRIST_PITCH_RAD = np.deg2rad(38.)
MAX_SELF_PENETRATION_M = .004

class ContactCascade(DualArmDynamics):
    """Contact cascade, with a genuine one-ball mode for motion diagnostics."""

    def __init__(self, tau=DEFAULT_THROW_INTERVAL_S, flight=DEFAULT_FLIGHT_TIME_S,
                 torso_stand_height=DEFAULT_STAND_HEIGHT_M,
                 obs_noise=DEFAULT_OBSERVATION_NOISE_STD_M, verbose=False, n_balls=DEFAULT_BALL_COUNT, seed=0):
        if n_balls not in (1, 2, 3):
            raise ValueError('contact cascade supports one, two, or three balls')
        offset = torso_stand_height - DEFAULT_STAND_HEIGHT_M
        super().__init__('right', height=SOURCE_POSITION_RIGHT_M[2] + offset, n_balls=n_balls, torso_stand_height=torso_stand_height)
        m, d = (self.model, self.data)
        self.tau = tau
        self.torso_stand_height = torso_stand_height
        self.verbose = verbose
        self.obs_noise = obs_noise
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        warm_file = Path(__file__).with_name('leap_cascade_seed_plans.json')
        self.set_optimizer_inputs(json.loads(warm_file.read_text()) if warm_file.exists() else {})
        self.flight = flight
        self.max_planned_apex = MAX_PLANNED_APEX_M + offset
        self.planning = True
        self.silent = False
        self.commands = {'right': np.array([.31, .70, 3.0, -.06]), 'left': np.array([.31, -.70, 3.0, -.06])}
        self.balls = [(int(m.joint(f'ball{i}_free').qposadr[0]), int(m.joint(f'ball{i}_free').dofadr[0])) for i in range(n_balls)]
        self.ball_geom_ids = [m.geom(f'ball{i}').id for i in range(n_balls)]
        self.ball_geom_set = set(self.ball_geom_ids)
        self.ball_position_indices = np.array([index for q, _ in self.balls for index in range(q, q + 7)])
        self.ball_velocity_indices = np.array([index for _, v in self.balls for index in range(v, v + 6)])
        self.ball_body_ids = m.geom_bodyid[self.ball_geom_ids].copy()
        if any(m.equality(index).name.startswith('grasp_') for index in range(m.neq)):
            raise RuntimeError('contact-only model contains a hand-ball grasp equality')
        self.contact_only_checks = 0
        self.hand_geoms = {}
        self.palm_geoms = {}
        for side, prefix in [('right', 'rh_'), ('left', 'lh_')]:
            self.hand_geoms[side] = set()
            self.palm_geoms[side] = set()
            for g in range(m.ngeom):
                name = m.body(m.geom_bodyid[g]).name
                if name.startswith(prefix):
                    self.hand_geoms[side].add(g)
                    if name == prefix + 'palm':
                        self.palm_geoms[side].add(g)
        self.paths = {}
        for side, h in self.hands.items():
            h['p'] = SOURCE_POSITION_RIGHT_M.copy()
            h['p'][1] *= 1. if side == 'right' else -1.
            h['p'][2] += offset
            seed = PREFERRED_THROW_RIGHT.copy()
            if side == 'left':
                seed *= np.array([1, -1, -1, 1, -1, 1, -1])
            source_pose = h['kin'].solve(h['p'], np.array([0.0, 0.0, 1.0]), seed, q_nominal=seed)
            if not source_pose.success:
                raise ValueError(f'unreachable source posture: {side}')
            h['q'] = source_pose.q
            d.qpos[h['kin'].qpos_adr] = h['q']
            d.ctrl[h['acts']] = h['q']
            h.update(mode='hold', held=0 if side == 'right' else 1, throw_time=0.1 if side == 'right' else 0.1 + tau, release_pending=None, reception=None, carry=None, contact_age=0.0, path=None)
            c = self.commands[side]
            self.paths[side] = ThrowPath(h['kin'], h['p'], c[:3], h['q'], h['catch_p'], bank=c[3])
            h['catch_q'] = self.paths[side].catch_q.copy()
            fingers = h['fingers']
            h['closure_onset'] = fingers.closure_onset.copy()
            h['closure_duration'] = fingers.closure_duration.copy()
            h['grip_map'] = fingers.grip_map.copy()
        mujoco.mj_forward(m, d)
        torso_rotation=d.xmat[m.body('torso_link').id].reshape(3,3)
        for side, hand in self.hands.items():
            elbow=d.xpos[m.body(side+'_elbow_link').id]
            shoulder=d.xpos[m.body(side+'_shoulder_pitch_link').id]
            hand['source_elbow_local']=torso_rotation.T@(elbow-shoulder)
        for side, i in ([('right', 0), ('left', 1)] if n_balls >= 2 else [('right', 0)]):
            q, v = self.balls[i]
            d.qpos[q:q + 3] = d.site_xpos[self.hands[side]['seat']] + [0, 0, BALL_SEAT_CENTER_OFFSET_M]
            d.qvel[v:v + 6] = 0
        if n_balls == 3:
            remaining = 0.1 + NOMINAL_RELEASE_DELAY_S + self.flight - tau
            landing = self.hands['right']['catch_p'] + [0, 0, SUPPORTED_BALL_CENTER_OFFSET_M]
            landing[0] += INITIAL_AIRBORNE_FORWARD_OFFSET_M
            # Seed the already-airborne ball from the same low, narrow cascade
            # as the new scoop, rather than the previous high-speed throw.
            velocity_end = INITIAL_AIRBORNE_LANDING_VELOCITY_M_S.copy()
            q, v = self.balls[2]
            d.qpos[q:q + 3] = landing - velocity_end * remaining + [0, 0, -0.5 * self.gravity_m_s2 * remaining ** 2]
            d.qvel[v:v + 3] = velocity_end + [0, 0, self.gravity_m_s2 * remaining]
        mujoco.mj_forward(m, d)
        self.initial_ball_states = [dict(position_m=d.qpos[q:q+3].tolist(),
                                         velocity_m_s=d.qvel[v:v+3].tolist())
                                    for q, v in self.balls]
        for side, i in ([('right', 0), ('left', 1)] if n_balls >= 2 else [('right', 0), ('left', 0)]):
            hand = self.hands[side]
            hand['fingers'].calibrate_wrapping(d, hand['seat'], self.ball_geom_ids[i])
        for side, i in ([('right', 0), ('left', 1)] if n_balls >= 2 else [('right', 0)]):
            self.begin_capture(side, i)
        if n_balls == 1:
            self.hands['left'].update(mode='wait', held=None)
        self.initial_contact_evidence = [h['contact_evidence'].copy() for h in self.hands.values() if h['held'] is not None]
        self.targets = {2: 'right'} if n_balls == 3 else {}
        self.throw_count = 0
        self.exchange_pair = 0
        self.catch_count = 0
        self.ball_contacts = 0
        self.minimum_distance = 10.0
        self.events = []
        self.airborne = {2: 0} if n_balls == 3 else {}
        self.last_release = {}
        self.next_event = 1
        self.caught_events = set()
        self.minimum_height = 10.0
        self.drop_event = None
        self.failure_event = None
        self.airborne_physics_checks = 0
        self.max_height = 0.0
        self.max_palm_height = 0.0
        self.min_shoulder_above_elbow = 10.0
        self.max_self_penetration = 0.0
        self.palm_ranges = {side: dict(ymin=10.0, ymax=-10.0, zmin=10.0, zmax=-10.0) for side in self.hands}
        self.active_strokes = {}
        self.stroke_metrics = []
        self.max_joint_limit_violation = 0.0
        self.posture_samples = []
        self.torso_samples = []
        self.next_posture_sample = 0.
        from .metrics import ExecutionMetrics
        self.execution_metrics = ExecutionMetrics(m)
        self.waist_position_indices = [int(m.joint(name).qposadr[0]) for name in ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint')]
        self.min_throw_palm_up = 1.0
        self.min_throw_fingers_forward = 1.0

    def set_optimizer_inputs(self, inputs):
        """Keep forecast guesses independent of live optimizer warm starts.

        Updating a successful live command must not retroactively change the
        other arm's predicted motion inside an earlier reception rollout.
        Flat dictionaries remain accepted for initial calibration.
        """
        warm = dict(inputs.get('forecast', inputs))
        execution = dict(inputs.get('execution', {}))
        for group in (warm, execution):
            for key, values in group.items():
                if not isinstance(key, str) or not key.startswith('throw:'):
                    continue
                command = np.asarray(values, dtype=float)
                if command.ndim != 1 or command.size not in (4, 5) or not np.all(np.isfinite(command)):
                    raise ValueError(f'{key}: expected four motion parameters and optional finger release lead')
                if command.size == 5 and abs(command[4]) > MAX_FINGER_RELEASE_LEAD_S:
                    raise ValueError(f'{key}: finger release lead exceeds {MAX_FINGER_RELEASE_LEAD_S} s')
        self.warm, self.execution_warm = warm, execution

    def begin_capture(self, side, i):
        """Start finger closure only after a real palm contact; this registers evidence and does not attach or reposition the ball."""
        if not self.ball_contact(i, side, palm_only=True):
            raise RuntimeError('capture requires real palm contact')
        m, d = (self.model, self.data)
        h = self.hands[side]
        ball = self.ball_geom_ids[i]
        evidence = []
        for c in d.contact:
            other = c.geom2 if c.geom1 == ball else c.geom1 if c.geom2 == ball else -1
            if other in self.palm_geoms[side] and c.dist <= 1e-05 and (c.efc_address >= 0):
                evidence.append((float(c.dist), m.body(m.geom_bodyid[other]).name))
        distance, body = min(evidence)
        h['contact_evidence'] = dict(contact_distance=distance, contact_body=body)
        normal = d.site_xmat[h['seat']].reshape(3, 3)[:, 2]
        center = d.qpos[self.balls[i][0]:self.balls[i][0] + 3] - d.site_xpos[h['seat']]
        h['contact_evidence'].update(palm_side=float(center @ normal), tangent_offset=float(np.linalg.norm(center - center @ normal * normal)))
        h['contact_time'] = self.data.time if self.data.time > 0 else -0.2
        h['grip_offsets']=np.zeros(len(h['fingers'].digits))
        h['grip_force']=np.zeros(len(h['fingers'].digits))
        h['digit_contact_times'] = {}
        h['opposition_offset'] = np.zeros(4)
        torso_rotation=d.xmat[m.body('torso_link').id].reshape(3,3)
        h['touch_elbow_local']=torso_rotation.T@(d.xpos[m.body(side+'_elbow_link').id]-d.xpos[m.body(side+'_shoulder_pitch_link').id])
        h['closure_start'] = d.ctrl[h['fingers'].act_indices].copy()
        h['opening_ready'] = False

    def ball_contact(self, i, side, palm_only=False):
        ball = self.ball_geom_ids[i]
        hand = self.palm_geoms[side] if palm_only else self.hand_geoms[side]
        for c in self.data.contact:
            other = c.geom2 if c.geom1 == ball else c.geom1 if c.geom2 == ball else -1
            if other < 0:
                continue
            if other in hand and c.dist <= 1e-05 and (c.efc_address >= 0):
                return True
        return False

    def spatial(self, h):
        spatial = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, h['seat'], spatial, 0)
        return spatial

    def holding_axis(self, hand):
        """Ease the initial loaded palm into the same posture used after catches."""
        if hand.get('contact_time', -1.) >= 0.:
            return LOADED_SOURCE_AXIS
        phase = float(np.clip(self.data.time / INITIAL_HOLD_TILT_DURATION_S, 0., 1.))
        weight = phase**3 * (10. - 15.*phase + 6.*phase**2)
        angle = np.deg2rad(LOADED_SOURCE_TILT_DEG) * weight
        return np.array([np.sin(angle), 0., np.cos(angle)])

    def ready_for_throw(self, side):
        """Use the same physical readiness gate in forecasts and live control."""
        hand = self.hands[side]
        if hand['held'] is None:
            return False
        q, v = self.balls[hand['held']]
        normal = self.data.site_xmat[hand['seat']].reshape(3, 3)[:, 2]
        offset = self.data.qpos[q:q+3] - self.data.site_xpos[hand['seat']] - SUPPORTED_BALL_CENTER_OFFSET_M*normal
        relative = self.data.qvel[v:v+3] - self.spatial(hand)[3:]
        return bool(np.linalg.norm(offset[:2]) <= .045 and np.linalg.norm(relative) <= .18
                    and normal[2] >= .85 and hand.get('capture_confirmed', True)
                    and self.ball_contact(hand['held'], side))

    def log(self, kind, side, i, **extra):
        if self.silent:
            return
        event = dict(time=round(self.data.time, 4), kind=kind, side=side, ball=i, **extra)
        self.events.append(event)
        if self.verbose:
            print(event, flush=True)
        elif kind == 'release' and getattr(self, 'progress_interval', 0) and extra['count'] % self.progress_interval == 0:
            completed = len(self.caught_events - {0})
            print(f"[Progress] {extra['count']} releases | {completed} complete flights | sim time {self.data.time:.3f} s", flush=True)

    def airborne_clearance(self, launched_ball):
        """Predict sphere separation only while both balls are freely flying.

        Equal gravity cancels in relative motion, so the closest point is
        analytic. Stop at the first nominal palm interception and exclude
        balls already touching a hand. This only reads the simulated state.
        """
        d = self.data
        held = {h['held'] for h in self.hands.values() if h['held'] is not None}
        def flight(ball):
            q, v = self.balls[ball]
            position, velocity = d.qpos[q:q+3], d.qvel[v:v+3]
            height = self.hands[self.targets[ball]]['catch_p'][2] + SUPPORTED_BALL_CENTER_OFFSET_M
            disc = velocity[2]**2 + 2*self.gravity_m_s2*(position[2]-height)
            remaining = max(0., (velocity[2]+np.sqrt(max(0.,disc)))/self.gravity_m_s2)
            return position, velocity, remaining
        position, velocity, remaining = flight(launched_ball)
        closest = 10.
        for other in self.airborne:
            if other == launched_ball or other in held:
                continue
            other_position, other_velocity, other_remaining = flight(other)
            relative = position-other_position
            speed = velocity-other_velocity
            horizon = min(.25, remaining, other_remaining)
            time = np.clip(-relative@speed/max(float(speed@speed),1e-12),0.,horizon)
            closest = min(closest,float(np.linalg.norm(relative+time*speed)))
        return closest

    def plan_throw(self, side):
        """Optimize an actuator-driven launch using private physics rollouts restored before returning to the live state."""
        live = self.__dict__.copy()
        hand = self.hands[side]
        ball = hand['held']
        q, v = self.balls[ball]
        mujoco.mj_copyData(hand['kin'].data, self.model, self.data)
        initial = mujoco.MjData(self.model)
        mujoco.mj_copyData(initial, self.model, self.data)
        target = self.hands['left' if side == 'right' else 'right']['catch_p'] + [0, 0, SUPPORTED_BALL_CENTER_OFFSET_M]
        initial_time = self.data.time
        start = self.data.site_xpos[hand['seat']].copy()
        initial_q = self.data.qpos[hand['kin'].qpos_adr].copy()
        initial_axis = self.data.site_xmat[hand['seat']].reshape(3, 3)[:, 2].copy()
        input_key = f'throw:{side}:{self.throw_count}'
        input_command = np.asarray(self.execution_warm.get(input_key, self.warm.get(input_key, self.commands[side])), dtype=float)
        release_lead = float(input_command[4]) if input_command.size == 5 else 0.

        def evaluate(command):
            mujoco.mj_copyData(hand['kin'].data, self.model, initial)
            path = ThrowPath(hand['kin'], start, command[:3], initial_q, hand['catch_p'], bank=command[3], initial_axis=initial_axis, catch_q=hand['catch_q'])
            path.planned_time = initial_time
            path.finger_release_lead_s = release_lead
            trial = mujoco.MjData(self.model)
            mujoco.mj_copyData(trial, self.model, initial)
            self.data = trial
            self.hands = {s: dict(h) for s, h in live['hands'].items()}
            self.paths = dict(live['paths'])
            self.paths[side] = path
            self.targets = dict(live['targets'])
            self.airborne = dict(live['airborne'])
            self.caught_events = set(live['caught_events'])
            self.events = []
            self.planning = False
            self.silent = True
            self.minimum_distance = 10.
            stroke_points = [start]
            fingers_forward = 1.
            try:
                while trial.time < initial_time + 0.35 - 1e-08:
                    self.step()
                    try:
                        self.check_forbidden_contacts()
                    except RuntimeError as failure:
                        raise ValueError(f'infeasible launch: {failure}') from failure
                    stroke_points.append(trial.site_xpos[hand['seat']].copy())
                    fingers_forward = min(fingers_forward, float(trial.site_xmat[hand['seat']].reshape(3,3)[0,1]))
                    if ball in self.airborne:
                        break
                if ball not in self.airborne:
                    raise ValueError('launch did not clear the hand')
                disc = trial.qvel[v + 2] ** 2 + 2 * self.gravity_m_s2 * (trial.qpos[q + 2] - target[2])
                remaining = (trial.qvel[v + 2] + np.sqrt(max(0.0, disc))) / self.gravity_m_s2
                error = trial.qpos[q:q + 3] + trial.qvel[v:v + 3] * remaining + [0, 0, -0.5 * self.gravity_m_s2 * remaining ** 2] - target
                free_elapsed = trial.time - self.hands[side]['release_pending']
                error[2] = THROW_TIME_ERROR_WEIGHT * (remaining + free_elapsed - self.flight)
                apex = trial.qpos[q+2] + max(0., trial.qvel[v+2])**2/(2*self.gravity_m_s2)
                error = np.r_[error, APEX_PENALTY_WEIGHT*max(0., apex-self.max_planned_apex)]
                points = np.asarray(stroke_points)
                rise = float(points[-1,2]-start[2])
                lateral = float(np.ptp(points[:,1]))
                error = np.r_[error, THROW_POSTURE_PENALTY_WEIGHT*np.array([
                    max(0., MIN_PLANNED_UPSTROKE_M-rise),
                    max(0., MIN_PLANNED_VERTICAL_RATIO*lateral-rise),
                    max(0., MIN_PLANNED_FINGERS_FORWARD-fingers_forward)])]
                clearance = min(self.minimum_distance, self.airborne_clearance(ball))
                error = np.r_[error, BALL_SPACING_PENALTY_WEIGHT*max(0., MIN_PLANNED_BALL_SPACING_M-clearance)]
                path.predicted_release_velocity = trial.qvel[v:v + 3].copy()
                path.predicted_release_time = float(trial.time)
                # Release alone does not prove that the loaded wrist can stop.
                # Check the short empty-hand transition too, retaining the
                # release state above for the ballistic objective and timing.
                braking_end = trial.time + THROW_BRAKING_PREVIEW_S
                braking_apex = apex
                while trial.time < braking_end - 1e-08:
                    self.step()
                    try:
                        self.check_forbidden_contacts()
                    except RuntimeError as failure:
                        raise ValueError(f'infeasible launch braking: {failure}') from failure
                    braking_apex = max(braking_apex, float(trial.qpos[q+2] + max(0., trial.qvel[v+2])**2/(2*self.gravity_m_s2)))
                # A later physical finger contact can change the ball's energy
                # after separation confirmation. Include that in the ceiling.
                error[3] = APEX_PENALTY_WEIGHT*max(0., braking_apex-self.max_planned_apex)
            finally:
                self.__dict__.clear()
                self.__dict__.update(live)
            return (np.asarray(error), path)
        key = f'throw:{side}:{self.throw_count}'
        # A delayed retry starts from its previous best command. Keep this
        # transient guess separate from fixed forecasts and exported inputs.
        guess = (hand['retry_throw_command'] if hand.get('retry_throw_key') == key else
                 self.execution_warm.get(key, self.warm.get(key, self.commands[side])))
        command = np.array(guess, dtype=float)[:4].copy()
        try:
            error, path = evaluate(command)
        except ValueError:
            # A seed calibrated for a different contact state may not release
            # at all. Treat that as an infeasible trial, just as below, rather
            # than aborting the live simulation before trying nearby launches.
            feasible = []
            offsets = ([np.array([0., 0., vertical, bank])
                        for vertical in COARSE_THROW_VERTICAL_OFFSETS_M_S
                        for bank in COARSE_THROW_BANK_OFFSETS_RAD if vertical or bank]
                       + [np.array([forward, 0., 0., bank])
                          for forward in COARSE_THROW_FORWARD_OFFSETS_M_S
                          for bank in COARSE_THROW_WIDE_BANK_OFFSETS_RAD if forward or bank])
            for offset in offsets:
                candidate = command + offset
                try:
                    ep, pp = evaluate(candidate)
                except ValueError:
                    continue
                feasible.append((float(np.linalg.norm(ep)), candidate.copy(), ep, pp))
                if feasible[-1][0] < .02:
                    break
            if not feasible:
                self.last_plan_error = float('inf')
                self.log('plan_rejected', side, ball, reason='no feasible contact release',
                         command=(np.r_[command, release_lead] if release_lead else command).tolist())
                return None
            _, command, error, path = min(feasible, key=lambda trial: trial[0])
        best = (np.linalg.norm(error), command.copy(), path, error.copy())
        if best[0] > .02:
            # A flight-time error of 30 ms needs a meaningful vertical-speed
            # correction. Milliradian-scale contact changes can make tiny
            # finite differences noisy, so try a ballistic correction first.
            correction = np.r_[-BALLISTIC_POSITION_CORRECTION_GAIN*error[:2],
                               -BALLISTIC_TIME_CORRECTION_GAIN*error[2], 0.]
            if error[3] > 0.:
                correction[2] = min(correction[2], -2.*error[3])
            candidate = command + np.clip(correction, [-.2,-.2,-.5,0.], [.2,.2,.5,0.])
            try:
                ep, pp = evaluate(candidate)
                if np.linalg.norm(ep) < best[0]:
                    command, error, path = candidate, ep, pp
                    best = (np.linalg.norm(ep), candidate.copy(), pp, ep.copy())
            except ValueError:
                pass
        if best[0] > .02:
            base = best[1].copy()
            for vertical in COARSE_THROW_VERTICAL_OFFSETS_M_S:
                for bank in COARSE_THROW_BANK_OFFSETS_RAD:
                    if vertical == 0. and bank == 0.:
                        continue
                    candidate = base + [0., 0., vertical, bank]
                    try:
                        ep, pp = evaluate(candidate)
                    except ValueError:
                        continue
                    if np.linalg.norm(ep) < best[0]:
                        best = (np.linalg.norm(ep), candidate.copy(), pp, ep.copy())
            _, command, path, error = best
        if best[0] > .02:
            # Contact after a catch can lie on a different release branch from
            # the initially seated ball. Forward speed and palm bank must also
            # receive a wider probe before relying on local derivatives.
            base = best[1].copy()
            for forward in COARSE_THROW_FORWARD_OFFSETS_M_S:
                for bank in COARSE_THROW_WIDE_BANK_OFFSETS_RAD:
                    if forward == 0. and bank == 0.:
                        continue
                    candidate = base + [forward, 0., 0., bank]
                    try:
                        ep, pp = evaluate(candidate)
                    except ValueError:
                        continue
                    if np.linalg.norm(ep) < best[0]:
                        best = (np.linalg.norm(ep), candidate.copy(), pp, ep.copy())
                    if best[0] < .02:
                        break
                if best[0] < .02:
                    break
            _, command, path, error = best
        for it in range(8):
            if best[0] < 0.02:
                break
            columns = []
            for index, step in [(0, .02), (1, .02), (2, .05), (3, .008)]:
                perturbed = command.copy()
                perturbed[index] += step
                try:
                    ep, pp = evaluate(perturbed)
                    columns.append((ep - error) / step)
                    if np.linalg.norm(ep) < best[0]:
                        best = (np.linalg.norm(ep), perturbed.copy(), pp, ep.copy())
                    if best[0] < .02:
                        break
                except ValueError:
                    columns.append(np.zeros_like(error))
            if best[0] < .02:
                break
            delta = np.linalg.lstsq(np.column_stack(columns), -error, rcond=0.001)[0]
            delta *= min(1, 1 / max(0.001, np.max(abs(delta) / [0.20, 0.15, 0.50, .02])))
            improved = False
            for fraction in [1.0, 0.4]:
                candidate = command.copy()
                candidate += delta * fraction
                try:
                    ep, pp = evaluate(candidate)
                except ValueError:
                    continue
                if np.linalg.norm(ep) < best[0]:
                    command, error, path = (candidate, ep, pp)
                    best = (np.linalg.norm(ep), candidate.copy(), pp, ep.copy())
                    improved = True
                    break
            if not improved:
                break
        _, command, path, error = best
        if best[0] > 0.02:
            rng = np.random.default_rng(17 + self.throw_count)
            for generation in range(2):
                base = best[1].copy()
                scale = np.array([0.08, 0.16, 0.1, 0.012]) * 0.55 ** generation
                heuristic = base.copy()
                heuristic[:3] -= best[3][:3] * [0.8, 1.1, 0.6]
                for candidate in [heuristic] + [base + rng.normal(size=4) * scale for _ in range(15)]:
                    try:
                        ep, pp = evaluate(candidate)
                    except ValueError:
                        continue
                    if np.linalg.norm(ep) < best[0]:
                        best = (np.linalg.norm(ep), candidate.copy(), pp, ep.copy())
                if best[0] < 0.01:
                    break
            _, command, path, error = best
        if best[0] > 0.02:
            for iteration in range(4):
                command = best[1].copy()
                error = best[3].copy()
                columns = []
                for i, step in enumerate([0.003, 0.004, 0.006, 0.0007]):
                    candidate = command.copy()
                    candidate[i] += step
                    try:
                        ep, pp = evaluate(candidate)
                        columns.append((ep - error) / step)
                        if np.linalg.norm(ep) < best[0]:
                            best = (np.linalg.norm(ep), candidate.copy(), pp, ep.copy())
                        if best[0] < .015:
                            break
                    except ValueError:
                        columns.append(np.zeros_like(error))
                if best[0] < .015:
                    break
                delta = np.linalg.lstsq(np.column_stack(columns), -error, rcond=0.005)[0]
                delta *= min(1.0, 1 / max(0.001, np.max(abs(delta) / [0.05, 0.08, 0.08, 0.008])))
                improved = False
                for fraction in [1.0, 0.4, 0.15]:
                    try:
                        ep, pp = evaluate(command + delta * fraction)
                    except ValueError:
                        continue
                    if np.linalg.norm(ep) < best[0]:
                        best = (np.linalg.norm(ep), command + delta * fraction, pp, ep.copy())
                        improved = True
                        break
                if best[0] < 0.015 or not improved:
                    break
            _, command, path, error = best
        self.commands[side] = command
        hand['retry_throw_key'] = key
        hand['retry_throw_command'] = command.copy()
        self.last_plan_error = float(np.linalg.norm(error))
        actual_error = error.copy()
        self.log('plan', side, ball, error=actual_error[:2].round(4).tolist(), timing_error=round(float(actual_error[2] / THROW_TIME_ERROR_WEIGHT), 4), score=round(self.last_plan_error,5), objective_components=actual_error.tolist(), command=(np.r_[command, release_lead] if release_lead else command).tolist(), predicted_release_velocity=path.predicted_release_velocity.tolist(), predicted_release_time=path.predicted_release_time)
        return path

    @staticmethod
    def reception_bounds(side):
        lower = -RECEPTION_PARAMETER_LIMITS.copy()
        upper = RECEPTION_PARAMETER_LIMITS.copy()
        if side == 'right':
            lower[4] = -RECEIVE_INWARD_BANK
        else:
            upper[4] = RECEIVE_INWARD_BANK
        return lower, upper

    def reception_reference(self, side, parameters):
        """Build an offset catch at the current physical palm pose.

        The live optimizer and forecasts use the same reference construction.
        Only actuator references are returned; no ball state is changed.
        """
        hand = self.hands[side]
        parameters = np.asarray(parameters, dtype=float)
        data = self.data
        _, nominal_stop = hand['reception']
        bank = RECEIVE_INWARD_BANK if side == 'right' else -RECEIVE_INWARD_BANK
        axis = np.r_[parameters[3:] + [-RECEIVE_PALM_BACK_TILT, bank], 1.]
        axis /= np.linalg.norm(axis)
        mujoco.mj_copyData(hand['kin'].data, self.model, data)
        path = CartesianPalmMotion(
            hand['kin'], data.time, nominal_stop.t0-data.time,
            data.site_xpos[hand['seat']].copy(), self.spatial(hand)[3:].copy(),
            data.qpos[hand['kin'].qpos_adr].copy(),
            nominal_stop.start + parameters[:3], nominal_stop.velocity,
            axis=axis, nominal=hand['catch_q'], empty_arc=True)
        stop = SmoothStop(nominal_stop.t0, nominal_stop.duration,
                          nominal_stop.start + parameters[:3],
                          nominal_stop.velocity, self.gravity_m_s2)
        return path, stop, axis

    @staticmethod
    def apply_reception_reference(hand, path, stop, axis):
        hand['reception'] = (path.cartesian, stop)
        hand['approach_joint'] = path
        hand['approach_axis'] = axis
        hand['receiving_end'] = stop.sample(stop.t1)[0]

    def plan_reception(self, side):
        live = self.__dict__.copy()
        hand = self.hands[side]
        original = mujoco.MjData(self.model)
        mujoco.mj_copyData(original, self.model, self.data)
        _, old_stop = hand['reception']
        impact = old_stop.t0
        pre_time = impact - 0.015
        next_throw = max(hand['throw_time'] + 2*self.tau, impact + MIN_CATCH_RECOVERY_DURATION_S)
        lower, upper = self.reception_bounds(side)
        inward_bank = RECEIVE_INWARD_BANK if side == 'right' else -RECEIVE_INWARD_BANK

        def evaluate(parameters):
            trial = mujoco.MjData(self.model)
            mujoco.mj_copyData(trial, self.model, original)
            self.data = trial
            self.hands = {s: dict(h) for s, h in live['hands'].items()}
            self.paths = dict(live['paths'])
            self.targets = dict(live['targets'])
            self.airborne = dict(live['airborne'])
            self.caught_events = set(live['caught_events'])
            self.events = []
            self.planning = False
            self.silent = True
            path = None
            stop = None
            axis = np.r_[parameters[3:] + [-RECEIVE_PALM_BACK_TILT, inward_bank], 1.0]
            axis /= np.linalg.norm(axis)
            collision_free = True
            first_forbidden_contact = None

            def advance():
                nonlocal collision_free, first_forbidden_contact
                self.step()
                if collision_free:
                    try:
                        self.check_forbidden_contacts()
                    except RuntimeError:
                        # Silent rollouts skip the live guard. A collision
                        # must still prevent a zero-error catch certificate.
                        collision_free = False
                        first_forbidden_contact = self.failure_event

            try:
                path, stop, axis = self.reception_reference(side, parameters)
                h = self.hands[side]
                self.apply_reception_reference(h, path, stop, axis)
                while trial.time < pre_time - 1e-08:
                    advance()
                q, v = self.balls[hand['incoming']]
                n = trial.site_xmat[hand['seat']].reshape(3, 3)[:, 2]
                clearance = .0223 + .015 * (1-RECEIVE_VELOCITY_RATIO) * abs(trial.qvel[v+2])
                pre_error = trial.qpos[q:q+3] - trial.site_xpos[hand['seat']] - clearance*n
                # A catch certificate must include time to recover for the
                # next throw. Contact alone can settle after that beat and
                # leave the following launch without a usable starting state.
                while trial.time < impact + MAX_RECEPTION_SETTLE_FORECAST_S - 1e-08:
                    advance()
                    receiving = self.hands[side]
                    if (trial.time >= next_throw-THROW_READY_MARGIN_S and
                            receiving.get('held') == hand['incoming'] and
                            receiving.get('candidate_ball') == hand['incoming'] and
                            receiving.get('capture_confirmed', False) and receiving['mode'] == 'hold'):
                        break
                    if trial.time >= next_throw-1e-8:
                        break
                actual = self.hands[side]
                settled_center = trial.qpos[q:q+3] - trial.site_xpos[hand['seat']]
                settled_normal = trial.site_xmat[hand['seat']].reshape(3,3)[:,2]
                centered = np.linalg.norm(settled_center - (settled_center @ settled_normal)*settled_normal) < .018
                touched = actual.get('candidate_ball') == hand['incoming'] and actual.get('held') == hand['incoming']
                captured = touched and actual.get('capture_confirmed', False)
                ready = bool(captured and actual['mode'] == 'hold' and
                             trial.time <= next_throw+1e-8 and
                             (len(self.balls) != 3 or actual['throw_time'] <= next_throw+1e-8) and
                             self.ready_for_throw(side))
                timely = touched and abs(actual.get('contact_time', -10.) - impact) < .035
                path.catch_diagnostics = dict(
                    centered=bool(centered), timely=bool(timely), captured=bool(captured),
                    ready_for_next_throw=ready, next_throw_time=float(next_throw),
                    collision_free=bool(collision_free),
                    first_forbidden_contact=first_forbidden_contact,
                    supporting_digits=self.supporting_digits(hand['incoming'], side),
                    stable_age_s=float(actual.get('stable_age', 0.)),
                    settled_relative_speed=float(np.linalg.norm(
                        trial.qvel[self.balls[hand['incoming']][1]:self.balls[hand['incoming']][1]+3]
                        - self.spatial(actual)[3:])),
                    contact_time=actual.get('contact_time') if touched else None)
                if collision_free and centered and timely and captured and ready and self.ball_contact(hand['incoming'], side) and np.linalg.norm(trial.qvel[self.balls[hand['incoming']][1]:self.balls[hand['incoming']][1] + 3] - self.spatial(actual)[3:]) < .4:
                    error = np.zeros(5)
                else:
                    q, v = self.balls[hand['incoming']]
                    # Once physical contact occurs, optimize the settled cup
                    # position, rather than a pre-impact error that can be zero
                    # while the ball subsequently rolls out toward the fingers.
                    position_error = settled_center - SUPPORTED_BALL_CENTER_OFFSET_M*settled_normal if touched else pre_error
                    if not touched:
                        timing_error = .04
                    elif timely:
                        timing_error = 0.
                    else:
                        timing_error = float(np.clip(.8*(actual['contact_time']-impact), -.08, .08))
                    error = np.r_[position_error, 0. if captured and collision_free and ready else .04, timing_error]
            except ValueError:
                error = np.full(5, 10.0)
            finally:
                self.__dict__.clear()
                self.__dict__.update(live)
            return (error, path, stop, axis)
        initial = np.array([.025, .015 if side == 'left' else -.015, 0., 0., 0.])
        key = f'return:{side}:{self.throw_count}'
        parameters = np.clip(np.array(self.execution_warm.get(key, self.warm.get(key, hand.get('return_parameters', initial))), dtype=float), lower, upper)
        error, path, stop, axis = evaluate(parameters)
        best = (np.linalg.norm(error), parameters.copy(), error.copy(), path, stop, axis)
        if best[0] > 0:
            lateral = .015 if side == 'left' else -.015
            for offset in [np.zeros(5), np.array([.025, lateral, 0., 0., 0.]), np.array([0.0, 0.0, -0.025, 0.0, 0.0]), np.array([-0.02, 0.0, -0.025, 0.0, 0.0]), np.array([0.02, 0.0, -0.025, 0.0, 0.0]), np.array([0.0, 0.0, -0.04, 0.0, 0.0])]:
                pp = np.clip(offset, lower, upper)
                ep, path, stop, axis = evaluate(pp)
                score = np.linalg.norm(ep)
                if score < best[0]:
                    best = (score, pp.copy(), ep.copy(), path, stop, axis)
                if best[0] == 0:
                    break
        for _ in range(5):
            if best[0] < 0.012:
                break
            parameters = best[1].copy()
            error = best[2].copy()
            columns = []
            diagnostics = getattr(best[3], 'catch_diagnostics', {})
            steps = (RECEPTION_FINE_DIFFERENCE if diagnostics.get('captured') and diagnostics.get('timely') and diagnostics.get('ready_for_next_throw')
                     else RECEPTION_FINITE_DIFFERENCE)
            for i, step in enumerate(steps):
                pp = parameters.copy()
                pp[i] = np.clip(pp[i]+step, lower[i], upper[i])
                if pp[i] == parameters[i]:
                    pp[i] = np.clip(pp[i]-step, lower[i], upper[i])
                displacement = pp[i]-parameters[i]
                ep, path, stop, axis = evaluate(pp)
                columns.append((ep - error) / displacement)
                if np.linalg.norm(ep) < best[0]:
                    best = (np.linalg.norm(ep), pp.copy(), ep.copy(), path, stop, axis)
                if best[0] < .012:
                    break
            if best[0] < .012:
                break
            delta = np.linalg.lstsq(np.column_stack(columns), -error, rcond=0.005)[0]
            delta *= min(1.0, 1 / max(0.001, np.max(abs(delta) / [0.025, 0.02, 0.025, 0.1, 0.1])))
            improved = False
            for fraction in [1.0, 0.4]:
                pp = np.clip(parameters + fraction * delta, lower, upper)
                ep, path, stop, axis = evaluate(pp)
                score = np.linalg.norm(ep)
                if score < best[0]:
                    best = (score, pp.copy(), ep.copy(), path, stop, axis)
                    improved = True
                    break
            if not improved:
                break
        if best[0] > 0.02:
            rng = np.random.default_rng(42 + self.throw_count)
            for _ in range(16):
                pp = np.clip(best[1] + rng.normal(size=5) * [0.012, 0.01, 0.02, 0.05, 0.05], lower, upper)
                ep, path, stop, axis = evaluate(pp)
                score = np.linalg.norm(ep)
                if score < best[0]:
                    best = (score, pp.copy(), ep.copy(), path, stop, axis)
        _, parameters, error, path, stop, axis = best
        if path is None:
            raise ValueError('no reachable reception path')
        diagnostics = getattr(path, 'catch_diagnostics', {})
        self.log('return_plan', side, hand['incoming'], error=error.round(4).tolist(),
                 score=float(np.linalg.norm(error)), parameters=parameters.tolist(),
                  diagnostics=diagnostics)
        if not all(diagnostics.get(key, False) for key in ('captured', 'collision_free', 'ready_for_next_throw')):
            self.failure_event = dict(reason='NO_FEASIBLE_RECEPTION', time=float(self.data.time), side=side,
                                      ball=hand['incoming'])
            raise RuntimeError('NO_FEASIBLE_RECEPTION: no collision-free catch ready for the next throw')
        hand['return_parameters'] = parameters
        self.apply_reception_reference(hand, path, stop, axis)

    def step(self):
        """Advance contact dynamics. Count releases only after finger opening and 12 ms of actual hand separation."""
        m, d = (self.model, self.data)
        ball_positions_before_control = d.qpos[self.ball_position_indices].copy()
        ball_velocities_before_control = d.qvel[self.ball_velocity_indices].copy()
        mujoco.mj_step1(m, d)
        t = d.time
        zero = np.zeros(3)
        if len(self.balls) == 2 and self.throw_count >= 2 and self.throw_count % 2 == 0 and self.exchange_pair < self.throw_count // 2:
            ready = all(h['held'] is not None and h['mode'] == 'hold' and h.get('capture_confirmed', False) for h in self.hands.values())
            if ready:
                self.exchange_pair = self.throw_count // 2
                leader = 'left' if self.exchange_pair % 2 else 'right'
                for side, hand in self.hands.items():
                    hand['throw_time'] = t + .15 + (0. if side == leader else self.tau)
                self.log('exchange_ready', leader, -1, pair=self.exchange_pair)
            else:
                for hand in self.hands.values():
                    if hand['mode'] == 'hold':
                        hand['throw_time'] = max(hand['throw_time'], t + .10)
        for side, h in self.hands.items():
            if h['mode'] == 'wait':
                # The Cartesian target defines the catch. Keep the arm's
                # comfortable inner posture as its null-space preference;
                # the outer IK solution is only a reachability/branch seed.
                self.return_servo(h, d, h['q'], position=h['catch_p'], velocity=zero, acceleration=zero)
            if h['mode'] == 'hold' and t >= h['throw_time'] - 1e-08:
                if not self.ready_for_throw(side):
                    h['throw_time'] = t + 0.02
                elif self.planning:
                    candidate = self.plan_throw(side)
                    if self.last_plan_error > 0.02:
                        h['throw_time'] = t + THROW_REPLAN_INTERVAL_S
                    else:
                        self.paths[side] = candidate
            if h['mode'] == 'hold' and t >= h['throw_time'] - 1e-08:
                if not self.planning and abs(getattr(self.paths[side], 'planned_time', -10.) - t) > 1e-8:
                    # A private rollout can cross the OTHER hand's next throw.
                    # Rebuild its reference at that future state instead of
                    # replaying the previous throw's old Cartesian origin.
                    command = np.asarray(self.warm.get(f'throw:{side}:{self.throw_count}', self.commands[side]))
                    mujoco.mj_copyData(h['kin'].data, m, d)
                    self.paths[side] = ThrowPath(h['kin'], d.site_xpos[h['seat']].copy(), command[:3], d.qpos[h['kin'].qpos_adr].copy(), h['catch_p'], bank=command[3], catch_q=h['catch_q'])
                    self.paths[side].finger_release_lead_s = float(command[4]) if command.size == 5 else 0.
                    self.paths[side].planned_time = t
                h['mode'] = 'throw'
                h['stroke'] = t
                # Continue from the yielded holding target, without squeezing
                # back to a preset cradle immediately before opening.
                h['release_start'] = d.ctrl[h['fingers'].act_indices].copy()
                h['path'] = self.paths[side]
                h['release_pending'] = None
                q, v = self.balls[h['held']]
                self.log('stroke', side, h['held'], offset=(d.qpos[q:q + 3] - d.site_xpos[h['seat']]).round(4).tolist(), velocity=d.qvel[v:v + 3].round(3).tolist())
            if h['mode'] == 'throw':
                age = t - h['stroke']
                if age >= .11 and (not h['opening_ready']):
                    h['opening_ready'] = True
                if age <= h['path'].duration and h['held'] is not None:
                    self.follow_palm(h, d, h['path'], age, h['q'])
                else:
                    h['mode'] = 'return'
                    qnow = d.qpos[h['kin'].qpos_adr].copy()
                    mujoco.mj_copyData(h['kin'].data, m, d)
                    start = d.site_xpos[h['seat']].copy()
                    velocity = self.spatial(h)[3:].copy()
                    normal = d.site_xmat[h['seat']].reshape(3, 3)[:, 2].copy()
                    h['brake'] = CartesianPalmMotion(h['kin'], t, .30, start, velocity, qnow, h['catch_p'], np.zeros(3), axis=np.array([0., 0., 1.]), nominal=h['catch_q'], empty_arc=h['held'] is None, arc_peak=.075)
            if h['mode'] == 'return':
                self.follow_palm(h, d, h['brake'], t, h['q'])
            if h['held'] is None and h['reception'] is None:
                for i, target in list(self.targets.items()):
                    if target != side:
                        continue
                    q, v = self.balls[i]
                    p = d.qpos[q:q + 3]
                    vel = d.qvel[v:v + 3]
                    if self.obs_noise and (not self.silent):
                        p = p + self.rng.normal(0, self.obs_noise, 3)
                    disc = vel[2] ** 2 + 2 * self.gravity_m_s2 * (p[2] - h['catch_p'][2] - SUPPORTED_BALL_CENTER_OFFSET_M)
                    dt = (vel[2] + np.sqrt(max(0, disc))) / self.gravity_m_s2
                    if 0.025 < dt < 0.3:
                        catch = h['catch_p'].copy()
                        catch[:2] = p[:2] + vel[:2] * dt
                        speed = vel.copy() * RECEIVE_TANGENTIAL_VELOCITY_RATIO
                        speed[2] = (vel[2] - self.gravity_m_s2 * dt) * RECEIVE_VELOCITY_RATIO
                        end = catch + 0.06 * speed
                        approach = Quintic(t, dt, d.site_xpos[h['seat']].copy(), self.spatial(h)[3:], zero, catch, speed, zero)
                        stop = SmoothStop(t + dt, CATCH_CUSHION_DURATION_S, catch, speed, self.gravity_m_s2)
                        h.update(mode='receive', reception=(approach, stop), incoming=i, receiving_end=end, contact_age=0.0, touchdown=False)
                        bank = RECEIVE_INWARD_BANK if side == 'right' else -RECEIVE_INWARD_BANK
                        h['approach_axis'] = np.array([-RECEIVE_PALM_BACK_TILT, bank, 1.0])
                        h['approach_axis'] /= np.linalg.norm(h['approach_axis'])
                        mujoco.mj_copyData(h['kin'].data, m, d)
                        path = CartesianPalmMotion(h['kin'], t, dt, d.site_xpos[h['seat']].copy(), self.spatial(h)[3:].copy(), d.qpos[h['kin'].qpos_adr].copy(), catch, speed, axis=h['approach_axis'], nominal=h['catch_q'], empty_arc=True)
                        h['approach_joint'] = path
                        h['reception'] = (path.cartesian, stop)
                        if self.planning:
                            self.plan_reception(side)
                        else:
                            # Forecast the other hand's calibrated catch as
                            # well as its throw. Rebuild at this trial's actual
                            # origin; live optimizer guesses remain separate.
                            key = f'return:{side}:{self.throw_count}'
                            if key in self.warm:
                                parameters = np.clip(np.asarray(self.warm[key], dtype=float),
                                                     *self.reception_bounds(side))
                                reference = self.reception_reference(side, parameters)
                                self.apply_reception_reference(h, *reference)
                        self.log('prepare', side, i, impact=round(t + dt, 3), point=catch.round(3).tolist())
            if h['mode'] == 'receive':
                i = h['incoming']
                q, v = self.balls[i]
                approach, stop = h['reception']
                if self.ball_contact(i, side, palm_only=True):
                    h['touchdown'] = True
                if t <= approach.t1:
                    pp, vv, aa = approach.sample(t)
                    self.follow_palm(h, d, h['approach_joint'], t, h['q'])
                else:
                    pp, vv, aa = stop.sample(t)
                    self.return_servo(h, d, h['catch_q'], position=pp, velocity=vv, acceleration=aa, axis_goal=h['approach_axis'])
                relative = d.qvel[v:v + 3] - self.spatial(h)[3:]
                error = d.qpos[q:q + 3] - d.site_xpos[h['seat']] - [0, 0, SUPPORTED_BALL_CENTER_OFFSET_M]
                normal = d.site_xmat[h['seat']].reshape(3, 3)[:, 2]
                center = d.qpos[q:q + 3] - d.site_xpos[h['seat']]
                palm_side = float(center @ normal)
                tangent = np.linalg.norm(center - palm_side * normal)
                stable = self.ball_contact(i, side, palm_only=True) and palm_side > 0.01 and (tangent < 0.025) and (abs(relative @ normal) < 4.0) and (np.linalg.norm(error) < 0.07)
                h['contact_age'] = h['contact_age'] + m.opt.timestep if stable else 0.0
                if h['held'] is None and h['contact_age'] >= m.opt.timestep:
                    self.begin_capture(side, i)
                    h['held'] = i
                    h['candidate_flight'] = self.airborne[i]
                    h['candidate_ball'] = i
                    h['capture_confirmed'] = False
                    h['stable_age'] = 0.0
                    h['touch_palm'] = d.site_xpos[h['seat']].copy()
                    h['lowest_capture_palm_z'] = float(d.site_xpos[h['seat'], 2])
                    h['touch_ball_velocity'] = d.qvel[v:v + 3].copy()
                    h['touch_palm_velocity'] = self.spatial(h)[3:].copy()
                    self.log('touch', side, i, release_id=h['candidate_flight'], **h['contact_evidence'])
                    h['mode'] = 'carry'
                    h['reception'] = None
                    end_time = max(h['throw_time'] + 2 * self.tau, t + MIN_CATCH_RECOVERY_DURATION_S)
                    h['throw_time'] = end_time
                    mujoco.mj_copyData(h['kin'].data, m, d)
                    h['carry'] = CartesianPalmMotion(h['kin'], t, end_time - t - 0.015, d.site_xpos[h['seat']].copy(), self.spatial(h)[3:].copy(), d.qpos[h['kin'].qpos_adr].copy(), h['p'], np.zeros(3), axis=LOADED_SOURCE_AXIS, nominal=h['q'], tilt_deg=LOADED_RECOVERY_TILT_DEG, arc_peak=LOADED_RECOVERY_PEAK_S, cushion_duration=CAPTURE_BRAKING_DURATION_S)
                    self.log('carry', side, i, next_throw=round(end_time, 3))
            if h['mode'] == 'carry':
                motion = h['carry']
                self.follow_palm(h, d, motion, t, h['q'])
                if t >= motion.t1 + 0.005:
                    h['mode'] = 'hold'
            if h['mode'] == 'hold':
                q, v = self.balls[h['held']]
                self.return_servo(h, d, h['q'], position=h['p'], velocity=zero, acceleration=zero, axis_goal=self.holding_axis(h))
            if h['mode'] == 'throw':
                d.ctrl[h['fingers'].act_indices] = self.release_fingers(h, t)
            elif h['mode'] in ('return', 'wait'):
                h['fingers'].apply_posture(d, 'flat')
            elif h['mode'] == 'receive' and h['held'] is None:
                phase = np.clip((t - h['reception'][1].t0 + RECEIVE_PREPARE_LEAD_S) / RECEIVE_PREPARE_DURATION_S, 0., 1.)
                weight = phase**3 * (10 - 15*phase + 6*phase**2)
                fingers = h['fingers']
                d.ctrl[fingers.act_indices] = (1-weight)*fingers.get_ctrl_array('flat') + weight*fingers.get_ctrl_array('catch')
            else:
                d.ctrl[h['fingers'].act_indices] = self.holding_fingers(h, t)
        if not np.array_equal(d.qpos[self.ball_position_indices], ball_positions_before_control) or not np.array_equal(d.qvel[self.ball_velocity_indices], ball_velocities_before_control):
            raise RuntimeError('control modified a live ball position or velocity')
        if np.any(d.qfrc_applied[self.ball_velocity_indices]) or np.any(d.xfrc_applied[self.ball_body_ids]):
            raise RuntimeError('an artificial force was applied to a ball')
        if not self.silent:
            self.contact_only_checks += len(self.balls)
        mujoco.mj_step2(m, d)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        for side, h in self.hands.items():
            i = h['held']
            if i is not None and h.get('candidate_ball') == i and not h.get('capture_confirmed', False):
                h['lowest_capture_palm_z'] = min(h['lowest_capture_palm_z'], float(d.site_xpos[h['seat'], 2]))
                q, v = self.balls[i]
                normal = d.site_xmat[h['seat']].reshape(3, 3)[:, 2]
                center = d.qpos[q:q + 3] - d.site_xpos[h['seat']]
                relative = d.qvel[v:v + 3] - self.spatial(h)[3:]
                digits = self.supporting_digits(i, side)
                # Preserve only a few milliseconds of real digit contact.
                # Replace the dictionary so private rollouts cannot mutate
                # live tactile history through their shallow hand copies.
                recent = {digit: stamp for digit, stamp in h.get('digit_contact_times', {}).items()
                          if d.time-stamp <= DIGIT_CONTACT_MEMORY_S}
                recent.update({digit: float(d.time) for digit in digits})
                h['digit_contact_times'] = recent
                supported = self.ball_contact(i, side) and .010 < center @ normal < .06 and np.linalg.norm(center - (center @ normal) * normal) < .04 and np.linalg.norm(relative) < .18 and 'th' in recent and len(recent) >= 2
                h['stable_age'] = h['stable_age'] + m.opt.timestep if supported else 0.0
                # Contact support alone is not a completed cushioned catch.
                # Match the existing regression's speed-decay requirements,
                # including catches confirmed during the loaded recovery.
                impact_absorbed = (
                    np.linalg.norm(relative) < CAPTURE_RELATIVE_SPEED_RATIO * np.linalg.norm(h['touch_ball_velocity']-h['touch_palm_velocity'])
                    and np.linalg.norm(d.qvel[v:v+3]) < np.linalg.norm(h['touch_ball_velocity']))
                if (d.time - h['contact_time'] >= .070 and h['stable_age'] >= .020
                        and 'th' in digits and len(digits) >= 2
                        and self.ball_contact(i, side, palm_only=True) and impact_absorbed):
                    h['capture_confirmed'] = True
                    captured_event = self.airborne.pop(i)
                    self.caught_events.add(captured_event)
                    self.targets.pop(i)
                    if captured_event > 0:
                        self.catch_count += 1
                    self.log('catch', side, i, release_id=captured_event,
                             palm_touch_time=float(h['contact_time']),
                             contact_settle_time=float(d.time - h['contact_time']),
                               cushion_depth=float(h['touch_palm'][2] - h['lowest_capture_palm_z']),
                               palm_drop_at_settle=float(h['touch_palm'][2] - d.site_xpos[h['seat'], 2]),
                             incoming_ball_speed=float(np.linalg.norm(h['touch_ball_velocity'])),
                               touch_palm_vertical_velocity=float(h['touch_palm_velocity'][2]),
                             initial_relative_speed=float(np.linalg.norm(h['touch_ball_velocity'] - h['touch_palm_velocity'])),
                             settled_relative_speed=float(np.linalg.norm(relative)),
                             settled_ball_speed=float(np.linalg.norm(d.qvel[v:v + 3])),
                             supporting_digits=self.supporting_digits(i, side),
                             **h['contact_evidence'])
            if i is not None and h['mode'] in ('throw', 'return'):
                q, v = self.balls[i]
                if h['opening_ready'] and (not self.ball_contact(i, side)) and (d.qvel[v + 2] > 1.0):
                    if h['release_pending'] is None:
                        h['release_pending'] = d.time
                    if d.time - h['release_pending'] >= RELEASE_CONFIRMATION_S:
                        self.throw_count += 1
                        self.airborne[i] = self.next_event
                        self.next_event += 1
                        self.targets[i] = 'left' if side == 'right' else 'right'
                        h['held'] = None
                        h['capture_confirmed'] = False
                        h['candidate_ball'] = None
                        h['digit_contact_times'] = {}
                        self.log('release', side, i, count=self.throw_count,
                                 separation_time=float(h['release_pending']),
                                 confirmation_delay=float(d.time-h['release_pending']),
                                 position=d.qpos[q:q + 3].tolist(), velocity=d.qvel[v:v + 3].tolist())
                else:
                    h['release_pending'] = None
        for i, (q, v) in enumerate(self.balls):
            self.minimum_height = min(self.minimum_height, float(d.qpos[q + 2]))
            if d.qpos[q + 2] < BALL_DROP_HEIGHT_M and (not self.silent):
                self.drop_event = dict(ball=i, time=float(d.time), height=float(d.qpos[q + 2]))
                self.failure_event = dict(reason='DROPPED', **self.drop_event)
                raise RuntimeError(f'DROPPED: ball {i} at {d.time:.3f}, throws={self.throw_count}')
            for j in range(i):
                distance = float(np.linalg.norm(d.qpos[q:q + 3] - d.qpos[self.balls[j][0]:self.balls[j][0] + 3]))
                self.minimum_distance = min(self.minimum_distance, distance)
        ids = self.ball_geom_set
        if not self.silent:
            self.airborne_physics_checks += len(self.airborne)
        self.ball_contacts += sum((c.geom1 in ids and c.geom2 in ids for c in d.contact))
        for c in d.contact:
            if c.geom1 in ids and c.geom2 in ids and (self.ball_contacts < 10):
                self.log('collision', '', -1, pair=[m.geom(c.geom1).name, m.geom(c.geom2).name])
        if not self.silent:
            self.record_pose_metrics()
            self.check_forbidden_contacts()
        # Constraint forces correspond to the contact rows after step2.
        # Use them for the next control tick, not before the solver runs.
        for hand in self.hands.values():
            if hand['held'] is not None and hand['mode'] in ('hold','carry'):
                self.update_grip(hand,d.time)

    def check_forbidden_contacts(self):
        """Fail on ball/link, floor and cross-arm contact, as in v5.

        Hand self-contact is part of grasping. G1 also has adjacent torso
        links, so only cross-arm robot contacts are classified here.
        """
        def owner(body):
            name = self.model.body(body).name
            if name.startswith(('rh_', 'right_')):
                return 'right'
            if name.startswith(('lh_', 'left_')):
                return 'left'
            return None

        hands = self.hand_geoms['right'] | self.hand_geoms['left']
        for contact in self.data.contact:
            if contact.efc_address < 0 or contact.dist > 0:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            reason = None
            if a in self.ball_geom_set or b in self.ball_geom_set:
                other = b if a in self.ball_geom_set else a
                if other in self.ball_geom_set:
                    reason = 'BALL_COLLISION'
                elif other not in hands:
                    reason = 'DROPPED' if self.model.geom(other).name in ('ground', 'torso_stand') else 'HIT_ARM'
            else:
                first, second = owner(self.model.geom_bodyid[a]), owner(self.model.geom_bodyid[b])
                if first and second and first != second:
                    reason = 'ARM_COLLISION'
            if reason is not None:
                labels = [self.model.geom(g).name or self.model.body(self.model.geom_bodyid[g]).name for g in (a, b)]
                self.failure_event = dict(reason=reason, time=float(self.data.time),
                                          geoms=labels)
                raise RuntimeError(f'{reason}: {self.failure_event["geoms"]}')
        self.check_pose_limits()

    def check_pose_limits(self):
        """Stop a failed recovery before the arms leave the validated workspace.

        Use the same instantaneous limits as the existing motion regressions;
        private forecasts must reject these states as well as the live run.
        """
        model, data = self.model, self.data
        offset = getattr(self, 'torso_stand_height', DEFAULT_STAND_HEIGHT_M)-DEFAULT_STAND_HEIGHT_M

        def fail(part, quantity, actual, limit):
            self.failure_event = dict(reason='POSE_LIMIT', time=float(data.time),
                                      part=part, quantity=quantity,
                                      actual=float(actual), limit=float(limit))
            raise RuntimeError(f'POSE_LIMIT: {part} {quantity}={actual:.6f}, limit={limit:.6f}')

        for index, (q, _) in enumerate(self.balls):
            if data.qpos[q+2] >= MAX_BALL_HEIGHT_M+offset:
                fail(f'ball{index}', 'ball_height_m', data.qpos[q+2], MAX_BALL_HEIGHT_M+offset)
        for side, hand in self.hands.items():
            palm_z = float(data.site_xpos[hand['seat'], 2])
            if palm_z > MAX_PALM_HEIGHT_M+offset:
                fail(side, 'palm_height_m', palm_z, MAX_PALM_HEIGHT_M+offset)
            clearance = (data.xpos[model.body(side+'_shoulder_pitch_link').id, 2]
                         - data.xpos[model.body(side+'_elbow_link').id, 2])
            if clearance < MIN_SHOULDER_ELBOW_CLEARANCE_M:
                fail(side, 'shoulder_elbow_clearance_m', clearance, MIN_SHOULDER_ELBOW_CLEARANCE_M)
            elbow_out = (abs(data.xpos[model.body(side+'_elbow_link').id, 1])
                         - abs(data.xpos[model.body(side+'_shoulder_pitch_link').id, 1]))
            if elbow_out >= MAX_ELBOW_OUT_M:
                fail(side, 'elbow_out_m', elbow_out, MAX_ELBOW_OUT_M)
            names = ARM_JOINTS_RIGHT if side == 'right' else ARM_JOINTS_LEFT
            ranges = model.jnt_range[[model.joint(name).id for name in names]]
            q = data.qpos[hand['kin'].qpos_adr]
            violations = np.maximum(ranges[:, 0]-q, q-ranges[:, 1])
            index = int(np.argmax(violations))
            if violations[index] > MAX_ARM_JOINT_VIOLATION_RAD:
                fail(names[index], 'joint_limit_violation_rad', violations[index], MAX_ARM_JOINT_VIOLATION_RAD)
            # A physically legal wrist angle can still exceed the original
            # motion regression's comfort limit. Reject that launch/reception
            # inside the private forecast, before it reaches the live run.
            wrist_pitch = abs(float(q[5]))
            if wrist_pitch >= MAX_WRIST_PITCH_RAD:
                fail(side+'_wrist_pitch_joint', 'wrist_pitch_rad', wrist_pitch, MAX_WRIST_PITCH_RAD)
        for contact in data.contact:
            if contact.dist >= -MAX_SELF_PENETRATION_M:
                continue
            names = [model.body(model.geom_bodyid[g]).name for g in (contact.geom1, contact.geom2)]
            if any(name.startswith('ball') for name in names):
                continue
            if all(name.startswith('rh_') for name in names) or all(name.startswith('lh_') for name in names):
                continue
            fail('/'.join(names), 'self_penetration_m', -contact.dist, MAX_SELF_PENETRATION_M)

    def supporting_digits(self, ball, side):
        digits = set()
        prefix = 'rh_' if side == 'right' else 'lh_'
        for index, contact in enumerate(self.data.contact):
            other = contact.geom2 if contact.geom1 == self.ball_geom_ids[ball] else contact.geom1 if contact.geom2 == self.ball_geom_ids[ball] else -1
            if other not in self.hand_geoms[side]:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            if force[0] <= .02:
                continue
            name = self.model.body(self.model.geom_bodyid[other]).name
            digit = name[len(prefix):len(prefix) + 2]
            if digit in self.hands[side]['fingers'].digits:
                digits.add(digit)
        return sorted(digits)

    def release_fingers(self, h, t):
        """Ease out of the cup with small digit offsets instead of opening in lockstep."""
        fingers = h['fingers']
        closed = h['release_start'] if 'release_start' in h else self.holding_fingers(h, t)
        relaxed = fingers.get_ctrl_array('flat')
        starts = fingers.release_onset
        lead = getattr(h.get('path'), 'finger_release_lead_s', 0.)
        if lead:
            t += lead
        phase = np.clip((t - h['stroke'] - starts) / .052, 0., 1.)
        weights = phase**3 * (10 - 15*phase + 6*phase**2)
        return (1 - weights) * closed + weights * relaxed

    def update_grip(self, h, t):
        """Yield excess grip through real digit forces, without attaching the ball."""
        side = h['kin'].side
        if t - h.get('contact_time', t) < CAPTURE_FORCE_FEEDBACK_DELAY_S or not self.ball_contact(h['held'], side):
            return
        # Complete contact-supported wrapping before reducing the grip. Early
        # fingertip forces include impact loads and should not open the cup.
        confirmed = h.get('capture_confirmed', True)
        if confirmed and h.get('mode') != 'hold':
            return
        groups = h['fingers'].digits
        forces = np.zeros(len(groups))
        ball = self.ball_geom_ids[h['held']]
        prefix = 'rh_' if side == 'right' else 'lh_'
        for index, contact in enumerate(self.data.contact):
            other = contact.geom2 if contact.geom1 == ball else contact.geom1 if contact.geom2 == ball else -1
            if other not in self.hand_geoms[side] or contact.efc_address < 0:
                continue
            digit = self.model.body(self.model.geom_bodyid[other]).name[len(prefix):len(prefix)+2]
            if digit not in groups:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            forces[groups.index(digit)] += max(0., wrench[0])
        if not confirmed:
            # A geometric touch may carry no normal force. Establish real
            # digit support before asking for capture confirmation. Impact
            # forces may stop closure, but cannot open an unconfirmed cup.
            closing = np.clip(.5*(1.-forces), 0., 1.)
            h['grip_offsets'] = np.minimum(.03, h['grip_offsets'] + self.model.opt.timestep*closing)
            return
        # Replace arrays rather than mutating them: prediction hands are copies.
        h['grip_force'] = .85*h['grip_force'] + .15*forces
        target = np.array([1., 1., 1., .35])
        rate = np.clip(.5*(target - h['grip_force']), -1., 1.)
        h['grip_offsets'] = np.clip(h['grip_offsets'] + self.model.opt.timestep*rate, -.18, .03)

    def holding_fingers(self, h, t):
        phase = np.clip((t - h.get('contact_time', -0.2) - h['closure_onset']) / h['closure_duration'], 0.0, 1.0)
        weight = phase ** 3 * (10 - 15 * phase + 6 * phase ** 2)
        closed = h['fingers'].get_ctrl_array('cradle')
        if h.get('held') is not None:
            q, _ = self.balls[h['held']]
            rotation = self.data.site_xmat[h['seat']].reshape(3,3)
            error = rotation.T @ (self.data.qpos[q:q+3] - self.data.site_xpos[h['seat']]) - [0.,0.,BALL_SEAT_CENTER_OFFSET_M]
            closed = closed + h['fingers'].wrapping_correction(error)
            fingers = h['fingers']
            thumb_rows = np.array([index for index,digit in enumerate(fingers.actuator_digits) if digit == 'th'])
            if not h.get('capture_confirmed', True) and t-h['contact_time'] >= DELAYED_OPPOSITION_ONSET_S:
                h['opposition_offset'] = fingers.opposition_correction(
                    self.data, self.ball_geom_ids[h['held']], closed, h['opposition_offset'])
            closed[thumb_rows] += h['opposition_offset']
        closed = closed + h['grip_map'] @ h.get('grip_offsets', np.zeros(len(h['fingers'].digits)))
        bounds = self.model.actuator_ctrlrange[h['fingers'].act_indices]
        closed = np.clip(closed, bounds[:, 0], bounds[:, 1])
        return (1 - weight) * h.get('closure_start', h['fingers'].get_ctrl_array('catch')) + weight * closed

    def record_pose_metrics(self):
        """Measure the real trajectory, excluding private planning rollouts."""
        m, d = (self.model, self.data)
        self.execution_metrics.record(d)
        sample_posture = d.time >= self.next_posture_sample
        self.max_height = max(self.max_height, *(float(d.qpos[q + 2]) for q, _ in self.balls))
        for side, hand in self.hands.items():
            self.max_palm_height = max(self.max_palm_height, float(d.site_xpos[hand['seat'], 2]))
            shoulder = d.xpos[m.body(side + '_shoulder_pitch_link').id, 2]
            elbow = d.xpos[m.body(side + '_elbow_link').id, 2]
            self.min_shoulder_above_elbow = min(self.min_shoulder_above_elbow, float(shoulder - elbow))
            joints = [m.joint(n).id for n in (ARM_JOINTS_RIGHT if side == 'right' else ARM_JOINTS_LEFT)]
            q = d.qpos[hand['kin'].qpos_adr]
            limits = m.jnt_range[joints]
            self.max_joint_limit_violation = max(self.max_joint_limit_violation, float(np.max(np.maximum(limits[:, 0] - q, q - limits[:, 1]))))
            if sample_posture:
                elbow_point = d.xpos[m.body(side + '_elbow_link').id]
                shoulder_point = d.xpos[m.body(side + '_shoulder_pitch_link').id]
                forearm = d.xpos[m.body(side + '_wrist_yaw_link').id] - elbow_point
                forearm = forearm / np.linalg.norm(forearm)
                forward = d.site_xmat[hand['seat']].reshape(3, 3)[:, 1]
                self.posture_samples.append((float(np.arccos(np.clip(forearm @ forward, -1., 1.))), float(abs(elbow_point[1]) - abs(shoulder_point[1])), float(abs(q[5]))))
            point = d.site_xpos[hand['seat']].copy()
            rotation = d.site_xmat[hand['seat']].reshape(3, 3)
            limits = self.palm_ranges[side]
            limits['ymin'] = min(limits['ymin'], float(point[1]))
            limits['ymax'] = max(limits['ymax'], float(point[1]))
            limits['zmin'] = min(limits['zmin'], float(point[2]))
            limits['zmax'] = max(limits['zmax'], float(point[2]))
            if hand['mode'] == 'throw':
                self.active_strokes.setdefault(side, []).append(point)
                self.min_throw_palm_up = min(self.min_throw_palm_up, float(rotation[2, 2]))
                self.min_throw_fingers_forward = min(self.min_throw_fingers_forward, float(rotation[0, 1]))
            elif side in self.active_strokes:
                points = np.array(self.active_strokes.pop(side))
                self.stroke_metrics.append(dict(side=side, rise=float(points[-1, 2] - points[0, 2]), sideways=float(np.ptp(points[:, 1])), vertical=float(np.ptp(points[:, 2]))))
        if sample_posture:
            self.torso_samples.append(d.qpos[self.waist_position_indices].copy())
            self.next_posture_sample += .01
        for contact in d.contact:
            names = [m.body(m.geom_bodyid[g]).name for g in (contact.geom1, contact.geom2)]
            if any((name.startswith('ball') for name in names)):
                continue
            if all((name.startswith('rh_') for name in names)) or all((name.startswith('lh_') for name in names)):
                continue
            self.max_self_penetration = max(self.max_self_penetration, -float(contact.dist))

    def style_summary(self):
        """Robot alignment metrics; these are not universal human joint limits."""
        if not self.posture_samples:
            return {}
        values = np.asarray(self.posture_samples)
        return dict(wrist_angle_p95_deg=float(np.rad2deg(np.percentile(values[:, 0], 95))),
                    wrist_angle_rms_deg=float(np.rad2deg(np.sqrt(np.mean(values[:, 0]**2)))),
                    max_elbow_out_m=float(np.max(values[:, 1])),
                    elbow_out_p95_m=float(np.percentile(values[:, 1], 95)),
                    wrist_pitch_p95_deg=float(np.rad2deg(np.percentile(values[:, 2], 95))),
                    wrist_pitch_max_deg=float(np.rad2deg(np.max(values[:, 2]))),
                    torso_range_deg=np.rad2deg(np.ptp(np.asarray(self.torso_samples), axis=0)).tolist())

    def conditions_summary(self):
        """Effective experiment settings, including calibrated differences."""
        from .model import SKIN_SUFFIXES, SKIN_SOLREF, LEAP_SEAT_SURFACE
        from .contact_control import SCOOP_END_ANGLE_RAD, THROW_PALM_BACK_TILT, THROW_LATERAL_PALM_LEAN, ELBOW_LATERAL_POSTURE_GAIN
        from .contact_control import WRIST_THROW_LOOKAHEAD_S
        motor = self.hands['right']['fingers'].act_indices[0]
        ball = self.ball_geom_ids[0]
        pulse = self.paths['right'].cartesian
        soft_links = any(self.model.geom_contype[g] and
                         not self.model.geom(g).name.removeprefix('rh_').removeprefix('lh_').startswith(SKIN_SUFFIXES) and
                         np.array_equal(self.model.geom_solref[g], SKIN_SOLREF)
                         for g in self.hand_geoms['right'] | self.hand_geoms['left'])
        return dict(mujoco_version=mujoco.__version__, timestep_s=float(self.model.opt.timestep),
                    gravity_m_s2=self.model.opt.gravity.tolist(),
                    ball_count=len(self.balls),
                    ball_diameter_m=float(2*self.model.geom_size[ball, 0]),
                    ball_mass_kg=float(self.model.body_mass[self.model.geom_bodyid[ball]]),
                    finger_position_gain=float(self.model.actuator_gainprm[motor, 0]),
                    finger_velocity_gain=float(-self.model.actuator_biasprm[motor, 2]),
                    finger_force_limit_nm=self.model.actuator_forcerange[motor].tolist(),
                    hand_material_rgba={prefix + name: self.model.mat_rgba[self.model.material(prefix + name).id].tolist()
                                        for prefix in ('rh_', 'lh_') for name in ('black', 'white')},
                    soft_finger_links=bool(soft_links),
                    observation='MuJoCo position and velocity truth',
                    partial_prediction_noise_m=float(self.obs_noise),
                    observation_noise_seed=self.seed,
                    source_positions_m={side: h['p'].tolist() for side, h in self.hands.items()},
                    catch_positions_m={side: h['catch_p'].tolist() for side, h in self.hands.items()},
                    wrist_seat_offsets_m={side: (
                        self.data.xmat[self.model.body(side+'_wrist_yaw_link').id].reshape(3, 3).T @
                        (self.data.site_xpos[h['seat']] - self.data.xpos[self.model.body(side+'_wrist_yaw_link').id])
                    ).tolist() for side, h in self.hands.items()},
                    palm_mount_quaternions_wxyz={side: self.model.body_quat[
                        self.model.body(prefix+'palm').id].tolist()
                        for side, prefix in (('right', 'rh_'), ('left', 'lh_'))},
                    initial_ball_states=self.initial_ball_states,
                    palm_seat_surfaces_m={side: list(point) for side, point in LEAP_SEAT_SURFACE.items()},
                    throw_interval_s=float(self.tau), flight_target_s=float(self.flight),
                    torso_stand_height_m=float(getattr(self, 'torso_stand_height', DEFAULT_STAND_HEIGHT_M)),
                    planned_apex_limit_m=float(self.max_planned_apex),
                    scoop_acceleration_duration_s=float(pulse.launch.phase.duration),
                    scoop_braking_duration_s=float(pulse.brake.duration),
                    post_release_braking_preview_s=THROW_BRAKING_PREVIEW_S,
                    scoop_end_angle_rad=float(SCOOP_END_ANGLE_RAD),
                    throw_palm_back_tilt=float(THROW_PALM_BACK_TILT),
                    throw_lateral_palm_lean=float(THROW_LATERAL_PALM_LEAN),
                    throw_command_fields=['vx', 'vy', 'vz', 'fore_aft_normal_bias', 'finger_release_lead_s'],
                    default_finger_release_lead_s=0.,
                    finger_release_lead_limit_s=MAX_FINGER_RELEASE_LEAD_S,
                    loaded_source_tilt_deg=LOADED_SOURCE_TILT_DEG,
                    initial_hold_tilt_duration_s=INITIAL_HOLD_TILT_DURATION_S,
                    throw_ready_margin_s=THROW_READY_MARGIN_S,
                    minimum_loaded_recovery_s=MIN_CATCH_RECOVERY_DURATION_S,
                    elbow_lateral_posture_gain=ELBOW_LATERAL_POSTURE_GAIN,
                    wrist_throw_lookahead_s=WRIST_THROW_LOOKAHEAD_S,
                    waist_yaw_stiffness=float(self.model.actuator_gainprm[self.model.actuator('waist_yaw_joint').id, 0]),
                    digit_contact_memory_s=DIGIT_CONTACT_MEMORY_S,
                    capture_force_feedback_delay_s=CAPTURE_FORCE_FEEDBACK_DELAY_S,
                    capture_relative_speed_ratio=CAPTURE_RELATIVE_SPEED_RATIO,
                    release_confirmation_s=RELEASE_CONFIRMATION_S)

    @property
    def total_catches_with_initial(self):
        return self.catch_count + int(0 in self.caught_events)

    def run_continuous(self, throws=30, timeout=25.0, guard=False):
        """Verify real releases and catches. Guard mode continues until every requested release ID has been caught, including the final flight."""
        required = set(range(1, throws + 1))
        failure = self.failure_event['reason'] if self.failure_event is not None else None
        try:
            while failure is None and self.data.time < timeout:
                if self.throw_count >= throws and (not guard or required.issubset(self.caught_events)):
                    break
                self.step()
                if self.ball_contacts:
                    raise RuntimeError('ball-ball contact detected')
        except RuntimeError as error:
            failure = str(error)
        if failure is None and not (self.throw_count >= throws and
                                     (not guard or required.issubset(self.caught_events))):
            failure = 'TIMEOUT'
        success = failure is None and self.throw_count >= throws and (not guard or required.issubset(self.caught_events)) and (self.ball_contacts == 0) and (self.minimum_height > BALL_DROP_HEIGHT_M)
        releases = [event for event in self.events if event['kind'] == 'release']
        caught = [event for event in self.events if event['kind'] == 'catch']
        releases_by_hand = {side: sum(event['side'] == side for event in releases) for side in self.hands}
        catches_by_hand = {side: sum(event['side'] == side and event['release_id'] > 0 for event in caught)
                           for side in self.hands}
        completed_by_source = {side: sum(event['side'] == side and event['count'] in required & self.caught_events
                                        for event in releases) for side in self.hands}
        separated = {event['count']: event['separation_time'] for event in releases}
        flight_times = [event['palm_touch_time'] - separated[event['release_id']]
                        for event in caught if event['release_id'] in required & separated.keys()]
        return dict(
            success=success,
            error=failure,
            requested_throws=throws,
            throws=self.throw_count,
            catches=self.catch_count,
            total_catches_with_initial=self.total_catches_with_initial,
            releases_by_hand=releases_by_hand,
            catches_by_hand=catches_by_hand,
            completed_requested_by_throwing_hand=completed_by_source,
            initial_airborne_catches=sum(event['release_id'] == 0 for event in caught),
            counting='global release IDs; catches exclude initial airborne ball unless explicitly noted',
            failure_event=self.failure_event,
            flight_target_s=self.flight,
            conditions=self.conditions_summary(),
            flight_time_range_s=[min(flight_times), max(flight_times)] if flight_times else None,
            mean_flight_time_s=float(np.mean(flight_times)) if flight_times else None,
            validated_flights=len(required & self.caught_events),
            guard=guard,
            guard_throws=max(0, self.throw_count - throws),
            time=float(self.data.time),
            min_distance=self.minimum_distance,
            min_surface_gap_m=(float(self.minimum_distance-2*self.model.geom_size[self.ball_geom_ids[0], 0])
                               if len(self.balls)>1 else None),
            min_height=self.minimum_height,
            ball_contacts=self.ball_contacts,
            drops=int(self.drop_event is not None or
                      (self.failure_event is not None and self.failure_event['reason'] == 'DROPPED')),
            airborne_physics_checks=self.airborne_physics_checks,
            style=self.style_summary(),
            execution=self.execution_metrics.summary(),
            grasp_assistance=False,
            grasp_constraints=0,
            contact_only_checks=self.contact_only_checks,
            max_joint_limit_violation=self.max_joint_limit_violation,
            max_height=self.max_height,
            max_palm_height=self.max_palm_height,
            minimum_shoulder_above_elbow=self.min_shoulder_above_elbow,
            maximum_self_penetration=self.max_self_penetration,
            min_upstroke=min((s['rise'] for s in self.stroke_metrics), default=0.0),
            min_upstroke_vertical_ratio=min((s['rise'] / max(0.001, s['sideways']) for s in self.stroke_metrics), default=0.0),
            min_throw_palm_up=self.min_throw_palm_up,
            min_throw_fingers_forward=self.min_throw_fingers_forward,
            palm_ranges=self.palm_ranges,
            stroke_metrics=self.stroke_metrics,
        )

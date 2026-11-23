"""Position-observed, contact-only single-ball reception for residual RL.

This is a reception subtask, not the complete cascade controller. Ball states
are initialized only in reset; actions change limited motor targets only.
"""
from dataclasses import asdict, dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from jugglebot.humanoid.contact_control import DualArmDynamics, SOURCE_POSITION_RIGHT_M
from jugglebot.humanoid.cascade import ContactCascade
from jugglebot.humanoid.model import BALL_SEAT_CENTER_OFFSET_M, DEFAULT_STAND_HEIGHT_M
from jugglebot.robustness.observation import BallObserver
from jugglebot.robustness.perception import PerceptionConfig
from .reception_control import ContactPositionTracker, ReceptionReference


UNSAFE_REASONS = frozenset({'FORBIDDEN_CONTACT', 'POSE_LIMIT', 'EXTERNAL_BALL_FORCE',
                            'BALL_TOO_HIGH', 'NONFINITE_STATE'})


@dataclass(frozen=True)
class CatchConfig:
    control_period_s: float = .02
    servo_period_s: float = .001
    reuse_forward_cache: bool = True
    tracking_measurement_std_m: float = .003
    tracking_process_noise: float = 1.
    horizon_s: float = .8
    stable_duration_s: float = .2
    initial_height_m: float = .20
    initial_downward_speed_m_s: float = .3
    landing_spread_m: float = .05
    incoming_speed_spread_m_s: float = .12
    height_spread_m: float = .015
    downward_speed_spread_m_s: float = .10
    feedback_enabled: bool = True
    nominal_reach_m: float = .065
    nominal_speed_m_s: float = 1.
    nominal_acceleration_m_s2: float = 30.
    cushion_depth_m: float = .025
    cushion_duration_s: float = .12
    closure_duration_s: float = .04
    grip_bias_rad: float = 0.
    max_offset_m: float = .015
    max_offset_speed_m_s: float = .15
    max_grip_offset_rad: float = .15
    noise_std_m: float = 0.
    delay_s: float = 0.
    dropout: float = 0.
    success_reward: float = 10.
    failure_reward: float = -10.
    unsafe_failure_reward: float = -20.
    shaping_weight: float = 0.
    action_cost_per_s: float = .05
    action_rate_cost_per_s: float = .02
    gamma: float = .99

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError('Configuration must be finite')
        for key in ('control_period_s', 'servo_period_s', 'tracking_measurement_std_m',
                    'tracking_process_noise', 'horizon_s', 'stable_duration_s',
                    'initial_height_m', 'max_offset_m', 'max_offset_speed_m_s',
                    'max_grip_offset_rad', 'nominal_reach_m', 'nominal_speed_m_s',
                    'nominal_acceleration_m_s2', 'cushion_duration_s', 'closure_duration_s'):
            if values[key] <= 0:
                raise ValueError(f'{key} must be positive')
        if (self.horizon_s <= self.stable_duration_s or self.initial_downward_speed_m_s < 0
                or self.landing_spread_m < 0 or not 0 < self.gamma < 1
                or min(self.incoming_speed_spread_m_s, self.height_spread_m,
                       self.downward_speed_spread_m_s, self.cushion_depth_m) < 0
                or self.height_spread_m >= self.initial_height_m
                or self.downward_speed_spread_m_s > self.initial_downward_speed_m_s
                or self.success_reward <= 0 or self.failure_reward >= 0
                or self.unsafe_failure_reward > self.failure_reward
                or min(self.shaping_weight, self.action_cost_per_s, self.action_rate_cost_per_s) < 0):
            raise ValueError('Invalid task or reward configuration')
        PerceptionConfig(self.noise_std_m, self.delay_s, self.dropout)


@dataclass(frozen=True)
class CatchEvidence:
    contact: bool
    relative_speed_m_s: float
    tangent_m: float
    normal_m: float
    ball_height_m: float
    forbidden_contact: bool = False
    pose_violation: bool = False
    external_ball_force: bool = False


class CatchAcceptance:
    """Reward-independent task monitor, checked at every physical step."""
    def __init__(self, stable_duration_s=.2):
        self.required = stable_duration_s
        self.stable_s = 0.
        self.done = False
        self.reason = None

    def update(self, evidence, dt, time_s):
        if self.done:
            return False, self.reason
        reason = None
        if not np.all(np.isfinite([evidence.relative_speed_m_s, evidence.tangent_m,
                                    evidence.normal_m, evidence.ball_height_m])):
            reason = 'NONFINITE_STATE'
        elif evidence.external_ball_force:
            reason = 'EXTERNAL_BALL_FORCE'
        elif evidence.forbidden_contact:
            reason = 'FORBIDDEN_CONTACT'
        elif evidence.pose_violation:
            reason = 'POSE_LIMIT'
        elif evidence.ball_height_m < .4:
            reason = 'BALL_DROPPED'
        elif evidence.ball_height_m >= 1.3:
            reason = 'BALL_TOO_HIGH'
        stable = (evidence.contact and evidence.relative_speed_m_s < .15
                  and evidence.tangent_m < .04 and .01 < evidence.normal_m < .06)
        self.stable_s = self.stable_s + dt if stable else 0.
        if reason is None and self.stable_s + 1e-10 >= self.required:
            reason = 'SUCCESS'
        # No new ballistic excursion after the incoming catch window.
        if reason is None and time_s > .4 and not evidence.contact and evidence.normal_m > .12:
            reason = 'MISSED_CATCH_WINDOW'
        if reason:
            self.done, self.reason = True, reason
        return reason == 'SUCCESS', reason


def transition_reward(config, previous_phi, phi, elapsed_s, action, previous_action,
                      terminal_reason=None):
    """Single event reward, bounded shaping and time-integrated control costs.

    The horizon is a task deadline (terminal), not an artificial truncation.
    Shaping terminal potential is zero for every outcome.
    """
    terminal = terminal_reason is not None
    failure = (config.unsafe_failure_reward if terminal_reason in UNSAFE_REASONS
               else config.failure_reward)
    parts = dict(event=(config.success_reward if terminal_reason == 'SUCCESS'
                        else failure if terminal else 0.),
                 shaping=config.shaping_weight * (
                     config.gamma * (0. if terminal else phi) - previous_phi),
                 action=-config.action_cost_per_s * elapsed_s * float(np.mean(action**2)),
                 rate=-config.action_rate_cost_per_s * elapsed_s
                      * float(np.mean(((action-previous_action)/.2)**2)))
    return float(sum(parts.values())), parts


class _CatchDynamics(DualArmDynamics):
    def elbow_out_limit(self, side):
        return .10

    def servo(self, hand, data, q, dq=None, ddq=None):
        if dq is not None or ddq is not None:
            return super().servo(hand, data, q, dq, ddq)
        # At rest M @ ddq is exactly zero; avoid expanding the full mass
        # matrix for the idle arm. Preserve the existing force protection.
        dof, acts = hand['kin'].dof_adr, hand['acts']
        kp = self.model.actuator_gainprm[acts, 0]
        kv = -self.model.actuator_biasprm[acts, 2]
        feed = data.qfrc_bias[dof]-data.qfrc_passive[dof]
        torque = kp*(q-data.qpos[hand['kin'].qpos_adr])+kv*(-data.qvel[dof])+feed
        torque = self.protect_arm(hand, data, torque)
        data.ctrl[acts] = data.qpos[hand['kin'].qpos_adr]+(torque+kv*data.qvel[dof])/kp


class ResidualCatchEnv(gym.Env):
    """Four residuals: palm XYZ offset and one finger flexion synergy.

    Observation: estimated relative ball position/velocity, validity/age,
    measured arm q/dq, finger q/dq, palm velocity, contact, time, applied action.
    No true ball velocity, landing target, RNG seed or future trajectory is
    included. Nominal interception uses the same position-only fitted state.
    """
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 50}
    ENVIRONMENT_VERSION = 6
    OBSERVATION_SIZE = 73
    TIME_INDEX = 58

    def __init__(self, config=None, render_mode=None):
        self.config = config or CatchConfig()
        self.render_mode = render_mode
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('Unsupported render mode')
        self.robot = _CatchDynamics(height=float(SOURCE_POSITION_RIGHT_M[2]))
        self.model, self.data = self.robot.model, self.robot.data
        self.hand = self.robot.hands['left']
        self.bq, self.bv = self.robot.bq, self.robot.bv
        self.ball_geom = self.model.geom('ball0').id
        self.ball_body = int(self.model.geom_bodyid[self.ball_geom])
        self.hand_geoms = {g for g in range(self.model.ngeom)
                           if self.model.body(self.model.geom_bodyid[g]).name.startswith('lh_')}
        self.palm_geoms = {g for g in self.hand_geoms
                           if self.model.body(self.model.geom_bodyid[g]).name == 'lh_palm'}
        self.robot.balls = [(self.bq, self.bv)]
        self.robot.torso_stand_height = DEFAULT_STAND_HEIGHT_M
        self.robot.failure_event = None
        self.physics_dt = float(self.model.opt.timestep)
        self.substeps = round(self.config.control_period_s/self.physics_dt)
        if self.substeps < 1 or not np.isclose(self.substeps*self.physics_dt, self.config.control_period_s):
            raise ValueError('Control period must be an integer number of physics steps')
        self.servo_substeps = round(self.config.servo_period_s/self.physics_dt)
        if (self.servo_substeps < 1 or self.substeps % self.servo_substeps
                or not np.isclose(self.servo_substeps*self.physics_dt, self.config.servo_period_s)):
            raise ValueError('Servo period must divide the control period on physics ticks')
        self.action_space = spaces.Box(-1., 1., (4,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.OBSERVATION_SIZE,), np.float32)
        self._renderer = None
        self._initial_qpos = self.data.qpos.copy()
        self._initial_ctrl = self.data.ctrl.copy()
        self._active = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('Reset options are not supported')
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._initial_qpos
        self.data.ctrl[:] = self._initial_ctrl
        self.hand['fingers'].set_posture_qpos(self.data, 'catch')
        mujoco.mj_forward(self.model, self.data)
        self.nominal_palm = self.data.site_xpos[self.hand['seat']].copy()
        self.nominal_q = self.data.qpos[self.hand['kin'].qpos_adr].copy()
        landing = self.nominal_palm + [0., 0., BALL_SEAT_CENTER_OFFSET_M]
        landing[:2] += self.np_random.uniform(-self.config.landing_spread_m,
                                            self.config.landing_spread_m, 2)
        height = self.config.initial_height_m+self.np_random.uniform(-self.config.height_spread_m,
                                                                    self.config.height_spread_m)
        down = self.config.initial_downward_speed_m_s+self.np_random.uniform(
            -self.config.downward_speed_spread_m_s, self.config.downward_speed_spread_m_s)
        lateral = self.np_random.uniform(-self.config.incoming_speed_spread_m_s,
                                         self.config.incoming_speed_spread_m_s, 2)
        flight = (-down+np.sqrt(down**2+2*self.robot.gravity_m_s2*height))/self.robot.gravity_m_s2
        self.data.qpos[self.bq:self.bq+3] = landing + np.r_[-lateral*flight, height]
        self.data.qvel[self.bv:self.bv+6] = np.r_[lateral, -down, np.zeros(3)]
        mujoco.mj_forward(self.model, self.data)
        noise_seed = int(self.np_random.integers(0, 2**31))
        self.observer = BallObserver([(self.bq, self.bv)], mode='sampled',
                                    position_step_s=self.physics_dt,
                                    perception=PerceptionConfig(self.config.noise_std_m,
                                        self.config.delay_s, self.config.dropout, noise_seed))
        self.observer.update(self.data)
        self.tracker = ContactPositionTracker(self.model.opt.gravity, self.physics_dt,
            measurement_std_m=self.config.tracking_measurement_std_m,
            process_noise=self.config.tracking_process_noise)
        self._update_tracker()
        self.reference = ReceptionReference(self.nominal_palm, self.robot.gravity_m_s2, self.config)
        self.acceptance = CatchAcceptance(self.config.stable_duration_s)
        self.action = np.zeros(4)
        self.touch_time = None
        self.first_hand_contact_s = None
        self.palm_contact = False
        self.servo_ticks = self.servo_updates = self.post_contact_estimates = 0
        self.residual_active_steps = 0
        self.position_residual_active_steps = 0
        self.tracking_squared_error = np.zeros(2)
        self.elapsed_steps = 0
        self.return_sum = 0.
        self.reward_sums = dict(event=0., shaping=0., action=0., rate=0.)
        self.last_evidence = None
        self.robot.failure_event = None
        self.initial_ball_height = float(self.data.qpos[self.bq+2])
        self.max_ball_height = self.initial_ball_height
        self.max_speed = self.max_offset = self.max_force = 0.
        self._active = True
        self.phi = self._potential()
        return self._observation(), self._info(None)

    def _palm_velocity(self):
        value = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                self.hand['seat'], value, 0)
        return value[3:]

    def _potential(self):
        # Truth is used for reward and acceptance only, never for policy input.
        error = self.data.qpos[self.bq:self.bq+2] - self.data.site_xpos[self.hand['seat'], :2]
        return float(np.exp(-np.dot(error, error)/.03**2))

    def _observation(self):
        estimate = self.tracker.estimate(float(self.data.time))
        valid = estimate is not None and estimate.valid
        palm_v = self._palm_velocity()
        position = np.zeros(3) if not valid else np.asarray(estimate.position_m)-self.data.site_xpos[self.hand['seat']]
        velocity = np.zeros(3) if not valid else np.asarray(estimate.velocity_m_s)-palm_v
        age = .1 if estimate is None else min(.1, estimate.age_s)
        fingers = self.hand['fingers']
        joint_ids = fingers.joint_ids
        values = np.r_[np.clip(position, -2, 2), np.clip(velocity/5., -10, 10),
                       float(valid), age/.05,
                       self.data.qpos[self.hand['kin'].qpos_adr],
                       self.data.qvel[self.hand['kin'].dof_adr]/10.,
                       self.data.qpos[self.model.jnt_qposadr[joint_ids]],
                       self.data.qvel[self.model.jnt_dofadr[joint_ids]]/10.,
                       palm_v, float(self._residual_permitted()),
                       self.data.time/self.config.horizon_s, self.action,
                       self.reference.sample(float(self.data.time))[0]-self.data.site_xpos[self.hand['seat']],
                       self.reference.sample(float(self.data.time))[1],
                       np.clip(self.tracker.uncertainty(float(self.data.time))/[.01, 1.], 0., 10.),
                       float(self._grip_closing_permitted()), float(self._position_residual_permitted())]
        return values.astype(np.float32)

    def _evidence(self):
        contact = forbidden = palm = False
        for i, c in enumerate(self.data.contact):
            other = c.geom2 if c.geom1 == self.ball_geom else c.geom1 if c.geom2 == self.ball_geom else -1
            if other < 0 or c.efc_address < 0 or c.dist > 1e-5:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            self.max_force = max(self.max_force, float(force[0]))
            forbidden |= other not in self.hand_geoms
            contact |= other in self.hand_geoms and force[0] > .02
            palm |= other in self.palm_geoms and force[0] > .02
        rotation = self.data.site_xmat[self.hand['seat']].reshape(3, 3)
        center = rotation.T @ (self.data.qpos[self.bq:self.bq+3]-self.data.site_xpos[self.hand['seat']])
        speed = float(np.linalg.norm(self.data.qvel[self.bv:self.bv+3]-self._palm_velocity()))
        pose = False
        try:
            ContactCascade.check_pose_limits(self.robot)
        except RuntimeError:
            pose = True
        external = (np.any(self.data.xfrc_applied[self.ball_body])
                    or np.any(self.data.qfrc_applied[self.bv:self.bv+6]) or np.any(self.data.eq_active))
        self.palm_contact = bool(palm)
        if contact and self.first_hand_contact_s is None:
            self.first_hand_contact_s = float(self.data.time)
        if palm and self.touch_time is None:
            self.touch_time = float(self.data.time)
            self.reference.capture(self.touch_time, self.data.site_xpos[self.hand['seat']],
                                    self._palm_velocity(),
                                    residual_offset=self.action[:3]*self.config.max_offset_m)
        self.max_speed = max(self.max_speed, speed)
        self.max_ball_height = max(self.max_ball_height, float(self.data.qpos[self.bq+2]))
        return CatchEvidence(bool(contact), speed, float(np.linalg.norm(center[:2])), float(center[2]),
                             float(self.data.qpos[self.bq+2]), bool(forbidden), bool(pose), bool(external))

    def _control(self):
        self.servo_ticks += 1
        if (self.servo_ticks-1) % self.servo_substeps:
            return
        self.servo_updates += 1
        for side, hand in self.robot.hands.items():
            target = hand['q'].copy()
            if side == 'left':
                position, velocity, acceleration = self.reference.sample(float(self.data.time))
                self.robot.return_servo(hand, self.data, self.nominal_q,
                    position=position+self.action[:3]*self.config.max_offset_m,
                    velocity=velocity, acceleration=acceleration, axis_goal=np.array([0., 0., 1.]))
                fingers = hand['fingers']
                phase = 0. if self.touch_time is None else np.clip(
                    (self.data.time-self.touch_time)/self.config.closure_duration_s, 0., 1.)
                blend = phase**3*(10-15*phase+6*phase**2)
                ctrl = ((1-blend)*fingers.get_ctrl_array('catch')
                        + blend*fingers.get_ctrl_array('cradle'))
                ctrl += (self.config.grip_bias_rad+self.action[3]*self.config.max_grip_offset_rad) * fingers.grip_map.sum(axis=1)
                bounds = self.model.actuator_ctrlrange[fingers.act_indices]
                self.data.ctrl[fingers.act_indices] = np.clip(ctrl, bounds[:, 0], bounds[:, 1])
            else:
                self.robot.servo(hand, self.data, target)

    def _update_tracker(self):
        if self.observer.observations:
            self.tracker.update(self.observer.observations[0])

    def _residual_permitted(self):
        return self.last_evidence is not None and self.last_evidence.contact

    def _grip_closing_permitted(self):
        return self.palm_contact and self._residual_permitted()

    def _position_residual_permitted(self):
        # Finger contact can relax the funnel. Additional closing requires
        # palm support; translation waits for the wrapping phase to finish.
        if (self.touch_time is None or not self._grip_closing_permitted()
                or self.data.time-self.touch_time < self.config.closure_duration_s):
            return False
        now = float(self.data.time)
        return (self.tracker.estimate(now) is not None
                and self.tracker.uncertainty(now)[0] <= self.config.max_offset_m)

    def step(self, action):
        if not self._active:
            raise RuntimeError('reset is required before stepping or after termination')
        request = np.asarray(action, dtype=float)
        if request.shape != (4,) or not np.all(np.isfinite(request)):
            raise ValueError('Expected four finite residual actions')
        request = request.copy()
        if not self._residual_permitted():
            request[:] = 0.
        else:
            self.residual_active_steps += 1
            if not self._grip_closing_permitted():
                request[3] = min(request[3], 0.)
            if not self._position_residual_permitted():
                request[:3] = 0.
            else:
                self.position_residual_active_steps += 1
        previous = self.action.copy()
        self.reference.advance(self.tracker.estimate(float(self.data.time)), float(self.data.time),
                               in_contact=self._residual_permitted())
        change = np.r_[np.full(3, self.config.max_offset_speed_m_s/self.config.max_offset_m), 5.]
        self.action += np.clip(np.clip(request, -1, 1)-self.action,
                               -change*self.config.control_period_s, change*self.config.control_period_s)
        self.max_offset = max(self.max_offset, float(np.max(np.abs(self.action[:3]))*self.config.max_offset_m))
        start = float(self.data.time)
        reason = None
        for _ in range(self.substeps):
            # reset() and the preceding tick end with mj_forward: position,
            # velocity, mass and contact caches already match this state.
            # Control writes ctrl only, so repeating mj_step1 is redundant.
            if not self.config.reuse_forward_cache:
                mujoco.mj_step1(self.model, self.data)
            self._control()
            mujoco.mj_step2(self.model, self.data)
            # Refresh geometry/velocity caches to the endpoint, without integration.
            mujoco.mj_forward(self.model, self.data)
            self.observer.update(self.data)
            if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                reason = 'NONFINITE_STATE'
                break
            self.last_evidence = self._evidence()
            if self.touch_time is not None:
                self.tracker.notify_contact(self.last_evidence.contact, float(self.data.time), self._palm_velocity())
            self._update_tracker()
            _, reason = self.acceptance.update(self.last_evidence, self.physics_dt, float(self.data.time))
            if reason:
                break
        if reason is None and self.data.time >= self.config.horizon_s-1e-10:
            reason = 'TASK_DEADLINE'
        self.elapsed_steps += 1
        if self.touch_time is not None:
            estimate = self.tracker.estimate(float(self.data.time))
            if estimate is not None and estimate.valid:
                self.post_contact_estimates += 1
                self.tracking_squared_error += [
                    np.sum((np.asarray(estimate.position_m)-self.data.qpos[self.bq:self.bq+3])**2),
                    np.sum((np.asarray(estimate.velocity_m_s)-self.data.qvel[self.bv:self.bv+3])**2)]
        phi = self._potential()
        reward, parts = transition_reward(self.config, self.phi, phi, self.data.time-start,
                                           self.action, previous, reason)
        self.phi = phi
        self.return_sum += reward
        for key, value in parts.items():
            self.reward_sums[key] += value
        terminated = reason is not None
        self._active = not terminated
        info = self._info(reason)
        info['reward_terms'] = parts
        return self._observation(), reward, terminated, False, info

    def _info(self, reason):
        return dict(is_success=reason == 'SUCCESS', reason=reason, simulated_time_s=float(self.data.time),
                    environment_version=self.ENVIRONMENT_VERSION,
                    servo_updates=self.servo_updates,
                    residual_active_steps=self.residual_active_steps,
                    position_residual_active_steps=self.position_residual_active_steps,
                    first_hand_contact_s=self.first_hand_contact_s,
                    first_palm_contact_s=self.touch_time,
                    post_contact_estimates=self.post_contact_estimates,
                    contact_measurement_updates=self.tracker.contact_updates,
                    post_contact_position_rmse_m=float(np.sqrt(self.tracking_squared_error[0]/max(1, self.post_contact_estimates))),
                    post_contact_velocity_rmse_m_s=float(np.sqrt(self.tracking_squared_error[1]/max(1, self.post_contact_estimates))),
                    nominal_feedback=self.config.feedback_enabled,
                    nominal_intercept_updates=self.reference.updates,
                    rejected_nominal_updates=self.reference.rejected_updates,
                    stable_contact_s=self.acceptance.stable_s, episode_return=self.return_sum,
                    reward_sums=dict(self.reward_sums), max_ball_height_m=self.max_ball_height,
                    final_evidence=None if self.last_evidence is None else asdict(self.last_evidence),
                    max_relative_speed_m_s=self.max_speed, max_residual_offset_m=self.max_offset,
                    peak_contact_force_n=self.max_force)

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera='front')
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

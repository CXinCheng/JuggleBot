"""Bounded arm dynamics and smooth references for the G1 juggling controller."""
import mujoco
import numpy as np
from .planning import Quintic
from .model import load_g1_dual_arm, ARM_JOINTS_RIGHT, ARM_JOINTS_LEFT
from .model import DEFAULT_STAND_HEIGHT_M, BALL_SEAT_CENTER_OFFSET_M
from .kinematics import HumanoidArmKinematics, IKResult7DOF
from .throwing import LeapFingers, get_juggling_postures
# Numerical IK branch seed only. The commanded ready posture is solved for
# the full-size hands and their adapter transforms; these are not motor targets.
PREFERRED_THROW_RIGHT = np.array([-0.32875641264113137, -0.14, 0.06106186896137782, -0.6476226875603018, 0.13268647750065796, 0.9556197341724287, -0.4624240963309207])
EMPTY_RETURN_TILT_DEG = 5.0
WAIST_YAW_STIFFNESS = 1800.0
ELBOW_LATERAL_POSTURE_GAIN = 8.0
WRIST_BRAKE_ONSET_RAD = np.deg2rad(32.)
WRIST_BRAKE_BOUND_RAD = np.deg2rad(36.)
WRIST_THROW_LOOKAHEAD_S = .02
# Inner releases leave a separate lane for the descending ball.
SOURCE_POSITION_RIGHT_M = np.array([.33, -.14, .82])
CATCH_POSITION_RIGHT_M = np.array([.30, -.22, .86])
SCOOP_END_ANGLE_RAD = .7
SCOOP_ACCELERATION_DURATION_S = .14
SCOOP_BRAKING_DURATION_S = .10
MIN_SCOOP_REFERENCE_RISE_M = .09
THROW_PALM_BACK_TILT = .18
THROW_LATERAL_PALM_LEAN = .20

def _cross3(a, b):
    """Three-vector cross product without NumPy's broadcasting dispatch."""
    result = np.empty(3)
    mujoco.mju_cross(result, a, b)
    return result


def bounded_lstsq(matrix, desired, limits):
    """Small active-set least squares with exact box constraints."""
    x = np.zeros(len(limits))
    active = np.zeros(len(limits), dtype=int)
    for _ in range(50):
        free = active == 0
        candidate = x.copy()
        if np.any(free):
            candidate[free] = np.linalg.lstsq(matrix[:, free], desired - matrix[:, ~free] @ x[~free], rcond=1e-09)[0]
        direction = candidate - x
        violated = free & (np.abs(candidate) > limits + 1e-09)
        if np.any(violated):
            fractions = np.full(len(x), np.inf)
            fractions[violated] = (np.sign(direction[violated]) * limits[violated] - x[violated]) / direction[violated]
            hit = int(np.argmin(fractions))
            x += np.clip(fractions[hit], 0, 1) * direction
            active[hit] = 1 if direction[hit] > 0 else -1
            x[hit] = active[hit] * limits[hit]
            continue
        x = candidate
        gradient = matrix.T @ (matrix @ x - desired)
        violation = active * gradient
        if np.max(violation) < 1e-08:
            return np.clip(x, -limits, limits)
        active[int(np.argmax(violation))] = 0
    return np.clip(x, -limits, limits)

class UprightKinematics(HumanoidArmKinematics):
    """Solve a complete palm orientation with bounded reference joint angles."""

    def solve(self, target_pos, target_axis, q_init, q_nominal=None, max_iters=100):
        q = np.clip(np.asarray(q_init, dtype=float), self.q_min, self.q_max)
        nominal = q if q_nominal is None else np.asarray(q_nominal)
        normal = np.asarray(target_axis) / np.linalg.norm(target_axis)
        forward = np.array([1.0, 0.0, 0.0])
        forward -= normal * (normal @ forward)
        forward /= np.linalg.norm(forward)
        rotation = np.column_stack([np.cross(forward, normal), forward, normal])
        override = getattr(self, 'target_rotation', None)
        if override is not None:
            rotation = override.copy()
        for iteration in range(max_iters):
            position, axis = self.fk(q)
            actual = self.data.site_xmat[self.site_id].reshape(3, 3)
            angular = 0.5 * sum((np.cross(actual[:, i], rotation[:, i]) for i in range(3)), np.zeros(3))
            error = np.r_[np.asarray(target_pos) - position, angular]
            pe = float(np.linalg.norm(error[:3]))
            ae = float(np.linalg.norm(angular))
            if pe < 2e-05 and ae < 0.0002:
                return IKResult7DOF(q, True, pe, ae, iteration)
            jp, jr = self.jacobian(q)
            jac = np.vstack([jp, jr])
            null = np.eye(7) - np.linalg.pinv(jac, rcond=0.0001) @ jac
            preference = 0.03 * null @ (nominal - q)
            matrix = np.vstack([jac, 0.008 * np.eye(7)])
            lower = np.maximum(self.q_min - q, -0.15)
            upper = np.minimum(self.q_max - q, 0.15)
            center = (lower + upper) / 2
            limits = (upper - lower) / 2
            delta = center + bounded_lstsq(matrix, np.r_[error, 0.008 * preference] - matrix @ center, limits)
            q = np.clip(q + delta, self.q_min, self.q_max)
        return IKResult7DOF(q, False, pe, ae, max_iters)

def exp_rotation(vector):
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew

def log_rotation(rotation):
    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))
    vector = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    return 0.5 * vector if angle < 1e-06 else angle / (2 * np.sin(angle)) * vector

def left_jacobian(vector):
    angle = np.linalg.norm(vector)
    x, y, z = vector
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    if angle < 1e-05:
        return np.eye(3) + 0.5 * skew + skew @ skew / 6
    return np.eye(3) + (1 - np.cos(angle)) / angle ** 2 * skew + (angle - np.sin(angle)) / angle ** 3 * skew @ skew

class ScoopCurve:
    def __init__(self, duration, start, initial_velocity, velocity, initial_acceleration=None):
        self.start=np.asarray(start).copy()
        initial_acceleration = np.zeros(3) if initial_acceleration is None else np.asarray(initial_acceleration)
        self.angle=SCOOP_END_ANGLE_RAD
        self.b=max(MIN_SCOOP_REFERENCE_RISE_M/(1-np.cos(self.angle)),abs(velocity[2])*duration/(2.25*self.angle*np.sin(self.angle)))
        self.a=self.b*velocity[1]/max(1.,velocity[2])*np.tan(self.angle)
        rate=velocity[2]/(self.b*np.sin(self.angle))
        # Cancel the ellipse curvature's vertical acceleration at release.
        # Zero phase acceleration alone still accelerates the palm upward.
        phase_acceleration=-np.cos(self.angle)/np.sin(self.angle)*rate*rate
        self.phase=Quintic(0,duration,0,0,0,self.angle,rate,phase_acceleration)
        self.x=Quintic(0,duration,start[0],initial_velocity[0],initial_acceleration[0],start[0]+.08*velocity[0],velocity[0],0)
        self.correction=Quintic(0,duration,np.zeros(3),np.r_[0.,initial_velocity[1:]],np.r_[0.,initial_acceleration[1:]],np.zeros(3),np.zeros(3),np.zeros(3))

    def sample(self,t):
        s,ds,dds=self.phase.sample(t)
        p=self.start+np.array([0.,self.a*np.sin(s),self.b*(1-np.cos(s))])
        v=np.array([0.,self.a*np.cos(s)*ds,self.b*np.sin(s)*ds])
        a=np.array([0.,self.a*(np.cos(s)*dds-np.sin(s)*ds*ds),self.b*(np.sin(s)*dds+np.cos(s)*ds*ds)])
        xp,xv,xa=self.x.sample(t);p[0],v[0],a[0]=xp,xv,xa
        cp,cv,ca=self.correction.sample(t)
        return p+cp,v+cv,a+ca

class ScoopPulse:
    """Accelerate along the ellipse, then brake the open palm to let contact separate."""
    def __init__(self, start, initial_velocity, velocity, initial_acceleration=None):
        self.launch=ScoopCurve(SCOOP_ACCELERATION_DURATION_S,start,initial_velocity,velocity,initial_acceleration)
        p,v,a=self.launch.sample(SCOOP_ACCELERATION_DURATION_S)
        self.brake=Quintic(SCOOP_ACCELERATION_DURATION_S,SCOOP_BRAKING_DURATION_S,p,v,a,p+.035*v,np.zeros(3),np.zeros(3))

    def sample(self,t):
        return self.launch.sample(t) if t<=SCOOP_ACCELERATION_DURATION_S else self.brake.sample(t)


class CartesianPalmMotion:
    """Smooth palm translation and orientation, tracked through bounded arm dynamics."""

    def __init__(self, kin, t0, duration, start, velocity, q0, goal, end_velocity, axis=None, nominal=None, empty_arc=False, arc_peak=None, tilt_deg=None, cushion_duration=None):
        m = kin.model
        d = kin.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        jp = np.zeros((3, m.nv))
        jr = np.zeros_like(jp)
        jpd = np.zeros_like(jp)
        jrd = np.zeros_like(jp)
        mujoco.mj_jacSite(m, d, jp, jr, kin.site_id)
        mujoco.mj_jacDot(m, d, jpd, jrd, d.site_xpos[kin.site_id], int(m.site_bodyid[kin.site_id]))
        a0 = np.clip(jp @ d.qacc + jpd @ d.qvel, -50.0, 50.0)
        alpha0 = np.clip(jr @ d.qacc + jrd @ d.qvel, -200.0, 200.0)
        self.initial_linear_velocity = jp @ d.qvel
        self.initial_linear_acceleration = a0.copy()
        self.empty_arc = empty_arc
        self.has_tilt = empty_arc or tilt_deg is not None
        self.tilt_amplitude = np.deg2rad(EMPTY_RETURN_TILT_DEG if tilt_deg is None else tilt_deg)
        self.empty_tilt_segments = None
        if self.has_tilt and arc_peak is not None:
            rise = min(float(arc_peak), duration / 2)
            amplitude = self.tilt_amplitude
            self.empty_tilt_segments = (
                Quintic(t0, rise, 0., 0., 0., amplitude, 0., 0.),
                Quintic(t0 + rise, duration - rise, amplitude, 0., 0., 0., 0., 0.),
            )
        self.t0 = t0
        self.duration = duration
        self.t1 = self.end = t0 + duration
        self.cartesian = Quintic(t0, duration, start, velocity, a0, goal, end_velocity, np.zeros(3))
        self.capture_cushion = None
        if cushion_duration is not None:
            stop_time = min(float(cushion_duration), .4*duration)
            stopped = np.asarray(start) + .5*stop_time*np.asarray(velocity)
            self.capture_cushion = (
                Quintic(t0, stop_time, start, velocity, a0, stopped, np.zeros(3), np.zeros(3)),
                Quintic(t0+stop_time, duration-stop_time, stopped, np.zeros(3), np.zeros(3), goal, end_velocity, np.zeros(3)),
            )
        self.initial_rotation = d.site_xmat[kin.site_id].reshape(3, 3).copy()
        normal = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis).copy()
        normal /= np.linalg.norm(normal)
        forward = np.array([1.0, 0.0, 0.0])
        forward -= normal * (normal @ forward)
        forward /= np.linalg.norm(forward)
        goal_rotation = np.column_stack([np.cross(forward, normal), forward, normal])
        self.angular = Quintic(t0, duration, np.zeros(3), jr @ d.qvel, alpha0, log_rotation(goal_rotation @ self.initial_rotation.T), np.zeros(3), np.zeros(3))

    def sample(self, t):
        if self.capture_cushion is not None:
            stop, recover = self.capture_cushion
            return (stop if t <= stop.t1 else recover).sample(t)
        return self.cartesian.sample(t)

    def orientation(self, t):
        vector, speed, accel = self.angular.sample(t)
        jac = left_jacobian(vector)
        epsilon = 0.0001
        derivative = (left_jacobian(vector + epsilon * speed) - left_jacobian(vector - epsilon * speed)) / (2 * epsilon)
        rotation = exp_rotation(vector) @ self.initial_rotation
        omega = jac @ speed
        alpha = jac @ accel + derivative @ speed
        if self.has_tilt:
            # A pitch pulse follows the forearm during empty return or loaded
            # recovery. Its first two derivatives vanish at both boundaries.
            s = np.clip((t-self.t0)/self.duration, 0., 1.)
            amplitude = self.tilt_amplitude
            bump = amplitude * 64*s**3*(1-s)**3
            db = amplitude * 192*s*s*(1-s)**2*(1-2*s)/self.duration
            ddb = amplitude * 384*s*(1-s)*(1-5*s+5*s*s)/self.duration**2
            if self.empty_tilt_segments is not None:
                rise, fall = self.empty_tilt_segments
                bump, db, ddb = (rise if t <= rise.t1 else fall).sample(t)
            tilt = exp_rotation(np.array([0., -bump, 0.]))
            dw = np.array([0., -db, 0.])
            alpha = tilt@alpha + np.cross(dw, tilt@omega) + [0., -ddb, 0.]
            omega = dw + tilt@omega
            rotation = tilt@rotation
        return rotation, omega, alpha

class ThrowPath(CartesianPalmMotion):
    """Continuous elliptical scoop tracked directly as a Cartesian task."""

    def __init__(self, kin, start, velocity, q0, catch, bank=0.0, initial_axis=None, catch_q=None):
        duration = .25
        velocity = np.asarray(velocity)
        jp, _ = kin.jacobian(q0)
        initial_velocity = jp @ kin.data.qvel[kin.dof_adr]
        # Historical `bank` shifts the fore/aft normal, not the lateral lean.
        # The latter mirrors with the throw direction.
        normal = np.array([bank - THROW_PALM_BACK_TILT,
                           THROW_LATERAL_PALM_LEAN * np.sign(velocity[1]), 1.])
        normal /= np.linalg.norm(normal)
        curve = ScoopPulse(start, initial_velocity, velocity)
        super().__init__(kin, 0., duration, start, initial_velocity, q0,
                         curve.sample(duration)[0], np.zeros(3), axis=normal, nominal=q0)
        self.initial_q = np.asarray(q0).copy()
        # Continue measured palm velocity and bounded acceleration, including
        # waist reaction and the preceding catch/recovery.
        # The correction vanishes at release, retaining the scoop's tangent.
        self.cartesian = ScoopPulse(start, self.initial_linear_velocity, velocity,
                                    self.initial_linear_acceleration)
        if catch_q is None:
            result = kin.solve(catch, np.array([0., 0., 1.]), q0, q_nominal=q0)
            if not result.success:
                raise ValueError(f'unreachable catch posture: {result}')
            self.catch_q = result.q.copy()
        else:
            self.catch_q = np.asarray(catch_q).copy()


class SmoothStop:
    """Follow the incoming ball downward and decelerate to rest without endpoint jerk."""

    def __init__(self, t0, duration, start, velocity, gravity_m_s2):
        self.t0 = t0
        self.duration = duration
        self.t1 = t0 + duration
        self.start = np.asarray(start)
        self.velocity = np.asarray(velocity)
        self.gravity_m_s2 = gravity_m_s2

    def sample(self, t):
        s = np.clip((t - self.t0) / self.duration, 0, 1)
        integral = 2.5 * s ** 4 - 3 * s ** 5 + s ** 6
        smooth = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        derivative = 30 * s * s * (1 - s) ** 2
        return (self.start + self.velocity * self.duration * (s - integral), self.velocity * (1 - smooth), -self.velocity / self.duration * derivative)

    def support(self, t):

        def axis(tt):
            a = self.sample(tt)[2]
            n = np.r_[a[:2], self.gravity_m_s2 + a[2]]
            return n / np.linalg.norm(n)
        step = 0.002
        n = axis(t)
        nm = axis(t - step)
        np1 = axis(t + step)
        return (n, np.cross(n, (np1 - nm) / (2 * step)), np.cross(n, (np1 - 2 * n + nm) / step ** 2))

class DualArmDynamics:
    """Shared G1 arm dynamics. Torque commands pass through the existing position actuators and retain their joint force limits."""

    def __init__(self, side='right', height=0.8, n_balls=1, torso_stand_height=DEFAULT_STAND_HEIGHT_M):
        self.model = load_g1_dual_arm(n_balls=n_balls, torso_stand_height=torso_stand_height)
        m = self.model
        # Ballistic prediction assumes vertical downward gravity; read its magnitude from the model.
        if np.any(m.opt.gravity[:2] != 0) or m.opt.gravity[2] >= 0:
            raise ValueError('contact juggling requires vertical downward model gravity')
        self.gravity_m_s2 = -float(m.opt.gravity[2])
        offset = torso_stand_height - DEFAULT_STAND_HEIGHT_M
        self.data = mujoco.MjData(m)
        self.side = side
        self.hands = {}
        for which, names in [('right', ARM_JOINTS_RIGHT), ('left', ARM_JOINTS_LEFT)]:
            kin = UprightKinematics(m, which, tcp_site=('r_' if which == 'right' else 'l_') + 'ball_seat')
            sign = 1 if which == 'right' else -1
            lower = np.array([-1.1, -0.65, -1.2, -1.02, -1.2, -0.7, -0.9])
            upper = np.array([0.8, -0.03, 1.2, 1.2, 1.2, 1.5, 0.9])
            if which == 'left':
                mirror = np.array([1, -1, -1, 1, -1, 1, -1])
                lower, upper = (np.minimum(lower * mirror, upper * mirror), np.maximum(lower * mirror, upper * mirror))
            kin.q_min = np.maximum(kin.q_min, lower)
            kin.q_max = np.minimum(kin.q_max, upper)
            p = (SOURCE_POSITION_RIGHT_M if which == side else CATCH_POSITION_RIGHT_M).copy()
            p[1] *= sign
            p[2] = height if which == side else CATCH_POSITION_RIGHT_M[2] + offset
            seed = PREFERRED_THROW_RIGHT.copy()
            if which == 'left':
                seed *= np.array([1, -1, -1, 1, -1, 1, -1])
            result = kin.solve(p, np.array([0.0, 0.0, 1.0]), seed, q_nominal=seed)
            if not result.success:
                raise ValueError(f'bad initial IK {which}: {result}')
            acts = np.array([m.actuator(n).id for n in names])
            m.actuator_gainprm[acts, 0] = 1500
            m.actuator_biasprm[acts, 1] = -1500
            m.actuator_biasprm[acts, 2] *= np.sqrt(3)
            fingers = LeapFingers(m, which, postures=get_juggling_postures(which))
            # Keep LEAP motor gains identical to v5 and the upstream model.
            # G1 arm tuning above does not alter the shared hand hardware.
            self.data.qpos[kin.qpos_adr] = result.q
            self.data.ctrl[acts] = result.q
            fingers.set_posture_qpos(self.data, 'cradle')
            catch = CATCH_POSITION_RIGHT_M.copy()
            catch[1] *= sign
            catch[2] += offset
            self.hands[which] = dict(kin=kin, acts=acts, fingers=fingers, p=p, q=result.q, seat=kin.site_id, catch_p=catch)
            self.hands[which]['force_limits'] = m.jnt_actfrcrange[[m.joint(n).id for n in names], 1].copy()
        for name in ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']:
            ai = m.actuator(name).id
            m.actuator_gainprm[ai, 0] = 2000
            m.actuator_biasprm[ai, 1] = -2000
            m.actuator_biasprm[ai, 2] *= 2
        # Damped reaction to arm momentum; no prescribed torso animation.
        for name, stiffness in [('waist_yaw_joint', WAIST_YAW_STIFFNESS), ('waist_roll_joint', 1400.)]:
            actuator = m.actuator(name).id
            m.actuator_gainprm[actuator, 0] = stiffness
            m.actuator_biasprm[actuator, 1] = -stiffness
            m.actuator_biasprm[actuator, 2] *= np.sqrt(stiffness / 2000.)
        self.bq = m.joint('ball0_free').qposadr[0]
        self.bv = m.joint('ball0_free').dofadr[0]
        mujoco.mj_forward(m, self.data)
        hand = self.hands[side]
        self.data.qpos[self.bq:self.bq + 3] = self.data.site_xpos[hand['seat']] + [0, 0, BALL_SEAT_CENTER_OFFSET_M]
        self.data.qvel[self.bv:self.bv + 6] = 0
        mujoco.mj_forward(m, self.data)
        self.mass = np.zeros((m.nv, m.nv))

    def servo(self, hand, data, q, dq=None, ddq=None):
        """Track a joint position, velocity and acceleration reference with inverse-dynamics feed-forward."""
        dq = np.zeros(7) if dq is None else dq
        ddq = np.zeros(7) if ddq is None else ddq
        mujoco.mj_fullM(self.model, data, self.mass)
        dof = hand['kin'].dof_adr
        acts = hand['acts']
        kp = self.model.actuator_gainprm[acts, 0]
        kv = -self.model.actuator_biasprm[acts, 2]
        feed = self.mass[np.ix_(dof, dof)] @ ddq + data.qfrc_bias[dof] - data.qfrc_passive[dof]
        torque = kp * (q - data.qpos[hand['kin'].qpos_adr]) + kv * (dq - data.qvel[dof]) + feed
        torque = self.protect_arm(hand, data, torque)
        data.ctrl[acts] = data.qpos[hand['kin'].qpos_adr] + (torque + kv * data.qvel[dof]) / kp

    def return_servo(self, hand, data, target, position=None, axis_goal=None, velocity=None, acceleration=None, angular_velocity=None, angular_acceleration=None, balance=False, orientation=None):
        """Solve a bounded Cartesian acceleration task, prioritizing palm leveling over rotation about its normal."""
        m = self.model
        dof = hand['kin'].dof_adr
        acts = hand['acts']
        site = hand['seat']
        mujoco.mj_fullM(m, data, self.mass)
        rhs_mass = np.zeros((8, m.nv))
        rhs_mass[np.arange(7), dof] = 1.0
        rhs_mass[7] = data.qfrc_actuator + data.qfrc_passive - data.qfrc_bias + data.qfrc_constraint
        rhs_mass[7, dof] -= data.qfrc_actuator[dof]
        solved = np.zeros_like(rhs_mass)
        mujoco.mj_solveM(m, data, solved, rhs_mass)
        response = solved[:7].T
        free_acceleration = solved[7]
        limits = hand['force_limits']
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        jpd = np.zeros((3, m.nv))
        jrd = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, data, jp, jr, site)
        mujoco.mj_jacDot(m, data, jpd, jrd, data.site_xpos[site], int(m.site_bodyid[site]))
        axis = data.site_xmat[site].reshape(3, 3)[:, 2]
        omega = jr @ data.qvel
        axis_dot = _cross3(omega, axis)
        projection = np.eye(3) - np.outer(axis, axis)
        projection_dot = -np.outer(axis_dot, axis) - np.outer(axis, axis_dot)
        if position is None:
            mujoco.mj_copyData(hand['kin'].data, m, data)
            goal, _ = hand['kin'].fk(target)
        else:
            goal = np.asarray(position)
        normal = np.array([0., 0., 1.]) if axis_goal is None else np.asarray(axis_goal).copy()
        normal /= np.linalg.norm(normal)
        forward = np.array([1., 0., 0.])
        forward -= normal * (normal @ forward)
        forward /= np.linalg.norm(forward)
        goal_rotation = np.column_stack([_cross3(forward, normal), forward, normal]) if orientation is None else np.asarray(orientation).copy()
        position_error = goal - data.site_xpos[site]
        stopping = np.sqrt(2 * 10 * abs(position_error))
        desired_linear_velocity = np.clip(25 * position_error, -stopping, stopping)
        desired_linear_velocity = np.clip(desired_linear_velocity, -2.0, 2.0)
        linear = (desired_linear_velocity - jp @ data.qvel) / 0.015
        if velocity is not None:
            linear = np.asarray(acceleration) + 225 * position_error + 30 * (np.asarray(velocity) - jp @ data.qvel)
        goal_axis = np.array([0.0, 0.0, 1.0]) if axis_goal is None else axis_goal
        omega_goal = np.zeros(3) if angular_velocity is None else angular_velocity
        alpha_goal = np.zeros(3) if angular_acceleration is None else angular_acceleration
        angular_kp, angular_kd = (220, 30) if velocity is None else (600, 45)
        angular = alpha_goal + angular_kp * _cross3(axis, goal_axis) - angular_kd * (projection @ (omega - omega_goal))
        if not balance:
            base_axis = goal_rotation[:, 2]
            vector = _cross3(base_axis, goal_axis)
            skew = np.array([[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]])
            goal_rotation = (np.eye(3) + skew + skew @ skew / max(1e-06, 1 + base_axis @ goal_axis)) @ goal_rotation
            actual = data.site_xmat[site].reshape(3, 3)
            orientation_error = 0.5 * sum((_cross3(actual[:, i], goal_rotation[:, i]) for i in range(3)), np.zeros(3))
            projection = np.eye(3)
            projection_dot = np.zeros((3, 3))
            spin = np.outer(axis, axis)
            angular = alpha_goal + angular_kp * (_cross3(axis, goal_axis) + spin @ orientation_error) - angular_kd * (omega - omega_goal)
        jlinear = jp[:, dof]
        jangular = (projection @ jr)[:, dof]
        linear -= jpd @ data.qvel
        angular -= (projection @ jrd + projection_dot @ jr) @ data.qvel
        posture_error = target - data.qpos[hand['kin'].qpos_adr]
        wrist_pitch = data.qpos[hand['kin'].qpos_adr][5]
        # A soft comfort band lets the wrist assist the arm during the scoop.
        posture_error[5] = 2.0*(np.clip(wrist_pitch, -.45, .45) - wrist_pitch)
        elbow = m.body(hand['kin'].side + '_elbow_link').id
        shoulder = m.body(hand['kin'].side + '_shoulder_pitch_link').id
        elbow_jacobian = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, data, elbow_jacobian, None, elbow)
        sign = -1. if hand['kin'].side == 'right' else 1.
        # Reference the upper arm as well as the palm. The native shoulder
        # chain already offsets the elbow laterally by about 5 cm at rest.
        torso_rotation = data.xmat[m.body('torso_link').id].reshape(3, 3)
        source = hand.get('source_elbow_local', np.array([.047, sign*.055, -.18]))
        mode = hand.get('mode', 'hold')
        if mode == 'carry' and hand.get('carry') is not None:
            motion = hand['carry']
            phase = np.clip((data.time-motion.t0)/(.60*motion.duration), 0., 1.)
            weight = phase**3*(10 - 15*phase + 6*phase**2)
            touch = hand.get('touch_elbow_local', source + np.array([.043, 0., 0.]))
            forward_offset = touch[0] + (source[0]-touch[0])*weight
        elif mode == 'throw':
            phase = np.clip((data.time-hand['stroke'])/.15, 0., 1.)
            forward_offset = source[0] + .030*phase**3*(10 - 15*phase + 6*phase**2)
        else:
            forward_offset = source[0] if mode == 'hold' else source[0] + .043
        offset = torso_rotation.T @ (data.xpos[elbow]-data.xpos[shoulder])
        lateral_error = source[1]-offset[1]
        forward_error = forward_offset-offset[0]
        posture_error += ELBOW_LATERAL_POSTURE_GAIN * (torso_rotation[:,1]@elbow_jacobian[:,dof]) * lateral_error
        posture_error += 12. * (torso_rotation[:,0]@elbow_jacobian[:,dof]) * forward_error
        null = 100 * posture_error - 24 * data.qvel[dof]
        joint_jacobian = np.vstack([jp[:, dof], (np.eye(3) - np.outer(axis, axis)) @ jr[:, dof]])
        posture_projection = np.eye(7) - np.linalg.pinv(joint_jacobian, rcond=.001) @ joint_jacobian
        angular_weight = 0.5 * (np.eye(3) - np.outer(axis, axis)) + 0.06 * np.outer(axis, axis)
        matrix = np.vstack([jp @ response, angular_weight @ jr @ response, 0.60 * posture_projection @ response[dof], np.eye(7) * 0.0003])
        desired = np.r_[linear - jp @ free_acceleration, angular_weight @ (angular - jr @ free_acceleration), 0.60 * posture_projection @ (null - free_acceleration[dof]), np.zeros(7)]
        if balance:
            inverse = np.linalg.inv(self.mass[np.ix_(dof, dof)])
            bias = data.qfrc_bias[dof] - data.qfrc_passive[dof]
            matrix = np.vstack([jlinear @ inverse, 0.2 * jangular @ inverse, 0.01 * inverse])
            desired = np.r_[linear + jlinear @ inverse @ bias, 0.2 * (angular + jangular @ inverse @ bias), 0.01 * (null + inverse @ bias)]
            hessian = matrix.T @ matrix + 1e-07 * np.eye(7)
            rhs = matrix.T @ desired
            torque = np.clip(np.linalg.solve(hessian, rhs), -limits, limits)
            for _ in range(15):
                for i in range(7):
                    torque[i] = max(-limits[i], min(limits[i], float(torque[i] - (hessian[i] @ torque - rhs[i]) / hessian[i, i])))
        else:
            torque = bounded_lstsq(matrix, desired, limits)
        kp = m.actuator_gainprm[acts, 0]
        kv = -m.actuator_biasprm[acts, 2]
        torque = self.protect_arm(hand, data, torque, response, free_acceleration)
        data.ctrl[acts] = data.qpos[hand['kin'].qpos_adr] + (torque + kv * data.qvel[dof]) / kp

    def protect_arm(self, hand, data, torque, response=None, free_acceleration=None):
        import numpy as np
        import mujoco
        limits = hand['force_limits']
        torque = np.clip(torque, -limits, limits)
        dof = hand['kin'].dof_adr
        q = data.qpos[hand['kin'].qpos_adr]
        v = data.qvel[dof]
        sign = 1.0 if hand['kin'].side == 'right' else -1.0
        shoulder = sign * q[1]
        speed = sign * v[1]
        shoulder_near = shoulder + max(0.0, speed) * 0.08 >= -0.06
        braking = hand.get('mode') != 'throw'
        elbow_near = braking and q[3] + min(0.0, v[3]) * 0.08 < -0.8
        # A loaded scoop reverses its wrist velocity within the stroke.
        # Extrapolating that velocity for 80 ms brakes an otherwise safe
        # launch before its planned reversal. Keep protection active, using
        # a short lookahead while throwing and the recovery horizon otherwise.
        wrist_horizon = .08 if braking else WRIST_THROW_LOOKAHEAD_S
        wrist_near = q[5] + max(0.0, v[5]) * wrist_horizon > WRIST_BRAKE_ONSET_RAD
        elbow_upper = q[3] + max(0., v[3]) * .08 > 1.35
        wrist_lower = q[5] + min(0., v[5]) * .08 < -WRIST_BRAKE_ONSET_RAD
        if not (shoulder_near or elbow_near or wrist_near or elbow_upper or wrist_lower):
            return torque
        if response is None:
            rhs = np.zeros((8, self.model.nv))
            rhs[np.arange(7), dof] = 1.0
            rhs[7] = data.qfrc_actuator + data.qfrc_passive - data.qfrc_bias + data.qfrc_constraint
            rhs[7, dof] -= data.qfrc_actuator[dof]
            solved = np.zeros_like(rhs)
            mujoco.mj_solveM(self.model, data, solved, rhs)
            response = solved[:7].T
            free_acceleration = solved[7]
        inequalities = []
        if shoulder_near:
            inequalities.append((sign * response[dof[1]], 200 * (-0.015 - shoulder) - 28 * speed - sign * free_acceleration[dof[1]]))
        if elbow_near:
            inequalities.append((-response[dof[3]], 400 * (q[3] + 0.94) + 40 * v[3] + free_acceleration[dof[3]]))
        if wrist_near:
            inequalities.append((response[dof[5]], 400 * (WRIST_BRAKE_BOUND_RAD - q[5]) - 40 * v[5] - free_acceleration[dof[5]]))
        if elbow_upper:
            inequalities.append((response[dof[3]], 200 * (1.55 - q[3]) - 28 * v[3] - free_acceleration[dof[3]]))
        if wrist_lower:
            inequalities.append((-response[dof[5]], 200 * (q[5] + WRIST_BRAKE_BOUND_RAD) + 28 * v[5] + free_acceleration[dof[5]]))
        for _ in range(4):
            for row, bound in inequalities:
                if row @ torque <= bound:
                    continue
                high = 1.0
                while row @ np.clip(torque - high * row, -limits, limits) > bound and high < 1000000.0:
                    high *= 2
                low = 0.0
                for _ in range(25):
                    mid = (low + high) / 2
                    if row @ np.clip(torque - mid * row, -limits, limits) > bound:
                        low = mid
                    else:
                        high = mid
                torque = np.clip(torque - high * row, -limits, limits)
        return torque

    def follow_palm(self, hand, data, motion, t, target):
        p, v, a = motion.sample(t)
        rotation, omega, alpha = motion.orientation(t)
        self.return_servo(hand, data, target, position=p, velocity=v, acceleration=a, axis_goal=rotation[:, 2], angular_velocity=omega, angular_acceleration=alpha, orientation=rotation)

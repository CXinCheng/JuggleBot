"""Single-arm harness for the UR5e + LEAP hand (v4's handsim.py, in the shared package from v5 on).

HandSim works on a single-arm model (model.load_arm(side)): arm tracking (position servos +
inverse-dynamics feed-forward on a contact-free copy), hand postures, ball placement and contact
bookkeeping (palm / finger / arm / ground / other ball), the 180 deg/s arm joint-speed check.
Several balls (load_arm(n_balls=...)): select_ball(i) picks the one the ball_* methods act on.

The hand-frame constants are those measured on the right hand in v4; for the left hand they are
mapped with model.palm_point (mirror image).

LineStroke: v4's straight stroke with a fixed hand orientation (E2 / E3b).
"""

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from .model import FINGER_TORQUE, MIRROR, Q_READY, load_arm, palm_point, parked_position

ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
ARM_SPEED_LIMIT = np.pi  # rad/s, UR5e datasheet
# Right-hand palm frame (v4): ball resting on the flat palm / in the cradle; bowl seat (E3c)
BALL_ON_PALM = np.array([-0.046, -0.036, -0.035])
BOWL_TILT = np.radians(30.0)
BOWL_SEAT = np.array([-0.0355, -0.036, -0.0771])
# Q_READY (model, v1/v2 "ready": flange up): the right arm holds the palm as in v4; the left arm is
# the right one's mirror image (same joint angles, mirrored pose), so it is the nominal of both.
Q_READY_LEFT = Q_READY.copy()


def q_ready(side: str) -> np.ndarray:
    return (Q_READY if side == "right" else Q_READY_LEFT).copy()


def shift_site_in_palm(model: mujoco.MjModel, site: str, offset_palm, side: str = "right") -> None:
    """Move a hand-fixed site by offset_palm (m, in the RIGHT palm's convention: +x from the heel
    towards the finger roots, -y towards the open ring-finger side; mirrored for the left hand)."""
    off = np.asarray(offset_palm, dtype=float)
    if side == "left":
        off = MIRROR @ off
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    sid = model.site(site).id
    R_body = d.xmat[model.site_bodyid[sid]].reshape(3, 3)
    R_palm = d.xmat[model.body("leap_palm").id].reshape(3, 3)
    model.site_pos[sid] += R_body.T @ R_palm @ off


def set_bowl_sites(model: mujoco.MjModel, side: str = "right", tilt: float = BOWL_TILT, approach: float = 0.025) -> None:
    """Place bowl_seat at the bowl seat and bowl_point approach (m) further out along the bowl axis -- the
    palm normal tilted by `tilt` towards the heel -- with its z axis on the bowl axis (v4 E3c)."""
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    seat = model.site("bowl_seat").id
    body = model.site_bodyid[seat]
    R_b, p_b = d.xmat[body].reshape(3, 3), d.xpos[body]
    palm = model.body("leap_palm").id
    R_p, p_p = d.xmat[palm].reshape(3, 3), d.xpos[palm]
    a_w = R_p @ np.array([-np.sin(tilt), 0.0, -np.cos(tilt)])  # -z = out of the palm, -x = towards the heel
    seat_w = p_p + R_p @ palm_point(side, BOWL_SEAT)
    x_w = R_p @ np.array([1.0, 0.0, 0.0])
    x_w = x_w - (x_w @ a_w) * a_w
    x_w /= np.linalg.norm(x_w)
    R_site = R_b.T @ np.column_stack([x_w, np.cross(a_w, x_w), a_w])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R_site.flatten())
    for name, pos in (("bowl_seat", seat_w), ("bowl_point", seat_w + approach * a_w)):
        sid = model.site(name).id
        model.site_pos[sid] = R_b.T @ (pos - p_b)
        model.site_quat[sid] = quat


@dataclass
class BallContacts:
    palm: bool = False
    finger: bool = False  # any finger or thumb link
    arm: bool = False  # arm links (not the hand)
    ground: bool = False  # ground or a pedestal
    ball: bool = False  # another ball

    @property
    def hand(self) -> bool:
        return self.palm or self.finger


class HandSim:
    def __init__(self, model: mujoco.MjModel | None = None, side: str = "right"):
        self.side = side
        self.model = m = load_arm(side) if model is None else model
        self.data = mujoco.MjData(m)
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self.arm_dof = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self.arm_act = np.array([m.actuator(j.replace("_joint", "")).id for j in ARM_JOINTS])
        self.hand_joints = [m.joint(j).name for j in range(m.njnt) if m.joint(j).name.startswith("leap_")]
        self.hand_qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.hand_joints])
        self.hand_act = np.array([m.actuator(n + "_act").id for n in self.hand_joints])
        self.hand_ctrl_lo = m.actuator_ctrlrange[self.hand_act, 0].copy()
        self.hand_ctrl_hi = m.actuator_ctrlrange[self.hand_act, 1].copy()
        names = ["ball"] + [f"ball_{i}" for i in range(1, m.nbody) if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ball_{i}") >= 0]
        self.balls = []  # (qpos adr, dof adr, geom) per ball
        for name in names:
            joint = m.joint(f"{name}_free").id
            self.balls.append((m.jnt_qposadr[joint], m.jnt_dofadr[joint], m.geom(name).id))
        self.ball_geoms = {b[2] for b in self.balls}
        self.select_ball(0)
        self.ball_radius = float(m.geom_size[self.ball_geom][0])
        self.palm = m.body("leap_palm").id
        self.ground = m.geom("ground").id
        self.ground_geoms = {g for g in (self.ground, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pedestal")) if g >= 0}
        self.palm_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.palm}
        self.hand_geoms = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("leap_")}
        self.arm_geoms = {g for g in range(m.ngeom)
                          if g not in self.hand_geoms and g not in self.ground_geoms and g not in self.ball_geoms}
        self.kp = m.actuator_gainprm[self.arm_act, 0].copy()
        self.kd = -m.actuator_biasprm[self.arm_act, 2].copy()
        self.dynamics = ArmDynamics(m)
        self.speed_peak = 0.0

    # ------------------------------------------------------------ state
    def select_ball(self, i: int) -> None:
        self.ball_qadr, self.ball_dadr, self.ball_geom = self.balls[i]

    def set_ball_state(self, i: int, p, v) -> None:
        qa, da, _ = self.balls[i]
        self.data.qpos[qa:qa + 3] = p
        self.data.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
        self.data.qvel[da:da + 3] = v
        self.data.qvel[da + 3:da + 6] = 0

    def park_ball(self, i: int) -> None:
        self.set_ball_state(i, parked_position(i), np.zeros(3))

    def reset(self, q_arm=None, hand=None) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        q_arm = q_ready(self.side) if q_arm is None else q_arm
        d.qpos[self.arm_qadr] = q_arm
        d.ctrl[self.arm_act] = q_arm
        self.set_hand(np.zeros(len(self.hand_joints)) if hand is None else hand, teleport=True)
        mujoco.mj_forward(m, d)
        self.speed_peak = 0.0

    def set_hand(self, targets: np.ndarray, teleport: bool = False) -> None:
        targets = np.clip(targets, self.hand_ctrl_lo, self.hand_ctrl_hi)
        self.data.ctrl[self.hand_act] = targets
        if teleport:
            self.data.qpos[self.hand_qadr] = targets

    def palm_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.palm].copy(), self.data.xmat[self.palm].reshape(3, 3).copy()

    def place_ball(self, p_palm_right: np.ndarray) -> None:
        """Selected ball centre at p_palm_right (RIGHT palm convention, mirrored for the left hand), at rest."""
        p, R = self.palm_pose()
        self.data.qpos[self.ball_qadr:self.ball_qadr + 3] = p + R @ palm_point(self.side, p_palm_right)
        self.data.qpos[self.ball_qadr + 3:self.ball_qadr + 7] = (1, 0, 0, 0)
        self.data.qvel[self.ball_dadr:self.ball_dadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def ball_state(self):
        return (self.data.qpos[self.ball_qadr:self.ball_qadr + 3].copy(),
                self.data.qvel[self.ball_dadr:self.ball_dadr + 3].copy())

    def ball_contacts(self, ball: int | None = None) -> BallContacts:
        d, out = self.data, BallContacts()
        geom = self.ball_geom if ball is None else self.balls[ball][2]
        for i in range(d.ncon):
            c = d.contact[i]
            if geom not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == geom else c.geom1
            if other in self.ball_geoms:
                out.ball = True
            elif other in self.palm_geoms:
                out.palm = True
            elif other in self.hand_geoms:
                out.finger = True
            elif other in self.ground_geoms:
                out.ground = True
            elif other in self.arm_geoms:
                out.arm = True
        return out

    # ------------------------------------------------------------ control
    def arm_torque_ff(self, q, qd, qdd) -> np.ndarray:
        return self.dynamics(q, qd, qdd, self.data.qpos[self.hand_qadr])

    def control_arm(self, q, qd=None, qdd=None) -> None:
        qd = np.zeros(6) if qd is None else qd
        qdd = np.zeros(6) if qdd is None else qdd
        tau = self.arm_torque_ff(q, qd, qdd)
        self.data.ctrl[self.arm_act] = q + (self.kd / self.kp) * qd + tau / self.kp

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        self.speed_peak = max(self.speed_peak, float(np.max(np.abs(self.data.qvel[self.arm_dof]))) / ARM_SPEED_LIMIT)

    def finger_torque_ratio(self) -> float:
        return float(np.max(np.abs(self.data.actuator_force[self.hand_act]))) / FINGER_TORQUE


class ArmDynamics:
    """Inverse dynamics of one arm (+ hand, fingers at the given qpos) on a contact-free copy of a
    single-arm model, balls parked far away. Shared by HandSim and the multi-arm world."""

    def __init__(self, model: mujoco.MjModel):
        self.m = copy.deepcopy(model)
        self.m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self.arm_dof = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self.hand_qadr = np.array([m.jnt_qposadr[j] for j in range(m.njnt) if m.joint(j).name.startswith("leap_")])
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 3] = (10.0, 10.0 + j, 5.0)
        self.force_limit = m.actuator_forcerange[[m.actuator(j.replace("_joint", "")).id for j in ARM_JOINTS], 1].copy()

    def __call__(self, q, qd, qdd, q_hand=None) -> np.ndarray:
        d = self.d
        d.qpos[self.arm_qadr] = q
        if q_hand is not None:
            d.qpos[self.hand_qadr] = q_hand
        d.qvel[:] = 0
        d.qvel[self.arm_dof] = qd
        d.qacc[:] = 0
        d.qacc[self.arm_dof] = qdd
        mujoco.mj_inverse(self.m, d)
        return d.qfrc_inverse[self.arm_dof].copy()


class LineStroke:
    """The flange moves along a straight line through the release configuration q_r, with a fixed
    hand orientation: q(s) integrated (RK2) from dq/ds = J+ [d; 0] at the flange, outwards from
    s = 0 (s = distance along d, m). Timed by a smoothstep speed profile: accelerate to V over
    T1 = 1.5 V / a_up (peak acceleration a_up), then stop over T2 = 1.5 V / a_down; s = 0 (peak
    speed) is reached at t_peak = t0 + T1."""

    DS = 0.0005  # m

    def __init__(self, model: mujoco.MjModel, q_r: np.ndarray, direction: np.ndarray, V: float, a_up: float,
                 a_down: float, t0: float = 0.0):
        m = model
        self.d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        self.V, self.t0 = V, t0
        self.T1, self.T2 = 1.5 * V / a_up, 1.5 * V / a_down
        self.t_peak, self.t_end = t0 + self.T1, t0 + self.T1 + self.T2
        s_lo, s_hi = -V * self.T1 / 2, V * self.T2 / 2
        scratch = mujoco.MjData(m)
        qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        dof = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        site = m.site("flange").id
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        task = np.concatenate([self.d, np.zeros(3)])

        def f(q):
            scratch.qpos[qadr] = q
            mujoco.mj_kinematics(m, scratch)
            mujoco.mj_comPos(m, scratch)
            mujoco.mj_jacSite(m, scratch, jp, jr, site)
            return np.linalg.pinv(np.vstack([jp[:, dof], jr[:, dof]])) @ task

        n_lo, n_hi = int(np.ceil(-s_lo / self.DS)) + 1, int(np.ceil(s_hi / self.DS)) + 1
        self.s = np.arange(-n_lo, n_hi + 1) * self.DS
        i0 = n_lo
        self.q = np.zeros((len(self.s), 6))
        self.q[i0] = q_r
        for step, idx in ((self.DS, range(i0, len(self.s) - 1)), (-self.DS, range(i0, 0, -1))):
            for i in idx:
                k1 = f(self.q[i])
                self.q[i + int(np.sign(step))] = self.q[i] + step * f(self.q[i] + 0.5 * step * k1)
        self.dq = np.array([f(q) for q in self.q])
        self.ddq = np.gradient(self.dq, self.DS, axis=0)
        self.q_start = self._at(s_lo)[0]
        self.q_end = self._at(s_hi)[0]

    def _at(self, s):
        x = np.clip((s - self.s[0]) / self.DS, 0, len(self.s) - 1.000001)
        i = int(x)
        w = x - i
        return [(1 - w) * a[i] + w * a[i + 1] for a in (self.q, self.dq, self.ddq)]

    def path_speed(self, t: float) -> tuple[float, float, float]:
        """(s, ds/dt, d2s/dt2) at time t."""
        V, T1, T2 = self.V, self.T1, self.T2
        tau = t - self.t0
        if tau <= 0:
            return -V * T1 / 2, 0.0, 0.0
        if tau < T1:
            x = tau / T1
            return -V * T1 / 2 + V * T1 * (x ** 3 - x ** 4 / 2), V * (3 * x ** 2 - 2 * x ** 3), V * (6 * x - 6 * x ** 2) / T1
        if tau < T1 + T2:
            x = (tau - T1) / T2
            return V * T2 * (x - x ** 3 + x ** 4 / 2), V * (1 - 3 * x ** 2 + 2 * x ** 3), V * (-6 * x + 6 * x ** 2) / T2
        return V * T2 / 2, 0.0, 0.0

    def sample(self, t):
        s, sd, sdd = self.path_speed(t)
        q, dq, ddq = self._at(s)
        return q, dq * sd, ddq * sd ** 2 + dq * sdd

    def max_joint_speed(self, n: int = 400) -> float:
        ts = np.linspace(self.t0, self.t_end, n)
        return max(float(np.max(np.abs(self.sample(t)[1]))) for t in ts)

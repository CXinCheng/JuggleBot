"""Simulation harness for v4: UR5e + LEAP hand + one juggling-sized ball.

* load_model(): compiles scene.xml through MjSpec and adds what the MJCF cannot express
  for an attached sub-model:
    - gravcomp = 1 on the hand bodies (a real arm controller compensates its payload; the
      fingers' own servos would hold them against gravity as well)
    - finger torque limits: the Menagerie model has none. LEAP uses Dynamixel XC330-M288-T
      servos run at a 600-750 mA current limit (Shaw et al., RSS 2023); FINGER_TORQUE is a
      conservative estimate of the torque at that current (to be verified).
    - compliant, damped palm and fingertip contacts (SKIN_*): the Menagerie hand is nearly rigid
      (solref 0.0001); a padded palm / fingertips give way a few mm and damp the impact of a
      caught ball. solmix makes these parameters dominate MuJoCo's per-contact averaging.
* HandSim: arm tracking (position servos + inverse-dynamics feed-forward, as in v2), hand
  postures (position targets for the 16 finger joints), ball placement and contact
  bookkeeping (palm / finger / other), and the 180 deg/s arm joint-speed check.
"""

import copy
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SCENE = HERE / "scene.xml"
ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
FINGERS = ("if", "mf", "rf")
FINGER_TORQUE = 0.5  # N m, estimated XC330-M288-T torque at the LEAP current limit (unverified)
ARM_SPEED_LIMIT = np.pi  # rad/s, UR5e datasheet
SKIN_SOLREF = (0.006, 1.2)  # time constant (s), damping ratio (> 1: no rebound)
SKIN_SOLIMP = (0.9, 0.95, 0.004)  # softer impedance, 4 mm transition width
SKIN_SOLMIX = 100.0
SKIN_GEOMS = ("leap_palm_collision_", "leap_if_tip", "leap_mf_tip", "leap_rf_tip", "leap_th_tip")
Q_READY = np.array([0.302, -1.649, -2.206, -0.857, -1.571, -1.873])  # v1/v2 "ready": flange up at (0.45, 0, 0.43)


def load_model(scene: Path = SCENE, finger_torque: float = FINGER_TORQUE, skin: bool = True,
               pedestal: float = 0.0) -> mujoco.MjModel:
    """pedestal > 0: the robot base stands on a box this high (m) -- the stretched-out arm, where the
    hand moves fastest vertically, is then high enough to throw, catch and give way below."""
    spec = mujoco.MjSpec.from_file(str(scene))
    if pedestal > 0:
        spec.body("base").pos[2] += pedestal
        spec.worldbody.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, pedestal / 2],
                                pos=[0, 0, pedestal / 2], rgba=[0.35, 0.35, 0.38, 1])
    if skin:
        for geom in spec.geoms:
            if geom.name.startswith(SKIN_GEOMS):
                geom.solref = list(SKIN_SOLREF)
                geom.solimp = list(SKIN_SOLIMP) + [0.5, 2.0]
                geom.solmix = SKIN_SOLMIX
    for body in spec.bodies:
        if body.name.startswith("leap_"):
            body.gravcomp = 1.0
    for act in spec.actuators:
        if act.name.startswith("leap_"):
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-finger_torque, finger_torque]
    return spec.compile()


@dataclass
class BallContacts:
    palm: bool = False
    finger: bool = False  # any finger or thumb link
    arm: bool = False  # arm links (not the hand)
    ground: bool = False

    @property
    def hand(self) -> bool:
        return self.palm or self.finger


class HandSim:
    def __init__(self, model: mujoco.MjModel | None = None):
        self.model = m = load_model() if model is None else model
        self.data = mujoco.MjData(m)
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self.arm_dof = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self.arm_act = np.array([m.actuator(j.replace("_joint", "")).id for j in ARM_JOINTS])
        self.hand_joints = [m.joint(j).name for j in range(m.njnt) if m.joint(j).name.startswith("leap_")]
        self.hand_qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.hand_joints])
        self.hand_act = np.array([m.actuator(n + "_act").id for n in self.hand_joints])
        self.hand_ctrl_lo = m.actuator_ctrlrange[self.hand_act, 0].copy()
        self.hand_ctrl_hi = m.actuator_ctrlrange[self.hand_act, 1].copy()
        joint = m.joint("ball_free").id
        self.ball_qadr, self.ball_dadr = m.jnt_qposadr[joint], m.jnt_dofadr[joint]
        self.ball_geom = m.geom("ball").id
        self.ball_radius = float(m.geom_size[self.ball_geom][0])
        self.palm = m.body("leap_palm").id
        self.ground = m.geom("ground").id
        # the pedestal (if any) counts as ground: the ball is lost if it gets there
        self.ground_geoms = {g for g in (self.ground, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pedestal")) if g >= 0}
        self.palm_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.palm}
        self.hand_geoms = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("leap_")}
        self.arm_geoms = {g for g in range(m.ngeom)
                          if g not in self.hand_geoms and g not in self.ground_geoms and g != self.ball_geom}
        # Arm servos: force = kp (ctrl - q) - kd qd
        self.kp = m.actuator_gainprm[self.arm_act, 0].copy()
        self.kd = -m.actuator_biasprm[self.arm_act, 2].copy()
        # Contact-free copy for inverse dynamics (ball parked far away)
        self._idm = copy.deepcopy(m)
        self._idm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self._idd = mujoco.MjData(self._idm)
        self._idd.qpos[self.ball_qadr:self.ball_qadr + 3] = (10.0, 10.0, 5.0)
        self.speed_peak = 0.0

    # ------------------------------------------------------------ state
    def reset(self, q_arm=Q_READY, hand=None) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_qadr] = q_arm
        d.ctrl[self.arm_act] = q_arm
        self.set_hand(np.zeros(len(self.hand_joints)) if hand is None else hand, teleport=True)
        d.qpos[self.ball_qadr:self.ball_qadr + 3] = (1.5, 0.0, self.ball_radius)
        mujoco.mj_forward(m, d)
        self.speed_peak = 0.0

    def set_hand(self, targets: np.ndarray, teleport: bool = False) -> None:
        targets = np.clip(targets, self.hand_ctrl_lo, self.hand_ctrl_hi)
        self.data.ctrl[self.hand_act] = targets
        if teleport:
            self.data.qpos[self.hand_qadr] = targets

    def palm_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.palm].copy(), self.data.xmat[self.palm].reshape(3, 3).copy()

    def place_ball(self, p_palm: np.ndarray) -> None:
        """Ball centre at p_palm (palm frame), at rest."""
        p, R = self.palm_pose()
        self.data.qpos[self.ball_qadr:self.ball_qadr + 3] = p + R @ p_palm
        self.data.qpos[self.ball_qadr + 3:self.ball_qadr + 7] = (1, 0, 0, 0)
        self.data.qvel[self.ball_dadr:self.ball_dadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def ball_in_palm(self) -> np.ndarray:
        p, R = self.palm_pose()
        return R.T @ (self.data.qpos[self.ball_qadr:self.ball_qadr + 3] - p)

    def ball_state(self):
        return (self.data.qpos[self.ball_qadr:self.ball_qadr + 3].copy(),
                self.data.qvel[self.ball_dadr:self.ball_dadr + 3].copy())

    def ball_contacts(self) -> BallContacts:
        d, out = self.data, BallContacts()
        for i in range(d.ncon):
            c = d.contact[i]
            if self.ball_geom not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.ball_geom else c.geom1
            if other in self.palm_geoms:
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
        d = self._idd
        d.qpos[self.arm_qadr] = q
        d.qpos[self.hand_qadr] = self.data.qpos[self.hand_qadr]
        d.qvel[:] = 0
        d.qvel[self.arm_dof] = qd
        d.qacc[:] = 0
        d.qacc[self.arm_dof] = qdd
        mujoco.mj_inverse(self._idm, d)
        return d.qfrc_inverse[self.arm_dof].copy()

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


class LineStroke:
    """The flange moves along a straight line through the release configuration q_r, with a fixed
    hand orientation: q(s) integrated (RK2) from dq/ds = J+ [d; 0] at the flange, outwards from
    s = 0 (s = distance along d, m). Timed by a smoothstep speed profile: accelerate to V over
    T1 = 1.5 V / a_up (peak acceleration a_up), then stop over T2 = 1.5 V / a_down; s = 0 (peak
    speed) is reached at t_peak = t0 + T1."""

    DS = 0.0005  # m

    def __init__(self, sim: HandSim, q_r: np.ndarray, direction: np.ndarray, V: float, a_up: float,
                 a_down: float, t0: float = 0.0):
        m = sim.model
        self.d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        self.V, self.t0 = V, t0
        self.T1, self.T2 = 1.5 * V / a_up, 1.5 * V / a_down
        self.t_peak, self.t_end = t0 + self.T1, t0 + self.T1 + self.T2
        s_lo, s_hi = -V * self.T1 / 2, V * self.T2 / 2
        scratch = mujoco.MjData(m)
        site = m.site("flange").id
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        task = np.concatenate([self.d, np.zeros(3)])

        def f(q):
            scratch.qpos[sim.arm_qadr] = q
            mujoco.mj_kinematics(m, scratch)
            mujoco.mj_comPos(m, scratch)
            mujoco.mj_jacSite(m, scratch, jp, jr, site)
            return np.linalg.pinv(np.vstack([jp[:, sim.arm_dof], jr[:, sim.arm_dof]])) @ task

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

"""Simulation harness shared by the v2 experiments.

* Tracking: the arm's position servos (force = kp (ctrl - q) - kd qd) get
  ctrl = q_d + (kd / kp) qd_d + tau_ff / kp with tau_ff = M qdd_d + c(q_d, qd_d) from
  inverse dynamics at the desired state, i.e. PD tracking plus full feed-forward
  (v1 used M qdd only; at throwing speeds the velocity-product terms matter).
* Joint-speed limit: MuJoCo has no joint-speed limit, so the UR5e's 180 deg/s is
  checked on the *actual* joint velocities every step; exceeding it fails the trial.
* Ball / cup contacts: classified per step (cup base, cup wall, other robot geoms,
  ground) for release detection and re-contact checks.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from kinematics import ArmKinematics
from throwing import InverseDynamics

JOINT_SPEED_LIMIT = np.pi  # rad/s, UR5e datasheet (every joint)


class JointSpeedGuard:
    def __init__(self, limit: float = JOINT_SPEED_LIMIT):
        self.limit = limit
        self.peak = 0.0  # max |qd| / limit seen so far
        self.violated_at: float | None = None

    def update(self, t: float, qvel: np.ndarray) -> None:
        r = float(np.max(np.abs(qvel))) / self.limit
        self.peak = max(self.peak, r)
        if r > 1.0 and self.violated_at is None:
            self.violated_at = t


@dataclass
class BallContacts:
    base: bool = False
    wall: bool = False
    robot: bool = False  # any other robot geom
    ground: bool = False

    @property
    def cup(self) -> bool:
        return self.base or self.wall


class ArmSim:
    def __init__(self, scene: Path):
        self.model = m = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(m)
        self.kin = ArmKinematics(m)
        self.idyn = InverseDynamics(m, self.kin)
        self.q_ready = m.key_qpos[m.key("ready").id][self.kin.qpos_adr].copy()
        joint = m.joint("ball_free").id
        self.ball_qadr, self.ball_dadr = m.jnt_qposadr[joint], m.jnt_dofadr[joint]
        self.ball_geom = m.geom("ball").id
        self.ball_radius = float(m.geom_size[self.ball_geom][0])
        self.cup_base = m.geom("cup_base").id
        self.cup_walls = {g for g in range(m.ngeom) if m.geom(g).name.startswith("cup_wall")}
        self.ground = m.geom("ground").id
        self.robot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in self.kin._robot_bodies}
        self.seat = m.site("ball_seat").id
        self.cup_body = m.body("cup").id
        self.kp = m.actuator_gainprm[:, 0].copy()
        self.kd = -m.actuator_biasprm[:, 2].copy()
        self.force_limit = m.actuator_forcerange[:, 1].copy()
        self.ctrl_lo, self.ctrl_hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        # Cup cavity in the ball_seat frame (z along the cup axis, 0 = ball resting on the base).
        walls = sorted(self.cup_walls)
        self.inner_radius = min(np.linalg.norm(m.geom_pos[g][:2]) - m.geom_size[g][0] for g in walls)
        self.rim_height = min(m.geom_pos[g][2] + m.geom_size[g][2] for g in walls) - m.site_pos[self.seat][2]
        self.guard = JointSpeedGuard()
        self.max_force_ratio = 0.0
        self.ctrl_clamped = 0  # steps with a servo command outside ctrlrange

    # ------------------------------------------------------------ state
    def reset(self, q: np.ndarray, ball_offset: np.ndarray | None = None) -> None:
        """Arm at q (at rest), ball resting on the cup base, shifted by ball_offset (x, y in the cup frame)."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.kin.qpos_adr] = q
        d.ctrl[:] = q
        mujoco.mj_kinematics(m, d)
        seat = d.site_xpos[self.seat].copy()
        rot = d.site_xmat[self.seat].reshape(3, 3)
        off = np.zeros(3) if ball_offset is None else np.array([ball_offset[0], ball_offset[1], 0.0])
        d.qpos[self.ball_qadr:self.ball_qadr + 3] = seat + rot @ (off + np.array([0, 0, 0.0005]))
        d.qpos[self.ball_qadr + 3:self.ball_qadr + 7] = (1, 0, 0, 0)
        mujoco.mj_forward(m, d)
        self.guard = JointSpeedGuard()
        self.max_force_ratio = 0.0
        self.ctrl_clamped = 0

    def ball_state(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.qpos[self.ball_qadr:self.ball_qadr + 3].copy()
        vel = self.data.qvel[self.ball_dadr:self.ball_dadr + 3].copy()
        return pos, vel

    def ball_in_cup_frame(self) -> np.ndarray:
        """Ball centre in the ball_seat frame (z along the cup axis, 0 = resting on the base)."""
        d = self.data
        rot = d.site_xmat[self.seat].reshape(3, 3)
        return rot.T @ (d.qpos[self.ball_qadr:self.ball_qadr + 3] - d.site_xpos[self.seat])

    def ball_relative_velocity(self) -> np.ndarray:
        """Ball velocity relative to the cup material point at the ball centre (world frame)."""
        m, d = self.model, self.data
        cup_vel, ball_vel = np.zeros(6), np.zeros(6)  # [angular; linear] at the body origins
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.cup_body, cup_vel, 0)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[self.ball_geom], ball_vel, 0)
        p_ball = d.qpos[self.ball_qadr:self.ball_qadr + 3]
        return ball_vel[3:] - (cup_vel[3:] + np.cross(cup_vel[:3], p_ball - d.xpos[self.cup_body]))

    def ball_in_cup(self, tol: float = 0.002) -> bool:
        """Ball centre inside the cavity: within the wall radius and not above the rim (the cup is
        shallower than the ball, as in v1)."""
        p = self.ball_in_cup_frame()
        return (np.linalg.norm(p[:2]) <= self.inner_radius - self.ball_radius + tol
                and -tol <= p[2] <= self.rim_height + tol)

    def ball_contacts(self) -> BallContacts:
        d = self.data
        out = BallContacts()
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == self.ball_geom:
                other = c.geom2
            elif c.geom2 == self.ball_geom:
                other = c.geom1
            else:
                continue
            if other == self.cup_base:
                out.base = True
            elif other in self.cup_walls:
                out.wall = True
            elif other == self.ground:
                out.ground = True
            elif other in self.robot_geoms:
                out.robot = True
        return out

    # ------------------------------------------------------------ control / stepping
    def control(self, traj) -> None:
        q_d, qd_d, qdd_d = traj.sample(self.data.time)
        tau = self.idyn(q_d, qd_d, qdd_d)
        ctrl = q_d + (self.kd / self.kp) * qd_d + tau / self.kp
        if np.any(ctrl < self.ctrl_lo) or np.any(ctrl > self.ctrl_hi):
            self.ctrl_clamped += 1
        self.data.ctrl[:] = ctrl

    def step(self) -> None:
        d = self.data
        mujoco.mj_step(self.model, d)
        self.guard.update(d.time, d.qvel[self.kin.dof_adr])
        self.max_force_ratio = max(self.max_force_ratio, float(np.max(np.abs(d.actuator_force) / self.force_limit)))


# ---------------------------------------------------------------- throw rollout
SETTLE = 0.3  # s at the stroke start pose before the throw (the ball settles on the base)
FREE_CONFIRM = 0.03  # s without any ball-robot contact that confirms the release
THROW_TIMEOUT = 2.0  # s after the stroke start


@dataclass
class ThrowRollout:
    outcome: str  # OK (past the apex, no re-contact) / RECONTACT / GROUND / NO_RELEASE
    t_release: float = np.nan  # s since the stroke start: first contact-free step of the final release
    p_release: np.ndarray = None  # ball state at that step
    v_release: np.ndarray = None
    wall_at_release: bool = False  # the ball touched the cup wall in the last contact step
    rattles: int = 0  # contact lost and regained before the final release
    speed_peak: float = 0.0  # max |qd| / 180 deg/s over the run (actual)
    speed_violation: bool = False
    force_peak: float = 0.0  # max |actuator force| / limit
    ctrl_clamped: int = 0


def simulate_throw(sim: ArmSim, plan, ball_offset=None, recorder=None) -> ThrowRollout:
    """Run one throw (ball settled at the stroke start, then the stroke, then hold) until the
    ball is past its apex. The release is the last loss of ball-robot contact after which the
    ball stays free for FREE_CONFIRM."""
    from planning import Hold, Trajectory  # local: planning.py has no simulation dependencies

    d = sim.data
    sim.reset(plan.q_start, ball_offset)
    traj = Trajectory([Hold(0.0, SETTLE, plan.q_start), plan.trajectory(SETTLE),
                       Hold(SETTLE + plan.duration, np.inf, plan.q_end)])
    res = ThrowRollout("NO_RELEASE")
    state, candidate, last_contact = "held", None, None
    while d.time < SETTLE + THROW_TIMEOUT:
        sim.control(traj)
        sim.step()
        if recorder is not None:
            recorder.capture(d)
        c = sim.ball_contacts()
        touching = c.cup or c.robot
        if c.ground:
            res.outcome = "GROUND"
            break
        if state == "held":
            if touching:
                last_contact = c
            elif d.time > SETTLE:
                candidate, state = (d.time, *sim.ball_state(), last_contact), "maybe_free"
        elif state == "maybe_free":
            if touching:
                res.rattles += 1
                state, last_contact = "held", c
            elif d.time - candidate[0] >= FREE_CONFIRM:
                state = "free"
        else:
            if touching:
                res.outcome = "RECONTACT"  # the cup caught up with the ball on its way out
                break
            if sim.ball_state()[1][2] < 0:
                res.outcome = "OK"  # past the apex: the rest is ballistic
                break
    res.speed_peak, res.speed_violation = sim.guard.peak, sim.guard.violated_at is not None
    res.force_peak, res.ctrl_clamped = sim.max_force_ratio, sim.ctrl_clamped
    if candidate is not None and state != "held":
        t, p, v, contact = candidate
        res.t_release, res.p_release, res.v_release = t - SETTLE, p, v
        res.wall_at_release = bool(contact is not None and contact.wall)
    return res

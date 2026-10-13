"""Forward / inverse kinematics of the UR5e + LEAP hand on a private scratch MjData (v4).

Copied from v2 (cup) with the task point moved to the flange site. The IK task stays 5-D:
flange position + flange axis (= palm normal). The rotation about the palm normal (which
way the fingers point) is left to the null-space pull towards the nominal configuration;
the finger frame matters for the hand (it is not symmetric like the cup), so callers check
it where needed. The scratch MjData keeps the fingers flat (qpos 0).

v2: fk / jacobian also accept another site of the cup (e.g. ball_seat, where the
ball centre sits when it rests on the cup base).
"""

from dataclasses import dataclass

import mujoco
import numpy as np

ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
TCP_SITE = "flange"

# Solver settings
IK_MAX_ITERS = 150
IK_DAMPING = 1e-2  # lambda of damped least squares
IK_MAX_STEP = 0.2  # max joint update per iteration (rad)
IK_NULLSPACE_GAIN = 0.1
IK_POS_TOL = 1e-5  # m (tight: the dense Cartesian phases are finite-differenced)
IK_AXIS_TOL = 1e-4  # rad

# Ground / self-collision check: geoms of the robot must stay this far above the floor.
MIN_LINK_HEIGHT = 0.05  # m


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    pos_err: float  # m
    axis_err: float  # rad
    iters: int


class ArmKinematics:
    """Kinematics helper for the 6 arm joints, independent of the simulation state."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)  # scratch copy; never stepped
        joints = [model.joint(n) for n in ARM_JOINTS]
        self.qpos_adr = np.array([j.qposadr[0] for j in joints])
        self.dof_adr = np.array([j.dofadr[0] for j in joints])
        self.q_min = np.array([j.range[0] for j in joints])
        self.q_max = np.array([j.range[1] for j in joints])
        self.site_id = model.site(TCP_SITE).id
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

        # Robot geoms with collision enabled (for the clearance / contact check).
        robot_root = model.body("shoulder_link").id
        self._robot_bodies = {b for b in range(model.nbody)
                              if self._is_descendant(b, robot_root)}
        self._ground_geom = model.geom("ground").id
        # Park the ball far away so it never shows up in the scratch contact check.
        ball_adr = model.jnt_qposadr[model.joint("ball_free").id]
        self.data.qpos[ball_adr:ball_adr + 3] = (10.0, 10.0, 5.0)

    def _is_descendant(self, body: int, root: int) -> bool:
        while body != 0:
            if body == root:
                return True
            body = self.model.body_parentid[body]
        return False

    # ------------------------------------------------------------------ FK
    def fk(self, q: np.ndarray, site: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (site position, cup axis) for joint vector q; site defaults to cup_mouth."""
        self._set_q(q)
        mujoco.mj_kinematics(self.model, self.data)
        return self._site_pose(site)

    def jacobian(self, q: np.ndarray, site: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (3x6 positional, 3x6 rotational) Jacobian of a cup site (default cup_mouth) at q."""
        self._set_q(q)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # cdof, needed by mj_jacSite
        return self._site_jac(site)

    def _set_q(self, q):
        self.data.qpos[self.qpos_adr] = q

    def _site_pose(self, site=None):
        site = self.site_id if site is None else site
        pos = self.data.site_xpos[site].copy()
        axis = self.data.site_xmat[site].reshape(3, 3)[:, 2].copy()  # all cup sites share the cup axis
        return pos, axis

    def _site_jac(self, site=None):
        site = self.site_id if site is None else site
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, site)
        return self._jacp[:, self.dof_adr].copy(), self._jacr[:, self.dof_adr].copy()

    # ------------------------------------------------------------------ IK
    def solve(self, target_pos: np.ndarray, target_axis: np.ndarray, q_init: np.ndarray,
              q_nominal: np.ndarray | None = None, max_iters: int = IK_MAX_ITERS) -> IKResult:
        """Damped-least-squares IK for (cup_mouth position, cup axis)."""
        target_axis = target_axis / np.linalg.norm(target_axis)
        q = np.clip(np.array(q_init, dtype=float), self.q_min, self.q_max)
        pos_err = axis_err = np.inf
        for it in range(max_iters):
            self._set_q(q)
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            pos, axis = self._site_pose()

            e_pos = target_pos - pos
            # Small rotation that turns the current axis onto the target axis.
            cross = np.cross(axis, target_axis)
            axis_err = float(np.arctan2(np.linalg.norm(cross), axis @ target_axis))
            e_rot = cross if np.linalg.norm(cross) < 1e-9 else cross / np.linalg.norm(cross) * axis_err
            pos_err = float(np.linalg.norm(e_pos))
            if pos_err < IK_POS_TOL and axis_err < IK_AXIS_TOL:
                return IKResult(q, True, pos_err, axis_err, it)

            jp, jr = self._site_jac()
            # Rotation about the cup axis is free: project it out of the task.
            p_perp = np.eye(3) - np.outer(axis, axis)
            jac = np.vstack([jp, p_perp @ jr])
            err = np.concatenate([e_pos, e_rot])

            jjt = jac @ jac.T + IK_DAMPING ** 2 * np.eye(6)
            dq = jac.T @ np.linalg.solve(jjt, err)
            if q_nominal is not None:
                # Exact (undamped) null-space projector, so the posture term cannot
                # leak into the task and stall convergence. jac has rank 5.
                null = np.eye(len(q)) - np.linalg.pinv(jac, rcond=1e-6) @ jac
                dq += null @ (IK_NULLSPACE_GAIN * (q_nominal - q))

            step = np.max(np.abs(dq))
            if step > IK_MAX_STEP:
                dq *= IK_MAX_STEP / step
            q = np.clip(q + dq, self.q_min, self.q_max)

        return IKResult(q, False, pos_err, axis_err, max_iters)

    # ------------------------------------------------------------------ checks
    def is_collision_free(self, q: np.ndarray) -> bool:
        """True if at q the robot touches neither the ground nor itself and stays above the floor."""
        self._set_q(q)
        mujoco.mj_fwdPosition(self.model, self.data)  # kinematics + collision detection
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if b1 in self._robot_bodies or b2 in self._robot_bodies:
                return False
        for g in range(self.model.ngeom):
            if (self.model.geom_bodyid[g] in self._robot_bodies and self.model.geom_contype[g]
                    and self.data.geom_xpos[g][2] < MIN_LINK_HEIGHT):
                return False
        return True

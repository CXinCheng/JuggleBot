"""Forward and Inverse Kinematics for Unitree G1 7-DOF arms (Stage v6).

Supports 7-DOF redundant inverse kinematics using Damped Least Squares (DLS)
with null-space projection towards a preferred nominal arm posture.
"""

from dataclasses import dataclass
import mujoco
import numpy as np

from .model import (
    ARM_JOINTS_LEFT,
    ARM_JOINTS_RIGHT,
    Q_READY_ARM_LEFT,
    Q_READY_ARM_RIGHT,
)

IK_MAX_ITERS = 150
IK_DAMPING = 1e-2
IK_MAX_STEP = 0.2
IK_NULLSPACE_GAIN = 0.1
IK_POS_TOL = 1e-5
IK_AXIS_TOL = 1e-4


@dataclass
class IKResult7DOF:
    q: np.ndarray
    success: bool
    pos_err: float
    axis_err: float
    iters: int


class HumanoidArmKinematics:
    """7-DOF Kinematics solver for one arm of Unitree G1."""

    def __init__(self, model: mujoco.MjModel, side: str = "right", tcp_site: str | None = None):
        self.model = model
        self.side = side
        self.data = mujoco.MjData(model)
        self.joint_names = ARM_JOINTS_RIGHT if side == "right" else ARM_JOINTS_LEFT
        self.prefix = "r_" if side == "right" else "l_"
        self.tcp_site = tcp_site or f"{self.prefix}catch_point"

        joints = [model.joint(n) for n in self.joint_names]
        self.qpos_adr = np.array([j.qposadr[0] for j in joints])
        self.dof_adr = np.array([j.dofadr[0] for j in joints])
        self.q_min = np.array([j.range[0] for j in joints])
        self.q_max = np.array([j.range[1] for j in joints])
        self.site_id = model.site(self.tcp_site).id
        self.q_nominal = Q_READY_ARM_RIGHT if side == "right" else Q_READY_ARM_LEFT

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def _set_q(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_adr] = q

    def fk(self, q: np.ndarray, site: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (site position, cup/hand axis) for joint vector q (7-DOF)."""
        site = self.site_id if site is None else site
        self._set_q(q)
        mujoco.mj_kinematics(self.model, self.data)
        pos = self.data.site_xpos[site].copy()
        axis = self.data.site_xmat[site].reshape(3, 3)[:, 2].copy()  # z-axis of site
        return pos, axis

    def jacobian(self, q: np.ndarray, site: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (3x7 positional, 3x7 rotational) Jacobian of arm joints at q."""
        site = self.site_id if site is None else site
        self._set_q(q)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, site)
        return self._jacp[:, self.dof_adr].copy(), self._jacr[:, self.dof_adr].copy()

    def solve(
        self,
        target_pos: np.ndarray,
        target_axis: np.ndarray,
        q_init: np.ndarray,
        q_nominal: np.ndarray | None = None,
        max_iters: int = IK_MAX_ITERS,
    ) -> IKResult7DOF:
        """7-DOF Damped-least-squares IK with null-space posture projection."""
        target_axis = target_axis / np.linalg.norm(target_axis)
        q = np.clip(np.array(q_init, dtype=float), self.q_min, self.q_max)
        q_nom = self.q_nominal if q_nominal is None else q_nominal
        pos_err = axis_err = np.inf

        for it in range(max_iters):
            self._set_q(q)
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            pos = self.data.site_xpos[self.site_id].copy()
            axis = self.data.site_xmat[self.site_id].reshape(3, 3)[:, 2].copy()

            e_pos = target_pos - pos
            cross = np.cross(axis, target_axis)
            axis_err = float(np.arctan2(np.linalg.norm(cross), axis @ target_axis))
            e_rot = cross if np.linalg.norm(cross) < 1e-9 else cross / np.linalg.norm(cross) * axis_err
            pos_err = float(np.linalg.norm(e_pos))

            if pos_err < IK_POS_TOL and axis_err < IK_AXIS_TOL:
                return IKResult7DOF(q, True, pos_err, axis_err, it)

            jp, jr = self.jacobian(q)
            # Free rotation around normal: project out of rotational task
            p_perp = np.eye(3) - np.outer(axis, axis)
            jac = np.vstack([jp, p_perp @ jr])  # 6 x 7 task Jacobian
            err = np.concatenate([e_pos, e_rot])

            # Damped least squares
            jjt = jac @ jac.T + (IK_DAMPING ** 2) * np.eye(6)
            dq = jac.T @ np.linalg.solve(jjt, err)

            # Redundant null-space projection (7-DOF has 1 or more redundant DOFs)
            pinv_jac = np.linalg.pinv(jac, rcond=1e-4)
            null_proj = np.eye(len(q)) - pinv_jac @ jac
            dq += null_proj @ (IK_NULLSPACE_GAIN * (q_nom - q))

            step = np.max(np.abs(dq))
            if step > IK_MAX_STEP:
                dq *= IK_MAX_STEP / step
            q = np.clip(q + dq, self.q_min, self.q_max)

        return IKResult7DOF(q, False, pos_err, axis_err, max_iters)


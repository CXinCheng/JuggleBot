"""Shared helpers for the 2+1 three-ball feasibility study (single LEAP hand on the v6 G1)."""
import mujoco, numpy as np
from jugglebot.humanoid.model import (load_g1_dual_arm, Q_READY_ARM_RIGHT, Q_READY_ARM_LEFT,
                                      ARM_JOINTS_RIGHT, ARM_JOINTS_LEFT, BALL_RADIUS, BALL_SEAT_CENTER_OFFSET_M)
from jugglebot.humanoid.throwing import LeapFingers, LEAP_JOINTS

R = BALL_RADIUS


class HandRig:
    """One hand of the v6 model, arm frozen kinematically at the ready pose.

    Seat frame: x lateral, y toward fingertips, z out of the palm.
    Hand/arm joints are frozen except the 16 finger joints (position actuators, +-0.5 Nm)
    and `free` extra joints (e.g. wrist pitch for tilt experiments).
    """

    def __init__(self, side="right", n_balls=3, free_arm=()):
        self.side = side
        self.m = load_g1_dual_arm(n_balls=n_balls)
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        self.fingers = LeapFingers(m, side)
        self.prefix = "rh_" if side == "right" else "lh_"
        names, q = ((ARM_JOINTS_RIGHT, Q_READY_ARM_RIGHT) if side == "right" else (ARM_JOINTS_LEFT, Q_READY_ARM_LEFT))
        self.arm_names = names
        self.q_arm_ready = q.copy()
        for other_names, other_q in ((ARM_JOINTS_RIGHT, Q_READY_ARM_RIGHT), (ARM_JOINTS_LEFT, Q_READY_ARM_LEFT)):
            for j, v in zip(other_names, other_q):
                d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
        self.seat = m.site(("r_" if side == "right" else "l_") + "ball_seat").id
        self.balls = [(int(m.joint(f"ball{i}_free").qposadr[0]), int(m.joint(f"ball{i}_free").dofadr[0])) for i in range(n_balls)]
        self.ball_geoms = [m.geom(f"ball{i}").id for i in range(n_balls)]
        self.hand_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(self.prefix)}
        self.finger_q = m.jnt_qposadr[self.fingers.joint_ids]
        self.finger_v = m.jnt_dofadr[self.fingers.joint_ids]
        self.free_q = [int(m.jnt_qposadr[m.joint(n).id]) for n in free_arm]
        self.free_v = [int(m.jnt_dofadr[m.joint(n).id]) for n in free_arm]
        # Hold the robot with stiff position servos and full gravity compensation on every
        # non-ball body, so tilting `opt.gravity` is exactly an inertial-frame change for the hand.
        ball_bodies = {int(m.geom_bodyid[g]) for g in self.ball_geoms}
        for b in range(m.nbody):
            m.body_gravcomp[b] = 0.0 if b in ball_bodies else 1.0
        self.hold_acts = []
        for a in range(m.nu):
            if m.actuator(a).name.startswith(("rh_", "lh_")):
                continue
            kp = 2000.0 if "waist" in m.actuator(a).name else 1500.0
            m.actuator_gainprm[a, 0] = kp
            m.actuator_biasprm[a, 1] = -kp
            m.actuator_biasprm[a, 2] = -0.05 * kp ** 0.5 * 4 if m.actuator_biasprm[a, 2] == 0 else m.actuator_biasprm[a, 2] * np.sqrt(3)
            self.hold_acts.append(a)
        self.hold_ctrl = {}
        for a in self.hold_acts:
            j = m.actuator_trnid[a, 0]
            self.hold_ctrl[a] = float(d.qpos[m.jnt_qposadr[j]])
        mujoco.mj_forward(m, d)
        self.ctrl_lo = m.actuator_ctrlrange[self.fingers.act_indices, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[self.fingers.act_indices, 1].copy()
        self.q_init = d.qpos.copy()

    # ---- frames -------------------------------------------------------------
    def seat_frame(self):
        d = self.d
        return d.site_xpos[self.seat].copy(), d.site_xmat[self.seat].reshape(3, 3).copy()

    def to_world(self, local):
        p, Rm = self.seat_frame()
        return p + Rm @ np.asarray(local)

    def ball_local(self, i):
        p, Rm = self.seat_frame()
        return Rm.T @ (self.d.qpos[self.balls[i][0]:self.balls[i][0] + 3] - p)

    def ball_vel_local(self, i):
        _, Rm = self.seat_frame()
        return Rm.T @ self.d.qvel[self.balls[i][1]:self.balls[i][1] + 3]

    # ---- state --------------------------------------------------------------
    def reset(self, finger_q, ctrl, ball_local_positions, park=(-2.5, -2.5)):
        m, d = self.m, self.d
        d.qpos[:] = self.q_init
        d.qvel[:] = 0
        for a, v in self.hold_ctrl.items():
            d.ctrl[a] = v
        d.qpos[self.finger_q] = finger_q
        d.ctrl[self.fingers.act_indices] = ctrl
        mujoco.mj_forward(m, d)
        for i, (q0, v0) in enumerate(self.balls):
            if i < len(ball_local_positions) and ball_local_positions[i] is not None:
                d.qpos[q0:q0 + 3] = self.to_world(ball_local_positions[i])
            else:
                d.qpos[q0:q0 + 3] = [park[0], park[1] + .3 * i, R]
            d.qpos[q0 + 3:q0 + 7] = [1, 0, 0, 0]
        d.time = 0
        mujoco.mj_forward(m, d)

    def step(self, n=1):
        m, d = self.m, self.d
        for _ in range(n):
            mujoco.mj_step(m, d)

    def ball_hand_contacts(self, i):
        """Number of hand geoms touching ball i, and total normal force."""
        m, d = self.m, self.d
        g = self.ball_geoms[i]
        n, fsum, parts = 0, 0.0, set()
        f6 = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            if g in (con.geom1, con.geom2):
                other = con.geom2 if con.geom1 == g else con.geom1
                if m.geom_bodyid[other] in self.hand_bodies:
                    mujoco.mj_contactForce(m, d, c, f6)
                    n += 1; fsum += f6[0]; parts.add(m.body(m.geom_bodyid[other]).name[len(self.prefix):])
        return n, fsum, parts

    def ball_ball_contact_force(self, i, j):
        m, d = self.m, self.d
        f6 = np.zeros(6); tot = 0.0
        for c in range(d.ncon):
            con = d.contact[c]
            if {con.geom1, con.geom2} == {self.ball_geoms[i], self.ball_geoms[j]}:
                mujoco.mj_contactForce(m, d, c, f6); tot += f6[0]
        return tot

    def set_gravity_local(self, g_local):
        """Equivalent to a hand without rotation accelerating by a = g_world - g_eff.
        g_local: effective gravity expressed in the (ready-pose) seat frame."""
        _, Rm = self.seat_frame0
        self.m.opt.gravity[:] = Rm @ np.asarray(g_local)

    @property
    def seat_frame0(self):
        if not hasattr(self, "_sf0"):
            mujoco.mj_forward(self.m, self.d)  # ready pose frame (fingers do not move the seat)
            self._sf0 = self.seat_frame()
        return self._sf0

"""Throw primitives for the cup end-effector (v2).

The ball rests on the cup base; there is nothing to open. It leaves the cup on its
own as soon as the cup decelerates harder than gravity along the cup axis, i.e. when
the specific force at the ball seat, f = a_seat - g, stops pressing the ball into
the base (f . axis <= 0). The release velocity is the seat velocity at that instant.

Both primitives move the arm along a one-parameter path q(s) with one shared speed
profile s(t) (smoothstep in path speed): accelerate over t_acc, decelerate over
t_dec. s = 0 is the end of the acceleration phase, where the path speed peaks; the
ball actually leaves a few ms into the deceleration.

translate : the cup moves along a straight Cartesian line with a fixed orientation.
            Clean release (the ball is only pressed into the base), but slow: to
            keep the cup from turning, wrist_1 must cancel the forearm rotation, so
            with 180 deg/s joints both arm links turn at most pi rad/s.
whip      : straight line in joint space through the release configuration, using
            shoulder_pan and the three parallel pitch joints (shoulder_lift, elbow,
            wrist_1); wrist_2 / wrist_3 stay put. The cup turns and the pitch rates
            add up along the chain. But the ball seat follows a curved path, and the
            centripetal load pins the ball to the cup wall: it rides up the wall and
            leaves 30-50 ms late, slow and with a sideways kick (see README).
linewhip  : the whip's joints and speed, but the ball seat moves along a straight
            line (like translate) while the cup is free to turn: no centripetal load
            at the seat, only the cup's rotation tilts the axis against the load.

Release-point search (ThrowPlanner.plan): IK over a grid of candidate release points
(cup axis aligned with the specific force at peak acceleration), rank them by the
joint-speed ratio of the release velocity, then build and verify the full stroke for
the best ones: joint speed, feed-forward torque, joint range, servo command range,
collisions, ball kept on the base until release, ball clearing the rim afterwards.
The commanded release velocity is corrected a few times so that the *predicted*
release state hits the target.
"""

import copy
from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

from kinematics import ArmKinematics

WHIP_JOINTS = np.array([0, 1, 2, 3])  # shoulder_pan, shoulder_lift, elbow, wrist_1


@dataclass
class ThrowConfig:
    t_acc: float = 0.15  # s, acceleration phase (peak path acceleration 1.5 V / t_acc)
    t_dec: float = 0.12  # s, deceleration phase (peak 1.5 V / t_dec; release only needs > g, the
    #                      rest just stops the arm, and a harder stop mostly costs shoulder torque)
    v_limit: float = np.pi  # rad/s, UR5e datasheet: 180 deg/s on every joint
    vel_frac: float = 0.9  # planned joint speed <= vel_frac * v_limit
    torque_frac: float = 0.8  # feed-forward torque <= torque_frac * actuator limit
    joint_margin: float = 0.05  # rad, distance kept from joint range limits
    min_press: float = 2.0  # m/s^2, f . axis during acceleration (ball firmly on the base)
    clear_tol: float = 0.001  # m, slack of the rim-clearance check
    # Candidate release points (ball_seat positions), world frame
    grid_x: tuple = (0.0, 0.9)
    grid_y: tuple = (-0.7, 0.7)
    grid_z: tuple = (0.3, 0.9)
    grid_step: float = 0.1
    max_reach: float = 0.95  # m, prune: shoulder -> seat distance
    refine_step: float = 0.025  # m, local grid around the best candidates
    n_refine: int = 3
    n_verify: int = 40  # candidates (best first) for which the full stroke is built
    n_correct: int = 4  # release-velocity correction iterations
    dt: float = 0.001  # s, trajectory sampling


class SpeedProfile:
    """Path parameter s(t), in seconds at full speed; the path speed ds/dt rises 0 -> 1 -> 0."""

    def __init__(self, t_acc: float, t_dec: float):
        self.t_acc, self.t_dec = t_acc, t_dec
        self.s_min, self.s_max = -t_acc / 2, t_dec / 2

    def __call__(self, tau: float) -> tuple[float, float, float]:
        """(s, ds/dt, d2s/dt2) at tau = time since the end of the acceleration phase."""
        if tau <= -self.t_acc:
            return self.s_min, 0.0, 0.0
        if tau < 0.0:
            T = self.t_acc
            u = (tau + T) / T
            return -T / 2 + T * (u ** 3 - u ** 4 / 2), 3 * u ** 2 - 2 * u ** 3, (6 * u - 6 * u ** 2) / T
        if tau < self.t_dec:
            T = self.t_dec
            u = tau / T
            return T * (u - u ** 3 + u ** 4 / 2), 1 - 3 * u ** 2 + 2 * u ** 3, (-6 * u + 6 * u ** 2) / T
        return self.s_max, 0.0, 0.0


@dataclass
class Path:
    """Joint path sampled on a uniform grid of s, with dq/ds and d2q/ds2."""
    s: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray

    def at(self, s: float):
        f = np.clip((s - self.s[0]) / (self.s[1] - self.s[0]), 0, len(self.s) - 1)
        i = min(int(f), len(self.s) - 2)
        w = f - i
        lerp = lambda a: (1 - w) * a[i] + w * a[i + 1]
        return lerp(self.q), lerp(self.dq), lerp(self.ddq)


class PathTrajectory:
    """Trajectory segment (same interface as planning.py): path q(s) timed by a SpeedProfile."""

    def __init__(self, t0: float, profile: SpeedProfile, path: Path):
        self.t0, self.profile, self.path = t0, profile, path
        self.t_peak = t0 + profile.t_acc  # nominal release (peak speed)
        self.t1 = self.t_peak + profile.t_dec

    def sample(self, t):
        s, sd, sdd = self.profile(t - self.t_peak)
        q, dq, ddq = self.path.at(s)
        return q, dq * sd, ddq * sd ** 2 + dq * sdd


@dataclass
class ThrowPlan:
    mode: str
    q_release: np.ndarray  # configuration at the nominal release point (s = 0)
    qd_peak: np.ndarray  # joint velocity there
    path: Path
    profile: SpeedProfile
    t_release: float  # predicted release time, relative to the stroke start
    p_release: np.ndarray  # predicted ball state at release
    v_release: np.ndarray
    v_target: np.ndarray
    speed_ratio: float  # max |qd| / v_limit over the stroke
    torque_ratio: float  # max |tau_ff| / actuator limit over the stroke
    min_press: float  # min f . axis during acceleration (m/s^2)
    lateral_ratio: float  # max |f_perp| / (f . axis) during acceleration (> friction -> the ball rolls)
    exit_angle: float  # angle between cup axis and release velocity (rad)
    axis: np.ndarray  # cup axis at the nominal release point
    v_cmd: np.ndarray = None  # commanded seat velocity at the nominal release point
    calibrated: bool = False  # release state measured in simulation (calibration.py), not predicted

    @property
    def duration(self) -> float:
        return self.profile.t_acc + self.profile.t_dec

    def trajectory(self, t0: float) -> PathTrajectory:
        return PathTrajectory(t0, self.profile, self.path)

    @property
    def q_start(self) -> np.ndarray:
        return self.path.q[0]

    @property
    def q_end(self) -> np.ndarray:
        return self.path.q[-1]


def specific_force_axis(v_release: np.ndarray, a_peak: float, gravity: np.ndarray) -> np.ndarray:
    """Cup axis along the specific force at peak acceleration, so the ball sits centred on the base."""
    f = a_peak * v_release / np.linalg.norm(v_release) - gravity
    return f / np.linalg.norm(f)


def minimax_nullspace(u: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """min over r of max_i |u_i + r n_i| (convex, piecewise linear): try all kinks. Returns (r, value)."""
    rs = [0.0]
    for i in range(len(u)):
        for j in range(len(u)):
            for sgn in (1.0, -1.0):  # u_i + r n_i = sgn (u_j + r n_j)
                den = n[i] - sgn * n[j]
                if abs(den) > 1e-12:
                    rs.append((sgn * u[j] - u[i]) / den)
    rs = np.array(rs)
    vals = np.max(np.abs(u[None, :] + rs[:, None] * n[None, :]), axis=1)
    k = int(np.argmin(vals))
    return float(rs[k]), float(vals[k])


class InverseDynamics:
    """tau = M(q) qdd + c(q, qd) for the arm (gravity is cancelled by gravcomp).

    Runs on a contact-free copy of the model: contact forces of a colliding pose would
    otherwise show up as (huge) joint torques. Collisions are checked separately.
    """

    def __init__(self, model: mujoco.MjModel, kin: ArmKinematics):
        self.model = copy.deepcopy(model)
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.kin = kin
        self.data = mujoco.MjData(self.model)

    def __call__(self, q, qd, qdd) -> np.ndarray:
        d = self.data
        d.qpos[self.kin.qpos_adr] = q
        d.qvel[:] = 0.0
        d.qvel[self.kin.dof_adr] = qd
        d.qacc[:] = 0.0
        d.qacc[self.kin.dof_adr] = qdd
        mujoco.mj_inverse(self.model, d)
        return d.qfrc_inverse[self.kin.dof_adr].copy()


@dataclass
class Rejections:
    """Why candidates were rejected (diagnostics for infeasible targets)."""
    counts: dict = field(default_factory=dict)

    def add(self, reason: str) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1

    def __str__(self) -> str:
        return ", ".join(f"{k} {v}" for k, v in sorted(self.counts.items(), key=lambda kv: -kv[1])) or "-"


class ThrowPlanner:
    def __init__(self, model: mujoco.MjModel, kin: ArmKinematics, cfg: ThrowConfig, q_nominal: np.ndarray):
        self.model, self.kin, self.cfg, self.q_nominal = model, kin, cfg, q_nominal
        self.gravity = model.opt.gravity.copy()
        self.idyn = InverseDynamics(model, kin)
        self.seat = model.site("ball_seat").id
        self.mouth = model.site("cup_mouth").id
        self.ball_radius = float(model.geom_size[model.geom("ball").id][0])
        # Cup cavity relative to the ball seat (cup frame z = cup axis).
        walls = [g for g in range(model.ngeom) if model.geom(g).name.startswith("cup_wall")]
        self.inner_radius = min(np.linalg.norm(model.geom_pos[g][:2]) - model.geom_size[g][0] for g in walls)
        seat_z = model.site_pos[self.seat][2]
        self.rim_height = min(model.geom_pos[g][2] + model.geom_size[g][2] for g in walls) - seat_z
        self.mouth_offset = model.site_pos[self.mouth][2] - seat_z
        self.torque_limit = model.actuator_forcerange[:, 1].copy()
        self.kp = model.actuator_gainprm[:, 0].copy()
        self.kd = -model.actuator_biasprm[:, 2].copy()
        self.ctrl_lo, self.ctrl_hi = model.actuator_ctrlrange[:, 0].copy(), model.actuator_ctrlrange[:, 1].copy()
        mujoco_data = mujoco.MjData(model)
        mujoco.mj_forward(model, mujoco_data)
        self.shoulder = mujoco_data.body("shoulder_link").xpos.copy()
        self.profile = SpeedProfile(cfg.t_acc, cfg.t_dec)
        self.last_rejections = Rejections()

    # ------------------------------------------------------------ velocity solutions
    def release_velocity(self, mode: str, q: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Joint velocity at q giving the ball seat velocity v, with the smallest max |qd_i|.

        Returns (qd, r, n): the solution is u + r n with u the min-norm solution and n the
        1-D null-space direction of the task (needed to continue the translate path).
        """
        jp, jr = self.kin.jacobian(q, self.seat)
        if mode == "translate":
            axis = self.kin.fk(q, self.seat)[1]
            jac = np.vstack([jp, (np.eye(3) - np.outer(axis, axis)) @ jr])  # rank 5: turning about the axis is free
            task = np.concatenate([v, np.zeros(3)])
            u = np.linalg.pinv(jac, rcond=1e-6) @ task
            n = np.linalg.svd(jac)[2][-1]
        elif mode in ("whip", "linewhip"):
            jac = jp[:, WHIP_JOINTS]
            u4 = np.linalg.pinv(jac, rcond=1e-6) @ v
            n4 = np.linalg.svd(jac)[2][-1]
            u, n = np.zeros(6), np.zeros(6)
            u[WHIP_JOINTS], n[WHIP_JOINTS] = u4, n4
        else:
            raise ValueError(mode)
        r, _ = minimax_nullspace(u, n)
        return u + r * n, r, n

    # ------------------------------------------------------------ paths
    def _path(self, mode: str, q_r: np.ndarray, v_cmd: np.ndarray) -> Path | None:
        cfg = self.cfg
        h = cfg.dt  # path step in s (1 ms at full speed)
        s_grid = np.arange(np.floor(self.profile.s_min / h), np.ceil(self.profile.s_max / h) + 1) * h
        i0 = int(round(-s_grid[0] / h))  # index of s = 0
        qd_r, r, n_r = self.release_velocity(mode, q_r, v_cmd)
        if mode == "whip":
            q = q_r[None, :] + s_grid[:, None] * qd_r[None, :]
            dq = np.repeat(qd_r[None, :], len(s_grid), axis=0)
            return Path(s_grid, q, dq, np.zeros_like(q))

        # translate / linewhip: integrate dq/ds = f(q) = J+ v + r n(q) (RK4) outwards from s = 0,
        # keeping the release solution's null-space share r. J f = v: the seat moves along a straight
        # line at speed |v|; for translate the task also holds the cup axis still.
        axis = self.kin.fk(q_r, self.seat)[1]
        n_ref = [n_r]

        def f(q):
            jp, jr = self.kin.jacobian(q, self.seat)
            if mode == "translate":
                jac = np.vstack([jp, (np.eye(3) - np.outer(axis, axis)) @ jr])
                n = np.linalg.svd(jac)[2][-1]
                qd = np.linalg.pinv(jac, rcond=1e-6) @ np.concatenate([v_cmd, np.zeros(3)])
            else:
                jac = jp[:, WHIP_JOINTS]
                n, qd = np.zeros(6), np.zeros(6)
                n[WHIP_JOINTS] = np.linalg.svd(jac)[2][-1]
                qd[WHIP_JOINTS] = np.linalg.pinv(jac, rcond=1e-6) @ v_cmd
            n = n if n @ n_ref[0] >= 0 else -n  # keep the null-space direction continuous
            n_ref[0] = n
            return qd + r * n

        q = np.zeros((len(s_grid), 6))
        dq = np.zeros_like(q)
        q[i0] = q_r
        for direction, idx in ((1, range(i0, len(s_grid) - 1)), (-1, range(i0, 0, -1))):
            n_ref[0] = n_r
            for i in idx:
                step = direction * h
                k1 = f(q[i])
                dq[i] = k1
                k2 = f(q[i] + step / 2 * k1)
                k3 = f(q[i] + step / 2 * k2)
                k4 = f(q[i] + step * k3)
                q[i + direction] = q[i] + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            dq[idx[-1] + direction] = f(q[idx[-1] + direction])
        ddq = np.gradient(dq, h, axis=0)
        return Path(s_grid, q, dq, ddq)

    # ------------------------------------------------------------ verification
    def _evaluate(self, mode: str, path: Path, axis_r: np.ndarray, v_target_fn, rej: Rejections,
                  check_release: bool = True) -> ThrowPlan | None:
        """Sample the stroke, check all constraints, predict the release. None if infeasible.

        check_release=False skips the checks that rest on the analytic release prediction (ball held,
        early release, rim clearance) -- for calibration, where a simulated rollout replaces them.
        """
        cfg, kin = self.cfg, self.kin
        traj = PathTrajectory(0.0, self.profile, path)
        ts = np.arange(0.0, traj.t1 + 1e-9, cfg.dt)
        Q, QD, QDD = (np.array(x) for x in zip(*(traj.sample(t) for t in ts)))

        speed_ratio = float(np.max(np.abs(QD)) / cfg.v_limit)
        if speed_ratio > cfg.vel_frac + 1e-6:
            rej.add("joint speed")
            return None
        if np.any(Q < kin.q_min + cfg.joint_margin) or np.any(Q > kin.q_max - cfg.joint_margin):
            rej.add("joint range")
            return None
        for k in list(range(0, len(ts), 10)) + [len(ts) - 1]:
            if not kin.is_collision_free(Q[k]):
                rej.add("collision")
                return None

        pos, axes, vel = np.zeros((len(ts), 3)), np.zeros((len(ts), 3)), np.zeros((len(ts), 3))
        tau = np.zeros((len(ts), 6))
        for k in range(len(ts)):
            jp, _ = kin.jacobian(Q[k], self.seat)
            pos[k], axes[k] = kin.fk(Q[k], self.seat)
            vel[k] = jp @ QD[k]
            tau[k] = self.idyn(Q[k], QD[k], QDD[k])
        torque_ratio = float(np.max(np.abs(tau) / self.torque_limit))
        if torque_ratio > cfg.torque_frac:
            rej.add("torque")
            return None
        ctrl = Q + (self.kd / self.kp) * QD + tau / self.kp  # what the servo will be commanded
        if np.any(ctrl < self.ctrl_lo) or np.any(ctrl > self.ctrl_hi):
            rej.add("servo command range")
            return None

        # Release: first instant the specific force stops pressing the ball onto the base.
        acc = np.gradient(vel, cfg.dt, axis=0)
        f = acc - self.gravity
        press = np.einsum("ij,ij->i", f, axes)
        released = np.nonzero(press <= 0.0)[0]
        k_peak = int(round(self.profile.t_acc / cfg.dt))  # end of the acceleration phase
        if len(released) == 0:
            if check_release:
                rej.add("no release")
                return None
            released = [k_peak]
        k_rel = int(released[0])
        accel = slice(0, k_peak)  # (press falls to 0 at release by definition; check only up to the peak)
        if check_release and k_rel < k_peak:
            rej.add("early release")  # e.g. centrifugal pull of a turning cup
            return None
        if check_release and np.min(press[accel]) < cfg.min_press:
            rej.add("ball not held")
            return None
        f_perp = f[accel] - press[accel, None] * axes[accel]
        lateral = float(np.max(np.linalg.norm(f_perp, axis=1) / press[accel]))
        p_rel, v_rel, t_rel = pos[k_rel], vel[k_rel], ts[k_rel]

        # After release the ball must leave through the opening without touching the rim.
        if check_release and not self._clears_rim(ts, pos, axes, k_rel, p_rel, v_rel, t_rel):
            rej.add("rim")
            return None

        v_target = v_target_fn(p_rel)
        exit_angle = float(np.arccos(np.clip(v_rel @ axes[k_rel] / np.linalg.norm(v_rel), -1, 1)))
        i0 = int(np.argmin(np.abs(path.s)))
        return ThrowPlan(mode, path.q[i0].copy(), path.dq[i0].copy(), path, self.profile, t_rel,
                         p_rel.copy(), v_rel.copy(), v_target, speed_ratio, torque_ratio,
                         float(np.min(press[accel])), lateral, exit_angle, axis_r.copy())

    def _clears_rim(self, ts, pos, axes, k_rel, p_rel, v_rel, t_rel) -> bool:
        """Ball centre (ballistic from release) vs the cup cavity: it must not touch the wall or rim
        on its way out, nor be caught up by the base again. A ball that never rises fully above the
        rim within the window is fine as long as it stays inside the opening (a near-vertical throw
        falls back into the same cup: the cup follows it up for ~10 cm before it stops)."""
        cfg, r = self.cfg, self.ball_radius
        t_extra = np.arange(ts[-1] + cfg.dt, ts[-1] + 0.15, cfg.dt)  # the cup holds its final pose
        t_all = np.concatenate([ts[k_rel:], t_extra])
        seat = np.vstack([pos[k_rel:], np.repeat(pos[-1:], len(t_extra), axis=0)])
        ax = np.vstack([axes[k_rel:], np.repeat(axes[-1:], len(t_extra), axis=0)])
        dt = t_all - t_rel
        ball = p_rel + v_rel * dt[:, None] + 0.5 * self.gravity * dt[:, None] ** 2
        rel = ball - seat
        h = np.einsum("ij,ij->i", rel, ax)
        radial = np.linalg.norm(rel - h[:, None] * ax, axis=1)
        for hk, rk in zip(h, radial):
            if hk >= self.rim_height + r:
                return True  # fully above the rim
            if hk < -cfg.clear_tol or rk > self.inner_radius - r + cfg.clear_tol:
                return False
        return True

    # ------------------------------------------------------------ search
    def _candidates(self, axis: np.ndarray, points: np.ndarray, rej: Rejections):
        """IK for each candidate seat point; returns [(q, p)] of collision-free solutions."""
        out = []
        seed = self.q_nominal
        for p in points:
            if np.linalg.norm(p - self.shoulder) > self.cfg.max_reach:
                rej.add("out of reach")
                continue
            ik = self.kin.solve(p + self.mouth_offset * axis, axis, seed, self.q_nominal)
            if not ik.success:
                rej.add("ik")
                continue
            if not self.kin.is_collision_free(ik.q):
                rej.add("collision")
                continue
            out.append((ik.q, p))
        return out

    def _grid(self, center=None, half=None, step=None) -> np.ndarray:
        cfg = self.cfg
        if center is None:
            axes = [np.arange(lo, hi + 1e-9, cfg.grid_step) for lo, hi in (cfg.grid_x, cfg.grid_y, cfg.grid_z)]
        else:
            axes = [np.arange(c - half, c + half + 1e-9, step) for c in center]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    def plan(self, mode: str, v_target_fn: Callable[[np.ndarray], np.ndarray],
             v_nominal: np.ndarray) -> ThrowPlan | None:
        """Best feasible throw. v_target_fn(p_release) -> required release velocity; v_nominal
        (the target velocity for a typical release point) fixes the cup axis."""
        cfg = self.cfg
        rej = Rejections()
        self.last_rejections = rej
        a_peak = 1.5 * np.linalg.norm(v_nominal) / cfg.t_acc
        axis = specific_force_axis(v_nominal, a_peak, self.gravity)

        def ratio(q, p):
            """Cheap ranking: joint-speed ratio of the release velocity alone; inf if the stroke ends
            (linearised: q +- qd * s) collide. The fastest poses are low and stretched out, and their
            acceleration stroke tends to run into the floor."""
            qd = self.release_velocity(mode, q, v_target_fn(p))[0]
            for s in (self.profile.s_min, self.profile.s_max):
                if not self.kin.is_collision_free(q + qd * s):
                    rej.add("stroke collision")
                    return np.inf
            return np.max(np.abs(qd)) / cfg.v_limit

        cands = [(ratio(q, p), q, p) for q, p in self._candidates(axis, self._grid(), rej)]
        cands = sorted((c for c in cands if np.isfinite(c[0])), key=lambda c: c[0])
        # Refine around the best few: the fastest poses sit at the edge of the workspace.
        local = []
        for _, _, p in cands[:cfg.n_refine]:
            pts = self._grid(p, cfg.grid_step / 2, cfg.refine_step)
            local += [(ratio(q, pp), q, pp) for q, pp in self._candidates(axis, pts, rej)]
        cands = sorted((c for c in cands + local if np.isfinite(c[0])), key=lambda c: c[0])

        for rho, q_r, p in cands[:cfg.n_verify]:
            if rho > cfg.vel_frac:
                rej.add("joint speed")
                break  # sorted: the rest is slower still
            plan = None
            v_cmd = v_target_fn(p).copy()
            for _ in range(cfg.n_correct):
                plan = self.build(mode, q_r, v_cmd, axis, v_target_fn, rej)
                if plan is None:
                    break
                err = plan.v_target - plan.v_release
                if np.linalg.norm(err) < 0.005:
                    break
                v_cmd = v_cmd + err
            if plan is not None:
                return plan
        return None

    def build(self, mode: str, q_r: np.ndarray, v_cmd: np.ndarray, axis: np.ndarray,
              v_target_fn: Callable[[np.ndarray], np.ndarray], rej: Rejections | None = None,
              check_release: bool = True) -> ThrowPlan | None:
        """Stroke through the release configuration q_r with commanded seat velocity v_cmd; None if
        any constraint fails. Used by plan() and by the simulation-based calibration."""
        rej = Rejections() if rej is None else rej
        path = self._path(mode, q_r, v_cmd)
        plan = self._evaluate(mode, path, axis, v_target_fn, rej, check_release) if path is not None else None
        if plan is not None:
            plan.v_cmd = np.asarray(v_cmd, dtype=float).copy()
        return plan

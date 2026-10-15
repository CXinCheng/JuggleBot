"""Target selection (step 2) and motion planning (step 3) of the catching pipeline.

A plan consists of three phases, all expressed as one joint-space trajectory:

    approach : joint-space quintic from the current desired state to the
               pre-catch pose q_pre, then hold there (arrive early)
    match    : straight Cartesian line p_pre -> p_c, accelerating from rest so
               the cup moves along the ball direction with speed u at t_c
    cushion  : straight line p_c -> p_c + d_cush * v_hat, decelerating to rest

The "static" strategy skips match/cushion: the cup waits motionless at p_c.

v2: an optional torque_ok(segment) -> bool filter on the approach quintic, so the
joint acceleration limits can be raised for the short juggling flights without
planning motions the actuators cannot follow.
v4: PlannerConfig.cushion_dir = "down" runs the match / cushion lines straight down
instead of along the incoming ball (the hand gives way under the ball).
v4 E5: strategy "direct" -- no rest before the catch: one quintic from the current (possibly
moving) state straight to the catch pose, arriving with the cushion speed along the ball, then
the cushion line. For catches right after a throw, when the arm is still moving fast the other way.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .kinematics import ArmKinematics
from .prediction import BallisticPredictor


@dataclass
class PlannerConfig:
    strategy: str = "cushion"  # "cushion" or "static"
    # Catch-point constraints
    z_min: float = 0.40  # m, lowest catch height (cup_mouth); the wrist hangs ~0.2 m below it
    z_max: float = 0.85  # m
    z_min_end: float = 0.33  # m, lowest point the cushion may push the cup to
    max_reach: float = 0.95  # m, coarse prune: distance shoulder -> cup_mouth
    max_tilt: float = np.deg2rad(40.0)  # max angle between cup axis and vertical
    # Joint limits used for timing (UR5e: 180 deg/s on every joint)
    v_max: np.ndarray = field(default_factory=lambda: np.full(6, np.pi))  # rad/s
    a_max: np.ndarray = field(default_factory=lambda: np.array([15.0, 15.0, 15.0, 25.0, 25.0, 25.0]))  # rad/s^2
    min_wait: float = 0.10  # s the cup must rest at q_pre before the match phase starts
    earliest_catch: float = 0.15  # s, candidates start this far after "now"
    candidate_dt: float = 0.01  # s between candidate catch times
    # Cushion parameters
    alpha: float = 0.4  # cup speed at t_c as a fraction of the ball speed
    cup_speed_max: float = 0.7  # m/s (needs up to ~4 rad/s per m/s at the wrist)
    d_pre: float = 0.05  # m, match-phase length (upstream of p_c)
    d_cush: float = 0.10  # m, cushion-phase length (downstream of p_c)
    # Direction of the match / cushion lines: "ball" (v1: along the incoming ball) or "down"
    # (v4: straight down, the hand gives way under the ball like a human catch)
    cushion_dir: str = "ball"
    dense_dt: float = 0.001  # s, resolution of the IK-resolved Cartesian phases
    # Cost weights
    joint_weights: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.2]))
    tilt_weight: float = 0.5  # per rad
    # direct strategy: weight of the (weighted) joint distance from the catch pose to CatchPlanner.q_next
    # (e.g. the next throw's stroke start) in the cost -- catch where the carry afterwards is short
    next_weight: float = 0.0


# ---------------------------------------------------------------------------
# Trajectory primitives. Every segment answers sample(t) -> (q, qd, qdd).
# ---------------------------------------------------------------------------
class Quintic:
    """Quintic polynomial between full (pos, vel, acc) boundary states, vectorised over dims."""

    def __init__(self, t0, duration, x0, v0, a0, x1, v1, a1):
        self.t0, self.t1, self.T = t0, t0 + duration, duration
        T = duration
        x0, v0, a0, x1, v1, a1 = (np.asarray(v, dtype=float) for v in (x0, v0, a0, x1, v1, a1))
        dx = x1 - x0
        self.c = np.array([
            x0,
            v0,
            a0 / 2,
            (20 * dx - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * T ** 2) / (2 * T ** 3),
            (-30 * dx + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * T ** 2) / (2 * T ** 4),
            (12 * dx - 6 * (v1 + v0) * T + (a1 - a0) * T ** 2) / (2 * T ** 5),
        ])

    def sample(self, t):
        s = np.clip(t - self.t0, 0.0, self.T)
        c = self.c
        x = c[0] + s * (c[1] + s * (c[2] + s * (c[3] + s * (c[4] + s * c[5]))))
        v = c[1] + s * (2 * c[2] + s * (3 * c[3] + s * (4 * c[4] + s * 5 * c[5])))
        a = 2 * c[2] + s * (6 * c[3] + s * (12 * c[4] + s * 20 * c[5]))
        return x, v, a

    def peaks(self, n=60):
        """Max |vel| and |acc| over the segment, per dim (sampled)."""
        ts = self.t0 + np.linspace(0, self.T, n)
        vs, accs = zip(*[self.sample(t)[1:] for t in ts])
        return np.max(np.abs(vs), axis=0), np.max(np.abs(accs), axis=0)


class Hold:
    def __init__(self, t0, t1, q):
        self.t0, self.t1, self.q = t0, t1, np.asarray(q, dtype=float)

    def sample(self, t):
        z = np.zeros_like(self.q)
        return self.q, z, z


class Dense:
    """Joint trajectory sampled on a uniform grid, linearly interpolated."""

    def __init__(self, t0, dt, qs):
        self.t0, self.dt, self.qs = t0, dt, np.asarray(qs)
        self.t1 = t0 + dt * (len(qs) - 1)
        self.qds = np.gradient(self.qs, dt, axis=0)
        self.qdds = np.gradient(self.qds, dt, axis=0)

    def sample(self, t):
        f = np.clip((t - self.t0) / self.dt, 0, len(self.qs) - 1)
        i = min(int(f), len(self.qs) - 2)
        w = f - i
        lerp = lambda arr: (1 - w) * arr[i] + w * arr[i + 1]
        return lerp(self.qs), lerp(self.qds), lerp(self.qdds)


class Trajectory:
    """Piecewise trajectory; holds the last segment's end state forever."""

    def __init__(self, segments):
        self.segments = segments

    @property
    def t_end(self) -> float:
        """Time after which the arm is commanded to stand still."""
        last = self.segments[-1]
        return last.t0 if isinstance(last, Hold) else last.t1

    def sample(self, t):
        for seg in self.segments:
            if t < seg.t1:
                return seg.sample(t)
        q, _, _ = self.segments[-1].sample(self.segments[-1].t1)
        z = np.zeros_like(q)
        return q, z, z


def min_time_quintic(t0, q0, v0, a0, q1, v_max, a_max) -> Quintic:
    """Shortest quintic (to rest at q1) that respects joint velocity / acceleration limits."""
    dq = np.abs(q1 - q0)
    # Rest-to-rest bounds: peak vel = 1.875 dq/T, peak acc = 5.77 dq/T^2.
    T = max(0.05, np.max(1.875 * dq / v_max), np.max(np.sqrt(5.77 * dq / a_max)))
    zero = np.zeros_like(q0)
    # When replanning mid-motion the start state itself may sit at a limit; tolerate it.
    v_lim = np.maximum(v_max, np.abs(v0)) * 1.01
    a_lim = np.maximum(a_max, np.abs(a0)) * 1.01
    for _ in range(60):
        seg = Quintic(t0, T, q0, v0, a0, q1, zero, zero)
        vp, ap = seg.peaks()
        if np.all(vp <= v_lim) and np.all(ap <= a_lim):
            return seg
        T *= 1.05
    return seg


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
@dataclass
class CatchPlan:
    t_catch: float
    p_catch: np.ndarray  # cup_mouth (= predicted ball centre) at t_catch
    v_ball: np.ndarray  # predicted ball velocity at t_catch
    axis: np.ndarray  # cup axis during the catch
    cup_speed: float  # cup speed along v_hat at t_catch
    q_pre: np.ndarray
    q_end: np.ndarray
    t_match: float  # start of the match phase (end of the waiting period)
    trajectory: Trajectory
    n_candidates: int = 0
    n_feasible: int = 0


def cup_axis_for(v_ball: np.ndarray, max_tilt: float) -> np.ndarray:
    """Cup opening faces the incoming ball (-v_hat), tilted at most max_tilt from vertical."""
    a = -v_ball / np.linalg.norm(v_ball)
    tilt = np.arccos(np.clip(a[2], -1, 1))
    if tilt <= max_tilt:
        return a
    horiz = a[:2] / max(np.linalg.norm(a[:2]), 1e-9)
    return np.array([np.sin(max_tilt) * horiz[0], np.sin(max_tilt) * horiz[1], np.cos(max_tilt)])


class CatchPlanner:
    def __init__(self, kin: ArmKinematics, cfg: PlannerConfig, q_nominal: np.ndarray,
                 shoulder_pos: np.ndarray, torque_ok=None):
        self.kin, self.cfg, self.q_nominal, self.shoulder = kin, cfg, q_nominal, shoulder_pos
        self.torque_ok = torque_ok
        self.rejections = Counter()  # direct strategy: why candidates were dropped in the last plan()
        self.q_next = None  # direct strategy: see PlannerConfig.next_weight

    def _phase_geometry(self, v_ball, cushion: bool):
        cfg = self.cfg
        if not cushion:
            return 0.0, 0.0, 0.0
        u = min(cfg.alpha * np.linalg.norm(v_ball), cfg.cup_speed_max)
        return u, cfg.d_pre, cfg.d_cush

    def plan(self, pred: BallisticPredictor, t_now: float, state: tuple,
             min_wait: float | None = None) -> CatchPlan | None:
        """Select a catch target and build the trajectory.

        state = desired (q, qd, qdd) at t_now, so a replan continues smoothly from
        wherever the current trajectory is. With the cushion strategy, a throw for which
        no cushioned catch is feasible falls back to a static catch.
        """
        min_wait = self.cfg.min_wait if min_wait is None else min_wait
        if self.cfg.strategy == "direct":
            return self._plan_direct(pred, t_now, state)
        for cushion in ((True, False) if self.cfg.strategy == "cushion" else (False,)):
            plan = self._plan(pred, t_now, state, min_wait, cushion)
            if plan is not None:
                return plan
        return None

    def _plan(self, pred, t_now, state, min_wait, cushion) -> CatchPlan | None:
        cfg = self.cfg
        q_now, v_now, a_now = state

        t_low = pred.time_at_height(cfg.z_min)
        if t_low is None:
            return None
        times = np.arange(t_now + cfg.earliest_catch, t_low + 1e-9, cfg.candidate_dt)

        seed = q_now
        candidates = []
        for t in times:
            p_c, v_b = pred.predict(t)
            if v_b[2] >= 0 or not cfg.z_min <= p_c[2] <= cfg.z_max:
                continue
            v_hat = self._line_dir(v_b)
            axis = cup_axis_for(v_b, cfg.max_tilt)
            u, d_pre, d_cush = self._phase_geometry(v_b, cushion)
            p_pre, p_end = p_c - d_pre * v_hat, p_c + d_cush * v_hat
            if p_end[2] < cfg.z_min_end or np.linalg.norm(p_pre - self.shoulder) > cfg.max_reach:
                continue
            t_match = t - (2 * d_pre / u if u > 0 else 0.0)

            ik = self.kin.solve(p_pre, axis, seed, self.q_nominal)
            if not ik.success or not self.kin.is_collision_free(ik.q):
                continue
            seed = ik.q  # neighbouring candidates have similar solutions
            approach = min_time_quintic(t_now, q_now, v_now, a_now, ik.q, cfg.v_max, cfg.a_max)
            if approach.t1 + min_wait > t_match:
                continue  # cannot get there (and settle) in time
            if self.torque_ok is not None and not self.torque_ok(approach):
                continue
            tilt = np.arccos(np.clip(axis[2], -1, 1))
            cost = np.linalg.norm(cfg.joint_weights * (ik.q - q_now)) + cfg.tilt_weight * tilt
            candidates.append((cost, t, p_c, v_b, axis, u, d_pre, d_cush, t_match, ik.q, approach))

        candidates.sort(key=lambda c: c[0])
        for cost, t, p_c, v_b, axis, u, d_pre, d_cush, t_match, q_pre, approach in candidates:
            segments = [approach, Hold(approach.t1, t_match, q_pre)]
            q_end = q_pre
            if u > 0:
                dense = self._cartesian_phases(t, p_c, v_b, axis, u, d_pre, d_cush, q_pre)
                if dense is None:
                    continue
                q_end = dense.qs[-1]
                segments += [dense, Hold(dense.t1, np.inf, q_end)]
            else:
                segments[-1] = Hold(approach.t1, np.inf, q_pre)
            return CatchPlan(t, p_c, v_b, axis, u, q_pre, q_end, t_match, Trajectory(segments),
                             len(times), len(candidates))
        return None

    def _plan_direct(self, pred, t_now, state) -> CatchPlan | None:
        cfg = self.cfg
        q_now, v_now, a_now = state
        rej = self.rejections = Counter()
        t_low = pred.time_at_height(cfg.z_min)
        if t_low is None:
            rej["below z_min"] += 1
            return None
        times = np.arange(t_now + cfg.earliest_catch, t_low + 1e-9, cfg.candidate_dt)
        seed = q_now
        candidates = []
        for t in times:
            p_c, v_b = pred.predict(t)
            if v_b[2] >= 0 or not cfg.z_min <= p_c[2] <= cfg.z_max:
                rej["rising / height"] += 1
                continue
            v_hat = self._line_dir(v_b)
            axis = cup_axis_for(v_b, cfg.max_tilt)
            u = min(cfg.alpha * np.linalg.norm(v_b), cfg.cup_speed_max)
            if (p_c + cfg.d_cush * v_hat)[2] < cfg.z_min_end or np.linalg.norm(p_c - self.shoulder) > cfg.max_reach:
                rej["reach / cushion height"] += 1
                continue
            ik = self.kin.solve(p_c, axis, seed, self.q_nominal)
            if not ik.success or not self.kin.is_collision_free(ik.q):
                rej["IK / collision"] += 1
                continue
            seed = ik.q
            jp, jr = self.kin.jacobian(ik.q)
            jac = np.vstack([jp, (np.eye(3) - np.outer(axis, axis)) @ jr])  # keep the axis, as the cushion line does
            qd_c = np.linalg.pinv(jac, rcond=1e-6) @ np.concatenate([u * v_hat, np.zeros(3)])
            approach = Quintic(t_now, t - t_now, q_now, v_now, a_now, ik.q, qd_c, np.zeros_like(q_now))
            vp, ap = approach.peaks()
            if np.any(vp > cfg.v_max):
                rej["joint speed"] += 1
                continue
            if np.any(ap > cfg.a_max):
                rej["joint acceleration"] += 1
                continue
            if self.torque_ok is not None and not self.torque_ok(approach):
                rej["torque"] += 1
                continue
            tilt = np.arccos(np.clip(axis[2], -1, 1))
            cost = np.linalg.norm(cfg.joint_weights * (ik.q - q_now)) + cfg.tilt_weight * tilt
            if self.q_next is not None and cfg.next_weight:
                cost += cfg.next_weight * np.linalg.norm(cfg.joint_weights * (ik.q - self.q_next))
            candidates.append((cost, t, p_c, v_b, axis, u, ik.q, approach))
        candidates.sort(key=lambda c: c[0])
        for cost, t, p_c, v_b, axis, u, q_c, approach in candidates:
            dense = self._cushion_line(t, p_c, v_b, axis, u, q_c)
            if dense is None:
                rej["cushion line"] += 1
                continue
            q_end = dense.qs[-1]
            segments = [approach, dense, Hold(dense.t1, np.inf, q_end)]
            return CatchPlan(t, p_c, v_b, axis, u, q_c, q_end, t, Trajectory(segments), len(times), len(candidates))
        return None

    def _cushion_line(self, t_c, p_c, v_b, axis, u, q_c) -> Dense | None:
        """IK-resolve the cushion line from p_c (speed u along the ball) to rest d_cush further on."""
        cfg = self.cfg
        v_hat = self._line_dir(v_b)
        t_k = 2 * cfg.d_cush / u
        cushion = Quintic(t_c, t_k, 0.0, u, 0.0, cfg.d_cush, 0.0, 0.0)
        qs, q = [], q_c
        for t in np.arange(t_c, t_c + t_k + 1e-9, cfg.dense_dt):
            ik = self.kin.solve(p_c + cushion.sample(t)[0] * v_hat, axis, q, None, max_iters=30)
            if not ik.success:
                return None
            q = ik.q
            qs.append(q)
        dense = Dense(t_c, cfg.dense_dt, qs)
        if np.any(np.max(np.abs(dense.qds), axis=0) > cfg.v_max * 1.05):
            return None
        if not self.kin.is_collision_free(dense.qs[-1]):
            return None
        return dense

    def _line_dir(self, v_b: np.ndarray) -> np.ndarray:
        if self.cfg.cushion_dir == "down":
            return np.array([0.0, 0.0, -1.0])
        return v_b / np.linalg.norm(v_b)

    def _cartesian_phases(self, t_c, p_c, v_b, axis, u, d_pre, d_cush, q_pre) -> Dense | None:
        """IK-resolve the match + cushion lines into a dense joint trajectory."""
        cfg = self.cfg
        v_hat = self._line_dir(v_b)
        t_m, t_k = 2 * d_pre / u, 2 * d_cush / u  # average speed u/2 in each phase
        # Arc length s along the line, s = 0 at p_pre. Zero acceleration at the joints.
        match = Quintic(t_c - t_m, t_m, 0.0, 0.0, 0.0, d_pre, u, 0.0)
        cushion = Quintic(t_c, t_k, d_pre, u, 0.0, d_pre + d_cush, 0.0, 0.0)
        ts = np.arange(t_c - t_m, t_c + t_k + 1e-9, cfg.dense_dt)
        p_pre = p_c - d_pre * v_hat

        qs, q = [], q_pre
        for t in ts:
            s = (match if t < t_c else cushion).sample(t)[0]
            ik = self.kin.solve(p_pre + s * v_hat, axis, q, None, max_iters=30)
            if not ik.success:
                return None
            q = ik.q
            qs.append(q)
        dense = Dense(ts[0], cfg.dense_dt, qs)
        if np.any(np.max(np.abs(dense.qds), axis=0) > cfg.v_max * 1.05):
            return None
        if not self.kin.is_collision_free(dense.qs[-1]):
            return None
        return dense

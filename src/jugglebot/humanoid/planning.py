"""Motion planning and trajectory generation for Unitree G1 7-DOF arms (Stage v6).

Provides:
- Quintic polynomial primitives and minimum-time quintic trajectory generation.
- Piecewise trajectory container with continuous velocity and acceleration.
- 7-DOF CatchPlanner for intercepting ballistic balls with LEAP Hand cradle.
"""

from dataclasses import dataclass, field
import numpy as np
import mujoco

from .kinematics import HumanoidArmKinematics
from jugglebot.prediction import BallisticPredictor


@dataclass
class PlannerConfig7DOF:
    """Configuration constraints for G1 7-DOF arm trajectory planning."""
    # Catch height constraints in world frame (stand is 0.75m, waist is ~0.8m)
    z_min: float = 0.52   # m
    z_max: float = 0.82   # m
    z_min_end: float = 0.46
    max_reach: float = 0.65  # m from shoulder link to catch_point
    max_tilt: float = np.deg2rad(45.0)  # max deviation from vertical

    # 7-DOF velocity and acceleration limits
    v_max: np.ndarray = field(default_factory=lambda: np.full(7, np.pi))
    a_max: np.ndarray = field(default_factory=lambda: np.array([15.0, 15.0, 15.0, 25.0, 25.0, 25.0, 25.0]))

    # Catch cushion parameters
    alpha: float = 0.35            # hand speed at contact as fraction of ball speed
    cup_speed_max: float = 0.65    # m/s maximum cushion retreat velocity
    d_cush: float = 0.05           # m cushion retreat distance
    cushion_dir: str = "down"      # give way straight down to absorb falling momentum
    dense_dt: float = 0.002        # s


class Quintic:
    """Vectorized 5th-order polynomial with full boundary conditions."""

    def __init__(self, t0: float, duration: float, x0, v0, a0, x1, v1, a1):
        self.t0 = float(t0)
        self.duration = max(float(duration), 1e-4)
        self.t1 = self.t0 + self.duration
        T = self.duration

        x0, v0, a0 = (np.asarray(x, dtype=float) for x in (x0, v0, a0))
        x1, v1, a1 = (np.asarray(x, dtype=float) for x in (x1, v1, a1))
        dx = x1 - x0

        self.c = np.array([
            x0,
            v0,
            a0 / 2.0,
            (20.0 * dx - (8.0 * v1 + 12.0 * v0) * T - (3.0 * a0 - a1) * T**2) / (2.0 * T**3),
            (-30.0 * dx + (14.0 * v1 + 16.0 * v0) * T + (3.0 * a0 - 2.0 * a1) * T**2) / (2.0 * T**4),
            (12.0 * dx - 6.0 * (v1 + v0) * T + (a1 - a0) * T**2) / (2.0 * T**5),
        ])

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.clip(t - self.t0, 0.0, self.duration)
        c = self.c
        x = c[0] + s * (c[1] + s * (c[2] + s * (c[3] + s * (c[4] + s * c[5]))))
        v = c[1] + s * (2 * c[2] + s * (3 * c[3] + s * (4 * c[4] + s * 5 * c[5])))
        a = 2 * c[2] + s * (6 * c[3] + s * (12 * c[4] + s * 20 * c[5]))
        return x, v, a

    def peaks(self, n: int = 50) -> tuple[np.ndarray, np.ndarray]:
        ts = self.t0 + np.linspace(0.0, self.duration, n)
        vs, accs = zip(*[self.sample(t)[1:] for t in ts])
        return np.max(np.abs(vs), axis=0), np.max(np.abs(accs), axis=0)


class Hold:
    """Holds a joint configuration stationary over [t0, t1]."""

    def __init__(self, t0: float, t1: float, q: np.ndarray):
        self.t0, self.t1 = float(t0), float(t1)
        self.q = np.asarray(q, dtype=float)

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.zeros_like(self.q)
        return self.q.copy(), z, z


class Dense:
    """Dense trajectory interpolated from a sampled joint angle array."""

    def __init__(self, t0: float, dt: float, qs: np.ndarray):
        self.t0 = float(t0)
        self.dt = float(dt)
        self.qs = np.asarray(qs, dtype=float)
        self.t1 = self.t0 + self.dt * (len(self.qs) - 1)
        self.qds = np.gradient(self.qs, self.dt, axis=0)
        self.qdds = np.gradient(self.qds, self.dt, axis=0)

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = np.clip((t - self.t0) / self.dt, 0.0, len(self.qs) - 1)
        i = min(int(f), len(self.qs) - 2)
        w = f - i
        q = (1 - w) * self.qs[i] + w * self.qs[i + 1]
        qd = (1 - w) * self.qds[i] + w * self.qds[i + 1]
        qdd = (1 - w) * self.qdds[i] + w * self.qdds[i + 1]
        return q, qd, qdd


class Trajectory:
    """Composite piecewise trajectory across multiple segments."""

    def __init__(self, segments):
        self.segments = segments

    @property
    def t_end(self) -> float:
        last = self.segments[-1]
        return last.t0 if isinstance(last, Hold) else last.t1

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for seg in self.segments:
            if t < seg.t1:
                return seg.sample(t)
        q, _, _ = self.segments[-1].sample(self.segments[-1].t1)
        z = np.zeros_like(q)
        return q, z, z


def min_time_quintic(
    t0: float,
    q0: np.ndarray,
    v0: np.ndarray,
    a0: np.ndarray,
    q1: np.ndarray,
    v_max: np.ndarray,
    a_max: np.ndarray,
) -> Quintic:
    """Find shortest quintic segment respecting 7-DOF velocity and acceleration limits."""
    dq = np.abs(q1 - q0)
    T = max(0.06, float(np.max(1.875 * dq / v_max)), float(np.max(np.sqrt(5.77 * dq / a_max))))
    zero = np.zeros_like(q0)
    v_lim = np.maximum(v_max, np.abs(v0)) * 1.02
    a_lim = np.maximum(a_max, np.abs(a0)) * 1.02

    for _ in range(50):
        seg = Quintic(t0, T, q0, v0, a0, q1, zero, zero)
        vp, ap = seg.peaks()
        if np.all(vp <= v_lim) and np.all(ap <= a_lim):
            return seg
        T *= 1.06
    return seg


@dataclass
class CatchPlan7DOF:
    t_catch: float
    p_catch: np.ndarray
    v_ball: np.ndarray
    axis: np.ndarray
    q_catch: np.ndarray
    trajectory: Trajectory


class HumanoidCatchPlanner:
    """Plans 7-DOF bimanual interception trajectories for Unitree G1."""

    def __init__(self, kin: HumanoidArmKinematics, cfg: PlannerConfig7DOF | None = None):
        self.kin = kin
        self.cfg = cfg or PlannerConfig7DOF()

    def plan_direct(
        self,
        t_now: float,
        q_curr: np.ndarray,
        v_curr: np.ndarray,
        a_curr: np.ndarray,
        t_catch: float,
        p_catch: np.ndarray,
        v_ball: np.ndarray,
        target_axis: np.ndarray | None = None,
    ) -> CatchPlan7DOF | None:
        """Plan direct approach from moving arm state into catch pose at t_catch."""
        if t_catch <= t_now + 0.08:
            return None

        # Desired cup normal defaults to nominal upward ready palm axis
        if target_axis is None:
            _, target_axis = self.kin.fk(self.kin.q_nominal)

        ik_res = self.kin.solve(p_catch, target_axis, q_init=q_curr)
        if not ik_res.success:
            return None

        q_target = ik_res.q
        approach = min_time_quintic(
            t0=t_now,
            q0=q_curr,
            v0=v_curr,
            a0=a_curr,
            q1=q_target,
            v_max=self.cfg.v_max,
            a_max=self.cfg.a_max,
        )

        if approach.t1 > t_catch:
            return None  # arm cannot arrive in time

        hold = Hold(approach.t1, t_catch + 0.3, q_target)
        traj = Trajectory([approach, hold])

        return CatchPlan7DOF(
            t_catch=t_catch,
            p_catch=p_catch,
            v_ball=v_ball,
            axis=target_axis,
            q_catch=q_target,
            trajectory=traj,
        )

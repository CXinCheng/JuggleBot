"""Stage v3: geometry / timing check of two-arm, two-ball patterns.

Side by side, both bases facing +x: arm A at y = +Y, arm B at y = -Y. Each period P, A
throws ball 1 to B with flight time F_A (stroke starts at t = 0) and B throws ball 2 to A
with flight time F_B (stroke starts at t = s). Throw and catch points (world):

    T_A = (x0 + delta/2,  w, z0)   C_A = (x0 - delta/2,  w, z0)
    T_B = (x0 - delta/2, -w, z0)   C_B = (x0 + delta/2, -w, z0)

Two patterns:
  s1    : F_A = F_B, s = 0, delta > 0 -- the agreed synchronous crossing pattern. With equal
          flights the balls meet at the same time and height, so only the depth offset delta
          keeps them apart, and each arm throws at one depth and catches at another.
          Result: at 180 deg/s only one layout works, with ~8 mm between the balls.
  async : F_A != F_B, s > 0, delta = 0 -- unequal flights ("one high, one low", like a
          shower): the balls cross at different heights in the same plane, so each arm
          throws and catches at the same point. (Equal flights collide in a common plane
          for ANY time offset and endpoint heights: the crossing heights are always equal.)

Hand timing (Shannon): A is empty from its throw to the catch of ball 2, V_A = s + F_B;
B is empty for V_B = F_A - s; the dwells take up the rest of the period.

Each arm is planned in its own base frame with the single-arm modules (jugglebot.*, on
arm_scene.xml) -- joint trajectories do not depend on where the base stands:
  * throw: linewhip with joint-space follow-through (ThrowConfig.follow_through: the straight
    line runs into the elbow singularity near the workspace edge), release region around T with
    the depth pinned, ranked by the stroke-end -> catch-pose time, rollout-verified, and (unless
    --no-calibrate) calibrated in simulation to reach the other arm's C at t_release + F
  * catch: the ball the other arm throws, planned from the end of its own stroke
  * carry the caught ball back to its stroke start; the period is set by the slower arm
Checks in the two-arm scene (scene.xml; base spacing set through MjSpec): ball-ball clearance
while both balls fly, arm-arm clearance over a period, ball-link clearance (cups excluded).

Usage (from the repo root):
    uv run python v3_two_arm_s1/two_arm_geometry.py --pattern async --no-calibrate
    uv run python v3_two_arm_s1/two_arm_geometry.py --pattern s1 --delta 0.08 --no-calibrate
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from jugglebot.calibration import calibrate
from jugglebot.planning import CatchPlan, CatchPlanner, Hold, PlannerConfig, Trajectory, cup_axis_for, min_time_quintic
from jugglebot.prediction import BallisticPredictor
from jugglebot.simcore import ArmSim, simulate_throw
from jugglebot.throwing import ThrowConfig, ThrowPlan, ThrowPlanner

HERE = Path(__file__).resolve().parent
ARM_SCENE = HERE / "arm_scene.xml"
TWO_ARM_SCENE = HERE / "scene.xml"
ARMS = ("a", "b")
JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")

# Release region around T. Depth (x) is pinned, so each ball path stays in its vertical plane
# (with +-8 cm in x, as in v2, the S1 paths converged to 4-31 mm centre distance).
RELEASE_REGION = (np.array([0.0, -0.08, -0.15]), np.array([0.0, 0.08, 0.15]))
CATCH_A_MAX = np.array([40.0, 40.0, 40.0, 60.0, 60.0, 60.0])  # rad/s^2, torque-checked (as v2 exp. B)
CATCH_TORQUE_FRAC = 0.9
SETTLE_AFTER_CATCH = 0.15  # s from the end of the catch motion until carrying starts
SETTLE_BEFORE_THROW = 0.2  # s
TRANSFER_V_MAX = np.full(6, 0.8 * np.pi)
TRANSFER_A_MAX = np.array([12.0, 12.0, 12.0, 20.0, 20.0, 20.0])
REPLAY_DT = 0.005  # s, sampling of the clearance checks


@dataclass
class Pattern:
    Y: float  # base half-spacing (m)
    w: float  # work-point half-distance (m); throw distance 2 w
    F_A: float  # flight time of A's throw (s)
    F_B: float  # flight time of B's throw (s)
    s: float = 0.0  # B's stroke starts this long after A's (s; negative: before)
    x0: float = 0.65
    delta: float = 0.0  # depth offset between the two ball paths (m)
    z0: float = 0.47

    def base(self, arm: str) -> np.ndarray:
        return np.array([0.0, self.Y if arm == "a" else -self.Y, 0.0])

    def throw_point(self, arm: str) -> np.ndarray:
        k = 1.0 if arm == "a" else -1.0
        return np.array([self.x0 + k * self.delta / 2, k * self.w, self.z0])

    def catch_point(self, arm: str) -> np.ndarray:
        k = 1.0 if arm == "a" else -1.0
        return np.array([self.x0 - k * self.delta / 2, k * self.w, self.z0])

    def flight(self, arm: str) -> float:
        return self.F_A if arm == "a" else self.F_B

    def start(self, arm: str) -> float:
        """Stroke start in the first period (the earlier arm starts at t = 0)."""
        return max(0.0, -self.s) if arm == "a" else max(0.0, self.s)


def other(arm: str) -> str:
    return "b" if arm == "a" else "a"


# ------------------------------------------------------------------ two-arm model
def load_two_arm_model(Y: float, arm_model: mujoco.MjModel | None = None) -> mujoco.MjModel:
    """scene.xml with the bases at y = +-Y. Checks that the cup contact parameters equal the
    planning model's (the calibration relies on identical ball-cup contacts)."""
    spec = mujoco.MjSpec.from_file(str(TWO_ARM_SCENE))
    spec.frame("a_mount").pos = [0.0, Y, 0.0]
    spec.frame("b_mount").pos = [0.0, -Y, 0.0]
    model = spec.compile()
    if arm_model is not None:
        for name in ("cup_base", "cup_wall00"):
            for arm in ARMS:
                g, g1 = model.geom(f"{arm}_{name}"), arm_model.geom(name)
                assert np.allclose(g.solref, g1.solref) and np.allclose(g.solimp, g1.solimp), \
                    f"contact parameters of {arm}_{name} differ from the planning model"
    return model


def arm_qpos_adr(model: mujoco.MjModel, arm: str) -> np.ndarray:
    return np.array([model.jnt_qposadr[model.joint(f"{arm}_{j}").id] for j in JOINTS])


def collision_geoms(model: mujoco.MjModel, arm: str, cups: bool = True) -> list[int]:
    out = []
    for g in range(model.ngeom):
        body = model.body(model.geom_bodyid[g]).name
        if not body.startswith(f"{arm}_") or (model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0):
            continue
        if not cups and body == f"{arm}_cup":
            continue
        out.append(g)
    return out


def min_distance(model, data, g1s, g2s, distmax=0.5) -> float:
    best = distmax
    fromto = np.zeros(6)
    for a in g1s:
        for b in g2s:
            best = min(best, mujoco.mj_geomDistance(model, data, a, b, distmax, fromto))
    return best


# ------------------------------------------------------------------ per-arm planning
class ArmPlanner:
    """Single-arm planning in the arm's base frame (shared by both arms)."""

    def __init__(self, speed_scale: float = 1.0):
        """speed_scale > 1 relaxes the 180 deg/s joint-speed limit (agreed fallback; must be reported)."""
        self.sim = ArmSim(ARM_SCENE)
        cfg = ThrowConfig(follow_through="joint", v_limit=speed_scale * np.pi)
        self.tp = ThrowPlanner(self.sim.model, self.sim.kin, cfg, self.sim.q_ready)
        tp = self.tp

        def torque_ok(seg, n=15):
            for t in np.linspace(seg.t0, seg.t1, n):
                if np.any(np.abs(tp.idyn(*seg.sample(t))) > CATCH_TORQUE_FRAC * tp.torque_limit):
                    return False
            return True

        self.cp = CatchPlanner(self.sim.kin, PlannerConfig(v_max=np.full(6, speed_scale * np.pi), a_max=CATCH_A_MAX,
                                                           min_wait=0.05, earliest_catch=0.10),
                               self.sim.q_ready, tp.shoulder, torque_ok)
        self.g = self.sim.model.opt.gravity

    def throw(self, T: np.ndarray, target: np.ndarray, C: np.ndarray, v_in: np.ndarray, F: float,
              calibrated: bool = True) -> ThrowPlan | None:
        """All points in this arm's base frame. v_in: velocity of the ball this arm will catch at C.
        calibrated=False skips the simulation calibration (the nominal, analytically predicted
        release is used: geometry / timing only)."""
        sim, tp, g = self.sim, self.tp, self.g
        fn = lambda p: (target - p - 0.5 * g * F ** 2) / F
        ik = sim.kin.solve(C, cup_axis_for(v_in, self.cp.cfg.max_tilt), sim.q_ready, sim.q_ready)
        if not ik.success:
            return None
        q_catch, cfg = ik.q, self.cp.cfg

        def rank(q_start, q_end):
            dq = np.abs(q_catch - q_end)
            return max(np.max(1.875 * dq / cfg.v_max), np.max(np.sqrt(5.77 * dq / cfg.a_max)))

        verifier = lambda plan: simulate_throw(sim, plan).outcome == "OK"
        finalize = (lambda pl: calibrate(sim, tp, pl, target, pl.t_release + F)[0]) if calibrated else None
        return tp.plan("linewhip", fn, fn(T), (T + RELEASE_REGION[0], T + RELEASE_REGION[1]),
                       verifier, rank, finalize)

    def catch(self, q_end: np.ndarray, t_stroke_end: float, t_rel: float, p_rel: np.ndarray,
              v_rel: np.ndarray) -> CatchPlan | None:
        pred = BallisticPredictor(self.g)
        pred.t_ref, pred.p0, pred.v0 = t_rel, p_rel, v_rel
        return self.cp.plan(pred, t_stroke_end, (q_end, np.zeros(6), np.zeros(6)))


@dataclass
class ArmCycle:
    throw: ThrowPlan
    catch: CatchPlan
    trajectory: Trajectory  # (wait) -> throw -> catch -> settle -> carry -> rest, absolute times
    t_ready: float  # when this arm could start its next throw (absolute)
    wait: float  # rest at the pre-catch pose before the match phase (catch slack)


@dataclass
class Result:
    pattern: Pattern
    status: str  # OK or the first failure
    cycles: dict | None = None
    period: float = np.nan
    ball_ball: float = np.nan  # m, min centre distance while both balls fly
    arm_arm: float = np.nan  # m, min geom distance between the arms over a period
    ball_link: float = np.nan  # m, min distance ball surface - arm links (cups excluded) during the flights
    plan_time: float = np.nan


def evaluate(pat: Pattern, ap: ArmPlanner, calibrated: bool = True) -> Result:
    t0 = time.perf_counter()
    g = ap.g
    res = Result(pat, "OK")
    plans = {}
    for arm in ARMS:
        o = other(arm)
        b = pat.base(arm)
        T, C, target = pat.throw_point(arm) - b, pat.catch_point(arm) - b, pat.catch_point(o) - b
        F_in = pat.flight(o)
        v_in = (pat.catch_point(arm) - pat.throw_point(o) - 0.5 * g * F_in ** 2) / F_in + g * F_in
        plan = ap.throw(T, target, C, v_in, pat.flight(arm), calibrated)
        if plan is None:
            res.status = f"NO_THROW_{arm.upper()} ({ap.tp.last_rejections})"
            res.plan_time = time.perf_counter() - t0
            return res
        plans[arm] = plan

    # Ball flights in the world frame, absolute times: (t_release, p_release, v_release, F)
    flights = {arm: (pat.start(arm) + plans[arm].t_release, plans[arm].p_release + pat.base(arm),
                     plans[arm].v_release, pat.flight(arm)) for arm in ARMS}

    cycles = {}
    for arm in ARMS:
        o, mine, t_start = other(arm), plans[arm], pat.start(arm)
        t_rel, p_rel, v_rel, _ = flights[o]
        catch = ap.catch(mine.q_end, t_start + mine.duration, t_rel, p_rel - pat.base(arm), v_rel)
        if catch is None:
            res.status = f"NO_CATCH_{arm.upper()}"
            res.plan_time = time.perf_counter() - t0
            return res
        t_carry = catch.trajectory.t_end + SETTLE_AFTER_CATCH
        move = min_time_quintic(t_carry, catch.q_end, np.zeros(6), np.zeros(6), mine.q_start, TRANSFER_V_MAX, TRANSFER_A_MAX)
        segs = [Hold(0.0, t_start, mine.q_start), mine.trajectory(t_start)] + catch.trajectory.segments[:-1]
        segs += [Hold(segs[-1].t1, t_carry, catch.q_end), move, Hold(move.t1, np.inf, mine.q_start)]
        cycles[arm] = ArmCycle(mine, catch, Trajectory(segs), move.t1 + SETTLE_BEFORE_THROW,
                               catch.t_match - catch.trajectory.segments[0].t1)
    res.cycles = cycles
    # Each arm throws again one period after its first throw: P must let both be ready.
    res.period = max(cycles[arm].t_ready - pat.start(arm) for arm in ARMS)

    # ---- ball-ball clearance while both are in the air
    t_lo = max(f[0] for f in flights.values())
    t_hi = min(f[0] + f[3] for f in flights.values())
    ts = np.arange(t_lo, t_hi, 0.001)
    pos = {}
    for arm, (t_rel, p_rel, v_rel, _) in flights.items():
        tau = (ts - t_rel)[:, None]
        pos[arm] = p_rel + v_rel * tau + 0.5 * g * tau ** 2
    res.ball_ball = float(np.min(np.linalg.norm(pos["a"] - pos["b"], axis=1))) if len(ts) else np.inf

    # ---- arm-arm and ball-link clearance in the two-arm scene (one period from A's throw)
    model = load_two_arm_model(pat.Y, ap.sim.model)
    data = mujoco.MjData(model)
    adr = {arm: arm_qpos_adr(model, arm) for arm in ARMS}
    geoms = {arm: collision_geoms(model, arm) for arm in ARMS}
    links = collision_geoms(model, "a", cups=False) + collision_geoms(model, "b", cups=False)
    ball_g, ball_adr = model.geom("ball1").id, model.jnt_qposadr[model.joint("ball1_free").id]
    res.arm_arm, res.ball_link = np.inf, np.inf
    for t in np.arange(0.0, res.period + abs(pat.s), REPLAY_DT):
        for arm in ARMS:
            data.qpos[adr[arm]] = cycles[arm].trajectory.sample(t)[0]
        for t_rel, p_rel, v_rel, F in flights.values():  # one ball geom, placed on each flight in turn
            tau = t - t_rel
            if 0.03 <= tau <= F - 0.03:  # skip the moments in / next to a cup
                data.qpos[ball_adr:ball_adr + 3] = p_rel + v_rel * tau + 0.5 * g * tau ** 2
                mujoco.mj_kinematics(model, data)
                res.ball_link = min(res.ball_link, min_distance(model, data, [ball_g], links))
        mujoco.mj_kinematics(model, data)
        res.arm_arm = min(res.arm_arm, min_distance(model, data, geoms["a"], geoms["b"]))
    res.plan_time = time.perf_counter() - t0
    return res


def describe(r: Result) -> str:
    p = r.pattern
    head = (f"Y={p.Y:.2f} d={2 * p.w:.2f} F_A/F_B={p.F_A:.3f}/{p.F_B:.3f} s={p.s:+.2f} delta={p.delta:.2f}")
    if r.status != "OK":
        return f"{head}: {r.status} ({r.plan_time:.0f} s)"
    c = r.cycles
    return (f"{head}: OK | wait A/B {1e3 * c['a'].wait:.0f}/{1e3 * c['b'].wait:.0f} ms"
            f" | joint speed A/B {c['a'].throw.speed_ratio:.2f}/{c['b'].throw.speed_ratio:.2f}"
            f" torque {c['a'].throw.torque_ratio:.2f}/{c['b'].throw.torque_ratio:.2f}"
            f" | ball-ball {1e3 * r.ball_ball:.0f} mm, arm-arm {1e3 * r.arm_arm:.0f} mm, ball-link {1e3 * r.ball_link:.0f} mm"
            f" | period {r.period:.2f} s ({r.plan_time:.0f} s)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v3: two-arm, two-ball pattern geometry check.")
    p.add_argument("--pattern", choices=["s1", "async"], default="async")
    p.add_argument("--Y", nargs="+", type=float, default=[0.25, 0.30, 0.35], help="base half-spacing (m)")
    p.add_argument("--inset", nargs="+", type=float, default=[0.05, 0.10],
                   help="work points this far from each base towards the middle (m): w = Y - inset")
    p.add_argument("--F", nargs="+", type=float, default=[0.45],
                   help="s1: flight time; async: flight time of the high throw (s)")
    p.add_argument("--F-low", nargs="+", type=float, default=[0.30, 0.35], help="async: low throw (s)")
    p.add_argument("--offset", type=float, default=None,
                   help="async: B's stroke start after A's (s); default: middle of the ball-clearance window")
    p.add_argument("--high", choices=["a", "b"], default="a", help="async: which arm throws high")
    p.add_argument("--x0", type=float, default=0.65)
    p.add_argument("--delta", type=float, default=None, help="depth offset (m); default s1: 0.08, async: 0")
    p.add_argument("--z0", type=float, default=0.47)
    p.add_argument("--speed-scale", type=float, default=1.0,
                   help="relax the 180 deg/s joint-speed limit by this factor (fallback; report it)")
    p.add_argument("--no-calibrate", action="store_true",
                   help="geometry / timing only: skip the simulation calibration of the throws")
    return p.parse_args()


def async_offset(F_hi: float, F_lo: float, w: float, z0: float, g: float = 9.81) -> float:
    """Offset s (low throw starts after the high one) in the middle of the window where the two
    planar flights keep > 0.1 m apart; assumes release / catch at z0 and t_release ~ equal."""
    best = []
    for s in np.arange(-0.2, F_hi, 0.005):
        ts = np.arange(max(0.0, s), min(F_hi, s + F_lo), 0.001)
        if len(ts) == 0:
            continue
        y1, z1 = w - 2 * w * ts / F_hi, 0.5 * g * ts * (F_hi - ts)
        y2, z2 = -w + 2 * w * (ts - s) / F_lo, 0.5 * g * (ts - s) * (F_lo - (ts - s))
        if np.min(np.hypot(y1 - y2, z1 - z2)) > 0.1 and min(s + F_lo, F_hi - s) >= 0.3:
            best.append(s)
    return float(np.median(best)) if best else 0.0


def main() -> int:
    args = parse_args()
    ap = ArmPlanner(args.speed_scale)
    delta = args.delta if args.delta is not None else (0.08 if args.pattern == "s1" else 0.0)
    for Y in args.Y:
        for inset in args.inset:
            w = Y - inset
            for F in args.F:
                for F_lo in (args.F_low if args.pattern == "async" else [F]):
                    if args.pattern == "s1":
                        pat = Pattern(Y, w, F, F, 0.0, args.x0, delta, args.z0)
                    else:
                        s = args.offset if args.offset is not None else async_offset(F, F_lo, w, args.z0)
                        F_A, F_B = (F, F_lo) if args.high == "a" else (F_lo, F)
                        pat = Pattern(Y, w, F_A, F_B, s if args.high == "a" else -s, args.x0, delta, args.z0)
                    print(describe(evaluate(pat, ap, not args.no_calibrate)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

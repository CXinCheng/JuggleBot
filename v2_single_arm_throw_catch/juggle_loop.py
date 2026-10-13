"""Experiment B (v2): one arm, one ball, continuous throw-catch between two points.

The ball alternates between P_A and P_B: the arm throws from near P_A so that the
ball arrives at P_B after F seconds, catches it there, carries it to the start of
the next throw (near P_B, aimed at P_A), and so on. Per cycle the arm goes through
what each arm of the two-arm pattern S1 has to do:

    throw -> flight F (get from the stroke end to the catch pose in time) -> catch
          -> dwell (settle, carry the ball to the next stroke start) -> throw

Offline setup, per direction (setup_loop):
  * linewhip throw with its release point near the throwing point, ranked by how fast
    the arm gets from the stroke end to the catch pose at the target (throwing.py),
    rollout-verified and calibrated in simulation to hit the target point at
    t_release + F (calibration.py)
Online, per cycle (JuggleLoop.run):
  * the catch is planned when the throw is scheduled, from the calibrated nominal
    release; its approach starts when the stroke ends
  * after the release the ball is observed at OBS_RATE (optionally noisy) and the
    catch is re-planned as in v1 when the predicted catch point moves
  * success: ball inside the cup and at rest relative to it after the catch motion

A run stops at the first failure or after --cycles catches. Failures: NO_CATCH_PLAN,
DROPPED (ball on the ground), ROBOT_HIT (ball touched the arm outside the cup),
NOT_CAUGHT (not in the cup / not at rest in time), SPEED_LIMIT (actual joint speed
above 180 deg/s), NO_RELEASE.

Usage (from the repo root):
    uv run python v2_single_arm_throw_catch/juggle_loop.py                       # 10 runs x 20 catches
    uv run python v2_single_arm_throw_catch/juggle_loop.py --runs 1 --viewer     # watch (macOS: mjpython)
    uv run python v2_single_arm_throw_catch/juggle_loop.py --obs-noise 0.005 --video videos/loop.mp4
"""

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from calibration import calibrate
from planning import CatchPlanner, Hold, PlannerConfig, Trajectory, cup_axis_for, min_time_quintic
from prediction import BallisticPredictor
from simcore import ArmSim, FREE_CONFIRM, simulate_throw
from throwing import ThrowConfig, ThrowPlan, ThrowPlanner
from video import VideoRecorder, ViewerPacer

DEFAULT_SCENE = Path(__file__).resolve().parent / "scene.xml"

# Perception / re-planning (as in v1)
OBS_RATE = 100.0  # Hz
OBS_WINDOW = 0.05  # s of observations after the release before the first refit
REPLAN_PERIOD = 0.05  # s
REPLAN_THRESHOLD = 0.005  # m, re-plan if the predicted catch point moved more than this
REPLAN_LOCK = 0.03  # s before the match phase: no more re-planning
REPLAN_MIN_WAIT = 0.02  # s at the pre-catch pose after a re-planned approach

# Cycle timing
SETTLE_BEFORE_THROW = 0.2  # s at the stroke start pose (the ball settles on the base)
HOLD_CAUGHT = 0.1  # s the ball must rest in the cup (relative speed < CAUGHT_SPEED) after the catch motion
CAUGHT_SPEED = 0.05  # m/s
CATCH_TIMEOUT = 1.0  # s after the planned catch time
# Carrying the ball to the next stroke start. Faster carrying shortens the dwell (~1.5 s at half
# these limits, ~0.9 s at twice) but leaves the ball less settled: release errors grow from ~6
# to ~250 mm/s. These values: dwell ~1.15 s, release error <= ~25 mm/s.
TRANSFER_V_MAX = np.full(6, 0.8 * np.pi)  # rad/s
TRANSFER_A_MAX = np.array([12.0, 12.0, 12.0, 20.0, 20.0, 20.0])  # rad/s^2

# Catching with the short juggling flights: higher joint accelerations than v1, but every
# approach is checked against the actuator torque limits.
CATCH_A_MAX = np.array([40.0, 40.0, 40.0, 60.0, 60.0, 60.0])  # rad/s^2
CATCH_TORQUE_FRAC = 0.9
RELEASE_REGION = (np.array([-0.08, -0.08, -0.15]), np.array([0.08, 0.08, 0.15]))  # m around the throwing point


@dataclass
class LoopSetup:
    points: tuple  # (P_A, P_B), ball centre targets
    F: float
    plans: list  # [A -> B, B -> A], calibrated


@dataclass
class CycleRecord:
    k: int
    release_dt: float = np.nan  # s, actual - nominal (calibrated) release time
    release_dv: float = np.nan  # m/s, actual - nominal release velocity (propagated to the nominal time)
    flight: float = np.nan  # s, release -> ball enters the cup
    wait: float = np.nan  # s the cup rests at the pre-catch pose (slack of the catch approach)
    replans: int = 0
    impact_speed: float = np.nan  # m/s, ball relative to the cup when it enters
    dwell: float = np.nan  # s, ball enters the cup -> next release
    # timeline (absolute sim time): throw start, release, ball enters the cup, catch confirmed
    t_throw: float = np.nan
    t_release: float = np.nan
    t_entered: float = np.nan
    t_caught: float = np.nan
    force_peak: float = 0.0  # max |actuator force| / limit during the cycle
    force_peak_t: float = np.nan


@dataclass
class RunResult:
    catches: int = 0
    outcome: str = "CAP"  # CAP (reached --cycles) or the failure
    cycles: list = field(default_factory=list)
    speed_peak: float = 0.0
    force_peak: float = 0.0
    sim_time: float = 0.0


def torque_checker(planner: ThrowPlanner, frac: float = CATCH_TORQUE_FRAC):
    def ok(seg, n=15):
        for t in np.linspace(seg.t0, seg.t1, n):
            if np.any(np.abs(planner.idyn(*seg.sample(t))) > frac * planner.torque_limit):
                return False
        return True
    return ok


def setup_loop(sim: ArmSim, tp: ThrowPlanner, cp: CatchPlanner, P_A, P_B, F, verbose=True) -> LoopSetup | None:
    """Plan and calibrate both throws (A -> B, B -> A)."""
    g = sim.model.opt.gravity
    verifier = lambda plan: simulate_throw(sim, plan).outcome == "OK"
    plans = []
    for P_from, P_to, name in ((P_A, P_B, "A -> B"), (P_B, P_A, "B -> A")):
        t0 = time.perf_counter()
        fn = lambda p, P_to=P_to: (P_to - p - 0.5 * g * F ** 2) / F
        v_in = fn(P_from) + g * F  # ball velocity on arrival
        ik = sim.kin.solve(P_to, cup_axis_for(v_in, cp.cfg.max_tilt), sim.q_ready, sim.q_ready)
        if not ik.success:
            print(f"  {name}: no catch pose at the target")
            return None
        q_catch, cfg = ik.q, cp.cfg

        def rank(q_start, q_end):  # rest-to-rest move time stroke end -> catch pose
            dq = np.abs(q_catch - q_end)
            return max(np.max(1.875 * dq / cfg.v_max), np.max(np.sqrt(5.77 * dq / cfg.a_max)))

        finalize = lambda pl, P_to=P_to: calibrate(sim, tp, pl, P_to, pl.t_release + F)[0]
        region = (P_from + RELEASE_REGION[0], P_from + RELEASE_REGION[1])
        plan = tp.plan("linewhip", fn, fn(P_from), region, verifier, rank, finalize)
        if plan is None:
            print(f"  {name}: no feasible throw ({tp.last_rejections})")
            return None
        if verbose:
            print(f"  {name}: release at {plan.p_release.round(3)}, |v| {np.linalg.norm(plan.v_release):.2f} m/s, "
                  f"joint speed {plan.speed_ratio:.2f}, torque {plan.torque_ratio:.2f}, "
                  f"stroke end -> catch pose ~{rank(plan.q_start, plan.q_end):.2f} s "
                  f"({time.perf_counter() - t0:.1f} s)", flush=True)
        plans.append(plan)
    return LoopSetup((P_A, P_B), F, plans)


class JuggleLoop:
    def __init__(self, sim: ArmSim, setup: LoopSetup, cp: CatchPlanner, obs_noise: float = 0.0,
                 noise_rng: np.random.Generator | None = None):
        self.sim, self.setup, self.cp, self.obs_noise = sim, setup, cp, obs_noise
        self.noise_rng = np.random.default_rng() if noise_rng is None else noise_rng
        self.g = sim.model.opt.gravity
        self.obs_every = max(1, round(1.0 / (OBS_RATE * sim.model.opt.timestep)))

    def _nominal_predictor(self, plan: ThrowPlan, t_throw: float) -> BallisticPredictor:
        pred = BallisticPredictor(self.g)
        pred.t_ref, pred.p0, pred.v0 = t_throw + plan.t_release, plan.p_release.copy(), plan.v_release.copy()
        return pred

    def run(self, max_cycles: int, rng: np.random.Generator, offset: float = 0.01,
            recorder=None, verbose: bool = False) -> RunResult:
        sim, d, plans = self.sim, self.sim.data, self.setup.plans
        res = RunResult()
        r, phi = offset * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        sim.reset(plans[0].q_start, r * np.array([np.cos(phi), np.sin(phi)]))
        pre = [Hold(0.0, SETTLE_BEFORE_THROW, plans[0].q_start)]  # segments before the next throw
        t_throw = SETTLE_BEFORE_THROW
        t_entered_prev = None

        for k in range(max_cycles):
            plan = plans[k % 2]
            rec = CycleRecord(k)
            t_stroke_end = t_throw + plan.duration
            catch = self.cp.plan(self._nominal_predictor(plan, t_throw), t_stroke_end,
                                 (plan.q_end, np.zeros(6), np.zeros(6)))
            if catch is None:
                res.outcome = "NO_CATCH_PLAN"
                break
            traj = Trajectory(pre + [plan.trajectory(t_throw)] + catch.trajectory.segments)
            outcome, rec, traj = self._cycle(plan, traj, catch, t_throw, t_stroke_end, rec, recorder)
            if t_entered_prev is not None and np.isfinite(rec.t_release):
                res.cycles[-1].dwell = rec.t_release - t_entered_prev
            res.cycles.append(rec)
            if outcome != "CAUGHT":
                res.outcome = outcome
                break
            res.catches += 1
            t_entered_prev = rec.t_entered
            if verbose:
                print(f"    catch {k + 1}: release dt {1e3 * rec.release_dt:+.1f} ms, dv {1e3 * rec.release_dv:.0f} mm/s, "
                      f"wait {1e3 * rec.wait:.0f} ms, replans {rec.replans}, impact {rec.impact_speed:.2f} m/s")
            # Carry the ball to the next stroke start, let it settle there.
            nxt = plans[(k + 1) % 2]
            q_now = traj.sample(d.time)[0]
            move = min_time_quintic(d.time, q_now, np.zeros(6), np.zeros(6), nxt.q_start, TRANSFER_V_MAX, TRANSFER_A_MAX)
            t_throw = move.t1 + SETTLE_BEFORE_THROW
            pre = [Hold(0.0, d.time, q_now), move, Hold(move.t1, t_throw, nxt.q_start)]

        res.speed_peak, res.force_peak, res.sim_time = sim.guard.peak, sim.max_force_ratio, d.time
        return res

    def _cycle(self, plan, traj, catch, t_throw, t_stroke_end, rec, recorder):
        """Simulate one throw -> catch. Returns (outcome, record, trajectory in force at the end) --
        the catch may have been re-planned, and the next motion must start from where it ends."""
        sim, d = self.sim, self.sim.data
        rec.t_throw = t_throw
        state, t_cand, cand = "held", None, None  # release detection, as in simcore.simulate_throw
        obs_t, obs_p, last_fit = [], [], -np.inf
        predictor = BallisticPredictor(self.g)
        t_entered, hold_start, step, left_cup = None, None, 0, False
        while True:
            t = d.time
            # --- observe, refit, re-plan (only after the release and the end of the stroke)
            locked = t >= catch.t_match - REPLAN_LOCK
            if state != "held" and not locked and step % self.obs_every == 0:
                p, _ = sim.ball_state()
                obs_t.append(t)
                obs_p.append(p + (self.noise_rng.normal(0.0, self.obs_noise, 3) if self.obs_noise else 0.0))
            if (state == "free" and not locked and t >= t_stroke_end and t >= obs_t[0] + OBS_WINDOW
                    and t - last_fit >= REPLAN_PERIOD):
                predictor.fit(obs_t, obs_p)
                last_fit = t
                if np.linalg.norm(predictor.predict(catch.t_catch)[0] - catch.p_catch) > REPLAN_THRESHOLD:
                    new = self.cp.plan(predictor, t, traj.sample(t), min_wait=REPLAN_MIN_WAIT)
                    if new is not None:
                        catch, traj = new, new.trajectory
                        rec.replans += 1

            sim.control(traj)
            sim.step()
            step += 1
            fr = float(np.max(np.abs(d.actuator_force) / sim.force_limit))
            if fr > rec.force_peak:
                rec.force_peak, rec.force_peak_t = fr, d.time
            if recorder is not None:
                recorder.capture(d)
            if sim.guard.violated_at is not None:
                return "SPEED_LIMIT", rec, traj
            c = sim.ball_contacts()
            if c.ground:
                return "DROPPED", rec, traj
            if c.robot:
                return "ROBOT_HIT", rec, traj

            # --- release
            if state == "held":
                if not c.cup and d.time > t_throw:
                    state, t_cand, cand = "maybe_free", d.time, sim.ball_state()
                    obs_t, obs_p = [], []
                elif d.time > t_stroke_end + 0.2:
                    return "NO_RELEASE", rec, traj
            elif state == "maybe_free":
                if c.cup:
                    state = "held"
                elif d.time - t_cand >= FREE_CONFIRM:
                    state = "free"
                    rec.t_release = t_cand
                    tau = (t_throw + plan.t_release) - t_cand
                    rec.release_dt = -tau
                    v_cmp = cand[1] + self.g * tau
                    rec.release_dv = float(np.linalg.norm(v_cmp - plan.v_release))
            else:
                # --- catch (the ball must have left the cavity before it can enter it again)
                inside = sim.ball_in_cup()
                left_cup = left_cup or not inside
                if inside and left_cup and t_entered is None:
                    t_entered = rec.t_entered = d.time
                    rec.flight = t_entered - t_cand
                    rec.impact_speed = float(np.linalg.norm(sim.ball_relative_velocity()))
                    rec.wait = catch.t_match - catch.trajectory.segments[0].t1
                if (t_entered is not None and inside and d.time >= traj.t_end
                        and np.linalg.norm(sim.ball_relative_velocity()) < CAUGHT_SPEED):
                    hold_start = d.time if hold_start is None else hold_start
                    if d.time - hold_start >= HOLD_CAUGHT:
                        rec.t_caught = d.time
                        return "CAUGHT", rec, traj
                else:
                    hold_start = None
                if d.time > catch.t_catch + CATCH_TIMEOUT:
                    return "NOT_CAUGHT", rec, traj


def summarize(results: list[RunResult], cap: int) -> None:
    n = len(results)
    catches = np.array([r.catches for r in results])
    print(f"\nRuns: {n}, catches per run: median {np.median(catches):.0f}, min {catches.min()}, max {catches.max()} "
          f"(cap {cap}); runs reaching the cap: {sum(r.outcome == 'CAP' for r in results)}/{n}")
    print("Outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r.outcome for r in results).items())))
    cyc = [c for r in results for c in r.cycles if np.isfinite(c.flight)]
    if cyc:
        stat = lambda name, xs, s, u: print(f"  {name}: mean {s * np.nanmean(xs):.1f} {u}, max {s * np.nanmax(xs):.1f} {u}")
        stat("release timing error |dt|", [abs(c.release_dt) for c in cyc], 1e3, "ms")
        stat("release velocity error  ", [c.release_dv for c in cyc], 1e3, "mm/s")
        stat("flight (release -> cup)  ", [c.flight for c in cyc], 1e3, "ms")
        print(f"  wait at pre-catch pose  : mean {1e3 * np.nanmean([c.wait for c in cyc]):.0f} ms, "
              f"min {1e3 * np.nanmin([c.wait for c in cyc]):.0f} ms")
        stat("impact speed (rel. cup)  ", [c.impact_speed for c in cyc], 1, "m/s")
        stat("dwell (catch -> release) ", [c.dwell for c in cyc if np.isfinite(c.dwell)], 1e3, "ms")
        print(f"  re-plans per catch      : mean {np.mean([c.replans for c in cyc]):.2f}")
    print(f"  peak joint speed {max(r.speed_peak for r in results):.2f} x 180 deg/s, "
          f"peak actuator force {max(r.force_peak for r in results):.2f} x limit")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v2 experiment B: single-arm two-point throw-catch loop.")
    p.add_argument("--A", nargs=3, type=float, default=[0.6, -0.075, 0.55], help="point A (ball centre, m)")
    p.add_argument("--B", nargs=3, type=float, default=[0.6, 0.075, 0.55], help="point B (ball centre, m)")
    p.add_argument("--F", type=float, default=0.4, help="flight time (s)")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--cycles", type=int, default=20, help="stop a run after this many catches")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-noise", type=float, default=0.0, help="std. dev. of the ball position observations (m)")
    p.add_argument("--offset", type=float, default=0.01, help="max initial ball offset on the cup base (m)")
    p.add_argument("--video", type=Path, default=None, help="record the first run (mp4, real time)")
    p.add_argument("--viewer", action="store_true", help="watch the runs in the MuJoCo viewer")
    p.add_argument("--speed", type=float, default=0.5, help="viewer playback speed (x real time)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sim = ArmSim(args.scene)
    tp = ThrowPlanner(sim.model, sim.kin, ThrowConfig(), sim.q_ready)
    cp = CatchPlanner(sim.kin, PlannerConfig(a_max=CATCH_A_MAX, min_wait=0.05, earliest_catch=0.10),
                      sim.q_ready, tp.shoulder, torque_checker(tp))
    P_A, P_B = np.array(args.A), np.array(args.B)
    print(f"Setup: A {P_A}, B {P_B}, |AB| {np.linalg.norm(P_B - P_A):.3f} m, F {args.F} s")
    setup = setup_loop(sim, tp, cp, P_A, P_B, args.F)
    if setup is None:
        return 1

    rng, noise_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(args.seed).spawn(2))
    viewer = pacer = None
    if args.viewer:
        try:
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
        except RuntimeError as exc:
            hint = " (on macOS run it with mjpython)" if sys.platform == "darwin" else ""
            raise SystemExit(f"Could not launch viewer{hint}: {exc}")
        viewer.cam.lookat[:], viewer.cam.distance = (0.5, 0.0, 0.55), 1.8
        viewer.cam.azimuth, viewer.cam.elevation = 200.0, -12.0
        pacer = ViewerPacer(viewer, args.speed)

    loop = JuggleLoop(sim, setup, cp, args.obs_noise, noise_rng)
    results = []
    for i in range(args.runs):
        if viewer is not None and not viewer.is_running():
            break
        rec = pacer
        if args.video is not None and i == 0:
            rec = VideoRecorder(sim.model, args.video, speed=1.0, camera="loop")
        t0 = time.perf_counter()
        r = loop.run(args.cycles, rng, args.offset, rec, args.verbose)
        if isinstance(rec, VideoRecorder):
            rec.close()
        if pacer is not None:
            pacer.linger(1.0)
        results.append(r)
        print(f"  run {i + 1}: {r.catches} catches, {r.outcome} (sim {r.sim_time:.1f} s, wall {time.perf_counter() - t0:.1f} s)",
              flush=True)
    if viewer is not None:
        viewer.close()
    if results:
        summarize(results, args.cycles)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Experiment A (v2): characterise cup throws in physics.

For each (primitive, flight time F, horizontal distance d) the throw planner finds a
release point and stroke (throwing.py). The target: land d metres along the heading,
at the release height, F seconds after the release, i.e. release velocity
v = (d / F) h + (g F / 2) z, target point p* = p_release + v F + g F^2 / 2 at
t* = t_release + F.

The stroke then runs in MuJoCo, the ball starting at a random spot on the cup base
(--offset), and the actual release is compared with the plan:

    dt_release : actual - planned release time
    dv, dp     : actual - planned ball velocity / position, the actual state propagated
                 to the planned release time (a timing offset alone is not counted twice)
    miss       : |ball at t* - p*|, the error the catcher has to absorb
    wall       : share of releases where the ball last touched the cup wall

"raw" uses the analytic release prediction of throwing.py; with --calibrate the
command is first calibrated in simulation (calibration.py, ball centred), and "cal"
reports the remaining, random part of the error.

Usage (from the repo root):
    uv run python v2_single_arm_throw_catch/throw_test.py                        # default sweep
    uv run python v2_single_arm_throw_catch/throw_test.py --mode whip --F 0.45 --d 0.4 --trials 20 --calibrate
    uv run python v2_single_arm_throw_catch/throw_test.py --video videos/       # one 0.25x clip per config
    uv run python v2_single_arm_throw_catch/throw_test.py --mode linewhip --F 0.5 --d 0.4 --trials 3 --calibrate --viewer

--viewer plays every trial in the MuJoCo viewer at --speed x real time (default 0.25 =
4x slow motion). On macOS start it with `mjpython` instead of `python`.
"""

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from calibration import calibrate
from simcore import ArmSim, ThrowRollout, simulate_throw
from throwing import ThrowConfig, ThrowPlan, ThrowPlanner
from video import VideoRecorder, ViewerPacer

DEFAULT_SCENE = Path(__file__).resolve().parent / "scene.xml"
HEADINGS = {"lateral": np.array([0.0, 1.0, 0.0]), "radial": np.array([1.0, 0.0, 0.0])}


@dataclass
class TrialResult:
    mode: str
    F: float
    d: float
    variant: str  # raw / cal
    trial: int
    outcome: str
    dt_release: float = np.nan  # s
    dv: float = np.nan  # m/s
    dv_rel: float = np.nan  # dv / |v_planned|
    angle: float = np.nan  # rad
    dp: float = np.nan  # m
    miss: float = np.nan  # m, |ball(t*) - p*|
    wall_at_release: bool = False
    rattles: int = 0
    speed_peak: float = 0.0
    speed_violation: bool = False
    force_peak: float = 0.0
    ctrl_clamped: int = 0


def evaluate(roll: ThrowRollout, plan: ThrowPlan, p_target, t_target, g, **meta) -> TrialResult:
    res = TrialResult(outcome=roll.outcome, wall_at_release=roll.wall_at_release, rattles=roll.rattles,
                      speed_peak=roll.speed_peak, speed_violation=roll.speed_violation,
                      force_peak=roll.force_peak, ctrl_clamped=roll.ctrl_clamped, **meta)
    if roll.outcome != "OK":
        return res
    res.dt_release = roll.t_release - plan.t_release
    tau = plan.t_release - roll.t_release
    p_cmp = roll.p_release + roll.v_release * tau + 0.5 * g * tau ** 2
    v_cmp = roll.v_release + g * tau
    res.dv = float(np.linalg.norm(v_cmp - plan.v_release))
    res.dv_rel = res.dv / float(np.linalg.norm(plan.v_release))
    cos = v_cmp @ plan.v_release / (np.linalg.norm(v_cmp) * np.linalg.norm(plan.v_release))
    res.angle = float(np.arccos(np.clip(cos, -1, 1)))
    res.dp = float(np.linalg.norm(p_cmp - plan.p_release))
    T = t_target - roll.t_release
    res.miss = float(np.linalg.norm(roll.p_release + roll.v_release * T + 0.5 * g * T ** 2 - p_target))
    return res


def summarize(rows: list[TrialResult]) -> str:
    ok = [r for r in rows if r.outcome == "OK"]
    out = f"{len(ok)}/{len(rows)} ok"
    for o in ("RECONTACT", "GROUND", "NO_RELEASE"):
        n = sum(r.outcome == o for r in rows)
        out += f", {o} {n}" if n else ""
    if ok:
        f = lambda xs, s: f"{s * np.mean(xs):.1f}/{s * np.max(xs):.1f}"
        out += (f" | dt_rel {f([abs(r.dt_release) for r in ok], 1e3)} ms"
                f" | dv {f([r.dv for r in ok], 1e3)} mm/s ({f([r.dv_rel for r in ok], 100)}%)"
                f" | angle {f([r.angle for r in ok], 180 / np.pi)} deg"
                f" | miss {f([r.miss for r in ok], 1e3)} mm"
                f" | wall {100 * np.mean([r.wall_at_release for r in ok]):.0f}%")
    out += (f" | speed {max(r.speed_peak for r in rows):.2f}{' VIOLATION' if any(r.speed_violation for r in rows) else ''}"
            f" force {max(r.force_peak for r in rows):.2f}")
    if any(r.ctrl_clamped for r in rows):
        out += f" ctrl-clamped {max(r.ctrl_clamped for r in rows)} steps"
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v2 experiment A: cup throw characterisation.")
    p.add_argument("--mode", nargs="+", default=["translate", "whip", "linewhip"],
                   choices=["translate", "whip", "linewhip"])
    p.add_argument("--F", nargs="+", type=float, default=[0.35, 0.40, 0.45, 0.50], help="flight times (s)")
    p.add_argument("--d", nargs="+", type=float, default=[0.0, 0.2, 0.4], help="horizontal distances (m)")
    p.add_argument("--heading", default="lateral", help="lateral (+y), radial (+x) or an angle in degrees")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--offset", type=float, default=0.01, help="max radial offset of the ball on the base (m)")
    p.add_argument("--vel-frac", type=float, default=ThrowConfig.vel_frac, help="planned joint speed / limit")
    p.add_argument("--calibrate", action="store_true", help="also calibrate in simulation and test again")
    p.add_argument("--video", type=Path, default=None, help="directory: record trial 0 of each config (0.25x)")
    p.add_argument("--viewer", action="store_true", help="watch every trial in the MuJoCo viewer")
    p.add_argument("--speed", type=float, default=0.25, help="viewer playback speed (x real time)")
    p.add_argument("--csv", type=Path, default=None, help="write per-trial results")
    p.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.heading in HEADINGS:
        heading = HEADINGS[args.heading]
    else:
        a = np.radians(float(args.heading))
        heading = np.array([np.cos(a), np.sin(a), 0.0])
    sim = ArmSim(args.scene)
    planner = ThrowPlanner(sim.model, sim.kin, ThrowConfig(vel_frac=args.vel_frac), sim.q_ready)
    g = sim.model.opt.gravity
    rng = np.random.default_rng(args.seed)
    offsets = []  # the same ball offsets for every config and variant
    for _ in range(args.trials):
        r, phi = args.offset * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        offsets.append(r * np.array([np.cos(phi), np.sin(phi)]))

    viewer = None
    if args.viewer:
        try:
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
        except RuntimeError as exc:
            hint = " (on macOS run it with mjpython)" if sys.platform == "darwin" else ""
            raise SystemExit(f"Could not launch viewer{hint}: {exc}")
        viewer.cam.lookat[:], viewer.cam.distance = (0.3, -0.3, 0.5), 2.2
        viewer.cam.azimuth, viewer.cam.elevation = 200.0, -15.0
    pacer = ViewerPacer(viewer, args.speed) if viewer is not None else None

    def run(plan, variant, label, p_target, t_target, meta):
        rows = []
        for k, off in enumerate(offsets):
            if viewer is not None and not viewer.is_running():
                raise SystemExit("Viewer closed.")
            rec = pacer
            if args.video is not None and k == 0:
                rec = VideoRecorder(sim.model, args.video / f"throw_{label}_{variant}.mp4", speed=0.25)
            rows.append(evaluate(simulate_throw(sim, plan, off, rec), plan, p_target, t_target, g,
                                 variant=variant, trial=k, **meta))
            if isinstance(rec, VideoRecorder):
                rec.close()
            if pacer is not None:
                pacer.linger(1.0)
        return rows

    all_rows = []
    for mode in args.mode:
        for d in args.d:
            for F in args.F:
                v = d / F * heading - 0.5 * g * F
                head = f"{mode:9s} d={d:.2f} F={F:.2f} |v|={np.linalg.norm(v):.2f}"
                t0 = time.perf_counter()
                plan = planner.plan(mode, lambda p, v=v: v, v)
                if plan is None:
                    print(f"{head}: infeasible ({planner.last_rejections})", flush=True)
                    continue
                print(f"{head}: plan {time.perf_counter() - t0:.1f}s, joint speed {plan.speed_ratio:.2f}, "
                      f"torque {plan.torque_ratio:.2f}, lateral {plan.lateral_ratio:.2f}, "
                      f"release at {plan.p_release.round(2)}", flush=True)
                p_target = plan.p_release + plan.v_target * F + 0.5 * g * F ** 2
                t_target = plan.t_release + F
                meta, label = dict(mode=mode, F=F, d=d), f"{mode}_d{d:.2f}_F{F:.2f}"
                rows = run(plan, "raw", label, p_target, t_target, meta)
                print(f"    raw: {summarize(rows)}", flush=True)
                all_rows += rows
                if args.calibrate:
                    cal, hist = calibrate(sim, planner, plan, p_target, t_target)
                    steps = " -> ".join(f"{1e3 * h:.0f}" for h in hist)
                    if cal is None:
                        print(f"    cal: failed (miss {steps} mm)", flush=True)
                        continue
                    rows = run(cal, "cal", label, p_target, t_target, meta)
                    print(f"    cal: miss {steps} mm in {len(hist) - 1} corrections, joint speed "
                          f"{cal.speed_ratio:.2f}, torque {cal.torque_ratio:.2f} | {summarize(rows)}", flush=True)
                    all_rows += rows

    if viewer is not None:
        viewer.close()
    if args.csv is not None and all_rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(asdict(all_rows[0]).keys()))
            w.writeheader()
            w.writerows(asdict(r) for r in all_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

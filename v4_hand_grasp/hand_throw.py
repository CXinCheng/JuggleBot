"""E2 (v4): throwing with the LEAP hand -- grasp, accelerate, open the fingers at a set time.

The hand holds the ball in the grasp posture (grasp_test.py) and moves along a straight line
through the release point with a fixed orientation (handsim.LineStroke): accelerate to V with
peak acceleration a_up, then stop with peak deceleration a_down (> 1 g). The release is
commanded by the fingers, described by two parameters:

    t_open : when the fingers start to open, relative to the peak hand speed (s; < 0 = before)
    T_open : how long they take to go from the grasp to the open (flat) posture (s), linearly
             in the joint targets (the torque-limited servos may lag behind)

Once the fingers no longer hold it and the hand decelerates faster than gravity, the ball
leaves: t_open therefore picks the point of the deceleration at which it goes (and so its
speed), and T_open how much the opening fingers disturb it. Each setting is repeated with
small random disturbances -- ball position in the hand (+-OFFSET), opening start jitter
(+-JITTER), finger target noise (+-TARGET_NOISE) -- to measure how repeatable the release is.

Recorded per throw: release time (last hand contact, then free for FREE_CONFIRM) relative to
the peak, ball velocity (speed, angle from the throw direction, sideways component), ball spin,
flange speed at release, and the ballistic apex above the release point.

Usage (from the repo root):
    uv run python v4_hand_grasp/hand_throw.py                                   # default sweep
    uv run python v4_hand_grasp/hand_throw.py --t-open -0.02 0 0.02 --T-open 0.02 --trials 20
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from grasp_test import postures
from handsim import HandSim, LineStroke, Q_READY
from kinematics import ArmKinematics
from video import VideoRecorder

G = 9.81
P_RELEASE = np.array([0.85, 0.0, 0.30])  # flange at the peak speed; palm up (arm stretched out: ~2 m/s possible)
BALL_ON_PALM = np.array([-0.046, -0.036, -0.035])
FREE_CONFIRM = 0.03  # s
SPEED_MARGIN = 0.9  # planned arm joint speed <= 0.9 x 180 deg/s
OFFSET = 0.003  # m, initial ball offset on the palm (uniform in a square)
JITTER = 0.003  # s, opening start jitter (uniform)
TARGET_NOISE = 0.02  # rad, finger target noise (uniform)


@dataclass
class Throw:
    outcome: str  # THROWN / NO_RELEASE / DROPPED / HIT_ARM
    t_release: float = np.nan  # s after the peak hand speed
    speed: float = np.nan  # m/s
    angle: float = np.nan  # rad between ball velocity and throw direction
    sideways: float = np.nan  # m/s, velocity component perpendicular to the throw direction
    spin: float = np.nan  # rad/s
    hand_speed: float = np.nan  # m/s, flange speed at release
    apex: float = np.nan  # m above the release point


def release_config(kin: ArmKinematics, p: np.ndarray = P_RELEASE) -> np.ndarray:
    ik = kin.solve(p, np.array([0.0, 0.0, 1.0]), Q_READY, Q_READY)
    if not ik.success:
        raise RuntimeError("no IK solution at the release point")
    return ik.q


def throw_once(sim: HandSim, stroke: LineStroke, t_open: float, T_open: float, rng: np.random.Generator,
               noise: bool = True, recorder: VideoRecorder | None = None) -> Throw:
    m, d = sim.model, sim.data
    hands = postures(sim)
    grasp, cradle, flat = hands["grasp"], hands["cradle"], hands["flat"]
    off = rng.uniform(-OFFSET, OFFSET, 2) if noise else np.zeros(2)
    jitter = rng.uniform(-JITTER, JITTER) if noise else 0.0
    grasp_n = grasp + (rng.uniform(-TARGET_NOISE, TARGET_NOISE, grasp.shape) if noise else 0.0)
    open_n = flat + (rng.uniform(-TARGET_NOISE, TARGET_NOISE, flat.shape) if noise else 0.0)

    # Settle: ball onto the cradled palm at the stroke start pose, then close the grasp.
    sim.reset(stroke.q_start, cradle)
    sim.place_ball(BALL_ON_PALM + np.array([off[0], off[1], -sim.ball_radius - 0.002]))
    t_close, t_go = 0.4, 0.9
    shift = t_go - stroke.t0  # stroke times are relative to its own t0
    while d.time < t_go:
        if d.time >= t_close:
            sim.set_hand(grasp_n)
        sim.control_arm(stroke.q_start)
        sim.step()
        if recorder is not None and d.time > t_go - 0.15:
            recorder.capture(d)

    t_peak = stroke.t_peak + shift
    t_start_open = t_peak + t_open + jitter
    flange = m.site("flange").id
    res = Throw("NO_RELEASE")
    state, cand = "held", None
    while d.time < stroke.t_end + shift + 0.6:
        t = d.time
        w = np.clip((t - t_start_open) / max(T_open, 1e-6), 0.0, 1.0)
        sim.set_hand(grasp_n + w * (open_n - grasp_n))
        sim.control_arm(*stroke.sample(t - shift))
        sim.step()
        if recorder is not None:
            recorder.capture(d)
        c = sim.ball_contacts()
        if c.ground:
            res.outcome = "DROPPED"
            break
        if c.arm:
            res.outcome = "HIT_ARM"
            break
        if state == "held":
            if not c.hand and d.time > t_peak - stroke.T1:
                p, v = sim.ball_state()
                omega = d.qvel[sim.ball_dadr + 3:sim.ball_dadr + 6].copy()
                vf = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, flange, vf, 0)
                cand, state = (d.time, p, v, omega, float(np.linalg.norm(vf[3:]))), "maybe"
        elif state == "maybe":
            if c.hand:
                state = "held"
            elif d.time - cand[0] >= FREE_CONFIRM:
                state = "free"
        else:
            if c.hand:
                break  # fell back into the hand: the free part is what counts
            if sim.ball_state()[1][2] < 0 and recorder is None:
                break  # past the apex (a recording continues until the ball is back or down)
    if cand is not None and state != "held":
        t_rel, p, v, omega, hs = cand
        u = stroke.d
        res.outcome = "THROWN" if res.outcome == "NO_RELEASE" else res.outcome
        res.t_release = t_rel - t_peak
        res.speed = float(np.linalg.norm(v))
        res.angle = float(np.arccos(np.clip(v @ u / res.speed, -1, 1)))
        res.sideways = float(np.linalg.norm(v - (v @ u) * u))
        res.spin = float(np.linalg.norm(omega))
        res.hand_speed = hs
        res.apex = float(max(v[2], 0.0) ** 2 / (2 * G))
    return res


def summarize(rows: list[Throw]) -> str:
    ok = [r for r in rows if r.outcome == "THROWN"]
    out = f"{len(ok)}/{len(rows)} thrown"
    for o in ("NO_RELEASE", "DROPPED", "HIT_ARM"):
        n = sum(r.outcome == o for r in rows)
        out += f", {o} {n}" if n else ""
    if ok:
        f = lambda xs, s=1.0, p=2: f"{s * np.mean(xs):.{p}f} +- {s * np.std(xs):.{p}f}"
        out += (f" | release {f([r.t_release for r in ok], 1e3, 1)} ms after peak"
                f" | speed {f([r.speed for r in ok])} m/s (hand {f([r.hand_speed for r in ok])})"
                f" | angle {f([r.angle for r in ok], 180 / np.pi, 1)} deg, sideways {f([r.sideways for r in ok], 1e3, 0)} mm/s"
                f" | spin {f([r.spin for r in ok], 1, 1)} rad/s | apex {f([r.apex for r in ok], 1e3, 0)} mm")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="v4 E2: throwing by opening the fingers at a set time.")
    p.add_argument("--V", type=float, default=1.7, help="peak hand speed (m/s)")
    p.add_argument("--a-up", type=float, default=3.0, help="peak acceleration (g)")
    p.add_argument("--a-down", type=float, default=3.0, help="peak deceleration (g)")
    p.add_argument("--tilt", type=float, default=0.0, help="throw direction tilted from vertical (deg)")
    p.add_argument("--heading", type=float, default=0.0,
                   help="azimuth of the tilt (deg, world): 0 = +x = towards the fingertips, 90 = +y = the open "
                        "ring-finger side, 180 = towards the heel of the palm, 270 = the thumb side")
    p.add_argument("--release", nargs=3, type=float, default=list(P_RELEASE), help="flange position at the peak speed (m)")
    p.add_argument("--t-open", nargs="+", type=float, default=[-0.04, -0.02, 0.0, 0.02, 0.04],
                   help="finger opening start relative to the peak speed (s)")
    p.add_argument("--T-open", nargs="+", type=float, default=[0.01, 0.03, 0.06], help="opening duration (s)")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--video", type=Path, default=None, help="record the first throw of each setting (mp4, 0.25x)")
    args = p.parse_args()

    sim = HandSim()
    kin = ArmKinematics(sim.model)
    q_r = release_config(kin, np.array(args.release))
    a, h = np.radians(args.tilt), np.radians(args.heading)
    direction = np.array([np.sin(a) * np.cos(h), np.sin(a) * np.sin(h), np.cos(a)])
    stroke = LineStroke(sim, q_r, direction, args.V, args.a_up * G, args.a_down * G)
    speed = stroke.max_joint_speed() / np.pi
    print(f"stroke: V {args.V} m/s, up {args.a_up} g, down {args.a_down} g, tilt {args.tilt} deg towards {args.heading} deg; "
          f"peak arm joint speed {speed:.2f} x 180 deg/s; peak at t = {stroke.t_peak:.3f} s, stops at {stroke.t_end:.3f} s")
    if speed > SPEED_MARGIN:
        print(f"  over the planning margin ({SPEED_MARGIN}); lower --V")
        return 1
    rng = np.random.default_rng(args.seed)
    for T_open in args.T_open:
        for t_open in args.t_open:
            rows = []
            for k in range(args.trials):
                rec = None
                if args.video is not None and k == 0:
                    rec = VideoRecorder(sim.model, args.video / f"hand_throw_t{1e3 * t_open:+.0f}ms_T{1e3 * T_open:.0f}ms.mp4",
                                        speed=0.25, camera="throw")
                rows.append(throw_once(sim, stroke, t_open, T_open, rng, not args.no_noise, rec))
                if rec is not None:
                    rec.close()
            print(f"t_open {1e3 * t_open:+4.0f} ms, T_open {1e3 * T_open:3.0f} ms: {summarize(rows)}"
                  f" | arm speed peak {sim.speed_peak:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

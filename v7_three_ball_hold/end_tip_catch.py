"""Ending, human-style: alpha (first ball) pinched at the fingertips, beta (second ball) must land in the palm.

Static hand (10 deg fingers-down), fingers at the hold2f hold posture. beta is released at 0.25 m from the landing point
(the palm slot where the retained ball sat in hold2f) with a relative velocity of `speed` m/s along a chosen approach
direction in the hand (seat) frame. Sweeps azimuth/elevation/speed and checks alpha drift and beta capture.
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from search_hold2 import Problem, tilt_g
from feaslib import R

_P = None; _X = None


def _init(fn):
    global _P, _X
    _P = Problem("right", 10.0)
    _X = np.array(json.loads(Path(fn).read_text())["x"])


def trial(args):
    az_deg, el_deg, speed, finger_mode = args
    P, x = _P, _X
    rig = P.rig; f = rig.fingers
    hold = np.clip(x[:16], rig.ctrl_lo, rig.ctrl_hi)
    land = np.array([0.012, -0.012, 0.022])               # palm slot used by the retained ball in hold2f
    alpha = np.array([x[18], x[19], x[20] + 0.022])        # fingertip slot (hold2f: ball1 = thrown ball)
    rig.set_gravity_local(tilt_g(10.0))
    rig.reset(P.flat, hold, [None, alpha])
    # ball index 1 is alpha here; ball 0 = beta (parked until released)
    rig.step(300)
    a0 = rig.ball_local(1).copy()
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    # direction of travel (unit): downward component sin(el) from horizontal, horizontal heading az in the seat x-y plane
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)])
    start = land - 0.25 * d
    q0, v0 = rig.balls[0]
    _, Rm = rig.seat_frame()
    rig.d.qpos[q0:q0 + 3] = rig.to_world(start); rig.d.qvel[v0:v0 + 3] = Rm @ (speed * d)
    hit = False
    for k in range(900):
        rig.step(1)
        if finger_mode == 1 and k == 120:      # after the first touch, nothing changes (kept for later variants)
            pass
    b, a = rig.ball_local(0), rig.ball_local(1)
    n, fs, parts = rig.ball_hand_contacts(0)
    ok = bool(np.linalg.norm(a - a0) < 0.02 and np.linalg.norm(b - land) < 0.04 and np.linalg.norm(rig.ball_vel_local(0)) < 0.05
              and np.linalg.norm(rig.ball_vel_local(1)) < 0.05 and n >= 2)
    return dict(args=args, ok=ok, alpha_drift=float(np.linalg.norm(a - a0)), beta=b.round(3).tolist(), beta_err=float(np.linalg.norm(b - land)),
                beta_contacts=n, parts=sorted(parts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("hold"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    grid = [(az, el, sp, 0) for az in range(0, 360, 45) for el in (45, 65, 85) for sp in (1.0, 2.0)]
    with Pool(16, initializer=_init, initargs=(a.hold,)) as pool:
        res = pool.map(trial, grid)
    good = [r for r in res if r["ok"]]
    print("success", len(good), "/", len(res))
    for r in sorted(res, key=lambda r: r["alpha_drift"])[:8]:
        print(r["args"], "ok" if r["ok"] else "--", "alpha_drift %.3f beta_err %.3f n %d" % (r["alpha_drift"], r["beta_err"], r["beta_contacts"]), r["beta"], r["parts"])
    print("by azimuth (ok count of 6):", {az: sum(r["ok"] for r in res if r["args"][0] == az) for az in range(0, 360, 45)})
    if a.out: Path(a.out).write_text(json.dumps(res))

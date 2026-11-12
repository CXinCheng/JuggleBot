"""Robustness of a found two-ball grasp: tilt envelope and hand-acceleration pulses."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from search_hold2 import Problem, tilt_g
from feaslib import R

DIRS = {"+z(up)": (0, 0, 1), "-z(down)": (0, 0, -1), "+y(finger)": (0, 1, 0), "-y(heel)": (0, -1, 0),
        "+x": (1, 0, 0), "-x": (-1, 0, 0)}


def retained(rig):
    ok = True
    for i in range(2):
        p = rig.ball_local(i)
        if np.linalg.norm(p - np.array([0, 0.0, 0])) > .13 or p[2] < -.03:
            ok = False
    return ok


def settle(prob, x, T=0.9):
    return prob.simulate(x, T=T)


def pulse_test(prob, x, direction, accel, dur=0.12, tail=0.35):
    rig = settle(prob, x)
    if not retained(rig):
        return False
    base = tilt_g(prob.tilt)
    a = accel * np.asarray(direction, float)
    # half-sine pulse in the hand acceleration, 1 ms steps
    n = int(dur * 1000)
    for k in range(n):
        s = np.sin(np.pi * (k + .5) / n)
        rig.set_gravity_local(base - a * s)
        rig.step(1)
    rig.set_gravity_local(base)
    rig.step(int(tail * 1000))
    sp = max(np.linalg.norm(rig.ball_vel_local(i)) for i in range(2))
    return retained(rig) and sp < 0.15


def tilt_scan(prob, x):
    res = {}
    for th in (-5, 0, 5, 10, 15, 20, 25, 30):
        rig = prob.simulate(x, tilt=th)
        res[f"pitch{th}"] = retained(rig) and max(np.linalg.norm(rig.ball_vel_local(i)) for i in range(2)) < .1
    for ro in (-15, -10, -5, 5, 10, 15):
        rig = prob.simulate(x, g_local=tilt_g(prob.tilt, ro))
        res[f"roll{ro}"] = retained(rig) and max(np.linalg.norm(rig.ball_vel_local(i)) for i in range(2)) < .1
    return res


def max_accel(prob, x, direction, accels=(2, 4, 6, 8, 10, 14, 18, 24, 30), dur=0.12):
    best = 0
    for a in accels:
        if pulse_test(prob, x, direction, a, dur):
            best = a
        else:
            break
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("json"); ap.add_argument("--dur", type=float, default=0.12)
    a = ap.parse_args()
    cfg = json.loads(Path(a.json).read_text())
    prob = Problem(cfg["side"], cfg["tilt_deg"]); x = np.array(cfg["x"])
    rig = settle(prob, x)
    print("settled:", [(m["pos"].round(3).tolist(), round(m["speed"], 3), m["parts"]) for m in prob.metrics(rig)])
    print("tilt envelope:", tilt_scan(prob, x))
    out = {}
    for name, dvec in DIRS.items():
        out[name] = max_accel(prob, x, dvec, dur=a.dur)
        print(f"max sustained-half-sine hand accel {name:12s}: {out[name]} m/s^2", flush=True)
    cfg["max_accel"] = out
    Path(a.json).write_text(json.dumps(cfg))

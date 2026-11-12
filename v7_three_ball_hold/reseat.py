"""After A is thrown: move the finger-caged ball B back into the palm cradle during the empty return.

The hand follows the recorded v6 right-hand empty-return motion (effective-gravity replay, t=0.74..1.50 s,
second right release at 0.912, next stroke starts ~1.38). Fingers: hold -> opener -> cradle (3 keyframes).
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from replay_throw import load_rec, g_local_series, Problem
from feaslib import R

T0, T_LEAVE, T_SEAT = 0.74, 0.91, 1.38


def smooth(a):
    a = np.clip(a, 0, 1)
    return a ** 3 * (10 - 15 * a + 6 * a * a)


def run_reseat(prob, x, params, t_end=1.52, log=False, gl=None):
    ts, d1, hold_open, d2, opener_mix, use_catch = params
    rig = prob.rig; f = rig.fingers
    if gl is None:
        t, ps, rot, ctrl, acc = load_rec(); gl = g_local_series(t, rot, acc)
    K0 = np.clip(x[:16], rig.ctrl_lo, rig.ctrl_hi)
    K2 = f.get_ctrl_array("cradle")
    K1 = (f.get_ctrl_array("catch") if use_catch > .5 else f.get_ctrl_array("flat")) * opener_mix + K0 * (1 - opener_mix)
    rig2 = prob.simulate(x, T=0.6)
    hist = []
    for k in range(int(T0 * 1000), int(t_end * 1000)):
        tt = k / 1000.0
        rig.set_gravity_local(gl[k])
        a1 = smooth((tt - ts) / d1)
        a2 = smooth((tt - (ts + d1 + hold_open)) / d2)
        c = (1 - a1) * K0 + a1 * K1
        c = (1 - a2) * c + a2 * K2
        rig.d.ctrl[f.act_indices] = c
        rig.step(1)
        if log and k % 20 == 0:
            hist.append((tt, rig.ball_local(1).copy(), np.linalg.norm(rig.ball_vel_local(1)), rig.ball_hand_contacts(1)[0]))
    return rig, hist


def score(rig):
    B = rig.ball_local(1)
    err = np.linalg.norm(B - np.array([0, 0, 0.022]))
    n, fs, parts = rig.ball_hand_contacts(1)
    sp = np.linalg.norm(rig.ball_vel_local(1))
    return err, sp, n, fs, sorted(parts)


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text()); x = np.array(cfg["x"])
    P = Problem("right", 10.0)
    t, ps, rot, ctrl, acc = load_rec(); gl = g_local_series(t, rot, acc)
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    best = None
    for i in range(int(sys.argv[3]) if len(sys.argv) > 3 else 150):
        p = (rng.uniform(.90, 1.12), rng.uniform(.05, .22), rng.uniform(0, .2), rng.uniform(.05, .2),
             rng.uniform(0.3, 1.0), float(rng.integers(0, 2)))
        rig, _ = run_reseat(P, x, p, t_end=T_SEAT + .12, gl=gl)
        err, sp, n, fs, parts = score(rig)
        A = rig.ball_local(0)
        cost = err + 0.2 * sp + (0.1 if n < 3 else 0)
        if best is None or cost < best[0]:
            best = (cost, p, err, sp, n, fs, parts); print(i, "cost %.3f" % cost, [round(v, 3) for v in p], "err %.3f sp %.3f n %d F %.1f" % (err, sp, n, fs), parts, flush=True)
    Path(sys.argv[1].replace(".json", "_reseat.json")).write_text(json.dumps(dict(params=list(best[1]), err=best[2], speed=best[3])))

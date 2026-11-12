"""Ending, step 1: move the first-caught ball alpha from the palm cradle to the fingertip pinch slot (hold2f 'T' slot).

Single hand, static arm; the hand tilt is a quasi-static effective-gravity profile (palm fingers-down angle theta(t)).
Fingers: v6 'cradle' -> opener -> hold2f pinch posture (index/middle/thumb) with the other digits left open for the next catch.
Search over timing/tilt parameters (random search + CMA refinement).
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from search_hold2 import Problem, tilt_g
from feaslib import R

HOLD = Path(__file__).parent / "results" / "hold2f_right_s20.json"
_P = None; _X = None


def _init():
    global _P, _X
    _P = Problem("right", 10.0); _X = np.array(json.loads(HOLD.read_text())["x"])


def smooth(a):
    a = np.clip(a, 0, 1)
    return a ** 3 * (10 - 15 * a + 6 * a * a)


def run(p, P=None, X=None, trace=False, T_end=0.9):
    """p = [theta_peak, t_up, d_up, hold_peak, d_down, t_open, d_open, hold_open, d_pinch, open_mix]"""
    P = P or _P; X = X if X is not None else _X
    rig = P.rig; f = rig.fingers
    theta_peak, t_up, d_up, hold_peak, d_down, t_open, d_open, hold_open, d_pinch, wf, wc = p
    if wf + wc > 1: wf, wc = wf / (wf + wc), wc / (wf + wc)
    cradle = f.get_ctrl_array("cradle"); flat = f.get_ctrl_array("flat"); catch = f.get_ctrl_array("catch")
    pinch = np.clip(X[:16], rig.ctrl_lo, rig.ctrl_hi)
    # pinch for the three holders (if, mf, th); the other digits stay in the open catch bowl for beta
    holders = np.array([d in ("if", "mf", "th") for d in f.actuator_digits])
    K_pinch = np.where(holders, pinch, catch)
    K_open = wf * flat + wc * catch + (1 - wf - wc) * cradle
    rig.set_gravity_local(tilt_g(10.0))
    rig.reset(cradle, cradle, [[0.0, 0.0, 0.022]])
    rig.step(250)
    a0 = rig.ball_local(0).copy(); hist = []
    for k in range(int(T_end * 1000)):
        t = k / 1000.0
        up = smooth((t - t_up) / d_up); down = smooth((t - (t_up + d_up + hold_peak)) / d_down)
        th = 10.0 + (theta_peak - 10.0) * (up - down)
        rig.set_gravity_local(tilt_g(th))
        a1 = smooth((t - t_open) / d_open); a2 = smooth((t - (t_open + d_open + hold_open)) / d_pinch)
        c = (1 - a1) * cradle + a1 * K_open
        c = (1 - a2) * c + a2 * K_pinch
        rig.d.ctrl[f.act_indices] = c
        rig.step(1)
        if trace and k % 25 == 0:
            hist.append((t, th, rig.ball_local(0).copy(), rig.ball_hand_contacts(0)[0]))
    return rig, a0, hist


def score(rig, X):
    slot = np.array([X[18], X[19], X[20] + 0.022])
    b = rig.ball_local(0); n, fs, parts = rig.ball_hand_contacts(0)
    err = float(np.linalg.norm(b - slot)); sp = float(np.linalg.norm(rig.ball_vel_local(0)))
    return dict(err=err, speed=sp, contacts=n, force=float(fs), parts=sorted(parts), pos=b.round(3).tolist(),
                ok=bool(err < 0.02 and sp < 0.05 and n >= 3))


def cost(p):
    rig, _, _ = run(p)
    s = score(rig, _X)
    return s["err"] + 0.5 * s["speed"] + (0.1 if s["contacts"] < 3 else 0.0)


LO = np.array([10, 0.02, 0.03, 0.0, 0.03, 0.02, 0.03, 0.0, 0.03, 0.0, 0.0])
HI = np.array([40, 0.35, 0.20, 0.15, 0.20, 0.35, 0.20, 0.30, 0.20, 1.0, 1.0])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2160); ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "transfer_best.json"))
    a = ap.parse_args(); rng = np.random.default_rng(0)
    pops = [LO + rng.random(11) * (HI - LO) for _ in range(a.n)]
    with Pool(18, initializer=_init) as pool:
        res = pool.map(cost, pops)
    o = np.argsort(res); print("random search best costs", np.round(np.array(res)[o[:6]], 3), " success(<0.02):", int(np.sum(np.array(res) < 0.02)), "/", a.n)
    best = pops[o[0]]; best_cost = float(res[o[0]])
    # CMA-like refinement around the best few
    mean = np.mean([pops[i] for i in o[:8]], axis=0); sig = 0.15 * (HI - LO)
    with Pool(18, initializer=_init) as pool:
        for g in range(25):
            pop = [np.clip(mean + sig * rng.standard_normal(11), LO, HI) for _ in range(72)]
            cs = pool.map(cost, pop); oo = np.argsort(cs)
            if cs[oo[0]] < best_cost: best, best_cost = pop[oo[0]], cs[oo[0]]
            mean = np.mean([pop[i] for i in oo[:12]], axis=0); sig = 0.85 * sig + 0.15 * np.std([pop[i] for i in oo[:12]], axis=0) + 1e-4
            print("gen", g, "best", round(min(cs), 4), flush=True)
    _init(); rig, a0, hist = run(best, trace=True)
    for t, th, b, n in hist[::2]: print(round(t, 2), round(th, 1), b.round(3), n)
    print("BEST params", np.round(best, 3).tolist()); print(score(rig, _X))
    Path(a.out).write_text(json.dumps(dict(params=best.tolist(), score=score(rig, _X))))

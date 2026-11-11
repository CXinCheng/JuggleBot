"""Two-ball grasp where ball B is CAGED by the fingers (distal) and ball A just sits on the palm.

Goal: A can be scooped out by the palm while B stays. Costs:
 * no contact with the wrist/arm (rules out the wrist-shell pocket found by search_hold2b)
 * B distal (y>0.05) and finger-supported, A on palm centre
 * disturbance pulses on the settled pair (as before)
 * scoop replay (recorded v6 stroke accelerations): B must stay, A is free to leave
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).parent))
from search_hold2 import Problem, tilt_g
from search_hold2b import snapshot, restore, SCEN
from replay_throw import load_rec, g_local_series
from feaslib import R

_P = None; _GL = None; _K0 = None


def _init(side):
    global _P, _GL, _K0
    _P = Problem(side, 10.0)
    t, ps, rot, ctrl, acc = load_rec()
    _GL = g_local_series(t, rot, acc)
    if side == "left":  # recorded motion is the right hand; mirror x of the local frame
        _GL = _GL * np.array([-1, 1, 1])
    _K0 = int(0.74 / (t[1] - t[0]))


def non_hand_contacts(rig):
    m, d = rig.m, rig.d
    bad = 0.0
    f6 = np.zeros(6)
    for c in range(d.ncon):
        con = d.contact[c]
        for i in range(2):
            g = rig.ball_geoms[i]
            if g in (con.geom1, con.geom2):
                o = con.geom2 if con.geom1 == g else con.geom1
                b = m.geom_bodyid[o]
                if b not in rig.hand_bodies and o not in rig.ball_geoms:
                    mujoco.mj_contactForce(m, d, c, f6); bad += max(f6[0], 0.05)
    return bad


def evaluate(x, detail=False, prob=None):
    P = prob or _P
    rig = P.simulate(x, T=0.6)
    ms = P.metrics(rig)
    A, B = [rig.ball_local(i) for i in range(2)]
    rep = {}
    c = sum(2.0 * m["speed"] for m in ms)
    c += 8.0 * non_hand_contacts(rig)
    c += 20 * max(0, 0.05 - B[1]) + 20 * max(0, A[1] - 0.04)   # B distal, A not out on the fingers
    c += 20 * max(0, -0.035 - A[1])                              # A not off the heel
    if np.linalg.norm(A) > .13 or np.linalg.norm(B - [0, .06, 0]) > .13 or A[2] < -.02 or B[2] < -.04:
        return (40.0 + c, rep) if detail else 40.0 + c
    # B support from fingers (digit contacts), reward more digits
    digits = {p[:2] for p in ms[1]["parts"] if p[:2] in ("if", "mf", "rf", "th")}
    c -= 0.15 * len(digits)
    s0 = snapshot(rig); base = tilt_g(P.tilt); refs = [A.copy(), B.copy()]
    for name, dvec, acc in SCEN:
        restore(rig, s0)
        n = 120
        for k in range(n):
            rig.set_gravity_local(base - acc * np.asarray(dvec, float) * np.sin(np.pi * (k + .5) / n)); rig.step(1)
        rig.set_gravity_local(base); rig.step(250)
        wB = np.linalg.norm(rig.ball_local(1) - refs[1]); wA = np.linalg.norm(rig.ball_local(0) - refs[0])
        rep[name] = (wA, wB)
        c += min(wB, 0.3) * 10 + min(wA, 0.3) * 4
    # scoop replay: hand follows recorded v6 stroke accelerations, fingers hold; B must stay
    restore(rig, s0)
    for k in range(_K0 if prob is None else int(0.74 / 0.001), int(1.15 / 0.001)):
        gl = _GL[k] if prob is None else None
        if gl is None:
            t, ps, rot, ctrl, acc = load_rec(); gl = g_local_series(t, rot, acc)[k]
        rig.set_gravity_local(gl); rig.step(1)
    wB = np.linalg.norm(rig.ball_local(1) - refs[1]); upA = rig.ball_local(0)[2]
    rep["scoop"] = (float(upA), float(wB))
    c += min(wB, 0.4) * 25
    c += 2.0 * max(0.0, 0.15 - upA)   # A should leave the palm by > 15 cm during the stroke
    rig.set_gravity_local(base)
    return (c, rep) if detail else c


def _eval(x):
    return evaluate(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right"); ap.add_argument("--gens", type=int, default=80)
    ap.add_argument("--lam", type=int, default=72); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None); ap.add_argument("--out", required=True); ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    prob = Problem(a.side, 10.0)
    rng = np.random.default_rng(a.seed)
    mean = prob.x0.copy(); mean[16:21] = [0, -0.005, 0, 0.095, 0.0]
    mean[:16] = np.clip(mean[:16] + rng.normal(0, .15, 16), prob.lo, prob.hi)
    if a.init:
        mean = np.array(json.loads(Path(a.init).read_text())["x"])
    sig = prob.sig0.copy() * (0.7 if a.init else 1.2)
    n, lam = len(mean), a.lam; mu = lam // 4
    w = np.log(mu + .5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    best = (1e9, None); scale = .8; t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.side,)) as pool:
        for g in range(a.gens):
            pop = mean + scale * sig * rng.standard_normal((lam, n))
            costs = np.array(pool.map(_eval, list(pop)))
            o = np.argsort(costs)
            if costs[o[0]] < best[0]:
                best = (float(costs[o[0]]), pop[o[0]].copy())
            sel = pop[o[:mu]]; new = w @ sel
            sig = 0.8 * sig + 0.2 * np.maximum(np.sqrt(w @ (sel - mean) ** 2), 0.02 * prob.sig0)
            mean = new; scale = max(0.4, scale * .985)
            print(f"[{time.time()-t0:5.0f}s] gen {g:3d} best {best[0]:.3f} genbest {costs[o[0]]:.3f} med {np.median(costs):.2f}", flush=True)
            Path(a.out).write_text(json.dumps(dict(side=a.side, tilt_deg=10.0, cost=best[0], x=best[1].tolist())))
    c, rep = evaluate(best[1], detail=True, prob=prob)
    print("FINAL", c, {k: tuple(round(float(u), 3) for u in v) for k, v in rep.items()})
    rig = prob.simulate(best[1], T=0.6)
    for i, m in enumerate(prob.metrics(rig)):
        print(i, m["pos"].round(4), m["n"], round(m["f"], 2), m["parts"])


if __name__ == "__main__":
    main()

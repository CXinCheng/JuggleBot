"""Ending-hold feasibility: palm ball A held ONLY by the ring finger (LEAP has no pinky; 'rf' plays the pinky) [+ thumb in variant B] and the palm,
while the other fingers hold the last ball B at the fingertips (static hold under hand-acceleration pulses).
Variant A: A allowed {rf, palm}; B allowed {if, mf, th}.  Variant B: A allowed {rf, th, palm}; B allowed {if, mf}.
x = 16 ctrl + A(x,y) + B(x,y,z). balls[0]=A (palm), balls[1]=B (fingertips).
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from search_hold2 import Problem, tilt_g
from search_hold2b import snapshot, restore, SCEN
from search_hold2c import non_hand_contacts

VARIANT = {"A": ({"rf", "pa"}, {"if", "mf", "th"}), "B": ({"rf", "th", "pa"}, {"if", "mf"})}
_P = None; _V = "A"


def _init(side, v):
    global _P, _V
    _P = Problem(side, 10.0); _V = v


def klass(parts):
    return {p[:2] if p[:2] in ("if", "mf", "rf", "th") else "pa" for p in parts}


def evaluate(x, v=None, P=None, detail=False):
    P = P or _P; v = v or _V; allowA, allowB = VARIANT[v]
    rig = P.simulate(x, T=0.6); ms = P.metrics(rig)
    A, B = rig.ball_local(0), rig.ball_local(1)
    cA, cB = klass(ms[0]["parts"]), klass(ms[1]["parts"])
    c = 2.0 * sum(m["speed"] for m in ms) + 8.0 * non_hand_contacts(rig)
    c += 1.5 * len(cA - allowA) + 1.5 * len(cB - allowB) + 1.0 * ("rf" not in cA) + 0.5 * (len(cA) < 2) + 0.5 * (len(cB) < 2)
    rep = {"A": sorted(cA), "B": sorted(cB), "A_f": round(ms[0]["f"], 1), "B_f": round(ms[1]["f"], 1)}
    if np.linalg.norm(B - [0, .06, 0]) > .13 or np.linalg.norm(A) > .13 or A[2] < -.02 or B[2] < -.02:
        return (40 + c, rep) if detail else 40 + c
    c += 20 * max(0, 0.05 - B[1]) + 20 * max(0, np.linalg.norm(A[:2]) - 0.035)
    s0 = snapshot(rig); base = tilt_g(P.tilt); refs = [A.copy(), B.copy()]
    for name, dvec, acc in SCEN:
        restore(rig, s0)
        for k in range(120):
            rig.set_gravity_local(base - acc * np.asarray(dvec, float) * np.sin(np.pi * (k + .5) / 120)); rig.step(1)
        rig.set_gravity_local(base); rig.step(250)
        wA = np.linalg.norm(rig.ball_local(0) - refs[0]); wB = np.linalg.norm(rig.ball_local(1) - refs[1])
        rep[name] = (round(float(wA), 3), round(float(wB), 3)); c += 10 * min(wA, .3) + 10 * min(wB, .3)
    rig.set_gravity_local(base)
    return (c, rep) if detail else c


def _eval(x):
    return evaluate(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A"); ap.add_argument("--gens", type=int, default=60); ap.add_argument("--lam", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--procs", type=int, default=8); ap.add_argument("--out", required=True)
    ap.add_argument("--hold", default=str(Path(__file__).parent / "results" / "hold2f_right_s20.json"))
    a = ap.parse_args()
    prob = Problem("right", 10.0); rng = np.random.default_rng(a.seed)
    h = np.array(json.loads(Path(a.hold).read_text())["x"])[:21]
    mean = h.copy(); mean[16:18] = [0.0, -0.02]   # A toward the heel
    sig = np.r_[np.full(16, .3), .01, .01, .01, .01, .01]
    n, lam = len(mean), a.lam; mu = lam // 4
    w = np.log(mu + .5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    best = (1e9, None); scale = 1.0; t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=("right", a.variant)) as pool:
        for g in range(a.gens):
            pop = np.clip(mean + scale * sig * rng.standard_normal((lam, n)), np.r_[prob.lo, np.full(5, -.2)], np.r_[prob.hi, np.full(5, .2)])
            costs = np.array(pool.map(_eval, list(pop))); o = np.argsort(costs)
            if costs[o[0]] < best[0]: best = (float(costs[o[0]]), pop[o[0]].copy())
            sel = pop[o[:mu]]; new = w @ sel
            sig = 0.8 * sig + 0.2 * np.maximum(np.sqrt(w @ (sel - mean) ** 2), 0.02 * np.r_[np.full(16, .3), np.full(5, .01)])
            mean = new; scale = max(0.4, scale * .985)
            print(f"[{time.time()-t0:5.0f}s] gen {g:3d} best {best[0]:.3f} genbest {costs[o[0]]:.3f} med {np.median(costs):.2f}", flush=True)
            Path(a.out).write_text(json.dumps(dict(variant=a.variant, cost=best[0], x=best[1].tolist())))
    c, rep = evaluate(best[1], v=a.variant, P=prob, detail=True)
    print("FINAL", c, rep)
    rig = prob.simulate(best[1], T=0.6)
    for nm, m in zip("AB", prob.metrics(rig)):
        print(nm, m["pos"].round(4), m["n"], round(m["f"], 2), m["parts"])


if __name__ == "__main__":
    main()

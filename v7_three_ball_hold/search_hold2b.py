"""Multi-condition optimisation of a two-ball resting grasp (parallel sep-CMA-ES).

Each candidate is settled at 10 deg fingers-down tilt and then branched into disturbance scenarios:
hand-acceleration pulses (+-x, +-y lateral/longitudinal, -z) and slow tilt changes (5, 18 deg).
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).parent))
from search_hold2 import Problem, tilt_g
from feaslib import R

ACC = 8.0  # m/s^2 hand acceleration pulses (about 0.8 g)
SCEN = [("+x", (1, 0, 0), ACC), ("-x", (-1, 0, 0), ACC), ("+y", (0, 1, 0), ACC), ("-y", (0, -1, 0), ACC), ("-z", (0, 0, -1), 12.0), ("+z", (0, 0, 1), 20.0)]
TILTS = (5.0, 18.0)
_P = None


def _init(side):
    global _P
    _P = Problem(side, 10.0)


def snapshot(rig):
    d = rig.d
    return d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.time


def restore(rig, s):
    d = rig.d
    d.qpos[:] = s[0]; d.qvel[:] = s[1]; d.ctrl[:] = s[2]; d.time = s[3]
    mujoco.mj_forward(rig.m, d)


def ok_state(rig, ref):
    """Both balls near their settled place, slow, and still touching the hand."""
    worst = 0.0
    for i in range(2):
        drift = np.linalg.norm(rig.ball_local(i) - ref[i])
        worst = max(worst, drift)
    return worst


def evaluate(x, detail=False, prob=None):
    P = prob or _P
    rig = P.simulate(x, T=0.6)
    ref = [rig.ball_local(i) for i in range(2)]
    base_ms = P.metrics(rig)
    c = 0.0
    c += sum(2.0 * m["speed"] + (0.4 if m["n"] == 0 else 0) for m in base_ms)
    bad = [np.linalg.norm(r - np.array([0, 0, 0.0])) > 0.13 or r[2] < -0.03 for r in ref]
    if any(bad):
        return (30.0 + c, {}) if detail else 30.0 + c
    # require B to have at least one contact besides ball-ball, prefer more support
    c -= 0.05 * min(sum(m["n"] for m in base_ms), 8)
    s0 = snapshot(rig)
    rep = {}
    base = tilt_g(P.tilt)
    for name, dvec, acc in SCEN:
        restore(rig, s0)
        n = 120
        for k in range(n):
            rig.set_gravity_local(base - acc * np.asarray(dvec, float) * np.sin(np.pi * (k + .5) / n))
            rig.step(1)
        rig.set_gravity_local(base); rig.step(250)
        w = ok_state(rig, ref); rep[name] = w
        c += min(w, 0.3) * 10.0 + (3.0 if w > 0.03 and max(np.linalg.norm(rig.ball_local(i) - ref[i]) for i in range(2)) > 0.1 else 0.0)
    for th in TILTS:
        restore(rig, s0)
        g1 = tilt_g(th)
        for k in range(150):
            a = (k + 1) / 150.0
            rig.set_gravity_local((1 - a) * base + a * g1); rig.step(1)
        rig.step(250)
        w = ok_state(rig, ref); rep[f"tilt{th:g}"] = w
        c += min(w, 0.3) * 6.0
    rig.set_gravity_local(base)
    return (c, rep) if detail else c


def _eval(x):
    return evaluate(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right"); ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--lam", type=int, default=72); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=18)
    a = ap.parse_args()
    prob = Problem(a.side, 10.0)
    mean = prob.x0.copy()
    if a.init:
        mean = np.array(json.loads(Path(a.init).read_text())["x"])
    sig = prob.sig0.copy() * (0.6 if a.init else 1.0)
    rng = np.random.default_rng(a.seed)
    n, lam = len(mean), a.lam; mu = lam // 4
    w = np.log(mu + .5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    best = (1e9, None); scale = 0.7; t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.side,)) as pool:
        for g in range(a.gens):
            pop = mean + scale * sig * rng.standard_normal((lam, n))
            costs = np.array(pool.map(_eval, list(pop)))
            o = np.argsort(costs)
            if costs[o[0]] < best[0]:
                best = (float(costs[o[0]]), pop[o[0]].copy())
            sel = pop[o[:mu]]; new = w @ sel
            sig = 0.8 * sig + 0.2 * np.maximum(np.sqrt(w @ (sel - mean) ** 2), 0.02 * prob.sig0)
            mean = new; scale = max(0.4, scale * 0.98)
            print(f"[{time.time()-t0:5.0f}s] gen {g:3d} best {best[0]:.3f} genbest {costs[o[0]]:.3f} med {np.median(costs):.2f}", flush=True)
            Path(a.out).write_text(json.dumps(dict(side=a.side, tilt_deg=10.0, cost=best[0], x=best[1].tolist())))
    c, rep = evaluate(best[1], detail=True, prob=prob)
    print("FINAL", c, {k: round(v, 3) for k, v in rep.items()})
    rig = prob.simulate(best[1], T=0.6)
    for i, m in enumerate(prob.metrics(rig)):
        print(i, m["pos"].round(4), m["n"], round(m["f"], 2), m["parts"])


if __name__ == "__main__":
    main()

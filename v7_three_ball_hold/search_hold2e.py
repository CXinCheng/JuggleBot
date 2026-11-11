"""Human-style role swap: the THROWN ball T is pinched at the fingertips (index+middle), the RETAINED ball R sits in the
palm cradle held by ring finger + thumb + palm. At the throw, index+middle open (flat) while ring/thumb keep R.
balls[1] = T (thrown, free z), balls[0] = R (retained, palm). Parameters: 16 ctrl, 5 ball coords, t_open.
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).parent))
from search_hold2 import Problem, tilt_g
from search_hold2b import snapshot, restore, SCEN
from search_hold2c import non_hand_contacts
from replay_throw import load_rec, g_local_series
from feaslib import R

_P = None; _GL = None; _K0 = None; _OPEN = None


def _init(side):
    global _P, _GL, _K0, _OPEN
    _P = Problem(side, 10.0)
    t, ps, rot, ctrl, acc = load_rec()
    _GL = g_local_series(t, rot, acc)
    _K0 = int(0.74 / (t[1] - t[0]))
    f = _P.rig.fingers
    _OPEN = np.array([d in ("if", "mf") for d in f.actuator_digits])


def evaluate(x, detail=False, gl=None, open_mask=None, prob=None):
    P = prob or _P; GL = gl if gl is not None else _GL; mask = open_mask if open_mask is not None else _OPEN
    t_open = float(np.clip(x[21], 0.78, 0.90))
    rig = P.simulate(x[:21], T=0.6)
    ms = P.metrics(rig)
    Rr, T = rig.ball_local(0), rig.ball_local(1)
    c = sum(2.0 * m["speed"] for m in ms) + 8.0 * non_hand_contacts(rig)
    rep = {}
    c += 20 * max(0, 0.05 - T[1]) + 20 * max(0, np.linalg.norm(Rr[:2] - [0, 0]) - 0.03)
    if np.linalg.norm(T - [0, .06, 0]) > .13 or np.linalg.norm(Rr) > .13 or Rr[2] < -.02 or T[2] < -.02:
        return (40.0 + c, rep) if detail else 40.0 + c
    parts_R = {p[:2] for p in ms[0]["parts"]}
    c -= 0.15 * len({"rf", "th", "pa"} & parts_R)
    s0 = snapshot(rig); base = tilt_g(P.tilt); refs = [Rr.copy(), T.copy()]
    for name, dvec, acc in SCEN:
        restore(rig, s0)
        n = 120
        for k in range(n):
            rig.set_gravity_local(base - acc * np.asarray(dvec, float) * np.sin(np.pi * (k + .5) / n)); rig.step(1)
        rig.set_gravity_local(base); rig.step(250)
        wR = np.linalg.norm(rig.ball_local(0) - refs[0]); wT = np.linalg.norm(rig.ball_local(1) - refs[1])
        rep[name] = (wT, wR); c += min(wR, 0.3) * 10 + min(wT, 0.3) * 4
    # scoop with index+middle release
    restore(rig, s0)
    f = rig.fingers; ctrl0 = np.clip(x[:16], rig.ctrl_lo, rig.ctrl_hi); flat = f.get_ctrl_array("flat")
    maxF = 0.0
    for k in range(int(0.74 / 0.001), int(1.15 / 0.001)):
        tt = k / 1000.0
        a = np.clip((tt - t_open) / 0.05, 0, 1); a = a ** 3 * (10 - 15 * a + 6 * a * a)
        c_t = ctrl0.copy(); c_t[mask] = (1 - a) * ctrl0[mask] + a * flat[mask]
        rig.d.ctrl[f.act_indices] = c_t
        rig.set_gravity_local(GL[k]); rig.step(1)
        if k % 5 == 0:
            maxF = max(maxF, rig.ball_ball_contact_force(0, 1))
    wR = np.linalg.norm(rig.ball_local(0) - refs[0]); upT = rig.ball_local(1)[2]
    rep["scoop"] = (float(upT), float(wR), float(maxF))
    c += min(wR, 0.4) * 25 + 2.0 * max(0.0, 0.15 - upT)
    rig.set_gravity_local(base)
    return (c, rep) if detail else c


def _eval(x):
    return evaluate(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right"); ap.add_argument("--gens", type=int, default=80)
    ap.add_argument("--lam", type=int, default=72); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None); ap.add_argument("--out", required=True); ap.add_argument("--procs", type=int, default=9)
    a = ap.parse_args()
    prob = Problem(a.side, 10.0); rng = np.random.default_rng(a.seed)
    mean = np.r_[prob.x0, 0.86]
    mean[16:21] = [0, -0.005, 0, 0.09, 0.02]     # R palm centre (ball0), T distal (ball1)
    mean[:16] = np.clip(mean[:16] + rng.normal(0, .15, 16), prob.lo, prob.hi)
    if a.init:
        mean = np.array(json.loads(Path(a.init).read_text())["x"])
    sig = np.r_[prob.sig0, 0.02] * (0.7 if a.init else 1.2)
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
            sig = 0.8 * sig + 0.2 * np.maximum(np.sqrt(w @ (sel - mean) ** 2), 0.02 * np.r_[prob.sig0, 0.02])
            mean = new; scale = max(0.4, scale * .985)
            print(f"[{time.time()-t0:5.0f}s] gen {g:3d} best {best[0]:.3f} genbest {costs[o[0]]:.3f} med {np.median(costs):.2f}", flush=True)
            Path(a.out).write_text(json.dumps(dict(side=a.side, tilt_deg=10.0, cost=best[0], x=best[1].tolist())))
    t, ps, rot, ctrl, acc = load_rec(); gl = g_local_series(t, rot, acc)
    c, rep = evaluate(best[1], detail=True, gl=gl, open_mask=np.array([d in ("if", "mf") for d in prob.rig.fingers.actuator_digits]), prob=prob)
    print("FINAL", c, {k: tuple(round(float(u), 3) for u in v) for k, v in rep.items()})
    rig = prob.simulate(best[1][:21], T=0.6)
    for i, m in enumerate(prob.metrics(rig)):
        print("T" if i == 0 else "R", m["pos"].round(4), m["n"], round(m["f"], 2), m["parts"])


if __name__ == "__main__":
    main()

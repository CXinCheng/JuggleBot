"""Open-loop calibration of the first (fingertip) throw of the 2+1 start.

Parameters: v6 throw command [forward, lateral, vertical speed, bank] + finger-opening age t_open_age (s after stroke start).
Objective mirrors plan_throw: landing xy error at the other hand's catch point + weighted flight-time error + failure penalties.
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from cascade21 import TwoPlusOneCascade

TIME_W = 4.0   # weight of the flight-time error (s) relative to metres; plan_throw uses THROW_TIME_ERROR_WEIGHT


def evaluate(x, detail=False):
    c = TwoPlusOneCascade(t_open_age=float(np.clip(x[4], 0.05, 0.2)), calibration=None)
    c.planning = False
    c.warm['throw:right:0'] = np.asarray(x[:4], float)
    c.execution_warm['throw:right:0'] = np.asarray(x[:4], float)
    try:
        while c.data.time < 1.2 and c.throw_count < 1:
            c.step()
    except (RuntimeError, ValueError) as e:
        return (50.0, str(e)[:60]) if detail else 50.0
    ev = [e for e in c.events if e['kind'] == 'release']
    if not ev:
        return (40.0, 'no release') if detail else 40.0
    e = ev[0]; tgt = c.hands['left']['catch_p'] + [0, 0, 0.0223]
    p = np.array(e['position']); v = np.array(e['velocity']); g = c.gravity_m_s2
    rem = (v[2] + np.sqrt(max(0.0, v[2] ** 2 + 2 * g * (p[2] - tgt[2])))) / g
    land = p[:2] + v[:2] * rem
    xy = land - tgt[:2]
    flight = rem + (e['time'] - e['separation_time'])
    cost = float(np.linalg.norm(xy)) + TIME_W * abs(flight - c.flight)
    # the retained ball must still be in the hand after the throw (mode carry set)
    if c.hands['right'].get('two_phase'):
        cost += 5.0
    return (cost, dict(xy=xy.round(3).tolist(), flight=round(flight, 3), v=v.round(2).tolist(), t=e['time'])) if detail else cost


def _ev(x):
    return evaluate(x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gens", type=int, default=40); ap.add_argument("--lam", type=int, default=72)
    ap.add_argument("--procs", type=int, default=18); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "first_throw_calibration.json"))
    a = ap.parse_args()
    seed = json.loads((Path(__file__).parents[1] / "src/jugglebot/humanoid/leap_cascade_seed_plans.json").read_text())["forecast"]["throw:right:0"][:4]
    mean = np.r_[seed, 0.114]; sig = np.array([0.15, 0.2, 0.3, 0.03, 0.015]); rng = np.random.default_rng(a.seed)
    lam, mu = a.lam, a.lam // 4; w = np.log(mu + .5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    best = (1e9, None); scale = .8; t0 = time.time(); base_sig = sig.copy()
    with Pool(a.procs) as pool:
        for g in range(a.gens):
            pop = mean + scale * sig * rng.standard_normal((lam, 5))
            costs = np.array(pool.map(_ev, list(pop))); o = np.argsort(costs)
            if costs[o[0]] < best[0]: best = (float(costs[o[0]]), pop[o[0]].copy())
            sel = pop[o[:mu]]; new = w @ sel
            sig = 0.8 * sig + 0.2 * np.maximum(np.sqrt(w @ (sel - mean) ** 2), 0.03 * base_sig); mean = new; scale = max(0.4, scale * .985)
            print(f"[{time.time()-t0:4.0f}s] gen {g:2d} best {best[0]:.4f} genbest {costs[o[0]]:.4f} med {np.median(costs):.2f}", flush=True)
            Path(a.out).write_text(json.dumps(dict(cost=best[0], command=best[1][:4].tolist(), t_open_age=float(best[1][4]))))
    print("FINAL", best[0], evaluate(best[1], detail=True), best[1].round(4).tolist())

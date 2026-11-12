"""Open-loop calibration of the second right-hand throw (the retained ball R) of the 2+1 start.

v6 seeds for 'throw:right:2' were calibrated for a ball caught from the air and seated at the cradle centre; R leaves from the
hold2f cradle position and the ball thrown by the left hand is already flying towards this hand, so the launch must also clear it.
Parameters: [forward, lateral, vertical speed, bank]. Objective: landing xy error at the left catch point + flight-time error +
penalty for any failure (collision, pose limit, drop) during the 0.45 s following the release.
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from cascade21 import TwoPlusOneCascade

TIME_W = 4.0


def evaluate(x, detail=False):
    c = TwoPlusOneCascade()
    c.planning = False
    c.warm['throw:right:2'] = np.asarray(x[:4], float); c.execution_warm['throw:right:2'] = np.asarray(x[:4], float)
    t_fail = None
    try:
        while c.data.time < 2.2 and c.throw_count < 3:
            c.step()
        if c.throw_count >= 3:
            t_end = c.data.time + 0.45
            while c.data.time < t_end:
                c.step()
    except (RuntimeError, ValueError) as e:
        t_fail = str(e)[:70]
    rel = [e for e in c.events if e['kind'] == 'release' and e['side'] == 'right']
    if len(rel) < 2:
        return (60.0, t_fail or 'no second right release') if detail else 60.0
    e = rel[1]; tgt = c.hands['left']['catch_p'] + [0, 0, 0.0223]
    p = np.array(e['position']); v = np.array(e['velocity']); g = c.gravity_m_s2
    rem = (v[2] + np.sqrt(max(0.0, v[2] ** 2 + 2 * g * (p[2] - tgt[2])))) / g
    xy = p[:2] + v[:2] * rem - tgt[:2]
    flight = rem + (e['time'] - e['separation_time'])
    cost = float(np.linalg.norm(xy)) + TIME_W * abs(flight - c.flight) + (10.0 if t_fail else 0.0)
    return (cost, dict(xy=xy.round(3).tolist(), flight=round(flight, 3), v=v.round(2).tolist(), t=e['time'], fail=t_fail)) if detail else cost


def _ev(x):
    return evaluate(x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gens", type=int, default=40); ap.add_argument("--lam", type=int, default=72)
    ap.add_argument("--procs", type=int, default=18); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "second_throw_calibration.json"))
    a = ap.parse_args()
    seed = json.loads((Path(__file__).parents[1] / "src/jugglebot/humanoid/leap_cascade_seed_plans.json").read_text())["forecast"]["throw:right:2"][:4]
    mean = np.array(seed, float); sig = np.array([0.15, 0.2, 0.3, 0.03]); rng = np.random.default_rng(a.seed); base_sig = sig.copy()
    lam, mu = a.lam, a.lam // 4; w = np.log(mu + .5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    best = (1e9, None); scale = .8; t0 = time.time()
    with Pool(a.procs) as pool:
        for g in range(a.gens):
            pop = mean + scale * sig * rng.standard_normal((lam, 4))
            costs = np.array(pool.map(_ev, list(pop))); o = np.argsort(costs)
            if costs[o[0]] < best[0]: best = (float(costs[o[0]]), pop[o[0]].copy())
            sel = pop[o[:mu]]; new = w @ sel
            sig = 0.8 * sig + 0.2 * np.maximum(np.sqrt(w @ (sel - mean) ** 2), 0.03 * base_sig); mean = new; scale = max(0.4, scale * .985)
            print(f"[{time.time()-t0:4.0f}s] gen {g:2d} best {best[0]:.4f} genbest {costs[o[0]]:.4f} med {np.median(costs):.2f}", flush=True)
            Path(a.out).write_text(json.dumps(dict(cost=best[0], command=best[1].tolist())))
    print("FINAL", best[0], evaluate(best[1], detail=True), best[1].round(4).tolist())

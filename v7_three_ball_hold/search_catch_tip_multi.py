"""Multi-condition re-optimisation of the dynamic fingertip ending catch (warm start from search_catch_tip.py result).
Conditions = (relative speed, landing offset) pairs; cost = mean + 0.5*max over conditions."""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import search_catch_tip as S

CONDS = [(2.5, (0, 0)), (2.5, (.01, .01)), (2.5, (-.01, -.01)), (2.5, (.01, -.01)), (2.5, (-.01, .01)), (1.5, (0, 0)), (3.5, (0, 0)), (3.5, (.01, 0)), (1.5, (-.01, 0)), (3.5, (-.01, 0))]
_P = None; BASE = {}


def _init(cfg):
    global _P, BASE
    _P = S.Problem("right", 10.0); BASE = cfg


def relaxed_ok(m):
    return (m["a_drift"] < 0.02 and m["b_speed"] < 0.05 and m["a_speed"] < 0.05 and m["nb"] >= 2 and "rf" in m["a_parts"] and m["wrist"] == 0
            and np.linalg.norm(np.array(m["b_pos"]) - S.B_TGT) < 0.08)


def cond_cost(x, cond):
    cfg = dict(BASE); cfg["speed"] = cond[0]; cfg["off"] = [cond[1][0], cond[1][1], 0.0]
    rig, a0, bb, _ = S.run(x, P=_P, cfg=cfg); m = S.metrics(rig, a0, bb)
    dist = float(np.linalg.norm(np.array(m["b_pos"]) - S.B_TGT))
    c = 15 * min(m["a_drift"], .3) + 8 * min(max(0, dist - .03), .3) + 2 * m["b_speed"] + 2 * m["a_speed"] + 4.0 * m["wrist"] + 0.4 * (m["nb"] < 2) + 0.4 * ("rf" not in m["a_parts"])
    return c, relaxed_ok(m)


def multi(x):
    r = [cond_cost(x, c) for c in CONDS]; cs = np.array([a for a, _ in r])
    return cs.mean() + 0.5 * cs.max(), sum(b for _, b in r)


def _m(x):
    return multi(x)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--gens", type=int, default=40); ap.add_argument("--lam", type=int, default=40)
    ap.add_argument("--procs", type=int, default=10); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args(); d = json.loads(Path(a.init).read_text()); cfg = d["cfg"]
    mean = np.array(d["x"]); base_sig = np.r_[np.full(16, .2), np.full(16, .12), .03, .03]; sig = base_sig.copy()
    rng = np.random.default_rng(a.seed); n, lam = len(mean), a.lam; mu = lam // 4
    w = np.log(mu + .5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    best = (1e9, None); scale = .8; t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(cfg,)) as pool:
        for g in range(a.gens):
            pop = np.vstack([mean, mean + scale * sig * rng.standard_normal((lam - 1, n))])
            costs = np.array(pool.map(_m, list(pop))); o = np.argsort(costs)
            if costs[o[0]] < best[0]: best = (float(costs[o[0]]), pop[o[0]].copy())
            sel = pop[o[:mu]]; mean = w @ sel
            sig = 0.8 * sig + 0.2 * np.maximum(np.sqrt(w @ (sel - mean) ** 2), 0.02 * base_sig); scale = max(0.4, scale * .985)
            print(f"[{time.time()-t0:5.0f}s] gen {g:3d} best {best[0]:.3f} genbest {costs[o[0]]:.3f} med {np.median(costs):.2f}", flush=True)
            Path(a.out).write_text(json.dumps(dict(cost=best[0], cfg=cfg, x=best[1].tolist())))
    _init(cfg); c, k = multi(best[1]); print("FINAL", c, f"{k}/{len(CONDS)} conditions ok")


if __name__ == "__main__":
    main()

"""Optimise the ending posture: finger target `hold` (16) + closing time, so that alpha (wrist pocket) is retained
laterally while beta is caught, over catch-position offsets (0,0),(+-1cm,0),(0,+-1cm),(+1,+1)cm."""
import sys, json, time, argparse
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from end_catch2 import run, score, Problem, frame_data
OFFS = [(0, 0), (0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01), (0.01, 0.01), (-0.01, -0.01)]
P = None; FD = None
def init():
    global P, FD
    P = Problem("right", 10.0); FD = frame_data()
def cost(z):
    x = np.zeros(21); x[:16] = z[:16]; tc = float(np.clip(z[16], 1.0, 1.25))
    c = 0.0
    for off in OFFS:
        rig, a0, _ = run(P, x, t_close=tc, close_dur=0.10, offset=off, fd=FD)
        s = score(rig, a0)
        c += min(s["alpha_drift"], 1.0) * 4 + min(np.linalg.norm(np.array(s["beta"]) - [0, 0, .022]), .3) * 6 + 2 * min(s["alpha_speed"], 2) + 2 * min(s["beta_speed"], 2) + (0 if s["ok"] else 0.5)
    return c / len(OFFS)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gens", type=int, default=40); ap.add_argument("--lam", type=int, default=54); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    x0 = np.array(json.loads(Path(a.init).read_text())["x"])
    mean = np.r_[x0[:16], 1.27 - 0.07]; sig = np.r_[np.full(16, .2), .05]
    rng = np.random.default_rng(a.seed); n = 17; lam = a.lam; mu = lam // 4
    w = np.log(mu + .5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    best = (1e9, None); scale = .7; t0 = time.time()
    with Pool(18, initializer=init) as pool:
        for g in range(a.gens):
            pop = mean + scale * sig * rng.standard_normal((lam, n))
            cs = np.array(pool.map(cost, list(pop))); o = np.argsort(cs)
            if cs[o[0]] < best[0]: best = (float(cs[o[0]]), pop[o[0]].copy())
            sel = pop[o[:mu]]; new = w @ sel
            sig = 0.8 * sig + 0.2 * np.maximum(np.sqrt(w @ (sel - mean) ** 2), 0.02 * np.r_[np.full(16, .2), .05]); mean = new; scale = max(.4, scale * .98)
            print(f"[{time.time()-t0:5.0f}s] gen {g} best {best[0]:.3f} genbest {cs[o[0]]:.3f} med {np.median(cs):.2f}", flush=True)
            Path(a.out).write_text(json.dumps(dict(side="right", tilt_deg=10.0, cost=best[0], x=best[1][:16].tolist() + [0, 0, 0, 0, 0], t_close=float(np.clip(best[1][16], 1.0, 1.25)))))

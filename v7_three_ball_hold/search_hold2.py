"""Search for a stable two-ball resting grasp on one LEAP hand (physics-in-the-loop, sep-CMA-ES)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from feaslib import HandRig, R


def tilt_g(deg, roll_deg=0.0, scale=1.0):
    t, r = np.deg2rad(deg), np.deg2rad(roll_deg)
    return 9.81 * scale * np.array([np.sin(r) * np.cos(t), np.sin(t), -np.cos(t) * np.cos(r)])


class Problem:
    def __init__(self, side, tilt_deg):
        self.rig = HandRig(side, n_balls=3)
        self.side = side
        f = self.rig.fingers
        self.flat, self.cradle = f.get_ctrl_array("flat"), f.get_ctrl_array("cradle")
        self.lo, self.hi = self.rig.ctrl_lo, self.rig.ctrl_hi
        self.tilt = tilt_deg
        self.sgn = 1.0 if side == "right" else -1.0
        # x = [16 ctrl] + [x1, y1, x2, y2, z2]
        self.x0 = np.r_[self.cradle, 0.0, -0.005, 0.0, 0.095, 0.0]
        self.sig0 = np.r_[np.full(16, 0.25), 0.01, 0.01, 0.01, 0.01, 0.01]

    def decode(self, x):
        ctrl = np.clip(x[:16], self.lo, self.hi)
        b1 = [x[16], x[17], 0.022]
        b2 = [x[18], x[19], x[20] + 0.022]
        return ctrl, b1, b2

    def simulate(self, x, T=0.9, tilt=None, g_local=None, settle_close=True):
        rig = self.rig
        ctrl, b1, b2 = self.decode(x)
        rig.set_gravity_local(tilt_g(self.tilt if tilt is None else tilt) if g_local is None else g_local)
        rig.reset(self.flat, ctrl, [b1, b2])
        rig.step(int(T * 1000))
        return rig

    def metrics(self, rig):
        out = []
        for i in range(2):
            n, fs, parts = rig.ball_hand_contacts(i)
            out.append(dict(pos=rig.ball_local(i), speed=float(np.linalg.norm(rig.ball_vel_local(i))), n=n, f=fs, parts=sorted(parts)))
        return out

    def cost(self, x):
        rig = self.simulate(x)
        # accumulate the late-window motion as well
        c = 0.0
        ms = self.metrics(rig)
        for mt in ms:
            p = mt["pos"]
            out = max(0.0, np.linalg.norm(p - np.array([0, 0.04, 0.0])) - 0.10)
            c += 40 * out + 2.0 * mt["speed"] + (0.4 if mt["n"] == 0 else 0)
        dist = np.linalg.norm(ms[0]["pos"] - ms[1]["pos"])
        c += 20 * max(0.0, 2 * R - dist + 0.001)  # overlap (should not happen physically)
        c += 3 * max(0.0, dist - 2 * R - 0.01)    # balls drifting apart means not jointly cupped
        # extra: avoid solutions riding on very different contact count
        return c


def sep_cma(prob, gens, lam, seed, log):
    rng = np.random.default_rng(seed)
    n = len(prob.x0)
    mean, sig = prob.x0.copy(), prob.sig0.copy()
    mu = lam // 4
    w = np.log(mu + .5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    best = (1e9, None)
    scale = 0.6
    for g in range(gens):
        z = rng.standard_normal((lam, n))
        pop = mean + scale * sig * z
        costs = np.array([prob.cost(p) for p in pop])
        order = np.argsort(costs)
        if costs[order[0]] < best[0]:
            best = (float(costs[order[0]]), pop[order[0]].copy())
        sel = pop[order[:mu]]
        new_mean = w @ sel
        sig = 0.8 * sig + 0.2 * np.maximum(np.sqrt(w @ (sel - mean) ** 2), 0.02 * prob.sig0)
        mean = new_mean
        scale = max(0.35, scale * 0.97)
        log(f"gen {g:3d} best {best[0]:.4f} gen-best {costs[order[0]]:.4f} median {np.median(costs):.3f}")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right")
    ap.add_argument("--tilt", type=float, default=10.0)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--lam", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prob = Problem(a.side, a.tilt)
    t0 = time.time()
    best_cost, best_x = sep_cma(prob, a.gens, a.lam, a.seed, lambda s: print(f"[{time.time()-t0:5.0f}s] {s}", flush=True))
    rig = prob.simulate(best_x)
    ms = prob.metrics(rig)
    print("BEST", best_cost)
    for i, mt in enumerate(ms):
        print(i, mt["pos"].round(4), round(mt["speed"], 4), mt["n"], round(mt["f"], 2), mt["parts"])
    Path(a.out).write_text(json.dumps(dict(side=a.side, tilt_deg=a.tilt, cost=best_cost, x=best_x.tolist())))

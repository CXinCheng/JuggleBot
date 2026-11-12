"""Ending, human-style, FREE approach: alpha at the fingertips, beta falls onto a palm slot moved toward the heel.

Unlike search_end_tip.py (v6 recorded approach), beta is released from 0.25 m above/behind the slot with a chosen
relative velocity in the hand frame (we may command the hand to match the ball's horizontal motion).
Approach: --az (heading in the seat x-y plane, deg; 90 = travelling toward the fingertips, i.e. coming from the heel),
--el (descent angle from horizontal, deg; 90 = vertical), --speed (relative speed at start).
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from replay_throw import Problem
from search_hold2 import tilt_g
from search_hold2c import non_hand_contacts
from feaslib import R

SLOT = np.array([0.0, -0.025, 0.022])
CFG = {}
_P = None


def _init(cfg):
    global _P, CFG
    _P = Problem("right", 10.0); CFG = cfg


def run(x, P=None, cfg=None, trace=False):
    P = P or _P; cfg = cfg or CFG
    rig = P.rig; f = rig.fingers
    Ka = np.clip(x[:16], rig.ctrl_lo, rig.ctrl_hi); Kb = np.clip(x[16:32], rig.ctrl_lo, rig.ctrl_hi)
    alpha = np.array(x[32:35]); t_close = float(np.clip(x[35], 0.05, 0.30)); dur = float(np.clip(x[36], 0.04, 0.25))
    rig.set_gravity_local(tilt_g(10.0))
    rig.reset(P.flat, Ka, [None, alpha]); rig.step(300)
    a0 = rig.ball_local(1).copy()
    az, el = np.deg2rad(cfg["az"]), np.deg2rad(cfg["el"])
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)])
    start = SLOT - 0.25 * d + np.array(cfg.get("off", [0., 0., 0.]))
    q0, v0 = rig.balls[0]; _, Rm = rig.seat_frame()
    rig.d.qpos[q0:q0 + 3] = rig.to_world(start); rig.d.qvel[v0:v0 + 3] = Rm @ (cfg["speed"] * d)
    hist = []
    for k in range(1100):
        tt = k / 1000.0
        a = np.clip((tt - t_close) / dur, 0, 1); a = a ** 3 * (10 - 15 * a + 6 * a * a)
        rig.d.ctrl[f.act_indices] = (1 - a) * Ka + a * Kb
        rig.step(1)
        if trace and k % 50 == 0:
            hist.append((tt, rig.ball_local(1).copy(), rig.ball_local(0).copy()))
    return rig, a0, hist


def metrics(rig, a0):
    a, b = rig.ball_local(1), rig.ball_local(0)
    n, fs, parts = rig.ball_hand_contacts(0); na, fa, pa = rig.ball_hand_contacts(1)
    return dict(alpha_drift=float(np.linalg.norm(a - a0)), beta_err=float(np.linalg.norm(b - SLOT)), beta_speed=float(np.linalg.norm(rig.ball_vel_local(0))),
                alpha_speed=float(np.linalg.norm(rig.ball_vel_local(1))), beta_contacts=n, alpha_contacts=na, alpha_force=float(fa), beta_force=float(fs),
                wrist=float(non_hand_contacts(rig)), beta_parts=sorted(parts), alpha_parts=sorted(pa), alpha_pos=a.round(3).tolist(), beta_pos=b.round(3).tolist())


def cost(x):
    rig, a0, _ = run(x); m = metrics(rig, a0)
    c = 15 * min(m["alpha_drift"], .3) + 12 * min(m["beta_err"], .3) + 2 * m["beta_speed"] + 2 * m["alpha_speed"]
    return c + 0.4 * (m["beta_contacts"] < 2) + 0.4 * (m["alpha_contacts"] < 2) + 4.0 * m["wrist"]


def _cost(x):
    return cost(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", required=True); ap.add_argument("--az", type=float, required=True); ap.add_argument("--el", type=float, required=True)
    ap.add_argument("--speed", type=float, default=1.5); ap.add_argument("--gens", type=int, default=60); ap.add_argument("--lam", type=int, default=72)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True); ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    cfg = dict(az=a.az, el=a.el, speed=a.speed)
    h = np.array(json.loads(Path(a.hold).read_text())["x"]); P = Problem("right", 10.0); f = P.rig.fingers
    catch = f.get_ctrl_array("catch"); holders = np.array([d in ("if", "mf", "th") for d in f.actuator_digits])
    mean = np.r_[np.where(holders, h[:16], catch), h[:16], h[18], h[19], h[20] + 0.022, 0.2, 0.10]
    base_sig = np.r_[np.full(16, 0.25), np.full(16, 0.15), 0.01, 0.01, 0.01, 0.03, 0.03]; sig = base_sig.copy()
    rng = np.random.default_rng(a.seed); n, lam = len(mean), a.lam; mu = lam // 4
    w = np.log(mu + .5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    best = (1e9, None); scale = .8; t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(cfg,)) as pool:
        for g in range(a.gens):
            pop = mean + scale * sig * rng.standard_normal((lam, n))
            costs = np.array(pool.map(_cost, list(pop))); o = np.argsort(costs)
            if costs[o[0]] < best[0]: best = (float(costs[o[0]]), pop[o[0]].copy())
            sel = pop[o[:mu]]; new = w @ sel
            sig = 0.8 * sig + 0.2 * np.maximum(np.sqrt(w @ (sel - mean) ** 2), 0.02 * base_sig)
            mean = new; scale = max(0.4, scale * .985)
            print(f"[{time.time()-t0:5.0f}s] gen {g:3d} best {best[0]:.3f} genbest {costs[o[0]]:.3f} med {np.median(costs):.2f}", flush=True)
            Path(a.out).write_text(json.dumps(dict(cost=best[0], cfg=cfg, x=best[1].tolist())))
    _init(cfg); rig, a0, _ = run(best[1], P=Problem("right", 10.0), cfg=cfg)
    print("FINAL", cfg, best[0], metrics(rig, a0))


if __name__ == "__main__":
    main()

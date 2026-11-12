"""Dynamic ending catch: palm ball A held by ring finger + palm (variant A of pinky_A.json); last ball B falls from above/side and is
caught and held by index/middle/thumb at the fingertips. balls[0]=B (falling), balls[1]=A (held). Params: Ka(16) pre-catch posture,
Kb(16) final posture, t_close, dur. Target B pose and A pose taken from results/pinky_A.json.
--az heading in seat x-y plane (deg), --el descent angle (90 = vertical), --speed relative speed at start.
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from search_hold2 import Problem, tilt_g
from search_hold2c import non_hand_contacts
from search_pinky import klass

PINKY = np.array(json.loads((Path(__file__).parent / "results" / "pinky_A.json").read_text())["x"])
A_LOC = np.array([PINKY[16], PINKY[17], 0.022]); B_TGT = np.array([PINKY[18], PINKY[19], PINKY[20] + 0.022])
CFG = {}; _P = None


def _init(cfg):
    global _P, CFG
    _P = Problem("right", 10.0); CFG = cfg


def run(x, P=None, cfg=None, hist=False):
    P = P or _P; cfg = cfg or CFG; rig = P.rig; f = rig.fingers
    Ka = np.clip(x[:16], rig.ctrl_lo, rig.ctrl_hi); Kb = np.clip(x[16:32], rig.ctrl_lo, rig.ctrl_hi)
    t_close = float(np.clip(x[32], 0.05, 0.35)); dur = float(np.clip(x[33], 0.04, 0.25))
    rig.set_gravity_local(tilt_g(10.0))
    rig.reset(P.flat, Ka, [None, A_LOC]); rig.step(300)
    a0 = rig.ball_local(1).copy()
    az, el = np.deg2rad(cfg["az"]), np.deg2rad(cfg["el"])
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)])
    start = B_TGT - 0.25 * d + np.array(cfg.get("off", [0., 0., 0.]))
    q0, v0 = rig.balls[0]; _, Rm = rig.seat_frame()
    rig.d.qpos[q0:q0 + 3] = rig.to_world(start); rig.d.qvel[v0:v0 + 3] = Rm @ (cfg["speed"] * d)
    maxbb = 0.0; hs = []
    for k in range(1100):
        tt = k / 1000.0
        a = np.clip((tt - t_close) / dur, 0, 1); a = a ** 3 * (10 - 15 * a + 6 * a * a)
        rig.d.ctrl[f.act_indices] = (1 - a) * Ka + a * Kb
        rig.step(1)
        if k % 5 == 0: maxbb = max(maxbb, rig.ball_ball_contact_force(0, 1))
        if hist and k % 50 == 0: hs.append((tt, rig.ball_local(0).copy(), rig.ball_local(1).copy()))
    return rig, a0, maxbb, hs


def metrics(rig, a0, maxbb):
    a, b = rig.ball_local(1), rig.ball_local(0)
    nb, fb, pb = rig.ball_hand_contacts(0); na, fa, pa = rig.ball_hand_contacts(1)
    cA, cB = klass(pa), klass(pb)
    return dict(a_drift=float(np.linalg.norm(a - a0)), b_err=float(np.linalg.norm(b - B_TGT)), b_speed=float(np.linalg.norm(rig.ball_vel_local(0))),
                a_speed=float(np.linalg.norm(rig.ball_vel_local(1))), nb=nb, na=na, fb=float(fb), fa=float(fa), wrist=float(non_hand_contacts(rig)), maxbb=float(maxbb),
                a_parts=sorted(cA), b_parts=sorted(cB), a_pos=a.round(3).tolist(), b_pos=b.round(3).tolist())


def ok(m):
    return m["a_drift"] < 0.02 and m["b_err"] < 0.03 and m["b_speed"] < 0.05 and m["a_speed"] < 0.05 and m["nb"] >= 2 and "rf" in m["a_parts"] and m["wrist"] == 0


def cost(x):
    rig, a0, bb, _ = run(x); m = metrics(rig, a0, bb)
    c = 15 * min(m["a_drift"], .3) + 12 * min(m["b_err"], .3) + 2 * m["b_speed"] + 2 * m["a_speed"] + 4.0 * m["wrist"] + 0.01 * min(bb, 50)
    return c + 0.4 * (m["nb"] < 2) + 0.4 * ("rf" not in m["a_parts"])


def _cost(x):
    return cost(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--az", type=float, required=True); ap.add_argument("--el", type=float, required=True); ap.add_argument("--speed", type=float, default=1.5)
    ap.add_argument("--gens", type=int, default=60); ap.add_argument("--lam", type=int, default=72); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=6); ap.add_argument("--out", required=True)
    a = ap.parse_args(); cfg = dict(az=a.az, el=a.el, speed=a.speed)
    P = Problem("right", 10.0); f = P.rig.fingers
    catch = f.get_ctrl_array("catch"); catchers = np.array([dg in ("if", "mf", "th") for dg in f.actuator_digits])
    Kb0 = PINKY[:16]
    mean = np.r_[np.where(catchers, catch, Kb0), Kb0, 0.2, 0.10]
    base_sig = np.r_[np.full(16, .25), np.full(16, .15), .03, .03]; sig = base_sig.copy()
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
    _init(cfg); rig, a0, bb, _ = run(best[1], P=Problem("right", 10.0), cfg=cfg); m = metrics(rig, a0, bb)
    print("FINAL", cfg, best[0], "OK" if ok(m) else "FAIL", m)


if __name__ == "__main__":
    main()

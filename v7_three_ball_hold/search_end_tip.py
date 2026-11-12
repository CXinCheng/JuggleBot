"""Ending, human-style, optimised: alpha pinched at the fingertips, beta (v6 ball 1 final approach) caught by the palm.

Parameters (38): Ka[16] pre-catch finger targets (alpha holders closed, rest open bowl), Kb[16] post-catch hold targets,
alpha slot (ax, ay, az in seat frame), t_close, close_dur. Hand motion = recorded v6 final approach (as end_catch2).
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from end_catch import frame_data, rel_series
from end_catch2 import smooth
from replay_throw import g_local_series, Problem
from search_hold2 import tilt_g
from search_hold2c import non_hand_contacts
from feaslib import R

K_START, HANDOFF, K_STOP, K_END = 940, 1070, 1320, 1700
SLOT = np.array([0.012, -0.012, 0.022])
_P = None; _FD = None; _GL = None; _REL = None


def _init():
    global _P, _FD, _GL, _REL
    _P = Problem("right", 10.0); _FD = frame_data()
    _GL = g_local_series(_FD[0], _FD[1] if False else _FD[2], _FD[4]); _REL = rel_series(_FD)


def run(x, P=None, fd=None, gl=None, rel=None, trace=False, offset=(0., 0.)):
    P = P or _P; fd = fd or _FD; gl = gl if gl is not None else _GL; rel = rel or _REL
    rig = P.rig; f = rig.fingers
    Ka = np.clip(x[:16], rig.ctrl_lo, rig.ctrl_hi); Kb = np.clip(x[16:32], rig.ctrl_lo, rig.ctrl_hi)
    alpha = np.array(x[32:35]); t_close = float(np.clip(x[35], 1.10, 1.40)); dur = float(np.clip(x[36], 0.04, 0.25))
    rig.set_gravity_local(tilt_g(10.0))
    rig.reset(P.flat, Ka, [None, alpha])      # ball1 = alpha, ball0 = beta (guided)
    rig.step(300)
    a0 = rig.ball_local(1).copy()
    q0, v0 = rig.balls[0]; _, Rm = rig.seat_frame()
    relp, relv = rel[0] + np.array([offset[0], offset[1], 0.]), rel[1]
    hist = []
    for k in range(K_START, K_END):
        rig.set_gravity_local(gl[k] if k < K_STOP else tilt_g(10.0))
        if k < HANDOFF:
            rig.d.qpos[q0:q0 + 3] = rig.to_world(relp[k]); rig.d.qvel[v0:v0 + 3] = Rm @ relv[k]
        a = smooth((k / 1000.0 - t_close) / dur)
        rig.d.ctrl[f.act_indices] = (1 - a) * Ka + a * Kb
        rig.step(1)
        if trace and k % 25 == 0:
            hist.append((k / 1000.0, rig.ball_local(1).copy(), rig.ball_local(0).copy(), rig.ball_hand_contacts(0)[0]))
    return rig, a0, hist


def metrics(rig, a0):
    a, b = rig.ball_local(1), rig.ball_local(0)
    n, fs, parts = rig.ball_hand_contacts(0)
    na, fa, pa = rig.ball_hand_contacts(1)
    return dict(alpha_drift=float(np.linalg.norm(a - a0)), beta_err=float(np.linalg.norm(b - SLOT)), beta_speed=float(np.linalg.norm(rig.ball_vel_local(0))),
                alpha_speed=float(np.linalg.norm(rig.ball_vel_local(1))), beta_contacts=n, alpha_contacts=na, alpha_force=float(fa), beta_force=float(fs),
                wrist=float(non_hand_contacts(rig)), beta_parts=sorted(parts), alpha_parts=sorted(pa), alpha_pos=a.round(3).tolist(), beta_pos=b.round(3).tolist())


def cost(x):
    rig, a0, _ = run(x)
    m = metrics(rig, a0)
    c = 15 * min(m["alpha_drift"], .3) + 12 * min(m["beta_err"], .3) + 2 * m["beta_speed"] + 2 * m["alpha_speed"]
    c += 0.4 * (m["beta_contacts"] < 2) + 0.4 * (m["alpha_contacts"] < 2) + 4.0 * m["wrist"]
    return c


def _cost(x):
    return cost(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", required=True); ap.add_argument("--gens", type=int, default=70); ap.add_argument("--lam", type=int, default=72)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True); ap.add_argument("--procs", type=int, default=9)
    a = ap.parse_args()
    h = np.array(json.loads(Path(a.hold).read_text())["x"])
    P = Problem("right", 10.0); f = P.rig.fingers
    catch = f.get_ctrl_array("catch")
    holders = np.array([d in ("if", "mf", "th") for d in f.actuator_digits])
    Ka = np.where(holders, h[:16], catch)           # alpha holders keep the pinch, others form the open bowl
    mean = np.r_[Ka, h[:16], h[18], h[19], h[20] + 0.022, 1.27, 0.10]
    sig = np.r_[np.full(16, 0.25), np.full(16, 0.15), 0.01, 0.01, 0.01, 0.03, 0.03]
    rng = np.random.default_rng(a.seed); n, lam = len(mean), a.lam; mu = lam // 4
    w = np.log(mu + .5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    best = (1e9, None); scale = .8; t0 = time.time()
    with Pool(a.procs, initializer=_init) as pool:
        for g in range(a.gens):
            pop = mean + scale * sig * rng.standard_normal((lam, n))
            costs = np.array(pool.map(_cost, list(pop)))
            o = np.argsort(costs)
            if costs[o[0]] < best[0]: best = (float(costs[o[0]]), pop[o[0]].copy())
            sel = pop[o[:mu]]; new = w @ sel
            sig = 0.8 * sig + 0.2 * np.maximum(np.sqrt(w @ (sel - mean) ** 2), 0.02 * np.r_[np.full(16, 0.25), np.full(16, 0.15), 0.01, 0.01, 0.01, 0.03, 0.03])
            mean = new; scale = max(0.4, scale * .985)
            print(f"[{time.time()-t0:5.0f}s] gen {g:3d} best {best[0]:.3f} genbest {costs[o[0]]:.3f} med {np.median(costs):.2f}", flush=True)
            Path(a.out).write_text(json.dumps(dict(cost=best[0], x=best[1].tolist())))
    _init(); rig, a0, _ = run(best[1], P=Problem("right", 10.0))
    print("FINAL", best[0], metrics(rig, a0))


if __name__ == "__main__":
    main()

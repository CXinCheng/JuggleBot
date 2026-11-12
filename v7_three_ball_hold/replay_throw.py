"""Replay the real v6 right-hand throw (palm accelerations + orientation) on a two-ball grasp.

The hand frame is handled as an accelerating/rotating frame: effective gravity in the seat frame is
R(t)^T (g - a(t)). Rotational inertial terms (omega x omega x r, alpha x r) are neglected here.
"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).parent))
from search_hold2 import Problem

REC = Path(__file__).parent / "results" / "v6_right_throw.npz"


def load_rec(smooth=7):
    z = np.load(REC)
    t, pos, rot, ctrl = z["t"], z["pos"], z["rot"].reshape(-1, 3, 3), z["ctrl"]
    k = np.ones(smooth) / smooth
    ps = np.stack([np.convolve(np.pad(pos[:, i], smooth // 2, mode="edge"), k, mode="valid") for i in range(3)], 1)
    dt = t[1] - t[0]
    acc = np.gradient(np.gradient(ps, dt, axis=0), dt, axis=0)
    acc = np.stack([np.convolve(np.pad(acc[:, i], smooth // 2, mode="edge"), k, mode="valid") for i in range(3)], 1)
    return t, ps, rot, ctrl, acc


def g_local_series(t, rot, acc):
    g = np.array([0, 0, -9.81])
    return np.stack([rot[i].T @ (g - acc[i]) for i in range(len(t))])


def finger_progress(rec_ctrl, cradle, flat):
    span = flat - cradle
    w = np.zeros_like(rec_ctrl)
    big = np.abs(span) > 0.05
    w[:, big] = np.clip((rec_ctrl[:, big] - cradle[big]) / span[big], 0, 1)
    return w


def run(prob, x, t_end=0.95, verbose=True, release_scale=1.0, t_start=0.1):
    t, ps, rot, ctrl, acc = load_rec()
    gl = g_local_series(t, rot, acc)
    rig = prob.rig
    f = rig.fingers
    cradle, flat = f.get_ctrl_array("cradle"), f.get_ctrl_array("flat")
    w = finger_progress(ctrl, cradle, flat) * release_scale
    hold = np.clip(x[:16], rig.ctrl_lo, rig.ctrl_hi)
    k0 = int(t_start / (t[1] - t[0]))
    prob.simulate(x, T=0.6)  # settle at the 10 deg loaded tilt
    log = []
    n = int(min(t_end, t[-1]) / (t[1] - t[0]))
    dt_rec = t[1] - t[0]
    steps_per = max(1, int(round(dt_rec / rig.m.opt.timestep)))
    for k in range(k0, n):
        rig.set_gravity_local(gl[k])
        rig.d.ctrl[f.act_indices] = hold + w[k] * (flat - hold)
        rig.step(steps_per)
        if k % 10 == 0:
            log.append((t[k], rig.ball_local(0).copy(), rig.ball_local(1).copy(), np.linalg.norm(gl[k]) ))
    return rig, log, (t, gl, w)


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text())
    prob = Problem(cfg["side"], 10.0)
    x = np.array(cfg["x"])
    rig, log, (t, gl, w) = run(prob, x, t_start=float(sys.argv[2]) if len(sys.argv)>2 else 0.71, t_end=float(sys.argv[3]) if len(sys.argv)>3 else 1.5)
    for tt, a, b, gn in log:
        print(f"t={tt:.3f} |g_eff|={gn:5.1f}  A={a.round(3)}  B={b.round(3)}")
    print("release progress at end", w[-1].round(2))

"""E2: catch the second ball (beta) in the palm while the first ball (alpha) is already finger-caged.

Hand motion = recorded v6 right-hand empty return + catch (release 0.912 s, touch of ball 1 at 1.101 s, catch confirmed 1.27 s).
Beta (v6 ball 1, thrown by the left hand at 0.598 s) is initialised from its recorded world state converted into the
accelerating/rotating hand frame (including the omega x r term of the initial relative velocity).
"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).parent))
from replay_throw import load_rec, g_local_series, REC, Problem
from feaslib import R

K_START = 940  # ms; beta in flight, alpha caged, stroke finished (release at 912)


def frame_data():
    z = np.load(REC)
    t, ps, rot, ctrl, acc = load_rec()
    dt = t[1] - t[0]
    vh = np.gradient(ps, dt, axis=0)
    Rdot = np.gradient(rot, dt, axis=0)
    omega = np.zeros((len(t), 3))
    for k in range(len(t)):
        S = Rdot[k] @ rot[k].T
        omega[k] = [S[2, 1], S[0, 2], S[1, 0]]
    return t, ps, rot, ctrl, acc, vh, omega, z["balls_pos"], z["balls_vel"]


def place_beta(rig, idx, ps, rot, vh, omega, bp, bv, k):
    p_rel = bp[k, idx] - ps[k]
    v_rel = bv[k, idx] - vh[k] - np.cross(omega[k], p_rel)
    q0, v0 = rig.balls[idx]
    pl, vl = rot[k].T @ p_rel, rot[k].T @ v_rel
    rig.d.qpos[q0:q0 + 3] = rig.to_world(pl)
    _, Rm = rig.seat_frame()
    rig.d.qvel[v0:v0 + 3] = Rm @ vl
    return pl, vl


def rel_series(fd, idx=1):
    t, ps, rot = fd[0], fd[1], fd[2]; bp = fd[7]
    rel = np.einsum('kji,kj->ki', rot, bp[:, idx] - ps)          # R^T (p - p_seat)
    return rel, np.gradient(rel, t[1] - t[0], axis=0)


def run_catch(prob, x, finger_fn=None, k_start=K_START, k_end=1450, fd=None, trace=False, k_handoff=1085):
    """alpha = ball 0 settled in the finger cage; beta = ball 1 (v6 ball 1) caught by the palm."""
    fd = fd or frame_data()
    t, ps, rot, ctrl, acc, vh, omega, bp, bv = fd
    gl = g_local_series(t, rot, acc)
    rig = prob.rig; f = rig.fingers
    K0 = np.clip(x[:16], rig.ctrl_lo, rig.ctrl_hi)
    rig.set_gravity_local(np.array([0, np.sin(np.deg2rad(10)), -np.cos(np.deg2rad(10))]) * 9.81)
    rig.reset(prob.flat, K0, [[x[18], x[19], x[20] + 0.022]])      # alpha only, starts where B sat in the hold solution
    rig.step(600)
    alpha0 = rig.ball_local(0).copy()
    rel, relv = rel_series(fd)
    q1, v1 = rig.balls[1]
    _, Rm = rig.seat_frame()
    hist = []
    for k in range(k_start, k_end):
        rig.set_gravity_local(gl[k])
        if k < k_handoff:   # beta is guided along the recorded hand-frame trajectory, then released to physics
            rig.d.qpos[q1:q1 + 3] = rig.to_world(rel[k]); rig.d.qvel[v1:v1 + 3] = Rm @ relv[k]
        rig.d.ctrl[f.act_indices] = K0 if finger_fn is None else finger_fn(k / 1000.0, K0, rig)
        rig.step(1)
        if trace and k % 25 == 0:
            hist.append((k / 1000.0, rig.ball_local(0).copy(), rig.ball_local(1).copy(), rig.ball_hand_contacts(1)[0]))
    return rig, alpha0, hist, fd


def summarize(rig, alpha0, x):
    A = rig.ball_local(0); Bt = rig.ball_local(1)
    slot = np.array(x[16:19]) + [0, 0, 0]
    n, fs, parts = rig.ball_hand_contacts(1)
    return dict(alpha_drift=float(np.linalg.norm(A - alpha0)), beta=Bt.round(3).tolist(), beta_speed=float(np.linalg.norm(rig.ball_vel_local(1))),
                beta_to_slot=float(np.linalg.norm(Bt - slot)), beta_contacts=n, alpha_speed=float(np.linalg.norm(rig.ball_vel_local(0))),
                ab_gap=float(np.linalg.norm(A - Bt) - 2 * R), beta_parts=sorted(parts))


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text()); x = np.array(cfg["x"])
    P = Problem("right", 10.0)
    rig, a0, hist, fd = run_catch(P, x, trace=True)
    t, ps, rot = fd[0], fd[1], fd[2]; bp = fd[7]
    for h in hist:
        k = int(round(h[0] * 1000)); rec = rot[k].T @ (bp[k, 1] - ps[k])
        print(f"t={h[0]:.3f} alpha={h[1].round(3)} beta={h[2].round(3)} (v6 ball1 in seat frame {rec.round(3)}) nBeta={h[3]}")
    print(summarize(rig, a0, x))

"""E2 (revised): ending = alpha parked in the wrist-shell pocket, beta caught in the palm cup in front of it.

The pocket arrangement is the hold2b grasp (stable, but cannot be used for a later throw -- fine at the end).
Hand motion: recorded v6 final approach (from k_start ms); beta guided along its recorded hand-frame path until
`handoff`, then free physics. Fingers: open 'catch' bowl, then close to the hold2b posture over `close_dur` from `t_close`.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from end_catch import frame_data, rel_series, g_local_series, Problem
from search_hold2 import tilt_g
from feaslib import R

POCKET = [-0.0019, -0.0859, 0.0078]


def smooth(a):
    a = np.clip(a, 0, 1)
    return a ** 3 * (10 - 15 * a + 6 * a * a)


def run(prob, x, t_close=1.27, close_dur=0.10, handoff=1070, k_start=1000, k_end=1820, open_name="catch", fd=None, trace=False, use_v6_fingers=True, k_stop=1320, settle_s=0.5, final_tilt=10.0, offset=(0.0, 0.0)):
    fd = fd or frame_data()
    t, ps, rot = fd[0], fd[1], fd[2]
    gl = g_local_series(t, rot, fd[4])
    rig = prob.rig; f = rig.fingers
    hold = np.clip(x[:16], rig.ctrl_lo, rig.ctrl_hi); opn = f.get_ctrl_array(open_name)
    rig.set_gravity_local(tilt_g(0.0))
    rig.reset(prob.flat, fd[3][k_start], [POCKET]); rig.step(400)
    alpha0 = rig.ball_local(0).copy()
    rel, relv = rel_series(fd); rel = rel + np.array([offset[0], offset[1], 0.0]); q1, v1 = rig.balls[1]; _, Rm = rig.seat_frame()
    hist = []
    for k in range(k_start, k_end):
        rig.set_gravity_local(gl[k] if k < k_stop else tilt_g(final_tilt))
        if k < handoff:
            rig.d.qpos[q1:q1 + 3] = rig.to_world(rel[k]); rig.d.qvel[v1:v1 + 3] = Rm @ relv[k]
        a = smooth((k / 1000.0 - t_close) / close_dur)
        base = fd[3][min(k, k_stop)] if use_v6_fingers else opn
        rig.d.ctrl[f.act_indices] = (1 - a) * base + a * hold
        rig.step(1)
        if trace and k % 25 == 0:
            hist.append((k / 1000.0, rig.ball_local(0).copy(), rig.ball_local(1).copy(), rig.ball_hand_contacts(1)[0]))
    return rig, alpha0, hist


def score(rig, alpha0):
    A, Bt = rig.ball_local(0), rig.ball_local(1)
    n, fs, parts = rig.ball_hand_contacts(1)
    return dict(alpha_drift=float(np.linalg.norm(A - alpha0)), beta=Bt.round(3).tolist(), beta_speed=float(np.linalg.norm(rig.ball_vel_local(1))),
                alpha_speed=float(np.linalg.norm(rig.ball_vel_local(0))), beta_contacts=n, beta_force=float(fs),
                gap=float(np.linalg.norm(A - Bt) - 2 * R), beta_parts=sorted(parts),
                ok=bool(np.linalg.norm(A - alpha0) < .03 and np.linalg.norm(Bt - [0, 0, .022]) < .04 and np.linalg.norm(rig.ball_vel_local(1)) < .03 and np.linalg.norm(rig.ball_vel_local(0)) < .03 and n >= 2))


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text()); x = np.array(cfg["x"])
    P = Problem("right", 10.0)
    rig, a0, hist = run(P, x, trace=True)
    for h in hist: print(f"t={h[0]:.3f} alpha={h[1].round(3)} beta={h[2].round(3)} nBeta={h[3]}")
    print(score(rig, a0))

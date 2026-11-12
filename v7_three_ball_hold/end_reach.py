"""Arm reachability of the fingertip-ending catch inside the v6 cascade.

For the v6 recorded incoming ball (final approach to the right hand), choose a touchdown time Tc and an approach type
(direction in the hand frame + relative speed). That fixes the required palm position/orientation/velocity at touchdown:
  p_seat = p_ball(Tc) - R_h @ slot,   v_hand = v_ball(Tc) - speed * R_h @ d_local.
We solve 7-DOF IK (same limits/IK as the v6 controller) along the last 0.25 s of the approach and report joint-limit margin,
peak joint speed (Jacobian pseudo-inverse) and the required hand acceleration to reach v_hand from rest.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from jugglebot.humanoid.model import load_g1_dual_arm, Q_READY_ARM_RIGHT, BALL_RADIUS
from jugglebot.humanoid.contact_control import UprightKinematics, PREFERRED_THROW_RIGHT

REC = Path(__file__).parent / "results" / "v6_right_throw.npz"
SLOT = np.array([0.0, -0.025, 0.022])
G = np.array([0, 0, -9.81])


def make_kin():
    m = load_g1_dual_arm(n_balls=3)
    kin = UprightKinematics(m, "right", tcp_site="r_ball_seat")
    lower = np.array([-1.1, -0.65, -1.2, -1.02, -1.2, -0.7, -0.9]); upper = np.array([0.8, -0.03, 1.2, 1.2, 1.2, 1.5, 0.9])
    kin.q_min = np.maximum(kin.q_min, lower); kin.q_max = np.minimum(kin.q_max, upper)
    return m, kin


def hand_rot(tilt_deg=10.0):
    n = np.array([np.sin(np.deg2rad(tilt_deg)), 0, np.cos(np.deg2rad(tilt_deg))])
    f = np.array([1., 0, 0]) - n * n[0]; f /= np.linalg.norm(f)
    return np.column_stack([np.cross(f, n), f, n]), n


def plan(kin, p_ball0, v_ball0, tb0, Tc, d_local, speed, Rh, n, q_seed, T_pre=0.15, dt=0.01):
    """Return dict of reachability metrics for one (Tc, approach)."""
    ts = np.arange(Tc - T_pre, Tc + 1e-9, dt)
    pb = lambda t: p_ball0 + v_ball0 * (t - tb0) + 0.5 * G * (t - tb0) ** 2
    vb = lambda t: v_ball0 + G * (t - tb0)
    v_h = vb(Tc) - speed * (Rh @ d_local)          # constant hand velocity over the approach (straight line)
    # solve the touchdown pose first (continuation seed), then walk backwards along the approach
    q = q_seed.copy(); qs = []; ok = True; worst = 0.0; ok_touch = None
    for t in ts[::-1]:
        p_seat = pb(Tc) - Rh @ SLOT + v_h * (t - Tc)
        r = kin.solve(p_seat, n, q, q_nominal=PREFERRED_THROW_RIGHT)
        if ok_touch is None: ok_touch = bool(r.success)
        if not r.success: ok = False; worst = max(worst, r.pos_err)
        q = r.q; qs.append(q.copy())
    qs = np.array(qs)[::-1]
    dq = np.gradient(qs, dt, axis=0)
    margin = np.minimum(qs - kin.q_min, kin.q_max - qs).min()
    # joint speed required at touchdown from the Jacobian (translation only, orientation fixed)
    jp, jr = kin.jacobian(qs[-1])
    J = np.vstack([jp, jr]); qd = np.linalg.pinv(J, rcond=1e-3) @ np.r_[v_h, np.zeros(3)]
    return dict(ok=ok, ok_touch=ok_touch, ik_err=worst, margin_rad=float(margin), peak_qd_ik=float(np.abs(dq).max()), qd_touch=float(np.abs(qd).max()),
                v_hand=v_h.round(2).tolist(), p_seat=(pb(Tc) - Rh @ SLOT).round(3).tolist(), q=qs[-1].round(2).tolist())


def main_single():
    z = np.load(REC); t = z["t"]; bp = z["balls_pos"][:, 1]; bv = z["balls_vel"][:, 1]
    m, kin = make_kin(); Rh, n = hand_rot(10.0)
    k0 = int(0.95 / (t[1] - t[0])); tb0 = t[k0]
    print("v6 ball1 world state at 0.95 s:", bp[k0].round(3), bv[k0].round(2))
    k_touch = int(1.101 / (t[1] - t[0])); print("v6 seat at touch:", z["pos"][k_touch].round(3), " ball:", bp[k_touch].round(3))
    q_seed = PREFERRED_THROW_RIGHT.copy()
    types = {"vertical (az0,el90)": np.array([0, 0, -1.0]), "heel-side 60deg (az90,el60)": np.array([0, np.cos(np.deg2rad(60)), -np.sin(np.deg2rad(60))])}
    for name, d in types.items():
        for speed in (1.0, 1.5):
            for Tc in (1.00, 1.05, 1.10, 1.15):
                r = plan(kin, bp[k0], bv[k0], tb0, Tc, d, speed, Rh, n, q_seed)
                print(f"{name:28s} v_rel={speed} Tc={Tc:.2f} touch_ok={r['ok_touch']} approach_ok={r['ok']} ikerr={r['ik_err']:.4f} margin={r['margin_rad']:.2f} peak_qd={r['peak_qd_ik']:.1f} v_hand={r['v_hand']} seat={r['p_seat']}")


def workspace_scan(tilt_deg=10.0, speeds=(1.0, 1.5), T_pre=0.10):
    """Scan touchdown seat positions around the v6 catch point for the two successful approach types.
    Ball world velocity at touchdown is the v6 recorded one at ~1.10 s (a design choice: the left-hand throw can be retuned)."""
    z = np.load(REC); t = z["t"]; bv = z["balls_vel"][:, 1]
    vb = bv[int(1.10 / (t[1] - t[0]))]
    m, kin = make_kin(); Rh, n = hand_rot(tilt_deg)
    types = {"vertical": np.array([0, 0, -1.0]), "heel60": np.array([0, np.cos(np.deg2rad(60)), -np.sin(np.deg2rad(60))])}
    out = {}
    for name, d in types.items():
        for sp in speeds:
            v_h = vb - sp * (Rh @ d); n_ok = n_all = 0; best = None
            for x in np.arange(0.24, 0.361, 0.04):
                for y in np.arange(-0.30, -0.139, 0.04):
                    for zc in np.arange(0.80, 0.961, 0.04):
                        p_end = np.array([x, y, zc]); q = PREFERRED_THROW_RIGHT.copy(); good = True; mg = 9.0
                        for tau in np.arange(0.0, T_pre + 1e-9, 0.02):
                            r = kin.solve(p_end - v_h * tau, n, q, q_nominal=PREFERRED_THROW_RIGHT)
                            q = r.q
                            if not r.success: good = False; break
                            mg = min(mg, float(np.minimum(r.q - kin.q_min, kin.q_max - r.q).min()))
                        n_all += 1
                        if good:
                            jp, jr = kin.jacobian(r.q)  # velocity check at the earliest pose of the approach
                            jp0, jr0 = kin.jacobian(q); qd = np.linalg.pinv(np.vstack([jp0, jr0]), rcond=1e-3) @ np.r_[v_h, np.zeros(3)]
                            if np.abs(qd).max() < 15.0 and mg > 0.03:
                                n_ok += 1; best = best or (p_end.round(3).tolist(), float(np.abs(qd).max()), mg)
            out[(name, sp)] = (n_ok, n_all, v_h.round(2).tolist(), best)
            print(f"tilt {tilt_deg:>4}: {name:8s} v_rel {sp}: ok {n_ok}/{n_all}  v_hand {v_h.round(2).tolist()}  first-ok {best}", flush=True)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        for tl in (0.0, 10.0, 20.0):
            workspace_scan(tl)
    else:
        main_single()

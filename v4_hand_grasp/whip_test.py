"""E4 (v4): does a whip throw raise the hand's release speed under 180 deg/s?

The loop's throw (E2/E3b, handsim.LineStroke) translates the hand with a fixed orientation: with
the arm stretched out the joint rates partly cancel and the ball leaves at ~2 m/s at most. A whip
lets the hand turn, so the rates of the parallel pitch joints (shoulder_lift, elbow, wrist_1) add up
along the chain. With the cup this failed (v2: the curved seat path pinned the ball to the wall);
the hand grasps the ball, so the centripetal load is carried by the fingers and the release is
commanded by opening them.

Part A, instantaneous kinematic bound: at release configurations (bowl axis up, ball seat on a
grid in front of the robot), the largest vertical ball-seat speed with every arm joint <= 180 deg/s:
    translate : the hand does not turn (6-D task [v; 0]), as LineStroke
    whip4     : shoulder_pan + the three pitch joints, wrists 2/3 still, hand free to turn (v2's whip)
    whip6     : all six joints, hand free to turn (exact LP over the 3-D null space)

Part B, strokes (same speed profile as LineStroke: smoothstep to V over 1.5 V / a_up, stop over
1.5 V / a_down), simulated with the bowl hand of E3c and the E2 finger opening (the ball grasped at
the release pose and carried to the stroke start, then thrown as in hand_juggle.rollout):
    translate : LineStroke (fixed hand orientation)
    linewhip  : the ball seat moves on a straight line, whip4 joints, hand free to turn (v2)
    jointwhip : straight line in joint space (whip4 joints) -- the seat moves on an arc
For each, the largest V within 0.9 x 180 deg/s and 0.9 x the arm torque limits (feed-forward), joint
range and collisions is found by bisection; the throw is then rolled out and the actual release
speed, direction, spin and flight time (to a catch 0.1 / 0.2 m below the release) are reported.

Usage (from the repo root):
    uv run python v4_hand_grasp/whip_test.py
    uv run python v4_hand_grasp/whip_test.py --a-up 2 3 --a-down 2.5 3
"""

import argparse
import itertools
import sys

import mujoco
import numpy as np

from hand_juggle import Fingers, stroke_torque
from hand_throw import BALL_ON_PALM, FREE_CONFIRM, SPEED_MARGIN
from handsim import HandSim, LineStroke, Q_READY, load_model, set_bowl_sites
from kinematics import ArmKinematics

G = 9.81
WHIP = np.array([0, 1, 2, 3])
TORQUE_FRAC = 0.9


# ------------------------------------------------------------------ Part A
def minimax_affine(u: np.ndarray, N: np.ndarray) -> float:
    """min over r of max_i |u_i + (N r)_i| (N: n x k), by enumerating the LP's vertices."""
    n, k = N.shape
    best = np.inf
    rows = [(i, s) for i in range(n) for s in (1.0, -1.0)]  # s (u_i + N_i r) = t
    for combo in itertools.combinations(rows, k + 1):
        A = np.array([np.concatenate([s * N[i], [-1.0]]) for i, s in combo])
        b = np.array([-s * u[i] for i, s in combo])
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        r, t = x[:k], x[k]
        if t >= 0 and np.all(np.abs(u + N @ r) <= t + 1e-9):
            best = min(best, t)
    return best


def speed_bounds(kin: ArmKinematics, q: np.ndarray, d: np.ndarray) -> dict:
    """Largest seat speed along d with all |qd_i| <= pi, for the three primitives."""
    jp, jr = kin.jacobian(q)
    out = {}
    J6 = np.vstack([jp, jr])
    if np.linalg.cond(J6) < 1e6:
        out["translate"] = np.pi / np.max(np.abs(np.linalg.solve(J6, np.concatenate([d, np.zeros(3)]))))
    else:
        out["translate"] = 0.0
    for name, cols in (("whip4", WHIP), ("whip6", np.arange(6))):
        J = jp[:, cols]
        u = np.linalg.pinv(J, rcond=1e-6) @ d
        N = np.linalg.svd(J)[2][3:].T
        out[name] = np.pi / minimax_affine(u, N)
    return out


def part_a(sim: HandSim, base: np.ndarray) -> list:
    kb = ArmKinematics(sim.model, "bowl_seat")
    up = np.array([0.0, 0.0, 1.0])
    rows = []
    for x in (0.6, 0.7, 0.8, 0.9, 1.0):
        for z in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            p = base + np.array([x, 0.0, z])
            ik = kb.solve(p, up, Q_READY, Q_READY)
            if not ik.success or not kb.is_collision_free(ik.q):
                continue
            b = speed_bounds(kb, ik.q, up)
            rows.append((x, z, ik.q, b))
    print("Part A: largest vertical ball-seat speed, all joints <= 180 deg/s (bowl axis up, y = 0)")
    print("   x     z(base) | translate  whip4  whip6  (m/s)")
    for x, z, _, b in rows:
        print(f"  {x:.2f}  {z:.2f}    |  {b['translate']:5.2f}   {b['whip4']:5.2f}  {b['whip6']:5.2f}")
    for k in ("translate", "whip4", "whip6"):
        v = np.array([r[3][k] for r in rows])
        print(f"  {k:9s}: median {np.median(v):.2f}, max {v.max():.2f} m/s over {len(v)} poses")
    return rows


# ------------------------------------------------------------------ Part B strokes
class WhipStroke:
    """LineStroke's interface (t0, T1, T2, t_peak, t_end, q_start, q_end, sample, max_joint_speed),
    whip4 joints. mode 'jointwhip': q(s) = q_r + s qd_r / V (straight line in joint space), with qd_r
    the minimax whip4 solution for the seat velocity V d; 'linewhip': the seat moves along d at unit
    path speed, dq/ds = J4+ d + r n(q) with the release's null-space share r (RK2)."""

    DS = 0.0005

    def __init__(self, kin: ArmKinematics, q_r: np.ndarray, direction, V: float, a_up: float, a_down: float,
                 mode: str):
        self.d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        self.V, self.t0 = V, 0.0
        self.T1, self.T2 = 1.5 * V / a_up, 1.5 * V / a_down
        self.t_peak, self.t_end = self.T1, self.T1 + self.T2
        s_lo, s_hi = -V * self.T1 / 2, V * self.T2 / 2

        def sol(q):
            jp = kin.jacobian(q)[0][:, WHIP]
            u = np.linalg.pinv(jp, rcond=1e-6) @ self.d
            n = np.linalg.svd(jp)[2][-1]
            return u, n

        u, n = sol(q_r)
        rs = np.linspace(-3, 3, 6001)
        r0 = rs[np.argmin(np.max(np.abs(u[None] + rs[:, None] * n[None]), axis=1))]
        n_lo, n_hi = int(np.ceil(-s_lo / self.DS)) + 1, int(np.ceil(s_hi / self.DS)) + 1
        self.s = np.arange(-n_lo, n_hi + 1) * self.DS
        i0 = n_lo
        self.q = np.zeros((len(self.s), 6))
        self.q[i0] = q_r
        if mode == "jointwhip":
            dq0 = np.zeros(6)
            dq0[WHIP] = u + r0 * n
            self.q = q_r[None] + self.s[:, None] * dq0[None]
            self.dq = np.repeat(dq0[None], len(self.s), axis=0)
        else:
            n_ref = [n]

            def f(q):
                uu, nn = sol(q)
                nn = nn if nn @ n_ref[0] >= 0 else -nn
                n_ref[0] = nn
                out = np.zeros(6)
                out[WHIP] = uu + r0 * nn
                return out

            for step, idx in ((self.DS, range(i0, len(self.s) - 1)), (-self.DS, range(i0, 0, -1))):
                n_ref[0] = n
                for i in idx:
                    k1 = f(self.q[i])
                    self.q[i + int(np.sign(step))] = self.q[i] + step * f(self.q[i] + 0.5 * step * k1)
            n_ref[0] = n
            self.dq = np.array([f(q) for q in self.q])
        self.ddq = np.gradient(self.dq, self.DS, axis=0)
        self.q_start, self.q_end = self.q[0].copy(), self.q[-1].copy()

    _at = LineStroke._at
    path_speed = LineStroke.path_speed
    sample = LineStroke.sample
    max_joint_speed = LineStroke.max_joint_speed


def make_stroke(sim, kin, q_r, d, V, a_up, a_down, mode):
    if mode == "translate":
        return LineStroke(sim, q_r, d, V, a_up, a_down)
    return WhipStroke(kin, q_r, d, V, a_up, a_down, mode)


def feasible(sim, kin, stroke) -> str:
    if stroke.max_joint_speed() / np.pi > SPEED_MARGIN:
        return "speed"
    if stroke_torque(sim, stroke) > TORQUE_FRAC:
        return "torque"
    if np.any(stroke.q < kin.q_min + 0.05) or np.any(stroke.q > kin.q_max - 0.05):
        return "joint range"
    for q in stroke.q[:: max(1, len(stroke.q) // 25)]:
        if not kin.is_collision_free(q):
            return "collision"
    return ""


def max_speed(sim, kin, q_r, d, a_up, a_down, mode, lo=0.5, hi=6.0, tol=0.02):
    """Largest V (m/s) of a feasible stroke, by bisection; (V, limiting reason above V)."""
    if feasible(sim, kin, make_stroke(sim, kin, q_r, d, lo, a_up, a_down, mode)):
        return 0.0, "infeasible at 0.5 m/s"
    why = ""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        r = feasible(sim, kin, make_stroke(sim, kin, q_r, d, mid, a_up, a_down, mode))
        if r:
            hi, why = mid, r
        else:
            lo = mid
    return lo, why


def rollout_grasped(sim: HandSim, stroke, q_r: np.ndarray, fingers: Fingers, t_carry: float = 0.6):
    """As hand_juggle.rollout, but the ball is settled and grasped at the release pose (bowl axis up)
    and carried to the stroke start (a whip starts with the hand tilted: a ball placed there rolls out
    before the grasp closes; in the loop it arrives grasped). Returns (t_release from the stroke start,
    ball position, ball velocity) or None."""
    d = sim.data
    sim.reset(q_r, fingers.hands["settle"])
    sim.place_ball(BALL_ON_PALM + np.array([0.0, 0.0, -sim.ball_radius - 0.002]))
    t_grasp, t_move = 0.4, 0.9
    shift = t_move + t_carry + 0.2
    while d.time < shift:
        if d.time >= t_grasp:
            sim.set_hand(fingers.hands["grasp"])
        x = np.clip((d.time - t_move) / t_carry, 0.0, 1.0)
        w, wd, wdd = 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5, (30 * x ** 2 - 60 * x ** 3 + 30 * x ** 4) / t_carry, \
            (60 * x - 180 * x ** 2 + 120 * x ** 3) / t_carry ** 2
        dq = stroke.q_start - q_r
        sim.control_arm(q_r + w * dq, wd * dq, wdd * dq)
        sim.step()
        if sim.ball_contacts().ground:
            return None
    t_peak = shift + stroke.t_peak
    cand = None
    while d.time < shift + stroke.t_end + 0.2:
        sim.set_hand(fingers.opening(d.time, t_peak))
        sim.control_arm(*stroke.sample(d.time - shift))
        sim.step()
        c = sim.ball_contacts()
        if c.ground or c.arm:
            return None
        if c.hand:
            cand = None
        elif d.time > t_peak - stroke.T1:
            if cand is None:
                cand = (d.time - shift, *sim.ball_state())
            elif d.time - shift - cand[0] >= FREE_CONFIRM:
                return cand
    return None


def flight_time(vz: float, drop: float) -> float:
    return (vz + np.sqrt(vz ** 2 + 2 * G * drop)) / G


def part_b(sim, kb, rows, a_ups, a_downs, n_poses, modes, poses=None):
    fingers = Fingers(sim, -0.02, 0.03, bowl=True)
    up = np.array([0.0, 0.0, 1.0])
    print("\nPart B: strokes within 0.9 x 180 deg/s and 0.9 x torque; rollout with the bowl hand "
          "(fingers open 20 ms before the peak over 30 ms)")
    # the poses with the best whip4 bound, plus the E3c release pose region
    order = sorted(rows, key=lambda r: -r[3]["whip4"])[:n_poses]
    if poses:
        order = [r for xz in poses for r in rows if np.allclose(r[:2], xz)]
    for x, z, q_r, _ in order:
        for a_up, a_down in itertools.product(a_ups, a_downs):
            print(f"  pose x {x:.2f} z {z:.2f}, a_up {a_up:.1f} g, a_down {a_down:.1f} g:")
            for mode in modes:
                V, why = max_speed(sim, kb, q_r, up, a_up * G, a_down * G, mode)
                if V <= 0:
                    print(f"    {mode:9s}: {why}")
                    continue
                st = make_stroke(sim, kb, q_r, up, V, a_up * G, a_down * G, mode)
                length = np.linalg.norm(kb.fk(st.q_end)[0] - kb.fk(st.q_start)[0])
                rel = rollout_grasped(sim, st, q_r, fingers)
                over = np.linalg.norm(kb.fk(st.q_end)[0] - kb.fk(q_r)[0])
                head = (f"    {mode:9s}: V {V:.2f} m/s (next limit: {why}), joint speed "
                        f"{st.max_joint_speed() / np.pi:.2f}, torque {stroke_torque(sim, st):.2f}, "
                        f"stroke {length:.2f} m ({st.T1:.3f} + {st.T2:.3f} s), stop {over:.2f} m past the release")
                if rel is None:
                    print(head + " -- no clean release")
                    continue
                t_rel, p, v = rel
                spin = np.linalg.norm(sim.data.qvel[sim.ball_dadr + 3:sim.ball_dadr + 6])
                ang = np.degrees(np.arccos(np.clip(v[2] / np.linalg.norm(v), -1, 1)))
                print(head + f"\n               release {1e3 * (t_rel - st.t_peak):+.0f} ms, |v| {np.linalg.norm(v):.2f} m/s, "
                      f"{ang:.1f} deg off vertical, spin {spin:.1f} rad/s; F {flight_time(v[2], 0.1):.2f} / "
                      f"{flight_time(v[2], 0.2):.2f} s (catch 0.1 / 0.2 m lower)", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="v4 E4: whip vs translate throws with the LEAP hand.")
    p.add_argument("--pedestal", type=float, default=0.4)
    p.add_argument("--a-up", nargs="+", type=float, default=[2.0])
    p.add_argument("--a-down", nargs="+", type=float, default=[2.5])
    p.add_argument("--poses", type=int, default=3, help="Part B: number of release poses (best whip4 bound first)")
    p.add_argument("--pose", nargs=2, type=float, action="append", default=None,
                   help="Part B: release pose x z (from Part A's grid); repeatable, overrides --poses")
    p.add_argument("--modes", nargs="+", default=["translate", "linewhip", "jointwhip"])
    p.add_argument("--skip-b", action="store_true")
    args = p.parse_args()
    sim = HandSim(load_model(pedestal=args.pedestal))
    set_bowl_sites(sim.model)
    kb = ArmKinematics(sim.model, "bowl_seat")
    base = np.array([0.0, 0.0, args.pedestal])
    rows = part_a(sim, base)
    if not args.skip_b:
        part_b(sim, kb, rows, args.a_up, args.a_down, args.poses, args.modes, args.pose)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Hand postures, finger schedule, whip strokes and throw calibration (v4 E2-E5, shared from v5 on).

* posture() / postures(): LEAP joint targets (v4 grasp_test); the same for both hands (the
  abduction joints, which change sign, stay at 0).
* Fingers: the bowl hand of E3c -- settle / catch / catch_grasp postures and the E2 opening
  (grasp -> flat from t_peak + t_open over T_open).
* WhipStroke: E4's straight-seat-line whip ('linewhip') and joint-space whip ('jointwhip'), with
  LineStroke's interface; feasible() / stroke_torque(): 0.9 x 180 deg/s, 0.9 x torque, joint range,
  collisions.
* rollout_grasped(): one throw on a single-arm HandSim (ball grasped at the release pose, carried
  to the stroke start); calibrate_whip(): iterate the commanded release velocity in rollouts until
  the ball reaches a target F after the release (v4 hand_cascade).
"""

from dataclasses import dataclass

import numpy as np

from .handsim import ARM_SPEED_LIMIT, BALL_ON_PALM, HandSim, LineStroke
from .kinematics import ArmKinematics
from .planning import min_time_quintic

WHIP = np.array([0, 1, 2, 3])  # shoulder_pan + the three parallel pitch joints
SPEED_MARGIN = 0.9  # planned arm joint speed <= 0.9 x 180 deg/s
TORQUE_FRAC = 0.9  # planned stroke feed-forward torque <= 0.9 x actuator limits
FREE_CONFIRM = 0.03  # s without hand contact = released
BOWL_MCP = 1.1  # rad, knuckle flexion of the bowl (E3c)


def posture(sim: HandSim, curl: float, th_cmc: float, th_mcp: float = 0.0, th_ipl: float = 0.0,
            pip: float | None = None) -> np.ndarray:
    """Fingers: mcp = curl, pip = pip (default curl), dip = 0.6 pip; thumb joints as given."""
    J = {n: i for i, n in enumerate(sim.hand_joints)}
    q = np.zeros(len(J))
    pip = curl if pip is None else pip
    for f in ("if", "mf", "rf"):
        q[J[f"leap_{f}_mcp"]], q[J[f"leap_{f}_pip"]], q[J[f"leap_{f}_dip"]] = curl, pip, 0.6 * pip
    q[J["leap_th_cmc"]], q[J["leap_th_mcp"]], q[J["leap_th_ipl"]] = th_cmc, th_mcp, th_ipl
    return q


def postures(sim: HandSim) -> dict:
    return {"flat": posture(sim, 0.0, 0.0), "cradle": posture(sim, 0.8, 1.2), "grasp": posture(sim, 1.2, 1.2, 0.5)}


class Fingers:
    """Bowl hand (E3c): settle / catch in the bowl, close to catch_grasp on palm contact; the throw
    opens grasp -> flat from t_peak + t_open over T_open (E2)."""

    def __init__(self, sim: HandSim, t_open: float = -0.02, T_open: float = 0.03):
        self.hands = postures(sim)
        self.hands["settle"] = posture(sim, BOWL_MCP, 1.2, pip=0.3)
        self.hands["catch"] = posture(sim, BOWL_MCP, 1.2, pip=0.3)
        self.hands["catch_grasp"] = posture(sim, BOWL_MCP + 0.3, 1.2, 0.5, pip=1.2)
        self.t_open, self.T_open = t_open, T_open

    def opening(self, t, t_peak):
        w = np.clip((t - t_peak - self.t_open) / max(self.T_open, 1e-6), 0.0, 1.0)
        return self.hands["grasp"] + w * (self.hands["flat"] - self.hands["grasp"])


class WhipStroke:
    """LineStroke's interface (t0, T1, T2, t_peak, t_end, q_start, q_end, sample, max_joint_speed) with
    the whip joints. 'jointwhip': straight line in joint space through q_r with the minimax WHIP
    solution for the seat velocity V d; 'linewhip': the seat (kin's site) moves along d at unit path
    speed, dq/ds = J4+ d + r n(q) with the release's null-space share r (RK2)."""

    DS = 0.0005

    def __init__(self, kin: ArmKinematics, q_r: np.ndarray, direction, V: float, a_up: float, a_down: float,
                 mode: str = "linewhip"):
        self.d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        self.V, self.t0 = V, 0.0
        self.T1, self.T2 = 1.5 * V / a_up, 1.5 * V / a_down
        self.t_peak, self.t_end = self.T1, self.T1 + self.T2
        s_lo, s_hi = -V * self.T1 / 2, V * self.T2 / 2

        def sol(q):
            jp = kin.jacobian(q)[0][:, WHIP]
            return np.linalg.pinv(jp, rcond=1e-6) @ self.d, np.linalg.svd(jp)[2][-1]

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


def stroke_torque(sim: HandSim, stroke, n: int = 200) -> float:
    """Peak inverse-dynamics arm torque along the stroke, as a fraction of the actuator limits."""
    lim = sim.dynamics.force_limit
    return max(float(np.max(np.abs(sim.arm_torque_ff(*stroke.sample(t))) / lim)) for t in np.linspace(stroke.t0, stroke.t_end, n))


def feasible(sim: HandSim, kin: ArmKinematics, stroke) -> str:
    """'' if the stroke keeps 0.9 x 180 deg/s, 0.9 x torque, the joint range and is collision-free."""
    if stroke.max_joint_speed() / ARM_SPEED_LIMIT > SPEED_MARGIN:
        return "speed"
    if stroke_torque(sim, stroke) > TORQUE_FRAC:
        return "torque"
    if np.any(stroke.q < kin.q_min + 0.05) or np.any(stroke.q > kin.q_max - 0.05):
        return "joint range"
    for q in stroke.q[:: max(1, len(stroke.q) // 25)]:
        if not kin.is_collision_free(q):
            return "collision"
    return ""


def rollout_grasped(sim: HandSim, stroke, q_r: np.ndarray, fingers: Fingers, t_carry: float = 0.6):
    """One throw: the ball settled and grasped at the release pose (bowl axis up), carried to the stroke
    start, thrown. Returns (t_release from the stroke start, ball position, ball velocity) or None."""
    d = sim.data
    sim.select_ball(0)
    sim.reset(q_r, fingers.hands["settle"])
    sim.place_ball(BALL_ON_PALM + np.array([0.0, 0.0, -sim.ball_radius - 0.002]))
    t_grasp, t_move = 0.4, 0.9
    shift = t_move + t_carry + 0.2
    carry = min_time_quintic(t_move, q_r, np.zeros(6), np.zeros(6), stroke.q_start, np.full(6, 1.0), np.full(6, 4.0))
    while d.time < shift:
        if d.time >= t_grasp:
            sim.set_hand(fingers.hands["grasp"])
        sim.control_arm(*(carry.sample(d.time) if d.time >= t_move else (q_r, None, None)))
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


@dataclass
class Throw:
    stroke: object  # t0 = 0
    target: np.ndarray  # ball centre aimed at, F after the release (arm base frame)
    t_release: float  # nominal, from the stroke start
    p_release: np.ndarray
    v_release: np.ndarray
    q_release: np.ndarray


def calibrate_whip(sim: HandSim, kb: ArmKinematics, q_r: np.ndarray, target, F: float, fingers: Fingers,
                   a_up: float, a_down: float, iters: int = 8, tol: float = 0.005, accept: float = 0.03,
                   gain: float = 0.7, verbose: bool = True) -> Throw | None:
    """Iterate the commanded seat velocity of a linewhip through q_r (kb: bowl_seat kinematics) in
    rollouts until the released ball reaches target F after the release."""
    g = sim.model.opt.gravity
    target = np.asarray(target, dtype=float)
    need = lambda p: (target - p - 0.5 * g * F ** 2) / F
    v_cmd = need(kb.fk(q_r)[0])
    best = (np.inf, None)
    for i in range(iters):
        stroke = WhipStroke(kb, q_r, v_cmd, float(np.linalg.norm(v_cmd)), a_up, a_down, "linewhip")
        why = feasible(sim, kb, stroke)
        if why:
            if verbose:
                print(f"    stroke infeasible ({why}) at |v| {np.linalg.norm(v_cmd):.2f} m/s")
            return None
        rel = rollout_grasped(sim, stroke, q_r, fingers)
        if rel is None:
            if verbose:
                print("    no release in the calibration rollout")
            return None
        t_rel, p, v = rel
        err = need(p) - v
        if verbose:
            print(f"    iter {i}: |v_cmd| {np.linalg.norm(v_cmd):.3f} m/s, release v {v.round(3)}, error "
                  f"{1e3 * np.linalg.norm(err):.0f} mm/s, joint speed {stroke.max_joint_speed() / np.pi:.2f}, "
                  f"torque {stroke_torque(sim, stroke):.2f}")
        if np.linalg.norm(err) < best[0]:
            best = (float(np.linalg.norm(err)), Throw(stroke, target, t_rel, p, v, q_r.copy()))
        if best[0] < tol:
            break
        v_cmd = v_cmd + gain * err
    if best[0] > accept:
        if verbose:
            print(f"    calibration did not converge (best {1e3 * best[0]:.0f} mm/s)")
        return None
    return best[1]

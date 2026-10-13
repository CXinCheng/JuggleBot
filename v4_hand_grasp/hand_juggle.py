"""E3b (v4): one arm, one ball, LEAP hand -- continuous self throw-catch between two points.

v2's experiment B with the hand: the ball alternates between two release points R_A, R_B
(GAP apart in y, flange positions relative to the robot base). Per cycle:

    throw (E2: grasp, straight stroke, fingers open at t_open) -> flight F
          -> catch (E3a: catch posture, the arm gives way along the falling ball -- ~10 deg off
             vertical here -- and the fingers close on palm contact)
          -> carry the grasped ball to the next stroke start -> throw back

The robot stands on a pedestal (handsim.load_model(pedestal=...)): the hand moves fastest
vertically with the arm stretched out at about shoulder height, and the catch -- below the
release, with the wrist hanging under the hand and the arm giving way -- needs room below that.

Offline, per direction (calibrate_throw): the stroke's peak hand velocity is iterated in
rollouts (ball settled in the cradle, then grasped, as in E2) until the ball, released at
p_release, reaches the target -- the other release point's ball position, DROP lower -- F after
the release. Online, per cycle: the catch is planned from the nominal release when the throw is
scheduled (its approach starts at the end of the stroke), then re-planned from observations of
the ball (OBS_RATE, optionally noisy) as in v1/v2.

Choices that made the loop reliable (see README, E3b):
  * the ball fit uses the calibrated nominal release as a prior (prediction.fit_with_prior) --
    with the short flight, plain least squares on noisy observations misplaces the catch by a
    few cm, and the hand (unlike the cup, a funnel) does not forgive that;
  * the stroke decelerates at 2.5 g (3 g saturates shoulder_lift with the arm stretched out, 2 g
    releases too late to get to the catch);
  * the hand is turned 90 deg about the palm normal (--yaw): the open ring-finger side of the
    LEAP hand (no little finger) then faces away from the throw line -- a ball arriving with a
    sideways velocity towards that side can be squeezed out of it by the closing fingers.

A run stops at the first failure or after --cycles catches. Failures: NO_CATCH_PLAN, NO_RELEASE,
DROPPED (ball on the ground / pedestal), HIT_ARM, NOT_CAUGHT (not resting in the hand in time),
SPEED_LIMIT (actual arm joint speed above 180 deg/s).

Usage (from the repo root):
    uv run python v4_hand_grasp/hand_juggle.py                          # 10 runs x 20 catches
    uv run python v4_hand_grasp/hand_juggle.py --obs-noise 0.005
    uv run python v4_hand_grasp/hand_juggle.py --runs 1 --cycles 6 --video videos/hand_juggle.mp4
"""

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from grasp_test import posture, postures
from hand_throw import BALL_ON_PALM, FREE_CONFIRM, OFFSET, SPEED_MARGIN, release_config
from handsim import HandSim, LineStroke, Q_READY, load_model
from kinematics import ArmKinematics
from planning import CatchPlanner, Hold, PlannerConfig, Trajectory, min_time_quintic
from prediction import BallisticPredictor
from video import VideoRecorder

G = 9.81
# Perception / re-planning (as v1/v2)
OBS_RATE, OBS_WINDOW, REPLAN_PERIOD = 100.0, 0.05, 0.05
REPLAN_THRESHOLD, REPLAN_LOCK, REPLAN_MIN_WAIT = 0.005, 0.03, 0.02
# Catching (E3a), with v2's higher joint accelerations for the short flights, torque-checked
CATCH_A_MAX = np.array([40.0, 40.0, 40.0, 60.0, 60.0, 60.0])  # rad/s^2
CATCH_TORQUE_FRAC = 0.8  # planned (feed-forward) torque; the servo's corrections add up to ~0.1 more
STROKE_TORQUE_FRAC = 0.9
CATCH_CURL = 0.3  # rad, finger curl of the catch posture
SEAT_TOL, CAUGHT_SPEED, HOLD_CAUGHT = 0.05, 0.05, 0.1  # m from ball_seat (the grasp may hold it off-centre), m/s relative, s
CATCH_TIMEOUT = 1.0  # s after the planned catch time
# Prior on the release state for the ball fit (--prior): the calibrated nominal release, with
# about the spread measured in the loop (release velocity error ~40 mm/s)
PRIOR_SIGMA_P, PRIOR_SIGMA_V = 0.005, 0.03  # m, m/s per axis
SIGMA_OBS_MIN = 0.001  # m, observation weight floor (noise-free observations)
# Carrying the grasped ball to the next stroke start (v2's limits), then a short pause
TRANSFER_V_MAX = np.full(6, 0.8 * np.pi)
TRANSFER_A_MAX = np.array([12.0, 12.0, 12.0, 20.0, 20.0, 20.0])
SETTLE_BEFORE_THROW = 0.2  # s


@dataclass
class LoopThrow:
    stroke: LineStroke  # t0 = 0
    target: np.ndarray  # ball centre aimed at, F after the release
    t_release: float  # nominal, from the stroke start
    p_release: np.ndarray
    v_release: np.ndarray


class StrokeSegment:
    """A LineStroke shifted to start at t_throw, as a Trajectory segment."""

    def __init__(self, stroke: LineStroke, t_throw: float):
        self.stroke, self.shift = stroke, t_throw
        self.t0, self.t1 = t_throw + stroke.t0, t_throw + stroke.t_end

    def sample(self, t):
        return self.stroke.sample(t - self.shift)


@dataclass
class CycleRecord:
    k: int
    release_dv: float = np.nan  # m/s, actual - nominal release velocity (propagated to the nominal time)
    release_dt: float = np.nan  # s, actual - nominal release time
    flight: float = np.nan  # s, release -> first palm contact
    impact_speed: float = np.nan  # m/s, ball relative to ball_seat at the first hand contact
    replans: int = 0
    dwell: float = np.nan  # s, palm contact -> next release
    t_release: float = np.nan
    t_touch: float = np.nan
    diag: str = ""  # state at a failure


@dataclass
class RunResult:
    catches: int = 0
    outcome: str = "CAP"
    cycles: list = field(default_factory=list)
    speed_peak: float = 0.0
    arm_force: float = 0.0
    finger_torque: float = 0.0
    sim_time: float = 0.0


class Fingers:
    """Finger targets: the grasp; the E2 opening (grasp -> flat from t_peak + t_open over T_open)."""

    def __init__(self, sim: HandSim, t_open: float, T_open: float):
        self.hands = postures(sim)
        self.hands["catch"] = posture(sim, CATCH_CURL, 1.2)
        self.t_open, self.T_open = t_open, T_open

    def opening(self, t, t_peak):
        w = np.clip((t - t_peak - self.t_open) / max(self.T_open, 1e-6), 0.0, 1.0)
        return self.hands["grasp"] + w * (self.hands["flat"] - self.hands["grasp"])


def rollout(sim: HandSim, stroke: LineStroke, fingers: Fingers):
    """Throw from a settled grasp (as E2, no noise); returns (t_release from the stroke start,
    ball position, ball velocity) or None."""
    d = sim.data
    sim.reset(stroke.q_start, fingers.hands["cradle"])
    sim.place_ball(BALL_ON_PALM + np.array([0.0, 0.0, -sim.ball_radius - 0.002]))
    shift = 0.9
    while d.time < shift:
        if d.time >= 0.4:
            sim.set_hand(fingers.hands["grasp"])
        sim.control_arm(stroke.q_start)
        sim.step()
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


def stroke_torque(sim: HandSim, stroke: LineStroke, n: int = 200) -> float:
    """Peak inverse-dynamics arm torque along the stroke, as a fraction of the actuator limits."""
    lim = sim.model.actuator_forcerange[sim.arm_act, 1]
    return max(float(np.max(np.abs(sim.arm_torque_ff(*stroke.sample(t))) / lim))
               for t in np.linspace(stroke.t0, stroke.t_end, n))


def calibrate_throw(sim, kin, R, target, F, fingers, a_up, a_down, yaw=0.0, iters=8, tol=0.005, accept=0.03,
                    gain=0.7, verbose=True):
    """Peak hand velocity of the stroke through flange position R (palm up, the hand turned by yaw
    about the palm normal), iterated in rollouts until the released ball reaches target F later.
    The release jitters by ~+-20 mm/s between nearby commands (the opening fingers), so the update is
    damped and the best iterate is kept if it is within accept (m/s). Returns a LoopThrow or None."""
    g = sim.model.opt.gravity
    q_r = release_config(kin, R)
    q_r[5] += yaw  # wrist_3 turns the hand about the flange axis = the palm normal
    need = lambda p: (target - p - 0.5 * g * F ** 2) / F
    v_cmd = need(R + np.array([0.0, 0.0, 0.122]))
    best = (np.inf, None)
    for i in range(iters):
        stroke = LineStroke(sim, q_r, v_cmd, float(np.linalg.norm(v_cmd)), a_up, a_down)
        speed = stroke.max_joint_speed() / np.pi
        torque = stroke_torque(sim, stroke)
        if speed > SPEED_MARGIN or torque > STROKE_TORQUE_FRAC:
            print(f"    stroke needs {speed:.2f} x 180 deg/s, {torque:.2f} x the arm torque limits "
                  f"(> {SPEED_MARGIN} / {STROKE_TORQUE_FRAC})")
            return None
        rel = rollout(sim, stroke, fingers)
        if rel is None:
            print("    no release in the calibration rollout")
            return None
        t_rel, p, v = rel
        err = need(p) - v
        if verbose:
            print(f"    iter {i}: |v_cmd| {np.linalg.norm(v_cmd):.3f} m/s, release v {v.round(3)}, "
                  f"error {1e3 * np.linalg.norm(err):.0f} mm/s, joint speed {speed:.2f}, torque {torque:.2f}")
        if np.linalg.norm(err) < best[0]:
            best = (float(np.linalg.norm(err)), LoopThrow(stroke, target, t_rel, p, v))
        if best[0] < tol:
            break
        v_cmd = v_cmd + gain * err
    if best[0] > accept:
        print(f"    calibration did not converge (best {1e3 * best[0]:.0f} mm/s)")
        return None
    return best[1]


class HandJuggleLoop:
    def __init__(self, sim: HandSim, throws: list[LoopThrow], fingers: Fingers, base_z: float,
                 obs_noise: float, noise_rng: np.random.Generator, cushion_dir: str = "ball",
                 d_pre: float = 0.03, d_cush: float = 0.08, yaw: float = 0.0, prior: bool = True):
        self.sim, self.throws, self.fingers = sim, throws, fingers
        self.obs_noise, self.noise_rng, self.prior = obs_noise, noise_rng, prior
        m = sim.model
        self.g = m.opt.gravity
        self.seat = m.site("ball_seat").id
        self.obs_every = max(1, round(1.0 / (OBS_RATE * m.opt.timestep)))
        self.force_limit = m.actuator_forcerange[sim.arm_act, 1]
        kin = ArmKinematics(m, "catch_point")
        d0 = mujoco.MjData(m)
        d0.qpos[sim.arm_qadr] = Q_READY
        mujoco.mj_forward(m, d0)
        cfg = PlannerConfig(strategy="cushion", z_min=base_z + 0.20, z_max=base_z + 0.90, z_min_end=base_z + 0.10,
                            v_max=np.full(6, 0.9 * np.pi), a_max=CATCH_A_MAX, min_wait=0.05, earliest_catch=0.10,
                            cushion_dir=cushion_dir, d_pre=d_pre, d_cush=d_cush, cup_speed_max=0.7)
        q_nominal = Q_READY + np.array([0, 0, 0, 0, 0, yaw])  # the catch poses keep the hand's yaw
        self.cp = CatchPlanner(kin, cfg, q_nominal, d0.body("shoulder_link").xpos.copy(), self._torque_ok)

    def _torque_ok(self, seg, n=15):
        lim = CATCH_TORQUE_FRAC * self.force_limit
        return all(np.all(np.abs(self.sim.arm_torque_ff(*seg.sample(t))) <= lim) for t in np.linspace(seg.t0, seg.t1, n))

    def seat_relative(self):
        m, d = self.sim.model, self.sim.data
        v = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.seat, v, 0)
        p, vb = self.sim.ball_state()
        return p - d.site_xpos[self.seat], vb - v[3:]

    def _nominal(self, th: LoopThrow, t_throw: float) -> BallisticPredictor:
        pred = BallisticPredictor(self.g)
        pred.t_ref, pred.p0, pred.v0 = t_throw + th.t_release, th.p_release.copy(), th.v_release.copy()
        return pred

    def run(self, max_cycles: int, rng: np.random.Generator, recorder=None, verbose=False) -> RunResult:
        sim, d, hands = self.sim, self.sim.data, self.fingers.hands
        res = RunResult()
        first = self.throws[0].stroke
        sim.reset(first.q_start, hands["cradle"])
        off = rng.uniform(-OFFSET, OFFSET, 2)
        sim.place_ball(BALL_ON_PALM + np.array([off[0], off[1], -sim.ball_radius - 0.002]))
        while d.time < 0.9:  # settle in the cradle, then grasp
            if d.time >= 0.4:
                sim.set_hand(hands["grasp"])
            sim.control_arm(first.q_start)
            sim.step()
        t_throw = d.time
        pre = [Hold(0.0, t_throw, first.q_start)]
        t_touch_prev = None
        for k in range(max_cycles):
            th = self.throws[k % 2]
            seg = StrokeSegment(th.stroke, t_throw)
            catch = self.cp.plan(self._nominal(th, t_throw), seg.t1, (th.stroke.q_end, np.zeros(6), np.zeros(6)))
            if catch is None:
                res.outcome = "NO_CATCH_PLAN"
                break
            traj = Trajectory(pre + [seg] + catch.trajectory.segments)
            outcome, rec, traj = self._cycle(k, th, seg, traj, catch, res, recorder)
            if t_touch_prev is not None and np.isfinite(rec.t_release):
                res.cycles[-1].dwell = rec.t_release - t_touch_prev
            res.cycles.append(rec)
            if outcome != "CAUGHT":
                res.outcome = outcome
                break
            res.catches += 1
            t_touch_prev = rec.t_touch
            if verbose:
                print(f"    catch {k + 1}: release dt {1e3 * rec.release_dt:+.1f} ms, dv {1e3 * rec.release_dv:.0f} mm/s, "
                      f"flight {1e3 * rec.flight:.0f} ms, replans {rec.replans}, impact {rec.impact_speed:.2f} m/s")
            nxt = self.throws[(k + 1) % 2].stroke
            q_now = traj.sample(d.time)[0]
            move = min_time_quintic(d.time, q_now, np.zeros(6), np.zeros(6), nxt.q_start, TRANSFER_V_MAX, TRANSFER_A_MAX)
            t_throw = move.t1 + SETTLE_BEFORE_THROW
            pre = [Hold(0.0, d.time, q_now), move, Hold(move.t1, t_throw, nxt.q_start)]
        res.speed_peak, res.sim_time = sim.speed_peak, d.time
        return res

    def _cycle(self, k, th, seg, traj, catch, res, recorder):
        """One throw -> catch. Returns (outcome, record, the trajectory in force at the end)."""
        sim, d, hands = self.sim, self.sim.data, self.fingers.hands
        rec = CycleRecord(k)
        t_peak = seg.shift + th.stroke.t_peak
        state, cand = "held", None
        obs_t, obs_p, last_fit, step = [], [], -np.inf, 0
        predictor = BallisticPredictor(self.g)
        closed, hold_start = False, None
        while True:
            t = d.time
            locked = t >= catch.t_match - REPLAN_LOCK
            if state == "free" and not np.isfinite(rec.t_touch) and not locked:
                if step % self.obs_every == 0:
                    obs_t.append(t)
                    p = sim.ball_state()[0]
                    obs_p.append(p + (self.noise_rng.normal(0, self.obs_noise, 3) if self.obs_noise else 0.0))
                if obs_t and t >= seg.t1 and t >= obs_t[0] + OBS_WINDOW and t - last_fit >= REPLAN_PERIOD:
                    if self.prior:
                        predictor.fit_with_prior(obs_t, obs_p, seg.shift + th.t_release, th.p_release, th.v_release,
                                                 max(self.obs_noise, SIGMA_OBS_MIN), PRIOR_SIGMA_P, PRIOR_SIGMA_V)
                    else:
                        predictor.fit(obs_t, obs_p)
                    last_fit = t
                    if np.linalg.norm(predictor.predict(catch.t_catch)[0] - catch.p_catch) > REPLAN_THRESHOLD:
                        new = self.cp.plan(predictor, t, traj.sample(t), min_wait=REPLAN_MIN_WAIT)
                        if new is not None:
                            catch, traj = new, new.trajectory
                            rec.replans += 1
            # fingers: open to throw; catch posture once the stroke is over; grasp on palm contact
            if state != "free":
                sim.set_hand(self.fingers.opening(t, t_peak))
            elif closed:
                sim.set_hand(hands["grasp"])
            elif t >= seg.t1:
                sim.set_hand(hands["catch"])
            sim.control_arm(*traj.sample(t))
            sim.step()
            step += 1
            if recorder is not None:
                recorder.capture(d)
            res.arm_force = max(res.arm_force, float(np.max(np.abs(d.actuator_force[sim.arm_act]) / self.force_limit)))
            res.finger_torque = max(res.finger_torque, sim.finger_torque_ratio())
            if sim.speed_peak > 1.0:
                return "SPEED_LIMIT", rec, traj
            c = sim.ball_contacts()
            if c.ground:
                return "DROPPED", rec, traj
            if c.arm:
                return "HIT_ARM", rec, traj
            if state == "held":
                if not c.hand and d.time > t_peak - th.stroke.T1:
                    state, cand = "maybe", (d.time, *sim.ball_state())
                elif d.time > seg.t1 + 0.2:
                    return "NO_RELEASE", rec, traj
            elif state == "maybe":
                if c.hand:
                    state = "held"
                elif d.time - cand[0] >= FREE_CONFIRM:
                    state, obs_t, obs_p = "free", [], []
                    rec.t_release = cand[0]
                    tau = seg.shift + th.t_release - cand[0]
                    rec.release_dt = -tau
                    rec.release_dv = float(np.linalg.norm(cand[2] + self.g * tau - th.v_release))
            else:
                if c.hand and not np.isfinite(rec.t_touch):
                    rec.t_touch = d.time
                    rec.flight = d.time - rec.t_release
                    rec.impact_speed = float(np.linalg.norm(self.seat_relative()[1]))
                closed = closed or c.palm
                if np.isfinite(rec.t_touch):
                    rel_p, rel_v = self.seat_relative()
                    if (c.hand and np.linalg.norm(rel_p) < SEAT_TOL and np.linalg.norm(rel_v) < CAUGHT_SPEED
                            and d.time >= traj.t_end):
                        hold_start = d.time if hold_start is None else hold_start
                        if d.time - hold_start >= HOLD_CAUGHT:
                            return "CAUGHT", rec, traj
                    else:
                        hold_start = None
                if d.time > catch.t_catch + CATCH_TIMEOUT:
                    rel_p, rel_v = self.seat_relative()
                    rec.diag = (f"ball {1e3 * rel_p.round(3)} mm / {1e3 * np.linalg.norm(rel_v):.0f} mm/s from ball_seat, "
                                f"palm {c.palm} finger {c.finger}, touched {np.isfinite(rec.t_touch)}, closed {closed}, "
                                f"in palm frame {1e3 * sim.ball_in_palm().round(3)} mm")
                    return "NOT_CAUGHT", rec, traj


def summarize(results: list[RunResult], cap: int) -> None:
    n = len(results)
    catches = np.array([r.catches for r in results])
    print(f"\nRuns: {n}, catches per run: median {np.median(catches):.0f}, min {catches.min()}, max {catches.max()} "
          f"(cap {cap}); runs reaching the cap: {sum(r.outcome == 'CAP' for r in results)}/{n}")
    print("Outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r.outcome for r in results).items())))
    cyc = [c for r in results for c in r.cycles if np.isfinite(c.flight)]
    if cyc:
        stat = lambda name, xs, s, u: xs and print(f"  {name}: mean {s * np.nanmean(xs):.1f} {u}, max {s * np.nanmax(xs):.1f} {u}")
        stat("release timing error |dt|", [abs(c.release_dt) for c in cyc], 1e3, "ms")
        stat("release velocity error  ", [c.release_dv for c in cyc], 1e3, "mm/s")
        stat("flight (release -> hand) ", [c.flight for c in cyc], 1e3, "ms")
        stat("impact speed (rel. hand) ", [c.impact_speed for c in cyc], 1, "m/s")
        stat("dwell (catch -> release) ", [c.dwell for c in cyc if np.isfinite(c.dwell)], 1e3, "ms")
        print(f"  re-plans per catch      : mean {np.mean([c.replans for c in cyc]):.2f}")
    print(f"  peak arm joint speed {max(r.speed_peak for r in results):.2f} x 180 deg/s, "
          f"arm force {max(r.arm_force for r in results):.2f} x limit, "
          f"finger torque {max(r.finger_torque for r in results):.2f} x limit")


def main() -> int:
    p = argparse.ArgumentParser(description="v4 E3b: single-arm self throw-catch loop with the LEAP hand.")
    p.add_argument("--pedestal", type=float, default=0.4, help="height of the robot base above the ground (m)")
    p.add_argument("--release", nargs=2, type=float, default=[0.85, 0.30],
                   help="x, z of the release points (flange at the peak speed) relative to the robot base (m)")
    p.add_argument("--gap", type=float, default=0.15, help="distance between the two release points, in y (m)")
    p.add_argument("--F", type=float, default=0.40, help="flight time to the target (s)")
    p.add_argument("--drop", type=float, default=0.10, help="target this much below the release ball position (m)")
    p.add_argument("--t-open", type=float, default=-0.02, help="finger opening start relative to the peak speed (s)")
    p.add_argument("--T-open", type=float, default=0.03, help="finger opening duration (s)")
    p.add_argument("--a-up", type=float, default=2.0,
                   help="stroke peak acceleration (g); 3 g saturates shoulder_lift with the arm stretched out")
    p.add_argument("--a-down", type=float, default=2.5,
                   help="stroke peak deceleration (g; > 1 for the ball to leave); 3 g saturates shoulder_lift, "
                        "2 g releases too late to catch")
    p.add_argument("--cushion-dir", choices=["ball", "down"], default="ball",
                   help="the hand gives way along the incoming ball (~10 deg off vertical here) or straight down")
    p.add_argument("--d-pre", type=float, default=0.03, help="m of motion before the contact")
    p.add_argument("--d-cush", type=float, default=0.08, help="m of motion after the contact")
    p.add_argument("--yaw", type=float, default=90.0,
                   help="hand turned about the palm normal (deg, wrist_3) from the ready pose's orientation; at 0 the open "
                        "ring-finger side faces the throw line (+y), at 90 it faces away from it")
    p.add_argument("--no-prior", action="store_true", help="plain least-squares ball fit (v1/v2)")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--cycles", type=int, default=20, help="stop a run after this many catches")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-noise", type=float, default=0.0, help="std. dev. of the ball position observations (m)")
    p.add_argument("--video", type=Path, default=None, help="record the first run (mp4, real time, camera 'loop')")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    sim = HandSim(load_model(pedestal=args.pedestal))
    kin = ArmKinematics(sim.model)
    fingers = Fingers(sim, args.t_open, args.T_open)
    base = np.array([0.0, 0.0, args.pedestal])
    x, z = args.release
    R = [base + np.array([x, -args.gap / 2, z]), base + np.array([x, args.gap / 2, z])]
    ball_up = np.array([0.0, 0.0, 0.122 - args.drop])  # ball_seat above the flange, minus the drop
    print(f"Setup: pedestal {args.pedestal} m, release points {R[0].round(3)} / {R[1].round(3)} (flange), "
          f"F {args.F} s, drop {args.drop} m, t_open {1e3 * args.t_open:+.0f} ms, T_open {1e3 * args.T_open:.0f} ms; "
          f"hand yaw {args.yaw} deg; catch: cushion {args.cushion_dir} {args.d_pre}/{args.d_cush} m, "
          f"obs noise {1e3 * args.obs_noise:.0f} mm, {'least-squares' if args.no_prior else 'release-prior'} fit")
    throws = []
    for i, name in ((0, "A -> B"), (1, "B -> A")):
        t0 = time.perf_counter()
        print(f"  {name}:")
        th = calibrate_throw(sim, kin, R[i], R[1 - i] + ball_up, args.F, fingers, args.a_up * G, args.a_down * G,
                             np.radians(args.yaw))
        if th is None:
            return 1
        tilt = np.degrees(np.arccos(th.stroke.d[2]))
        print(f"    -> stroke V {th.stroke.V:.2f} m/s tilted {tilt:.1f} deg, release {1e3 * (th.t_release - th.stroke.t_peak):.0f} ms "
              f"after the peak, |v| {np.linalg.norm(th.v_release):.2f} m/s, aim error "
              f"{1e3 * np.linalg.norm((th.target - th.p_release - 0.5 * sim.model.opt.gravity * args.F ** 2) / args.F - th.v_release):.0f} mm/s, joint speed {th.stroke.max_joint_speed() / np.pi:.2f} "
              f"({time.perf_counter() - t0:.1f} s)", flush=True)
        throws.append(th)

    rng, noise_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(args.seed).spawn(2))
    loop = HandJuggleLoop(sim, throws, fingers, args.pedestal, args.obs_noise, noise_rng, args.cushion_dir,
                          args.d_pre, args.d_cush, np.radians(args.yaw), not args.no_prior)
    results = []
    for i in range(args.runs):
        rec = None
        if args.video is not None and i == 0:
            rec = VideoRecorder(sim.model, args.video, speed=1.0, camera="loop")
        t0 = time.perf_counter()
        r = loop.run(args.cycles, rng, rec, args.verbose)
        if rec is not None:
            rec.close()
        results.append(r)
        print(f"  run {i + 1}: {r.catches} catches, {r.outcome} (sim {r.sim_time:.1f} s, wall {time.perf_counter() - t0:.1f} s)"
              + (f" -- {r.cycles[-1].diag}" if r.cycles and r.cycles[-1].diag else ""), flush=True)
    summarize(results, args.cycles)
    return 0


if __name__ == "__main__":
    sys.exit(main())

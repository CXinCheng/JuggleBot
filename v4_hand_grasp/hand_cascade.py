"""E5 (v4): one hand of a three-ball cascade -- can a single arm keep the cascade's timing?

Cascade (siteswap 3), beat tau: each hand throws every 2 tau, every ball is rethrown 3 tau after its
last throw, so with flight time F, dwell D (ball in the hand) and empty time E:

    F + D = 3 tau,   D + E = 2 tau   =>   D + 3E = 2F,   tau in [(F + D_min) / 3, F - E_min]

This arm is one hand (H); the other hand (O) is virtual: a launcher that releases the balls exactly
as a mirror image of H would (the plane of symmetry halfway between the two throw points), with
optional noise on the release velocity. In H's frame (relative to the robot base, ball centres):

    H throws from (x, 0, z)            to O's catch point (x + dx, -s - w, z - drop)   (throw inside,
    O throws from (x + depth, -s, z)   to H's catch point (x + dx + depth, +w, z - drop)  catch outside)

depth: the two hands' balls fly in planes depth apart (x), as the two ball paths of S1 -- without
it the ball H has just thrown and the one coming down to H pass within ~1 cm of each other.

H's throw is E4's straight-seat-line whip (whip_test.WhipStroke 'linewhip'), calibrated offline in
rollouts (as hand_juggle.calibrate_throw) to reach O's catch point F after the release; O's
nominal release is its mirror image, shifted by depth. Schedule: H releases at T0 + 2k tau, O at T0 + (2k - 1) tau.

Per cycle H:
  * throws: the stroke's acceleration, release (fingers open as in E2);
  * goes for the catch straight from the stroke, `switch` s after its peak speed: the catch
    approach starts from the arm's moving state, replacing the stroke's own deceleration
    (fallback: from the end of the full stroke). --catch direct (default, planning strategy
    "direct"): one quintic to the catch pose, arriving with the cushion speed along the ball, then
    the cushion; --catch cushion: E3a's approach to rest, wait, match, cushion. Bowl hand, fingers
    close on palm contact. Planned from O's nominal release
    before the throw, re-planned from observations (release-prior fit) after the switch;
  * as soon as the cushion is over and the ball sits in the hand, carries it to the next stroke
    start (fastest quintic within 0.9 x 180 deg/s, CATCH_A_MAX and the torque fraction), waits
    there at least `settle` s, throws at the scheduled time. The caught ball must sit in the
    hand when the stroke starts: in contact, within SEAT_TOL of the ball seat, moving less than
    --held-speed relative to it (E3b's 0.05 m/s is too strict here: the ball is often still
    rolling in the grasp after the fast carry).

Up to four balls take turns: the held one, the incoming one(s), and the one just thrown, which is
parked PARK_AFTER after its release (its landing error at O's catch point is computed from the
release state).

A run stops at the first failure or after --throws throws. Failures as in hand_juggle.py, plus
LATE (the carry cannot reach the stroke start by the scheduled throw), INCOMING_HIT (the incoming
ball touches the hand before the catch phase), BALL_COLLISION (two balls touch), OUTGOING_HIT (the
thrown ball touches the robot after its release).

Usage (from the repo root):
    uv run python v4_hand_grasp/hand_cascade.py --tau 0.35 0.37 0.39 --runs 2
    uv run python v4_hand_grasp/hand_cascade.py --launch-noise 0.03 --obs-noise 0.005 --runs 5 --throws 50 --seed 1
    uv run python v4_hand_grasp/hand_cascade.py --runs 1 --throws 8 --video videos/hand_cascade.mp4
"""

import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

import mujoco
import numpy as np

from hand_juggle import (CATCH_A_MAX, CATCH_TIMEOUT, CATCH_TORQUE_FRAC, OBS_RATE, OBS_WINDOW,
                         PRIOR_SIGMA_P, PRIOR_SIGMA_V, REPLAN_LOCK, REPLAN_MIN_WAIT, REPLAN_PERIOD, REPLAN_THRESHOLD,
                         SEAT_TOL, SIGMA_OBS_MIN, Fingers, LoopThrow, StrokeSegment, stroke_torque)
from hand_throw import BALL_ON_PALM, FREE_CONFIRM
from handsim import HandSim, Q_READY, load_model, set_bowl_sites
from kinematics import ArmKinematics
from planning import CatchPlanner, Hold, PlannerConfig, Trajectory, min_time_quintic
from prediction import BallisticPredictor
from video import VideoRecorder
from whip_test import WhipStroke, feasible, rollout_grasped

G = 9.81
PARK_AFTER = 0.12  # s after the release: the thrown ball leaves the scene
CARRY_V_MAX = np.full(6, 0.9 * np.pi)
CLEAR_DIST = 0.15  # m, thrown ball this far from the ball seat before the fingers rise to catch
SEAT_WAIT = 0.1  # s after the cushion: a closed hand whose ball is not within SEAT_TOL of the seat by then has missed


@dataclass
class Cycle:
    k: int
    release_dv: float = np.nan  # m/s, H's release velocity error (propagated to the nominal time)
    land_err: float = np.nan  # m, horizontal miss at O's catch height
    E: float = np.nan  # s, H's release -> first hand contact of the incoming ball
    D: float = np.nan  # s, that contact -> next release
    carry_spare: float = np.nan  # s, the carry to this stroke ended this long before the stroke start
    catch_spare: float = np.nan  # s, cushion strategy: the (last) approach reached its pre-catch pose this early (beyond min_wait)
    catch_acc: float = np.nan  # direct strategy: peak joint acceleration of the (last) approach / CATCH_A_MAX
    replans: int = 0
    impact: float = np.nan  # m/s, incoming ball relative to the ball seat at the first contact
    t_release: float = np.nan
    t_touch: float = np.nan


@dataclass
class Run:
    throws: int = 0
    outcome: str = "CAP"
    cycles: list = field(default_factory=list)
    speed_peak: float = 0.0
    arm_force: float = 0.0
    diag: str = ""


class Until:
    """A segment cut short at t1 (the catch approach takes over from the stroke there)."""

    def __init__(self, seg, t1):
        self.seg, self.t0, self.t1 = seg, seg.t0, t1

    def sample(self, t):
        return self.seg.sample(t)


def mirror(p, s):
    """Reflect across the plane y = -s / 2 (halfway between the two throw points)."""
    return np.array([p[0], -s - p[1], p[2]])


def calibrate_whip(sim, kb, q_r, target, F, fingers, a_up, a_down, iters=8, tol=0.005, accept=0.03, gain=0.7):
    """hand_juggle.calibrate_throw for the linewhip: iterate the commanded seat velocity in rollouts until
    the released ball reaches target F after the release. Returns a LoopThrow or None."""
    g = sim.model.opt.gravity
    need = lambda p: (target - p - 0.5 * g * F ** 2) / F
    v_cmd = need(kb.fk(q_r)[0])
    best = (np.inf, None)
    for i in range(iters):
        stroke = WhipStroke(kb, q_r, v_cmd, float(np.linalg.norm(v_cmd)), a_up, a_down, "linewhip")
        why = feasible(sim, kb, stroke)
        if why:
            print(f"    stroke infeasible ({why}) at |v| {np.linalg.norm(v_cmd):.2f} m/s")
            return None
        rel = rollout_grasped(sim, stroke, q_r, fingers)
        if rel is None:
            print("    no release in the calibration rollout")
            return None
        t_rel, p, v = rel
        err = need(p) - v
        print(f"    iter {i}: |v_cmd| {np.linalg.norm(v_cmd):.3f} m/s, release v {v.round(3)}, error "
              f"{1e3 * np.linalg.norm(err):.0f} mm/s, joint speed {stroke.max_joint_speed() / np.pi:.2f}, "
              f"torque {stroke_torque(sim, stroke):.2f}")
        if np.linalg.norm(err) < best[0]:
            best = (float(np.linalg.norm(err)), LoopThrow(stroke, target, t_rel, p, v))
        if best[0] < tol:
            break
        v_cmd = v_cmd + gain * err
    if best[0] > accept:
        print(f"    calibration did not converge (best {1e3 * best[0]:.0f} mm/s)")
        return None
    return best[1]


class CascadeHand:
    def __init__(self, sim: HandSim, th: LoopThrow, fingers: Fingers, base_z: float, s: float,
                 launch_noise: float, obs_noise: float, rng: np.random.Generator, z_min: float, settle: float,
                 d_pre: float, d_cush: float, min_wait: float, switch: float, strategy: str = "direct",
                 torque_frac: float = CATCH_TORQUE_FRAC, max_reach: float = 0.95, a_scale: float = 1.0,
                 depth: float = 0.0, held_speed: float = 0.15):
        self.sim, self.th, self.fingers, self.switch = sim, th, fingers, switch
        self.launch_noise, self.obs_noise, self.rng, self.settle = launch_noise, obs_noise, rng, settle
        self.torque_frac, self.held_speed = torque_frac, held_speed
        self.recorder = None
        self.a_max = a_scale * CATCH_A_MAX  # planning limit only: the torque check is the physical one
        m = sim.model
        self.g = m.opt.gravity
        self.seat = m.site("ball_seat").id
        self.obs_every = max(1, round(1.0 / (OBS_RATE * m.opt.timestep)))
        self.force_limit = m.actuator_forcerange[sim.arm_act, 1]
        kin = ArmKinematics(m, "bowl_point")
        d0 = mujoco.MjData(m)
        d0.qpos[sim.arm_qadr] = Q_READY
        mujoco.mj_forward(m, d0)
        cfg = PlannerConfig(strategy=strategy, z_min=base_z + z_min, z_max=base_z + 0.90, z_min_end=base_z + z_min - 0.10,
                            v_max=np.full(6, 0.9 * np.pi), a_max=self.a_max, min_wait=min_wait, earliest_catch=0.05,
                            cushion_dir="ball", d_pre=d_pre, d_cush=d_cush, cup_speed_max=0.7, max_reach=max_reach)
        self.cp = CatchPlanner(kin, cfg, Q_READY, d0.body("shoulder_link").xpos.copy(), self._torque_ok)
        # O's nominal release: H's calibrated release, mirrored
        self.p_O = mirror(th.p_release, s) + np.array([depth, 0.0, 0.0])
        self.v_O = th.v_release * np.array([1.0, -1.0, 1.0])
        self.target_O = th.target  # where H's throw should arrive (O's catch point)

    def _torque_ok(self, seg, n=15):
        lim = self.torque_frac * self.force_limit
        return all(np.all(np.abs(self.sim.arm_torque_ff(*seg.sample(t))) <= lim) for t in np.linspace(seg.t0, seg.t1, n))

    def _carry(self, t0, q0, q1):
        """Fastest rest-to-rest quintic within CARRY_V_MAX / CATCH_A_MAX and the torque fraction."""
        a = self.a_max.copy()
        for _ in range(12):
            move = min_time_quintic(t0, q0, np.zeros(6), np.zeros(6), q1, CARRY_V_MAX, a)
            if self._torque_ok(move):
                return move
            a = a / 1.25
        return move

    def _rel(self, ball):
        """Position and velocity of a ball relative to the ball seat."""
        m, d = self.sim.model, self.sim.data
        v = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.seat, v, 0)
        qa, da, _ = self.sim.balls[ball]
        return d.qpos[qa:qa + 3] - d.site_xpos[self.seat], d.qvel[da:da + 3] - v[3:]

    def _clear(self, ball):
        return np.linalg.norm(self._rel(ball)[0]) > CLEAR_DIST

    def land_error(self, p, v):
        """Horizontal miss of a ball released with (p, v) at O's catch height (descending)."""
        z = self.target_O[2]
        a, b, c = 0.5 * self.g[2], v[2], p[2] - z
        disc = b * b - 4 * a * c
        if disc < 0:
            return np.inf
        t = (-b - np.sqrt(disc)) / (2 * a)
        q = p + v * t + 0.5 * self.g * t ** 2
        return float(np.linalg.norm((q - self.target_O)[:2]))

    # ------------------------------------------------------------ run
    def run(self, tau: float, n_throws: int, verbose=False) -> Run:
        sim, d, hands, th = self.sim, self.sim.data, self.fingers.hands, self.th
        stroke = th.stroke
        res = self.res = Run()
        # start: ball 0 settled and grasped at the release pose, carried to the stroke start
        held = 0
        sim.select_ball(held)
        q_r = stroke.q[np.argmin(np.abs(stroke.s))]
        sim.reset(q_r, hands["settle"])
        sim.place_ball(BALL_ON_PALM + np.array([0.0, 0.0, -sim.ball_radius - 0.002]))
        t_grasp, t_move = 0.4, 0.9
        carry = min_time_quintic(t_move, q_r, np.zeros(6), np.zeros(6), stroke.q_start, np.full(6, 1.0), np.full(6, 4.0))
        traj = Trajectory([Hold(0.0, t_move, q_r), carry, Hold(carry.t1, np.inf, stroke.q_start)])
        T0 = max(carry.t1 + 0.2 + th.t_release, tau + 0.05)
        t_rel = lambda k: T0 + 2 * k * tau  # H's nominal releases
        self.launches = [T0 + (2 * k - 1) * tau for k in range(n_throws + 1)]  # O's releases
        self.flights = []  # balls launched by O, in order: [ball, t_launch, nominal predictor, obs_t, obs_p]
        self.parked = set(range(1, len(sim.balls)))
        self.outgoing = None  # (ball, t_release) of H's last throw, until it is parked

        while d.time < t_move:  # settle, grasp
            if d.time >= t_grasp:
                sim.set_hand(hands["grasp"])
            sim.control_arm(*traj.sample(d.time))
            if (r := self._step()):
                res.outcome = r
                return self._finish(res)
        t_touch_prev = None
        for k in range(n_throws):
            t_start = t_rel(k) - th.t_release
            rec = Cycle(k, carry_spare=t_start - carry.t1)
            res.cycles.append(rec)
            if rec.carry_spare < self.settle:
                res.outcome = "LATE"
                res.diag = f"carry ends {1e3 * rec.carry_spare:+.0f} ms before the stroke start (need {1e3 * self.settle:.0f})"
                break
            # O releases the incoming ball tau before H's release, normally during the carry / wait
            while not self.flights and d.time < t_start:
                sim.set_hand(hands["grasp"])
                sim.control_arm(*traj.sample(d.time))
                if (r := self._step()) or sim.ball_contacts(held).ground:
                    res.outcome = r or "DROPPED"
                    return self._finish(res)
            if not self.flights:
                res.outcome, res.diag = "NO_INCOMING", "tau shorter than the stroke's acceleration"
                break
            flight = self.flights[0]
            seg = StrokeSegment(stroke, t_start)
            pre = traj.segments[:-1] + [Hold(traj.segments[-1].t0, t_start, stroke.q_start)]
            t_sw = t_start + stroke.t_peak + self.switch  # the catch approach takes over from the stroke here
            catch = self.cp.plan(flight[2], t_sw, seg.sample(t_sw))
            if catch is not None:
                traj = Trajectory(pre + [Until(seg, t_sw)] + catch.trajectory.segments)
            else:  # fall back: finish the stroke, then go
                t_sw = seg.t1
                why = dict(self.cp.rejections)
                catch = self.cp.plan(flight[2], t_sw, (stroke.q_end, np.zeros(6), np.zeros(6)))
                if catch is None:
                    res.outcome = "NO_CATCH_PLAN"
                    res.diag = f"from the switch: {why}; from the stroke end: {dict(self.cp.rejections)}"
                    break
                traj = Trajectory(pre + [seg] + catch.trajectory.segments)
            outcome, traj = self._cycle(rec, seg, t_sw, traj, catch, held, flight)
            if t_touch_prev is not None and np.isfinite(rec.t_release):
                res.cycles[-2].D = rec.t_release - t_touch_prev
            if outcome != "CAUGHT":
                res.outcome = outcome
                break
            res.throws = k + 1
            self.flights.pop(0)
            held = flight[0]  # the caught ball
            sim.select_ball(held)
            t_touch_prev = rec.t_touch
            if verbose:
                print(f"    throw {k + 1}: dv {1e3 * rec.release_dv:.0f} mm/s, miss {1e3 * rec.land_err:.0f} mm, "
                      f"E {rec.E:.3f} s, carry spare {1e3 * rec.carry_spare:.0f} ms, catch spare "
                      f"{1e3 * rec.catch_spare:.0f} ms / acc {rec.catch_acc:.2f}, replans {rec.replans}, impact {rec.impact:.2f} m/s, "
                      f"{'switch' if t_sw < seg.t1 else 'full stroke'}")
            # carry the caught ball straight on to the next stroke start
            q_now = traj.sample(d.time)[0]
            carry = self._carry(d.time, q_now, stroke.q_start)
            traj = Trajectory([Hold(0.0, d.time, q_now), carry, Hold(carry.t1, np.inf, stroke.q_start)])
        return self._finish(res)

    def _margin(self, rec, catch, wait):
        approach = catch.trajectory.segments[0]
        if self.cp.cfg.strategy == "direct":
            rec.catch_acc = float(np.max(approach.peaks()[1] / self.cp.cfg.a_max))
        else:
            rec.catch_spare = catch.t_match - approach.t1 - wait

    def _collision_diag(self) -> str:
        """Which balls touch, where (relative to the ball seat), and their roles."""
        sim, d = self.sim, self.sim.data
        out = [getattr(self, "catch_state", "")]
        roles = {f[0]: f"incoming (launched {1e3 * (d.time - f[1]):.0f} ms ago)" for f in self.flights}
        if self.outgoing is not None:
            roles[self.outgoing[0]] = f"outgoing (released {1e3 * (d.time - self.outgoing[1]):.0f} ms ago)"
        for i in range(d.ncon):
            c = d.contact[i]
            pair = [k for k, b in enumerate(sim.balls) if b[2] in (c.geom1, c.geom2)]
            if len(pair) == 2:
                where = 1e3 * (c.pos - d.site_xpos[self.seat])
                out.append(" / ".join(roles.get(k, f"ball {k} (held)") for k in pair) + f" at {where.round(0)} mm from the seat")
        return "; ".join(out)

    def _finish(self, res):
        res.speed_peak = self.sim.speed_peak
        return res

    def _step(self) -> str:
        """Launch O's balls on schedule, step, record observations, retire H's thrown ball."""
        sim, d = self.sim, self.sim.data
        if self.launches and d.time >= self.launches[0]:
            t = self.launches.pop(0)
            i = min(self.parked)
            self.parked.discard(i)
            v = self.v_O + (self.rng.normal(0, self.launch_noise, 3) if self.launch_noise else 0.0)
            sim.set_ball_state(i, self.p_O, v)
            pred = BallisticPredictor(self.g)
            pred.t_ref, pred.p0, pred.v0 = t, self.p_O.copy(), self.v_O.copy()
            self.flights.append([i, t, pred, [], []])
        sim.step()
        if self.recorder is not None:
            self.recorder.capture(d)
        self.res.arm_force = max(self.res.arm_force, float(np.max(np.abs(d.actuator_force[sim.arm_act]) / self.force_limit)))
        if round(d.time / sim.model.opt.timestep) % self.obs_every == 0:
            for f in self.flights:
                qa = sim.balls[f[0]][0]
                f[3].append(d.time)
                f[4].append(d.qpos[qa:qa + 3] + (self.rng.normal(0, self.obs_noise, 3) if self.obs_noise else 0.0))
        if self.outgoing is not None:
            c = sim.ball_contacts(self.outgoing[0])
            if c.hand or c.arm or c.ground:
                return "OUTGOING_HIT"
            if c.ball:
                self.res.diag = self._collision_diag()
                return "BALL_COLLISION"
            if d.time >= self.outgoing[1] + PARK_AFTER:
                sim.park_ball(self.outgoing[0])
                self.parked.add(self.outgoing[0])
                self.outgoing = None
        if sim.speed_peak > 1.0:
            return "SPEED_LIMIT"
        return ""

    def _cycle(self, rec, seg, t_sw, traj, catch, held, flight):
        """H's throw of ball `held`, then the catch of the incoming ball. Returns (outcome, trajectory)."""
        sim, d, hands, th = self.sim, self.sim.data, self.fingers.hands, self.th
        ball_in, t_launch, _, obs_t, obs_p = flight
        t_peak = seg.shift + th.stroke.t_peak
        state, cand, checked = "held", None, False
        self.catch_state = "before the release"
        last_fit = -np.inf
        predictor = BallisticPredictor(self.g)
        closed = False
        self._margin(rec, catch, self.cp.cfg.min_wait)
        sim.select_ball(held)
        while True:
            t = d.time
            if not checked and t >= seg.t0:  # the ball caught last cycle must sit in the hand by now
                checked = True
                rel_p, rel_v = self._rel(held)
                if (not sim.ball_contacts(held).hand or np.linalg.norm(rel_p) > SEAT_TOL
                        or np.linalg.norm(rel_v) > self.held_speed):
                    self.res.diag = f"held ball {1e3 * rel_p.round(3)} mm / {1e3 * np.linalg.norm(rel_v):.0f} mm/s from ball_seat"
                    return "NOT_CAUGHT", traj
            locked = t >= catch.t_match - REPLAN_LOCK
            if (np.isnan(rec.t_touch) and not locked and t >= t_sw and t >= t_launch + OBS_WINDOW
                    and t - last_fit >= REPLAN_PERIOD):
                predictor.fit_with_prior(obs_t, obs_p, t_launch, self.p_O, self.v_O,
                                         max(self.obs_noise, SIGMA_OBS_MIN), PRIOR_SIGMA_P, PRIOR_SIGMA_V)
                last_fit = t
                if np.linalg.norm(predictor.predict(catch.t_catch)[0] - catch.p_catch) > REPLAN_THRESHOLD:
                    new = self.cp.plan(predictor, t, traj.sample(t), min_wait=REPLAN_MIN_WAIT)
                    if new is not None:
                        catch, traj = new, new.trajectory
                        self._margin(rec, catch, REPLAN_MIN_WAIT)
                        rec.replans += 1
            # fingers: open to throw; catch posture once the thrown ball is clear of the hand and the
            # incoming one descends; grasp on palm contact
            da = sim.balls[ball_in][1]
            if state != "free":
                sim.set_hand(self.fingers.opening(t, t_peak))
            elif closed:
                sim.set_hand(hands["catch_grasp"])
            elif t >= t_sw and d.qvel[da + 2] < 0 and (self.outgoing is None or self._clear(self.outgoing[0])):
                sim.set_hand(hands["catch"])
            sim.control_arm(*traj.sample(t))
            if (r := self._step()):
                return r, traj
            c_held, c_in = sim.ball_contacts(held), sim.ball_contacts(ball_in)
            if c_in.ball or (state != "free" and c_held.ball):
                self.res.diag = self._collision_diag()
                return "BALL_COLLISION", traj
            if c_in.ground or (state != "free" and c_held.ground):
                return "DROPPED", traj
            if c_in.arm or (state != "free" and c_held.arm):
                return "HIT_ARM", traj
            if c_in.hand and (state != "free" or t < t_sw):
                return "INCOMING_HIT", traj
            if state == "held":
                if not c_held.hand and t > t_peak - th.stroke.T1:
                    state, cand = "maybe", (t, *sim.ball_state())
                elif t > seg.t1 + 0.2:
                    return "NO_RELEASE", traj
            elif state == "maybe":
                if c_held.hand:
                    state = "held"
                elif t - cand[0] >= FREE_CONFIRM:
                    state = "free"
                    rec.t_release = cand[0]
                    dt = seg.shift + th.t_release - cand[0]
                    rec.release_dv = float(np.linalg.norm(cand[2] + self.g * dt - th.v_release))
                    rec.land_err = self.land_error(cand[1], cand[2])
                    self.outgoing = (held, cand[0])
            else:
                if c_in.hand and np.isnan(rec.t_touch):
                    rec.t_touch = t
                    rec.E = t - rec.t_release
                    rec.impact = float(np.linalg.norm(self._rel(ball_in)[1]))
                closed = closed or c_in.palm
                self.catch_state = f"incoming ball touched the hand {1e3 * (t - rec.t_touch):.0f} ms ago, palm contact (fingers closed) {closed}"                     if np.isfinite(rec.t_touch) else "incoming ball not touched yet"
                if closed and t >= traj.t_end:  # cushion over: carry on if the ball is in the hand
                    rel_p = self._rel(ball_in)[0]
                    if c_in.hand and np.linalg.norm(rel_p) < SEAT_TOL:
                        return "CAUGHT", traj
                    if t >= traj.t_end + SEAT_WAIT:
                        self.res.diag = (f"grasped off the seat: ball {1e3 * rel_p.round(3)} mm from ball_seat, "
                                         f"in hand {c_in.hand}")
                        return "NOT_CAUGHT", traj
                if t > catch.t_catch + CATCH_TIMEOUT:
                    rel_p, rel_v = self._rel(ball_in)
                    self.res.diag = (f"ball {1e3 * rel_p.round(3)} mm / {1e3 * np.linalg.norm(rel_v):.0f} mm/s from "
                                     f"ball_seat, touched {np.isfinite(rec.t_touch)}, closed {closed}")
                    return "NOT_CAUGHT", traj


def summarize(runs: list[Run], cap: int) -> str:
    full = sum(r.outcome == "CAP" for r in runs)
    fails = {}
    for r in runs:
        if r.outcome != "CAP":
            fails[r.outcome] = fails.get(r.outcome, 0) + 1
    cyc = [c for r in runs for c in r.cycles]
    def stat(name, scale=1.0):
        x = np.array([getattr(c, name) for c in cyc], dtype=float) * scale
        x = x[np.isfinite(x)]
        return (x.mean(), x.min(), x.max()) if len(x) else (np.nan,) * 3
    E, D, cs, ks, ka = stat("E"), stat("D"), stat("carry_spare", 1e3), stat("catch_spare", 1e3), stat("catch_acc")
    dv, miss = stat("release_dv", 1e3), stat("land_err", 1e3)
    return (f"{full}/{len(runs)} runs reached {cap} throws; failures {fails or '-'}; {sum(r.throws for r in runs)} throws caught\n"
            f"      E {E[0]:.3f} ({E[1]:.3f}-{E[2]:.3f}) s, D {D[0]:.3f} ({D[1]:.3f}-{D[2]:.3f}) s; spare: carry "
            f"{cs[0]:.0f} (min {cs[1]:.0f}) ms, catch approach {ks[0]:.0f} (min {ks[1]:.0f}) ms / peak acc {ka[0]:.2f} (max {ka[2]:.2f})\n"
            f"      H release dv {dv[0]:.0f} (max {dv[2]:.0f}) mm/s, miss at O's catch point {miss[0]:.0f} (max {miss[2]:.0f}) mm; "
            f"joint speed peak {max(r.speed_peak for r in runs):.2f}, arm force peak {max(r.arm_force for r in runs):.2f}")


def main() -> int:
    p = argparse.ArgumentParser(description="v4 E5: one hand of a three-ball cascade (single arm + mirrored launcher).")
    p.add_argument("--pedestal", type=float, default=0.4)
    p.add_argument("--release", nargs=2, type=float, default=[0.8, 0.25], help="x, z of H's release (ball centre) from the base (m)")
    p.add_argument("--s", type=float, default=-0.2,
                   help="O's throw point at y = -s (m); negative: O on H's left, so this LEAP right hand gets the "
                        "balls moving towards its thumb side (positive: towards the open ring-finger side)")
    p.add_argument("--w", type=float, default=-0.15,
                   help="catch point this far outside the own throw point (m, same sign as s); the ball just thrown "
                        "and the one coming down pass ~0.95 |w| apart (centres)")
    p.add_argument("--dx", type=float, default=0.0, help="catch points this far further out in x (m)")
    p.add_argument("--depth", type=float, default=0.0, help="O's ball paths this far further out in x than H's (m)")
    p.add_argument("--drop", type=float, default=0.0, help="catch points this far below the release (m)")
    p.add_argument("--F", type=float, default=0.60, help="flight time to the catch point (s)")
    p.add_argument("--tau", nargs="+", type=float, default=[0.37], help="beat (s)")
    p.add_argument("--a-up", type=float, default=3.0, help="g")
    p.add_argument("--a-down", type=float, default=4.0, help="g (the stroke's own deceleration; used by the fallback)")
    p.add_argument("--switch", type=float, default=0.03, help="catch approach takes over this long after the peak speed (s)")
    p.add_argument("--catch", choices=["direct", "cushion"], default="direct",
                   help="direct: one quintic from the moving arm to the catch pose; cushion: E3a's approach, rest, match")
    p.add_argument("--catch-torque", type=float, default=0.9,
                   help="feed-forward torque limit of the catch approach and the carry (fraction of the actuator limits)")
    p.add_argument("--catch-a", type=float, default=2.0,
                   help="joint acceleration limit of the catch approach and the carry, x hand_juggle.CATCH_A_MAX "
                        "(a planning parameter; the torque check is the physical limit)")
    p.add_argument("--held-speed", type=float, default=0.15,
                   help="max. speed of the held ball relative to the ball seat when the stroke starts (m/s)")
    p.add_argument("--max-reach", type=float, default=1.0, help="catch planner's shoulder -> catch point prune (m)")
    p.add_argument("--z-min", type=float, default=0.0, help="lowest catch height above the base (m)")
    p.add_argument("--d-pre", type=float, default=0.03, help="match phase length (m)")
    p.add_argument("--d-cush", type=float, default=0.03, help="cushion length (m)")
    p.add_argument("--min-wait", type=float, default=0.02, help="rest at the pre-catch pose (s)")
    p.add_argument("--settle", type=float, default=0.01, help="min. rest at the stroke start before a throw (s)")
    p.add_argument("--launch-noise", type=float, default=0.0, help="std. dev. per axis of O's release velocity (m/s)")
    p.add_argument("--obs-noise", type=float, default=0.0)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--throws", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", type=Path, default=None, help="record the first run of the first tau (mp4, camera 'loop')")
    p.add_argument("--video-speed", type=float, default=0.5, help="simulated s per video s")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    sim = HandSim(load_model(pedestal=args.pedestal, extra_balls=3))
    set_bowl_sites(sim.model)
    kb = ArmKinematics(sim.model, "bowl_seat")
    fingers = Fingers(sim, -0.02, 0.03, bowl=True)
    base = np.array([0.0, 0.0, args.pedestal])
    x, z = args.release
    R = base + np.array([x, 0.0, z])
    target = base + np.array([x + args.dx, -args.s - args.w, z - args.drop])
    ik = kb.solve(R, np.array([0.0, 0.0, 1.0]), Q_READY, Q_READY)
    if not ik.success:
        print("no IK solution at the release point")
        return 1
    print(f"Setup: pedestal {args.pedestal} m, H release {R.round(3)}, O's catch point {target.round(3)} "
          f"(s {args.s}, w {args.w}, dx {args.dx}, depth {args.depth}, drop {args.drop} m), F {args.F} s, linewhip {args.a_up}/{args.a_down} g, "
          f"switch {1e3 * args.switch:.0f} ms, {args.catch} catch d_pre/d_cush {args.d_pre}/{args.d_cush} m, wait {args.min_wait} s, "
          f"settle {args.settle} s, catch accel x{args.catch_a}, torque {args.catch_torque}, launch noise {1e3 * args.launch_noise:.0f} mm/s, obs noise {1e3 * args.obs_noise:.0f} mm")
    print("  calibrating H's throw:")
    th = calibrate_whip(sim, kb, ik.q, target, args.F, fingers, args.a_up * G, args.a_down * G)
    if th is None:
        return 1
    st = th.stroke
    print(f"    -> V {st.V:.2f} m/s, release {1e3 * (th.t_release - st.t_peak):+.0f} ms after the peak, |v| "
          f"{np.linalg.norm(th.v_release):.2f} m/s, stroke {st.T1:.3f} + {st.T2:.3f} s, joint speed "
          f"{st.max_joint_speed() / np.pi:.2f}, torque {stroke_torque(sim, st):.2f}")
    print("  cascade: D = 3 tau - F, E = F - tau (nominal, to the catch point)")
    for tau in args.tau:
        rng = np.random.default_rng(args.seed)
        hand = CascadeHand(sim, th, fingers, args.pedestal, args.s, args.launch_noise, args.obs_noise, rng,
                           args.z_min, args.settle, args.d_pre, args.d_cush, args.min_wait, args.switch, args.catch, args.catch_torque, args.max_reach, args.catch_a, args.depth, args.held_speed)
        runs = []
        t0 = time.perf_counter()
        for i in range(args.runs):
            if args.video is not None and i == 0 and tau == args.tau[0]:
                hand.recorder = VideoRecorder(sim.model, args.video, speed=args.video_speed, camera="loop")
            r = hand.run(tau, args.throws, args.verbose)
            if hand.recorder is not None:
                hand.recorder.close()
                hand.recorder = None
            runs.append(r)
            if args.verbose or r.outcome != "CAP":
                print(f"    tau {tau:.3f} run {i + 1}: {r.throws} throws caught, {r.outcome} {r.diag}")
        print(f"  tau {tau:.3f} s (nominal D {3 * tau - args.F:.3f}, E {args.F - tau:.3f}): {summarize(runs, args.throws)} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""v5: two UR5e arms with LEAP hands (right + left, side by side) juggling -- two-ball synchronous
exchange (S1) and the three-ball cascade, from one generic throw-event table.

Layout: both bases face +x on 0.4 m pedestals, the right arm (r_, right hand) at y = -Y, the left
arm (l_, left hand) at y = +Y; the left arm (UR5e + hand) is the exact mirror image of the right one
(jugglebot.hands.model.mirror_spec). Each arm is planned in its own base frame on its single-arm model
(jugglebot.hands); the joint trajectories run in the shared scene.

Patterns (world points, ball centres; x0, z0 = work point, relative to the pedestal top for z):
  sync2   : both hands throw at t = kP (P = period). The two ball paths are depth apart in x so
            they do not meet mid-air (v3: equal flights in one plane always collide):
              R throws from (x0 + depth/2, -w, z0) to L's catch point (x0 + depth/2, +w, z0)
              L throws from (x0 - depth/2, +w, z0) to R's catch point (x0 - depth/2, -w, z0)
            Each hand catches the other's k-th throw after its own k-th throw (E = F, D = P - F).
  cascade3: siteswap 3, beat tau: R releases at 2k tau, L at (2k + 1) tau; throw inside, catch
            outside (v4 E5's geometry): R throws from (x0, -s/2, z0) to L's catch (x0, +s/2 + w, z0),
            L from (x0, +s/2, z0) to R's catch (x0, -s/2 - w, z0). L catches R's k-th throw, R
            catches L's (k - 1)-th; the third ball is fed in at t = -tau as if L had thrown it
            (a launcher releasing it with L's nominal release state -- a LEAP hand holds one ball).

Per throw type (one per hand) the linewhip is calibrated offline on the thrower's single-arm model
(jugglebot.hands.throwing.calibrate_whip). Each hand then runs v4 E5's cycle: stroke -> switch to
the "direct" catch of its next incoming ball (planned from the thrower's nominal release, re-planned
from observations with the release-prior fit) -> carry to the next stroke start -> wait -> throw.
New against E5: the carry starts --grasp-delay after the palm contact straight from the moving
cushion (one quintic from the arm's state to the stroke start, no stop), and the catch cost prefers
catch poses near the next stroke start (--next-weight).

A run stops at the first failure (of either hand) or after --throws throws per hand. Failures as
v4 E5 (LATE, NO_CATCH_PLAN, NOT_CAUGHT, DROPPED, HIT_ARM, INCOMING_HIT, NO_RELEASE, SPEED_LIMIT,
BALL_COLLISION, OUTGOING_HIT), plus ARM_COLLISION (the two robots touch).

Usage (from the repo root):
    uv run python v5_two_arm_hands/juggle.py --pattern sync2
    uv run python v5_two_arm_hands/juggle.py --pattern cascade3
    uv run python v5_two_arm_hands/juggle.py --pattern cascade3 --launch-noise 0 --obs-noise 0.005 --runs 5 --throws 30
    uv run python v5_two_arm_hands/juggle.py --pattern cascade3 --runs 1 --throws 8 --video videos/v5_cascade3.mp4
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from jugglebot.hands.handsim import Q_READY, BALL_ON_PALM, set_bowl_sites
from jugglebot.hands.kinematics import ArmKinematics
from jugglebot.hands.model import ArmPlacement, load_arm
from jugglebot.hands.planning import CatchPlanner, Hold, PlannerConfig, Trajectory, min_time_quintic
from jugglebot.hands.prediction import BallisticPredictor
from jugglebot.hands.throwing import FREE_CONFIRM, Fingers, Throw, calibrate_whip, stroke_torque
from jugglebot.hands.world import World
from jugglebot.video import VideoRecorder, ViewerPacer

G = 9.81
GRAVITY = np.array([0.0, 0.0, -G])
# Perception / re-planning (v1-v4)
OBS_RATE, OBS_WINDOW, REPLAN_PERIOD = 100.0, 0.05, 0.05
REPLAN_THRESHOLD, REPLAN_LOCK, REPLAN_MIN_WAIT = 0.005, 0.03, 0.02
PRIOR_SIGMA_P, PRIOR_SIGMA_V, SIGMA_OBS_MIN = 0.005, 0.03, 0.001
# Catching / carrying (v4 E5)
CATCH_A_MAX = np.array([40.0, 40.0, 40.0, 60.0, 60.0, 60.0])  # rad/s^2, x --catch-a (planning parameter)
SEAT_TOL, SEAT_WAIT, CATCH_TIMEOUT = 0.05, 0.1, 1.0
CLEAR_DIST = 0.15  # m, thrown ball this far from the seat before the fingers rise to catch
PARK_AFTER = 0.25  # s after the catch: nothing else to do with a ball? (unused: balls stay in play)


# ------------------------------------------------------------------ pattern
@dataclass
class HandPlan:
    prefix: str
    throw_point: np.ndarray  # world
    target: np.ndarray  # world: the other hand's catch point
    t_release: list  # nominal release times (pattern time)
    incoming: list  # per throw k: (source prefix or "launcher", index into the source's throws)


@dataclass
class Pattern:
    name: str
    hands: dict  # prefix -> HandPlan
    held: dict  # prefix -> ball held at the start
    launches: list = field(default_factory=list)  # (time, ball, source prefix whose nominal release it copies)
    F: float = 0.6


def make_pattern(args, n: int) -> Pattern:
    z = args.pedestal + args.z0
    if args.pattern == "sync2":
        hx, d = args.x0, args.depth / 2
        R_T, L_C = np.array([hx + d, -args.w, z]), np.array([hx + d, args.w, z])
        L_T, R_C = np.array([hx - d, args.w, z]), np.array([hx - d, -args.w, z])
        times = [k * args.period for k in range(n)]
        hands = {"r_": HandPlan("r_", R_T, L_C, times, [("l_", k) for k in range(n)]),
                 "l_": HandPlan("l_", L_T, R_C, times, [("r_", k) for k in range(n)])}
        return Pattern("sync2", hands, {"r_": 0, "l_": 1}, [], args.F)
    tau, s, w = args.tau, args.s, args.w_cascade
    R_T, R_C = np.array([args.x0, -s / 2, z]), np.array([args.x0, -s / 2 - w, z])
    L_T, L_C = np.array([args.x0, s / 2, z]), np.array([args.x0, s / 2 + w, z])
    hands = {"r_": HandPlan("r_", R_T, L_C, [2 * k * tau for k in range(n)], [("l_", k - 1) for k in range(n)]),
             "l_": HandPlan("l_", L_T, R_C, [(2 * k + 1) * tau for k in range(n)], [("r_", k) for k in range(n)])}
    hands["r_"].incoming[0] = ("launcher", 0)
    return Pattern("cascade3", hands, {"r_": 0, "l_": 1}, [(-tau, 2, "l_")], args.F)


# ------------------------------------------------------------------ hand controller
@dataclass
class Rec:
    k: int
    release_dv: float = np.nan
    land_err: float = np.nan
    E: float = np.nan
    D: float = np.nan
    carry_spare: float = np.nan
    delay: float = 0.0  # s the throw was put off (the carry was late)
    replans: int = 0
    impact: float = np.nan
    t_release: float = np.nan
    t_touch: float = np.nan


class Until:
    def __init__(self, seg, t1):
        self.seg, self.t0, self.t1 = seg, seg.t0, t1

    def sample(self, t):
        return self.seg.sample(t)


class Shifted:
    """A stroke (t0 = 0) shifted to start at t_start, as a trajectory segment."""

    def __init__(self, stroke, t_start):
        self.stroke, self.shift = stroke, t_start
        self.t0, self.t1 = t_start, t_start + stroke.t_end

    def sample(self, t):
        return self.stroke.sample(t - self.shift)


class Hand:
    """One arm's juggling cycle (v4 E5's CascadeHand, as a per-step state machine)."""

    def __init__(self, J: "Juggler", prefix: str, plan: HandPlan, throw: Throw, args):
        self.J, self.prefix, self.plan, self.th = J, prefix, plan, throw
        self.arm = J.world.arms[prefix]
        self.sim = self.arm.plan  # single-arm planning model (dynamics, torque checks)
        self.fingers = Fingers(self.sim)
        self.args = args
        m = self.sim.model
        kin = ArmKinematics(m, "bowl_point")
        d0 = mujoco.MjData(m)
        d0.qpos[self.sim.arm_qadr] = Q_READY
        mujoco.mj_forward(m, d0)
        base_z = args.pedestal
        self.a_max = args.catch_a * CATCH_A_MAX
        cfg = PlannerConfig(strategy="direct", z_min=base_z + args.z_min, z_max=base_z + 0.90,
                            z_min_end=base_z + args.z_min - 0.10, v_max=np.full(6, np.pi), a_max=self.a_max,
                            min_wait=0.02, earliest_catch=0.05, cushion_dir="ball", d_pre=0.03, d_cush=args.d_cush,
                            cup_speed_max=0.7, max_reach=args.max_reach, next_weight=args.next_weight)
        self.cp = CatchPlanner(kin, cfg, Q_READY, d0.body("shoulder_link").xpos.copy(), self._torque_ok)
        self.cp.q_next = throw.stroke.q_start  # every throw of this hand starts there
        self.force_limit = self.sim.dynamics.force_limit
        self.recs: list[Rec] = []

    # -- helpers in the arm frame
    def _torque_ok(self, seg, n=15):
        lim = self.args.catch_torque * self.force_limit
        return all(np.all(np.abs(self.sim.arm_torque_ff(*seg.sample(t))) <= lim) for t in np.linspace(seg.t0, seg.t1, n))

    def _carry(self, t0, q0, q1, v0=None, a0=None):
        a = self.a_max.copy()
        v0 = np.zeros(6) if v0 is None else v0
        a0 = np.zeros(6) if a0 is None else a0
        for i in range(12):
            move = min_time_quintic(t0, q0, v0, a0, q1, np.full(6, 0.9 * np.pi), a)
            if self._torque_ok(move):
                break
            a = a / 1.25
        self.carry_info = f"carry {1e3 * (move.t1 - t0):.0f} ms (accel limit x{0.8 ** i:.2f} for torque)"
        return move

    def nominal_incoming(self, k) -> BallisticPredictor | None:
        """Prediction (arm frame, absolute time) of the k-th incoming ball from its source's nominal release."""
        src, j = self.plan.incoming[k]
        J = self.J
        if src == "launcher":
            t, _, like = J.pattern.launches[j]
            th, t_rel = J.hands[like].th, J.T0 + t
            p, v = J.world_release(like)
        else:
            if j < 0 or j >= len(J.pattern.hands[src].t_release):
                return None
            t_rel = J.T0 + J.pattern.hands[src].t_release[j] + J.delay.get((src, j), 0.0)
            p, v = J.world_release(src)
        pred = BallisticPredictor(GRAVITY)
        pred.t_ref, pred.p0, pred.v0 = t_rel, self.arm.to_arm(p), self.arm.vec_to_arm(v)
        J.preds.setdefault((src, j), []).append(pred)  # shifted if the thrower puts the throw off
        return pred

    # -- setup
    def start(self, t_move: float):
        """Ball placed in the bowl at the release pose, grasped, carried to the stroke start."""
        q_r = self.th.q_release
        self.arm.reset(q_r, self.fingers.hands["settle"])
        self.carry = min_time_quintic(t_move, q_r, np.zeros(6), np.zeros(6), self.th.stroke.q_start, np.full(6, 1.0), np.full(6, 4.0))
        self.traj = Trajectory([Hold(0.0, t_move, q_r), self.carry, Hold(self.carry.t1, np.inf, self.th.stroke.q_start)])
        self.held = self.J.pattern.held[self.prefix]
        self.k = 0
        self.state = "setup"
        self.t_touch_prev = None

    def begin_cycle(self, t):
        """Plan throw k (stroke from the carry's end) and the catch of incoming k."""
        th, k, J = self.th, self.k, self.J
        self.t_start = J.T0 + self.plan.t_release[k] - th.t_release
        self.rec = Rec(k, carry_spare=self.t_start - self.carry.t1)
        self.recs.append(self.rec)
        late = self.args.settle - self.rec.carry_spare
        if late > 0:  # put the throw off a little (the receiver is told) or fail
            if late > self.args.max_delay:
                J.diag = f"{self.prefix} carry ends {1e3 * self.rec.carry_spare:+.0f} ms before the stroke start"
                return "LATE"
            self.rec.delay = late
            self.t_start += late
            J.delay[(self.prefix, k)] = late
            for pred in J.preds.get((self.prefix, k), []):
                pred.t_ref += late
        seg = Shifted(th.stroke, self.t_start)
        self.seg = seg
        pre = self.traj.segments[:-1] + [Hold(self.traj.segments[-1].t0, self.t_start, th.stroke.q_start)]
        self.nominal = self.nominal_incoming(k)
        if self.nominal is None:  # nothing more to catch: finish the stroke and stay
            self.traj = Trajectory(pre + [seg, Hold(seg.t1, np.inf, th.stroke.q_end)])
            self.catch, self.t_sw = None, seg.t1
        else:
            self.t_sw = self.t_start + th.stroke.t_peak + self.args.switch
            catch = self.cp.plan(self.nominal, self.t_sw, seg.sample(self.t_sw))
            if catch is not None:
                self.traj = Trajectory(pre + [Until(seg, self.t_sw)] + catch.trajectory.segments)
            else:
                why = dict(self.cp.rejections)
                self.t_sw = seg.t1
                catch = self.cp.plan(self.nominal, self.t_sw, (th.stroke.q_end, np.zeros(6), np.zeros(6)))
                if catch is None:
                    self.J.diag = f"{self.prefix} catch {k}: from the switch {why}, from the stroke end {dict(self.cp.rejections)}"
                    return "NO_CATCH_PLAN"
                self.traj = Trajectory(pre + [seg] + catch.trajectory.segments)
            self.catch = catch
        self.state, self.cand, self.checked, self.closed = "held", None, False, False
        self.last_fit = -np.inf
        self.predictor = BallisticPredictor(GRAVITY)
        return ""

    # -- per step: set targets (before the world step), then evaluate (after it)
    def command(self, t):
        hands, J, arm = self.fingers.hands, self.J, self.arm
        if self.state == "setup":
            if t >= 0.4:
                arm.set_hand(hands["grasp"])
        elif self.state in ("held", "maybe"):
            if hasattr(self, "seg") and t < self.seg.t0 - 0.03:
                arm.set_hand(hands["catch_grasp"])
            else:
                arm.set_hand(self.fingers.opening(t, self.seg.shift + self.th.stroke.t_peak))
        elif self.state == "free":
            inc = self.incoming_ball()
            if self.closed:
                arm.set_hand(hands["catch_grasp"])
            elif inc is not None and t >= self.t_sw and J.world.ball_state(inc)[1][2] < 0 and self._clear():
                arm.set_hand(hands["catch"])
            self._replan(t)
        elif self.state == "done":
            pass
        arm.control(*self.traj.sample(t))

    def incoming_ball(self):
        return self.J.assigned.get((self.prefix, self.k))

    def _clear(self):
        b = self.J.last_thrown.get(self.prefix)
        if b is None:
            return True
        return np.linalg.norm(self.J.world.rel_to_site(b, self.arm)[0]) > CLEAR_DIST

    def _replan(self, t):
        inc = self.incoming_ball()
        if inc is None or self.catch is None or np.isfinite(self.rec.t_touch):
            return
        obs_t, obs_p = self.J.obs[inc]
        t_launch = self.nominal.t_ref
        locked = t >= self.catch.t_match - REPLAN_LOCK
        if locked or t < self.t_sw or not obs_t or t < obs_t[0] + OBS_WINDOW or t - self.last_fit < REPLAN_PERIOD:
            return
        self.predictor.fit_with_prior(obs_t, [self.arm.to_arm(p) for p in obs_p], t_launch, self.nominal.p0, self.nominal.v0,
                                      max(self.args.obs_noise, SIGMA_OBS_MIN), PRIOR_SIGMA_P, PRIOR_SIGMA_V)
        self.last_fit = t
        if np.linalg.norm(self.predictor.predict(self.catch.t_catch)[0] - self.catch.p_catch) > REPLAN_THRESHOLD:
            new = self.cp.plan(self.predictor, t, self.traj.sample(t), min_wait=REPLAN_MIN_WAIT)
            if new is not None:
                self.catch, self.traj = new, new.trajectory
                self.rec.replans += 1

    def evaluate(self, t) -> str:
        J, W, arm, p = self.J, self.J.world, self.arm, self.prefix
        if self.state == "setup":
            if t >= self.carry.t1:
                return self.begin_cycle(t)
            return ""
        if self.state == "done":
            return ""
        if not self.checked and t >= self.seg.t0:  # the held ball must sit in the hand when the stroke starts
            self.checked = True
            rel_p, rel_v = W.rel_to_site(self.held, arm)
            if not W.contacts(self.held).hand(p) or np.linalg.norm(rel_p) > SEAT_TOL or np.linalg.norm(rel_v) > self.args.held_speed:
                J.diag = f"{p} held ball {1e3 * rel_p.round(3)} mm / {1e3 * np.linalg.norm(rel_v):.0f} mm/s from ball_seat"
                return "NOT_CAUGHT"
        c_held = W.contacts(self.held)
        inc = self.incoming_ball()
        c_in = W.contacts(inc) if inc is not None else None
        if self.state != "free":
            if c_held.ground:
                return "DROPPED"
            if c_held.link:
                return "HIT_ARM"
            if c_in is not None and c_in.hand(p):
                return "INCOMING_HIT"
        if self.state == "held":
            if not c_held.hand(p) and t > self.seg.shift + self.th.stroke.t_peak - self.th.stroke.T1:
                self.state, self.cand = "maybe", (t, *W.ball_state(self.held))
            elif t > self.seg.t1 + 0.2:
                return "NO_RELEASE"
        elif self.state == "maybe":
            if c_held.hand(p):
                self.state = "held"
            elif t - self.cand[0] >= FREE_CONFIRM:
                self.state = "free"
                self.rec.t_release = self.cand[0]
                t_nom = self.J.T0 + self.plan.t_release[self.k] + self.rec.delay
                p0, v0 = self.J.world_release(p)
                self.rec.release_dv = float(np.linalg.norm(self.cand[2] + GRAVITY * (t_nom - self.cand[0]) - v0))  # world
                self.rec.land_err = land_error(self.cand[1], self.cand[2], self.plan.target)
                J.on_release(p, self.k, self.held, self.cand[0])
                if self.t_touch_prev is not None and len(self.recs) >= 2:
                    self.recs[-2].D = self.cand[0] - self.t_touch_prev
                if self.catch is None:
                    self.state = "done"
        else:  # free: catching
            if inc is None:
                return ""
            if c_in.ground:
                return "DROPPED"
            if c_in.link:
                J.diag = f"{p} incoming ball hit the links of {sorted(c_in.link)}"
                return "HIT_ARM"
            if c_in.hand(p) and t < self.t_sw:
                return "INCOMING_HIT"
            if c_in.hand(p) and np.isnan(self.rec.t_touch):
                self.rec.t_touch = t
                self.rec.E = t - self.rec.t_release
                self.rec.impact = float(np.linalg.norm(W.rel_to_site(inc, arm)[1]))
            if not self.closed and p in c_in.palm:
                self.t_closed = t
            self.closed = self.closed or (p in c_in.palm)
            if (self.args.early_carry and self.closed and t >= self.t_closed + self.args.grasp_delay
                    and c_in.hand(p) and np.linalg.norm(W.rel_to_site(inc, arm)[0]) < SEAT_TOL):
                return self._caught(t, inc, moving=True)  # carry on straight from the cushion
            if self.closed and t >= self.traj.t_end:
                rel_p = W.rel_to_site(inc, arm)[0]
                if c_in.hand(p) and np.linalg.norm(rel_p) < SEAT_TOL:
                    return self._caught(t, inc)
                if t >= self.traj.t_end + SEAT_WAIT:
                    J.diag = f"{p} grasped off the seat: ball {1e3 * rel_p.round(3)} mm from ball_seat"
                    return "NOT_CAUGHT"
            if t > self.catch.t_catch + CATCH_TIMEOUT:
                J.diag = f"{p} catch {self.k}: touched {np.isfinite(self.rec.t_touch)}, palm contact {self.closed}"
                return "NOT_CAUGHT"
        return ""

    def _caught(self, t, inc, moving: bool = False) -> str:
        self.held = inc
        self.t_touch_prev = self.rec.t_touch
        if self.args.trace:
            pc = self.catch.p_catch
            print(f"      {self.prefix} catch {self.k}: E {self.rec.E:.3f} s, planned catch {1e3 * (self.catch.t_catch - self.J.T0):.0f} ms "
                  f"at {(pc - [0, 0, self.args.pedestal]).round(3)} (arm frame, z above the base), replans {self.rec.replans}, "
                  f"carry from t_touch + {1e3 * (t - self.rec.t_touch):.0f} ms", end="")
        self.k += 1
        self.J.caught[self.prefix] += 1
        if self.k >= len(self.plan.t_release):
            self.state = "done"
            q_now = self.traj.sample(t)[0]
            self.traj = Trajectory([Hold(0.0, np.inf, q_now)])
            return ""
        q_now, qd_now, qdd_now = self.traj.sample(t)
        if moving:
            self.carry = self._carry(t, q_now, self.th.stroke.q_start, qd_now, qdd_now)
        else:
            self.carry = self._carry(t, q_now, self.th.stroke.q_start)
        self.traj = Trajectory([Hold(0.0, t, q_now), self.carry, Hold(self.carry.t1, np.inf, self.th.stroke.q_start)])
        r = self.begin_cycle(t)
        if self.args.trace:
            print(f", {self.carry_info}, spare {1e3 * self.rec.carry_spare:.0f} ms")
        return r


def land_error(p, v, target) -> float:
    """Horizontal miss at the target height (descending branch) of a ball released with (p, v)."""
    a, b, c = 0.5 * GRAVITY[2], v[2], p[2] - target[2]
    disc = b * b - 4 * a * c
    if disc < 0:
        return np.inf
    s = (-b - np.sqrt(disc)) / (2 * a)
    q = p + v * s + 0.5 * GRAVITY * s ** 2
    return float(np.linalg.norm((q - target)[:2]))


# ------------------------------------------------------------------ juggler
@dataclass
class Run:
    outcome: str = "CAP"
    who: str = ""
    caught: dict = field(default_factory=dict)
    diag: str = ""
    speed_peak: float = 0.0
    force_peak: float = 0.0
    recs: dict = field(default_factory=dict)


class Juggler:
    def __init__(self, world: World, pattern: Pattern, throws: dict, args, rng):
        self.world, self.pattern, self.args, self.rng = world, pattern, args, rng
        self.throws = throws
        self.hands = {p: Hand(self, p, pattern.hands[p], throws[p], args) for p in pattern.hands}
        self.obs_every = max(1, round(1.0 / (OBS_RATE * world.model.opt.timestep)))
        self.recorder = None
        self.viewer = None

    def world_release(self, prefix):
        """Nominal release state of a hand's throw, in the world."""
        th, arm = self.throws[prefix], self.world.arms[prefix]
        return arm.to_world(th.p_release), arm.vec_to_world(th.v_release)

    def on_release(self, prefix, k, ball, t):
        """A hand released ball `ball` with its k-th throw: the receiving hand's matching catch gets it."""
        self.last_thrown[prefix] = ball
        for q, h in self.hands.items():
            for kk, (src, j) in enumerate(h.plan.incoming):
                if src == prefix and j == k:
                    self.assigned[(q, kk)] = ball
        self.obs[ball] = ([], [])

    def run(self, n_throws: int) -> Run:
        W, d, args = self.world, self.world.data, self.args
        W.reset()
        res = Run(caught={p: 0 for p in self.hands})
        self.caught = res.caught
        self.assigned, self.last_thrown, self.obs, self.diag = {}, {}, {}, ""
        self.delay, self.preds = {}, {}
        t_move = 0.9
        for p, h in self.hands.items():
            h.start(t_move)
        W.forward()
        for p, h in self.hands.items():
            W.place_in_palm(self.pattern.held[p], h.arm, BALL_ON_PALM + np.array([0.0, 0.0, -W.ball_radius - 0.002]))
        t_ready = max(h.carry.t1 for h in self.hands.values()) + 0.2
        self.T0 = max(t_ready + max(h.th.t_release for h in self.hands.values()),
                      t_ready + max(0.0, -min([t for t, _, _ in self.pattern.launches] + [0.0])) + 0.05)
        launches = sorted((self.T0 + t, b, like) for t, b, like in self.pattern.launches)
        for t, b, like in launches:  # the receiving hand's catch of a launched ball
            for q, h in self.hands.items():
                for kk, (src, j) in enumerate(h.plan.incoming):
                    if src == "launcher":
                        self.assigned[(q, kk)] = b
        t_end_cap = self.T0 + max(h.plan.t_release[-1] for h in self.hands.values()) + 3.0
        while d.time < t_end_cap:
            if self.viewer is not None and not self.viewer.is_running():
                break
            t = d.time
            while launches and t >= launches[0][0]:
                tl, b, like = launches.pop(0)
                p0, v0 = self.world_release(like)
                v = v0 + (self.rng.normal(0, args.launch_noise, 3) if args.launch_noise else 0.0)
                W.set_ball_state(b, p0, v)
                self.obs[b] = ([], [])
            for h in self.hands.values():
                h.command(t)
            W.step()
            if self.recorder is not None:
                self.recorder.capture(d)
            if round(d.time / W.model.opt.timestep) % self.obs_every == 0:
                for b, (ot, op) in self.obs.items():
                    p = W.ball_state(b)[0]
                    ot.append(d.time)
                    op.append(p + (self.rng.normal(0, args.obs_noise, 3) if args.obs_noise else 0.0))
            fail, who = "", ""
            if W.arm_arm_contact():
                fail = "ARM_COLLISION"
            for b in range(len(W.balls)):
                if not fail and W.contacts(b).ball:
                    fail = "BALL_COLLISION"
            for p, h in self.hands.items():
                if not fail:
                    r = h.evaluate(d.time)
                    if r:
                        fail, who = r, p
                if not fail and h.arm.speed_peak > 1.005:  # 0.5% numerical tolerance for 2ms discrete-time differentiation
                    fail, who = "SPEED_LIMIT", p
            if fail:
                res.outcome, res.who, res.diag = fail, who, self.diag
                break
            if all(h.state == "done" for h in self.hands.values()):
                break
        res.speed_peak = max(a.speed_peak for a in W.arms.values())
        res.force_peak = max(a.force_peak for a in W.arms.values())
        res.recs = {p: h.recs for p, h in self.hands.items()}
        return res


def summarize(runs: list[Run], n: int) -> str:
    full = sum(r.outcome == "CAP" for r in runs)
    fails = {}
    for r in runs:
        if r.outcome != "CAP":
            key = f"{r.outcome}({r.who})" if r.who else r.outcome
            fails[key] = fails.get(key, 0) + 1
    recs = [c for r in runs for rs in r.recs.values() for c in rs]
    def stat(name, scale=1.0):
        x = np.array([getattr(c, name) for c in recs], dtype=float) * scale
        x = x[np.isfinite(x)]
        return (x.mean(), x.min(), x.max()) if len(x) else (np.nan,) * 3
    E, D, cs, dl = stat("E"), stat("D"), stat("carry_spare", 1e3), stat("delay", 1e3)
    n_del = sum(c.delay > 0 for c in recs)
    dv, miss = stat("release_dv", 1e3), stat("land_err", 1e3)
    caught = sum(sum(r.caught.values()) for r in runs)
    return (f"{full}/{len(runs)} runs completed {n} throws per hand; failures {fails or '-'}; {caught} catches\n"
            f"      E {E[0]:.3f} ({E[1]:.3f}-{E[2]:.3f}) s, D {D[0]:.3f} ({D[1]:.3f}-{D[2]:.3f}) s, carry spare {cs[0]:.0f} (min {cs[1]:.0f}) ms, "
            f"throws put off {n_del} (max {dl[2]:.0f} ms)\n"
            f"      release dv {dv[0]:.0f} (max {dv[2]:.0f}) mm/s, miss at the catch point {miss[0]:.0f} (max {miss[2]:.0f}) mm; "
            f"joint speed peak {max(r.speed_peak for r in runs):.2f}, arm force peak {max(r.force_peak for r in runs):.2f}")


def main() -> int:
    p = argparse.ArgumentParser(description="v5: two arms with LEAP hands, two-ball sync exchange / three-ball cascade.")
    p.add_argument("--pattern", choices=["sync2", "cascade3"], default="cascade3")
    p.add_argument("--Y", type=float, default=None,
                   help="base half-spacing (m): right arm at y = -Y, left at +Y (default: 0.35 for sync2, 0.30 for cascade3)")
    p.add_argument("--toe-in", type=float, default=None,
                   help="deg each base is turned towards the middle (default: 0.0 for cascade3; so throw point straight "
                        "ahead for sync2)")
    p.add_argument("--pedestal", type=float, default=0.4)
    p.add_argument("--x0", type=float, default=None,
                   help="work points' x (m, from the bases' line) (default: 0.85 for sync2, 0.80 for cascade3)")
    p.add_argument("--z0", type=float, default=0.25, help="work points' height above the pedestal top (m)")
    p.add_argument("--F", type=float, default=None,
                   help="flight time to the catch point (s) (default: 0.5 for sync2, 0.6 for cascade3)")
    # sync2
    p.add_argument("--w", type=float, default=0.2, help="sync2: throw / catch points at y = +-w (m)")
    p.add_argument("--depth", type=float, default=0.12, help="sync2: x distance between the two ball paths (m)")
    p.add_argument("--period", type=float, default=1.2, help="sync2: both hands throw every period (s)")
    # cascade3
    p.add_argument("--tau", type=float, default=0.36, help="cascade3: beat (s)")
    p.add_argument("--s", type=float, default=0.2, help="cascade3: distance between the throw points (m)")
    p.add_argument("--w-cascade", type=float, default=0.14, help="cascade3: catch points this far outside the throw points (m)")
    # throw / catch (v4 E5's settings)
    p.add_argument("--a-up", type=float, default=3.0, help="g")
    p.add_argument("--a-down", type=float, default=4.0, help="g")
    p.add_argument("--switch", type=float, default=0.03)
    p.add_argument("--z-min", type=float, default=None,
                   help="min catch height above pedestal (m) (default: 0.18 for sync2, 0.24 for cascade3)")
    p.add_argument("--d-cush", type=float, default=0.03)
    p.add_argument("--settle", type=float, default=0.01)
    p.add_argument("--held-speed", type=float, default=0.15)
    p.add_argument("--catch-a", type=float, default=2.0)
    p.add_argument("--catch-torque", type=float, default=0.9)
    p.add_argument("--max-reach", type=float, default=1.0)
    p.add_argument("--max-delay", type=float, default=0.04,
                   help="a hand whose carry is late puts its throw off by up to this (s); the receiving hand re-plans")
    p.add_argument("--no-early-carry", dest="early_carry", action="store_false",
                   help="finish the cushion (to rest) before carrying the caught ball (v4 E5)")
    p.add_argument("--grasp-delay", type=float, default=0.04,
                   help="early carry: start this long after the palm contact (fingers closed)")
    p.add_argument("--next-weight", type=float, default=1.0,
                   help="catch cost: weight of the joint distance from the catch pose to the next stroke start")
    p.add_argument("--launch-noise", type=float, default=0.0, help="cascade3: std. dev. per axis of the fed-in ball's velocity (m/s)")
    p.add_argument("--obs-noise", type=float, default=0.0)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--throws", type=int, default=20, help="throws per hand (use 0 or --infinite to keep juggling)")
    p.add_argument("--infinite", action="store_true", help="keep throwing indefinitely until closed")
    p.add_argument("--viewer", action="store_true", help="watch live in the interactive MuJoCo GUI window")
    p.add_argument("--speed", type=float, default=1.0, help="viewer playback speed (1.0 = real time, 0.5 = 2x slow-mo)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--video-speed", type=float, default=0.5)
    p.add_argument("--camera", default="three_quarter")
    p.add_argument("--verbose", action="store_true", help="print the calibration iterations")
    p.add_argument("--trace", action="store_true", help="print every catch")
    args = p.parse_args()

    if args.pattern == "sync2":
        if args.Y is None:
            args.Y = 0.35
        if args.z_min is None:
            args.z_min = 0.18
        if args.x0 is None:
            args.x0 = 0.85
        if args.F is None:
            args.F = 0.5
    else:  # cascade3
        if args.Y is None:
            args.Y = 0.3
        if args.toe_in is None:
            args.toe_in = 0.0
        if args.z_min is None:
            args.z_min = 0.24
        if args.x0 is None:
            args.x0 = 0.80
        if args.F is None:
            args.F = 0.6

    throws_to_plan = 100_000 if (args.infinite or args.throws <= 0) else args.throws
    pattern = make_pattern(args, throws_to_plan)
    if args.toe_in is None:  # throw point straight ahead of each arm
        T = pattern.hands["r_"].throw_point
        args.toe_in = float(np.degrees(np.arctan2(T[1] + args.Y, T[0])))
    psi = np.radians(args.toe_in)
    placements = [ArmPlacement("r_", "right", (0.0, -args.Y), psi), ArmPlacement("l_", "left", (0.0, args.Y), -psi)]
    plan_models = {}
    for a in placements:
        m = load_arm(a.side, args.pedestal, n_balls=1)
        set_bowl_sites(m, a.side)
        plan_models[a.prefix] = m
    world = World(placements, args.pedestal, n_balls=3, plan_models=plan_models)
    print(f"Setup: {pattern.name}, bases at y = -+{args.Y} m on {args.pedestal} m pedestals, turned {args.toe_in:.1f} deg inwards, F {args.F} s"
          + (f", period {args.period} s, w {args.w} m, depth {args.depth} m" if pattern.name == "sync2"
             else f", tau {args.tau} s, s {args.s} m, w {args.w_cascade} m")
          + f", obs noise {1e3 * args.obs_noise:.0f} mm"
          + (f", {'infinite' if throws_to_plan == 100_000 else str(args.throws)} throws" + (", GUI viewer" if args.viewer else "")))
    throws = {}
    for pre, hp in pattern.hands.items():
        arm = world.arms[pre]
        kb = ArmKinematics(arm.plan.model, "bowl_seat")
        R, target = arm.to_arm(hp.throw_point), arm.to_arm(hp.target)
        ik = kb.solve(R, np.array([0.0, 0.0, 1.0]), Q_READY, Q_READY)
        if not ik.success:
            print(f"  {pre}: no IK solution at the throw point {hp.throw_point.round(3)}")
            return 1
        print(f"  {pre}: throw point {hp.throw_point.round(3)} (arm frame {R.round(3)}) -> {hp.target.round(3)}; calibrating")
        th = calibrate_whip(arm.plan, kb, ik.q, target, args.F, Fingers(arm.plan), args.a_up * G, args.a_down * G, verbose=args.verbose)
        if th is None:
            print("    calibration failed")
            return 1
        st = th.stroke
        print(f"    V {st.V:.2f} m/s, release {1e3 * (th.t_release - st.t_peak):+.0f} ms after the peak, joint speed "
              f"{st.max_joint_speed() / np.pi:.2f}, torque {stroke_torque(arm.plan, st):.2f}")
        throws[pre] = th
    rng = np.random.default_rng(args.seed)
    J = Juggler(world, pattern, throws, args, rng)

    viewer = pacer = None
    if args.viewer:
        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(world.model, world.data)
        except RuntimeError as exc:
            hint = " (on macOS run it with mjpython)" if sys.platform == "darwin" else ""
            raise SystemExit(f"Could not launch viewer{hint}: {exc}")
        viewer.cam.lookat[:] = [0.8, 0.0, 0.65]
        viewer.cam.distance = 2.4
        viewer.cam.azimuth, viewer.cam.elevation = 205.0, -15.0
        pacer = ViewerPacer(viewer, speed=args.speed)

    runs = []
    t0 = time.perf_counter()
    n_runs = 1 if (args.infinite or args.viewer) and args.runs == 2 else args.runs
    for i in range(n_runs):
        if viewer is not None and not viewer.is_running():
            break
        if args.video is not None and i == 0:
            J.recorder = VideoRecorder(world.model, args.video, speed=args.video_speed, camera=args.camera)
        elif pacer is not None:
            J.recorder = pacer
            J.viewer = viewer
        r = J.run(throws_to_plan)
        if isinstance(J.recorder, VideoRecorder):
            J.recorder.close()
            J.recorder = None
        runs.append(r)
        print(f"  run {i + 1}: caught {r.caught}, {r.outcome}{'(' + r.who + ')' if r.who else ''} {r.diag}", flush=True)
    if viewer is not None:
        viewer.close()
    if runs:
        print(f"  {summarize(runs, throws_to_plan)} ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

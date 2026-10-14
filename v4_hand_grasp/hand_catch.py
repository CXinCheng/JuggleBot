"""E3a (v4): catching a launched ball with the LEAP hand.

v1's pipeline (prediction.py, planning.py) with the hand in place of the cup: a launcher
throws the 85 mm / 120 g ball towards a random point of the workspace; the ball is observed at
OBS_RATE (optionally noisy), its parabola fitted, and the catch planned and re-planned as in v1
-- the task point is the hand's catch_point site (25 mm above where the ball rests, like v1's
cup mouth), the palm normal faces the incoming ball (tilt <= 40 deg), cushioned catch.

Finger strategies (the hand waits in the catch posture: the cradle's thumb, fingers curled only
--catch-curl = 0.3 rad -- the cradle's 0.8 rad fingers stand up into a ball coming in over the
fingertips and knock it away):
    cradle  : the hand stays in the catch posture -- a passive catch
    close   : when the ball touches the palm (--close-on), --close-delay later, the grasp posture
              (the torque-limited fingers wrap the ball)

Compliance (user's choice): an ordinary ball; the palm and fingertip contacts are compliant and
damped (handsim.SKIN_*), and the arm gives way: it starts moving down d_pre before the contact,
reaches min(0.4 |v_ball|, speed cap) there and stops d_cush lower (planning.cushion_dir = "down").

E3c, --hand bowl (the default): the hand waits shaped as a bowl -- fingers up from the knuckles (mcp 1.1, pip 0.3),
heel raised --bowl-tilt (30 deg) so the knuckle line is lowest -- and the planner aims the bowl
axis (bowl_point's z axis, handsim.set_bowl_sites) at the ball; on palm contact the fingers curl
over it. --obs-bias offsets all observations of a trial horizontally (an aim error). Each
configuration starts from its own waiting pose (opening up, --yaw).

Outcomes: SUCCESS (the ball rests in the hand -- within SEAT_TOL of ball_seat, slower than
SPEED_THRESHOLD relative to it for HOLD_DURATION after the arm has stopped), MISSED (never touched
the hand), BOUNCED_OUT / DROPPED (touched the hand, then lost / on the ground), HIT_ARM, NOT_SETTLED,
NO_FEASIBLE_TARGET.

Usage (from the repo root):
    uv run python v4_hand_grasp/hand_catch.py --trials 20                 # bowl hand
    uv run python v4_hand_grasp/hand_catch.py --trials 20 --hand flat     # E3a's flat palm
    uv run python v4_hand_grasp/hand_catch.py --trials 20 --fingers cradle
    uv run python v4_hand_grasp/hand_catch.py --trials 20 --obs-noise 0.005
"""

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from grasp_test import posture, postures
from handsim import HandSim, Q_READY, set_bowl_sites, shift_site_in_palm
from kinematics import ArmKinematics
from planning import CatchPlanner, Hold, PlannerConfig, Trajectory
from prediction import BallisticPredictor
from video import VideoRecorder

# Launcher (as v1; the through-box is ~7 cm higher: the hand's catch point sits higher than the cup mouth)
LAUNCHER_POS = np.array([1.8, 0.0, 0.65])
LAUNCHER_JITTER = np.array([0.05, 0.15, 0.05])
THROUGH_X, THROUGH_Y, THROUGH_Z = (0.35, 0.65), (-0.30, 0.30), (0.52, 0.77)
FLIGHT_TIME = (0.8, 1.1)
# Perception / re-planning (as v1)
OBS_RATE, OBS_WINDOW, REPLAN_PERIOD = 100.0, 0.05, 0.05
REPLAN_THRESHOLD, REPLAN_LOCK, REPLAN_MIN_WAIT = 0.005, 0.03, 0.02
# Catch success
SEAT_TOL = 0.03  # m from ball_seat
SPEED_THRESHOLD = 0.05  # m/s relative to the hand
HOLD_DURATION = 0.5  # s
TIMEOUT_AFTER_CATCH = 2.0  # s
MAX_TRIAL_TIME = 5.0
CLOSE_DELAY = 0.03  # s from first hand contact to closing the fingers
BOWL_MCP = 1.1  # rad, finger knuckle flexion of the bowl


@dataclass
class CatchResult:
    outcome: str
    impact_speed: float = np.nan  # ball speed relative to ball_seat at the first hand contact
    settle_time: float = np.nan  # s from first contact to resting
    n_replans: int = 0
    penetration: float = 0.0  # m, deepest ball-hand contact
    finger_torque: float = 0.0
    arm_force: float = 0.0
    arm_speed: float = 0.0


class HandCatcher:
    def __init__(self, sim: HandSim, fingers: str, obs_noise: float, strategy: str = "cushion",
                 cushion_dir: str = "down", d_pre: float = 0.03, d_cush: float = 0.08, speed_cap: float = 0.7,
                 catch_curl: float = 0.3, close_on: str = "palm", close_delay: float = 0.0, bowl: bool = False,
                 yaw: float = 0.0, obs_bias: float = 0.0):
        self.sim, self.fingers, self.obs_noise = sim, fingers, obs_noise
        self.close_on, self.close_delay = close_on, close_delay
        self.obs_bias = obs_bias  # m, a horizontal offset of all observations (random direction per trial)
        m = sim.model
        # bowl: the task site's z axis is the bowl axis (handsim.set_bowl_sites must have been called)
        self.kin = ArmKinematics(m, "bowl_point" if bowl else "catch_point")
        d0 = mujoco.MjData(m)
        d0.qpos[sim.arm_qadr] = Q_READY
        q_nominal = Q_READY + np.array([0, 0, 0, 0, 0, yaw])  # yaw: which way the fingers point
        # Waiting pose: the task site where it is at Q_READY, the palm (or bowl) opening straight up
        ik = self.kin.solve(self.kin.fk(Q_READY)[0], np.array([0.0, 0.0, 1.0]), q_nominal, q_nominal)
        if not ik.success:
            raise RuntimeError("no waiting pose")
        self.q_wait = ik.q
        mujoco.mj_forward(m, d0)
        # Joint speeds planned up to 0.9 x 180 deg/s (v1's planner used the full limit).
        # The hand gives way under the ball: it starts moving down d_pre before the contact, reaches
        # min(alpha |v_ball|, speed_cap) at the contact and stops d_cush further down.
        self.planner = CatchPlanner(self.kin, PlannerConfig(strategy=strategy, z_min=0.45, z_min_end=0.33,
                                                            v_max=np.full(6, 0.9 * np.pi), cushion_dir=cushion_dir,
                                                            d_pre=d_pre, d_cush=d_cush, cup_speed_max=speed_cap),
                                    q_nominal, d0.body("shoulder_link").xpos.copy())
        self.hands = postures(sim)
        # Catching posture: the cradle with the fingers curled catch_curl (0.8 = the cradle). Raised
        # fingers intercept a ball coming in over the fingertips and deflect it.
        self.hands["catch"] = posture(sim, catch_curl, 1.2)
        if bowl:  # fingers stand up from the knuckles; closing curls them over the ball
            self.hands["catch"] = posture(sim, BOWL_MCP, 1.2, pip=0.3)
            self.hands["grasp"] = posture(sim, BOWL_MCP + 0.3, 1.2, 0.5, pip=1.2)
        self.seat = m.site("ball_seat").id
        self.obs_every = max(1, round(1.0 / (OBS_RATE * m.opt.timestep)))

    def seat_relative(self):
        """Ball position and velocity relative to the ball_seat site (world frame)."""
        m, d = self.sim.model, self.sim.data
        v = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.seat, v, 0)
        p, vb = self.sim.ball_state()
        return p - d.site_xpos[self.seat], vb - v[3:]

    def launch(self, rng):
        sim, d = self.sim, self.sim.data
        sim.reset(self.q_wait, self.hands["catch"])
        p0 = LAUNCHER_POS + rng.uniform(-LAUNCHER_JITTER, LAUNCHER_JITTER)
        target = np.array([rng.uniform(*THROUGH_X), rng.uniform(*THROUGH_Y), rng.uniform(*THROUGH_Z)])
        T = rng.uniform(*FLIGHT_TIME)
        v0 = (target - p0 - 0.5 * sim.model.opt.gravity * T ** 2) / T
        d.qpos[sim.ball_qadr:sim.ball_qadr + 3] = p0
        d.qvel[sim.ball_dadr:sim.ball_dadr + 3] = v0
        mujoco.mj_forward(sim.model, d)

    def run(self, throw_rng, noise_rng, recorder=None) -> CatchResult:
        sim, d, m = self.sim, self.sim.data, self.sim.model
        self.launch(throw_rng)
        phi = noise_rng.uniform(0, 2 * np.pi) if self.obs_bias else 0.0
        bias = self.obs_bias * np.array([np.cos(phi), np.sin(phi), 0.0])
        predictor = BallisticPredictor(m.opt.gravity)
        traj = Trajectory([Hold(0.0, np.inf, self.q_wait)])
        plan, last_fit, step = None, -np.inf, 0
        obs_t, obs_p = [], []
        res = CatchResult("MISSED")
        timeout = MAX_TRIAL_TIME
        t_touch, t_trigger, hold_start = None, None, None
        force_limit = m.actuator_forcerange[sim.arm_act, 1]
        while d.time < timeout:
            t = d.time
            locked = plan is not None and t >= plan.t_match - REPLAN_LOCK
            if t_touch is None and not locked:
                if step % self.obs_every == 0:
                    p = sim.ball_state()[0]
                    obs_t.append(t)
                    obs_p.append(p + bias + (noise_rng.normal(0, self.obs_noise, 3) if self.obs_noise else 0.0))
                if t >= OBS_WINDOW and t - last_fit >= REPLAN_PERIOD:
                    predictor.fit(obs_t, obs_p)
                    last_fit = t
                    if plan is None or np.linalg.norm(predictor.predict(plan.t_catch)[0] - plan.p_catch) > REPLAN_THRESHOLD:
                        new = self.planner.plan(predictor, t, traj.sample(t),
                                                min_wait=None if plan is None else REPLAN_MIN_WAIT)
                        if new is not None:
                            res.n_replans += plan is not None
                            plan, traj = new, new.trajectory
                            timeout = plan.t_catch + TIMEOUT_AFTER_CATCH
            if self.fingers == "close" and t_trigger is not None and t - t_trigger >= self.close_delay:
                sim.set_hand(self.hands["grasp"])
            sim.control_arm(*traj.sample(t))
            sim.step()
            step += 1
            if recorder is not None:
                recorder.capture(d)
            res.finger_torque = max(res.finger_torque, sim.finger_torque_ratio())
            res.arm_force = max(res.arm_force, float(np.max(np.abs(d.actuator_force[sim.arm_act]) / force_limit)))
            for i in range(d.ncon):
                cc = d.contact[i]
                if sim.ball_geom in (cc.geom1, cc.geom2) and ({cc.geom1, cc.geom2} & sim.hand_geoms):
                    res.penetration = max(res.penetration, -float(cc.dist))
            c = sim.ball_contacts()
            if c.ground:
                res.outcome = "DROPPED" if t_touch is not None else "MISSED"
                break
            if c.arm:
                res.outcome = "HIT_ARM"
                break
            if c.hand and t_touch is None:
                t_touch = d.time
                res.impact_speed = float(np.linalg.norm(self.seat_relative()[1]))
            if t_trigger is None and (c.palm if self.close_on == "palm" else c.hand):
                t_trigger = d.time
            if t_touch is not None:
                rel_p, rel_v = self.seat_relative()
                resting = (c.hand and np.linalg.norm(rel_p) < SEAT_TOL and np.linalg.norm(rel_v) < SPEED_THRESHOLD
                           and d.time >= traj.t_end)
                if resting:
                    hold_start = d.time if hold_start is None else hold_start
                    if d.time - hold_start >= HOLD_DURATION:
                        res.outcome = "SUCCESS"
                        res.settle_time = hold_start - t_touch
                        break
                else:
                    hold_start = None
        res.arm_speed = sim.speed_peak
        if plan is None:
            res.outcome = "NO_FEASIBLE_TARGET"
        elif res.outcome == "MISSED" and t_touch is not None:
            res.outcome = "NOT_SETTLED" if sim.ball_contacts().hand else "BOUNCED_OUT"
        return res


def summarize(rows: list[CatchResult]) -> str:
    n = len(rows)
    counts = Counter(r.outcome for r in rows)
    out = f"success {counts['SUCCESS']}/{n} (" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ")"
    ok = [r for r in rows if np.isfinite(r.impact_speed)]
    if ok:
        out += f" | impact {np.mean([r.impact_speed for r in ok]):.2f}/{np.max([r.impact_speed for r in ok]):.2f} m/s"
    s = [r.settle_time for r in rows if np.isfinite(r.settle_time)]
    if s:
        out += f" | settle {np.mean(s):.2f}/{np.max(s):.2f} s"
    out += (f" | penetration {1e3 * max(r.penetration for r in rows):.1f} mm"
            f" | replans {np.mean([r.n_replans for r in rows]):.1f} | finger torque {max(r.finger_torque for r in rows):.2f}"
            f" | arm force {max(r.arm_force for r in rows):.2f} | arm speed {max(r.arm_speed for r in rows):.2f}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="v4 E3a: catching a launched ball with the LEAP hand.")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aim-dx", type=float, default=0.0,
                   help="aim the catch this far (m) from the palm centre towards the finger roots")
    p.add_argument("--fingers", choices=["close", "cradle"], nargs="+", default=["close", "cradle"])
    p.add_argument("--strategy", choices=["cushion", "static"], default="cushion")
    p.add_argument("--obs-noise", type=float, default=0.0)
    p.add_argument("--cushion-dir", choices=["down", "ball"], default="down")
    p.add_argument("--d-pre", type=float, default=0.03, help="m of downward motion before the contact")
    p.add_argument("--d-cush", type=float, default=0.08, help="m of downward motion after the contact")
    p.add_argument("--speed-cap", type=float, default=0.7, help="max hand speed at the contact (m/s)")
    p.add_argument("--catch-curl", type=float, default=0.3, help="finger curl while catching (0.8 = cradle)")
    p.add_argument("--close-on", choices=["palm", "hand"], default="palm", help="contact that triggers closing")
    p.add_argument("--close-delay", type=float, default=0.0, help="s from that contact to closing")
    p.add_argument("--obs-bias", type=float, default=0.0,
                   help="horizontal offset of all ball observations (m, random direction per trial): an aim error")
    p.add_argument("--hand", choices=["bowl", "flat"], default="bowl",
                   help="bowl (E3c, default): fingers up from the knuckles, heel raised by --bowl-tilt, the bowl axis "
                        "faces the ball; flat: the palm-up catch posture of E3a (fingers curled 0.3)")
    p.add_argument("--bowl-tilt", type=float, default=30.0, help="deg")
    p.add_argument("--yaw", type=float, default=0.0, help="hand turned about its axis from the ready pose (deg)")
    p.add_argument("--video", type=Path, default=None, help="record the first trial per finger strategy (mp4)")
    args = p.parse_args()
    args.bowl = args.hand == "bowl"
    sim = HandSim()
    if args.bowl:
        set_bowl_sites(sim.model, np.radians(args.bowl_tilt))
    if args.aim_dx:
        shift_site_in_palm(sim.model, "catch_point", (args.aim_dx, 0.0, 0.0))
    for fingers in args.fingers:
        catcher = HandCatcher(sim, fingers, args.obs_noise, args.strategy, args.cushion_dir, args.d_pre,
                              args.d_cush, args.speed_cap, args.catch_curl, args.close_on, args.close_delay, args.bowl,
                              np.radians(args.yaw), args.obs_bias)
        throw_rng, noise_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(args.seed).spawn(2))
        rows = []
        for k in range(args.trials):
            rec = None
            if args.video is not None and k == 0:
                rec = VideoRecorder(sim.model, args.video / f"hand_catch_{fingers}.mp4", speed=0.5, camera="close")
            rows.append(catcher.run(throw_rng, noise_rng, rec))
            if rec is not None:
                rec.close()
        print(f"fingers {fingers:6s} ({'bowl ' + str(args.bowl_tilt) + ' deg, ' if args.bowl else ''}yaw {args.yaw} deg, "
              f"{args.strategy} {args.cushion_dir} {args.d_pre}/{args.d_cush} m, cap {args.speed_cap} m/s,"
              f" obs noise {1e3 * args.obs_noise:.0f} mm, bias {1e3 * args.obs_bias:.0f} mm): {summarize(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

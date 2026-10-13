"""Stage 1 (v1) of the CS5478 Robot Juggling project: a UR5e catches a thrown ball in a cup.

Pipeline per trial:
    1. prediction.py  -- fit a ballistic model to (noisy) ball observations
    2. planning.py    -- choose a reachable catch point / time on the predicted path
    3. planning.py    -- IK + trajectory (approach, then match + cushion), tracked by
                         the arm's position servos with velocity/acceleration feed-forward
    4. this file      -- run the physics and decide whether the ball was caught

The ball is thrown from a launcher towards a random point in the arm's workspace.
The planner never sees that point; it only gets ball positions sampled at OBS_RATE.
While the ball is in flight the fit is refreshed every REPLAN_PERIOD; if the predicted
catch point moved, target selection and planning are redone from the current desired
arm state, until shortly before the match phase starts.

Usage (from the repo root):
    python v1_single_arm_catch/catch_sim.py                                  # one visualized trial
    python v1_single_arm_catch/catch_sim.py --headless --trials 50 --seed 0  # success rate
    python v1_single_arm_catch/catch_sim.py --strategy static                # baseline: cup waits still
    python v1_single_arm_catch/catch_sim.py --obs-noise 0.005                # 5 mm observation noise

On macOS the passive viewer must be launched with `mjpython` instead of `python`.
"""

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from kinematics import ArmKinematics
from planning import CatchPlan, CatchPlanner, Hold, PlannerConfig, Trajectory
from prediction import BallisticPredictor

DEFAULT_SCENE = Path(__file__).resolve().parent / "scene.xml"

# ---------------------------------------------------------------------------
# Throw generation (hidden from the planner)
# ---------------------------------------------------------------------------
LAUNCHER_POS = np.array([1.8, 0.0, 0.65])  # m
LAUNCHER_JITTER = np.array([0.05, 0.15, 0.05])  # m, uniform +-
THROUGH_X = (0.35, 0.65)  # the ball passes through a random point of this box ...
THROUGH_Y = (-0.30, 0.30)
THROUGH_Z = (0.45, 0.70)
FLIGHT_TIME = (0.8, 1.1)  # ... this many seconds after release

# ---------------------------------------------------------------------------
# Perception / replanning
# ---------------------------------------------------------------------------
OBS_RATE = 100.0  # Hz
OBS_WINDOW = 0.05  # s of observations before the first plan
REPLAN_PERIOD = 0.05  # s between refits
REPLAN_THRESHOLD = 0.005  # m, replan if the predicted catch point moved more than this
REPLAN_LOCK = 0.03  # s, no replanning this close to the start of the match phase
REPLAN_MIN_WAIT = 0.02  # s of rest required at q_pre after a replanned approach

# ---------------------------------------------------------------------------
# Catch-success criteria (as in v0, but relative to the moving cup)
# ---------------------------------------------------------------------------
SPEED_THRESHOLD = 0.05  # m/s, ball speed relative to the cup
HOLD_DURATION = 0.5  # s, must hold continuously after the arm has stopped
POSITION_TOLERANCE = 0.002  # m
TIMEOUT_AFTER_CATCH = 2.0  # s after the planned catch time
MAX_TRIAL_TIME = 5.0  # s, upper bound when no plan was ever found

VIEWER_LINGER = 3.0  # s
MARKER_DT = 0.03  # s between predicted-trajectory markers


@dataclass
class CupInfo:
    """Inner cavity of the (cylindrical) cup in the cup body frame (z = cup axis)."""
    radius: float
    base_top: float
    rim: float


@dataclass
class TrialResult:
    outcome: str  # SUCCESS / NO_FEASIBLE_TARGET / MISSED / BOUNCED_OUT / NOT_SETTLED
    t_catch: float = np.nan
    pred_err: float = np.nan  # |predicted - actual ball position| at t_catch (m)
    track_err: float = np.nan  # |planned - actual cup_mouth position| at t_catch (m)
    impact_speed: float = np.nan  # ball speed relative to the cup when it enters (m/s)
    cup_speed: float = np.nan  # planned cup speed at t_catch (0 = static catch)
    settle_time: float = np.nan  # s from entering the cup until it is at rest relative to the cup
    n_replans: int = 0
    max_force_ratio: float = 0.0  # max |actuator force| / force limit


class CatchSim:
    def __init__(self, scene: Path, cfg: PlannerConfig, obs_noise: float):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.obs_noise = obs_noise
        m = self.model

        self.key_ready = m.key("ready").id
        self.kin = ArmKinematics(m)
        self.q_ready = m.key_qpos[self.key_ready][self.kin.qpos_adr].copy()
        mujoco.mj_resetDataKeyframe(m, self.data, self.key_ready)
        mujoco.mj_forward(m, self.data)
        shoulder = self.data.body("shoulder_link").xpos.copy()
        self.planner = CatchPlanner(self.kin, cfg, self.q_ready, shoulder)

        joint = m.joint("ball_free").id
        self.ball_qadr, self.ball_dadr = m.jnt_qposadr[joint], m.jnt_dofadr[joint]
        self.ball_body = m.body("ball").id
        self.ball_radius = float(m.geom_size[m.geom("ball").id][0])
        self.cup_body = m.body("cup").id
        self.mouth_site = m.site("cup_mouth").id
        self.cup = self._cup_info()

        # Actuator: force = kp * (ctrl - q) - kd * qd  (Menagerie "general" position servo)
        self.kp = m.actuator_gainprm[:, 0].copy()
        self.kd = -m.actuator_biasprm[:, 2].copy()
        self.force_limit = m.actuator_forcerange[:, 1].copy()
        self.dof = self.kin.dof_adr
        self.obs_every = max(1, round(1.0 / (OBS_RATE * m.opt.timestep)))

    def _cup_info(self) -> CupInfo:
        m = self.model
        base = m.geom("cup_base").id
        # Ring of wall boxes cup_wall00, cup_wall01, ...; size[0] is the radial half-thickness.
        walls = [g for g in range(m.ngeom) if m.geom(g).name.startswith("cup_wall")]
        return CupInfo(radius=min(np.linalg.norm(m.geom_pos[g][:2]) - m.geom_size[g][0] for g in walls),
                       base_top=m.geom_pos[base][2] + m.geom_size[base][1],
                       rim=min(m.geom_pos[g][2] + m.geom_size[g][2] for g in walls))

    # ------------------------------------------------------------------ state
    def ball_state(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.qpos[self.ball_qadr:self.ball_qadr + 3].copy()
        vel = self.data.qvel[self.ball_dadr:self.ball_dadr + 3].copy()
        return pos, vel

    def ball_in_cup_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Ball position in the cup frame and ball velocity relative to the cup (world frame)."""
        d = self.data
        r_cup, x_cup = d.xmat[self.cup_body].reshape(3, 3), d.xpos[self.cup_body]
        p_ball = d.xpos[self.ball_body]
        cup_vel, ball_vel = np.zeros(6), np.zeros(6)  # [angular; linear], world frame, at body origin
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, self.cup_body, cup_vel, 0)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, self.ball_body, ball_vel, 0)
        v_cup_at_ball = cup_vel[3:] + np.cross(cup_vel[:3], p_ball - x_cup)
        return r_cup.T @ (p_ball - x_cup), ball_vel[3:] - v_cup_at_ball

    def ball_in_cup(self, p_local: np.ndarray) -> bool:
        # The cup is shallower than the ball (like a cupped hand), so only the ball centre
        # has to be below the rim.
        c, r, tol = self.cup, self.ball_radius, POSITION_TOLERANCE
        return (np.linalg.norm(p_local[:2]) <= c.radius - r + tol
                and c.base_top + r - tol <= p_local[2] <= c.rim)

    # ------------------------------------------------------------------ trial
    def reset(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Arm at "ready", ball launched towards a random point of the workspace."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_ready)
        p0 = LAUNCHER_POS + rng.uniform(-LAUNCHER_JITTER, LAUNCHER_JITTER)
        target = np.array([rng.uniform(*THROUGH_X), rng.uniform(*THROUGH_Y), rng.uniform(*THROUGH_Z)])
        flight = rng.uniform(*FLIGHT_TIME)
        v0 = (target - p0 - 0.5 * self.model.opt.gravity * flight ** 2) / flight
        self.data.qpos[self.ball_qadr:self.ball_qadr + 3] = p0
        self.data.qvel[self.ball_dadr:self.ball_dadr + 3] = v0
        mujoco.mj_forward(self.model, self.data)
        return p0, v0

    def apply_control(self, traj: Trajectory) -> None:
        q_d, qd_d, qdd_d = traj.sample(self.data.time)
        # Inertial feed-forward tau = M(q) qdd_d (gravity is compensated by gravcomp).
        acc = np.zeros(self.model.nv)
        acc[self.dof] = qdd_d
        tau = np.zeros(self.model.nv)
        mujoco.mj_mulM(self.model, self.data, tau, acc)
        # kp (ctrl - q) - kd qd  ==  kp (q_d - q) + kd (qd_d - qd) + tau_ff
        self.data.ctrl[:] = q_d + (self.kd / self.kp) * qd_d + tau[self.dof] / self.kp

    def run_trial(self, throw_rng, noise_rng, viewer=None, trial_idx=0, verbose=True) -> TrialResult:
        """Separate RNG streams keep the throws identical across strategies / noise levels."""
        m, d = self.model, self.data
        p0, v0 = self.reset(throw_rng)
        if verbose:
            print(f"\n[Trial {trial_idx + 1}] throw from ({p0[0]:+.2f}, {p0[1]:+.2f}, {p0[2]:.2f}) m, "
                  f"v0 = ({v0[0]:+.2f}, {v0[1]:+.2f}, {v0[2]:+.2f}) m/s")

        predictor = BallisticPredictor(m.opt.gravity)
        obs_t, obs_p = [], []
        traj = Trajectory([Hold(0.0, np.inf, self.q_ready)])
        plan: CatchPlan | None = None
        last_fit = -np.inf
        res = TrialResult("MISSED")
        timeout = MAX_TRIAL_TIME
        t_enter = None
        hold_start = None
        catch_logged = False

        wall_start = time.perf_counter()
        steps_per_frame = max(1, round(1.0 / (60 * m.opt.timestep)))
        step = 0
        while d.time < timeout:
            if viewer is not None and not viewer.is_running():
                break
            t = d.time
            ball_pos, _ = self.ball_state()

            # --- 1. observe ball, (re)fit, (re)plan --------------------------------
            locked = plan is not None and t >= plan.t_match - REPLAN_LOCK
            if step % self.obs_every == 0 and not locked:
                obs_t.append(t)
                obs_p.append(ball_pos + noise_rng.normal(0.0, self.obs_noise, 3) if self.obs_noise else ball_pos)
            if t >= OBS_WINDOW and not locked and t - last_fit >= REPLAN_PERIOD:
                predictor.fit(obs_t, obs_p)
                last_fit = t
                # Plan if there is no plan yet (keep trying as the estimate improves),
                # replan if the predicted catch point has moved.
                if plan is None or np.linalg.norm(predictor.predict(plan.t_catch)[0] - plan.p_catch) > REPLAN_THRESHOLD:
                    t0 = time.perf_counter()
                    new = self.planner.plan(predictor, t, traj.sample(t),
                                            min_wait=None if plan is None else REPLAN_MIN_WAIT)
                    if new is not None:
                        if plan is not None:
                            res.n_replans += 1
                        elif verbose:
                            pc = new.p_catch
                            print(f"  plan @ t = {t:.3f} s ({1e3 * (time.perf_counter() - t0):.0f} ms, "
                                  f"{new.n_feasible}/{new.n_candidates} feasible): catch at t = {new.t_catch:.3f} s, "
                                  f"p = ({pc[0]:+.3f}, {pc[1]:+.3f}, {pc[2]:.3f}) m, "
                                  f"|v_ball| = {np.linalg.norm(new.v_ball):.2f} m/s, cup speed = {new.cup_speed:.2f} m/s")
                        plan, traj = new, new.trajectory
                        timeout = plan.t_catch + TIMEOUT_AFTER_CATCH
                        self._draw_markers(viewer, predictor, t, plan)

            # --- 3. track the trajectory and step the physics ----------------------
            self.apply_control(traj)
            mujoco.mj_step(m, d)
            step += 1
            res.max_force_ratio = max(res.max_force_ratio,
                                      float(np.max(np.abs(d.actuator_force) / self.force_limit)))

            # --- 4. catch validation ----------------------------------------------
            if plan is not None and not catch_logged and d.time >= plan.t_catch:
                catch_logged = True
                res.t_catch = plan.t_catch
                res.cup_speed = plan.cup_speed
                # The ball moves ~5 mm per step, so compare at matching times: rewind the
                # ball to t_catch, and compare the cup with the plan at the current time.
                # (qpos is already at d.time; site_xpos is still from the previous step.)
                pos, vel = self.ball_state()
                res.pred_err = float(np.linalg.norm(pos - vel * (d.time - plan.t_catch) - plan.p_catch))
                mouth_actual = self.kin.fk(d.qpos[self.kin.qpos_adr])[0]
                mouth_planned = self.kin.fk(traj.sample(d.time)[0])[0]
                res.track_err = float(np.linalg.norm(mouth_actual - mouth_planned))

            p_local, v_rel = self.ball_in_cup_frame()
            inside = self.ball_in_cup(p_local)
            if inside and t_enter is None:
                t_enter = d.time
                res.impact_speed = float(np.linalg.norm(v_rel))
                if verbose:
                    print(f"  ball entered cup at t = {d.time:.3f} s, relative speed {res.impact_speed:.2f} m/s")
            if inside and np.linalg.norm(v_rel) < SPEED_THRESHOLD and d.time >= traj.t_end:
                hold_start = d.time if hold_start is None else hold_start
                if d.time - hold_start >= HOLD_DURATION:
                    res.outcome = "SUCCESS"
                    res.settle_time = hold_start - t_enter
                    break
            else:
                hold_start = None
            if not inside and d.xpos[self.ball_body][2] < self.ball_radius + 0.005:
                break  # ball on the ground

            if viewer is not None and step % steps_per_frame == 0:
                lag = d.time - (time.perf_counter() - wall_start)
                if lag > 0:
                    time.sleep(lag)
                viewer.sync()

        if plan is None:
            res.outcome = "NO_FEASIBLE_TARGET"
        elif res.outcome != "SUCCESS":
            inside = self.ball_in_cup(self.ball_in_cup_frame()[0])
            res.outcome = "NOT_SETTLED" if inside else ("BOUNCED_OUT" if t_enter is not None else "MISSED")

        if verbose:
            print(f"  {res.outcome}  (t = {d.time:.3f} s, prediction err {1e3 * res.pred_err:.1f} mm, "
                  f"tracking err {1e3 * res.track_err:.1f} mm, replans {res.n_replans}, "
                  f"max force {100 * res.max_force_ratio:.0f}% of limit)")

        if viewer is not None and viewer.is_running():
            viewer.sync()
            linger_end = time.perf_counter() + VIEWER_LINGER
            while viewer.is_running() and time.perf_counter() < linger_end:
                time.sleep(1 / 60)
                viewer.sync()
        return res

    # ------------------------------------------------------------------ viewer
    def _draw_markers(self, viewer, predictor, t_now, plan) -> None:
        """Predicted ball path (small yellow spheres) and the catch point (green)."""
        if viewer is None:
            return
        ts = np.arange(t_now, plan.t_catch + 0.1, MARKER_DT)
        points = [(p, 0.006, (1, 0.8, 0, 0.6)) for p in predictor.predict(ts)[0]]
        points.append((plan.p_catch, 0.015, (0, 1, 0, 0.6)))
        with viewer.lock():
            scn = viewer.user_scn
            scn.ngeom = 0
            for pos, size, rgba in points[:scn.maxgeom]:
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([size, 0, 0]), np.asarray(pos, dtype=float),
                                    np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
                scn.ngeom += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UR5e single-arm ball catching (v1, Stage 1).")
    parser.add_argument("--headless", action="store_true", help="run without the 3D viewer")
    parser.add_argument("--trials", type=int, default=1, help="number of trials to run")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="MJCF scene file")
    parser.add_argument("--strategy", choices=("cushion", "static"), default="cushion",
                        help="cushion: cup moves with the ball and decelerates; static: cup waits still")
    parser.add_argument("--obs-noise", type=float, default=0.0,
                        help="std. dev. of Gaussian noise on observed ball positions (m)")
    return parser.parse_args()


def summarize(results: list[TrialResult]) -> None:
    n = len(results)
    counts = Counter(r.outcome for r in results)
    print(f"\nSuccess rate: {counts['SUCCESS']}/{n} ({100.0 * counts['SUCCESS'] / n:.1f}%)")
    print("Outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    def stat(name, values, scale, unit):
        values = np.array([v for v in values if np.isfinite(v)])
        if len(values):
            print(f"{name}: mean {scale * values.mean():.2f} {unit}, max {scale * values.max():.2f} {unit}")
    stat("Prediction error at catch", [r.pred_err for r in results], 1e3, "mm")
    stat("Tracking error at catch  ", [r.track_err for r in results], 1e3, "mm")
    stat("Relative impact speed    ", [r.impact_speed for r in results], 1, "m/s")
    stat("Settle time in cup       ", [r.settle_time for r in results], 1, "s")
    n_static = sum(r.cup_speed == 0 for r in results)
    if n_static:
        print(f"Static catches (strategy static, or fallback when no cushioned catch is feasible): {n_static}")
    stat("Max actuator force       ", [100 * r.max_force_ratio for r in results], 1, "% of limit")


def main() -> int:
    args = parse_args()
    throw_rng, noise_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(args.seed).spawn(2))
    sim = CatchSim(args.scene, PlannerConfig(strategy=args.strategy), args.obs_noise)

    results = []
    if args.headless:
        for i in range(args.trials):
            results.append(sim.run_trial(throw_rng, noise_rng, trial_idx=i))
    else:
        try:
            viewer_ctx = mujoco.viewer.launch_passive(sim.model, sim.data)
        except RuntimeError as exc:
            if sys.platform == "darwin":
                print("On macOS the viewer must be started with `mjpython catch_sim.py`.")
            raise SystemExit(f"Could not launch viewer: {exc}")
        with viewer_ctx as viewer:
            for i in range(args.trials):
                if not viewer.is_running():
                    print("Viewer closed; stopping.")
                    break
                results.append(sim.run_trial(throw_rng, noise_rng, viewer=viewer, trial_idx=i))

    if results:
        summarize(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

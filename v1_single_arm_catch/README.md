# v1 — Stage 1 Single-Arm Catching: UR5e Dynamic Cup Catching

Stage 1 of the JuggleBot project: a single UR5e arm fitted with a cup end effector moves actively to catch a tossed ball in flight.
No reinforcement learning; this is a classical "prediction → target selection → IK/planning → execution verification" pipeline. `v0_physics_validation/` remains unchanged as the physics parameter baseline.

## Files

| File | Contents |
|---|---|
| `scene.xml` | World: floor, lights, camera, visual-only launcher marker, ball, `ready` keyframe. Includes `ur5e_cup.xml` |
| `ur5e_cup.xml` | Menagerie UR5e + cup end effector (changes noted in file header comments) |
| `../assets/universal_robots_ur5e/` | (Shared across stages at repository root) Original Menagerie meshes, `LICENSE` (BSD-3), and unmodified `ur5e.xml` for reference |
| `prediction.py` | 1. Ballistic prediction: least-squares fit of observed positions under known gravity |
| `kinematics.py` | 5-DoF DLS IK for cup rim position + cup axis, FK, collision checking (on a separate scratch `MjData`) |
| `planning.py` | 2. Catch-point selection + 3. trajectory generation (quintic / Cartesian linear segments) |
| `catch_sim.py` | Toss generation, observation, replanning, tracking control, 4. catch classification, CLI, statistics |

Model source: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `universal_robots_ur5e`, commit `0059d43`.

## Running (from the repository root)

```bash
uv run python v1_single_arm_catch/catch_sim.py                                     # Visualize 1 trial (yellow=predicted path, green=catch point)
uv run python v1_single_arm_catch/catch_sim.py --trials 5 --seed 0                 # Visualize multiple trials
uv run python v1_single_arm_catch/catch_sim.py --headless --trials 50 --seed 0     # Success-rate statistics
uv run python v1_single_arm_catch/catch_sim.py --headless --trials 50 --seed 0 --strategy static     # Baseline
uv run python v1_single_arm_catch/catch_sim.py --headless --trials 50 --seed 0 --obs-noise 0.005     # 5 mm observation noise
```

On macOS, use `mjpython` instead of `python` for visualization. To view/edit the scene: `cd v1_single_arm_catch && uv run python -m mujoco.viewer --mjcf=scene.xml`.

## Pipeline

**0. Ball Toss (hidden from planner)**: The launcher is randomly perturbed near (1.8, 0, 0.65) m; a random pass-through point is sampled in x∈[0.35, 0.65], y∈[−0.3, 0.3], z∈[0.45, 0.70] m with flight time T∈[0.8, 1.1] s, and initial velocity is solved as `v0 = (p_target − p0 − ½gT²)/T`. Ball speed at catch is ≈ 4.5–6 m/s, descending nearly vertically (≈ 15–25° from vertical).

**1. Trajectory Prediction** (`prediction.py`): Without air drag in MuJoCo, free flight follows `p(t) = p0 + v0·t + ½g·t²`. With known gravity, `p0` and `v0` enter linearly and are fit per axis by least squares. Ball center position is sampled at 100 Hz (optional Gaussian noise `--obs-noise`), observing for 50 ms before the first fit and refitting every 50 ms using all observations.

**2. Catch-Point Selection** (`planning.py: CatchPlanner.plan`): Candidate catch times `t_c` are sampled every 10 ms along the predicted trajectory and filtered by:
- descending branch with cup rim height z∈[0.40, 0.85] m (lower heights risk wrist-floor contact since the wrist hangs ≈ 0.2 m below the cup);
- cup axis aligned with −v̂ (opposite incoming ball velocity), tilted ≤ 40° from vertical;
- valid IK solution (position error < 10 µm), within joint limits, free of self-collision and floor collision;
- timing feasibility: shortest quintic from the current desired state respecting joint velocity (π rad/s) and acceleration (15 / 25 rad/s²) limits, leaving ≥ 0.1 s stationary wait at the pre-catch pose (≥ 0.02 s when replanning).

Among feasible candidates, the planner selects minimum weighted joint displacement (tilt as secondary cost). Under `cushion`, if no cushioned catch is feasible, it falls back to `static`.

**3. IK + Motion Planning**:
- IK (`kinematics.py`): 3-DoF cup rim position + 2-DoF cup axis alignment (spin about the cup axis is free) solved by damped least squares; the remaining 1-DoF redundancy is pulled toward `ready` via exact null-space projection.
- Three-segment trajectory:
  1. **Approach**: joint-space quintic from the current desired state (continuous position/velocity/acceleration, no jump on replan) to `q_pre`, followed by a stationary wait;
  2. **Match**: Cartesian line accelerating the cup from rest along the incoming direction to speed u = min(0.4·|v_ball|, 0.7 m/s) at `t_c` with the rim coincident with the ball center;
  3. **Cushion**: 10 cm deceleration to rest along the incoming direction.
  Segments 2 and 3 are discretized at 1 ms via per-step IK. `--strategy static` skips segments 2 and 3 and waits motionless at the catch point.
- Control: Menagerie position servo `τ = kp(ctrl − q) − kd·q̇` with `ctrl = q_d + (kd/kp)·q̇_d + τ_ff/kp` and `τ_ff = M(q)·q̈_d`, equivalent to PD tracking with velocity and inertial feedforward; gravity is compensated by `gravcomp`.
- Replanning: triggered when the updated fit moves the predicted catch point at `t_c` by > 5 mm; locked 30 ms before the Match segment begins.

**4. Catch Classification**: Evaluated in the cup frame: radially the entire ball must lie inside the inner wall, axially the ball center must not exceed the rim (cup depth 0.06 m < ball diameter, similar to a palm supporting a ball; 2 mm tolerance), relative speed < 0.05 m/s, and sustained ≥ 0.5 s after the arm stops → `SUCCESS`. Other outcomes: `NO_FEASIBLE_TARGET`, `MISSED` (never entered cup), `BOUNCED_OUT` (entered then exited), `NOT_SETTLED` (timed out before settling).

## Parameters

**Scene**

| Parameter | Value |
|---|---|
| Timestep / integrator / contact | 1 ms / implicitfast / same as v0 (`solref="0.003 0.5"`, `solimp="0.95 0.99"`) |
| Ball | Same as v0: radius 0.035 m, mass 0.057 kg, `condim=6` |
| Cup | Hollow cylinder sized to approximate a cupped dexterous hand: inner diameter 0.10 m, inner depth 0.06 m, wall thickness 5 mm, base thickness 1 cm, rim 0.07 m from flange, mass 0.2 kg, mounted at `attachment_site`. Wall built from 24 box segments |
| UR5e actuators | Menagerie defaults: kp = 2000 / 500, kd = 400 / 100, torque limits 150 / 28 N·m (shoulder-elbow / wrist) |
| `ready` pose | Cup rim at (0.45, 0, 0.50) m, facing upward |

**Planning / Trial** (`planning.py: PlannerConfig`, top-level constants in `catch_sim.py`)

| Parameter | Value |
|---|---|
| Observation | 100 Hz, 50 ms initial window before first plan, refit every 50 ms |
| Candidate catch points | 10 ms step, z∈[0.40, 0.85] m, tilt ≤ 40° |
| Joint limits (for timing) | Speed π rad/s; acceleration 15 (major) / 25 (wrist) rad/s² |
| Cushioning | α = 0.4, cup speed cap 0.7 m/s, Match 5 cm, Cushion 10 cm |
| Success criterion | Relative speed < 0.05 m/s for 0.5 s; timeout = t_c + 2 s |

Reason for the 0.7 m/s cup speed cap: in these poses, 1 m/s of cup velocity along the incoming trajectory requires 2–4 rad/s of joint speed; higher cup speeds exceed the UR5e π rad/s joint-speed limit.

## Reference Results

Environment: Windows 11, Python 3.12.5, MuJoCo 3.15.0, NumPy 2.5.3. All rows run `--headless --trials 50 --seed 0` with separate RNG streams for tosses and observation noise, so **every row evaluates the exact same 50 tosses**.

| Configuration | Success Rate | Rim Tracking Error @t_c (mean) | Relative Entry Speed (mean / max) | Time to Settle (mean / max) | Fallback to static |
|---|---|---|---|---|---|
| cushion | 50/50 | 0.22 mm | 4.43 / 5.11 m/s | 0.32 / 0.45 s | 0 |
| static | 50/50 | 0.03 mm | 5.13 / 5.81 m/s | 0.39 / 0.46 s | — |
| cushion, 5 mm noise | 50/50 | 0.19 mm | 4.52 / 5.73 m/s | 0.32 / 0.46 s | 7 |
| static, 5 mm noise | 50/50 | 0.03 mm | 5.13 / 5.82 m/s | 0.40 / 0.59 s | — |
| cushion, 10 mm noise | 50/50 | 0.15 mm | 4.69 / 5.51 m/s | 0.34 / 0.52 s | 21 |
| static, 10 mm noise | 50/50 | 0.04 mm | 5.14 / 5.83 m/s | 0.40 / 0.66 s | — |

- Without noise, prediction error at `t_c` is 0.00 mm (analytic parabola matches MuJoCo integration). Under 5 / 10 mm noise, mean error is ≈ 3–4 / 5.5 mm, max ≈ 9 / 13 mm.
- Peak actuator torque is ≈ 15–25% of limits (no saturation).
- Single planning call (selection + IK + discretization) takes ≈ 50 ms.

Additional checks (temporary scripts, not archived):
- IK: 200 random target positions and incoming directions in the catch region solved 200/200 collision-free with max position error 10 µm, ≈ 0.6 ms/solve.
- Quintic boundary conditions: random initial/final (position, velocity, acceleration) satisfied exactly.
- Offscreen rendering confirmed the cup tilts into the incoming ball and retains it.

**Conclusions and limitations**: Under these settings both `static` and `cushion` achieve 100% catches even after shrinking the cup to palm size (0.10 m inner diameter, 0.06 m depth, 0.07 m ball diameter), showing that the v0 contact parameters are heavily damped and nearly eliminate bounce. Cushioning reduces impact speed by ≈ 14% and settling time by ≈ 18%, but joint-speed limits cap cup speed at ≈ 15% of ball speed. Under noise, late replans often lack time for a full cushion stroke and fall back to `static`. Lower contact damping or faster tosses would be needed to separate the two strategies.

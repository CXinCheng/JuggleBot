# v2 — Stage 2: Single-Arm Cup Throwing and Two-Point Self-Throw/Catch

A single UR5e fitted with the v1 cup throws a ball and catches it repeatedly. Two experiments:

- **Experiment A (`throw_test.py`)**: Throwing characterization — maximum height/distance and release error under the 180°/s joint-speed limit.
- **Experiment B (`juggle_loop.py`)**: Single-arm, single-ball alternating self-throw and catch between points A and B, measuring consecutive catches. Each cycle (throw → move to catch within flight time F → catch → carry to next throw start → throw) mirrors what each arm performs in dual-arm S1 exchange.

Classical control only (no RL). `v1` remains unchanged; UR5e meshes live in the shared root `assets/` directory.

## Files

| File | Contents |
|---|---|
| `scene.xml` | Scene: floor, cameras (`side`, and `loop` for two-point cycling), ball (seated at cup base), `ready` keyframe |
| `ur5e_cup.xml` | Same UR5e + cup as v1, adding site `ball_seat` (ball center when seated) |
| `kinematics.py` | v1 IK / FK / collision checks; FK and Jacobian accept a target site |
| `prediction.py` | v1 ballistic least-squares fit (unchanged) |
| `planning.py` | v1 catch selection + trajectory with optional torque check `torque_ok` |
| `throwing.py` | Three throw motions, release prediction, release-point search, full-segment verification |
| `simcore.py` | Simulation wrapper: inverse-dynamics feedforward tracking, **actual joint-speed** 180°/s check, ball contact classification, single-throw replay |
| `calibration.py` | Iterative simulation-based throw command calibration |
| `throw_test.py` | Experiment A |
| `juggle_loop.py` | Experiment B |
| `video.py` | Offscreen MP4 recording; passive viewer pacing |
| `results/` | Raw outputs (`*.txt`) and per-trial data (`throw_test.csv`) backing this README |

## Running (from the repository root)

```bash
# Experiment A
uv run python v2_single_arm_throw_catch/throw_test.py --trials 10 --calibrate          # Full sweep (~5 min)
uv run python v2_single_arm_throw_catch/throw_test.py --mode linewhip --F 0.5 --d 0.4 --trials 3 --calibrate --viewer
uv run python v2_single_arm_throw_catch/throw_test.py --mode whip linewhip --F 0.5 --d 0.4 --video videos/

# Experiment B
uv run python v2_single_arm_throw_catch/juggle_loop.py                                 # 10 runs x up to 20 catches
uv run python v2_single_arm_throw_catch/juggle_loop.py --obs-noise 0.005               # 5 mm observation noise
uv run python v2_single_arm_throw_catch/juggle_loop.py --runs 1 --viewer               # Live viewer (default 0.5x speed)
uv run python v2_single_arm_throw_catch/juggle_loop.py --runs 1 --cycles 6 --video videos/loop.mp4
```

`--viewer` opens the MuJoCo window and `--speed` sets playback speed; on macOS use `mjpython` instead of `python`. Startup spends a few seconds on planning and calibration.

## Throwing Principle

The cup has no active release mechanism: the ball rests on the base, and **once cup deceleration along the cup axis exceeds gravity, support is lost and the ball launches**. Release velocity equals the velocity of `ball_seat` at that instant. Using specific force `f = a_seat − g`, the ball remains pressed against the base while `f · cup_axis > 0` and separates at the first instant `f · cup_axis ≤ 0`.

All three throw motions share the same smoothstep speed profile (`t_acc = 0.15 s`, `t_dec = 0.12 s`) and differ in path geometry:

| Motion | Path | Characteristics |
|---|---|---|
| `translate` | Linear `ball_seat`, fixed cup orientation | Cleanest release, but `wrist_1` must cancel forearm rotation; upper arm and forearm are capped at π rad/s, giving a theoretical cap of π(0.425 + 0.392) ≈ 2.57 m/s only at full horizontal extension |
| `whip` | Joint-space line (`shoulder_pan` + three parallel pitch joints) | Pitch joint speeds add along the chain (fast), but `ball_seat` follows an arc and centripetal load presses the ball against the wall |
| `linewhip` | Linear `ball_seat`, rotating cup (same 4 joints as `whip`) | Retains `whip` speed without centripetal wall loading |

Release-point search (`ThrowPlanner.plan`): solves IK on a workspace grid (aligning cup axis with peak acceleration specific force to center the ball), ranks candidates by required joint speed, and checks full trajectories for:
- joint speed ≤ 0.9 × 180°/s (planning margin);
- inverse-dynamics feedforward torque ≤ 0.8 × actuator limit (computed on a **contact-disabled** copy so collision forces are not mistaken for joint torque);
- joint limits, servo `ctrlrange`, self/floor collisions;
- continuous base support during acceleration (no early separation) and rim clearance after release.

Experiment B also uses: restricted release region (`region`), physics-replay verification for candidates rejected only by analytic release checks (`verifier`), ranking by post-throw time to reach the catch (`rank`), and trying the next candidate if calibration fails (`finalize`).

**180°/s hard constraint**: MuJoCo enforces no joint-speed limit natively. Planning keeps a 10% margin, and every simulation step checks **actual** joint speeds against π rad/s (`simcore.JointSpeedGuard`).

**Tracking control**: Uses Menagerie's position servo `force = kp(ctrl − q) − kd·q̇` with `ctrl = q_d + (kd/kp)·q̇_d + τ_ff/kp` and `τ_ff = M(q_d)·q̈_d + c(q_d, q̇_d)` from inverse dynamics (adding Coriolis/centrifugal terms beyond v1's `M·q̈`).

**Simulation calibration (`calibration.py`)**: Replays a centered ball to measure actual release `(t, p, v)`, computes the velocity required to hit the target at the target time from that release state, updates the commanded velocity, and rebuilds the trajectory until error < 2 mm (or < 10 mm if further updates violate constraints). **This treats the simulator itself as a perfect model**—a strong classical baseline that must be disclosed honestly and compared against future learned corrections.

## Experiment A: Throwing Characterization (`throw_test.py`, raw output `results/throw_test.txt`)

Target: land at the same height, `d` meters along `+y`, `F` seconds after release (`v = (d/F)·ŷ + (gF/2)·ẑ`). 10 trials per condition with shared random initial offsets ≤ 10 mm at the cup base. "miss" = ball-center distance from target at target time. `raw` = analytic release prediction; `cal` = after simulation calibration. Values are "mean / max".

**Feasible range** (planning joint speed ≤ 0.9 × 180°/s, feedforward torque ≤ 0.8 × limit):

| | d = 0 | d = 0.2 m | d = 0.4 m |
|---|---|---|---|
| translate | F = 0.35 s only | All infeasible | All infeasible |
| whip | F = 0.35–0.45 s | F = 0.40–0.50 s | F = 0.45–0.50 s |
| linewhip | F = 0.35–0.45 s | F = 0.40–0.50 s | F = 0.45–0.50 s |

**Release accuracy** (representative cases; full table in `results/throw_test.txt`):

| Case | Variant | Release Time Error | Release Speed Error | Miss | Wall Contact at Release |
|---|---|---|---|---|---|
| translate d=0, F=0.35 | raw | 2.9 / 5.0 ms | 1.4 / 1.6 % | 14 / 18 mm | 0 % |
| whip d=0.4, F=0.5 | raw | 37 / 38 ms (late) | 8.9 / 9.3 % | 99 / 104 mm | 100 % |
| whip d=0.4, F=0.5 | cal | Calibration failed | | | |
| linewhip d=0.4, F=0.5 | raw | 8.8 / 12 ms | 3.2 / 3.3 % | 34 / 36 mm | 100 % |
| linewhip d=0.4, F=0.5 | cal (4 iters) | 1.6 / 3.0 ms | 0.4 / 1.0 % | 5.0 / 12.8 mm | 100 % |
| linewhip d=0.2, F=0.45 | cal (2 iters) | 0.8 / 2.0 ms | 0.3 / 0.7 % | 3.2 / 6.0 mm | 100 % |
| linewhip d=0, F=0.40 | cal (2 iters) | 0.0 / 0.0 ms | 0.2 / 0.3 % | 1.2 / 2.3 mm | 100 % |

- All 7 feasible `linewhip` conditions calibrated cleanly: mean miss 1–7 mm, max 13 mm; joint speed 0.47–0.87, feedforward torque 0.46–0.76 (torque becomes the primary constraint; actual actuator force peaks at 0.85 × limit).
- Of 8 feasible `whip` conditions, 7 failed calibration and 1 barely passed (miss 14 / 26 mm): the ball slides along the wall, releasing 30–55 ms late and 7–11% slow with lateral drift.
- `translate` is already accurate (1.4%); calibration disables the analytic rim check and introduces 2/10 `RECONTACT` failures, so it should not be calibrated.
- `linewhip` d=0, F=0.35 had 2/10 `RECONTACT` events (ball brushed the rim after separation).

## Experiment B: Single-Arm Two-Point Self-Throw/Catch (`juggle_loop.py`, raw output `results/juggle_loop_noise*.txt`)

**Setup**: A = (0.6, −0.075, 0.55) m, B = (0.6, 0.075, 0.55) m (ball center, 0.15 m apart), F = 0.4 s. Both directions are planned and calibrated offline with release points within ±(8, 8, 15) cm of the throw point, ranked by estimated travel time to catch (≈ 0.18 s). Up to 20 catches per run, 10 runs (distinct initial ball offsets), 100 Hz observation.

**Each cycle**:
1. Pre-plan the catch from the calibrated nominal release state (v1 catch planner; approach starts when the throw finishes).
2. Observe post-release, refit every 50 ms, replan if predicted catch shifts > 5 mm (same as v1).
3. Catch criterion: ball inside cavity, relative speed < 0.05 m/s for 0.1 s after catch motion completes.
4. Carry ball to the next throw start (joint speed ≤ 0.8π, acceleration 12 / 20 rad/s²), settle for 0.2 s, and throw.

Catch-planner changes relative to v1: major-joint acceleration 15 → 40 rad/s², wrist 25 → 60 rad/s² (checking feedforward torque ≤ 0.9 × limit on every approach), `min_wait` 0.10 → 0.05 s, `earliest_catch` 0.15 → 0.10 s.

| Observation Noise | Runs Reaching 20 Catches | Consecutive Catches (median / min) | Failures | Remaining Wait Before Catch (mean / min) | Replans per Catch |
|---|---|---|---|---|---|
| 0 | 10 / 10 | 20 / 20 | — | 80 / 23 ms | 1.6 |
| 5 mm | 10 / 10 | 20 / 20 | — | 50 / 20 ms | 4.0 |
| 10 mm | 5 / 10 | 15 / 0 | 2 drops, 3 arm hits | 41 / 20 ms | 3.9 |

Common metrics across noise levels: mean release-time error ≈ 1.8 ms (max 4 ms), mean release-speed error 11–15 mm/s (max 27 mm/s), release-to-catch ≈ 0.39 s, relative entry speed ≈ 1.7 m/s (max 2.3), dwell (entry → next release) ≈ 1.16 s (cycle ≈ 1.55 s). Peak actual joint speed 0.87 × 180°/s (1.00 under 10 mm noise; see limitations), peak actuator force 0.59–0.70 × limit.

## Findings and Limitations

1. **`translate` cannot throw laterally**, matching its theoretical cap; **`whip`** slides along the wall and fails calibration; **`linewhip`** balances speed and release accuracy and is carried forward.
2. **Under 180°/s, one arm's release and catch points can be at most ≈ 0.15 m apart (F = 0.4 s)**. At d = 0.25 m the arm cannot arrive in time. Release selection must account for the whole cycle rather than throw speed alone.
3. **Speed and torque approach limits**: throw joint speed 0.84–0.86, catch approach acceleration 40 rad/s²; heavier dexterous hands will leave less margin.
4. **Simulation calibration assumes a perfect model**: suitable for fixed-cycle offline planning, not robustness to unknown objects or contacts.
5. **Wall contact at release occurs in 80–100% of `linewhip` throws**: absorbed by calibration rather than eliminated structurally.
6. **Catch planner uses π without the 0.9 margin**: under 10 mm noise, frequent replans push rounded peak joint speed to 1.00 × 180°/s with zero margin left.
7. **Dwell time ≈ 1.16 s**: faster carrying destabilizes the ball before release (release error jumps to ≈ 250 mm/s).
8. **Menagerie's position servo is heavily overdamped** (`kd/kp = 0.2 s`), so tracking relies primarily on feedforward.
9. Under 10 mm noise, success drops to 50% due to late replans (≈ 20 ms remaining wait); earlier locking, filtering, or prior-weighted fitting are candidate improvements.

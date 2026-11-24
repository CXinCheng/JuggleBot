# v8 Experiment and Interface Reference

This page documents experiment entry points, interfaces, and archived results. For first-time setup and runs, see the [v8 README](README.md) and [Quickstart](QUICKSTART_RL.md). Results in each section apply to the task, controller version, and evaluation configuration stated there.

`v8_robust_control` provides ball observations, state estimation, perception disturbances, residual training, and complete-task evaluation for three-ball juggling on the G1 upper body with dual LEAP Hands. The physics model reuses [v6](../v6_humanoid_hands/README.md), and the three-ball opening and ending of the complete task reuse [v7](../v7_three_ball_hold/README.md). Throws and catches are executed through physical hand-ball contact.

For the configurations and results cited in the report, see the [evidence index](../report/evidence/README.md). Some historical diagnostics rely on locally generated intermediate data or the original handover archive.

## Running

From the repository root:

```bash
uv sync
uv run python -m v8_robust_control.juggle --throws 1 --observation-mode truth
uv run python -m v8_robust_control.juggle --throws 1 --observation-mode sampled
```

| Flag | Default | Description |
| --- | --- | --- |
| `--throws` | 1 | Target number of completed flights |
| `--timeout` | auto | Simulation time limit (seconds) |
| `--observation-mode` | `truth` | `truth` reads exact position and velocity; `sampled` estimates state from position samples |
| `--observation-period` | 1/60 s | Position measurement period in sampled mode |
| `--observation-max-age` | 0.05 s | Measurement validity window from sample time; must be at least the sampling period |
| `--noise-mm` | 0 | Per-axis Gaussian position noise standard deviation (mm) |
| `--delay-ms` | 0 | Fixed measurement transmission delay (ms) |
| `--dropout` | 0 | Whole-frame drop probability in `[0, 1]` |
| `--seed` | 0 | Random seed for perception disturbances |
| `--output` | none | Output path for experiment JSON; must be a new file |
| `--video` | none | Output path for real-time MP4; must be a new file |

`juggle` starts with two balls in hand and one ball in the air, excluding the three-ball-in-hand opening and ending. One completed flight starts when a ball leaves the hand and ends upon a confirmed stable catch. Catching the initial airborne ball is not counted, so total releases can exceed the target flight count. Task failure returns a non-zero exit code.

### Complete Three-Ball Task

`complete_task` starts with two balls in the right hand and one in the left, executes the requested number of throws and catches, and ends with two balls in the right hand and one in the left. `--throws` must be an even integer >= 4 (default 30); `--timeout` defaults to 40 s; `--output` is required and must be a new file.

For commands that generate the reference results, see [Complete-Task Reference Results](#complete-task-reference-results).

`complete_task` supports the same `--observation-mode`, perception disturbance, seed, and video flags; the sampling period is fixed at 1/60 s and the max age is 0.05 s. Disturbances apply only in `sampled` mode. `--seed-plans` specifies initial planning seeds for the complete task (default: [complete_seed_plans.json](complete_seed_plans.json)).

## Observation Interface

[observation.py](../src/jugglebot/robustness/observation.py) defines the data structures for ball observations and state estimates. Position, velocity, and acceleration are in world coordinates (`m`, `m/s`, `m/s²`); timestamps use simulation time in seconds.

| Field | Description |
| --- | --- |
| `ball_id` | Ball model ID |
| `sample_time_s` | Sampling time of the latest measurement |
| `received_time_s` | Time when the measurement becomes available to the controller |
| `state_time_s` | Control time corresponding to the state estimate |
| `position_m` | Measured position or estimated position at control time |
| `velocity_m_s` | Measured or estimated velocity; `None` if unavailable |
| `acceleration_m_s2` | Estimated acceleration; `None` if unavailable |
| `valid` | Whether the measurement or state estimate is valid |

In sampled mode, `juggle` and undisturbed `complete_task` runs estimate velocity and acceleration from three consecutive position measurements and predict ball state at control time, whereas disturbed `complete_task` runs switch to the ballistic flight filter in the final v8 configuration. Position measurements pass sequentially through noise, frame dropout, and a delay queue before reaching the estimator. Reception planning uses valid, unexpired estimates.

Noise is sampled independently per ball and axis; frame dropout drops all balls in a single sample. Fixed delay is implemented with a FIFO queue, where `received_time_s` is the physics step at which the measurement is delivered. Fixing the seed replays the noise and dropout sequence.

Three-point finite differences amplify position noise; disturbance results under this estimator reflect sensitivity of the full observation-estimation-planning pipeline and cannot be attributed solely to the controller. The event-level interface defaults to three-point differences and also supports the filtered mode described below; the motor-residual environments use [SupportPositionTracker](transfer_observation.py) with a gravity prior and contact handling. Results from different entry points should be compared separately.

## Control and Scope

[ObservedCascade](../src/jugglebot/robustness/controller.py) predicts catch locations from ball state and plans throw/catch motions:

| Stage | Behavior |
| --- | --- |
| Throw | Reuses previously executed motion commands and adjusts finger release lead within bounded limits |
| Catch | Re-predicts on planning failure and adjusts the catch trajectory and wrist recovery posture |
| Empty-hand recovery | Starts palm recovery early for the next catch when a catch is delayed |

Before execution, motions are checked for catch stability, readiness for the next throw, collisions, and joint posture. Both observation modes share the same control logic.

The classical controller uses sampled estimates for online catch planning and complete-task ending predictions; holding control, contact detection, safety checks, and planning rollouts use simulation ground truth. For position estimation and policy inputs in residual environments, see [Residual Control](#residual-control).

[CompleteCascade](complete_task.py) reuses the three-ball opening and ending control and integrates observation handling with complete-task acceptance checks.

## Verification

```bash
uv run python -m unittest discover -s v8_robust_control -t . -v
uv run python -m unittest v6_humanoid_hands.test_evaluation -v
```

Tests cover timestamps, state estimation, invalid-data handling, contact authority transitions, planning rollout isolation, motion planning, and experiment record export.

Run zero-disturbance baselines:

<details>
<summary>Generate zero-disturbance baselines</summary>

```bash
uv run python -m v8_robust_control.verify_baselines --observation-mode truth --output-dir v8_robust_control/results
uv run python -m v8_robust_control.verify_baselines --observation-mode sampled --output-dir v8_robust_control/results
```

</details>

Each mode runs continuously to 50 completed flights and saves checkpoints at 30 and 50 flights. On acceptance failure, it saves the failure record and exits non-zero. Before re-running, use a different output directory or move existing files with the same names.

### Zero-Disturbance Reference Results

Default physics parameters with no observation noise, transmission delay, or frame dropout; sampling period 1/60 s, measurement max age 0.05 s.

| Mode and record | Completed flights | Total releases | Drops | Ball-ball collisions | Sim duration |
| --- | --- | --- | --- | --- | --- |
| [truth, 30 flights](results/truth_baseline30.json) | 30/30 | 32 | 0 | 0 | 10.210 s |
| [sampled, 30 flights](results/sampled_baseline30.json) | 30/30 | 32 | 0 | 0 | 10.236 s |
| [truth, 50 flights](results/truth_baseline50.json) | 50/50 | 52 | 0 | 0 | 16.620 s |
| [sampled, 50 flights](results/sampled_baseline50.json) | 50/50 | 52 | 0 | 0 | 16.671 s |

Each result file records acceptance metrics, run configuration, and SHA-256 checksums of source files and planning seeds.

### Complete-Task Reference Results

<details>
<summary>Generate complete-task reference results</summary>

```bash
uv run python -m v8_robust_control.complete_task --throws 30 --observation-mode truth --output v8_robust_control/results/complete_truth30.json
uv run python -m v8_robust_control.complete_task --throws 30 --observation-mode sampled --output v8_robust_control/results/complete_sampled30.json
uv run python -m v8_robust_control.complete_task --throws 50 --observation-mode truth --output v8_robust_control/results/complete_truth50.json
uv run python -m v8_robust_control.complete_task --throws 50 --observation-mode sampled --output v8_robust_control/results/complete_sampled50.json
```

</details>

Each zero-disturbance run executes independently and ends with all three balls held stably for 0.8 s continuously.

| Mode and record | Completed flights | Drops | Forbidden ball-ball contacts | Sim duration |
| --- | --- | --- | --- | --- |
| [truth, 30 flights](results/complete_truth30.json) | 30/30 | 0 | 0 | 11.147 s |
| [sampled, 30 flights](results/complete_sampled30.json) | 30/30 | 0 | 0 | 11.145 s |
| [truth, 50 flights](results/complete_truth50.json) | 50/50 | 0 | 0 | 17.562 s |
| [sampled, 50 flights](results/complete_sampled50.json) | 50/50 | 0 | 0 | 17.562 s |

### Complete-Task Acceptance Criteria

| Check | Requirement |
| --- | --- |
| Opening | All three balls must contact their assigned hands and remain continuously stable for at least 0.05 s before the first throw starts |
| Completion count | Exactly the requested number of releases, every release ID confirmed caught, and no ball airborne at termination |
| Final hold | All three balls continuously in hand contact for 0.8 s, relative palm speed < 0.15 m/s, ball center within 0.16 m of assigned palm center, ball height > 0.40 m |
| Ending approach | At least 25 ms approach margin beyond early arrival time (currently requires predicted remaining flight time > 125 ms) |
| Ball-ball contact | Only contact between balls 0 and 2 in the right hand before the first confirmed release is permitted, provided the retained ball still contacts the right hand and both balls stay within the support region (recorded separately). Any later ball-ball contact fails |
| Posture | At default stand height: ball height < 1.30 m, palm height <= 1.15 m, elbow at least 0.08 m below shoulder, elbow lateral abduction < 0.10 m, |wrist pitch| < 38 deg, joint limit exceedance <= 0.004 rad; limits are not relaxed during ending |
| Other contacts | Non-finger self-contact penetration (arm/torso, etc.) <= 4 mm; ball contact with ground, stand, or arm, and arm-arm collision all fail |
| Planning failure | At most 8 planning attempts per throw; infeasibility stops with `NO_FEASIBLE_THROW`. Infeasible catch planning stops with `NO_FEASIBLE_RECEPTION` |

The JSON records configuration, per-ball support evidence, posture metrics, and SHA-256 checksums of source and calibration files. Permitted opening pair contact is saved separately as `supported_pair_contact_steps` and `supported_pair_force_max_n`.

## Perception Disturbance Evaluation

Single disturbance run:

```bash
uv run python -m v8_robust_control.juggle --observation-mode sampled --throws 30 --noise-mm 3 --delay-ms 20 --dropout 0.1 --seed 0
```

Prints results to stdout by default; add `--output <new_file_path>` to save a single-run JSON.

Evaluate position noise, delay, frame dropout, and the combined condition:

<details>
<summary>Generate cascade disturbance evaluation</summary>

```bash
uv run python -m v8_robust_control.evaluate_perception --seeds 0 1 --throws 30 --jobs 3 --output v8_robust_control/results/perception_trials.jsonl --video-cases baseline noise_3mm combined
uv run python -m v8_robust_control.summarize_perception --input v8_robust_control/results/perception_trials.jsonl --output v8_robust_control/results/perception_sensitivity.md --csv-output v8_robust_control/results/perception_trials.csv --overwrite
```

</details>

| Disturbance | Tested levels |
| --- | --- |
| Position noise std | 0, 1, 3, 5, 10 mm |
| Fixed transmission delay | 0, 10, 20, 40 ms |
| Whole-frame dropout probability | 0, 5%, 15%, 30% |
| Combined condition | 3 mm noise, 20 ms delay, 10% dropout |

Single-factor sweeps vary one disturbance at a time. All conditions use the same controller, initial physical state, sampling period, and measurement max age across all seeds specified by `--seeds`. Disturbance ranges are for simulation sensitivity analysis and are not calibrated to real hardware sensors.

The JSONL stores configuration, per-trial outcomes, perception sequence checksums, and estimation errors. The summarizer checks trial completeness, counts, and configuration consistency before generating the report and CSV. Interrupted runs preserve completed records, and the summarizer requires all expected trials to be present.

Success requires completing all target flights and passing catch-stability, collision, and posture checks. When no feasible catch plan is found, the trial terminates with `NO_FEASIBLE_RECEPTION`. Planning failures, actual drops, and actual collisions are counted separately.

Results: [degradation report](results/perception_sensitivity.md), [per-trial metrics](results/perception_trials.csv), [raw records](results/perception_trials.jsonl).

For the complete task, pass `--task complete`; outputs use the `complete_` prefix:

<details>
<summary>Generate complete-task disturbance evaluation</summary>

```bash
uv run python -m v8_robust_control.evaluate_perception --task complete --seeds 0 1 --throws 30 --jobs 3 --output v8_robust_control/results/complete_perception_trials.jsonl --video-cases baseline noise_3mm combined
uv run python -m v8_robust_control.summarize_perception --input v8_robust_control/results/complete_perception_trials.jsonl --output v8_robust_control/results/complete_perception_sensitivity.md --csv-output v8_robust_control/results/complete_perception_trials.csv --overwrite
```

</details>

Complete-task results: [degradation report](results/complete_perception_sensitivity.md), [per-trial metrics](results/complete_perception_trials.csv), [raw records](results/complete_perception_trials.jsonl). On current `HEAD`, disturbed complete-task runs switch to the filtered observer (the archived three-point-estimator baseline is preserved at tag `report-v1`). `--jobs` defaults to 1. Two seeds per condition do not provide an accurate estimate of population success probability.

## Videos

Demo: [perception disturbance comparison video](../videos/v8_perception_before_rl.mp4). The overlay labels configuration, seed, completed flights, and stop reason; corresponding trial metadata is in [video_comparison.json](results/video_comparison.json).

`--video-cases` records raw video for the first seed of each selected case as `videos/v8_before_rl_<case>_seed<seed>.mp4`. Single runs accept `--video <path>`.

Raw videos record executed motion at 60 fps and exclude planning rollouts. On catch-planning failure, the trial stops immediately, so the video may be shorter than 1 s before the ball hits the floor.

| Raw video (seed 0) | Sim stop time | Completed flights | Outcome |
| --- | --- | --- | --- |
| [Zero disturbance](../videos/v8_before_rl_baseline_seed0.mp4) | 10.236 s | 30/30 | Completed |
| [3 mm position noise](../videos/v8_before_rl_noise_3mm_seed0.mp4) | 0.279 s | 0/30 | Catch planning failure |
| [Combined disturbance](../videos/v8_before_rl_combined_seed0.mp4) | 0.287 s | 0/30 | Catch planning failure |

Frame counts, playback rates, and trial outcomes are stored in the raw evaluation JSONL.

Generate the annotated side-by-side comparison video:

<details>
<summary>Generate cascade comparison video and metadata</summary>

```bash
uv run python -m v8_robust_control.compare_videos --input v8_robust_control/results/perception_trials.jsonl --cases baseline noise_3mm combined --seed 0 --output videos/v8_perception_before_rl.mp4 --metadata-output v8_robust_control/results/video_comparison.json
```

</details>

Side-by-side videos play at real-time speed and match the duration of the longest clip. Clips that terminate early freeze on their final frame labeled `STOPPED - final frame held` while keeping their original simulation stop time. `--hold-seconds` adds extra end-hold duration (default `0` s).

Passing a single case to `--cases` produces a standalone clip. `--output` sets the video path and `--metadata-output` sets the JSON path in `results/`; both must be new files.

Complete-task raw videos are named `videos/v8_complete_before_rl_<case>_seed<seed>.mp4`. Generate the side-by-side video:

<details>
<summary>Generate complete-task comparison video and metadata</summary>

```bash
uv run python -m v8_robust_control.compare_videos --input v8_robust_control/results/complete_perception_trials.jsonl --cases baseline noise_3mm combined --seed 0 --output videos/v8_complete_perception_before_rl.mp4 --metadata-output v8_robust_control/results/complete_video_comparison.json
```

</details>

Complete-task demos: [zero disturbance](../videos/v8_complete_before_rl_baseline_seed0.mp4), [perception disturbance comparison](../videos/v8_complete_perception_before_rl.mp4), [video metadata](results/complete_video_comparison.json).

### PPO Residual Policy Demos

The videos below load the PPO checkpoints selected in their training records. Control is nominal controller plus PPO residual, played at real-time speed; unlabeled cases use zero disturbance. A single successful video does not represent aggregate performance improvement.

| Task | Video | Metadata |
| --- | --- | --- |
| Full single-ball throw-and-catch | [Single ball](../videos/v8_ppo_single_seed0.mp4) | [single_seed0.json](results/policy_videos/single_seed0.json) |
| Two-ball exchange | [Two ball](../videos/v8_ppo_two_seed0.mp4) | [two_seed0.json](results/policy_videos/two_seed0.json) |
| Three-ball 30 throws + ending | [Three ball](../videos/v8_ppo_complete30_seed0.mp4) | [complete30_seed0.json](results/policy_videos/complete30_seed0.json) |
| Three-ball 30 throws + ending, `guarded_feedback`, 3 mm noise | [Perception noise](../videos/v8_ppo_feedback_noise3_seed0.mp4) | [feedback_noise3_seed0.json](results/policy_videos/feedback_noise3_seed0.json) |

In the `guarded_feedback` video, both added joint and finger corrections are zero, so the physical trajectory matches the nominal reference. This episode illustrates preserving the nominal trajectory under noisy measurements and is not evidence that PPO improves task performance.

Metadata records training and trial seeds, selected budget, model and video checksums, task configuration, and physical acceptance outcomes. Recording covers the full episode including initial hold and final catch hold; failed episodes keep their termination record and exit non-zero.

For recording commands, see [Recording Residual Policy Videos](#recording-residual-policy-videos).

## Planning Seed Plans

The complete task loads [complete_seed_plans.json](complete_seed_plans.json) by default. Seed plans store hand motion parameters used as initial guesses for planner rollouts and refinement; they do not replay or overwrite ball states. Reproducing existing baselines does not require recalibration.

<details>
<summary>Optional: recalibrate after model changes</summary>

After modifying geometry, contact parameters, or timing, recalibrate and verify candidate seed plans:

```bash
uv run python -m v8_robust_control.calibrate_complete --throws 50 --output v8_robust_control/results/complete_seed_candidate.json --report v8_robust_control/results/complete_calibration.jsonl
uv run python -m v8_robust_control.complete_task --throws 50 --observation-mode sampled --seed-plans v8_robust_control/results/complete_seed_candidate.json --output v8_robust_control/results/complete_candidate_sampled50.json
```

Calibration exports seed plans only after the complete task passes; the JSONL keeps per-round results, and `--input <seed_file>` sets the starting point.

</details>

## Residual Control

Residual environments apply bounded motor corrections on top of the nominal controller, covering single-ball reception, full single-ball throw-and-catch, and multi-ball tasks.

### Single-Ball Reception

Uses the same G1, dual LEAP Hand, and ball models to evaluate bounded residuals over the classical feedback controller. Each episode starts with a descending ball and ends when the left hand holds it stably.

| Item | Default setting |
| --- | --- |
| Incoming ball | Landing offset ±50 mm per axis; horizontal velocity ±0.12 m/s per axis; initial height 0.20 ±0.015 m above support point, downward velocity 0.30 ±0.10 m/s |
| Control and measurement | Policy 50 Hz, servo and physics step 1 kHz, position measurement 60 Hz; max measurement age 50 ms |
| State estimation | Free flight fits velocity from up to 8 position frames; post-contact uses Kalman estimation from position measurements, contact state, and palm velocity; never reads ground-truth ball velocity |
| Nominal controller | Predicts landing point from valid estimates and generates velocity- and acceleration-limited quintic trajectories; yields downward and closes fingers upon palm contact |
| Residual | Fingers may relax on force-bearing hand-ball contact and close further once palm-supported (synergy ±0.15 rad); once scheduled closing ends, palm remains loaded, and the ball estimate is valid with uncertainty <= 15 mm, palm XYZ ±15 mm per axis opens; displacement speed <= 0.15 m/s with finger rate limits |
| Success | Force-bearing hand-ball contact for 0.2 s continuously; relative speed < 0.15 m/s, tangential palm distance < 40 mm, normal distance 10–60 mm |
| Deadline | 0.8 s; failing to catch stably is counted as task failure |

Ball state is initialized only at reset; during execution only actuator targets are written, with no grasp constraints, external ball forces, or ball-state overwrites. An independent verifier checks catch stability, collisions, ball height, and posture at every physics step.

<details>
<summary>Observation and control interface</summary>

[ResidualCatchEnv](residual_env.py) provides the Gymnasium interface (environment version 6). The 73-D observation includes estimated relative ball position and velocity, validity and age, joint states, palm velocity, contacts, time, applied action, nominal reference position error and velocity, estimation uncertainty, and authority flags for extra closing and palm displacement. When measurements expire, the existing reference is retained without falling back to ground-truth ball state.

Disturbances affect only the ball position measurement pipeline; joints, contact signals, and physics parameters are not randomized.

The contact phase continues processing received position measurements. Contact transitions update the motion model and velocity prior; delayed measurements update at their sample time and predict forward to current control time. Contact signals alone do not synthesize position measurements, and expiration rules apply in both phases. Ground-truth estimation error is recorded for evaluation only.

Free-flight uncertainty is computed from position-fit residuals, sampling intervals, and prediction age. Contact filtering carries over the flight-estimated measurement variance with a 3 mm sensor-noise floor; it does not read the true disturbance level or true ball velocity.

Flight interception uses the nominal controller; the residual policy acts during post-contact cushioning and stabilization. Ballistic replanning pauses during force-bearing hand-ball contact. On palm-contact transitions, the nominal reference subtracts already-applied residuals so the reconstructed palm target stays continuous without double-counting displacement. All incoming-ball episodes are fully simulated and counted in evaluation.

Physics-step cache reuse reduces redundant computation without changing servo, integration, or per-step safety check frequencies. [benchmark.json](results/reception_refinement/benchmark.json) stores matched-scenario configuration, outcomes, and timing.

</details>

<details>
<summary>Reward definition</summary>

Under default settings, single-ball reception, full single-ball, and multi-ball environments share the same terminal rewards and action costs; multi-ball environments additionally record catch-return progress.

| Term | Per-step reward |
| --- | --- |
| Task success | One-time +10 |
| Normal failure | One-time -10 (drop, unconfirmed release, missed window, timeout) |
| Safety failure | One-time -20 (forbidden contact, posture violation, ball height limit, external force, non-finite state) |
| Request magnitude | `-0.05 * dt * mean(u^2)` |
| Adjacent request delta | `-0.02 * dt * mean(((u - previous_u) / 0.2)^2)` |
| Catch progress (multi-ball only) | `+1/N` per newly confirmed returned release ID, capped at +1 per episode (`N` = target throws) |
| Distance shaping, survival, holding, airtime, repeat touches | 0 |

`dt` is actual step duration (typically 0.02 s). `u` is the authority-gated, rate-limited normalized displacement and finger action in reception; the clipped, authority-gated request in full single-ball; and the similarly gated gain request in multi-ball (where zero request maps to XY gain 0.5 and finger gain 1). These costs do not directly measure joint motion, torque, or energy.

Ball height, forbidden contacts, posture limits, and catch criteria are checked by the independent verifier. The 0.8 s deadline is observable and returns `terminated=True`; it is not a training-budget truncation.

</details>

### Reference Results

Each condition uses test seeds 140000–140019. Candidate screening requires every training seed to preserve zero-disturbance catches, exceed the classical controller on paired disturbed totals, and not increase safety failures.

| Policy | Zero-disturbance catches (of 20) | Disturbed catches (of 100) | Safety failures (of 120) | Screening |
| --- | --- | --- | --- | --- |
| Classical controller | 19 | 67 | 15 | Baseline |
| Fixed residual | 19 | 72 | 15 | Passed |
| PPO, training seeds 0/1/2 | 19, 19, 19 | 72, 72, 71 | 15, 15, 15 | Passed |
| SAC, training seeds 0/1/2 | 19, 19, 17 | 73, 72, 50 | 15, 15, 20 | Failed |

The fixed residual matches PPO's catch count; extra benefit from a dynamic policy is not confirmed. This table covers only the single-ball reception subtask; active release and full three-ball acceptance are below. Limited-budget results do not imply algorithm convergence.

Results: [policy comparison report](results/reception_comparison/report.md), [PPO evaluation report](results/reception_refinement/report.md), [PPO learning curve](results/reception_refinement/learning_curve.png), [SAC learning curve](results/reception_comparison/sac_learning_curve.png), [throughput benchmark](results/reception_refinement/benchmark.json).

The fixed residual is selected on validation seeds from zero request, the initial outputs of the three PPO models, and their mean contact-phase requests, while keeping nominal feedback, contact authority, clipping, and rate limits. Because this control reuses PPO training assets, it does not separately compare parameter-search cost.

Feedback diagnostics include complete ball-measurement dropout, masking direct ball-state fields in the policy observation, and freezing the policy's initial output. The latter two still keep the nominal reference and classical feedback; the current small sample shows no additional gain from online policy feedback.

<details>
<summary>Training, independent evaluation, and report commands</summary>

PPO and SAC each use 3 training seeds at 4,096 control steps per seed, selecting the 2,048- or 4,096-step checkpoint on validation seeds. Training and validation cover zero disturbance, 1/3/10 mm noise, 40 ms delay, 30% dropout, and combined conditions; sampling probabilities are saved in the training records.

Validation seeds are 90000–90009, independent test seeds 140000–140019, and feedback diagnostic seeds 150000–150004. Checkpoint selection preserves zero-disturbance catches and safety failures first, then compares completions and residual magnitude (ties favor earlier budgets). Checkpoints advance along the same training run, evaluations isolate training RNG state, and save/load consistency is verified.

Nominal regression and throughput:

```bash
uv run python -m v8_robust_control.benchmark_reception --episodes 8 --output v8_robust_control/results/reception_refinement/benchmark.json
uv run python -m v8_robust_control.evaluate_residual --seed-start 1000 --episodes 20 --cases baseline --output v8_robust_control/results/reception_refinement/nominal.json
```

PPO training and archiving:

```bash
uv run python -m v8_robust_control.train_residual --algorithms ppo --steps 4096 --checkpoints 2048 4096 --seeds 0 1 2 --validation-seed-start 90000 --eval-episodes 10 --jobs 3 --output-dir build/v8_reception_refinement
uv run --with matplotlib python -m v8_robust_control.summarize_residual --input-dir build/v8_reception_refinement --output v8_robust_control/results/reception_refinement/training.json --report build/v8_reception_refinement/validation.md --curve-output v8_robust_control/results/reception_refinement/learning_curve.png
```

PPO independent test, feedback audit, and report:

```bash
uv run python -m v8_robust_control.evaluate_residual --input-dir build/v8_reception_refinement --seed-start 140000 --episodes 20 --cases baseline noise_3mm combined noise_10mm delay_40ms dropout_30pct --jobs 3 --output v8_robust_control/results/reception_refinement/test.json
uv run python -m v8_robust_control.audit_residual_feedback --input-dir build/v8_reception_refinement --seed-start 150000 --episodes 5 --output v8_robust_control/results/reception_refinement/feedback.json --checkpoints ppo_seed0_lr0.0003_shape0.zip ppo_seed1_lr0.0003_shape0.zip ppo_seed2_lr0.0003_shape0.zip
uv run python -m v8_robust_control.summarize_residual --input v8_robust_control/results/reception_refinement/test.json --training v8_robust_control/results/reception_refinement/training.json --nominal v8_robust_control/results/reception_refinement/nominal.json --feedback v8_robust_control/results/reception_refinement/feedback.json --report v8_robust_control/results/reception_refinement/report.md
```

Matched-budget SAC training and test:

```bash
uv run python -m v8_robust_control.train_residual --algorithms sac --steps 4096 --checkpoints 2048 4096 --seeds 0 1 2 --validation-seed-start 90000 --eval-episodes 10 --jobs 3 --output-dir build/v8_reception_sac
uv run --with matplotlib python -m v8_robust_control.summarize_residual --input-dir build/v8_reception_sac --output v8_robust_control/results/reception_comparison/sac_training.json --report build/v8_reception_sac/validation.md --curve-output v8_robust_control/results/reception_comparison/sac_learning_curve.png
uv run python -m v8_robust_control.evaluate_residual --input-dir build/v8_reception_sac --seed-start 140000 --episodes 20 --cases baseline noise_3mm combined noise_10mm delay_40ms dropout_30pct --jobs 3 --output v8_robust_control/results/reception_comparison/sac_test.json
```

Fixed-residual validation selection and independent test:

```bash
uv run python -m v8_robust_control.compare_residual select-fixed --input-dir build/v8_reception_refinement --jobs 3 --output v8_robust_control/results/reception_comparison/fixed_selection.json
uv run python -m v8_robust_control.compare_residual test-fixed --selection v8_robust_control/results/reception_comparison/fixed_selection.json --reference v8_robust_control/results/reception_refinement/test.json --output v8_robust_control/results/reception_comparison/fixed_test.json
```

Generate the policy comparison report from archives:

```bash
uv run python -m v8_robust_control.compare_residual report --ppo-training v8_robust_control/results/reception_refinement/training.json --ppo-test v8_robust_control/results/reception_refinement/test.json --sac-training v8_robust_control/results/reception_comparison/sac_training.json --sac-test v8_robust_control/results/reception_comparison/sac_test.json --fixed-selection v8_robust_control/results/reception_comparison/fixed_selection.json --fixed-test v8_robust_control/results/reception_comparison/fixed_test.json --output v8_robust_control/results/reception_comparison/report.md
```

</details>

### Full Single-Ball Nominal Reference

`single_ball` uses a physical single-ball model starting in the right hand, performing one physical release, flight, and left-hand catch. Acceptance requires >= 0.05 s continuous initial hold, exactly one release ID, return of that ID, and >= 0.2 s continuous hold in the left hand. Ball height, forbidden contacts, posture, and external forces are checked at every step; no further throw is scheduled after catch.

```bash
uv run python -m v8_robust_control.single_ball --observation-mode truth --output v8_robust_control/results/single_ball_truth.json
uv run python -m v8_robust_control.single_ball --observation-mode sampled --output v8_robust_control/results/single_ball_sampled.json
```

Both modes use zero perception disturbance; `--timeout` defaults to 5 s. Records save initial hold, release and catch events, final support evidence, and checksums of source and planning inputs.

Both reference runs complete one release and catch, with 0.180 s continuous initial hold, 0.200 s continuous final hold, and 1.095 s total simulation time.

This entry point verifies the nominal controller; the Gymnasium residual interface is below.

### Full Single-Ball Residual Environment

[TransferEnv](transfer_env.py) provides a Gymnasium `reset/step` interface for the full single-ball task. Nominal motor references are generated offline by the classical controller; training steps physics continuously without online planning rollouts. Ball states stored in the reference are used only for zero-residual consistency checks and never enter policy observations or control.

Ball state is estimated from timestamped position measurements. The [contact estimator](transfer_observation.py) compensates for support-hand motion and predicts under gravity after release; hand motion does not generate ball position measurements or refresh their validity. Position and velocity estimation errors are logged at control steps for evaluation only.

| Interface | Definition |
| --- | --- |
| Action | 6-D: two XY requests and one finger synergy per hand; defaults cap per-axis request at 5 mm, added joint target at 0.02 rad (rate <= 0.15 rad/s), finger synergy at 0.1 rad (rate <= 0.5 rad/s) |
| Observation | 191-D: estimated ball state relative to both palms, measurement validity/age, joint state, palm motion, contact load, time, applied corrections, and authority flags; reserves 3 ball slots (unused slots invalid in single-ball) |
| Authority | XY requires a valid position estimate within uncertainty limits; source hand stops correcting after release while catching hand retains pre-contact channel; fingers use measured contact authority, with extra closing requiring palm load |
| Physical initial variation | `seat_jitter_m` perturbs right-hand initial ball position only at reset; ball state is never overwritten during rollout |
| Acceptance | >= 0.05 s continuous initial hold, actual release, >= 0.3 s contact-free flight, target-palm touch within 0.4–0.75 s, >= 0.2 s continuous final hold; release within ±50 ms of nominal time, no re-contact with source hand after confirmed release |

Under fixed physical initial conditions, perception noise does not alter zero-residual motor replay. Physical initial variation and perception disturbance are therefore defined separately and held consistent across controllers; this baseline must not be interpreted as the noise tolerance of the online planner.

```bash
uv run python -m v8_robust_control.transfer_env prepare --output-dir build/v8_transfer_reference
uv run python -m v8_robust_control.transfer_env check --reference-dir build/v8_transfer_reference --episodes 5 --output v8_robust_control/results/full_transfer/zero_residual.json
uv run python -m v8_robust_control.train_transfer run --reference-dir build/v8_transfer_reference --output-dir build/v8_transfer_ppo --checkpoints 8192 16384 32768 --seed-start 380000 --episodes 5 --jobs 3 --output v8_robust_control/results/full_transfer/training.json
uv run python -m v8_robust_control.train_transfer report --input v8_robust_control/results/full_transfer/training.json --output v8_robust_control/results/full_transfer/report.md
```

The single-ball feedback-gain study uses 3 PPO training seeds and selects checkpoints on independent validation scenarios. Baselines include zero residual, the best of 13 fixed requests, the policy's per-step mean request on validation, and the output frozen after the first valid observation. Passing requires each training seed to preserve clean, perception-only, and safety metrics while strictly increasing disturbed completions; if not passed, independent testing is skipped. Single-ball and multi-ball tasks are accepted separately.

`report` summarizes training and baseline metrics. Single-ball acceptance does not substitute for two-ball, three-ball, or opening/ending residual policy acceptance.

<details>
<summary>Optional: independent test after passing single-ball feedback screening</summary>

Run the two commands below only when `summary.all_candidates_passed` is `true` in `full_transfer/training.json`. When `false`, skip independent testing and the combined report; the single-ball validation report, policy video recording, and multi-ball training can still run under their own prerequisites.

```bash
uv run python -m v8_robust_control.train_transfer test --training v8_robust_control/results/full_transfer/training.json --input-dir build/v8_transfer_ppo --seed-start 390000 --episodes 20 --jobs 3 --output v8_robust_control/results/full_transfer/test.json
uv run python -m v8_robust_control.train_transfer report --input v8_robust_control/results/full_transfer/training.json --test v8_robust_control/results/full_transfer/test.json --output v8_robust_control/results/full_transfer/independent_report.md
```

</details>

### Multi-Ball Residual Environment and Training

[CascadeEnv](cascade_env.py) supports two-ball exchange and the complete three-ball task with opening and ending. Nominal motor trajectories are generated offline; runtime feedback uses position estimates, physical contact, and analytic ending without forward physics rollouts.

| Interface | Definition |
| --- | --- |
| Action | Default `reference` controller uses 6-D gain requests: first 4 dims map to both-hand XY feedback gains in `[0, 1]` (zero request -> 0.5); last 2 dims map to finger feedback gains in `[0, 2]` (zero request -> 1) |
| Motor correction | XY tracks estimated state error relative to nominal trajectory, capped at 5 mm/axis; added joint target <= 0.02 rad (rate <= 0.15 rad/s); finger correction <= 0.55 rad (rate <= 5 rad/s) |
| Observation | 191-D: estimated ball states, nominal reference errors, measurement validity/age, joint and palm states, contact loads, timing, and action authority |
| Timing | Policy 50 Hz, physics step 1 kHz, position measurement 60 Hz, measurement max age 50 ms |
| Tasks | Two-ball executes one throw-catch per ball and holds 0.2 s continuously; three-ball executes the requested throws and holds all three balls 0.8 s continuously; inherits all physical, safety, and posture rules above |

At zero error, any bounded feedback gain preserves the nominal trajectory. Clean completion checks compatibility, while learning benefit is judged on disturbed conditions and controls. Only motor targets change during execution; ball state is set only at reset. The policy uses states and reference errors derived from position measurements; ground-truth ball velocity is used only for independent physical verification and error logging.

Nominal references store executed motor commands, position estimates, contact geometry, and closed-loop state trajectories. Wrapping activates when a single ball physically contacts the palm with a valid estimate; ending confirms support from palm/finger contact and estimated relative velocity, fitting a gravity flight model from at least three post-release position frames.

<details>
<summary>Generate references and zero-residual checks</summary>

```bash
uv run python -m v8_robust_control.cascade_env prepare --task two --output-dir build/v8_two_ball_reference
uv run python -m v8_robust_control.cascade_env prepare --task complete --throws 30 --output-dir build/v8_complete30_reference
uv run python -m v8_robust_control.cascade_env check --reference-dir build/v8_two_ball_reference --output v8_robust_control/results/full_transfer/two_ball_zero.json
uv run python -m v8_robust_control.cascade_env check --reference-dir build/v8_complete30_reference --output v8_robust_control/results/full_transfer/complete30_zero.json
```

</details>

PPO uses 3 training seeds. Two-ball warm-starts from the single-ball encoder while resetting reference-error channels and the action head; three-ball retains the two-ball policy and initializes the added ball observation channels. Default training conditions are clean, 3 mm position noise, 2 mm initial ball seat jitter, and their combination. Seat jitter is sampled per hand uniformly in ±2 mm on both horizontal axes (balls in the same hand share the offset); validation covers all 8 scenario classes. Checkpoints are selected first on clean complete-task completion, then disturbed completions and safety failures.

The 8 scenario classes are: clean, 3 mm noise only, 2 mm and 5 mm seat jitter, both seat jitters combined with 3 mm noise, 2 mm seat jitter + 20 ms delay, and 2 mm seat jitter + 3 mm noise + 20 ms delay + 10% dropout.

Rewards inherit the terminal rewards and request costs above plus the catch-return progress term. Repeat contacts or holding do not award extra progress.

<details>
<summary>Train two-ball and three-ball policies</summary>

```bash
uv run python -m v8_robust_control.train_cascade run --reference-dir build/v8_two_ball_reference --output-dir build/v8_two_gain_ppo --warm-start-study v8_robust_control/results/full_transfer/training.json --warm-start-dir build/v8_transfer_ppo --checkpoints 1024 4096 8192 --seed-start 380000 --episodes 3 --jobs 3 --output v8_robust_control/results/cascade_training/two.json
uv run python -m v8_robust_control.train_cascade report --input v8_robust_control/results/cascade_training/two.json --output v8_robust_control/results/cascade_training/two.md
uv run python -m v8_robust_control.train_cascade run --reference-dir build/v8_complete30_reference --output-dir build/v8_complete30_gain_ppo --warm-start-study v8_robust_control/results/cascade_training/two.json --warm-start-dir build/v8_two_gain_ppo --checkpoints 1024 4096 8192 --seed-start 380000 --episodes 3 --jobs 3 --output v8_robust_control/results/cascade_training/complete30.json
uv run python -m v8_robust_control.train_cascade report --input v8_robust_control/results/cascade_training/complete30.json --output v8_robust_control/results/cascade_training/complete30.md
```

</details>

Validation results: [two-ball exchange](results/cascade_training/two.md), [three-ball complete task](results/cascade_training/complete30.md).

[Independent evaluation](evaluate_cascade.py) freezes the candidate checkpoints, baselines, and test-seed protocol before running the fixed test suite. Clean complete-task completion, disturbance non-degradation, and strict feedback gain are reported separately; candidates failing robustness comparisons still retain their failure records. Independent seeds are never used for training or checkpoint re-selection.

<details>
<summary>Independent evaluation and report</summary>

```bash
uv run python -m v8_robust_control.evaluate_cascade run --training v8_robust_control/results/cascade_training/complete30.json --input-dir build/v8_complete30_gain_ppo --seed-start 390000 --episodes 5 --jobs 3 --protocol-output v8_robust_control/results/cascade_training/complete30_protocol.json --output v8_robust_control/results/cascade_training/complete30_test.json
uv run python -m v8_robust_control.evaluate_cascade report --training v8_robust_control/results/cascade_training/complete30.json --input v8_robust_control/results/cascade_training/complete30_test.json --output v8_robust_control/results/cascade_training/complete30_test.md
```

</details>

The evaluation exit code reflects whether all clean complete-task runs succeeded. Disturbance non-degradation requires non-decreasing completions on perception-noise and physical-seat-jitter suites with non-increasing safety failures; strict feedback gain additionally requires exceeding zero-residual, timed-gain, and own-frozen-gain baselines on physical initial-state disturbances.

Non-degradation is a relative comparison and does not mean disturbed tasks pass. Tying a baseline at zero completions is not evidence of robustness improvement.

Independent tests use seeds 390000–390004 (5 runs per scenario class). Clean repeats the same physical initial state to verify task compatibility and reset consistency.

| Controller | Clean (of 5) | 3 mm noise only (of 5) | Seat jitter & combined (of 30) | Safety failures (of 40) |
| --- | --- | --- | --- | --- |
| Zero-request gain | 5 | 0 | 0 | 29 |
| Timed gain | 5 | 0 | 0 | 31 |
| PPO, training seeds 0/1/2 | 5, 5, 5 | 0, 0, 0 | 0, 0, 0 | 33, 33, 32 |

All three PPO policies pass clean complete-task acceptance; none complete disturbed tasks, and all incur more safety failures than the zero-request baseline, failing both disturbance non-degradation and strict feedback-gain checks.

Results: [independent evaluation report](results/cascade_training/complete30_test.md), [raw record](results/cascade_training/complete30_test.json), [frozen protocol](results/cascade_training/complete30_protocol.json).

### Feedback Control Under Perception Disturbances

[feedback_env.py](feedback_env.py) provides optional feedback controllers sharing the multi-ball task, observation, and motor correction limits. Training and evaluation records store the controller name so checkpoints load in their matching environment.

| `--controller` | Request and control behavior |
| --- | --- |
| `reference` (default) | Gain mapping above; masks requests by contact authority at policy steps |
| `latched_feedback` | Same mapping, retains raw policy requests; checks physical control authority at every servo step |
| `guarded_feedback` | Retains raw requests; non-positive requests disable correction, positive `u` maps to XY gain `u^2` and finger gain `2u^2`; gates corrections using measurement uncertainty evidence |

In `reference`, a zero request maps to non-zero gain, so when contact authority changes between policy steps, a masked request can enable default feedback. `latched_feedback` preserves disable requests while still gating physical corrections by estimate validity, contact, and release authority.

`guarded_feedback` enables a deviation correction only after three consecutive distinct-timestamp measurements exceed 3x the estimated position standard deviation along the same principal axis. During initial holding, finger centering is allowed only under true support contact and valid position estimates. Ending similarly requires three consecutive significant deviations; without sufficient evidence it follows the nominal ending trajectory, and invalid or expired measurements clear accumulated evidence.

These controllers only modify motor targets. Physical verification, terminal rewards, and request magnitude/rate costs are unchanged; request costs are computed over raw requests that hold authority at the policy step. Because gain mappings differ, policies cannot be compared by episode return alone; evaluation uses completion counts and physical safety metrics.

`train_cascade run --controller motor_residual` exposes a bounded joint-residual interface controlling shoulder roll, elbow, and wrist pitch per arm (joint bias <= 0.02 rad, rate <= 0.15 rad/s at 1 kHz servo steps; nominal finger commands unchanged). Policy observations include ball position estimates and joint, finger, and palm errors relative to the nominal trajectory.

This interface retains the offline reference timing and does not expose actions for throw wait times, catch-path parameters, or contact state-machine transitions.

Correction authority is determined from joint position and velocity deviations during the 0.22 s pre-launch window and held thereafter. Joint encoders use noiseless simulated readings, so under perception-only disturbances corrections stay zero. This authority rule targets initial-state offsets and does not cover mid-run physical disturbances.

Adding `--motor-curriculum` trains short episodes by confirmed catch count and logs curriculum progress. Only targets below the full throw count can terminate early with a prefix reward; full-count runs still require final hold and receive no prefix reward. Independent evaluation always runs the full throw count with opening and ending checks. No policy on this interface has passed full disturbed-task acceptance.

The frozen-output baseline for the motor interface captures one policy output once all ball observations are valid, startup observation completes, and correction authority opens (holding zero request beforehand), then keeps that output fixed.

When motor authority is closed, PPO outputs zero request and masks policy log-prob and entropy gradients for those actions while the value network still learns returns.

Motor policies cap initial normalized gSDE action std at ~0.04, independent of environment action and rate limits.

Audit motor requests, control authority, and release errors:

```bash
uv run python -m v8_robust_control.audit_feedback --reference-dir build/v8_complete30_reference --controllers motor_residual --cases clean perception_extreme seat_2mm --requests zero positive negative --seeds 410200 410201 --jobs 2 --output build/v8_motor_action_audit.json
```

The audit records physical state at each confirmed release, nominal state at the same simulation time, position/velocity estimation errors, and terminal ball state. Ground truth is diagnostic only and never enters policy observations. `positive` and `negative` apply all +1 and all -1 requests; the disable request for joint residuals is `zero`.

Compare against the native online planner under the same initial offset:

```bash
uv run python -m v8_robust_control.audit_feedback --reference-dir build/v8_complete30_reference --native-timeout 2.2 --cases seat_2mm --seeds 410200 --jobs 1 --output build/v8_native_action_audit.json
```

`--native-timeout` sets the simulation time cap and switches to the native planner. Short diagnostics isolate differences between replay and online planning; timeouts do not count as complete-task successes.

<details>
<summary>Feedback diagnostics and policy training</summary>

First generate the single-ball, two-ball, and three-ball nominal references and single-ball training records above, then run:

```bash
uv run python -m v8_robust_control.audit_feedback --reference-dir build/v8_complete30_reference --seeds 410000 410001 --jobs 3 --output v8_robust_control/results/feedback_training/diagnosis.json
uv run python -m v8_robust_control.train_cascade run --controller guarded_feedback --reference-dir build/v8_two_ball_reference --output-dir build/v8_feedback_two_final_ppo --warm-start-study v8_robust_control/results/full_transfer/training.json --warm-start-dir build/v8_transfer_ppo --checkpoints 4096 --seed-start 410100 --episodes 3 --jobs 3 --output v8_robust_control/results/feedback_training/two.json
uv run python -m v8_robust_control.train_cascade report --input v8_robust_control/results/feedback_training/two.json --output v8_robust_control/results/feedback_training/two.md
uv run python -m v8_robust_control.train_cascade run --controller guarded_feedback --reference-dir build/v8_complete30_reference --output-dir build/v8_feedback_complete30_final_ppo --warm-start-study v8_robust_control/results/feedback_training/two.json --warm-start-dir build/v8_feedback_two_final_ppo --checkpoints 4096 --seed-start 410100 --episodes 3 --jobs 3 --output v8_robust_control/results/feedback_training/complete30.json
uv run python -m v8_robust_control.train_cascade report --input v8_robust_control/results/feedback_training/complete30.json --output v8_robust_control/results/feedback_training/complete30.md
```

</details>

Diagnostics compare default vs. disabled requests across contact transitions on identical initial states and disturbance sequences. Training uses 3 PPO seeds at 4,096 steps each (two-ball transfers the single-ball encoder with a reset action head; three-ball transfers the same-controller two-ball policy). Diagnostic, validation, and independent test seeds are disjoint.

Records: [contact-transition diagnosis](results/feedback_training/diagnosis.json), [two-ball validation](results/feedback_training/two.md), [three-ball validation](results/feedback_training/complete30.md).

Independent evaluation uses two suites: `reference` (physical seat jitter and combined disturbances) and `perception` (fixed physical initial state, varying position noise 1/3/5/10 mm, delay 20/40 ms, dropout 10%/30%, and combined). Each suite evaluates zero request, timed gain, three PPO policies, and their frozen-output controls.

<details>
<summary>Independent evaluation of feedback controllers</summary>

```bash
uv run python -m v8_robust_control.evaluate_cascade run --training v8_robust_control/results/cascade_training/complete30.json --input-dir build/v8_complete30_gain_ppo --suite perception --seed-start 510000 --episodes 5 --jobs 3 --protocol-output v8_robust_control/results/feedback_training/reference_perception_protocol.json --output v8_robust_control/results/feedback_training/reference_perception_test.json
uv run python -m v8_robust_control.evaluate_cascade report --training v8_robust_control/results/cascade_training/complete30.json --input v8_robust_control/results/feedback_training/reference_perception_test.json --output v8_robust_control/results/feedback_training/reference_perception_test.md
uv run python -m v8_robust_control.evaluate_cascade run --training v8_robust_control/results/feedback_training/complete30.json --input-dir build/v8_feedback_complete30_final_ppo --suite reference --seed-start 500000 --episodes 5 --jobs 3 --protocol-output v8_robust_control/results/feedback_training/reference_protocol.json --output v8_robust_control/results/feedback_training/reference_test.json
uv run python -m v8_robust_control.evaluate_cascade report --training v8_robust_control/results/feedback_training/complete30.json --input v8_robust_control/results/feedback_training/reference_test.json --output v8_robust_control/results/feedback_training/reference_test.md
uv run python -m v8_robust_control.evaluate_cascade run --training v8_robust_control/results/feedback_training/complete30.json --input-dir build/v8_feedback_complete30_final_ppo --suite perception --seed-start 510000 --episodes 5 --jobs 8 --protocol-output v8_robust_control/results/feedback_training/perception_protocol.json --output v8_robust_control/results/feedback_training/perception_test.json
uv run python -m v8_robust_control.evaluate_cascade report --training v8_robust_control/results/feedback_training/complete30.json --input v8_robust_control/results/feedback_training/perception_test.json --output v8_robust_control/results/feedback_training/perception_test.md
```

</details>

Perception tests compare `reference` and `guarded_feedback` on matched seeds. Because their zero requests correspond to different gains, cross-controller differences reflect the overall control configuration; PPO gain is judged against zero-request, timed, and frozen controls within the same controller.

Under fixed physical initial states, suppressing measurement corrections lacking deviation evidence preserves the nominal trajectory, which does not demonstrate the ability to correct real flight deviations. Passing perception-only disturbances does not imply passing physical seat jitter, and tying a simple baseline is not evidence of learning benefit.

Independent perception evaluation uses seeds 510000–510004 (5 runs per condition). All configurations complete 5/5 clean runs; disturbed completions below sum over 9 perception-disturbance classes, and safety failures sum over all 10 classes.

| Control configuration | Perception-disturbed completions | Safety failures |
| --- | --- | --- |
| `reference`, zero request | 0/45 | 41/50 |
| `guarded_feedback`, zero request | 45/45 | 0/50 |
| `guarded_feedback`, timed gain | 40/45 | 1/50 |
| `guarded_feedback`, PPO seed 0 | 41/45 | 2/50 |
| `guarded_feedback`, PPO seed 1 | 40/45 | 1/50 |
| `guarded_feedback`, PPO seed 2 | 41/45 | 0/50 |

PPO perception failures occur mainly at 5 mm and 10 mm noise; none of the three policies pass disturbance non-degradation relative to zero request, and none are recommended as robustness improvements. In physical seat-jitter tests (seeds 500000–500004), all 8 control configurations score 0/30 on seat-jitter and combined conditions. These results do not show solved physical initial variation or extra RL benefit.

Independent reports: [reference perception test](results/feedback_training/reference_perception_test.md), [guarded_feedback perception test](results/feedback_training/perception_test.md), [physical seat-jitter and combined test](results/feedback_training/reference_test.md). Five runs per class compare configurations on a fixed test set and do not provide an accurate population success probability.

### Task-Parameter Interface

[TaskSpaceCascade](task_control.py) accepts bounded parameters at the native controller's throw and catch decision points, retaining the contact state machine and finger feedback while omitting online physics rollouts. Callback inputs provide timestamped ball estimates, joint states, palm motion, and palm/finger normal forces, without ground-truth ball velocity. Low-level classical contact control still uses simulated contact geometry and ball state.

| Action | Bounds |
| --- | --- |
| Throw command residual | XYZ velocity ±0.75, ±0.75, ±1.25 m/s; palm normal bias ±0.25 |
| Finger release-lead residual | ±15 ms, subject to existing absolute lead limits |
| Throw defer | 5–25 ms per defer (aligned to physics steps); <= 35 ms total per throw, keeping retry limit |
| Catch target residual | XYZ ±60 mm per axis, normal bias ±0.25 per axis, further bounded by native catch parameter limits |
| Catch tilt-scale multiplier | 1–1.5, subject to existing model limits |

`decode_action` maps 6-D `[-1, 1]` actions to phase-specific parameters. Ending elbow feedback counters abduction drift; wrist-to-stand distance feedback damps approach speed in the nullspace of palm position and normal tasks, scaled to respect native torque limits. Corrections remain bounded by actuator margin and do not replace trajectory feasibility checks. Ball height, posture, 4 mm penetration cap, and final hold use the original acceptance rules.

The defer component is quantized to physics steps first, and quantized values below 5 ms are treated as zero so small actions do not force an unintended defer.

Throw launch retains the complete-task contact conditions: ball confirmed caught, hand-ball contact maintained, and ball position/velocity relative to palm within holding bounds. The training curriculum additionally tracks recovery completion as palm within 30 mm of the recovery target at speed <= 0.30 m/s; this metric does not block normal launches. When palm downward speed at contact exceeds 1.8 m/s, the reference braking segment shortens to <= 45 ms while preserving boundary states, total recovery end time, and posture; actual deceleration is produced by torque-limited joints and hand-ball contact.

Fixed nominal commands are stored in [event_seed_plans.json](event_seed_plans.json), exported from an accepted run of the native complete-task controller and independently replayed through the event entry point. Its role differs from `complete_seed_plans.json`, which initializes the online optimizer. Offline calibration gains of fixed commands do not count as RL gains.

Run zero parameter residual:

```bash
uv run python -m v8_robust_control.task_control --throws 30 --output build/v8_task_zero30.json
```

Check initial ball seat jitter:

```bash
uv run python -m v8_robust_control.task_control --seat-jitter-mm 2 --seed 410202 --output build/v8_task_seat2mm_seed410202.json
```

`--seat-jitter-mm` applies a uniform horizontal initial ball offset at reset (shared by both balls in the same hand); ball state is never modified after start. Add `--disable-stand-guard` to ablate wrist-stand distance feedback. This interface has not passed multi-seed complete-task acceptance under 2 mm seat jitter.

`--recovery-gate` additionally requires reaching the world-frame recovery condition before throw launch for timing comparison; `--disable-adaptive-braking` restores the original braking reference. Neither changes physical acceptance criteria.

After robot, trajectory, or contact parameter changes, re-export and verify candidate fixed commands:

```bash
uv run python -m v8_robust_control.calibrate_complete --throws 50 --output build/v8_native_guesses.json --report build/v8_native_calibration.jsonl --executed-output build/v8_event_candidate.json
uv run python -m v8_robust_control.task_control --throws 50 --estimator filtered --seed-plans build/v8_event_candidate.json --timeout 20 --output build/v8_event_candidate50.json
```

The first command saves online planner guesses and executed commands; the second verifies the candidate commands with online physics rollouts disabled. Update fixed nominal commands and references only after the candidate passes; passing the native controller alone does not imply passing the event entry point.

Check whether recorded planner decisions are expressible through this interface:

```bash
uv run python -m v8_robust_control.audit_feedback --reference-dir build/v8_complete30_reference --native-timeout 12 --cases seat_2mm --seeds 410200 --jobs 1 --output build/v8_task_oracle.json
uv run python -m v8_robust_control.task_control --oracle-input build/v8_task_oracle.json --case seat_2mm --seed 410200 --output build/v8_task_oracle_replay.json
```

See the multi-ball training section for generating `build/v8_complete30_reference`. Recorded plans include accept, reject, and retry decisions; replay verifies task source, MuJoCo version, and initial state. Add `--disable-ending-guard` to ablate the dedicated ending guard. Incomplete runs save actual outcomes and exit non-zero; exhausted logs are marked `ORACLE_EXHAUSTED`.

Recorded decisions use information from the same episode and test action expressiveness only, not causal policy or RL benefit. This interface has not completed multi-seed disturbance acceptance.

### Event-Driven Parameter Training

[TaskResidualEnv](task_env.py) wraps the task-parameter interface in Gymnasium, and [train_task.py](train_task.py) provides PPO/SAC pilots and baseline evaluations. Policies act at physical throw and catch decision points with training budgets counted in decisions; underlying servo/physics runs at 1 kHz and position measurements at 60 Hz. This entry point uses separate observation/action signatures and checkpoints are not interchangeable with the 191-D feedback-gain policies.

Zero residual here means the nominal controller with online physics rollouts disabled, which differs from `complete_task` (which retains online planning). The former serves as a same-environment residual baseline, while the latter evaluates the native complete controller. Because planning cost and guards differ, degradation under zero residual cannot be attributed directly to the native controller or used alone to prove the need for RL.

Event-level observations support `difference` (default three-point finite differences) or `filtered` (contact tracking plus ballistic fitting). Filtered mode combines palm proprioception and position measurements during contact; after release, it fits position and velocity under configured gravity using only samples collected after the last hand-ball contact. Flight estimates are emitted only after >= 3 valid flight frames; expired or invalid measurements are not fabricated from proprioception. Fitting uses original sample timestamps and accounts for sample-to-control latency and semi-implicit integration without reading true ball velocity or substituting nominal release speed.

Compare zero-residual complete tasks under both estimators:

```bash
uv run python -m v8_robust_control.train_task observation-baseline --estimators difference filtered --cases clean sensor_only_1mm sensor_only_3mm sensor_only_5mm seat_2mm combined --seeds 416100 416101 --jobs 3 --output build/v8_event_estimator_baseline30.json
uv run python -m v8_robust_control.train_task observation-baseline --estimators difference filtered --cases clean --seeds 0 --throws 50 --jobs 2 --output build/v8_event_estimator_baseline50.json
```

Reports save complete-task outcomes, estimator mode, fixed test configuration, and source checksums. Position and velocity errors are evaluated read-only at paused decision callbacks and never fed to the controller. This comparison trains no policy; estimator gains and RL gains are evaluated separately.

Under zero disturbance, both `difference` and `filtered` pass 30-throw and 50-throw complete-task acceptance (50-throw uses seed 0, zero ball-ball collisions, no grasp constraints, 0.8 s continuous final hold). Disturbed performance must be evaluated separately with the same fixed commands; lower estimation error does not substitute for complete-task and safety acceptance.

The 30-throw comparison uses seeds 416100 and 416101. The table lists confirmed returned catches per episode; complete-task success under every disturbed condition is 0/2 for both modes, so disturbance acceptance is not passed.

| Condition | Difference mode | Filtered mode |
| --- | --- | --- |
| Clean | 30, 30 | 30, 30 |
| Position noise 1 mm | 0, 1 | 4, 4 |
| Position noise 3 mm | 0, 0 | 2, 2 |
| Position noise 5 mm | 0, 0 | 2, 1 |
| Seat jitter 2 mm | 0, 1 | 0, 1 |
| Combined disturbance | 0, 0 | 0, 1 |

Generate the filtered-mode training reference:

```bash
uv run python -m v8_robust_control.train_task prepare --estimator filtered --output build/v8_task_filtered_reference.json
```

Subsequent `pilot`, `audit`, and `evaluate` calls passing this file to `--reference` inherit its estimator mode. Reference files, source code, and checkpoints must match; `difference` and `filtered` references cannot be mixed. Standalone runs also accept `task_control --estimator filtered`. This option does not change estimation in `juggle`, `complete_task`, or the motor-residual environments.

Observations include sampled ball estimates and validity, joint/palm state, palm/finger normal forces, nominal command, and event context. Reference deviations come from a separate zero-disturbance sampled run matched by event. All-ball sampled deviations plus active-side arm, finger, and palm proprioception determine residual authority: when measurements match the reference within numerical tolerance, zero residual is applied; otherwise the declared action bounds remain open. Unmatched events retain action authority, with exploration scale governed by the policy distribution.

Rewards award +1 per confirmed completed flight, +10 for complete-task success, and -20 for any task failure. Action magnitude and same-hand/same-phase action delta each incur a mean-squared penalty with weight 0.002; there is no survival or airtime reward. Ball height, collisions, posture, and final hold remain independently verified. Task deadlines count as failure; compute-budget truncations are flagged separately.

Parameters span control bounds, training settings, and acceptance criteria. Some are CLI flags and others are module constants. Changing control or observation definitions requires regenerating compatible references and checkpoints; hyperparameter tuning keeps independent validation and test splits.

| Category | Current setting | Defined in |
| --- | --- | --- |
| Throw correction | Velocity limits ±0.75, ±0.75, ±1.25 m/s; palm normal ±0.25; finger lead residual ±15 ms | `task_control.THROW_RESIDUAL_LIMITS` |
| Catch correction | Position ±60 mm/axis; two normal params ±0.25 each; tilt scale 1–1.5, subject to catch bounds | `task_control.RECEPTION_RESIDUAL_LIMITS`, `decode_action` |
| Throw defer | <= 25 ms per defer, <= 35 ms cumulative, quantized to physics step, subject to min retry interval | `task_control.MAX_DEFER_S`, `MAX_TOTAL_DEFER_S` |
| Disturbance & task | Initial seat jitter, perception noise, delay, dropout, throws, deadline, decision budget | `TaskConfig` |
| PPO updates | LR `1e-4`, rollout 64 decisions, batch size 32, 5 epochs, target KL 0.01, ent coef 0 | `train_task.make_model` |
| Exploration init | 6-D std bounds `(0.20, 0.05, 0.05, 0.05, 0.05, 0.05)` | `train_task.INITIAL_ACTION_STD_BOUNDS` |

RL reward terms are listed below; `task_imitation` and `task_corrections` optimize action-label and retention losses rather than these rewards.

| Term | Value or rule |
| --- | --- |
| Confirmed catch | +1 each |
| Complete-task success | Additional +10 |
| Task failure | One-time -20; compute-budget truncation is not penalized as task failure |
| Action magnitude & delta | `-0.002 * mean_sq` each |
| Curriculum prefix | Additional +2 on reaching current physical recovery goal; does not count as complete-task success |
| Optional potential shaping | `Phi(next) - Phi(current)`, `Phi in [-0.25, 0]`, enabled by `--curriculum-shaping` |
| Survival, hold time, airtime | 0 |

RL uses discount factor `gamma = 1`. Table values are raw units; `train_task pilot` defaults to `--reward-scale 0.01` for training returns only, while evaluation reports raw units. Ball height, collision, posture, and catch criteria are hard independent constraints and are not relaxed via reward weights.

Curriculum prefix completion means reaching the specified physical catch-and-recovery goal (e.g., `catch_goal=3` requires flights 1–3 caught and recovered, awarding +2; this is not 30-throw complete-task success, and 1/3/6 are recovery targets in the same three-ball scene rather than ball counts). Auto-curriculum advances on reset once the last 32 episodes reach >= 75% success. Default stages progress through 1 recovery at ±0.5/1/2 mm seat jitter, then 3/6/30 recoveries at ±2 mm. Supervised fine-tuning and closed-loop diagnostics use a fixed 3-recovery stage without auto-advancement or curriculum rewards. The full-task stage gives no prefix reward and requires complete-task acceptance.

Generate the nominal sampled reference and run a short pilot:

```bash
uv run python -m v8_robust_control.train_task prepare --output build/v8_task_event_reference.json
uv run python -m v8_robust_control.train_task pilot --reference build/v8_task_event_reference.json --steps 64 --train-seeds 0 --validation-seeds 610000 --test-seeds 630000 --train-cases seat_0p5mm seat_2mm --cases clean seat_2mm combined --jobs 1 --output-dir build/v8_task_event_pilot --output build/v8_task_event_pilot.json
```

The report compares zero residual, simple feedback, validation-fitted fixed and per-event requests, and PPO/SAC; test seeds are excluded from fitting. `combined` applies 2 mm seat jitter, 3 mm position noise, 20 ms delay, and 10% dropout simultaneously. References, checkpoints, and pilot reports are written under `build/` and require new paths.

Audit the effect of bounded actions on early failures:

```bash
uv run python -m v8_robust_control.train_task audit --reference build/v8_task_event_reference.json --seeds 410200 410201 410202 --profiles zero lower_third_throw_medium lower_first_throw lower_second_throw --jobs 3 --output build/v8_task_action_audit.json
```

The audit runs a fixed 1.8 s window and records actual failures and catches on each development seed. `safe_window` means no physical or posture failure occurred within the window, not complete-task success.

Audit first-throw corrections under 2 mm seat jitter on the filtered reference:

```bash
uv run python -m v8_robust_control.train_task audit --reference build/v8_task_filtered_reference.json --seeds 418000 418001 418002 --profiles zero forward_first_throw_small backward_first_throw_small lateral_first_throw_positive_small lateral_first_throw_negative_small earlier_first_release_small --jobs 3 --output build/v8_filtered_action_diagnosis.json
```

Velocity probes apply ±0.075 m/s forward/backward or lateral, and the release-lead probe applies +3 ms, keeping all other commands nominal. Reports pair runs by identical physical initial state and record decision time, estimate validity/error, action authority, decoded command, contact events, and terminal state. `paired_probes` compares actual non-zero actions, catch/recovery deltas, and safety regressions. Ground truth is read-only diagnostic data and never enters policy or reward.

Audit later throws and the first catch on early failures:

```bash
uv run python -m v8_robust_control.train_task audit --reference build/v8_task_filtered_reference.json --seeds 418000 418001 418002 --profiles zero higher_first_reception lower_second_throw lateral_second_throw_positive lateral_second_throw_negative lower_third_throw_medium --jobs 3 --output build/v8_filtered_event_diagnosis.json
```

Verify that an action had a chance to execute before failure before judging catch or safety changes. Lack of gain from a finite probe set does not prove the action space infeasible, and higher windowed catches do not substitute for complete-task acceptance. Seeds inspected during development are excluded from independent testing.

Train with the catch-and-recovery curriculum:

```bash
uv run python -m v8_robust_control.train_task pilot --reference build/v8_task_event_reference.json --curriculum --curriculum-shaping --development-only --steps 512 --train-seeds 0 1 2 --validation-seeds 410200 410201 410202 410203 410204 410205 --jobs 3 --output-dir build/v8_task_development_ppo --output build/v8_task_development_ppo.json
uv run python -m v8_robust_control.train_task pilot --algorithm sac --reference build/v8_task_event_reference.json --curriculum --curriculum-shaping --development-only --steps 512 --train-seeds 0 1 2 --validation-seeds 410200 410201 410202 410203 410204 410205 --jobs 3 --output-dir build/v8_task_development_sac --output build/v8_task_development_sac.json
```

`--development-only` evaluates only first catch and recovery on development seeds without running independent complete-task tests (`all_screened` is `null`). Full-task training uses a larger budget and independent evaluation seeds:

```bash
uv run python -m v8_robust_control.train_task pilot --reference build/v8_task_event_reference.json --curriculum --curriculum-shaping --steps 4096 --train-seeds 0 1 2 --validation-seeds 890000 --test-seeds 910000 910001 --cases clean seat_2mm combined --jobs 3 --output-dir build/v8_task_confirm_ppo --output build/v8_task_confirm.json
```

Training and development evaluation use the same 12 s base configuration, terminating early once the curriculum catch-and-recovery goal is met. Re-evaluate saved checkpoints:

```bash
uv run python -m v8_robust_control.train_task evaluate --training build/v8_task_development_ppo.json --development-only --seeds 410200 410201 410202 410203 410204 410205 --jobs 3 --output build/v8_task_evaluation_ppo.json
uv run python -m v8_robust_control.train_task evaluate --training build/v8_task_development_sac.json --development-only --seeds 410200 410201 410202 410203 410204 410205 --jobs 3 --output build/v8_task_evaluation_sac.json
uv run python -m v8_robust_control.train_task evaluate --training build/v8_task_development_ppo.json --training-seed 2 --seeds 910000 910001 --validation-seeds 890000 --cases clean seat_2mm combined --output build/v8_task_complete_evaluation.json
```

`evaluate` only loads saved policies and binds the report to the original training record, model, and reference checksums. Omitting `--development-only` runs the full 30-throw task and compares zero residual, simple feedback, fixed, and per-event controls (the latter two fit only validation decisions where residual authority was open). `--training-seed` selects the checkpoint and `--input-dir` overrides the model directory.

The curriculum first requires 1 catch and recovery under 0.5, 1, and 2 mm seat jitter, then raises the consecutive goal to 3, 6, and 30. Short goals require confirmed catches plus palm return within 30 mm of the launch origin at <= 0.30 m/s, awarding +2 per short episode (logged separately in training reward). Curriculum advances when >= 75% of the last 32 episodes pass. The 30-throw stage has no early termination or prefix bonus and requires final hold.

`--curriculum-shaping` adds bounded potential shaping on sampled error: `F = Phi(next) - Phi(current)`, `Phi in [-0.25, 0]`, normalized by 20 mm position and 0.5 m/s velocity deviation. Terminal potential is zero at physical or curriculum-goal termination, so total shaping from a given initial state is independent of termination time.

Independent evaluation always runs the full 30-throw task without curriculum truncation or shaping. Initial exploration std is capped at 0.20 on dim 0 (throw forward/backward velocity or catch forward/backward position bias) and 0.05 on the remaining dims, squashed and clipped to native bounds. Reports log curriculum progress and complete-task outcomes separately. The task controller brakes elbows near joint limits via native actuators and torque limits; drops, collisions, and posture violations follow the original rules.

Finite-horizon runs use `gamma = 1` so deferring does not discount failure penalties. `--reward-scale` defaults to `0.01` for optimizer returns only while preserving reward ratios; evaluations and event ledgers use raw units.

`--algorithm` defaults to `ppo`. SAC uses identical observations, actions, curriculum, and rewards, with zero-mean gSDE exploration for the first 128 decisions, replay updates every 4 decisions thereafter, and entropy coefficient scaled by `0.01` to match raw units. Reports store algorithm settings, interaction counts, wall-clock time, and checkpoint checksums; short-budget runs screen configurations and do not represent converged training.

The evaluation entry point checks environment version, observation/action signatures, runtime source, and reference checksums. Short budgets verify interfaces and pipelines; partial catch counts do not replace complete-task success, and multi-seed robustness requires separate acceptance.

### Learnability of Observations and Corrective Actions

[audit_task_learning](audit_task_learning.py) checks whether pre-throw sampled observations predict effective small corrections. The action library includes zero correction, forward velocity +0.1 m/s, vertical velocity +0.1 m/s, finger opening lead -3 ms, and the combined velocity correction, applied only to the first throw (all other events use zero residual).

```bash
uv run python -m v8_robust_control.audit_task_learning --reference build/v8_task_event_reference.json --jobs 3 --output-dir build/v8_task_learning_audit
```

Defaults use 12 train, 6 validation, and 6 independent test seeds with uniform per-axis horizontal seat jitter in `[-0.5, 0.5] mm` (shared per hand). Every candidate action is executed from each initial state, recording physical catch, recovery, and safety failure; selectors read only policy observations (no seed or ground truth). Both full 306-D observations and 40-D local features (ball relative position/velocity, palm position/velocity error, palm/finger contact forces) are fitted. Hyperparameters are selected on validation only, and selectors are saved and frozen before test collection.

To reuse prior data and add demonstrations (merging old train/val/test into the new training split with fresh validation and test seeds):

```bash
uv run python -m v8_robust_control.audit_task_learning --reference build/v8_task_event_reference.json --training-data build/v8_task_learning_audit --train-seeds 410300 410301 410302 410303 410304 410305 410306 410307 410308 410309 410310 410311 --validation-seeds 410330 410331 410332 410333 410334 410335 410336 410337 --test-seeds 410350 410351 410352 410353 410354 410355 410356 410357 410358 410359 410360 410361 --jobs 3 --output-dir build/v8_task_learning_local
```

Output directory stores the protocol, datasets, frozen selectors, and `report.json`, comparing zero correction, train- and validation-calibrated fixed corrections, full- and local-observation selectors, and three label-shuffled local-feature controls, all replayed in physics. Fixed corrections separate nominal calibration from state-dependent feedback.

`test_library_coverage` counts test initial states where at least one library action achieves first catch and recovery; it is not a deployable policy score. Seat jitter is initial-state randomization rather than sensor noise, delay, or dropout. This audit does not replace complete three-ball or RL robustness acceptance.

The prefix still runs the three-ball opening, so later throws may occur before first catch/recovery with nominal commands; failures can arise from the first throw or other balls. Library failure does not prove the full residual space infeasible, and lack of selector gain does not prove RL ineffective.

### Multi-Event Planner Demonstrations

[task_demonstrations](task_demonstrations.py) records sampled observations at online throw/catch decision points and invokes the native planner on an isolated simulation copy. Planner and residual environment share low-level guards; the live environment executes only decoder-validated bounded parameters. Planner rejections or out-of-bounds decisions are logged and stop that diagnostic episode.

```bash
uv run python -m v8_robust_control.task_demonstrations --reference build/v8_task_event_reference.json --seeds 410400 410401 410402 --jobs 3 --output-dir build/v8_joint_demonstrations
```

Defaults compare zero residual, fixed first-throw correction, planner on first throw only, and planner on all throws and catches within the first four releases. The default target is first physical catch and recovery; `--goal` increases the recovery target and `--control-releases` sets the controllable event range. Episodes use the 30-throw base config and original safety criteria; prefix completion does not equal complete-task completion.

Outputs include the protocol, per-episode records, and `report.json`, storing pre-decision observations, requests, expressible normalized actions, and rollout time. Because the planner teacher accesses full simulation state at extra compute cost, its score does not count as a learned policy or RL gain.

`--planner-budget-s` (default 120 s) and `--trial-budget-s` (default 300 s) cap per-query and per-episode compute time. Exceeding budget is marked `TEACHER_RESOURCE_LIMIT` and reported separately, not as a physical failure or action infeasibility.

The teacher keeps the native planner's retry cache and executed throw history solely for optimization inside isolated copies, never writing live ball state. Completed episodes save immediately while preserving requested summary order.

Each query's `computation` records initialization, reference-build, and physics-rollout time, rollout step counts, and budget-check stage. Catch candidates log parameters, capture diagnostics, first forbidden contact, and post-contact rollout steps; opposing-hand predicted trajectories are counted separately. Incomplete rollouts have no capture diagnostic and are not judged infeasible. Post-contact step counts diagnose compute cost and do not exempt forbidden contacts.

Replay successful demonstrations and evaluate observation-only demonstration selectors on fresh seeds:

```bash
uv run python -m v8_robust_control.task_demonstrations --reference build/v8_task_event_reference.json --evaluate build/v8_joint_demonstrations --seeds 410420 410421 410422 410423 410424 410425 --jobs 3 --output-dir build/v8_joint_students
```

Evaluation uses only joint demonstrations that met the physical recovery goal, saving and reloading the event table before testing. Baselines include zero residual, fixed first-throw correction, per-event demonstration mean, first-throw-only selector, and nearest-demonstration selector per event using local observations. The latter two read only current observations without invoking the planner; training data, normalization, and event tables exclude test seeds. Replaying recorded actions on their source scenes checks expressiveness only and is not an independent policy score. This pilot includes no PPO/SAC training.

### Consecutive Recovery and Imitation Initialization

Collect demonstrations of 3 consecutive physical catches and recoveries with control over the first 6 releases:

```bash
uv run python -m v8_robust_control.task_demonstrations --reference build/v8_task_event_reference.json --seeds 410401 410402 410403 410404 410405 410406 --goal 3 --control-releases 6 --modes planner_joint --planner-budget-s 180 --trial-budget-s 600 --jobs 3 --output-dir build/v8_joint_recovery3
```

Candidate trajectories from the native planner can also be exported and verified through the same action interface:

```bash
uv run python -m v8_robust_control.audit_feedback --reference-dir build/v8_complete30_reference --native-timeout 2.2 --cases seat_2mm --seeds 410200 --jobs 1 --output build/v8_recorded_candidates.json
uv run python -m v8_robust_control.task_demonstrations --reference build/v8_task_event_reference.json --oracle-input build/v8_recorded_candidates.json --seeds 410200 --jitter-mm 2 --goal 3 --control-releases 6 --modes recorded_joint --jobs 1 --output-dir build/v8_joint_candidate3
```

See the multi-ball training section for preparing `build/v8_complete30_reference`. Export covers 2.2 s only; eligibility as a demonstration depends on achieving 3 physical catches and recoveries in the current environment, ignoring the source record's success flag. Candidate replay uses pre-recorded decisions and is not a causal policy score.

[task_imitation](task_imitation.py) initializes PPO's actor from qualifying demonstrations using disjoint training, closed-loop validation, and test seeds (test seeds never enter training, normalization, or checkpoint selection):

```bash
uv run python -m v8_robust_control.task_imitation --demonstrations build/v8_joint_recovery3 build/v8_joint_candidate3 --reference build/v8_task_event_reference.json --validation-seeds 410450 410451 --test-seeds 410460 410461 410462 410463 410464 410465 --goal 3 --control-releases 6 --training-seeds 0 1 2 --epochs 2000 --jobs 3 --output-dir build/v8_task_imitation
```

Only demonstrations meeting the recovery goal are used, weighted equally per episode, with input statistics computed on the training set only; the value network is not trained on action labels. Checkpoints are saved at zero initialization and fixed epochs, selected on closed-loop validation by safety failures, goal completions, and recovered flights. Checkpoints store input normalization and event range (executing zero residual outside that range) and can be loaded into PPO for continued training.

Demonstrations may span different initial jitter amplitudes (recorded per episode); validation and test jitter is set by `--jitter-mm` (default ±0.5 mm/axis).

Outputs include `protocol.json`, `training.json`, PPO checkpoints, and `report.json`, reporting action fit, source replay, float32 replay, closed-loop training rollouts, and independent test rollouts alongside zero residual, first-throw fixed calibration, per-event fixed calibration, and nearest-demonstration selector. Training replays use each demonstration's original jitter amplitude. Low action error does not imply closed-loop recovery, and imitation initialization does not imply RL or complete three-ball acceptance.

Full-budget initialization files are `ppo_bc_seed*_epoch2000.zip`, with closed-loop training-seed results in `fit_rollouts`. `selection.json` and `evaluated` record the validation-selected checkpoint and its test outcomes, which may fall back to zero initialization and therefore differ from final-epoch test performance.

### Closed-Loop Corrective Demonstrations

[task_corrections](task_corrections.py) starts from the full-budget imitation checkpoint, executes the first two policy decisions on fresh initial states, then hands over to the planner. The planner generates bounded labels on isolated physics copies at each decision; records save policy action, expert label, and executed action separately. Planner rejection or compute timeout ends collection for that episode without fabricating labels.

```bash
uv run python -m v8_robust_control.task_corrections --parent-dir build/v8_task_imitation --reference build/v8_task_event_reference.json --demonstrations build/v8_joint_recovery3 build/v8_joint_candidate3 --train-seeds 410500 410501 --validation-seeds 410550 410551 --test-seeds 410560 410561 410562 410563 410564 410565 --epochs 1000 --jobs 2 --output-dir build/v8_task_corrections
```

Defaults to `ppo_bc_seed0_epoch2000.zip`. Correction, validation, and test seeds are disjoint from initialization. Training merges original qualifying demonstrations and accepted local correction labels for a fixed epoch budget while preserving input normalization, value network, and action bounds. Later episode failure does not invalidate accepted local labels, nor do those labels prove complete-task success.

Outputs include collection records, `corrected.zip`, fit records, and `report.json`, comparing pre/post-correction policies, zero residual, first-throw calibration, and per-event calibration using the fixed-budget model. Expert-takeover episodes do not count toward policy scores. This is supervised fine-tuning; 3-recovery results do not represent complete three-ball or robust RL acceptance.

To reuse correction labels with anchor retention on original demonstrations and add expert-from-start episodes:

```bash
uv run python -m v8_robust_control.task_corrections --parent-dir build/v8_task_imitation --reference build/v8_task_event_reference.json --demonstrations build/v8_joint_recovery3 build/v8_joint_candidate3 --correction-input build/v8_task_corrections --retain-anchors --learner-steps 0 --train-seeds 410700 410701 --validation-seeds 410750 410751 --test-seeds 410760 410761 410762 410763 410764 410765 --jobs 2 --output-dir build/v8_task_retention_coverage
```

`--learner-steps 0` runs expert actions from reset (default `2` executes two policy decisions before expert takeover). Both collection modes are logged separately and excluded from policy evaluation scores.

`--retain-anchors` freezes the trunk, value network, and input statistics, fitting an output-layer correction via regularized least squares constrained on original demonstration observations (ignoring `--epochs`). Four fixed update scales undergo physical replay; only candidates preserving original demonstration recovery enter validation selection, and the test set is evaluated once after selection. `selection.json` records candidate replays, validation outcomes, and selection criteria; falling back to zero update leaves `corrected.zip` identical to the parent model and indicates no new capability learned.

Training seeds from prior correction logs may be reused; validation and test seeds from initialization and reused logs never enter training. Disturbances here are initial seat jitter only, without perception noise, delay, or dropout.

To fine-tune the final hidden layer on the same labels without re-collecting:

```bash
uv run python -m v8_robust_control.task_corrections --parent-dir build/v8_task_imitation --reference build/v8_task_event_reference.json --demonstrations build/v8_joint_recovery3 build/v8_joint_candidate3 --correction-input build/v8_task_retention_coverage --retain-anchors --partial-actor --epochs 6000 --validation-seeds 410850 410851 --test-seeds 410860 410861 410862 410863 410864 410865 --jobs 2 --output-dir build/v8_task_partial_actor
```

`--partial-actor` updates the last actor hidden layer and output layer while freezing all other weights. The loss sums correction-action MSE plus an anchor-retention term (weight 100) on parent actions, with cosine LR decay from `3e-4` to `1e-6`. Checkpoints at 0, 1/4, 1/2, and full budget are filtered by demonstration physical regression and selected on independent validation (test excluded from selection). Requires `--retain-anchors`; `--train-seeds` may be omitted when only reusing labels, or specified with fresh seeds for new collection. If `epoch=0` is selected, `corrected` test results correspond to the parent model, not a trained policy.

Per-event fixed calibration uses the same original demonstrations and local correction labels (`fixed_event.json`). Label fit and closed-loop recovery are reported separately; supervised fine-tuning does not count as PPO reward training.

CLI flags for partial-actor fine-tuning (recorded in the protocol):

| Flag | Default | Purpose |
| --- | --- | --- |
| `--actor-learning-rate` | `3e-4` | Initial learning rate |
| `--actor-final-learning-rate` | `1e-6` | Final cosine-schedule learning rate |
| `--anchor-weight` | `100` | Weight on parent-action retention term; does not replace physical regression filtering |
| `--epochs` | `1000` | Optimization epochs (example above uses `6000`) |

### Closed-Loop Divergence Diagnostics

Compare the parent model and a candidate checkpoint on original demonstration initial states using single-action, single-component, prefix, and magnitude-scaled substitutions:

```bash
uv run python -m v8_robust_control.task_corrections diagnose --parent-dir build/v8_task_imitation --candidate-dir build/v8_task_partial_actor --candidate partial_epoch6000.zip --reference build/v8_task_event_reference.json --demonstrations build/v8_joint_recovery3 build/v8_joint_candidate3 --jobs 2 --output-dir build/v8_task_closed_loop_axes
```

Records capture the first effective command difference, measurement difference, native physical events, and terminal reasons for each control. Pairwise alignment stops once event IDs or simulation timestamps diverge so different phases are not conflated. Physical events are diagnostic only and never enter policy observations; all controls execute through the standard action interface without altering live ball state or external forces. Diagnostics use training initial states only and do not serve as policy benefit or independent test acceptance.

### Opening Calibration and Phased Authority

[calibrate_opening](calibrate_opening.py) compares local first-throw parameter candidates and measures catch stability, safety, and airborne ball-surface clearance under small velocity probes. Local search keeps nominal preparation duration and sweeps first-throw forward and vertical velocity offsets over -10/0/+10 mm/s; validation and test add ±1 mm/s velocity probes per axis.

```bash
uv run python -m v8_robust_control.calibrate_opening --reference build/v8_task_event_reference.json --phase-policy-dir build/v8_task_partial_actor --parent-dir build/v8_task_imitation --local-velocity-grid --train-seeds 411200 411201 --validation-seeds 411220 411221 --test-seeds 411240 411241 411242 411243 --jobs 3 --output-dir build/v8_opening_local_calibration
```

Screening, validation, and disturbed tests target 3 physical recoveries alongside a clean 30-throw complete-task check. Screening keeps candidates passing clean 3-recovery, then selects on validation safety and completions (test excluded). Omitting `--local-velocity-grid` compares 0/100/200 ms added prep time and fixed first-throw offsets. Outputs include per-trial records, selection rationale, and `report.json`; selecting a candidate does not imply passing independent acceptance.

Nominal calibration inherits the `--reference` estimator mode and executes fixed commands directly with zero-innovation residual gating off; previews and standard calibration share the same estimator. Existing policies retain their training reference and gating: the first two releases and all throws disallow corrections, opening position corrections only on later catch events (default ±2 mm/axis via `--reception-cap-mm`, bounded by the native 60 mm cap). This control evaluates an old policy under a new authority map and does not represent new RL training on this interface.

Clearance tracks only confirmed-released balls not contacting either palm using read-only ground truth (never entering policy observations or rewards). Calibration never alters live ball states, external forces, contact models, or safety thresholds; calibration gains and RL gains are reported separately.

Jointly calibrate the first three throws and early catches:

```bash
uv run python -m v8_robust_control.calibrate_opening --reference build/v8_task_event_reference.json --joint-opening --train-seeds 412000 412001 --validation-seeds 412020 412021 --test-seeds 412040 412041 412042 412043 --jobs 3 --output-dir build/v8_joint_opening_calibration
```

`--joint-opening` compares coordinated forward velocity offsets, release lead offsets, and early catch position offsets across the first three throws through the native action interface (reverting to nominal throw commands after release 3 and nominal catch commands after release 6). Screening uses 3 recoveries, validation and disturbed tests use 6 recoveries, plus a clean 30-throw check. Validation also includes clean 6-recovery to ensure opening improvements do not break later handoffs.

Margin logs record airborne ball-surface clearance, palm/wrist distance to stand (sampled every 10 ms, capped at 50 mm), elbow posture margin, and robot contact penetration (checked every physics step alongside native safety checks). Margins are diagnostic and tie-breaking only, not extra rewards or substitutes for complete-task acceptance.

<details>
<summary>Third-throw height and corresponding catch-height calibration</summary>

Keeps the first two throws nominal and applies -0.05/-0.10/-0.15 m/s vertical velocity offset to the third throw with 0/-5/+5 mm catch height offset starting after release 4 (reverting after release 6), under the same first-throw velocity probes.

```bash
uv run python -m v8_robust_control.calibrate_opening --reference build/v8_task_event_reference.json --retained-height-grid --train-seeds 413000 413001 --validation-seeds 413020 413021 --test-seeds 413040 413041 413042 413043 --jobs 3 --output-dir build/v8_retained_height_calibration
```

Reuses 3-recovery screening, 6-recovery validation/test, and clean 30-throw verification with unchanged height and collision thresholds. Calibration and RL training remain separate.

</details>

<details>
<summary>Catch trajectory reachability and collision-avoidance diagnostics</summary>

```bash
uv run python -m v8_robust_control.calibrate_opening --reference build/v8_task_event_reference.json --reception-planner project --third-launch-dvz -0.2 --train-seeds 414000 414001 --validation-seeds 414020 414021 --test-seeds 414040 414041 414042 414043 --jobs 3 --output-dir build/v8_reception_constraint_calibration
```

`--reception-planner audit` previews the nominal catch trajectory; `project` searches bounded position and palm-normal corrections on preview failure (default `off` keeps nominal control). `--third-launch-dvz` additionally compares a third-throw vertical velocity offset without altering the first two throws.

Previews check approach and missed-catch cushioning trajectories in isolated kinematics for wrist angle, elbow, torso clearance, and robot collisions with other joints held fixed at finite samples; therefore previews do not prove dynamic collision avoidance or catch success, and post-catch recovery remains verified in physics. On search failure, the nominal trajectory runs and `fallback` is logged without counting preview failure as completion.

Per-episode `reception_plans` log candidates, constraint margins, and fallbacks. Calibration still requires clean complete-task retention, higher independent disturbed completions, and no safety regression. This diagnostic trains no policy and is not evidence of RL gain.

</details>

<details>
<summary>Missed-catch cushioning braking</summary>

`--uncaught-braking` compares nominal control against post-miss joint braking. After expected catch time, if the hand has not contacted the ball and palm/wrist is projected near the stand, cushioning tail tracking stops and torque-limited joint braking decelerates the arm. Inactive during approach, hand-ball contact, and holding (default off). Missed catches still follow original drop/timeout rules.

```bash
uv run python -m v8_robust_control.calibrate_opening --reference build/v8_task_event_reference.json --uncaught-braking --third-launch-dvz -0.2 --train-seeds 415000 415001 --validation-seeds 415020 415021 --test-seeds 415040 415041 --jobs 3 --output-dir build/v8_uncaught_braking_calibration
```

Tests pair braking on/off on matched initial states and command probes (and optional `--third-launch-dvz`). All combinations run independent disturbed tests and clean complete-task checks, including missed-catch cases failing initial screening. `summary.braking_pairs` in `report.json` stores paired metrics, and episodes log brake counts and first trigger state. Avoiding collisions, increasing completions, and achieving RL gain are distinct metrics and not interchangeable.

</details>

### Recording Residual Policy Videos

<details>
<summary>Record selected policies</summary>

Single-ball recording requires `training.json` and checkpoints from full single-ball training; two-ball and three-ball require their respective multi-ball training records and checkpoints.

```bash
uv run python -m v8_robust_control.record_policy --training v8_robust_control/results/full_transfer/training.json --input-dir build/v8_transfer_ppo --training-seed 0 --seed 0 --case clean --video videos/v8_ppo_single_seed0.mp4 --output v8_robust_control/results/policy_videos/single_seed0.json
uv run python -m v8_robust_control.record_policy --training v8_robust_control/results/cascade_training/two.json --input-dir build/v8_two_gain_ppo --training-seed 0 --seed 0 --case clean --video videos/v8_ppo_two_seed0.mp4 --output v8_robust_control/results/policy_videos/two_seed0.json
uv run python -m v8_robust_control.record_policy --training v8_robust_control/results/cascade_training/complete30.json --input-dir build/v8_complete30_gain_ppo --training-seed 0 --seed 0 --case clean --video videos/v8_ppo_complete30_seed0.mp4 --output v8_robust_control/results/policy_videos/complete30_seed0.json
uv run python -m v8_robust_control.record_policy --training v8_robust_control/results/feedback_training/complete30.json --input-dir build/v8_feedback_complete30_final_ppo --training-seed 0 --seed 510000 --case sensor_only_3mm --video videos/v8_ppo_feedback_noise3_seed0.mp4 --output v8_robust_control/results/policy_videos/feedback_noise3_seed0.json
```

`--training-seed` selects the training seed, `--seed` sets the episode seed, and `--case` chooses a training/evaluation scenario. Video and metadata paths must be new files.

</details>

## Result Reproduction

`results/` stores raw records, summary metrics, reports, and video metadata for the reference experiments. Running the commands in the corresponding sections regenerates:

| Result file | Entry point |
| --- | --- |
| `truth_baseline30/50.json`, `sampled_baseline30/50.json` | `verify_baselines` (run `truth` and `sampled` separately) |
| `complete_truth30/50.json`, `complete_sampled30/50.json` | `complete_task` (run all 4 combinations) |
| `perception_trials.jsonl`, `perception_trials.csv`, `perception_sensitivity.md` | `evaluate_perception` -> `summarize_perception` |
| `complete_perception_trials.jsonl`, `complete_perception_trials.csv`, `complete_perception_sensitivity.md` | Same with `--task complete` |
| `video_comparison.json`, `complete_video_comparison.json` | Evaluation with video recording -> `compare_videos` |
| [reception_refinement/](results/reception_refinement/) | Nominal regression, throughput, PPO training archive, independent test, feedback audit, report |
| [reception_comparison/](results/reception_comparison/) | SAC training archive & test, fixed-residual selection & test, comparison report |
| `single_ball_truth.json`, `single_ball_sampled.json` | `single_ball` (`truth` and `sampled`) |
| [full_transfer/](results/full_transfer/) | Single- and multi-ball zero-residual checks, single-ball PPO feedback-gain study |
| [policy_videos/](results/policy_videos/) | `record_policy` loading selected PPO checkpoints |
| [cascade_training/](results/cascade_training/) | Multi-ball PPO validation, frozen protocol, independent test, report |
| [feedback_training/](results/feedback_training/) | Contact-transition diagnosis, feedback controller training, perception & seat-jitter independent evaluations |

Full throw-and-catch results are generated in order: nominal references -> single-ball training -> two-ball training -> three-ball training -> independent evaluation -> policy recording.

Except for summary commands that explicitly accept `--overwrite`, output paths must be new. When regenerating results, also check:

- `build/v8_reception_refinement/`, `build/v8_reception_sac/`: if retained and matching current source, skip training and start from archiving (move existing `validation.md` or pick a new report path); fresh training requires new directories and updated downstream input paths.
- `build/v8_transfer_reference/`, `build/v8_transfer_ppo/`: store full single-ball nominal motor references and trained checkpoints; use new directories when re-running.
- `build/v8_two_ball_reference/`, `build/v8_complete30_reference/` and `build/v8_two_gain_ppo/`, `build/v8_complete30_gain_ppo/`: two-ball/three-ball references and PPO checkpoints; use new directories and sync downstream paths when re-running.
- `build/v8_feedback_two_final_ppo/`, `build/v8_feedback_complete30_final_ppo/`: feedback-controller two-ball and three-ball PPO checkpoints.
- `videos/`: recording and compositing commands refuse to overwrite existing MP4s; move existing files or pass `--video-dir` / `--output`.

Reports can be regenerated from raw records. Reception training archives can be re-exported via `summarize_residual`; if raw records for full throw-and-catch training are missing, re-run training and evaluation. Video metadata is regenerated by re-recording after moving or renaming existing MP4s. Unit tests do not depend on `results/`.

Reproduction uses the same tasks, configurations, seeds, and acceptance pipeline, without guaranteeing byte-for-byte identity. Wall-clock times, parallel log ordering, and model checksums may vary; dependency, hardware, or numerical differences may affect training and contact outcomes, so judge reproduction by the re-run acceptance metrics. Model compatibility is checked via environment version, source, and checkpoint checksums; modifying observation or control code requires retraining.

# Report evidence

| Record | Scope |
| --- | --- |
| [Dual-UR5e benchmark](../../v5_two_arm_hands/README.md) | Five nominal runs of 60 catches each using two arms and two LEAP Hands. |
| [G1 long-run cascade baselines](../../v8_robust_control/EXPERIMENT_REFERENCE.md#zero-disturbance-reference-results) | 50 confirmed flights per observation mode on the G1 cascade. |
| [Cascade-only perception baseline](../../v8_robust_control/results/perception_sensitivity.md) and [complete-task baseline](../../v8_robust_control/results/complete_perception_sensitivity.md) | Two runs per condition for noise, delay, and frame loss. Cascade-only uses the three-point estimator; on current HEAD the complete-task sweep uses the filtered observer under disturbances (archived three-point version is at tag `report-v1`). |
| [Official perception runs](official_perception_runs.json) | Two zero-noise sampled runs and one completed 1 mm measurement-noise run, with original calibration. |
| [Nominal retry run](nominal_retry_run.json) | Seed 440002 at 1 mm with the second search for the fourth throw initialized from the nominal command. |
| [Late reception pair](late_reception_pair.json) | Same-state short replay at 9.124 s with the filtered ball estimate and the exact simulated state as alternative planner inputs. |
| [Original-ending independent results](independent_440100_results.json) | Five paired seeds testing state-triggered braking with the original ending timing. |
| [Same-state ending comparison](ending_same_state_pairing.json) and [full online regressions](ending_regression4.json) | Four targeted trajectories comparing the 0.50 s and 0.35 s start thresholds, followed by four full online runs with the revised ending. |
| [Revised-ending frozen protocol](independent_440200_config.json) and [paired results](independent_440200_results.json) | Five fresh seeds under 1 mm position measurement noise and zero physical initial offsets. Both arms use the revised ending and native throw retries. |
| [Seed 440203 regression diagnosis](seed_440203_regression_diagnosis.json) | Same-state short replay around the braking trigger. The later throw-planning rejections are identified in the original logs. |
| [Seed 440203 warm-start replay](seed_440203_warmstart_downstream.json) and [full online result](seed_440203_warmstart_full30.json) | A short downstream pair and a targeted run from reset. The warm start changes the first search of Throw 12 only. |
| [Generic warm-start rule](warmstart_snapshot_rule.json) and [development snapshot results](warmstart_snapshot_results.json) | Four selected same-state search comparisons with the frozen forward correction and measured probe overhead. |
| [First-search follow-up](warmstart_first_search_followup.json) | Three development snapshots, including one native acceptance that regressed when the correction was applied on the first search. |
| [Retry-only fallback pilot](retry_only_fallback_pilot.json) | Three completed short-horizon pairs and one pending pair. Target catch counts use executed catches and exclude the pending pair. |
| [Retry-only frozen protocol](retry_only_independent_config.json) and [independent online results](retry_only_independent_results.json) | Three fresh seed pairs under 1 mm position measurement noise and zero initial offsets. Both groups use the revised ending and braking rule. The fallback fired once and required no extra probe. The summary includes all six branch records. |
| [Perception-only PPO evaluation](../../v8_robust_control/results/feedback_training/perception_test.md) | Zero, timed, learned, and frozen residual comparisons with no physical initial offsets. |
| [Rule and opening PPO ablations](../../v8_robust_control/results/closed_loop_bvp/ablation_final_report.md) | Seat-jitter sweep, fixed-bias comparison, and perception stress on the development set. |
| [Opening PPO training record](../../v8_robust_control/results/closed_loop_bvp/step2b_rl_pilot.json) | Training and validation protocols, selected checkpoint, and development outcomes. |

The JSON files preserve the experiment summaries and frozen protocols. Use the [quickstart](../../v8_robust_control/QUICKSTART_RL.md) for the standard controller and training entry points.

### Retry-only comparison

The [runner](../../v8_robust_control/experiments/retry_fallback.py) uses the archived configuration and verifies the [prior interface smoke results](retry_only_interface_smoke.txt). From the repository root, run:

```sh
uv sync --locked
uv run --locked python -m v8_robust_control.experiments.retry_fallback --check-archive
```

Omit `--check-archive` to run the six online branches. By default, new outputs go under `build/v8_retry_fallback_reproduction/`. The runner resumes matching completed branches in that directory. Pass `--output-dir` with a new directory for a fresh run.

The first-search follow-up, retry-only pilot, and retry-only independent summary and configuration are preserved copies of the original run archives. Full logs and temporary diagnostics remain local build artifacts.

The submitted report links to the `report-v1` code and evidence snapshot. Run the commands above from that tag to use the matching runner and archived inputs.

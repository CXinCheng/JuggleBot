# Run the v8 controller and try residual RL

Run these commands from the repository root with `uv` installed. Each command that writes a result uses a path under the local `build/` directory.

```bash
uv sync --locked
uv run python -m unittest v8_robust_control.test_task_env -v
uv run python -m v8_robust_control.complete_task --throws 30 --observation-mode sampled --output build/v8_team_sampled30.json
```

## Report evidence and retry reproduction

Use the `report-v1` tag for the code and evidence referenced in the submitted report. The [evidence index](../report/evidence/README.md) links the frozen configuration and archived results. Check the archive from the repository root:

```bash
uv run --locked python -m v8_robust_control.experiments.retry_fallback --check-archive
```

Omit `--check-archive` to run the six online branches. New results are written under `build/`.

## Single-ball reception

Use the same reception environment for PPO and SAC. Start with one short run to check the training pipeline. Use distinct output directories for repeat runs. For the full comparison budget and evaluation commands, see [reception results and reproduction](EXPERIMENT_REFERENCE.md#参考结果).

```bash
uv run python -m v8_robust_control.train_residual --algorithms ppo --steps 256 --checkpoints 256 --seeds 0 --eval-episodes 1 --jobs 1 --output-dir build/v8_team_reception_ppo
uv run python -m v8_robust_control.train_residual --algorithms sac --steps 256 --checkpoints 256 --seeds 0 --eval-episodes 1 --jobs 1 --output-dir build/v8_team_reception_sac
```

## Three-ball opening residual

The opening pilot uses PPO and a sampled reference trajectory. For fresh training, use a clean checkout or move any existing `build/v8_step2b_pilot_ckpts/` aside. The trainer creates that local directory and reuses its baseline cache when present. Generate a reference from the current checkout so its source fingerprints match. The archived model and evaluation records are linked below.

Run these commands from the repository root in Bash or PowerShell. The model output path is absolute and stays inside that checkout.

```bash
uv run --locked python -m v8_robust_control.train_task prepare --estimator filtered --output build/v8_team_opening_reference.json
uv run --locked python -m v8_robust_control.train_closed_loop_rl --reference build/v8_team_opening_reference.json --iterations 1 --workers 1 --model-out "$PWD/build/v8_team_opening.zip" --report-out build/v8_team_opening.json
```

For the archived pilot, see the [model](models/ppo_cl_opening_pilot.zip), [training record](results/closed_loop_bvp/step2b_rl_pilot.json), and [ablation results](results/closed_loop_bvp/ablation_final_report.md). One iteration uses 256 active decisions and still runs the built-in baseline and validation sweeps. Use the archived record to inspect a completed run before starting a new sweep.

## Interpret the results

- Sensor noise, delay, and frame loss change measurements. Seat jitter changes a ball's physical initial position. Compare policies on matched seeds, conditions, and controller settings.
- Use confirmed catches, completed episodes, and safety terminations together. For the full task, also check the final 0.8 s of continuous support. A single successful video is an example of execution.
- The opening pilot reduced aggregate unsafe terminations in the development seat-jitter sweep while confirming fewer catches. Under combined perception disturbances, it regressed on both metrics. Keep `450000..450019` outside training, tuning, and exploratory runs.
- The [late-reception summaries](../report/evidence/README.md) document one 1 mm measurement-noise success and a separate trajectory-planning failure. See the [experiment reference](EXPERIMENT_REFERENCE.md) for additional evaluations.

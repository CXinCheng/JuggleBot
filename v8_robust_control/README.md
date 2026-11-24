# v8 Perception Disturbances and Control Evaluation

`v8_robust_control` runs the Unitree G1 upper-body and dual LEAP Hand juggling tasks in MuJoCo, providing ball state estimation, perception disturbances, complete-task evaluation, and residual policy training. See [Quickstart](QUICKSTART_RL.md) for setup and training commands.

## Running

Run from the repository root. Use a new output file path for each complete-task run:

~~~bash
uv sync --locked
uv run --locked python -m v8_robust_control.juggle --throws 1 --observation-mode sampled
uv run --locked python -m v8_robust_control.complete_task --throws 30 --observation-mode sampled --output build/v8_sampled30.json
~~~

- `juggle` starts with two balls in hand and one ball in flight. One complete flight runs from confirmed release to confirmed catch; catching the initial airborne ball is not counted.
- `complete_task` starts with three balls held steadily, executes the requested throws and catches, and ends with a three-ball hold. Full task success requires passing all safety checks and holding all three balls continuously for 0.8 s. `--throws` must be an even integer of at least 4.

In `sampled` mode, `--noise-mm`, `--delay-ms`, and `--dropout` set position noise standard deviation, measurement delivery delay, and whole-frame dropout probability, respectively, and `--seed` fixes the disturbance sequence. Initial held-ball seat offset is an independent physical disturbance and should be recorded separately when comparing policies.

## Testing

~~~bash
uv run --locked python -m unittest discover -s v8_robust_control -t . -v
~~~

## Results and Reproduction

- [Report evidence index](../report/evidence/README.md) provides the frozen configurations, per-seed results, and retry-fallback reproduction commands.
- [Experiment and interface reference](EXPERIMENT_REFERENCE.md) documents the control interfaces, historical evaluations, training commands, and videos. Write new evaluation outputs under `build/`.

Perception estimators, disturbance types, and task phases vary across experiments. Compare results using matched tasks, seeds, budgets, and safety criteria. The planner relies on online private physics rollouts, and reported simulation results have not been validated on hardware.

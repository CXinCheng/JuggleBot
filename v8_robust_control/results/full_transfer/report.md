# Full Single-Ball Residual PPO Validation

The task starts with the ball held in the right hand, executes a physical release, and holds it stably in the left hand for 0.2 s. Physical initial-state offsets are applied only at reset; perception disturbances affect position measurements independently.
Nominal motor references are generated offline, and training integrates physics continuously without online planning rollouts. Zero residual reproduces the nominal trajectory under clean initial conditions.
Fixed and per-step mean controls are selected on validation only; frozen output waits for the first valid position observation. This report is not an independent test conclusion.

| Controller | Clean | Perception noise only | Physical seat jitter & combined | Safety failures |
| --- | --- | --- | --- | --- |
| zero | 5/5 | 5/5 | 18/30 | 0/40 |
| fixed | 5/5 | 5/5 | 18/30 | 0/40 |
| timed | 5/5 | 5/5 | 28/30 | 0/40 |
| PPO seed 0 | 5/5 | 5/5 | 26/30 | 0/40 |
| PPO seed 1 | 5/5 | 5/5 | 25/30 | 1/40 |
| PPO seed 2 | 5/5 | 5/5 | 25/30 | 0/40 |

Candidate gates: {'0': False, '1': False, '2': False}.
Each training seed must preserve clean, perception-only, and safety metrics while strictly exceeding zero-residual, fixed, per-step mean, and own-frozen controls on disturbed completions.
Only after all seeds pass are fresh independent seeds evaluated; failure does not support scaling to two- or three-ball policies.

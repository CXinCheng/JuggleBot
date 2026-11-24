# 30-Throw Complete-Task Independent Evaluation

Controller: `reference`; suite: `perception`.

Complete-task capability, disturbance non-degradation, and strict feedback gain are verified separately. Every successful episode must pass release, catch, continuous support, collision, and posture checks.

At zero error, feedback gains preserve the nominal trajectory; clean counts verify compatibility, while disturbed counts and baselines test learning gain.

| Controller | Clean | 3 mm noise only | Other perception disturbances | Safety failures |
| --- | --- | --- | --- | --- |
| zero | 5/5 | 0/5 | 0/40 | 41/50 |
| timed | 5/5 | 0/5 | 0/40 | 38/50 |
| live_seed0 | 5/5 | 0/5 | 0/40 | 42/50 |
| frozen_seed0 | 5/5 | 0/5 | 0/40 | 42/50 |
| live_seed1 | 5/5 | 0/5 | 0/40 | 37/50 |
| frozen_seed1 | 5/5 | 0/5 | 0/40 | 44/50 |
| live_seed2 | 5/5 | 0/5 | 0/40 | 42/50 |
| frozen_seed2 | 5/5 | 0/5 | 0/40 | 38/50 |

| PPO training seed | Clean complete task | Disturbance non-degradation | Strict feedback gain |
| --- | --- | --- | --- |
| 0 | PASS | FAIL | FAIL |
| 1 | PASS | PASS | FAIL |
| 2 | PASS | FAIL | FAIL |

This suite varies only position measurement noise, transmission delay, and frame dropout, with no physical initial-state randomization.

| Condition | Zero request | Timed gain | PPO 0 | PPO 1 | PPO 2 |
| --- | --- | --- | --- | --- | --- |
| clean | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| sensor_only_3mm | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| noise_1mm | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| noise_5mm | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| noise_10mm | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| delay_20ms | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| delay_40ms | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| dropout_10pct | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| dropout_30pct | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| perception_combined | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |

Clean runs repeat the same physical initial state to verify complete-task and reset consistency; repeated seeds do not represent independent physical initial states.

Disturbance non-degradation requires non-decreasing completions on perception disturbances with non-increasing safety failures. Strict feedback gain additionally requires exceeding zero-request, timed, and own-frozen baselines on other perception disturbances.

Candidate checkpoints and timed baselines were fixed on the validation set; independent evaluation does not re-select policies. Configuration, model checksums, source hashes, and per-episode physical evidence are stored in the protocol and raw JSON.

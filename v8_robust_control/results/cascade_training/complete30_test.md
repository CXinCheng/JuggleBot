# 30-Throw Complete-Task Independent Evaluation

Controller: `reference`; suite: `reference`.

Complete-task capability, disturbance non-degradation, and strict feedback gain are verified separately. Every successful episode must pass release, catch, continuous support, collision, and posture checks.

At zero error, feedback gains preserve the nominal trajectory; clean counts verify compatibility, while disturbed counts and baselines test learning gain.

| Controller | Clean | 3 mm noise only | Physical seat jitter & combined | Safety failures |
| --- | --- | --- | --- | --- |
| zero | 5/5 | 0/5 | 0/30 | 29/40 |
| timed | 5/5 | 0/5 | 0/30 | 31/40 |
| live_seed0 | 5/5 | 0/5 | 0/30 | 33/40 |
| frozen_seed0 | 5/5 | 0/5 | 0/30 | 32/40 |
| live_seed1 | 5/5 | 0/5 | 0/30 | 33/40 |
| frozen_seed1 | 5/5 | 0/5 | 0/30 | 31/40 |
| live_seed2 | 5/5 | 0/5 | 0/30 | 32/40 |
| frozen_seed2 | 5/5 | 0/5 | 0/30 | 33/40 |

| PPO training seed | Clean complete task | Disturbance non-degradation | Strict feedback gain |
| --- | --- | --- | --- |
| 0 | PASS | FAIL | FAIL |
| 1 | PASS | FAIL | FAIL |
| 2 | PASS | FAIL | FAIL |

Clean runs repeat the same physical initial state to verify complete-task and reset consistency; repeated seeds do not represent independent physical initial states.

Disturbance non-degradation requires non-decreasing completions on perception noise and physical seat jitter with non-increasing safety failures. Strict feedback gain additionally requires exceeding zero-residual, timed, and own-frozen baselines on physical seat jitter.

Candidate checkpoints and timed baselines were fixed on the validation set; independent evaluation does not re-select policies. Configuration, model checksums, source hashes, and per-episode physical evidence are stored in the protocol and raw JSON.

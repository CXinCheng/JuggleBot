# two: 2-Throw Complete Validation

Task acceptance and RL gains over baselines are reported separately. Policies must preserve clean completion, safety, and disturbed completions; additional feedback gain is judged against timed and frozen controls.

At zero error, feedback gains preserve the nominal trajectory; the clean column verifies compatibility, while disturbed columns and baselines test learning gain.

| Controller | Clean | Perception noise only | Physical seat jitter & combined | Safety failures |
| --- | --- | --- | --- | --- |
| zero | 3/3 | 3/3 | 2/18 | 11/24 |
| timed | 3/3 | 3/3 | 2/18 | 11/24 |
| PPO seed 0 | 3/3 | 1/3 | 3/18 | 11/24 |
| PPO seed 1 | 3/3 | 3/3 | 2/18 | 11/24 |
| PPO seed 2 | 3/3 | 2/3 | 3/18 | 11/24 |

Clean complete-task acceptance: {'0': True, '1': True, '2': True}.

Disturbance non-degradation: {'0': False, '1': True, '2': False}.
Strictly exceeds zero, timed, and own-frozen controls: {'0': False, '1': False, '2': False}.

# Single-Ball Reception Policy Comparison

PPO and SAC each use three training seeds at 4,096 control steps per seed; checkpoints are selected on the validation set only.
Validation seeds: 90000–90009; test seeds: 140000–140019.

The fixed residual requests the same 4-D vector across all episodes; nominal feedback, contact authority, clipping, and rate limits remain active.
Candidates are zero residual plus each of the three PPO models' initial outputs and mean validation contact-phase requests; test outcomes are excluded from fitting and selection.

Constant selection uses the safety-first validation score, breaking ties toward earlier candidates with zero residual listed first. Because the fixed control reuses PPO training assets, parameter-search cost is not compared separately.

Selected constant: `ppo_seed2_contact_mean`; normalized request: `[0.1651602286603874, -0.05065660785848701, 0.24524226321905546, 0.18064974746487747]`.

| Policy | Zero-disturbance catches | Disturbed catches | Safety failures | Wins/Losses vs. fixed residual |
| --- | --- | --- | --- | --- |
| Classical controller | 19/20 | 67/100 | 15/120 | 0/5 |
| Fixed residual | 19/20 | 72/100 | 15/120 | 0/0 |
| PPO / seed 0 | 19/20 | 72/100 | 15/120 | 0/0 |
| PPO / seed 1 | 19/20 | 72/100 | 15/120 | 0/0 |
| PPO / seed 2 | 19/20 | 71/100 | 15/120 | 0/1 |
| SAC / seed 0 | 19/20 | 73/100 | 15/120 | 1/0 |
| SAC / seed 1 | 19/20 | 72/100 | 15/120 | 0/0 |
| SAC / seed 2 | 17/20 | 50/100 | 20/120 | 1/25 |

Safety failures include forbidden contacts, posture or ball-height violations, external forces, and non-finite states; they are counted separately from drops and deadline timeouts.
All policies share the same test scenes; pooling three training seeds does not create new independent test scenes.

## Catches by Condition

| Policy | baseline | noise_3mm | combined | noise_10mm | delay_40ms | dropout_30pct |
| --- | --- | --- | --- | --- | --- | --- |
| Classical controller | 19/20 | 17/20 | 12/20 | 9/20 | 12/20 | 17/20 |
| Fixed residual | 19/20 | 18/20 | 13/20 | 11/20 | 12/20 | 18/20 |
| PPO | 57/60 | 54/60 | 39/60 | 32/60 | 36/60 | 54/60 |
| SAC | 55/60 | 50/60 | 34/60 | 28/60 | 34/60 | 49/60 |

## Candidate Screening

- PPO: PASS; paired disturbed delta +14, safety failures per seed [15, 15, 15].
- SAC: FAIL; paired disturbed delta -6, safety failures per seed [15, 15, 20].
- Fixed residual: PASS under the same success and safety gate.

## Training Budget and Wall-Clock Time

| Algorithm/Seed | Training steps | Selected step | Training wall time (s) |
| --- | --- | --- | --- |
| PPO/0 | 4096 | 2048 | 319.75 |
| PPO/1 | 4096 | 4096 | 308.67 |
| PPO/2 | 4096 | 4096 | 315.45 |
| SAC/0 | 4096 | 4096 | 354.91 |
| SAC/1 | 4096 | 4096 | 350.91 |
| SAC/2 | 4096 | 2048 | 361.30 |

Wall-clock times come from actual learn calls excluding validation; affected by machine load and concurrency, they are not a strict algorithm speed ranking.
Equal interaction budgets do not imply equal gradient update counts or convergence of either algorithm.

## Conclusion and Scope

Results for the classical controller, fixed residual, and both RL algorithms inform whether further training is warranted; candidate screening does not trigger controller replacement.
This test did not show either algorithm consistently exceeding the fixed residual across all training seeds. Retain the classical controller and fixed residual as baselines and do not scale to three-ball RL on this basis.
This comparison covers one finite-budget setting and cannot establish general superiority of SAC or PPO.
The fixed residual searches only the finite candidate set above; any additional benefit from online policy feedback must be judged alongside feedback ablations.
The task covers catching a descending single ball only, excluding active throws, two-ball exchange, or the complete three-ball task.

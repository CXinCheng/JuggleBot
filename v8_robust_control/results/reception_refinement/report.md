# Single-Ball Reception Independent Test

The task is single-ball reception only, excluding active throws or the complete three-ball cycle.

Each perception condition uses 20 scene seeds: [140000, 140001, 140002, 140003, 140004, 140005, 140006, 140007, 140008, 140009, 140010, 140011, 140012, 140013, 140014, 140015, 140016, 140017, 140018, 140019].

| Policy | baseline | noise_3mm | combined | noise_10mm | delay_40ms | dropout_30pct |
| --- | --- | --- | --- | --- | --- | --- |
| zero | 19/20 | 17/20 | 12/20 | 9/20 | 12/20 | 17/20 |
| static | 9/20 | 9/20 | 9/20 | 9/20 | 9/20 | 9/20 |
| ppo | 57/60 | 54/60 | 39/60 | 32/60 | 36/60 | 54/60 |

zero is the nominal reception feedback controller; static disables landing prediction while keeping identical grasping and cushioning.
Learned policies are pooled across all training seeds; training seeds share the same test scenes and are not independent scene samples.
Policy evaluation disables exploration; failure reasons, catch evidence, and action metrics are stored in the raw JSON.


## Training and Validation

The task is catching a pre-launched descending ball, excluding active throws or the complete three-ball task.

zero is the classical feedback controller; static disables its landing prediction while keeping identical finger control and contact cushioning.

These results are on the validation set; final conclusions require independent tests and feedback ablations and cannot alone establish closed-loop noise robustness.

Training budget: 4096 control steps per seed; training seeds: [0, 1, 2].
Each perception condition uses 10 independent evaluation seeds with exploration disabled.

Trained policy rows come from each seed's selected checkpoint; all training seeds ran the full budget.

| Policy | Learning rate | Shaping weight | baseline | noise_1mm | noise_3mm | combined | noise_10mm | delay_40ms | dropout_30pct | Training time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero | — | — | 10/10 | 10/10 | 8/10 | 7/10 | 5/10 | 7/10 | 10/10 | 0.00 |
| static | — | — | 6/10 | 6/10 | 6/10 | 6/10 | 6/10 | 6/10 | 6/10 | 0.00 |
| ppo | 0.0003 | 0.0 | 30/30 | 30/30 | 24/30 | 21/30 | 15/30 | 22/30 | 30/30 | 943.87 |

Each control policy is evaluated once; trained-policy denominators pool all training seeds.
The shaping-weight column reflects training setup; evaluation returns use the unified reward config in the manifest, and policies are judged by independent verification metrics.
Different training seeds share the same evaluation scenes; repeated scenes are not independent samples.
This budget verifies the training pipeline and screens configurations; it does not establish convergence or general superiority of SAC or PPO.
The validation set is used for this screening; confirmation experiments must use fresh seeds. Wall-clock time reflects local concurrency load.
Nominal reception predicts landing from position samples and yields on contact; this subtask is not equivalent to the complete three-ball task.

## Learning Curve

Budgets advance continuously along the same training run; checkpoints are selected on the validation set only, excluding independent test scenes.

Training time sums actual learn calls excluding checkpoint evaluation; summing across seeds does not equal concurrent wall-clock duration.

| Algorithm/Seed | Control steps | baseline | noise_1mm | noise_3mm | combined | noise_10mm | delay_40ms | dropout_30pct | Selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo/0 | 2048 | 10/10 | 10/10 | 8/10 | 7/10 | 5/10 | 7/10 | 10/10 | Yes |
| ppo/0 | 4096 | 10/10 | 10/10 | 8/10 | 7/10 | 5/10 | 7/10 | 10/10 | — |
| ppo/1 | 2048 | 10/10 | 10/10 | 8/10 | 7/10 | 5/10 | 7/10 | 10/10 | — |
| ppo/1 | 4096 | 10/10 | 10/10 | 8/10 | 7/10 | 5/10 | 8/10 | 10/10 | Yes |
| ppo/2 | 2048 | 10/10 | 10/10 | 8/10 | 7/10 | 5/10 | 7/10 | 10/10 | — |
| ppo/2 | 4096 | 10/10 | 10/10 | 8/10 | 7/10 | 5/10 | 7/10 | 10/10 | Yes |

Learning curves use the same validation scenes; shaded bands show range across training seeds, not confidence intervals on success rate.
Larger budgets do not guarantee improvement, and checkpoint selection does not imply convergence.


## Nominal Controller Verification

The task is single-ball reception only, excluding active throws or the complete three-ball cycle.

Each perception condition uses 20 scene seeds: [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019].

| Policy | baseline |
| --- | --- |
| zero | 20/20 |
| static | 7/20 |

zero is the nominal reception feedback controller; static disables landing prediction while keeping identical grasping and cushioning.
Learned policies are pooled across all training seeds; training seeds share the same test scenes and are not independent scene samples.
Policy evaluation disables exploration; failure reasons, catch evidence, and action metrics are stored in the raw JSON.


## Feedback Diagnostics

| Policy/Mode | Case | Success |
| --- | --- | --- |
| zero / nominal_controls | baseline | 4/5 |
| zero / nominal_controls | combined | 2/5 |
| static / nominal_controls | baseline | 1/5 |
| static / nominal_controls | combined | 1/5 |
| zero / nominal_sensor_blackout | sensor_blackout | 1/5 |
| ppo_seed0 / unmodified | baseline | 4/5 |
| ppo_seed0 / unmodified | combined | 2/5 |
| ppo_seed0 / masked_ball | baseline | 4/5 |
| ppo_seed0 / masked_ball | combined | 2/5 |
| ppo_seed0 / frozen_initial | baseline | 4/5 |
| ppo_seed0 / frozen_initial | combined | 2/5 |
| ppo_seed1 / unmodified | baseline | 4/5 |
| ppo_seed1 / unmodified | combined | 2/5 |
| ppo_seed1 / masked_ball | baseline | 4/5 |
| ppo_seed1 / masked_ball | combined | 2/5 |
| ppo_seed1 / frozen_initial | baseline | 4/5 |
| ppo_seed1 / frozen_initial | combined | 2/5 |
| ppo_seed2 / unmodified | baseline | 4/5 |
| ppo_seed2 / unmodified | combined | 2/5 |
| ppo_seed2 / masked_ball | baseline | 4/5 |
| ppo_seed2 / masked_ball | combined | 2/5 |
| ppo_seed2 / frozen_initial | baseline | 4/5 |
| ppo_seed2 / frozen_initial | combined | 2/5 |


Masking policy ball state affects direct input fields only (nominal reference still carries ball info); frozen policy output still retains the classical controller.


ppo_seed0: normal policy 6/10, frozen initial output 6/10; both retain identical nominal feedback. This small sample shows no additional gain from online policy feedback.

ppo_seed1: normal policy 6/10, frozen initial output 6/10; both retain identical nominal feedback. This small sample shows no additional gain from online policy feedback.

ppo_seed2: normal policy 6/10, frozen initial output 6/10; both retain identical nominal feedback. This small sample shows no additional gain from online policy feedback.

## Screening Gate

Every training seed must preserve zero-disturbance catches and paired disturbed totals must exceed the classical controller. This gate does not trigger automatic replacement.


- ppo: passed candidate screening; zero-disturbance per seed [19, 19, 19], classical controller 19; paired disturbed delta +14.

  Safety failures: classical controller 15; per training seed [15, 15, 15]. Any seed increasing safety failures fails.

  Disturbed baseline completions 67/100; per-seed policy completions [72, 72, 71] (out of 100 each). Paired pooling repeats the same baseline 3 times and does not add independent scenes.


This budget and test set cannot establish general algorithm superiority; per-condition degradation and collisions must be checked jointly. The three-ball task is not yet wired to learned policies.


## Safety and Action Metrics

| Policy | Case | Episodes | Forbidden contact | Pose limit | Ball height | Other safety | Task deadline | Mean residual RMS | Max axis offset (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero | baseline | 20 | 1 | 0 | 0 | 0 | 0 | 0.0000 | 0.000 |
| zero | noise_3mm | 20 | 1 | 0 | 0 | 0 | 1 | 0.0000 | 0.000 |
| zero | combined | 20 | 4 | 0 | 0 | 0 | 3 | 0.0000 | 0.000 |
| zero | noise_10mm | 20 | 5 | 0 | 0 | 0 | 3 | 0.0000 | 0.000 |
| zero | delay_40ms | 20 | 2 | 0 | 0 | 0 | 1 | 0.0000 | 0.000 |
| zero | dropout_30pct | 20 | 2 | 0 | 0 | 0 | 1 | 0.0000 | 0.000 |
| ppo | baseline | 60 | 3 | 0 | 0 | 0 | 0 | 0.1087 | 4.157 |
| ppo | noise_3mm | 60 | 3 | 0 | 0 | 0 | 0 | 0.1057 | 4.149 |
| ppo | combined | 60 | 12 | 0 | 0 | 0 | 6 | 0.0927 | 4.036 |
| ppo | noise_10mm | 60 | 15 | 0 | 0 | 0 | 4 | 0.0814 | 3.935 |
| ppo | delay_40ms | 60 | 6 | 0 | 0 | 0 | 3 | 0.0589 | 3.861 |
| ppo | dropout_30pct | 60 | 6 | 0 | 0 | 0 | 0 | 0.1046 | 4.108 |


## Post-Contact Estimation

Errors are computed over valid estimation steps only; ground truth is diagnostic and never fed to policies. Episodes without valid estimates are not counted as zero error.

| Policy | Case | Valid estimate steps | Position RMSE (mm) | Velocity RMSE (m/s) |
| --- | --- | --- | --- | --- |
| zero | baseline | 260 | 2.215 | 0.194 |
| zero | noise_3mm | 291 | 5.397 | 0.266 |
| zero | combined | 273 | 12.283 | 0.370 |
| zero | noise_10mm | 315 | 13.945 | 0.436 |
| zero | delay_40ms | 160 | 12.213 | 0.340 |
| zero | dropout_30pct | 266 | 5.752 | 0.259 |
| ppo | baseline | 780 | 2.254 | 0.197 |
| ppo | noise_3mm | 807 | 5.427 | 0.261 |
| ppo | combined | 777 | 12.664 | 0.380 |
| ppo | noise_10mm | 859 | 13.449 | 0.409 |
| ppo | delay_40ms | 483 | 12.293 | 0.343 |
| ppo | dropout_30pct | 757 | 5.937 | 0.266 |

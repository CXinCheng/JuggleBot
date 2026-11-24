# Perception Disturbance Evaluation Results

Requested 30 complete flights per trial; test seeds: 0, 1.
Controller parameters are fixed, using position sampling and three-point state estimation. Noise is independent per-axis Gaussian noise, delay is a fixed FIFO delivery delay, and dropout drops whole frames independently.
Contacts, safety checks, and private planning rollouts use simulation ground truth. These results are a simulation sensitivity study; disturbance ranges are not calibrated to real hardware sensors.

| Case | Noise sigma (mm) | Delay (ms) | Dropout prob | Success/Trials | Mean completed flights | Total drops | Total ball-ball collisions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0 | 0 | 2/2 | 30.0 | 0 | 0 |
| noise_1mm | 1 | 0 | 0 | 0/2 | 0.0 | 0 | 0 |
| noise_3mm | 3 | 0 | 0 | 0/2 | 0.0 | 0 | 0 |
| noise_5mm | 5 | 0 | 0 | 0/2 | 0.0 | 0 | 0 |
| noise_10mm | 10 | 0 | 0 | 0/2 | 0.0 | 0 | 0 |
| delay_10ms | 0 | 10 | 0 | 2/2 | 30.0 | 0 | 0 |
| delay_20ms | 0 | 20 | 0 | 0/2 | 5.0 | 0 | 0 |
| delay_40ms | 0 | 40 | 0 | 0/2 | 8.0 | 0 | 0 |
| dropout_5pct | 0 | 0 | 0.05 | 0/2 | 24.5 | 0 | 0 |
| dropout_15pct | 0 | 0 | 0.15 | 1/2 | 27.0 | 0 | 0 |
| dropout_30pct | 0 | 0 | 0.3 | 0/2 | 2.5 | 0 | 0 |
| combined | 3 | 20 | 0.1 | 0/2 | 0.0 | 0 | 0 |

Trial counts per condition are limited; success counts summarize this experiment and are not an accurate estimate of population success probability.
Failures from planning rejections, posture limit violations, and expired measurements are retained in per-trial records. Estimation error is computed only over valid online airborne-ball estimates sampled per physics step; invalid estimates are counted separately.

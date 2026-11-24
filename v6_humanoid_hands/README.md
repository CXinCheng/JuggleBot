# v6 — G1 Upper Body and Full-Size Dual LEAP Hands

A fixed-base Unitree G1 upper body equipped with official left and right LEAP Hand models performing a three-ball cascade. Each hand has four fingers and 16 joints with original dimensions, mass, inertia, and joint limits; each ball has an 85 mm diameter and 120 g mass.

The controller uses elliptical inward scoops, downward catch cushioning, and contact-based holding. Balls are driven purely by gravity and physical contact without grasp attachment constraints.

## Running

From the repository root:

```bash
uv sync
uv run python v6_humanoid_hands/juggle.py --pattern cascade3 --viewer --infinite
uv run python v6_humanoid_hands/juggle.py --pattern cascade3 --throws 30
uv run python v6_humanoid_hands/juggle.py --pattern cascade3 --throws 30 --video videos/v6_cascade3.mp4
```

`--viewer --infinite` runs continuously in the viewer; `--throws 30` validates 30 complete flights; `--video` records an [MP4 video](../videos/v6_cascade3.mp4) at 1280×720, 60 fps (add `--speed 0.5` for half speed; overwrites existing file).

One complete flight consists of physical release followed by confirmed catch. Total releases may exceed `--throws` in order to catch the final target flight. Because the planner performs forward physics rollouts, wall-clock time can significantly exceed simulated time.

## Verification

```bash
uv run python -u v6_humanoid_hands/verify_g1_hands.py --log v6_humanoid_hands/results/assembly.txt
uv run python -u -m v6_humanoid_hands.test_cradle_contact -v --log v6_humanoid_hands/results/validation.txt
uv run python -m unittest v6_humanoid_hands.test_evaluation -v
```

These three commands check model assembly and FK/IK, contact and continuous juggling, and evaluation/logging utilities. `--log` writes and overwrites the specified log; omitting it runs checks without saving. Failures return a non-zero exit code.

| Check | Default Reference Result |
| --- | --- |
| Assembly, original physical parameters, and FK/IK | PASS |
| Contact and juggling regression | 6/6 PASS |
| Evaluation and logging unit tests | 13/13 PASS |
| Three-ball 30/50 complete flights | All caught, 0 drops, 0 ball-ball collisions |

The continuous juggling test checks ball height, collisions, contact dynamics, catch support, cushioning deceleration, and arm posture at every physics step. See [test_cradle_contact.py](test_cradle_contact.py) for exact assertions.

## Local Noise Experiment

`--obs-noise` adds Gaussian noise (in meters) to selected catch-position predictions; `--seed` sets the RNG seed (default 0). Example at 5 mm standard deviation:

```bash
uv run python v6_humanoid_hands/juggle.py --pattern cascade3 --throws 30 --obs-noise 0.005 --seed 1
```

This perturbation applies only to selected catch-position predictions; ball velocity and private rollout initial states still use simulation truth. It does not model full sensor sampling, delay, dropout, or actuator error.

Run the 10-trial sweep (5 noise levels × 2 seeds):

```bash
uv run python -m v6_humanoid_hands.evaluate_noise --noise-mm 0 1 3 5 10 --seeds 0 1 --throws 30 --output v6_humanoid_hands/results/noise_sensitivity.jsonl
```

The output JSONL path must not already exist. Runs serially by default (`--jobs 2` enables two workers). `--timeout` limits simulated time per run, not wall-clock time.

After completion, generate the summary and CSVs from the same JSONL:

```bash
uv run python -m v6_humanoid_hands.summarize_noise --input v6_humanoid_hands/results/noise_sensitivity.jsonl --output v6_humanoid_hands/results/noise_sensitivity.txt --csv-output v6_humanoid_hands/results/noise_trials.csv --execution-output v6_humanoid_hands/results/execution_demand.csv --overwrite
```

Summarization reads the existing JSONL without running simulations and can be run directly on the archived data. `--overwrite` replaces the target TXT/CSV files; both CSV flags are optional.

### Measured Results

Two seeds per noise level, 30 requested flights per run, default serial settings (`--jobs 1`):

| Position Noise Std | Seed 0 Flights | Seed 1 Flights | Successes | Termination Reason |
| --- | --- | --- | --- | --- |
| 0 mm | 30 | 30 | 2/2 | Both completed |
| 1 mm | 4 | 4 | 0/2 | Both reception planning rejection |
| 3 mm | 4 | 21 | 0/2 | Both reception planning rejection |
| 5 mm | 7 | 13 | 0/2 | Both reception planning rejection |
| 10 mm | 1 | 25 | 0/2 | Seed 0 planning rejection; Seed 1 drop |

Seven noisy runs stopped on `NO_FEASIBLE_RECEPTION`. Under 10 mm noise, seed 1 dropped below the 0.40 m threshold (≈ 0.398 m) at 9.071 s. All 10 runs had 0 ball-ball collisions.

With only two seeds per level, results do not estimate population success rates, and completed flights are not monotonic in noise magnitude. Zero noise yields identical trajectories across both seeds.

## Result Files

The commands above write to `v6_humanoid_hands/results/`:

| File | Contents |
| --- | --- |
| `assembly.txt` | Model assembly, physical parameters, FK/IK, and initial contact checks |
| `validation.txt` | Zero-noise contact and 30/50-flight regression log |
| `noise_sensitivity.jsonl` | Complete 10-trial noise record for verification and summary/CSV regeneration |
| `noise_sensitivity.txt` | Noise configuration, per-trial summary, stop reasons, and source/plan fingerprints |
| `noise_trials.csv` | Per-trial settings, flight counts, success flags, drops/collisions, and stop events |
| `execution_demand.csv` | Execution statistics across 49 joints for the zero-noise 30-flight baseline |

`execution_demand.csv` corresponds to **0 mm, seed 0, 30 complete flights**, reporting peak joint speed, peak/RMS actuator torque (converted from actuator force via gear ratio, excluding passive/constraint forces), and fraction of steps within 1% of either force limit. It does not include 50-flight per-joint data.

## Model and Control

| Parameter | Default |
| --- | --- |
| Robot | Fixed-base G1 upper body, 7 DoF/arm, dual LEAP Hands |
| Ball | 85 mm diameter, 120 g mass |
| Physics step & solver | 1 ms, `implicitfast`, elliptic friction cone, `impratio=100` |
| Alternating throw interval `tau` | 0.32 s |
| Target separation-to-touch `flight` | 0.53 s |
| LEAP finger torque limits | ±0.5 N·m |
| Hand gravity compensation | `gravcomp=1` |

Arms recover from outer catch positions, accelerate inward through a scoop, and cushion downward at reception while fingers close based on relative position, gap, and contact force. Confirmed catches require sustained palm-finger support and decaying relative speed. Live runs and private rollouts enforce collision, joint-range, and posture checks.

Implementation lives in [src/jugglebot/humanoid](../src/jugglebot/humanoid/): `model.py` (assembly), `kinematics.py` (kinematics), `cascade.py` (planning & control), `throwing.py` (finger control). Assets: [Unitree G1](../assets/unitree_g1/) and [LEAP Hand provenance](../assets/leap_hand/UPSTREAM.json).

### Recalibration (Optional)

Normal runs use `src/jugglebot/humanoid/leap_cascade_seed_plans.json` and require no recalibration. After modifying model or control parameters, export new seed plans with:

```bash
uv run python v6_humanoid_hands/calibrate.py --balls 3 --throws 50 --timeout 35 --output <output_path>
```

Exported files are not applied automatically; review them, replace `leap_cascade_seed_plans.json`, and rerun the full regression. `--initial-plans <file>` sets starting warm plans (`forecast` and `execution`). Throw plans contain four motion parameters plus optional `finger_release_lead_s` (±25 ms, default 0).

## Experimental Scope

Formal acceptance is zero-noise three-ball cascade starting with two balls in hand and one in flight (catching the initial airborne ball is not counted). The base is fixed with no locomotion or balance control.

Compared to v5, v6 uses the G1 upper body, official left/right hands, and exact-state input; v5 uses mirrored UR5e arms and sampled position estimation. v5 counts throws per hand; v6 counts global complete flights. `--tau` and `--flight` override defaults, but archived results apply only to defaults. Finger gains, torque limits, contact compliance, gravity compensation, mounting brackets, torque-speed curves, and thermal limits have not been identified on real hardware; simulation passes do not validate hardware deployment.

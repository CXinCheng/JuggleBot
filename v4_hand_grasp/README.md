# v4 — Stage 4: UR5e + LEAP Dexterous Hand (E1 Grasp/Release, E2 Timed-Open Throw, E3 Catch & Self-Juggle, Bowl Posture, E4 Whip Throw, E5 Single-Hand Three-Ball Cascade)

Dual-arm cup experiments (v3) were paused (see `git stash list`) and development moved directly to a dexterous hand. v4 addresses:

- **E1**: Can a LEAP Hand mounted on a UR5e hold a juggling-sized ball under throw-level accelerations and release it when open?
- **E2**: Can opening the fingers at a scheduled time (`t_open`, `T_open`) control throw velocity?
- **E3a**: Catching balls tossed by a launcher with the LEAP Hand.
- **E3b**: Single-arm, single-ball two-point self-throw/catch (hand version of v2 Experiment B).
- **E3c**: Human-like "bowl" hand posture (knuckles lowest, palm heel and fingers raised) compared against a flat palm.
- **E4**: How much release speed does whip throwing add when parallel arm joints rotate together?
- **E5**: Can one arm play one hand of a three-ball cascade (with the other hand replaced by a mirrored launcher)?

Classical control only (no RL).

## Files

| File | Contents |
|---|---|
| `ur5e_leap.xml` | Menagerie UR5e (same as v1/v2) with Menagerie right LEAP Hand attached at flange (prefix `leap_`). Sites: `ball_seat` (122 mm along flange axis), `catch_point` (+25 mm); `bowl_seat`, `bowl_point` placed by `handsim.set_bowl_sites()` |
| `scene.xml` | Floor, cameras (`close`, `throw`, `loop` for E3b), ball; `cone="elliptic"`, `impratio="100"` (recommended Menagerie LEAP settings) |
| `handsim.py` | Simulation wrapper (`load_model(extra_balls=)` for E5): sets hand `gravcomp`, finger torque limits, compliant palm/fingertip contacts, optional `pedestal` (E3b); PD + inverse-dynamics arm tracking; hand poses; contact classification; 180°/s guard; `LineStroke` |
| `kinematics.py` | v2 IK/FK with selectable target site (5 DoF: position + palm normal; spin handled in null space) |
| `prediction.py`, `planning.py` | v1/v2 ballistic fit and catch planner, adding `cushion_dir="down"`, `"direct"` strategy (E5: single quintic from moving arm without stopping), and `fit_with_prior()` (E3b) |
| `grasp_test.py` .. `hand_cascade.py` | E1 (`grasp_test.py`), E2 (`hand_throw.py`), E3a (`hand_catch.py`), E3b (`hand_juggle.py`), E4 (`whip_test.py`), E5 (`hand_cascade.py`) |
| `video.py` | v2 video recorder (fixed frame padding when recording starts at `t > 0`) |
| `results/` | Raw outputs, `postures.png`, `hand_throw_frames.png` |
| `../assets/leap_hand/` | Menagerie `leap_hand` (commit `0059d43`, MIT license) meshes, `LICENSE`, and original `right_hand.xml` (`meshdir="."`) |

## Running (from the repository root)

```bash
uv run python v4_hand_grasp/grasp_test.py --a-up 2 3 --a-down 0.5 1.5 3
uv run python v4_hand_grasp/grasp_test.py --images v4_hand_grasp/results     # Save posture screenshots
uv run python v4_hand_grasp/hand_throw.py                                    # E2 sweep (~3 min)
uv run python v4_hand_grasp/hand_throw.py --tilt 20 --heading 90 --t-open -0.02 0.02 --T-open 0.03
uv run python v4_hand_grasp/hand_throw.py --t-open 0.02 --T-open 0.03 --trials 1 --video videos/
uv run python v4_hand_grasp/hand_catch.py --trials 20                        # E3a/E3c: default bowl posture
uv run python v4_hand_grasp/hand_catch.py --trials 20 --hand flat            # E3a: flat palm
uv run python v4_hand_grasp/hand_catch.py --trials 20 --hand flat --obs-noise 0.005 --cushion-dir ball --d-pre 0.05 --d-cush 0.10
uv run python v4_hand_grasp/hand_catch.py --fingers close --obs-bias 0.03    # 3 cm artificial landing bias
uv run python v4_hand_grasp/hand_juggle.py --obs-noise 0.01                  # Self-juggle, default bowl
uv run python v4_hand_grasp/hand_juggle.py --hand flat --obs-noise 0.005     # E3b: flat palm
uv run python v4_hand_grasp/hand_juggle.py --runs 1 --cycles 6 --video videos/hand_juggle.mp4
uv run python v4_hand_grasp/whip_test.py --poses 5 --a-up 2 3                # E4 (~15 min)
uv run python v4_hand_grasp/whip_test.py --pose 0.9 0.3 --pose 0.9 0.2 --a-up 3 --a-down 2.5 4 5
uv run python v4_hand_grasp/hand_cascade.py --tau 0.35 0.37 0.39 --runs 2
uv run python v4_hand_grasp/hand_cascade.py --launch-noise 0.03 --obs-noise 0.005 --runs 5 --throws 50 --seed 1
uv run python v4_hand_grasp/hand_cascade.py --runs 1 --throws 8 --video videos/hand_cascade.mp4
```

## Ball and Contact Model

| | Human Hand | LEAP Hand |
|---|---|---|
| Scale | 1 | ≈ 1.3× human hand (Shaw et al., RSS 2023) |
| Juggling ball diameter | 65–67 mm (standard beanbag) | **85 mm** (= 65 mm × 1.3) |
| Ball mass | 120 g | **120 g** (LEAP grip pull-out force 19.5 N is comparable to human hand, so mass is not scaled by volume) |

- **Ball**: Standard ball with v0–v2 contact parameters (`solref=(0.003, 0.5)`).
- **Palm and fingertips** (`leap_palm_collision_*`, `leap_*_tip`): Replaced Menagerie's near-rigid `solref=0.0001` with a compliant damped pad `solref=(0.006, 1.2)`, `solimp=(0.9, 0.95, 0.004)`, `solmix=100` so hand compliance dominates ball-hand contacts.

## Model Settings

| Item | Value | Notes |
|---|---|---|
| Hand mass | 0.746 kg (model) | Hardware 595 g (paper); model is ≈ 25% heavier (conservative) |
| Finger actuators | Position servo kp = 3, **torque limit ±0.5 N·m** | Menagerie model omits torque limits; 0.5 N·m is a conservative estimate for Dynamixel XC330-M288-T at 600–750 mA (**to be verified**) |
| Gravity compensation | `gravcomp=1` on all hand bodies | Matches arm controller payload gravity compensation |
| Finger geometry | Index-to-ring base span 91 mm; finger length ≈ 0.11–0.12 m | Measured from model |

## Hand Postures

| Posture | Joint Targets | Notes |
|---|---|---|
| `flat` | All 0 | Reference; open target for throwing |
| `cradle` | Fingers mcp = pip = 0.8, dip = 0.48; thumb `th_cmc = 1.2` | Forms walls on finger, thumb, and palm-heel sides (6–34 mm clearance); **ring-finger side is open** (no pinky on LEAP) |
| `grasp` | Fingers 1.2 / 1.2 / 0.72; thumb `th_cmc = 1.2`, `th_mcp = 0.5` | Overdriven past contact so servos apply torque limit; tighter `th_cmc` (1.6–1.8) squeezes the ball out the open side |
| `catch` | Same as `cradle` with fingers bent only 0.3 | Upright cradle fingers deflect balls arriving obliquely from the fingertip side (E3a) |
| `bowl` (E3c, default) | Fingers MCP 1.1, PIP 0.3, thumb as `cradle`, palm heel raised 30°; closes to MCP 1.4, PIP 1.2 on touch | Knuckles lowest; ball settles into palm-finger crease |

## E1: Grasp / Release (`results/grasp_test.txt`)

Vertical linear translation from `ready` (palm up) accelerating at `a_up` to 0.9 m/s and decelerating at `a_down`. Separation > 20 ms counts as `THROWN`, else `HELD`.

| Posture | a_up | a_down = 0.5 g | a_down = 1.5 g | a_down = 3 g | Ball Slip While Held | Finger Torque / Limit |
|---|---|---|---|---|---|---|
| `flat` | 2–3 g | HELD | THROWN (≈ 105 ms) | THROWN (≈ 147 ms) | 2–4 mm | 0.33–0.50 |
| `cradle` | 2–3 g | HELD | THROWN (≈ 106 ms) | THROWN (≈ 148 ms) | 2–4 mm | 0.19–0.28 |
| `grasp` | 2–3 g | HELD | **HELD** | **HELD** | 1.6–1.7 mm | 1.00 (at limit by design) |

Peak arm joint speed 0.88–0.90 × 180°/s; peak arm actuator force 0.74 × limit (`shoulder_lift`), `wrist_1` 0.41 × limit. **Conclusion**: `grasp` holds 120 g through 3 g acceleration/deceleration; `flat`/`cradle` release above 1 g deceleration.

## E2: Timed Finger-Opening Throws (`hand_throw.py`)

Linear flange stroke (`LineStroke`, 3 g up / 3 g down, V = 1.7 m/s, arm joint speed 0.82 × 180°/s), interpolating from `grasp` to `flat` starting at `t_open` (relative to peak speed) over `T_open`. Random perturbations: ±3 mm ball offset, ±3 ms timing jitter, ±0.02 rad finger targets (10 trials/setting).

**Vertical throws (`results/hand_throw_vertical.txt`, mean ± std)**:

| t_open | T_open | Release (after peak) | Ball / Hand Speed (m/s) | Direction Error | Spin (rad/s) | Apex (mm) |
|---|---|---|---|---|---|---|
| −40 / −20 ms | 10–60 ms | 8–15 ms | 1.63–1.69 ± 0.01–0.03 / 1.62–1.70 | 1.4–3.4° | 0.9–2.4 | 135–146 ± 2–5 |
| 0 ms | 10 / 30 / 60 ms | 10 / 14 / 21 ms | 1.80 / 1.70 / 1.60 ± 0.04–0.07 (hand 1.65 / 1.60 / 1.50) | 2–3° | 2.4–3.8 | 164 / 147 / 130 ± 7–12 |
| +20 ms | 10 / 30 / 60 ms | 29 / 35 / 43 ms | 1.58 / 1.38 / 1.16 ± 0.06–0.07 (hand 1.30 / 1.15 / 0.94) | 2–4° | 2.6–4.3 | 127 / 96 / 68 ± 9–10 |
| +40 ms | 10 / 30 / 60 ms | 48 / 55 / 63 ms | 1.05 / 0.88 / 0.61 ± 0.06–0.09 (hand 0.77 / 0.57 / 0.38) | 6–7° | 3.1–5.3 | 56 / 40 / 19 ± 4–8 |

**Findings**: Opening before peak (`t_open < 0`) releases naturally at > 1 g deceleration; opening at/after peak monotonically modulates speed (1.8 → 0.6 m/s); compliant pad spring-back adds 0.1–0.3 m/s over hand speed with 2–5 rad/s spin and 2–7° angular error.

**Tilted throws (20° from vertical, `results/hand_throw_tilt20.txt`)**: 10/10 clean releases in all four palm directions (ring-side 1.7 m/s, thumb-side 1.7 m/s, fingertip 1.0 m/s, palm-heel 1.2 m/s), direction error 1.6–8.7°.

## E3a: Catching Launched Balls (`hand_catch.py --hand flat`, `results/hand_catch_compare.txt`)

v1 launcher (4.4–5.8 m/s at catch), 20 trials/setting:

| Arm Motion | Fingers | Zero Noise | 5 mm Noise | Relative Impact Speed (mean / max) | Max Penetration |
|---|---|---|---|---|---|
| Downward cushion | Close on touch | **20/20** | 13/20 | 3.04 / 3.54 m/s | 10.7 mm |
| Downward cushion | Passive | 0/20 | 0/20 | Same | — |
| Along-ball cushion | Close on touch | 20/20 | **19/20** | 3.10 / 3.55 m/s | 11.7 mm |
| Static wait | Close on touch | 20/20 | 17/20 | 3.46 / 3.83 m/s | 12.4 mm |

Active finger closure is required (passive 0/20 vs active 20/20); downward cushioning cuts impact speed by ≈ 12% but trails along-ball cushioning under 5 mm noise (13/20 vs 19/20).

## E3b: Single-Arm Two-Point Self-Juggling (`hand_juggle.py --hand flat`, `results/hand_juggle_noise*.txt`)

0.4 m pedestal, points A/B 15 cm apart (`F = 0.4 s`), 10 runs × up to 20 catches with prior-weighted ballistic fit (`fit_with_prior`):

| Observation Noise | Full 20-Catch Runs | Failures | Release Speed Error (mean / max) | Relative Contact Speed (mean / max) | Replans / Catch | Dwell |
|---|---|---|---|---|---|---|
| 0 | **10/10** | — | 45 / 144 mm/s | 1.4 / 1.9 m/s | 1.2 | 0.84 s |
| 5 mm | **10/10** | — | 47 / 144 mm/s | 1.3 / 1.9 m/s | 2.6 | 0.86 s |
| 10 mm | **10/10** | — | 52 / 144 mm/s | 1.3 / 1.9 m/s | 2.8 | 0.86 s |

Peak joint speed 0.91 × 180°/s, peak arm force 0.94 × limit. Long test (20 runs × 50 catches, `results/hand_juggle_long_*.txt`): rotating the hand 90° (gap perpendicular to throw plane) reduced `NOT_CAUGHT` gap-side slips from 4/1807 to 0/1903 (1/952 `DROPPED` at both 0 and 10 mm noise from a 313 mm/s release outlier; per-catch failure rate ≈ 0.1–0.5%). Release-speed error is ≈ 3× larger than the v2 cup (45–52 mm/s vs 11–15 mm/s).

## E3c: Bowl Posture (`--hand bowl`, default)

Raised palm heel (30°) with fingers MCP 1.1 / PIP 0.3 (closing to 1.4 / 1.2 on touch):
- Static capture grid (154 landing spots at 1.5 / 2.5 m/s): flat 120 / 114 vs **bowl 144 / 138**.
- E3a launched balls: 20/20 at 5 mm noise (vs 13/17 flat), impact speed 1.9–2.3 m/s (vs 3.0–3.1), penetration 6–7 mm (vs 11 mm); 4 cm bias catches doubled (11/19, 9/18 vs 5/15, 4/13).
- E3b self-juggling: **0/1000** failures in long 10 mm noise test (vs 1/952 flat); mean release-speed error **42–44 mm/s**; under 2 cm / 3 cm bias, full runs rose from 0/10 to **7/10** (3/163 failures) and **8/10** (2/164 failures).

## E4: Whip Throwing (`whip_test.py`, `results/whip_test.txt`, `results/whip_test_decel.txt`)

Within 0.9 × 180°/s and 0.9 × feedforward torque limit:
- `translate`: 1.83 m/s (`F = 0.42–0.46 s`), cascade margin ≈ 0.46 s.
- `linewhip` (3 g / 4 g): **2.99–3.16 m/s** (`F = 0.64–0.70 s`, 1.8–1.9° error, spin 2.5–4.8 rad/s), cascade margin ≈ 0.86–0.90 s — **selected for cascade**.
- `jointwhip`: up to 3.91 m/s (`F = 0.82–0.84 s`), limited by floor clearance and 5–7 rad/s spin.

## E5: Single-Hand Three-Ball Cascade (`hand_cascade.py`, `results/hand_cascade_*.txt`)

Mirrored virtual launcher `O` (`F = 0.6 s`, `|w| = 0.15 m`, `direct` catch strategy from moving arm):
- Zero noise (2 runs × 20 throws): `τ = 0.35, 0.37 s` achieved 40/40; `τ = 0.39 s` failed first catch (`NO_CATCH_PLAN`).
- With noise (`σ_launch = 30 mm/s`, 5 mm obs noise, 5 runs × 50 throws): `τ = 0.36 s` completed 3/5 (194 catches, 2 LATE; ≈ 1% per-throw failure); `τ = 0.37 s` across 3 seeds had ≈ 2.5% per-throw failure (10/406); `τ = 0.38 s` failed on first catch. Peak arm force reached 1.00 × limit (zero margin).

## Limitations / Open Items

- Finger torque limit (0.5 N·m) is estimated and needs hardware verification; continuous stall clamping would overheat real servos.
- Compliant pad parameters (6 ms / 1.2) are uncalibrated against physical materials.
- E5 uses an ideal mirrored launcher with fixed timing, stops between cushion/carry/stroke, and hits 1.00 × arm actuator force.

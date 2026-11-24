# v7 — "2+1" Three-Ball Start & Ending Feasibility Study

Goal: On top of v6 (fixed-base G1 upper body + dual LEAP Hands three-ball cascade), start with **all three balls in hand (2 in right hand, 1 in left hand)** and **end by catching all three balls back into hand (2+1)** without modifying v6 controller source code.

> Scope (2026-11-11): **Right hand only** holds/catches two balls; left hand always holds one ball. All results are simulation-only and do not constitute hardware validation.

## 1. Summary of Findings

| Question | Finding | Margin |
| --- | --- | --- |
| Can one hand stably hold two 85 mm balls? | **Yes**, with ball A in the palm and ball B pinched by index/middle/ring fingertips (≈ 24–28 N); resting B in the wrist-housing groove is **unusable** | Displacement ≤ 1 mm under 8 m/s² lateral, 12 m/s² down, 20 m/s² up pulses and 5°–18° tilt |
| Can A be thrown while retaining B? | **Yes** (v6 stroke replay): B drift ≈ 2 mm, A releases at ≈ 0.90 s | World release velocity (−0.30, 0.65, 2.4) vs v6 (−0.08, 0.55, 2.1) m/s; needs recalibration |
| Can B reseat into v6 cradle during recovery? | **Feasible but narrow**: random-timing success ≈ 2% (6/360, 7/360) | Eliminated by the human-like role swap in Sec. 8 |
| Ending: catch 2nd ball while holding 1st (wrist groove) | **Partially feasible**: β settles in palm across all offsets; α stays in groove in 9/25 cases within ±1 cm, fails at ±2 cm | Lateral acceleration (≈ 8 m/s²) dislodges α |
| Active palm→fingertip transfer of held ball | **Failed** (0/2160+ CMA samples; Sec. 11.4) | — |
| Human-like ending (α in palm held by ring finger + heel, β caught by index/middle/thumb, Secs. 12 & 14) | **Feasible**: static 0.000 m slip; dynamic robust within ±1 cm at 2–3.5 m/s | Integrated end-to-end in Sec. 15 (`stop_after` = 4, 6, 8, 12) |
| Left hand | **Not validated** | — |

## 2. Single-Hand Test Rig (`feaslib.py`)

Reuses `jugglebot.humanoid.model.load_g1_dual_arm` (3 balls) and `LeapFingers`; only one hand's 16 finger joints and the balls are free while the remaining joints are held at `Q_READY_ARM` via stiff servos (`kp=1500`, waist `2000`) with `gravcomp=1` on all non-ball bodies, so varying `opt.gravity` applies effective hand acceleration `g_eff = g − a_hand` in the seat frame (`x` lateral, `y` toward fingertips, `z` palm normal; nominal ball center `(0, 0, 0.022)`).

## 3. Pipeline Scripts and Example Commands

All commands run from the repository root (`uv run python ...`):

| Stage | Script(s) | Key Outputs | Section |
| --- | --- | --- | --- |
| 1–2. Static / multi-condition 2-ball hold | `search_hold2.py`, `robust_hold2.py`, `search_hold2b.py` | `results/hold2_*.json`, `hold2b_right_s0.json` | Sec. 4.1 |
| 3–5. v6 stroke recording, replay, clamped-B search | `record_v6_throw.py`, `replay_throw.py`, `search_hold2c.py` | `results/v6_right_throw.npz`, `hold2c_*.json`, `hold2d_*.json` | Sec. 4.2 |
| 6. B reseating | `reseat.py`, `reseat_search.py` | `results/hold2d_right_s1_1*_reseat_good.json` | Sec. 4.3 |
| 7. Wrist-groove ending catch | `end_catch.py`, `end_catch2.py`, `search_end.py` | `results/end_hold_right_s0.json` | Sec. 4.4 |
| 8. Human-like start (throw fingertip ball T, retain palm ball R) | `search_hold2e.py` | `results/hold2f_right_s20.json`, `hold2f_right_s21.json` | Sec. 8 |
| 9–10. Fingertip-held ending & arm reachability | `end_tip_catch.py`, `search_end_tip*.py`, `end_tip2_robust.py`, `end_reach.py` | `results/end_tip2_robust.json`, `end_reach_scan.txt` | Secs. 9–10 |
| 11. Full-body 2+1 start integration | `cascade21.py`, `run_cascade21.py`, `calibrate_first_throw.py`, `calibrate_second_throw.py`, `transfer.py` | `results/first_throw_calibration.json`, `second_throw_calibration.json`, `cascade21_*throws.log`, `transfer_best.json` | Sec. 11 |
| 12–14. Pinky/ring-finger palm hold + fingertip catch | `search_pinky.py`, `search_catch_tip*.py`, `catch_tip_robust.py`, `render_start.py`, `record_cascade21.py` | `results/pinky_A.json`, `catchtipM_az0_el90.json`, `cascade21_start.mp4` | Secs. 12–14 |
| 15–18. Full-body 2+1 ending, 30-throw diagnosis, seed recalibration, local noise | `end21.py`, `run_end21.py`, `diag_30.py`, `calibrate21.py`, `noise21.py` | `results/end21*_stop*.log`, `seeds_2plus1.json`, `diag_30_seeded.log`, `noise21_*.jsonl` | Secs. 15–18 |

```bash
uv run python v7_three_ball_hold/run_cascade21.py --throws 8 --timeout 12 --verbose     # ~100 s
uv run python v7_three_ball_hold/run_cascade21.py --throws 30 --timeout 30 --guard     # ~13 min
```

## 4. Detailed Early-Rig Results (Wrist Groove vs Clamped-B)

- **4.1 Groove rejection**: `search_hold2b` showed B rested mainly on `right_wrist_yaw_link` (1.81 N) rather than the hand (0.11 N) and flew out with A during deceleration. Any retained ball must be actively clamped.
- **4.2 Clamped-B (`hold2d_right_s1_10/11`)**: B held at ≈ 24–28 N by index/middle/ring tips; pulse/tilt slip ≤ 1 mm; stroke replay drift 1.8–2.0 mm; A releases at 0.898–0.901 s. Static ball-ball contact force ≈ 5.5 N conflicts with v6's zero-ball-contact check.
- **4.3 Reseating**: 6/360 (`s1_10`) and 7/360 (`s1_11`) random-timing successes; best samples 5–8 mm error.
- **4.4 Groove ending catch**: Fingertip-held α collides with β along v6's approach (`end_catch.py`); wrist-groove α survives 9/25 offsets (±1 cm mostly OK, ±2 cm fails) while β settles in the palm in all 25 cases. Note: v6's empty-hand recovery (0.912–1.0 s) produces +25 m/s² palm-normal effective gravity, lifting any unclamped ball off the palm.

## 5. Early-Rig Limitations

Inertial-frame gravity replay omits rotational terms (`ω×ω×r`, `α×r`), stiff servos omit arm tracking error and torso reaction, weights are hand-tuned, single/few seeds are not statistical, and only the right hand was tested.

## 6. Literature Notes (2026-11)

Ploeger & Peters (arXiv 2207.01414) leave start/stop to future work; arXiv 2608.26800 uses manual placement and no planned ending; MultiGrasp (arXiv 2310.15599) and Morgan et al. (arXiv 2201.07928) do not address dynamic 2+1 start/ending.

## 8. Human-Like Role Swap (`search_hold2e.py` → `hold2f_right_s20/s21.json`)

Ball R (retained) sits in the palm cradle (34–41 N, < 2 cm from v6 cradle center) while ball T (thrown) is pinched by index/middle/thumb fingertips (≈ 10 N) and released by opening index/middle fingers over 50 ms at `t_open`:
- **0 ball-ball contacts** during hold and stroke; R drift ≤ 0.9 mm; T releases at 0.875–0.876 s with (−0.080, 0.55, 2.10) m/s matching v6; **no reseating required**.
- Sensitivity: `vz` varies linearly with `t_open` at ≈ 0.04 m/s per ms (≈ 2 cm landing shift per ms at 0.53 s flight).

## 9–10. Fingertip-Held Ending & Arm Reachability (`end_tip*`, `end_reach.py`)

- Catching β into the palm while holding α at fingertips requires opening receiving fingers into a bowl and approaching from directly above (0°/90°, 8/25 at 1.5 m/s) or from the palm-heel side (90°/60°, 16–17/25 at 1.5–2.5 m/s within ±2 cm); v6's actual fingertip-side approach collides with α.
- 7-DoF IK reachability scan (`end_reach.py`, 100 grid points): at `θ = 10°`, `v_rel = 1.5 m/s`, **41%** of grid points pass for the palm-heel 60° approach (concentrated lower and outward: `z = 0.80–0.84`, `y = −0.26..−0.30`), with peak joint speeds 8–12 rad/s (< v6's 14 rad/s).

## 11. Full-Body 2+1 Start Integration (`cascade21.py`)

`TwoPlusOneCascade(ContactCascade)` starts with `hold2f` in the right hand (0.2 s settle delay `SETTLE_S`), then hands off to v6:
- **8 throws**: PASS (8 releases, 6 complete flights, 0 drops, 0 collisions, 2.71 s).
- **30 throws (before seed recalibration)**: stopped at 5.95 s (18 releases, 16 flights) on `NO_FEASIBLE_RECEPTION` (0 drops, 0 collisions).
- Key integration fixes: close fingers from open state (`0.2 s` settle); exempt `(R, T)` pair contact during initial 2-ball hold + 0.12 s post-release while logging it; offline-calibrate first throw (`calibrate_first_throw.py`: `[0.2507, 1.1304, 3.7577, −0.1013]`, `t_open = 0.1125 s`, 2 mm error); keep index/middle open 0.12 s post-release to avoid grazing T; two-segment return (0.28 s empty-hand brake + loaded carry) to stay within wrist pitch limit; calibrate second throw (`calibrate_second_throw.py`).
- **11.4 Palm→fingertip transfer (`transfer.py`)**: 0 successes across 2,160 random + 25 CMA generations (best error 8.9 cm); abandoned in favor of Sec. 12.

## 12–14. Ring-Finger Palm Hold + Fingertip Catch (`pinky_A.json`, `catchtipM_az0_el90.json`)

- **Static (`search_pinky.py`)**: Ring finger (`rf`, acting as pinky) + palm heel clamp ball A (14 N) while index/middle/thumb hold ball B (22 N); **0.000 m** slip across all 6 pulses (variant A). Eliminates palm→fingertip transfer.
- **Dynamic (`search_catch_tip_multi.py`, `catch_tip_robust.py`)**: Multi-condition optimization (1.5–3.5 m/s × ±1 cm) passes 10/10 training conditions; on ±2 cm grid, `az0/el90` achieves 13/25 (1.5 m/s), 19/25 (2.5 m/s), 13/25 (3.5 m/s), 0/25 (0.5 m/s), and 15/15 under ±20° azimuth / ±15° elevation perturbations.
- Start figures & video: `render_start.py` → `results/figs/start_*.png`; `record_cascade21.py` → `results/cascade21_start.mp4` (0.5x, 9 throws, 0 failures).

## 15. Full-Body 2+1 Ending Integration (`end21.py`, `run_end21.py`)

`EndingCascade(TwoPlusOneCascade)` runs `stop_after` throws (even, last throw by left hand), disables live further throws (`ready_for_throw` override on `silent=False` only), transitions right hand to `catchtipM_az0_el90.json` pre-catch pose `Ka`, arrives at predicted landing point 0.10 s early (`ARRIVE_EARLY_S`, zero hand velocity), closes fingers on timed schedule, holds `q_hold` in null space, and checks 0.8 s three-ball support (`v_rel < 0.15 m/s`, local radius `< 0.16 m`, `z > 0.4 m`):

| `stop_after` | `ELBOW_OUT_END_M = 0.12 m` (`end21b`) | Original v6 `0.10 m` (`end21c`) |
| --- | --- | --- |
| 4, 6, 8 | PASS | PASS |
| 10 | PASS | FAIL: `POSE_LIMIT elbow_out=0.100035` (+0.04 mm) |
| 12 | — | PASS |
| 2 | FAIL: `BALL_COLLISION ball0-ball1` (during start transition) | — |

Demo video: `results/end21_ending.mp4` (0.5x, `stop_after=4`). Single deterministic runs only.

## 16–17. 30-Throw Diagnosis & Seed Recalibration (`diag_30.py`, `calibrate21.py`, `seeds_2plus1.json`)

- **Root cause (Sec. 16)**: The 0.2 s shifted 2+1 timeline drifted away from v6's `leap_cascade_seed_plans.json`, causing Throw 18 (left) to take 4 retries (+15 ms delay) on top of an early Throw 17 (−13 ms), shrinking the reception window to 0.15–0.17 s and causing `NO_FEASIBLE_RECEPTION` (4.07 mm palm self-penetration).
- **Recalibration (Sec. 17)**: Fixed-point recalibration (`calibrate21.py`, caching only directly accepted commands, preserving `throw:right:0/2`) yielded `results/seeds_2plus1.json`: **30/30 complete flights** (31 releases, 10.38 s, 0 drops/collisions, 654 s wall time) and all `stop_after = 4/8/12` ending regressions PASS. Pass `seeds_file=None` to reproduce earlier unseeded runs.

## 18. Local Noise Check (`noise21.py`, `results/noise21_*.jsonl`)

Same local noise model as v6 (Gaussian noise on selected catch-position predictions only; true velocity/rollout state; `seeds_2plus1.json`):
- **Cascade (30 flights requested)**: 0 mm PASS (1/1); 1 mm 0/3 (11/18/9 releases, all `NO_FEASIBLE_RECEPTION`); 3 mm 0/3 (19/11/10 releases; seed 2 `DROPPED` ball 0 at 3.774 s); 5 mm 0/3 (15/20/9 releases, all `NO_FEASIBLE_RECEPTION`). Matches v6's own 0/6 failure mode under noise.
- **Ending (`stop_after = 8`)**: 9/9 PASS across 1/3/5 mm × seeds 0/1/2 (note: `_maybe_start_end` used true ball state without noise, so this does not test ending robustness to noisy landing estimates).

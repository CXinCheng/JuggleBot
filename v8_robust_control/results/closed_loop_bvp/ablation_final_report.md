# v8 Disturbance-Rejection Control & RL Improvement: Stagewise Ablation and Gate 1 Diagnostic Report

> **Status Statement: Interim Progress (Final Acceptance Criteria Not Yet Met)**
> - **Two independent conclusions separated explicitly**:
>   1. **Rule-based closed loop reduces unsafe terminations (frozen `ClosedLoopBvpPolicy`, `apex_trim_dvz = -0.15 m/s` + wrist incoming-ball path braking guard)**: Across the 6 held-ball seat-offset levels on the development comparison set (`440000..440009`), aggregate unsafe termination rate dropped from `48/60 (80.0%)` (fixed baseline) and `54/60 (90.0%)` (timing baseline) to **`28/60 (46.7%)`** (3 fewer than the initial `31/60`), mean confirmed catches rose from `1.18` to **`1.20`**, and mean survival duration exceeded both baselines (`1.516 s` vs `1.398 s / 1.257 s`; among 22 unsafe→safe flips, 18 lasted as long or longer and 4 ended earlier, showing no overall trend of trading safety for earlier drops). However, full 30-throw task completions across all 60 disturbed episodes remain **`0`**, and 2 matched paired regressions vs the fixed baseline remain (logged as regressions).
>   2. **RL further improving over the rule-based closed loop**: **Not demonstrated**. Historical `PPO-BC`, `6D-LCB`, and `7D-LCB` failed to demonstrate simultaneous catch gains and safety non-degradation; incremental RL must be trained and tested on the frozen `ClosedLoopResidualEnv` base after Gate 1.
> - **Dataset Scope Corrections**:
>   - Seeds `440000, 440001, 440002, 440005, 440008` in `440000..440009` were previously used by `build/v8_multi_seed_search.py` for 0.25 mm diagnostics; therefore `440000..440009` is downgraded to **Development Comparison Set** rather than an unseen independent test set.
>   - Repository-wide search (`v8_robust_control/`, `v7_three_ball_hold/`, `v6_humanoid_hands/`, `src/`, `build/`) confirmed **`450000..450019`** has **0 historical uses**; it remains strictly sealed as the final unseen test set.
>   - The 7-level sweep covers only `seat_jitter_m` (initial held-ball seat offset); combined perception disturbances (`noise_std_m`, `delay_s`, `dropout`) are evaluated separately in Section 6.3.

---

## 1. Acceptance Checklist (Interim Scope)

| Criterion | Measured Status | Stage Verdict |
| :--- | :--- | :---: |
| **1. Preserve zero-disturbance full task** | At `0.00 mm`, all policies achieve **10/10 success, 30.00 / 30 catches, 0/10 unsafe** on `440000..440009`; `ClosedLoopResidualEnv` is gated by `action_authority(obs)` and unit tests confirm non-zero learned actions execute as `0.0` at `0.00 mm`. | **Supported** |
| **2. Gain under disturbances** | Across 6 seat-offset levels (60 dev episodes): frozen rule closed loop averages **`1.20` catches** (`2.40` at `0.25 mm`) vs `1.00` (fixed) and `0.40` (timing); full 30-throw completions remain **`0/60`**; at `2.00 mm` closed loop averages `0.40` vs timing's `0.70`. | **Partial (thin gain)** |
| **3. Safety non-degradation** | Aggregate unsafe terminations drop from `48/60 (80.0%)` (fixed) and `54/60 (90.0%)` (timing) to **`28/60 (46.7%)`** with no overall early-drop tradeoff; however, matched pairs still have **2 regressions vs fixed (`440006@0.02`, `440000@0.25`)** and **3 vs timing**. | **Aggregate gain, isolated paired regressions** |
| **4. RL improvement** | Best safety comes from the rule closed loop, not RL. Legacy `PPO-BC` (trained at `0.25 mm` bare) hits `10/10` unsafe at `2.00 mm` (worse than fixed `6/10`), and remains `9/10`–`10/10` unsafe at `1.00–2.00 mm` even with safety guards. `6D-LCB` and `7D-LCB` raise catches at low/mid jitter but suffer `88.3%–90.0%` unsafe rates. | **Not demonstrated (pending Gate 2)** |

---

## 2. Single-Candidate Ablation Table on Dev Comparison Set (`440000..440009`)

> All candidates ran in `TaskResidualEnv` / `ClosedLoopTaskEnv(throws=30, horizon_s=12.0, estimator='filtered')` using `v8_robust_control/results/ablation_baselines/task_filtered_reference_final.json`. **Each column corresponds strictly to one frozen version.**

| Seat Offset (`seat_jitter_m`) | Fixed Baseline<br>(Zero Residual) | Timing Baseline<br>(`defer=0.5`) | Behavior Cloning<br>`PPO-BC` (Bare) | Offline RL<br>`6D-LCB` | Offline RL<br>`7D-LCB (v2)` | **Rule Closed Loop**<br>`ClosedLoopBvpPolicy` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00 mm** | `10/10`, `30.00` catches<br>Unsafe: `0/10` | `10/10`, `30.00` catches<br>Unsafe: `0/10` | `10/10`, `30.00` catches<br>Unsafe: `0/10` | `10/10`, `30.00` catches<br>Unsafe: `0/10` | `10/10`, `30.00` catches<br>Unsafe: `0/10` | **`10/10`, `30.00` catches**<br>**Unsafe: `0/10`** |
| **0.02 mm** | `1.00` (max 3)<br>Unsafe: `7/10` | `0.40` (max 1)<br>Unsafe: `9/10` | `1.10` (max 3)<br>Unsafe: `7/10` | `1.20` (max 5)<br>Unsafe: `9/10` | `1.20` (max 5)<br>Unsafe: `9/10` | `0.80` (max 2)<br>**Unsafe: `6/10`** |
| **0.10 mm** | `1.30` (max 3)<br>Unsafe: `8/10` | `0.40` (max 1)<br>Unsafe: `10/10` | `0.60` (max 2)<br>Unsafe: `9/10` | `2.90` (max 8)<br>Unsafe: `9/10` | `2.40` (max 6)<br>Unsafe: `10/10` | `1.30` (max 3)<br>**Unsafe: `4/10`** |
| **0.25 mm** | `2.00` (max 5)<br>Unsafe: `8/10` | `0.10` (max 1)<br>Unsafe: `7/10` | `1.50` (max 4)<br>Unsafe: `7/10` | `2.50` (max 6)<br>Unsafe: `9/10` | `1.90` (max 6)<br>Unsafe: `8/10` | `2.40` (max 5)<br>**Unsafe: `4/10`** |
| **0.50 mm** | `1.10` (max 3)<br>Unsafe: `10/10` | `0.60` (max 1)<br>Unsafe: `9/10` | `1.00` (max 5)<br>Unsafe: `7/10` | `1.20` (max 3)<br>Unsafe: `8/10` | `1.60` (max 5)<br>Unsafe: `9/10` | `1.60` (max 5)<br>**Unsafe: `4/10`** |
| **1.00 mm** | `0.40` (max 1)<br>Unsafe: `9/10` | `0.20` (max 1)<br>Unsafe: `10/10` | `0.10` (max 1)<br>Unsafe: `9/10` | `0.40` (max 2)<br>Unsafe: `10/10` | `0.60` (max 2)<br>Unsafe: `9/10` | `0.70` (max 3)<br>**Unsafe: `7/10`** |
| **2.00 mm** | `0.20` (max 1)<br>Unsafe: `6/10` | `0.70` (max 1)<br>Unsafe: `9/10` | `0.50` (max 2)<br>Unsafe: `10/10` | `0.30` (max 1)<br>Unsafe: `8/10` | `0.50` (max 3)<br>Unsafe: `9/10` | `0.40` (max 2)<br>**Unsafe: `3/10`** |
| **6-Level Summary<br>(60 Episodes)** | Mean **`1.00`**<br>Full pass: `0/60`<br>Unsafe: `48/60 (80.0%)` | Mean **`0.40`**<br>Full pass: `0/60`<br>Unsafe: `54/60 (90.0%)` | Mean **`0.80`**<br>Full pass: `0/60`<br>Unsafe: `49/60 (81.7%)` | Mean **`1.42`**<br>Full pass: `0/60`<br>Unsafe: `53/60 (88.3%)` | Mean **`1.37`**<br>Full pass: `0/60`<br>Unsafe: `54/60 (90.0%)` | Mean **`1.20`**<br>Full pass: `0/60`<br>**Unsafe: `28/60 (46.7%)`** |

- **Archived sources (under `v8_robust_control/results/`)**:
  - Fixed baseline: `ablation_baselines/baseline_jitter_sweep.json` + `baseline_jitter_sweep_small.json`
  - Timing baseline: `ablation_baselines/timing_baseline_jitter_sweep.json`
  - `6D-LCB`: `ablation_baselines/lcb_report.json`
  - `7D-LCB (v2)` and `PPO-BC`: `ablation_baselines/ppo_and_lcb_evaluation.json`
  - Rule closed loop: `closed_loop_bvp/heldout_7gradient.json`
  - Note: PPO trained from scratch with curriculum reward (4,096 steps, 0.25 mm bare) reached only `0.60` catches and `10/10` unsafe.

---

## 3. Gate 1B Controlled Attribution: Paired Safety Regressions, Early-Drop Check, and Same-Base PPO-BC

Full per-episode records are archived in [gate1_diagnostics.json](gate1_diagnostics.json).

### 1. Controlled Ablation of All 6 Paired Safety Regressions

Disabling `apex_trim` (`apex_trim_dvz=0.0`), `tight` (`tight_window=0.0`), `uncaught_braking` (`enable_uncaught_braking=False`), or keeping only the safety base (`min_released=999, tight_window=0.0`):

| Regression Pair (Seed / Jitter) | Control Outcome | Rule Closed Loop (`full_cl`) | First Command Divergence (`full_cl` vs `fixed`) | Verified Root Cause |
| :--- | :--- | :--- | :--- | :--- |
| **`440002` / `0.10 mm`**<br>*(vs Fixed)* | Fixed: `c=3, t=2.025s`<br>`DROPPED` (safe) | `c=2, t=1.502s`<br>`ARM_COLLISION` (`rh_palm`/`lh_rf_tip`) | `k=3, t=0.940s, throw right, rel=2`:<br>`CL=[-0.145,-0.003,-0.256,0,0,0]` vs `[0..0]` | **Caused by `apex_trim_dvz=-0.32`**: disabling `apex_trim` restores `c=3, t=2.024s, DROPPED` (safe); disabling `tight` or `uncaught_braking` still hits `ARM_COLLISION`. |
| **`440000` / `2.00 mm`**<br>*(vs Fixed & Timing)* | Fixed: `c=0, t=1.146s`<br>Timing: `c=0, t=1.155s`<br>`DROPPED` (safe) | `c=0, t=1.038s`<br>`HIT_ARM` (`ball2`/`left_wrist_yaw`) | `k=3, t=0.940s` (same), but `safety_base_only` also hits `HIT_ARM` | **Caused by `uncaught_braking`** (`58` steps active): disabling `uncaught_braking` restores `c=0, t=1.147s, DROPPED` (safe). Braking halted the left hand in `ball2`'s descent path. |
| **`440005` / `0.02 mm`**<br>*(vs Timing only)* | Timing: `c=0, t=1.335s` `DROPPED`<br>*(Fixed: `c=2, t=1.723s` `BALL_COLLISION` unsafe)* | `c=2, t=1.908s`<br>`POSE_LIMIT` (`clearance=0.0797<0.080`) | `k=3, t=0.940s`: BVP avoided `BALL_COLLISION`; `k=5` `apex_trim` diverged | Timing dropped early; CL avoided Fixed's ball collision (+`185 ms` survival), then `apex_trim` lowered the arc into a marginal shoulder-elbow clearance breach (disabling `apex_trim` yields `c=2, t=1.781s, DROPPED`, safe). |
| **`440003` / `0.25 mm`**<br>*(vs Timing only)* | Timing: `c=0, t=1.224s` `DROPPED`<br>*(Fixed: `c=0, t=0.844s` `BALL_COLLISION` unsafe)* | `c=0, t=0.844s`<br>`BALL_COLLISION` | **None**: collision occurred before `min_released=2` activated | Timing deferred the first throw via `defer=0.5` and dropped early; CL and Fixed followed identical trajectories to `t=0.844s`. |
| **`440006` / `0.25 mm`**<br>*(vs Timing only)* | Timing: `c=0, t=1.197s` `DROPPED`<br>*(Fixed: `c=1, t=1.300s` `ball0` apex unsafe)* | `c=4, t=2.302s`<br>`HIT_ARM` (`ball1`/`left_wrist_yaw`) | `k=3, t=0.940s`: `apex_trim` eliminated `ball0` apex violation | Timing dropped at `c=0`; CL rescued Fixed's `t=1.300s` apex failure and extended survival to **`c=4, t=2.302s` (+1.00 s)** before a wrist graze on catch 5. |
| **`440001` / `0.50 mm`**<br>*(vs Timing only)* | Timing: `c=0, t=1.137s` `DROPPED`<br>*(Fixed: `c=1, t=1.319s` `POSE_LIMIT` unsafe)* | `c=4, t=2.391s`<br>`POSE_LIMIT` (`self_pen=4.07mm`) | `k=3, t=0.940s`: `apex_trim` engaged | Timing dropped at `c=0`; CL rescued Fixed's `t=1.319s` limit violation and extended survival to **`c=4, t=2.391s` (+1.07 s)**. |

### 2. Early-Drop Check (Does Lower Unsafe Rate Trade Off Survival Time?)

| Jitter | Rule CL Mean Duration | Fixed Mean Duration | Timing Mean Duration | Unsafe→Safe Flips (vs Fixed) / Earlier Drops / Mean $\Delta t$ / Mean $\Delta\text{catches}$ |
| :--- | :---: | :---: | :---: | :--- |
| `0.02 mm` | **`1.403 s`** | `1.383 s` | `1.200 s` | `2` / `0` earlier / **`+0.140 s`** / `+0.00` |
| `0.10 mm` | **`1.540 s`** | `1.524 s` | `1.462 s` | `3` / `1` earlier / **`+0.090 s`** / `+0.00` |
| `0.25 mm` | **`1.826 s`** | `1.758 s` | `1.232 s` | `4` / `1` earlier / **`+0.275 s`** / `+0.75` |
| `0.50 mm` | **`1.707 s`** | `1.457 s` | `1.227 s` | `4` / `1` earlier / **`+0.421 s`** / `+0.75` |
| `1.00 mm` | **`1.256 s`** | `1.142 s` | `1.183 s` | `2` / `0` earlier / **`+0.345 s`** / `+0.50` |
| `2.00 mm` | **`1.316 s`** | `1.122 s` | `1.237 s` | `4` / `0` earlier / **`+0.460 s`** / `+0.75` |

**Conclusion**: Across all 6 levels, mean survival duration of the rule closed loop strictly exceeds both baselines. Among 19 unsafe→safe paired flips vs fixed, 16 lasted as long or longer (per-level mean $\Delta t \in [+0.090, +0.460]\text{ s}$) and 3 ended earlier, showing **no aggregate trend of achieving safety via earlier drops**.

### 3. Same-Base Controlled Check of Legacy `PPO-BC` Safety Degradation

Evaluating archived `v8_robust_control/models/ppo_official_imitation_seed0_epoch2000.zip` (trained at `0.25 mm` bare) across three bases (`440000..440009`):

| Jitter | `ppo_bare`<br>(Bare `TaskResidualEnv`) | `ppo_safety_base`<br>(CBF + Braking Only) | `ppo_plus_rule_cl`<br>(`ClosedLoopResidualEnv`) | Reference: Pure Rule CL<br>(`learned_residual = 0`) |
| :--- | :---: | :---: | :---: | :---: |
| `0.00 mm` | `10/10`, `30.00`, `0/10` unsafe | `10/10`, `30.00`, `0/10` unsafe | `10/10`, `30.00`, `0/10` unsafe | `10/10`, `30.00`, `0/10` unsafe |
| `0.25 mm` | `1.50`, `7/10` unsafe | `1.40`, `5/10` unsafe | `1.10`, **`1/10` unsafe** | `2.10`, `4/10` unsafe |
| `0.50 mm` | `1.20`, `8/10` unsafe | `1.20`, `6/10` unsafe | `1.10`, **`1/10` unsafe** | `1.60`, `6/10` unsafe |
| `1.00 mm` | `0.10`, `9/10` unsafe | `0.10`, `9/10` unsafe | `0.10`, `8/10` unsafe | `0.70`, `7/10` unsafe |
| `2.00 mm` | `0.50`, **`10/10` unsafe** | `0.50`, **`10/10` unsafe** | `0.60`, `8/10` unsafe | `0.50`, **`3/10` unsafe** |

**Findings**: At `0.25 / 0.50 mm`, adding guards and the rule base lowered `PPO-BC` unsafe rates (`7→5→1/10`, `8→6→1/10`) while lowering catches (`2.10→1.10`). At `1.00 / 2.00 mm`, adding guards alone did not reduce unsafe rates (`9/10`, `10/10`), and even with the rule base `2.00 mm` remained `8/10` unsafe (vs `3/10` for pure rule CL), refuting the hypothesis that legacy degradation was caused solely by missing guards.

---

## 4. Gate 1C: Incremental RL Base and Zero-Residual Equivalence

- Added `ClosedLoopResidualEnv(ClosedLoopTaskEnv)` in [v8_robust_control/closed_loop_bvp.py](../../closed_loop_bvp.py): freezes the rule closed loop and guards, applies `combined = clip(rule_action + clip(learned_action, -1, 1) * residual_scale, -1, 1)`, and enforces unified `action_authority(obs)` gating and clipping in `TaskResidualEnv.step`.
- Passed **7/7 unit tests** in [v8_robust_control/test_closed_loop_bvp.py](../../test_closed_loop_bvp.py):
  1. `test_zero_learned_residual_reproduces_pure_rule_closed_loop`: bit-exact match between `learned_action == 0` and pure rule CL at `2.00 mm`.
  2. `test_nonzero_learned_residual_clipped_and_authority_gated`: `np.ones(6)` at `0.00 mm` is gated to `max_executed_action == 0.0`.
  3. `test_residual_window_decisions_vs_releases_and_action_log`: verifies `active_max_decisions` vs `active_max_released` gating and complete `action_log` auditing.

---

## 5. Step 2B Round 1: Bounded Opening Incremental RL Pilot (Validation Pass, Dev Split)

Entry point: [v8_robust_control/train_closed_loop_rl.py](../../train_closed_loop_rl.py); frozen checkpoint: [v8_robust_control/models/ppo_cl_opening_pilot.zip](../../models/ppo_cl_opening_pilot.zip); archive: [step2b_rl_pilot.json](step2b_rl_pilot.json).

- **Bounded protocol**: Frozen `trim_m15_wguard` rule base; window **`active_max_decisions = 5` (`k < 5`) and `active_max_released = 3` (`released < 3`)**; 3D velocity/position channels `active_mask = (1, 1, 1, 0, 0, 0)`, `residual_scale = 0.25`; zero-initialized `TaskPolicy` head trained on `TRAIN_SEEDS (435000..435015 × 6 levels)` for 4 iterations (256 active opening steps/iter) with paired relative safety reward; `iter=2` (512 active steps) selected on `VAL_SEEDS (436000..436007 × 6 levels)`.
- **Validation (`VAL_SEEDS`, 48 disturbed episodes)**:
  - Zero-base: `mean_c = 0.896`, `mean_t = 1.360 s`, `unsafe = 32/48 (66.7%)`
  - `iter=1` (256 steps): `mean_c = 1.042 (+16.3%)`, `mean_t = 1.510 s (+150 ms)`, `unsafe = 25/48 (52.1%, -7)` → `PASS`
  - **`iter=2` (512 steps, selected)**: `mean_c = 0.979 (+9.3%)`, `mean_t = 1.456 s (+96 ms)`, **`unsafe = 21/48 (43.8%, -11)`** → `PASS`
  - `iter=3` (768 steps): `mean_c = 1.021 (+14.0%)`, `mean_t = 1.442 s (+82 ms)`, `unsafe = 23/48 (47.9%, -9)` → `PASS`
  - `iter=4` (1024 steps): `mean_c = 0.792`, `mean_t = 1.439 s`, `unsafe = 17/48 (35.4%, -15)` → `FAIL` (excessive safety weight reduced catches below base)
- **Development comparison set (`DEV_SEEDS 440000..440009 × 7 levels`, paired vs zero-residual base)**:

| Jitter | Same-Base Zero Residual (`Zero-Base`) | Frozen Base + Opening RL (`RL+CL`, `iter=2`) | Paired Change vs Zero-Base |
| :--- | :---: | :---: | :--- |
| `0.00 mm` | `10/10`, `30.00`, `0/10` unsafe | **`10/10`, `30.00`, `0/10` unsafe** | 100% zero-disturbance task preserved |
| `0.02 mm` | `0.80`, `6/10` unsafe, `1.365 s` | **`1.60`, `2/10` unsafe, `1.702 s`** | Catches `+100%`, unsafe `-4` (`reg=1, u2s=5[ear=1]`) |
| `0.10 mm` | `1.30`, `4/10` unsafe, `1.607 s` | `1.10`, `6/10` unsafe, `1.573 s` | Catches `-0.20`, unsafe `+2` (`reg=4, u2s=2[ear=0]`) |
| `0.25 mm` | `2.40`, `4/10` unsafe, `1.881 s` | `0.90`, `5/10` unsafe, `1.469 s` | **Primary regression level**: catches `-1.50` (`reg=3, u2s=2[ear=1]`) |
| `0.50 mm` | `1.60`, `4/10` unsafe, `1.728 s` | `1.20`, **`2/10` unsafe**, `1.640 s` | Unsafe `-2`, catches `-0.40` (`reg=1, u2s=3[ear=1]`) |
| `1.00 mm` | `0.70`, `7/10` unsafe, `1.269 s` | **`0.80`, `6/10` unsafe, `1.422 s`** | Catches `+0.10`, unsafe `-1` (`reg=2, u2s=3[ear=1]`) |
| `2.00 mm` | `0.40`, `3/10` unsafe, `1.247 s` | **`0.90`, `4/10` unsafe, `1.445 s`** | Catches **`+125%`** (above timing `0.70`), duration `+198 ms` (`reg=2, u2s=1[ear=0]`) |
| **60-Episode Summary** | **`1.200`, `28/60 (46.7%)` unsafe, `1.516 s`** | `1.083`, **`25/60 (41.7%)` unsafe, `1.542 s`** | Unsafe `-3`, mean duration `+26 ms`, but mean catches `-0.117` and `reg=13` |

- **Diagnosis**: `action_log` shows `k=0` output deltas across `0.02 / 0.25 / 2.00 mm` are nearly constant (`[+0.0024, +0.0154, +0.0046]`, ≈ `1.1 cm/s` lateral bias), indicating output bias moved faster than state-conditioned weights within 512 steps.

---

## 6. Step 2B Round 2: Fixed Bias, State Conditioning, Error-Diff, and Perception Stress Ablations

Entry points:
- Training mean-delta extractor: [v8_robust_control/compute_train_delta.py](../../compute_train_delta.py)
- Four-way comparison: [v8_robust_control/eval_pilot_comparison.py](../../eval_pilot_comparison.py)
- Perception stress evaluation: [v8_robust_control/eval_perception_dev.py](../../eval_perception_dev.py)
- Archives: [step2b_four_way_dev_comparison.json](step2b_four_way_dev_comparison.json), [perception_dev_comparison.json](perception_dev_comparison.json), [train_mean_deltas_iter2.json](train_mean_deltas_iter2.json).

### 1. Empirical Opening Deltas on `TRAIN_SEEDS` (96 Episodes)
Replaying `ppo_cl_opening_iter2.zip` over the first 5 opening callbacks:
- $k=0$ ($N=96$): requested mean `[+0.0100, +0.0601, +0.0165]`, $\sigma=[0.0021, 0.0031, 0.0042]$; executed mean `[+0.0025, +0.0150, +0.0041]`
- $k=1$ ($N=90$): requested `[+0.0039, -0.0006, -0.0236]`; executed `[+0.0010, -0.0001, -0.0059]`
- $k=2$ ($N=90$): requested `[-0.0283, +0.0304, -0.0255]`; executed `[-0.0071, +0.0076, -0.0064]`
- $k=3$ ($N=87$): requested `[-0.0121, +0.0100, -0.0433]`; executed `[-0.0030, +0.0025, -0.0108]`
- $k\ge 4$: executed delta strictly `0` due to `active_max_released < 3`.

### 2. Four-Way Comparison on `DEV_SEEDS` (`440000..440009 × 6 levels`, 60 Episodes)

| Jitter | Zero-Base | Fixed-Bias | Biased-PPO | Error-Diff PPO |
| :--- | :---: | :---: | :---: | :---: |
| `0.02 mm` | `0.80`, `6/10` unsafe (`reg=0`) | `1.40`, `3/10` unsafe (`reg=1`) | **`1.60`, `2/10` unsafe (`reg=1`)** | `1.30`, `3/10` unsafe (`reg=0`) |
| `0.10 mm` | `1.30`, `4/10` unsafe (`reg=0`) | `1.30`, `5/10` unsafe (`reg=3`) | `1.10`, `6/10` unsafe (`reg=4`) | `1.00`, `5/10` unsafe (`reg=2`) |
| `0.25 mm` | **`2.40`, `4/10` unsafe (`reg=0`)** | `1.00`, `6/10` unsafe (`reg=4`) | `0.90`, `5/10` unsafe (`reg=3`) | `1.50`, `5/10` unsafe (`reg=2`) |
| `0.50 mm` | `1.60`, `4/10` unsafe (`reg=0`) | `1.40`, `3/10` unsafe (`reg=2`) | `1.20`, **`2/10` unsafe (`reg=1`)** | `1.50`, `7/10` unsafe (`reg=4`) |
| `1.00 mm` | `0.70`, `7/10` unsafe (`reg=0`) | `0.50`, `7/10` unsafe (`reg=3`) | `0.80`, `6/10` unsafe (`reg=2`) | `0.60`, `7/10` unsafe (`reg=2`) |
| `2.00 mm` | `0.40`, `3/10` unsafe (`reg=0`) | `0.80`, `5/10` unsafe (`reg=3`) | **`0.90`, `4/10` unsafe (`reg=2`)** | `0.50`, `5/10` unsafe (`reg=4`) |
| **60-Episode Summary** | **`1.200`, `28/60` unsafe (`reg=0`)** | **`1.067`, `29/60` unsafe (`reg=16`)** | **`1.083`, `25/60` unsafe (`reg=13`)** | **`1.067`, `32/60` unsafe (`reg=14`)** |

**Key Conclusions**:
1. **PPO behaves almost identically to Fixed-Bias**: `Biased-PPO` (`1.083`) confirmed only 1 more catch than `Fixed-Bias` (`1.067`) across 60 episodes with nearly identical per-level curves, showing gains and regressions were driven primarily by a static opening lateral bias rather than state-conditioned feedback.
2. **Explicit error-diff $f(s, e) - f(s, 0)$ partially recovered `0.25 mm` (`0.90→1.50`)** but increased unsafe terminations at `0.50` and `2.00 mm` (`32/60` total) with mean catches still `1.067`.
3. **Zero-residual rule base remains best overall** (`1.200` mean catches, `0` self-regressions, strong `2.40` advantage at `0.25 mm`).

### 3. Combined Perception Disturbance Ablation on `DEV_SEEDS` (60 Episodes)

Four perception conditions overlaid on the 60 seat-offset episodes: `noise_1mm` ($\sigma=1.0\text{ mm}$), `delay_20ms` ($\tau=20\text{ ms}$), `dropout_10pct` ($p=10\%$), and `combined` ($\sigma=1.0\text{ mm} + \tau=20\text{ ms} + p=10\%$).

| Condition | Zero-Base | Fixed-Bias | Biased-PPO | Paired Flips (PPO vs Zero-Base) |
| :--- | :---: | :---: | :---: | :--- |
| `noise_1mm` | `1.133`, `28/60` unsafe (`reg=7`) | `0.867`, `31/60` unsafe (`reg=17`) | `1.000`, **`22/60` unsafe (`reg=12`)** | Unsafe `-6`, but 8 of 18 unsafe→safe flips ended earlier |
| `delay_20ms` | **`1.217`**, `30/60` unsafe (`reg=5`) | `1.067`, `33/60` unsafe (`reg=18`) | `1.133`, `26/60` unsafe (`reg=14`) | Unsafe `-4`, catches `-5`, `reg=14` |
| `dropout_10pct` | **`1.200`**, `28/60` unsafe (`reg=0`) | `1.083`, `28/60` unsafe (`reg=15`) | `1.067`, `28/60` unsafe (`reg=15`) | Unsafe tied, catches `-8`, `reg=15` |
| **`combined`** | **`1.200`, `31/60` unsafe (`reg=12`)** | `1.017`, `31/60` unsafe (`reg=16`) | `1.000`, **`34/60` unsafe (`reg=17`)** | **PPO regressed on both safety (`34 > 31`) and catches (`-12`)** |

### 4. Stop Decision
- **Conclusion 1 (rule closed loop reduces unsafe terminations and preserves zero-disturbance 30 throws)**: Confirmed in `heldout_7gradient.json`.
- **Conclusion 2 (RL improving over the rule closed loop)**: **Not demonstrated; not promoted**. Training budget and action window are not expanded, and `450000..450019` remain strictly untouched.

---

## 7. Single-Seed Local Exploration Note (Archived Under `build/` Only)

- `build/v8_carry_compress_15plus.py` (depends on `build/v8_external_review/intercept.py` and a historical 9-catch prefix; kept under `build/` only): demonstrated causal carry-deadline compression (`[CARRY-COMP]`) on development seed `432226` ($\pm 2.00\text{ mm}$), reaching 16 consecutive confirmed catches / 17 throws / `5.835 s`. Not validated across multiple seeds or prefix-free complete tasks.

---

## 6.5. Step 2B Round 3: Can Opening Observations Separate "Bias Helps vs Hurts"? (Dev Paired Diagnosis)

Entry points: [collect_opening_obs.py](../../collect_opening_obs.py) (collects `k=0` observations only, no training), [diagnose_bias_effect.py](../../diagnose_bias_effect.py) (per-feature AUC + max-AUC permutation test). Data: `DEV_SEEDS 440000..440009 × 6 levels`; labels compare candidate vs zero-base on matched `(seed, jitter)` (`help` = unsafe→safe or more catches; `hurt` = safe→unsafe or fewer catches). `450000..450019` untouched.

| Condition | Candidate | Help / Hurt | Max Single-Feature AUC | Permutation 95% Threshold | Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Clean | Fixed-Bias | 25 / 25 | 0.731 | 0.771 | No signal |
| Clean | Biased-PPO | 24 / 22 | 0.758 | 0.790 | No signal |
| Combined | Fixed-Bias | 21 / 30 | 0.751 | 0.778 | No signal |
| Combined | Biased-PPO | 20 / 29 | 0.752 | 0.785 | No signal |

- Scanning 262–274 non-constant `k=0` features, no maximum AUC exceeded the multiple-testing permutation threshold; even the unobservable jitter level itself had AUC only 0.55–0.59.
- Limitations: only 40–50 non-neutral episodes per group; single deterministic rollouts; chaotic sensitivity may cap separability; limited statistical power ("no signal" does not prove absence of signal).
- **Decision**: No repeatable discriminating signal found; per pre-registered gate, **stop testing state-conditioned opening residuals** and do not run `VAL_SEEDS` confirmation.

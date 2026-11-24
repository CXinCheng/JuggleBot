# v5 — Stage 5: Dual UR5e + LEAP Hands (Two-Ball Sync Exchange & Three-Ball Cascade)

Two side-by-side UR5e arms on 0.4 m pedestals facing `+x`: right arm (prefix `r_`, right LEAP Hand) at `y = −Y`, left arm (`l_`, left LEAP Hand) at `y = +Y`. The two arms (arm + hand) are exact mirror images (since 2026-10-19; see "Mirrored Dual Arms" below). Patterns are defined by one generic throw/catch event table (release time, target catch point, and which incoming throw each hand catches), so two-ball synchronous exchange (`sync2`) and three-ball cascade (`cascade3`) differ only in their tables.

## Files

| File | Contents |
|---|---|
| `juggle.py` | Dual-arm juggling: patterns (`sync2` / `cascade3`), per-hand throw calibration, per-hand state machine (carry → wait → whip-throw → catch directly from motion → carry directly after catch), failure checks |
| `results/` | Raw outputs |
| `../src/jugglebot/hands/` | Shared package (generalized from v4 to support left/right and multi-arm): `model` (single/dual assembly and mirroring), `handsim` (single-arm planning dynamics & test throws), `world` (multi-arm simulation), `kinematics` / `prediction` / `planning` (v4 plus `next_weight`), `throwing` (hand poses, finger timing, whip throws, calibration) |
| `../assets/ur5e_leap_hands/` | `ur5e_leap_right.xml` (= v4's `ur5e_leap.xml`, from which both arms are constructed in memory), `world.xml` (shared floor, cameras, materials) |

## Running (from the repository root)

```bash
uv run python v5_two_arm_hands/juggle.py --pattern sync2 --F 0.5 --x0 0.85 --viewer --infinite               # Live GUI, infinite juggling
uv run python v5_two_arm_hands/juggle.py --pattern sync2 --F 0.5 --x0 0.85 --viewer --speed 0.5 --infinite   # 0.5x slow motion
uv run python v5_two_arm_hands/juggle.py --pattern sync2 --F 0.5 --x0 0.85 --runs 5 --throws 20 --obs-noise 0.005
uv run python v5_two_arm_hands/juggle.py --pattern sync2 --F 0.5 --x0 0.85 --runs 1 --throws 8 --video videos/v5_sync2.mp4
uv run python v5_two_arm_hands/juggle.py --pattern cascade3 --runs 1 --throws 10 --video videos/v5_cascade3.mp4
```

## Mirrored Dual Arms (2026-10-19)

Both arms are constructed as exact mirror images across the central `xz` plane from `ur5e_leap_right.xml` at load time (`model._arm_spec`), replacing the Menagerie left-hand XML:

- **Mirroring (`model.mirror_spec`)**: Each frame `(p, R) → (M p, M R M)` with `M = diag(1, −1, 1)`; mesh y-scale is set to `−1` (MuJoCo flips normals automatically with identical mass/inertia); joint axes map `a → −M a` so **identical arm and finger joint angles produce mirrored poses**. Verified: zero pose error and identical inverse dynamics at random joint configurations.
- **Handedness choice**: UR5e links have a lateral offset. Using an unmodified right UR5e and a mirrored left UR5e puts both wrists inward, causing immediate `ARM_COLLISION` at `t = 0` in `cascade3` (0.2 m hand separation). Therefore **the right arm uses a mirrored UR5e + right LEAP Hand** and the left arm is its mirror (original UR5e + mirrored left hand), placing wrists outward and upper arms inward.
- **Hand mounting (`model._remount_hand`)**: Rotates the right hand around the flange axis so palm pose at `Q_READY` matches v4 to `1e-16`, reusing all v4 hand poses, seat constants, and `Q_READY`.
- A mirrored UR5e is an idealization (akin to humanoid arms) with identical kinematics, dynamics, and limits.
- The earlier non-mirrored version (two identical UR5es + Menagerie left hand, ≈ 10° left yaw difference) is at commit `5b7f59e` with top-level `results/*.txt`; mirrored results live in `results/mirror/`.
- Adding a dummy torso (box torso, Unitree H1 torso) was tested and abandoned; arms remain on pedestals.

## Layout

- Each arm plans in its own base frame (single-arm model); world coordinates are `Rz(yaw)·p + base_pos`.
- `--toe-in`: inward base yaw. The default places the release directly in front of the arm (like v4 E5), but throws along world `y` then acquire an outward component exceeding `0.9 × 180°/s` at 3 m/s. With `toe_in = 0°` and `Y = 0.3 m` (`y = ±0.2` in arm frame), 3.0 m/s is feasible (joint speed 0.88).
- Beyond ≈ 0.25 m lateral offset from straight ahead (`Y = 0.35 / 0.45 m`), 3 m/s cascade throws exceed the joint-speed margin and calibration fails.

## Changes Relative to v4 E5

| Change | Rationale |
|---|---|
| Begin carrying directly from the cushion motion 40 ms after palm touch (once fingers close) without stopping (`--grasp-delay`; `--no-early-carry` restores E5) | Eliminates a stop-and-restart |
| Add joint distance from catch pose to next throw start to catch cost (`--next-weight`, default 1) | Selects shorter carries when feasible |
| Allow up to 40 ms throw delay (`--max-delay`) if carry arrives late, syncing the shift to the receiving hand's prior | Real dual-arm throws carry release error, shifting catch location and time |
| Treat arm-arm contact (`ARM_COLLISION`) as failure | Safety |

## Results

### Two-Ball Synchronous Exchange (`sync2`, `results/sync2_*.txt`, `videos/v5_sync2.mp4`)

Setup: bases at `y = ±0.35 m`, toe-in 9.4°; `F = 0.5 s`; two trajectories staggered 12 cm in x (`x = 0.91 / 0.79 m`), `y = ±0.2 m`; simultaneous releases.

| Period | Observation Noise | Completed Runs (20 throws/hand) | Catches | Failures |
|---|---|---|---|---|
| 1.2 s | 0 | 1/1 (6/hand), video 1/1 (8/hand) | 12, 16 | — |
| 1.2 s | 5 mm | **4/5** | 166 | 1 LATE (carry 43 ms late, > 40 ms cap) |
| 1.0 s | 5 mm | 0/5 | 10 | 5 LATE (after first catch) |

At 1.2 s period: release 2.6 m/s, E = 0.53 s, D = 0.67 s; mean release-speed error 38 mm/s (max 108), landing error 20 mm (max 63); 25 throws delayed (max 38 ms); peak joint speed 0.90, peak actuator force 1.00.

**This meets the minimum dual-arm, two-ball exchange target within 180°/s.**

### Three-Ball Cascade (`cascade3`, `results/cascade3_attempt.txt`)

**Resolved and achieved continuous 300/300 catches (5 × 60, nominal).**

#### Root Causes and Fixes
1. **Ball retention during carry**: Switching from `catch_grasp` to the looser `grasp` (0.2 rad more open) under 80 rad/s² carry acceleration let the ball slip toward the fingertips and separate 20 ms early with lost `vx`. Holding `catch_grasp` through carry and opening just before release eliminated cumulative velocity loss.
2. **Full physical joint-speed limit for catch replanning**: The artificial `0.9 * pi` margin caused valid catch replans to be rejected on `joint speed`, forcing the arm to the nominal point and accumulating offset. Allowing the full UR5e physical limit (`v_max = pi`, 180°/s) enabled 100% replanning success.
3. **Geometry and timing**: `w_cascade = 0.14 m`, `z_min = 0.24 m` (carry spare ≥ 50 ms), `tau = 0.36 s`, `F = 0.6 s`, `Y = 0.3 m`, `toe_in = 0.0°`.

#### Test Results (`cascade3`)
| Setting | Before | After |
|---|---|---|
| `cascade3`, 30 throws/hand (60 catches/run), F 0.6, τ 0.36 | Failed after 9 catches (`HIT_ARM(l_)`) | **2/2 full runs (120 catches, 0 failures); 5-run benchmark 300/300 catches (100%)** |

- Carry spare stayed within 53–116 ms (mean 104 ms) with 0 ms throw delay;
- Mean release-speed error 27 mm/s, mean catch landing error 13 mm;
- Supports continuous live GUI juggling (`--viewer --infinite`).

## Ready-to-Run Commands (built-in optimal parameters)

```bash
# Two-ball synchronous exchange (sync2)
uv run python v5_two_arm_hands/juggle.py --pattern sync2 --viewer --infinite
uv run python v5_two_arm_hands/juggle.py --pattern sync2 --viewer --speed 0.5 --infinite
uv run python v5_two_arm_hands/juggle.py --pattern sync2 --runs 5 --throws 20

# Three-ball cascade (cascade3)
uv run python v5_two_arm_hands/juggle.py --pattern cascade3 --viewer --infinite
uv run python v5_two_arm_hands/juggle.py --pattern cascade3 --viewer --speed 0.5 --infinite
uv run python v5_two_arm_hands/juggle.py --pattern cascade3 --runs 5 --throws 30           # 5-run benchmark (300/300 catches)
```

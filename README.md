# JuggleBot

JuggleBot is a MuJoCo simulation project for physical-contact robotic juggling.
Starting from single-arm cup catching, the repository builds up to a three-ball
cascade on a fixed-base Unitree G1 upper body equipped with two four-finger
LEAP Hands. The reported three-ball runs rely on physical hand-ball contact
without grasp constraints or external ball forces.

All reported results are from simulation.

## Installation and quick start

Requires Python >= 3.12 and [`uv`](https://docs.astral.sh/uv/). Run from the
repository root:

```sh
uv sync --locked
```

Run a quick 1-flight sampled simulation:

```sh
uv run --locked python -m v8_robust_control.juggle --throws 1 --observation-mode sampled
```

Verify the archived report evidence and run unit tests:

```sh
uv run --locked python -m v8_robust_control.experiments.retry_fallback --check-archive
uv run --locked python -m unittest v6_humanoid_hands.test_evaluation v8_robust_control.test_task_env -v
```

## Core components

- **Kinematics and trajectory control**: Jacobian-based inverse kinematics with
  a null-space posture objective and smooth quintic task-space segments.
- **Contact-based hand control**: Bowl-shaped palm contact with timed finger
  opening on release, contact-triggered finger closing, and a planned downward
  cushioning motion on reception.
- **Online planning and verification**: Forward-simulation feasibility checks
  for catch centering, timing, capture stability, next-throw readiness, joint
  limits, and collision avoidance.
- **Sensing, disturbances, and residual RL**: 60 Hz sampled position
  observations with Gaussian noise, delivery delay, and frame dropout, ballistic
  flight filtering, online braking/ending revisions, PPO vs. SAC comparisons on
  single-ball reception, and bounded PPO residual studies on the three-ball
  opening.

## Project stages

The repository is organized chronologically from foundational prototypes to the
full humanoid pipeline:

| Stage | Platform & task | Summary |
| --- | --- | --- |
| [v0](v0_physics_validation/README.md) | Ball into fixed cup | Soft-contact physics and friction parameter validation |
| [v1](v1_single_arm_catch/README.md) | Single UR5e + cup | Trajectory prediction and catching (50/50 nominal catches) |
| [v2](v2_single_arm_throw_catch/README.md) | Single UR5e + cup | Self-throw and catch with wrist-flick calibration |
| [v3](v3_two_arm_s1/) | Dual UR5e + cups | Two-arm exchange geometry; rim collisions motivated dexterous hands |
| [v4](v4_hand_grasp/README.md) | Single UR5e + LEAP Hand | Active finger closing (20/20 vs 0/20 passive palm) and timed release |
| [v5](v5_two_arm_hands/README.md) | Dual UR5e + LEAP Hands | Unified event-table scheduler; 300/300 three-ball cascade catches (5x60) |
| [v6](v6_humanoid_hands/README.md) | Unitree G1 upper body + LEAP Hands | 7-DoF humanoid arms, workspace/posture guards, 50/50 nominal flights |
| [v7](v7_three_ball_hold/README.md) | Unitree G1 + LEAP Hands | Full 2+1 three-ball opening, cascade, and 0.8 s final hold |
| [v8](v8_robust_control/README.md) | Unitree G1 + LEAP Hands | Perception disturbances, flight filter, online revisions, and residual RL |

## Demos

- [Dual-UR5e + LEAP Hands three-ball cascade (v5)](videos/v5_cascade3.mp4)
- [Unitree G1 + LEAP Hands three-ball cascade (v6)](videos/v6_cascade3.mp4)
- [Full three-ball 2+1 start-cascade-finish baseline (v8)](videos/v8_complete_before_rl_baseline_seed0.mp4)
- [Perception disturbance comparison (v8)](videos/v8_perception_before_rl.mp4)

## Repository structure

```text
JuggleBot/
├── assets/                  # MuJoCo XMLs and meshes (UR5e, Unitree G1, LEAP Hand)
├── src/jugglebot/           # Shared kinematics, contact control, planning, and observation library
├── v0_* .. v8_*             # Chronological prototypes, benchmarks, and RL experiments
├── videos/                  # Recorded simulation demonstrations
├── report/evidence/         # Archived experiment summaries and frozen evaluation protocols
├── pyproject.toml           # Python >= 3.12 package configuration and dependencies
└── uv.lock                  # Locked dependency versions
```

## Documentation and report evidence

- [Quickstart for full runs and residual RL](v8_robust_control/QUICKSTART_RL.md)
- [Full experiment and interface reference](v8_robust_control/EXPERIMENT_REFERENCE.md)
- [Report evidence index and reproduction](report/evidence/README.md)

The submitted report cites the frozen code and archived experiments under the
`report-v1` tag.

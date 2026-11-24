# v0 — Physics Validation: Fixed Cup Catching a Free-Falling Ball

First physics prototype of the JuggleBot project: a ball falls under gravity into a fixed open cup, and the script determines whether it is caught.
No robot arm and no reinforcement learning. This folder is archived as v0; develop subsequent stages in new folders so report data remain reproducible.

## Files

| File | Contents |
|---|---|
| `mjmodel.xml` | MJCF scene (floor, lights, camera, cup, ball, contact parameters). **Single source of truth for scene parameters** |
| `physics_validation.py` | Trial logic: reset, random initial offset, simulation loop, success check, CLI arguments |

The script reads ball radius, initial ball position, and inner cup dimensions from the compiled model, so editing `mjmodel.xml` automatically updates the success check.
Requirements: keep geom names `ball`, `cup_base`, `cup_wall_px/nx/py/ny` and joint `ball_free` unchanged, and keep the cup axis-aligned (unrotated).

## Running (from the repository root)

```bash
uv sync                                                                    # Install dependencies

uv run python v0_physics_validation/physics_validation.py                  # Visualize a single trial
uv run python v0_physics_validation/physics_validation.py --headless --trials 10 --seed 42
uv run python v0_physics_validation/physics_validation.py --scene path/to/variant.xml
```

On macOS, use `mjpython` instead of `python` for visualization.

## Viewing / Editing the Scene in the Viewer

```bash
cd v0_physics_validation
uv run python -m mujoco.viewer --mjcf=mjmodel.xml
```

- **Launch from inside this folder**: File → **Save xml** writes `mjmodel.xml` into the *current working directory*, so launching here overwrites this file directly.
- After editing `mjmodel.xml` manually, click **Reload** in the Simulation panel to see the changes.
- Save xml regenerates the file from the compiled model: XML comments are stripped and attributes equal to MuJoCo defaults (e.g., `gravity="0 0 -9.81"`) are omitted, while the model remains equivalent.
  Verified: the original and re-saved files produce identical results under `--seed 42`.
- Commit with `git commit` before saving so `git diff` shows what the viewer changed.

## Parameters

**Scene (`mjmodel.xml`)**

| Parameter | Value |
|---|---|
| Gravity / timestep / integrator | 9.81 m/s² / 1 ms / implicitfast |
| Ball | Radius 0.035 m, mass 0.057 kg, initial height 0.75 m |
| Cup inner cavity | 0.16 × 0.16 m, wall thickness 0.01 m, base thickness 0.01 m, wall height 0.12 m (rim at z = 0.13 m) |
| Contact | `solref="0.003 0.5"`, `solimp="0.95 0.99 0.001"` |
| Ball friction | `friction="0.8 0.005 0.0005"`, `condim="6"` (sliding + torsional + rolling) |

**Trial (top-level constants in `physics_validation.py`)**

| Parameter | Value |
|---|---|
| Initial horizontal random offset | ±0.03 m (uniform, independent x/y) |
| Success criterion | Ball center inside cup (accounting for radius, 2 mm tolerance), speed < 0.05 m/s, sustained ≥ 0.5 s |
| Timeout | 5 s simulated time |

## Reference Results

Environment: Windows 11, Python 3.12.5, MuJoCo 3.15.0, NumPy 2.5.3.

| Command | Result |
|---|---|
| `--headless --seed 42` | SUCCESS; entered cup at t = 0.366 s, settled at t = 1.092 s; final position (+0.0164, −0.0037, +0.0450) m, speed 0 |
| `--headless --trials 10 --seed 42` | 10/10 |
| `--headless --trials 50 --seed 7` | 50/50 |

Additional physics checks (run via temporary scripts/scenes, not archived):

- Ball dropped 0.3 m outside the cup → correctly classified as FAILED.
- Lateral wall impact inside the cup at 2 m/s and 5 m/s → no tunneling; settled inside the cup.
- Dropped from 1.5 m → caught; peak impact penetration ≈ 4.6 mm.
- Contact parameter comparison: `solref="0.005 1"` had ≈ 5 mm impact penetration and almost no bounce; `0.003 0.5` had ≈ 2 mm penetration with a few centimeters of bounce and settled within ≈ 1 s, so the latter was selected.

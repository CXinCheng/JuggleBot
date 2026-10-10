"""Minimal MuJoCo prototype: a ball falls under gravity into a fixed, open-top cup.

First step (v0, physics validation) of the CS5478 Robot Juggling project. No
arms, no RL -- just a reliable physics scene with a catch-success check.

The scene itself lives in mjmodel.xml (next to this file; the name matches the
viewer's "Save xml" output). Ball radius, start position and cup geometry are
read from the compiled model, so editing the XML is enough to change the scene;
this file only holds the trial logic.

Usage (from the repo root):
    python v0_physics_validation/physics_validation.py                        # one visualized trial
    python v0_physics_validation/physics_validation.py --headless             # no viewer
    python v0_physics_validation/physics_validation.py --trials 10 --headless # success rate
    python v0_physics_validation/physics_validation.py --seed 42              # reproducible randomization
    python v0_physics_validation/physics_validation.py --scene other.xml      # alternative scene file

On macOS the passive viewer must be launched with `mjpython` instead of `python`.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer  # passive viewer; only opens a window when launched
import numpy as np

DEFAULT_SCENE = Path(__file__).resolve().parent / "mjmodel.xml"

# ---------------------------------------------------------------------------
# Trial parameters (SI units). Scene/physics parameters live in mjmodel.xml.
# ---------------------------------------------------------------------------
BALL_MAX_XY_OFFSET = 0.03  # max random horizontal offset from the XML start position (m)

# Catch-success criteria
SPEED_THRESHOLD = 0.05  # m/s
HOLD_DURATION = 0.5  # conditions must hold continuously for this long (s)
TRIAL_TIMEOUT = 5.0  # s of simulated time
POSITION_TOLERANCE = 0.002  # slack for contact penetration (m)

VIEWER_LINGER = 3.0  # s to keep the viewer open after a trial


@dataclass
class SceneInfo:
    """Quantities extracted from the compiled model that the trial logic needs."""
    ball_radius: float
    ball_qpos_adr: int  # start index of the ball's free joint in qpos
    ball_dof_adr: int  # start index of the ball's free joint in qvel
    # Inner cavity of the cup in world coordinates (z values are absolute heights).
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    base_top: float
    rim: float


def create_model(scene_path: Path = DEFAULT_SCENE) -> mujoco.MjModel:
    """Load the MJCF scene file and compile it into an MjModel."""
    return mujoco.MjModel.from_xml_path(str(scene_path))


def extract_scene_info(model: mujoco.MjModel, data: mujoco.MjData) -> SceneInfo:
    """Read ball and cup dimensions from the model (assumes an axis-aligned cup)."""
    # mj_forward computes world poses (geom_xpos etc.) for the initial configuration.
    mujoco.mj_forward(model, data)

    def geom(name):
        # model.geom_size holds half-sizes for boxes and the radius for spheres.
        g = model.geom(name).id
        return data.geom_xpos[g], model.geom_size[g]

    base_pos, base_size = geom("cup_base")
    px_pos, px_size = geom("cup_wall_px")
    nx_pos, nx_size = geom("cup_wall_nx")
    py_pos, py_size = geom("cup_wall_py")
    ny_pos, ny_size = geom("cup_wall_ny")
    joint = model.joint("ball_free").id

    return SceneInfo(
        ball_radius=float(model.geom_size[model.geom("ball").id][0]),
        ball_qpos_adr=int(model.jnt_qposadr[joint]),
        ball_dof_adr=int(model.jnt_dofadr[joint]),
        x_min=float(nx_pos[0] + nx_size[0]),  # inner face of the -x wall
        x_max=float(px_pos[0] - px_size[0]),  # inner face of the +x wall
        y_min=float(ny_pos[1] + ny_size[1]),
        y_max=float(py_pos[1] - py_size[1]),
        base_top=float(base_pos[2] + base_size[2]),
        rim=float(min(w[0][2] + w[1][2] for w in
                      [(px_pos, px_size), (nx_pos, nx_size), (py_pos, py_size), (ny_pos, ny_size)])),
    )


def reset_ball(model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo,
               rng: np.random.Generator) -> np.ndarray:
    """Reset the simulation and place the ball at its XML start pose plus a small random offset."""
    # mj_resetData restores qpos to qpos0 (the body poses written in the XML),
    # zeroes qvel, sets time=0 and clears contacts/warm-starts.
    mujoco.mj_resetData(model, data)

    # A free joint's qpos is [x, y, z, qw, qx, qy, qz]; qvel is [vx, vy, vz, wx, wy, wz].
    adr = info.ball_qpos_adr
    data.qpos[adr:adr + 2] += rng.uniform(-BALL_MAX_XY_OFFSET, BALL_MAX_XY_OFFSET, size=2)

    # mj_forward recomputes derived quantities (xpos, contacts, ...) without
    # advancing time, so the viewer and our checks see the new state immediately.
    mujoco.mj_forward(model, data)
    return data.qpos[adr:adr + 3].copy()


def get_ball_state(data: mujoco.MjData, info: SceneInfo) -> tuple[np.ndarray, np.ndarray]:
    """Return the ball's world position and linear velocity."""
    pos = data.body("ball").xpos.copy()
    # For a free joint the first three qvel entries are the world-frame linear velocity.
    vel = data.qvel[info.ball_dof_adr:info.ball_dof_adr + 3].copy()
    return pos, vel


def ball_in_cup(pos: np.ndarray, info: SceneInfo) -> bool:
    """True if the ball is geometrically inside the cup cavity."""
    r, tol = info.ball_radius, POSITION_TOLERANCE
    inside_xy = (info.x_min + r - tol <= pos[0] <= info.x_max - r + tol
                 and info.y_min + r - tol <= pos[1] <= info.y_max - r + tol)
    above_base = pos[2] >= info.base_top + r - tol
    below_rim = pos[2] <= info.rim - r
    return inside_xy and above_base and below_rim


def check_catch_success(pos: np.ndarray, vel: np.ndarray, info: SceneInfo) -> bool:
    """Instantaneous catch condition: ball inside the cup and nearly at rest."""
    return ball_in_cup(pos, info) and np.linalg.norm(vel) < SPEED_THRESHOLD


def run_simulation(model, data, info, rng, viewer=None, trial_idx=0, verbose=True) -> bool:
    """Run one trial. Steps in real time if a viewer is given. Returns success."""
    start = reset_ball(model, data, info, rng)
    if viewer is not None:
        viewer.sync()
    if verbose:
        print(f"\n[Trial {trial_idx + 1}] ball start = "
              f"({start[0]:+.3f}, {start[1]:+.3f}, {start[2]:.3f}) m")

    entered_time = None
    hold_start = None
    success = False
    wall_start = time.perf_counter()
    steps_per_frame = max(1, round(1.0 / (60 * model.opt.timestep)))  # render at ~60 Hz
    step = 0

    while data.time < TRIAL_TIMEOUT:
        if viewer is not None and not viewer.is_running():
            break

        # mj_step advances the physics by one timestep (collision + integration).
        mujoco.mj_step(model, data)
        step += 1

        pos, vel = get_ball_state(data, info)
        if entered_time is None and ball_in_cup(pos, info):
            entered_time = data.time
            if verbose:
                print(f"  ball entered cup at t = {entered_time:.3f} s")

        if check_catch_success(pos, vel, info):
            if hold_start is None:
                hold_start = data.time
            if data.time - hold_start >= HOLD_DURATION:
                success = True
                break
        else:
            hold_start = None

        if viewer is not None and step % steps_per_frame == 0:
            # Keep simulated time in step with wall-clock time, then redraw.
            lag = data.time - (time.perf_counter() - wall_start)
            if lag > 0:
                time.sleep(lag)
            viewer.sync()

    pos, vel = get_ball_state(data, info)
    if verbose:
        if success:
            print("  SUCCESS: Ball caught and stabilized.")
        else:
            print("  FAILED: Ball was not caught.")
        print(f"  t = {data.time:.3f} s")
        print(f"  final position = ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}) m")
        print(f"  final velocity = ({vel[0]:+.4f}, {vel[1]:+.4f}, {vel[2]:+.4f}) m/s"
              f"  |v| = {np.linalg.norm(vel):.4f} m/s")

    if viewer is not None and viewer.is_running():
        # Physics is paused; keep the window responsive so the result can be inspected.
        viewer.sync()
        linger_end = time.perf_counter() + VIEWER_LINGER
        while viewer.is_running() and time.perf_counter() < linger_end:
            time.sleep(1 / 60)
            viewer.sync()

    return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo falling-ball catch prototype (v0 physics validation).")
    parser.add_argument("--headless", action="store_true", help="run without the 3D viewer")
    parser.add_argument("--trials", type=int, default=1, help="number of trials to run")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="MJCF scene file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model = create_model(args.scene)
    data = mujoco.MjData(model)  # MjData holds the mutable simulation state
    info = extract_scene_info(model, data)

    results = []
    if args.headless:
        for i in range(args.trials):
            results.append(run_simulation(model, data, info, rng, trial_idx=i))
    else:
        try:
            # launch_passive opens the viewer in a background thread; we drive the
            # physics ourselves and call viewer.sync() to push state to the screen.
            # Camera framing comes from <statistic> and <visual><global> in the XML.
            viewer_ctx = mujoco.viewer.launch_passive(model, data)
        except RuntimeError as exc:
            if sys.platform == "darwin":
                print("On macOS the viewer must be started with `mjpython physics_validation.py`.")
            raise SystemExit(f"Could not launch viewer: {exc}")

        with viewer_ctx as viewer:
            for i in range(args.trials):
                if not viewer.is_running():
                    print("Viewer closed; stopping.")
                    break
                results.append(run_simulation(model, data, info, rng, viewer=viewer, trial_idx=i))

    if results:
        n_ok = sum(results)
        print(f"\nSuccess rate: {n_ok}/{len(results)} ({100.0 * n_ok / len(results):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

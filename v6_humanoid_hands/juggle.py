"""Run the G1 upper-body and LEAP Hand three-ball cascade.

Control and contact physics live in jugglebot.humanoid.cascade. This module
provides the v6 adapter, benchmark, viewer and video command-line entry point.
"""

import argparse
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import mujoco

from jugglebot.humanoid.cascade import ContactCascade, DEFAULT_BALL_COUNT, DEFAULT_OBSERVATION_NOISE_STD_M
from jugglebot.humanoid.model import DEFAULT_STAND_HEIGHT_M
from jugglebot.video import VideoRecorder, ViewerPacer


PATTERN = "cascade3"
DEFAULT_THROWS = 20
DEFAULT_BENCHMARK_TIMEOUT_S = 30.0
# Conservative simulated-time budget, separate from the throw cadence.
TIMEOUT_SECONDS_PER_THROW = 0.6
TIMEOUT_MARGIN_S = 5.0
VIDEO_FPS = 60
VIEWER_REPORT_INTERVAL_S = 1.0
PROGRESS_INTERVAL_THROWS = 10
# Framing of the validated fixed-base scene; positions are in metres.
CAMERA_TARGET_X_M = 0.22
CAMERA_TARGET_Y_M = 0.0
CAMERA_TARGET_HEIGHT_OFFSET_M = 0.25
CAMERA_DISTANCE_M = 1.30
CAMERA_AZIMUTH_DEG = 180.0
CAMERA_ELEVATION_DEG = -10.0


def benchmark_timeout(throws: int, minimum_s: float = 0.0) -> float:
    if throws < 1:
        raise ValueError("throws must be positive")
    return max(minimum_s, throws * TIMEOUT_SECONDS_PER_THROW + TIMEOUT_MARGIN_S)


class ArmState(Enum):
    READY = auto()
    THROW = auto()
    CUSHION = auto()


@dataclass
class BallTracker:
    """Model addresses used by the viewer and contact regression tests."""

    id: int
    name: str
    body_id: int
    qadr: int
    vadr: int


class _CascadeArmView:
    """Expose the arm/finger interface used by the viewer and contact tests."""

    def __init__(self, cascade, side):
        self.cascade = cascade
        self.model, self.data = cascade.model, cascade.data
        self.side = side
        self.prefix = "rh_" if side == "right" else "lh_"
        hand = cascade.hands[side]
        self.fingers = hand["fingers"]
        self.kin = hand["kin"]
        self.q_nominal = hand["q"]
        self.qpos_adr = hand["kin"].qpos_adr
        self.act_adr = hand["acts"]

    @property
    def held_ball_id(self):
        return self.cascade.hands[self.side]["held"]

    @property
    def state(self):
        mode = self.cascade.hands[self.side]["mode"]
        if mode == "throw":
            return ArmState.THROW
        if mode in ("receive", "carry"):
            return ArmState.CUSHION
        return ArmState.READY


class HumanoidJuggler:
    """Adapter for the validated three-ball contact controller only."""

    def __init__(
        self,
        pattern: str = PATTERN,
        n_balls: int = DEFAULT_BALL_COUNT,
        torso_stand_height: float = DEFAULT_STAND_HEIGHT_M,
        obs_noise: float = DEFAULT_OBSERVATION_NOISE_STD_M,
        tau: float | None = None,
        T_f: float | None = None,
        seed: int = 0,
    ):
        if pattern != PATTERN or n_balls != DEFAULT_BALL_COUNT:
            raise ValueError("v6 supports only pattern='cascade3' with n_balls=3")
        self.pattern = pattern
        self.n_balls = n_balls
        self.obs_noise = obs_noise
        self.torso_stand_height = torso_stand_height
        # Omitted overrides use ContactCascade's authoritative task defaults.
        timing = {}
        if tau is not None:
            timing["tau"] = tau
        if T_f is not None:
            timing["flight"] = T_f
        self._cascade = ContactCascade(
            torso_stand_height=torso_stand_height, obs_noise=obs_noise, seed=seed, **timing
        )
        self._cascade.progress_interval = PROGRESS_INTERVAL_THROWS
        self.tau, self.T_f = self._cascade.tau, self._cascade.flight
        self.model, self.data = self._cascade.model, self._cascade.data
        self.r_arm = _CascadeArmView(self._cascade, "right")
        self.l_arm = _CascadeArmView(self._cascade, "left")
        self.r_seat_id = self.model.site("r_ball_seat").id
        self.l_seat_id = self.model.site("l_ball_seat").id
        self.balls = [
            BallTracker(i, f"ball{i}", self.model.body(f"ball{i}").id, qadr, vadr)
            for i, (qadr, vadr) in enumerate(self._cascade.balls)
        ]
        self.ball_target = self._cascade.targets

    def __getattr__(self, name):
        cascade = self.__dict__.get("_cascade")
        if cascade is not None:
            return getattr(cascade, name)
        raise AttributeError(name)

    def benchmark_complete(self, throws):
        return self.throw_count >= throws and set(range(1, throws + 1)).issubset(self.caught_events)

    def step(self) -> None:
        """Advance one model timestep; ball motion remains contact physics."""
        self._cascade.step()

    def run_benchmark(self, max_throws: int = DEFAULT_THROWS,
                      timeout: float = DEFAULT_BENCHMARK_TIMEOUT_S) -> bool:
        timeout = benchmark_timeout(max_throws, timeout)
        print(f"[Benchmark] Verifying {max_throws} released-and-caught flights with pure contact and finger support...")
        self.benchmark_result = self._cascade.run_continuous(max_throws, timeout, guard=True)
        result = self.benchmark_result
        print(f"      Success: {result['success']} | validated flights: {result['validated_flights']} | "
              f"drops: {result['drops']} | ball contacts: {result['ball_contacts']} | "
              f"minimum center distance: {result['min_distance'] * 1000:.1f} mm")
        print(f"      Releases by hand: {result['releases_by_hand']} | catches by hand: {result['catches_by_hand']} | "
              f"completed requested flights by throwing hand: {result['completed_requested_by_throwing_hand']}")
        flight_range = result['flight_time_range_s']
        if flight_range is not None:
            print(f"      Separation-to-palm-contact time: {flight_range[0]*1000:.0f}-{flight_range[1]*1000:.0f} ms")
        execution = result['execution']
        if execution['samples']:
            rows = execution['actuators']
            speed = max(rows, key=lambda row: row['peak_speed_rad_s'])
            torque = max(rows, key=lambda row: row['peak_actuator_torque_nm'])
            near = max((row['near_force_limit_fraction'] or 0.) for row in rows)
            print(f"      Execution demand: peak speed {speed['peak_speed_rad_s']:.3f} rad/s ({speed['joint']}), "
                  f"peak actuator torque {torque['peak_actuator_torque_nm']:.3f} Nm ({torque['joint']}), "
                  f"max near-force-limit fraction {near:.1%}; hardware limits unverified")
        if result['error']:
            print(f"      Failure: {result['error']}")
        return result["success"]


def cascade_camera(torso_stand_height=DEFAULT_STAND_HEIGHT_M):
    """Frame both hands and the full ballistic arcs, including the apex."""
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [CAMERA_TARGET_X_M, CAMERA_TARGET_Y_M,
                        torso_stand_height + CAMERA_TARGET_HEIGHT_OFFSET_M]
    camera.distance = CAMERA_DISTANCE_M
    camera.azimuth, camera.elevation = CAMERA_AZIMUTH_DEG, CAMERA_ELEVATION_DEG
    return camera


def hide_calibration_sites(scene):
    """Keep palm calibration markers out of the presentation view."""
    for geom in scene.geoms[:scene.ngeom]:
        if geom.objtype == mujoco.mjtObj.mjOBJ_SITE:
            geom.rgba[3] = 0.


def run_viewer(juggler: HumanoidJuggler, speed: float = 1.0,
               infinite: bool = True, max_throws: int = DEFAULT_THROWS) -> None:
    """Launch the interactive viewer for the validated contact cascade."""
    import mujoco.viewer

    print("\nJuggleBot v6: G1 upper-body and dual LEAP Hand")
    print(f"Pattern: {juggler.pattern.upper()} | Speed: {speed}x | Position prediction noise: {juggler.obs_noise*1000:.1f}mm")
    print(">>> Space to Pause/Resume | Drag mouse to orbit | Esc to exit <<<")
    with mujoco.viewer.launch_passive(juggler.model, juggler.data) as viewer:
        viewer.opt.sitegroup[:] = 0
        camera = cascade_camera(juggler.torso_stand_height)
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance = camera.distance
        viewer.cam.azimuth, viewer.cam.elevation = camera.azimuth, camera.elevation
        pacer = ViewerPacer(viewer, speed=speed, fps=VIDEO_FPS)
        last_print = 0.0
        while viewer.is_running():
            juggler.step()
            pacer.capture(juggler.data)
            t = juggler.data.time
            if t - last_print >= VIEWER_REPORT_INTERVAL_S:
                print(f"[{t:5.1f}s] Throws: {juggler.throw_count:3d} | Catches: {juggler.catch_count:3d} | State: R={juggler.r_arm.state.name:<7s} L={juggler.l_arm.state.name:<7s}")
                last_print = t
            if not infinite and juggler.benchmark_complete(max_throws):
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", choices=[PATTERN], default=PATTERN,
                        help="Validated three-ball contact cascade.")
    parser.add_argument("--throws", type=int, default=DEFAULT_THROWS,
                        help="Number of released-and-caught flights to complete.")
    parser.add_argument("--viewer", action="store_true", help="Launch interactive 3D viewer.")
    parser.add_argument("--speed", type=float, default=1.0, help="Viewer/video speed (0.5 for half speed).")
    parser.add_argument("--infinite", action="store_true", help="Run the viewer until it is closed.")
    parser.add_argument("--obs-noise", type=float, default=DEFAULT_OBSERVATION_NOISE_STD_M,
                        help="Partial receive-position prediction noise (m), not full sensor simulation.")
    parser.add_argument("--seed", type=int, default=0, help="Nonnegative seed for partial observation noise.")
    parser.add_argument("--tau", type=float, help="Seconds between alternating throws; omitted uses the G1 calibration.")
    parser.add_argument("--flight", type=float, help="Target flight time in seconds; omitted uses the G1 calibration.")
    parser.add_argument("--video", type=Path, help="Save simulation to MP4 video path.")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.throws < 1 or args.speed <= 0 or args.obs_noise < 0:
        parser.error("--throws and --speed must be positive; --obs-noise must be nonnegative")
    if (args.tau is not None and args.tau <= 0) or (args.flight is not None and args.flight <= 0):
        parser.error("--tau and --flight must be positive")
    juggler = HumanoidJuggler(obs_noise=args.obs_noise, tau=args.tau, T_f=args.flight, seed=args.seed)
    print(f"[Setup] cascade3 | tau: {juggler.tau:.3f} s | target flight: {juggler.T_f:.3f} s | "
          f"ball diameter: {2*juggler.model.geom_size[juggler.model.geom('ball0').id, 0]*1000:.0f} mm | "
          f"observation: simulation truth (partial prediction noise {args.obs_noise*1000:.1f} mm, seed {args.seed})")
    conditions = juggler.conditions_summary()
    print(f"[Hand] position gain: {conditions['finger_position_gain']} | velocity gain: {conditions['finger_velocity_gain']} | "
          f"soft finger links: {conditions['soft_finger_links']} (v5: False)")
    if args.viewer:
        run_viewer(juggler, speed=args.speed, infinite=args.infinite, max_throws=args.throws)
    elif args.video:
        print(f"[Video] Recording {args.throws} throws to {args.video}...")
        rec = VideoRecorder(juggler.model, args.video, fps=VIDEO_FPS, speed=args.speed,
                            camera=cascade_camera(juggler.torso_stand_height))
        max_time = benchmark_timeout(args.throws)
        try:
            while not juggler.benchmark_complete(args.throws) and juggler.data.time < max_time:
                juggler.step()
                rec.capture(juggler.data, scene_callback=hide_calibration_sites)
        finally:
            rec.close()
        result = juggler._cascade.run_continuous(args.throws, juggler.data.time, guard=True)
        print(f"      Validation: {result['validated_flights']}/{args.throws} complete flights | "
              f"drops: {result['drops']} | ball contacts: {result['ball_contacts']}")
        print(f"      Releases by hand: {result['releases_by_hand']} | catches by hand: {result['catches_by_hand']} | "
              f"completed requested flights by throwing hand: {result['completed_requested_by_throwing_hand']}")
        if not result["success"]:
            raise RuntimeError(f"Video validation failed: {result}")
        print(f"      Saved: {args.video} ({rec.frames} frames).")
    elif not juggler.run_benchmark(max_throws=args.throws):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

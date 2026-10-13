"""Offscreen MP4 recording of a simulation (for the milestone videos).

Frames are rendered every `speed / fps` s of simulated time, so speed = 0.25
gives a 4x slow motion clip at the same frame rate.
"""

import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco


class VideoRecorder:
    def __init__(self, model: mujoco.MjModel, path: Path, fps: int = 60, speed: float = 1.0,
                 width: int = 1280, height: int = 720, camera: str | int = "side"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = camera
        self.frame_dt = speed / fps  # simulated seconds per frame
        self.next_t = 0.0
        self.writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                                         macro_block_size=8)
        self.frames = 0

    def capture(self, data: mujoco.MjData, scene_callback=None) -> None:
        """Render a frame if it is due. scene_callback(scene) may add marker geoms."""
        if data.time + 1e-9 < self.next_t:
            return
        # max(): a recording that starts later than t = 0 must not first catch up frame by frame
        self.next_t = max(self.next_t, data.time) + self.frame_dt
        self.renderer.update_scene(data, camera=self.camera)
        if scene_callback is not None:
            scene_callback(self.renderer.scene)
        self.writer.append_data(self.renderer.render())
        self.frames += 1

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()


class ViewerPacer:
    """Drop-in for VideoRecorder.capture() that drives a passive mujoco.viewer window instead:
    syncs at ~60 Hz and paces the simulation to `speed` x real time (0.25 = 4x slow motion)."""

    def __init__(self, viewer, speed: float = 0.25, fps: int = 60):
        self.viewer, self.speed = viewer, speed
        self.frame_dt = speed / fps
        self.next_t = self.last_t = 0.0
        self.wall0 = self.sim0 = None

    def capture(self, data: mujoco.MjData, scene_callback=None) -> None:
        if self.wall0 is None or data.time < self.last_t:  # first frame, or the sim was reset
            self.wall0, self.sim0, self.next_t = time.perf_counter(), data.time, data.time
        self.last_t = data.time
        if data.time + 1e-9 < self.next_t:
            return
        self.next_t = data.time + self.frame_dt
        lag = (data.time - self.sim0) / self.speed - (time.perf_counter() - self.wall0)
        if lag > 0:
            time.sleep(lag)
        self.viewer.sync()

    def linger(self, seconds: float = 1.0) -> None:
        end = time.perf_counter() + seconds
        while self.viewer.is_running() and time.perf_counter() < end:
            time.sleep(1 / 60)
            self.viewer.sync()
        self.wall0 = None

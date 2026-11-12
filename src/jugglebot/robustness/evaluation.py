"""Live-only evaluation telemetry and video capture, isolated from planning."""
import math
import numpy as np

from .controller import ObservedCascade


class EvaluatedCascade(ObservedCascade):
    def __init__(self, *args, recorder_factory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = recorder_factory(self.model) if recorder_factory else None
        self.error_count = self.invalid_count = 0
        self.position_squared_error = self.velocity_squared_error = 0.
        self.max_position_error = self.max_velocity_error = 0.
        self.max_measurement_age = 0.
        if self.recorder:
            self.recorder.capture(self.data, scene_callback=hide_sites)

    def step(self):
        live = not self.silent
        super().step()
        if not live:
            return
        # Evaluation truth never feeds the controller. Sampling remains on the
        # original pre-control clock: do not update the observer after stepping.
        for ball in self.airborne:
            if self.ball_observer.mode != 'sampled':
                continue
            state = self.ball_observer.estimate(ball, float(self.data.time))
            if state is None or not state.valid or state.acceleration_m_s2 is None:
                self.invalid_count += 1
                continue
            q, v = self.balls[ball]
            ep = float(np.linalg.norm(np.asarray(state.position_m)-self.data.qpos[q:q+3]))
            ev = float(np.linalg.norm(np.asarray(state.velocity_m_s)-self.data.qvel[v:v+3]))
            self.error_count += 1
            self.position_squared_error += ep*ep
            self.velocity_squared_error += ev*ev
            self.max_position_error = max(self.max_position_error, ep)
            self.max_velocity_error = max(self.max_velocity_error, ev)
            self.max_measurement_age = max(self.max_measurement_age, state.age_s)
        if self.recorder:
            self.recorder.capture(self.data, scene_callback=hide_sites)

    def estimation_summary(self):
        n = self.error_count
        return dict(scope='live airborne ball states, once per physics step', samples=n,
                    invalid_samples=self.invalid_count,
                    position_rmse_m=math.sqrt(self.position_squared_error/n) if n else None,
                    velocity_rmse_m_s=math.sqrt(self.velocity_squared_error/n) if n else None,
                    max_position_error_m=self.max_position_error if n else None,
                    max_velocity_error_m_s=self.max_velocity_error if n else None,
                    max_measurement_age_s=self.max_measurement_age if n else None)


def hide_sites(scene):
    import mujoco
    for geom in scene.geoms[:scene.ngeom]:
        if geom.objtype == mujoco.mjtObj.mjOBJ_SITE:
            geom.rgba[3] = 0.

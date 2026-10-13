"""Ball trajectory prediction (step 1 of the catching pipeline).

The ball is a point mass under constant gravity (MuJoCo applies no air drag by
default), so its trajectory is the parabola

    p(t) = p0 + v0 (t - t_ref) + 1/2 g (t - t_ref)^2.

With g known, p0 and v0 enter linearly, so they are recovered from noisy
position observations by ordinary least squares, independently per axis.
"""

import numpy as np


class BallisticPredictor:
    def __init__(self, gravity: np.ndarray):
        self.g = np.asarray(gravity, dtype=float)
        self.t_ref = 0.0
        self.p0 = np.zeros(3)
        self.v0 = np.zeros(3)

    def fit(self, times: np.ndarray, positions: np.ndarray) -> None:
        """Least-squares fit of (p0, v0) to observed positions (N x 3) at the given times."""
        times = np.asarray(times, dtype=float)
        positions = np.asarray(positions, dtype=float)
        if len(times) < 2:
            raise ValueError("need at least two observations")
        self.t_ref = float(times[-1])  # reference at the latest sample keeps extrapolation well-conditioned
        dt = times - self.t_ref
        # Remove the known gravity term; what remains is linear in (p0, v0).
        residual = positions - 0.5 * np.outer(dt ** 2, self.g)
        a = np.column_stack([np.ones_like(dt), dt])
        coef, *_ = np.linalg.lstsq(a, residual, rcond=None)
        self.p0, self.v0 = coef[0], coef[1]

    def predict(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicted position and velocity at absolute time(s) t."""
        dt = np.asarray(t, dtype=float) - self.t_ref
        if dt.ndim == 0:
            return self.p0 + self.v0 * dt + 0.5 * self.g * dt ** 2, self.v0 + self.g * dt
        dt = dt[:, None]
        return self.p0 + self.v0 * dt + 0.5 * self.g * dt ** 2, self.v0 + self.g * dt

    def time_at_height(self, z: float) -> float | None:
        """Latest time (descending branch) at which the ball centre is at height z, if any."""
        # 1/2 g_z s^2 + v0_z s + (p0_z - z) = 0, with s = t - t_ref and g_z < 0.
        a, b, c = 0.5 * self.g[2], self.v0[2], self.p0[2] - z
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        s = (-b - np.sqrt(disc)) / (2 * a)  # larger root because a < 0
        return self.t_ref + float(s)

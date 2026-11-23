"""Position-only ball estimation using measured support-palm kinematics."""
from bisect import bisect_right
from dataclasses import replace
import numpy as np
from .reception_control import ContactPositionTracker


class SupportPositionTracker(ContactPositionTracker):
    """Position-only tracking with measured support-palm motion as an input."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hand_times = []
        self.hand_samples = []

    def notify_support(self, now_s, positions, velocities, side):
        positions, velocities = np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)
        if (positions.shape != (2, 3) or velocities.shape != (2, 3)
                or not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities))
                or side not in (None, 0, 1) or not np.isfinite(now_s) or now_s < 0
                or self.hand_times and now_s <= self.hand_times[-1]):
            raise ValueError('Require monotonic measured palm poses/velocities and a support side')
        self.hand_times.append(float(now_s))
        self.hand_samples.append((positions.copy(), velocities.copy(), side))
        self._prediction = None
        velocity = np.zeros(3) if side is None else velocities[side]
        super().notify_contact(side is not None, now_s, velocity)

    def _side_at(self, stamp):
        index = bisect_right(self.hand_times, stamp+1e-12)-1
        if index < 0:
            return None
        return self.hand_samples[index][2]

    def _hand_at(self, stamp, side):
        index = bisect_right(self.hand_times, stamp+1e-12)-1
        if index < 0:
            raise ValueError('No palm observation at this sampling time')
        position, velocity, _ = self.hand_samples[index]
        if index+1 == len(self.hand_times) or abs(stamp-self.hand_times[index]) < 1e-12:
            return position[side]+(stamp-self.hand_times[index])*velocity[side], velocity[side]
        later_position, later_velocity, _ = self.hand_samples[index+1]
        fraction = (stamp-self.hand_times[index])/(self.hand_times[index+1]-self.hand_times[index])
        return ((1-fraction)*position[side]+fraction*later_position[side],
                (1-fraction)*velocity[side]+fraction*later_velocity[side])

    def _relative_fit(self):
        if len(self.history) < 3:
            return None
        sides = [self._side_at(frame.sample_time_s) for frame in self.history]
        if sides[0] is None or any(side != sides[0] for side in sides):
            return None
        palms = [self._hand_at(frame.sample_time_s, sides[0]) for frame in self.history]
        times = np.array([frame.sample_time_s-self.history[-1].sample_time_s for frame in self.history])
        relative = np.array([frame.position_m for frame in self.history])-np.array([pair[0] for pair in palms])
        position, velocity = np.linalg.lstsq(np.column_stack([np.ones(len(times)), times]), relative, rcond=None)[0]
        residual = relative-position-times[:, None]*velocity
        return position+palms[-1][0], velocity+palms[-1][1], times, residual

    def _latest_fit(self, acceleration=None):
        contact = self.in_contact if acceleration is None else np.all(np.asarray(acceleration) == 0.)
        fitted = self._relative_fit() if contact else None
        return super()._latest_fit(acceleration) if fitted is None else fitted[:2]

    def _flight_uncertainty(self, now_s, acceleration=None):
        contact = self.in_contact if acceleration is None else np.all(np.asarray(acceleration) == 0.)
        fitted = self._relative_fit() if contact else None
        if fitted is None:
            return super()._flight_uncertainty(now_s, acceleration)
        _, _, times, residual = fitted
        variance = max(self.measurement_variance, float(np.sum(residual**2)/(3*(len(times)-2))))
        spread = float(np.sum((times-times.mean())**2))
        age = now_s-self.history[-1].sample_time_s
        uncertainty = np.sqrt(variance*np.array([1/len(times)+(age-times.mean())**2/spread, 1/spread]))
        return uncertainty, variance

    def _advance(self, state, covariance, start, target, contact, side):
        dt = target-start
        state, covariance = super()._propagate(state, covariance, dt, contact)
        if contact and dt > 0:
            p0, v0 = self._hand_at(start, side)
            p1, v1 = self._hand_at(target, side)
            state[:3] += p1-p0-dt*v0
            state[3:] += v1-v0
        return state, covariance

    def _predict(self, target):
        if self._prediction is not None and self._prediction[0] == target:
            return self._prediction[1], self._prediction[2]
        state, covariance = self.state.copy(), self.covariance.copy()
        stamp = self.state_time
        contact, side = False, None
        for event in self.events:
            if event[0] <= stamp+1e-12:
                contact = event[1]
                side = self._side_at(event[0]) if contact else None
            elif event[0] <= target+1e-12:
                state, covariance = self._advance(state, covariance, stamp, event[0], contact, side)
                state, covariance = self._impact(state, covariance, event)
                stamp, contact = event[0], event[1]
                side = self._side_at(stamp) if contact else None
        state, covariance = self._advance(state, covariance, stamp, target, contact, side)
        self._prediction = (target, state, covariance)
        return state, covariance

    def _check_query_time(self, now_s):
        if not np.isfinite(now_s) or now_s < 0:
            raise ValueError('Query time must be finite and nonnegative')
        if self.hand_times and now_s < self.hand_times[-1]-1e-12:
            raise ValueError('Cannot query an online estimate before recorded proprioception')

    def update(self, observation):
        super().update(observation)
        if self.state is None and len(self.history) >= 3 and self.hand_times:
            _, velocities, side = self.hand_samples[-1]
            velocity = np.zeros(3) if side is None else velocities[side]
            # Initialize on receipt of the third frame. Waiting until the next
            # tactile tick would briefly use the free-flight gravity model
            # during contact, especially for delayed frames.
            super().notify_contact(side is not None, self.hand_times[-1], velocity)

    def estimate(self, now_s):
        self._check_query_time(now_s)
        estimate = super().estimate(now_s)
        if estimate is None or not self.in_contact or len(self.hand_times) < 2:
            return estimate
        if now_s > self.hand_times[-1]+1e-12:
            return replace(estimate, acceleration_m_s2=(0., 0., 0.))
        side = self.hand_samples[-1][2]
        if side is None or side != self.hand_samples[-2][2]:
            return estimate
        dt = self.hand_times[-1]-self.hand_times[-2]
        acceleration = (self.hand_samples[-1][1][side]-self.hand_samples[-2][1][side])/dt
        return replace(estimate, acceleration_m_s2=tuple(acceleration))

    def uncertainty(self, now_s):
        self._check_query_time(now_s)
        return super().uncertainty(now_s)

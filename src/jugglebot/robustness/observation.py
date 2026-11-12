"""v7 timestamped world-frame ball observations and position-only estimation.

All times are seconds on the simulation clock. Sampling and receipt times are
distinct even though the current source has no transport delay. This module
does not modify physics, add noise, or read velocities in sampled mode.
"""
from collections import deque
from dataclasses import dataclass, replace
import math

import numpy as np

DEFAULT_OBSERVATION_PERIOD_S = 1/60
DEFAULT_OBSERVATION_MAX_AGE_S = .05


def _vector(value):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError('Expected a finite world-frame 3-vector')
    return tuple(float(x) for x in array)


@dataclass(frozen=True)
class BallObservation:
    ball_id: int
    sample_time_s: float
    received_time_s: float
    position_m: tuple[float, float, float]
    valid: bool = True
    velocity_m_s: tuple[float, float, float] | None = None

    def __post_init__(self):
        if (not math.isfinite(self.sample_time_s) or not math.isfinite(self.received_time_s)
                or self.sample_time_s < 0 or self.received_time_s < self.sample_time_s):
            raise ValueError('Observation timestamps must satisfy 0 <= sample <= receipt')
        object.__setattr__(self, 'position_m', _vector(self.position_m))
        if self.velocity_m_s is not None:
            object.__setattr__(self, 'velocity_m_s', _vector(self.velocity_m_s))


@dataclass(frozen=True)
class BallEstimate:
    ball_id: int
    state_time_s: float
    sample_time_s: float
    received_time_s: float
    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float] | None
    acceleration_m_s2: tuple[float, float, float] | None
    valid: bool

    @property
    def age_s(self):
        return self.state_time_s - self.sample_time_s


class PositionEstimator:
    """Causal backward differences with a local constant-acceleration prediction.

One position is insufficient for velocity (invalid estimate). Two yield a
secant velocity; three yield velocity/acceleration from their quadratic
interpolant. No truth velocity or ballistic phase label is used. This is a
noiseless reference estimator; it is not a noise-filtering tracker.
"""

    def __init__(self, ball_id, max_age_s=DEFAULT_OBSERVATION_MAX_AGE_S, time_quantum_s=0.):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('max_age_s must be finite and positive')
        if not math.isfinite(time_quantum_s) or time_quantum_s < 0:
            raise ValueError('time_quantum_s must be finite and nonnegative')
        self.ball_id = ball_id
        self.max_age_s = float(max_age_s)
        self.time_quantum_s = float(time_quantum_s)
        self.reset()

    def _elapsed(self, start, stop):
        interval = stop-start
        if self.time_quantum_s:
            # MuJoCo integrates with h while data.time accumulates floating
            # additions. Recover elapsed physical ticks without changing stamps.
            return round(interval/self.time_quantum_s)*self.time_quantum_s
        return interval

    def reset(self):
        self._history = deque(maxlen=3)
        self._last = None

    def update(self, observation):
        if observation.ball_id != self.ball_id:
            raise ValueError('Observation belongs to another ball')
        if self._last is not None and (
                observation.sample_time_s <= self._last.sample_time_s
                or observation.received_time_s < self._last.received_time_s
                or self._elapsed(self._last.sample_time_s, observation.sample_time_s) <= 0):
            return False  # Duplicate/out-of-order data cannot change history.
        if (not observation.valid or (self._last is not None and
                self._elapsed(self._last.sample_time_s, observation.sample_time_s) > self.max_age_s)):
            self._history.clear()
        self._last = observation
        if observation.valid:
            self._history.append(observation)
        return True

    def estimate(self, now_s):
        if not math.isfinite(now_s):
            raise ValueError('Query time must be finite')
        last = self._last
        if last is None:
            return None
        if now_s < last.received_time_s:
            raise ValueError('Cannot use an observation before receipt')
        position = np.asarray(last.position_m)
        velocity = acceleration = None
        if len(self._history) >= 2:
            previous = self._history[-2]
            dt = self._elapsed(previous.sample_time_s, last.sample_time_s)
            velocity = (position - previous.position_m) / dt
            if len(self._history) == 3:
                first = self._history[0]
                earlier_dt = self._elapsed(first.sample_time_s, previous.sample_time_s)
                earlier_velocity = (np.asarray(previous.position_m) - first.position_m) / earlier_dt
                acceleration = 2 * (velocity - earlier_velocity) / (dt + earlier_dt)
                velocity = velocity + .5 * dt * acceleration
        age = self._elapsed(last.sample_time_s, now_s)
        if velocity is not None:
            position = position + age * velocity
            if acceleration is not None:
                position = position + .5 * age**2 * acceleration
                velocity = velocity + age * acceleration
        return BallEstimate(
            self.ball_id, float(now_s), last.sample_time_s, last.received_time_s,
            _vector(position), None if velocity is None else _vector(velocity),
            None if acceleration is None else _vector(acceleration),
            bool(last.valid and velocity is not None and age <= self.max_age_s),
        )


class MuJoCoBallSource:
    """Read detached snapshots from explicit free-joint addresses."""

    def __init__(self, ball_addresses):
        self.ball_addresses = tuple(ball_addresses)

    def read(self, data, include_velocity=False):
        stamp = float(data.time)
        return tuple(BallObservation(
            i, stamp, stamp, _vector(data.qpos[q:q+3]),
            velocity_m_s=_vector(data.qvel[v:v+3]) if include_velocity else None,
        ) for i, (q, v) in enumerate(self.ball_addresses))

    def truth(self, data, ball_id):
        q, v = self.ball_addresses[ball_id]
        stamp = float(data.time)
        return BallEstimate(ball_id, stamp, stamp, stamp, _vector(data.qpos[q:q+3]),
                            _vector(data.qvel[v:v+3]), None, True)


class BallObserver:
    """Live observation history; private planner rollouts must not update it.

    'truth' preserves the existing exact position/velocity input. 'sampled'
    reads positions at the requested period, quantized to physics ticks, and
    estimates at control time. Changing data.time backwards requires reset().
    """

    def __init__(self, ball_addresses, mode='truth', sample_period_s=DEFAULT_OBSERVATION_PERIOD_S,
                 max_age_s=DEFAULT_OBSERVATION_MAX_AGE_S, position_step_s=0.):
        if mode not in ('truth', 'sampled'):
            raise ValueError("observation mode must be 'truth' or 'sampled'")
        if not math.isfinite(sample_period_s) or sample_period_s <= 0:
            raise ValueError('sample_period_s must be finite and positive')
        if not math.isfinite(max_age_s) or max_age_s < sample_period_s:
            raise ValueError('max_age_s must be finite and at least the sample period')
        if not math.isfinite(position_step_s) or position_step_s < 0:
            raise ValueError('position_step_s must be finite and nonnegative')
        self.mode = mode
        self.sample_period_s = float(sample_period_s)
        self.max_age_s = float(max_age_s)
        self.position_step_s = float(position_step_s)
        self.source = MuJoCoBallSource(ball_addresses)
        self.estimators = tuple(PositionEstimator(i, max_age_s, position_step_s)
                                for i in range(len(self.source.ball_addresses)))
        self.reset()

    def reset(self):
        for estimator in self.estimators:
            estimator.reset()
        self.observations = ()
        self._last_update_s = None
        self._origin_s = None
        self._next_sample = 0

    def update(self, data):
        now = float(data.time)
        if not math.isfinite(now) or now < 0:
            raise ValueError('Simulation time must be finite and nonnegative')
        if self._last_update_s is not None:
            if now < self._last_update_s:
                raise ValueError('Clock moved backwards; reset the observer before reusing it')
            if now == self._last_update_s:
                return False
        self._last_update_s = now
        if self._origin_s is None:
            self._origin_s = now
        if (self.mode == 'sampled' and
                now + 1e-12 < self._origin_s + self._next_sample * self.sample_period_s):
            return False
        self.observations = self.source.read(data, include_velocity=self.mode == 'truth')
        if self.mode == 'sampled':
            for estimator, observation in zip(self.estimators, self.observations):
                estimator.update(observation)
            # Skipped periods are not backfilled with fabricated measurements.
            self._next_sample = math.floor((now-self._origin_s + 1e-12) / self.sample_period_s) + 1
        return True

    def estimate(self, ball_id, now_s):
        if not 0 <= ball_id < len(self.estimators):
            raise IndexError(ball_id)
        if self.mode == 'sampled':
            state = self.estimators[ball_id].estimate(now_s)
            if (state is not None and state.acceleration_m_s2 is not None
                    and self.position_step_s):
                # Semi-implicit q[k+1] = q[k] + h*v[k+1]. The derivative
                # of a quadratic fit to these positions is v + h*a/2.
                # Positions already predict the discrete trajectory correctly;
                # align only velocity to the simulator's endpoint convention.
                # Use estimated acceleration, never simulator qvel/qacc.
                velocity = np.asarray(state.velocity_m_s) - .5*self.position_step_s*np.asarray(state.acceleration_m_s2)
                state = replace(state, velocity_m_s=_vector(velocity))
            return state
        if not self.observations:
            return None
        last = self.observations[ball_id]
        if now_s != last.sample_time_s:
            raise ValueError('Truth mode requires update at the query time')
        return BallEstimate(ball_id, float(now_s), last.sample_time_s, last.received_time_s,
                            last.position_m, last.velocity_m_s, None, last.valid)

    def conditions(self):
        return dict(mode=self.mode, frame='world', clock='simulation_seconds',
                    sample_period_s=self.sample_period_s if self.mode == 'sampled' else None,
                    max_age_s=self.max_age_s if self.mode == 'sampled' else None,
                    position_step_s=self.position_step_s if self.mode == 'sampled' else None,
                    interval_clock=('integer_physics_ticks' if self.position_step_s else 'recorded_timestamps')
                                   if self.mode == 'sampled' else None,
                    velocity_convention=('semi_implicit_endpoint' if self.position_step_s else 'position_derivative')
                                        if self.mode == 'sampled' else 'simulation_truth',
                    estimator='three_position_backward_difference' if self.mode == 'sampled' else 'truth',
                    scope='live reception prediction', rollout_input='simulation_truth',
                    contact_and_safety_input='simulation_truth')

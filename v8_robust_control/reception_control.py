"""Causal ballistic position fit and bounded, contact-cushioned palm reference."""
from collections import deque

import numpy as np

from jugglebot.humanoid.planning import Quintic
from jugglebot.robustness.observation import BallEstimate


class BallisticPositionTracker:
    """Fit position and velocity from position-only frames and known gravity.

    Unlike a three-point acceleration estimate, this fit does not differentiate
    noisy position twice. It applies only before contact. No simulator velocity,
    initialization parameters or future measurements are consumed.
    """
    def __init__(self, gravity, physics_dt, max_age_s=.05, history=8):
        self.gravity = np.asarray(gravity, dtype=float).copy()
        self.physics_dt = physics_dt
        self.max_age_s = max_age_s
        self.history = deque(maxlen=history)

    def update(self, observation):
        if observation is None or not observation.valid:
            return
        if self.history and observation.sample_time_s <= self.history[-1].sample_time_s:
            return
        if self.history and observation.sample_time_s-self.history[-1].sample_time_s > self.max_age_s:
            self.history.clear()
        self.history.append(observation)

    def estimate(self, now_s):
        if len(self.history) < 3:
            return None
        last = self.history[-1]
        if now_s < last.received_time_s or now_s-last.sample_time_s > self.max_age_s:
            return None
        position, velocity = self._latest_fit()
        age = now_s-last.sample_time_s
        predicted = position+age*velocity+.5*(age**2+self.physics_dt*age)*self.gravity
        speed = velocity+age*self.gravity
        return BallEstimate(0, now_s, last.sample_time_s, last.received_time_s,
                            tuple(predicted), tuple(speed), tuple(self.gravity), True)

    def _latest_fit(self, acceleration=None):
        last = self.history[-1]
        acceleration = self.gravity if acceleration is None else np.asarray(acceleration)
        times = np.array([o.sample_time_s-last.sample_time_s for o in self.history])
        # q(t+d) = q(t) + v(t)*d + g*(d*d+h*d)/2 under semi-implicit integration.
        corrected = np.array([o.position_m for o in self.history])
        corrected -= .5*(times**2+self.physics_dt*times)[:, None]*acceleration
        design = np.column_stack([np.ones(len(times)), times])
        position, velocity = np.linalg.lstsq(design, corrected, rcond=None)[0]
        return position, velocity


class ContactPositionTracker(BallisticPositionTracker):
    """Position-only Kalman tracking after a measured palm contact.

    Free-flight estimates retain the ballistic fit. Tactile transitions change
    the acceleration prior and its uncertainty, not simulator ball state.
    Delayed frames are updated at sampling time; queries predict to receipt
    time using only contact transitions already observed by that query.
    """
    def __init__(self, gravity, physics_dt, measurement_std_m=.003, process_noise=1., **kwargs):
        super().__init__(gravity, physics_dt, **kwargs)
        self.measurement_variance = measurement_std_m**2
        self.process_noise = process_noise
        self.events = []
        self.state = self.covariance = self.state_time = None
        self.last_observation = None
        self.in_contact = False
        self.contact_updates = 0
        self._prediction = None

    def notify_contact(self, contact, now_s, palm_velocity):
        contact = bool(contact)
        if self.state is None and len(self.history) >= 3:
            acceleration = np.zeros(3) if self.events and self.history[0].sample_time_s >= self.events[0][0] else self.gravity
            p, v = self._latest_fit(acceleration)
            uncertainty, variance = self._flight_uncertainty(self.history[-1].sample_time_s, acceleration)
            self.measurement_variance = variance
            self.state = np.r_[p, v]
            self.covariance = np.diag([max(uncertainty[0]**2, variance)]*3
                                      +[max(uncertainty[1]**2, .1**2)]*3)
            self.last_observation = self.history[-1]
            self.state_time = self.last_observation.sample_time_s
            self._prediction = None
        if contact == self.in_contact:
            return
        if self.events and now_s < self.events[-1][0]:
            raise ValueError('Contact event times must be monotonic')
        self.events.append((float(now_s), contact, np.asarray(palm_velocity).copy()))
        self._prediction = None
        self.in_contact = contact
        if self.state is not None and abs(now_s-self.state_time) < 1e-12:
            self.state, self.covariance = self._impact(self.state, self.covariance, self.events[-1])

    def _impact(self, state, covariance, event):
        state, covariance = state.copy(), covariance.copy()
        if event[1]:
            # A noisy support-velocity prior from robot proprioception. Actual
            # ball velocity is inferred from subsequent position measurements.
            state[3:] = event[2]
        covariance[3:, 3:] += .5**2*np.eye(3)
        return state, covariance

    def _propagate(self, state, covariance, dt, contact):
        if dt <= 0:
            return state, covariance
        transition = np.eye(6)
        transition[:3, 3:] = dt*np.eye(3)
        acceleration = np.zeros(3) if contact else self.gravity
        state = transition@state+np.r_[.5*(dt**2+self.physics_dt*dt)*acceleration, dt*acceleration]
        variance = self.process_noise if contact else .25
        noise = variance*np.kron(np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]]), np.eye(3))
        return state, transition@covariance@transition.T+noise

    def _predict(self, target):
        if self._prediction is not None and self._prediction[0] == target:
            return self._prediction[1], self._prediction[2]
        state, covariance = self.state.copy(), self.covariance.copy()
        stamp = self.state_time
        contact = False
        for event in self.events:
            if event[0] <= stamp+1e-12:
                contact = event[1]
            elif event[0] <= target+1e-12:
                state, covariance = self._propagate(state, covariance, event[0]-stamp, contact)
                state, covariance = self._impact(state, covariance, event)
                stamp, contact = event[0], event[1]
        state, covariance = self._propagate(state, covariance, target-stamp, contact)
        self._prediction = (target, state, covariance)
        return state, covariance

    def update(self, observation):
        if self.state is None:
            return super().update(observation)
        if (observation is None or not observation.valid
                or observation.sample_time_s <= self.state_time
                or observation.received_time_s < self.last_observation.received_time_s):
            return
        state, covariance = self._predict(observation.sample_time_s)
        residual = np.asarray(observation.position_m)-state[:3]
        noise = self.measurement_variance*np.eye(3)
        gain = np.linalg.solve(covariance[:3, :3]+noise, covariance[:3, :]).T
        self.state = state+gain@residual
        complement = np.eye(6)
        complement[:, :3] -= gain
        self.covariance = complement@covariance@complement.T+gain@noise@gain.T
        self.state_time = observation.sample_time_s
        self.last_observation = observation
        self._prediction = None
        self.history.append(observation)
        sampled_contact = False
        for event in self.events:
            if event[0] <= observation.sample_time_s:
                sampled_contact = event[1]
        if sampled_contact:
            self.contact_updates += 1

    def estimate(self, now_s):
        if self.events and now_s < self.events[-1][0]:
            raise ValueError('Cannot query before an observed contact event')
        if self.state is None:
            return super().estimate(now_s)
        last = self.last_observation
        if now_s < last.received_time_s or now_s-last.sample_time_s > self.max_age_s:
            return None
        state, _ = self._predict(now_s)
        return BallEstimate(0, now_s, last.sample_time_s, last.received_time_s,
                            tuple(state[:3]), tuple(state[3:]),
                            tuple(np.zeros(3) if self.in_contact else self.gravity), True)

    def uncertainty(self, now_s):
        if self.state is None:
            if len(self.history) < 3:
                return np.array([np.sqrt(self.measurement_variance), .1])
            return self._flight_uncertainty(now_s)[0]
        _, covariance = self._predict(now_s)
        return np.sqrt([np.mean(np.diag(covariance)[:3]), np.mean(np.diag(covariance)[3:])])

    def _flight_uncertainty(self, now_s, acceleration=None):
        """Least-squares covariance from delivered positions, with a sensor floor.

        Contact dynamics do not calibrate measurement noise: the last flight
        fit sets the measurement variance once, at filter initialization.
        """
        acceleration = self.gravity if acceleration is None else np.asarray(acceleration)
        p, v = self._latest_fit(acceleration)
        last = self.history[-1].sample_time_s
        times = np.array([o.sample_time_s-last for o in self.history])
        corrected = np.array([o.position_m for o in self.history])
        corrected -= .5*(times**2+self.physics_dt*times)[:, None]*acceleration
        residual = corrected-p-times[:, None]*v
        variance = max(self.measurement_variance,
                       float(np.sum(residual**2)/(3*(len(times)-2))))
        spread = float(np.sum((times-times.mean())**2))
        age = now_s-last
        uncertainty = np.sqrt(variance*np.array([
            1/len(times)+(age-times.mean())**2/spread, 1/spread]))
        return uncertainty, variance


class ReceptionReference:
    """Slew/acceleration limited interception followed by tactile cushioning."""
    def __init__(self, origin, gravity_m_s2, config):
        self.origin = np.asarray(origin).copy()
        self.gravity = gravity_m_s2
        self.config = config
        self.segment = None
        self.cushion = None
        self.last_intercept = None
        self.updates = 0
        self.rejected_updates = 0
        self.followthrough = None

    def sample(self, time_s):
        if self.cushion is not None:
            return self.cushion.sample(time_s)
        if self.segment is None:
            return self.origin.copy(), np.zeros(3), np.zeros(3)
        if time_s > self.segment.t1 and self.followthrough is not None:
            return self.followthrough.sample(time_s)
        return self.segment.sample(time_s)

    def advance(self, estimate, time_s, in_contact=False):
        if self.cushion is not None or in_contact:
            return
        position, velocity, acceleration = self.sample(time_s)
        if not self.config.feedback_enabled or estimate is None or not estimate.valid:
            return
        ball = np.asarray(estimate.position_m)
        speed = np.asarray(estimate.velocity_m_s)
        height = ball[2]-self.origin[2]-.022
        discriminant = speed[2]**2+2*self.gravity*height
        if discriminant < 0:
            return
        remaining = (speed[2]+np.sqrt(discriminant))/self.gravity
        if not .04 < remaining < .5:
            return
        goal = self.origin.copy()
        goal[:2] = np.clip(ball[:2]+speed[:2]*remaining,
                          self.origin[:2]-self.config.nominal_reach_m,
                          self.origin[:2]+self.config.nominal_reach_m)
        landing_speed = speed.copy()
        landing_speed[2] -= self.gravity*remaining
        landing_speed[:2] *= .35
        landing_speed[2] = np.clip(.35*landing_speed[2], -.7, -.25)
        candidate = Quintic(time_s, remaining, position, velocity, acceleration,
                             goal, landing_speed, np.zeros(3))
        velocity_peak, acceleration_peak = candidate.peaks(n=12)
        if (np.max(velocity_peak) > self.config.nominal_speed_m_s
                or np.max(acceleration_peak) > self.config.nominal_acceleration_m_s2):
            self.rejected_updates += 1
            return
        self.segment = candidate
        stop_duration = min(self.config.cushion_duration_s, .05/abs(landing_speed[2]))
        self.followthrough = Quintic(candidate.t1, stop_duration, goal, landing_speed,
                                     np.zeros(3), goal+.5*stop_duration*landing_speed,
                                     np.zeros(3), np.zeros(3))
        self.last_intercept = goal.copy()
        self.updates += 1

    def capture(self, time_s, palm_position, palm_velocity, residual_offset=None):
        if self.cushion is not None:
            return
        start = np.asarray(palm_position).copy()
        # The servo adds the residual to this nominal reference. The measured
        # palm already contains it: subtract once to avoid a target jump at
        # contact, including when the arm has not fully tracked its command.
        if residual_offset is not None:
            start -= np.asarray(residual_offset)
        goal = start-np.array([0., 0., self.config.cushion_depth_m])
        self.cushion = Quintic(time_s, self.config.cushion_duration_s, start,
                               np.asarray(palm_velocity), np.zeros(3), goal,
                               np.zeros(3), np.zeros(3))

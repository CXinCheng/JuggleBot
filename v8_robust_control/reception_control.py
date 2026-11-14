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
        times = np.array([o.sample_time_s-last.sample_time_s for o in self.history])
        # q(t+d) = q(t) + v(t)*d + g*(d*d+h*d)/2 under semi-implicit integration.
        corrected = np.array([o.position_m for o in self.history])
        corrected -= .5*(times**2+self.physics_dt*times)[:, None]*self.gravity
        design = np.column_stack([np.ones(len(times)), times])
        position, velocity = np.linalg.lstsq(design, corrected, rcond=None)[0]
        age = now_s-last.sample_time_s
        predicted = position+age*velocity+.5*(age**2+self.physics_dt*age)*self.gravity
        speed = velocity+age*self.gravity
        return BallEstimate(0, now_s, last.sample_time_s, last.received_time_s,
                            tuple(predicted), tuple(speed), tuple(self.gravity), True)


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

    def advance(self, estimate, time_s):
        if self.cushion is not None:
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

    def capture(self, time_s, palm_position, palm_velocity):
        if self.cushion is not None:
            return
        start = np.asarray(palm_position).copy()
        goal = start-np.array([0., 0., self.config.cushion_depth_m])
        self.cushion = Quintic(time_s, self.config.cushion_duration_s, start,
                               np.asarray(palm_velocity), np.zeros(3), goal,
                               np.zeros(3), np.zeros(3))

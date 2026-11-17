"""Causal support-motion compensation and contact/free-flight transitions."""
import unittest
import numpy as np

from jugglebot.robustness.observation import BallObservation
from .reception_control import ContactPositionTracker
from .transfer_observation import SupportPositionTracker


class SupportTrackingTests(unittest.TestCase):
    def trajectory(self, delay=0., release=None, side=0, velocity_field=None):
        gravity = np.array([0., 0., -9.81])
        acceleration = np.array([2., -1., 3.])
        tracker = SupportPositionTracker(gravity, .001)
        old = ContactPositionTracker(gravity, .001)
        frames, errors, old_errors, estimates = [], [], [], []
        for tick in range(301):
            time = tick*.001
            palm = .5*(time*time+.001*time)*acceleration
            velocity = time*acceleration
            positions, velocities = np.zeros((2, 3)), np.zeros((2, 3))
            positions[side], velocities[side] = palm, velocity
            contact = release is None or time < release
            tracker.notify_support(time, positions, velocities, side if contact else None)
            old.notify_contact(contact, time, velocity)
            offset = np.array([.01, -.01, .022])
            if contact:
                position, speed = palm+offset, velocity
            else:
                origin = .5*(release*release+.001*release)*acceleration+offset
                flight = time-release
                speed = release*acceleration+flight*gravity
                position = origin+flight*release*acceleration+.5*(flight*flight+.001*flight)*gravity
            if tick % 17 == 0:
                frames.append(BallObservation(0, time, time+delay, tuple(position),
                                               velocity_m_s=velocity_field))
            while frames and frames[0].received_time_s <= time+1e-12:
                frame = frames.pop(0)
                tracker.update(frame)
                old.update(frame)
            estimate, previous = tracker.estimate(time), old.estimate(time)
            if estimate is not None:
                errors.append(np.r_[np.asarray(estimate.position_m)-position,
                                    np.asarray(estimate.velocity_m_s)-speed])
                estimates.append((estimate.position_m, estimate.velocity_m_s))
            if previous is not None:
                old_errors.append(np.asarray(previous.position_m)-position)
        return tracker, np.asarray(errors), np.asarray(old_errors), estimates

    def test_accelerating_support_uses_proprioception_without_ball_velocity(self):
        for side in (0, 1):
            for delay in (0., .02):
                with self.subTest(side=side, delay=delay):
                    tracker, errors, old_errors, _ = self.trajectory(delay=delay, side=side)
                    self.assertGreater(len(errors), 100)
                    self.assertLess(np.max(np.abs(errors)), 1e-9)
                    self.assertGreater(np.sqrt(np.mean(old_errors**2)), .0001)
                    np.testing.assert_allclose(tracker.estimate(.3).acceleration_m_s2, [2., -1., 3.], atol=1e-10)

    def test_release_returns_to_ballistics_and_ignores_later_support_acceleration(self):
        for delay in (0., .02):
            _, errors, _, _ = self.trajectory(delay=delay, release=.17)
            self.assertLess(np.max(np.abs(errors)), 1e-9)

    def test_observation_velocity_field_does_not_enter_the_filter(self):
        _, _, _, ordinary = self.trajectory(delay=.02)
        _, _, _, privileged = self.trajectory(delay=.02, velocity_field=(999., -888., 777.))
        np.testing.assert_array_equal(ordinary, privileged)

    def test_future_queries_do_not_invent_unobserved_palm_acceleration(self):
        tracker, _, _, _ = self.trajectory()
        current = tracker.estimate(.3)
        future = tracker.estimate(.31)
        np.testing.assert_allclose(future.position_m,
                                  np.asarray(current.position_m)+.01*np.asarray(current.velocity_m_s), atol=1e-10)
        np.testing.assert_allclose(future.velocity_m_s, current.velocity_m_s, atol=1e-10)
        np.testing.assert_array_equal(future.acceleration_m_s2, 0.)

    def test_proprioception_does_not_create_or_refresh_ball_measurements(self):
        tracker = SupportPositionTracker([0., 0., -9.81], .001)
        for tick in range(101):
            tracker.notify_support(tick*.001, np.zeros((2, 3)), np.zeros((2, 3)), 0)
            if tick in (0, 17, 34):
                tracker.update(BallObservation(0, tick*.001, tick*.001, (0., 0., .022)))
            if tick < 34 or tick > 84:
                self.assertIsNone(tracker.estimate(tick*.001))
        self.assertEqual(len(tracker.history), 3)

    def test_support_inputs_and_public_query_times_are_causal(self):
        tracker = SupportPositionTracker([0., 0., -9.81], .001)
        positions, velocities = np.zeros((2, 3)), np.zeros((2, 3))
        tracker.notify_support(0., positions, velocities, 0)
        positions[:] = 5.
        np.testing.assert_array_equal(tracker.hand_samples[0][0], 0.)
        for time in (0., -.001, float('nan')):
            with self.assertRaises(ValueError):
                tracker.notify_support(time, positions, velocities, 0)
        tracker.notify_support(.001, positions, velocities, None)
        with self.assertRaises(ValueError):
            tracker.estimate(0.)
        with self.assertRaises(ValueError):
            tracker.uncertainty(0.)
        with self.assertRaises(ValueError):
            tracker.notify_support(.002, np.zeros((3, 3)), velocities, 0)


if __name__ == '__main__':
    unittest.main()

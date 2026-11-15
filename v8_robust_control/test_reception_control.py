"""Position-only tracking, timestamp causality and predictive reception."""
from dataclasses import replace
import unittest
import numpy as np

from jugglebot.robustness.observation import BallObservation, BallEstimate
from .reception_control import BallisticPositionTracker, ContactPositionTracker, ReceptionReference
from .residual_env import CatchConfig


class ReceptionControlTests(unittest.TestCase):
    def test_tracker_matches_discrete_ballistics_without_velocity_measurements(self):
        g = np.array([0., 0., -9.81])
        p = np.array([.3, .2, 1.1])
        v = np.array([.2, -.1, -.3])
        tracker = BallisticPositionTracker(g, .001)
        for t in (0., .017, .034, .05):
            q = p+v*t+.5*g*(t*t+.001*t)
            tracker.update(BallObservation(0, t, t, tuple(q)))
        state = tracker.estimate(.06)
        np.testing.assert_allclose(state.position_m, p+v*.06+.5*g*(.06**2+.001*.06), atol=1e-12)
        np.testing.assert_allclose(state.velocity_m_s, v+.06*g, atol=1e-12)

    def test_tracker_rejects_future_stale_and_duplicate_frames(self):
        tracker = BallisticPositionTracker([0., 0., -9.81], .001)
        for t in (0., .017, .034):
            tracker.update(BallObservation(0, t, t+.02, (.3, .2, 1.1-t)))
        self.assertIsNone(tracker.estimate(.05))
        self.assertIsNone(tracker.estimate(.09))
        before = len(tracker.history)
        tracker.update(BallObservation(0, .017, .04, (100., 100., 100.)))
        self.assertEqual(len(tracker.history), before)

    def test_opposite_measured_landings_produce_opposite_references(self):
        origin = np.array([.3, .22, .86])
        for direction in (-1., 1.):
            reference = ReceptionReference(origin, 9.81, CatchConfig())
            state = BallEstimate(0, 0., 0., 0., tuple(origin+[direction*.03, 0., .18]),
                                 (0., 0., -.3), (0., 0., -9.81), True)
            reference.advance(state, 0.)
            self.assertEqual(reference.updates, 1)
            self.assertGreater(direction*(reference.sample(.1)[0][0]-origin[0]), 0.)
            previous = reference.segment
            reference.advance(None, .02)
            self.assertIs(reference.segment, previous)

    def test_static_reference_ignores_ball_measurement_but_keeps_cushion(self):
        origin = np.array([.3, .22, .86])
        reference = ReceptionReference(origin, 9.81, replace(CatchConfig(), feedback_enabled=False))
        state = BallEstimate(0, 0., 0., 0., (.35, .2, 1.1), (0., 0., -.3), None, True)
        reference.advance(state, 0.)
        np.testing.assert_array_equal(reference.sample(.1)[0], origin)
        reference.capture(.2, origin, np.zeros(3))
        self.assertLess(reference.sample(.3)[0][2], origin[2])

    def test_capture_does_not_apply_a_palm_residual_twice(self):
        origin = np.array([.3, .22, .86])
        offset = np.array([.008, -.006, .004])
        actual_palm = origin+offset
        velocity = np.array([.1, -.1, -.4])
        reference = ReceptionReference(origin, 9.81, CatchConfig())
        reference.capture(.2, actual_palm, velocity, residual_offset=offset)
        position, speed, _ = reference.sample(.2)
        np.testing.assert_allclose(position+offset, actual_palm, atol=1e-12)
        np.testing.assert_allclose(speed, velocity, atol=1e-12)
        final, _, _ = reference.sample(.2+reference.config.cushion_duration_s)
        np.testing.assert_allclose(final+offset, actual_palm-[0., 0., reference.config.cushion_depth_m], atol=1e-12)

    def test_ballistic_replanning_pauses_during_measured_contact(self):
        origin = np.array([.3, .22, .86])
        reference = ReceptionReference(origin, 9.81, CatchConfig())
        state = BallEstimate(0, 0., 0., 0., tuple(origin+[.02, 0., .18]),
                             (0., 0., -.3), (0., 0., -9.81), True)
        reference.advance(state, 0.)
        previous = reference.segment
        reference.advance(state, .02, in_contact=True)
        self.assertIs(reference.segment, previous)
        reference.advance(state, .02, in_contact=False)
        self.assertGreater(reference.updates, 1)


class ContactTrackerTests(unittest.TestCase):
    def test_flight_uncertainty_reflects_position_noise_and_forecast_age(self):
        rng = np.random.default_rng(22)
        errors = rng.normal(size=(8, 3))
        trackers = []
        for sigma in (.001, .015):
            tracker = ContactPositionTracker([0., 0., -9.81], .001)
            for i in range(8):
                stamp = i*.017
                position = np.array([.3+.1*stamp, .2, 1.-.3*stamp-4.905*(stamp**2+.001*stamp)])
                tracker.update(BallObservation(0, stamp, stamp, tuple(position+sigma*errors[i]),
                                               velocity_m_s=(999., 999., 999.)))
            trackers.append(tracker)
        low, high = [tracker.uncertainty(.12) for tracker in trackers]
        self.assertTrue(np.all(high > low*2))
        self.assertGreater(trackers[1].uncertainty(.16)[0], high[0])
        trackers[1].notify_contact(True, .125, np.zeros(3))
        self.assertGreater(trackers[1].measurement_variance, .003**2)

    def test_contact_measurements_keep_state_valid_without_truth_velocity(self):
        tracker = ContactPositionTracker([0., 0., -9.81], .001)
        rng = np.random.default_rng(5)
        initial = np.array([.3, .22, 1.])
        g = np.array([0., 0., -9.81])
        held = initial+.5*g*(.1**2+.001*.1)
        errors = []
        for i in range(25):
            time_s = i*.017
            position = initial+.5*g*(time_s**2+.001*time_s) if time_s < .1 else held
            if time_s >= .1:
                tracker.notify_contact(True, time_s, np.zeros(3))
            tracker.update(BallObservation(0, time_s, time_s, tuple(position+rng.normal(0., .001, 3)),
                                           velocity_m_s=(999., 999., 999.)))
            state = tracker.estimate(time_s)
            if time_s > .2:
                self.assertTrue(state.valid)
                errors.append(np.linalg.norm(np.asarray(state.position_m)-held))
                self.assertLess(np.linalg.norm(state.velocity_m_s), .25)
        self.assertLess(np.mean(errors), .005)
        self.assertGreater(tracker.contact_updates, 5)

    def test_late_frames_and_staleness_cannot_create_current_measurements(self):
        tracker = ContactPositionTracker([0., 0., -9.81], .001)
        for t in (0., .017, .034):
            tracker.update(BallObservation(0, t, t+.02, (.3, .22, 1.-t)))
        tracker.notify_contact(True, .06, np.zeros(3))
        tracker.update(BallObservation(0, .05, .07, (.3, .22, .95)))
        self.assertEqual(tracker.state_time, .05)
        self.assertIsNone(tracker.estimate(.11))
        with self.assertRaises(ValueError):
            tracker.estimate(.059)
        tracker.update(BallObservation(0, .068, .088, (.3, .22, .95)))
        self.assertTrue(tracker.estimate(.09).valid)

    def test_tactile_contact_alone_does_not_fabricate_ball_position(self):
        tracker = ContactPositionTracker([0., 0., -9.81], .001)
        tracker.notify_contact(True, .1, np.zeros(3))
        self.assertIsNone(tracker.estimate(.1))



if __name__ == '__main__':
    unittest.main()

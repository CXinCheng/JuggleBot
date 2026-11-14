"""Position-only tracking, timestamp causality and predictive reception."""
from dataclasses import replace
import unittest
import numpy as np

from jugglebot.robustness.observation import BallObservation, BallEstimate
from .reception_control import BallisticPositionTracker, ReceptionReference
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


if __name__ == '__main__':
    unittest.main()

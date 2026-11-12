"""Acceptance, isolation and bounds of the optional release-timing trials."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from jugglebot.humanoid.cascade import ContactCascade, MAX_FINGER_RELEASE_LEAD_S
from jugglebot.robustness.controller import ObservedCascade


class TestThrowReleaseTiming(unittest.TestCase):
    def setUp(self):
        self.command = np.array([.15, -1.23, 4.5, -.04])
        self.path = SimpleNamespace(finger_release_lead_s=.02)
        self.error = np.array([.03, .035, .02, 0., 0., 0., 0., 0.])

    def controller(self, cls=ObservedCascade):
        controller = cls.__new__(cls)
        controller.log = Mock()
        return controller

    def call(self, controller, evaluate, lead=.02, error=None):
        return controller.try_throw_release_timing('left', 2, self.command, self.path,
                                                   self.error if error is None else error, lead, evaluate)

    def test_accepted_motion_plan_does_not_enter_timing_search(self):
        c = self.controller()
        c.throw_release_lead_candidates = Mock()
        evaluate = Mock()
        error = np.zeros(8)
        selected = self.call(c, evaluate, error=error)
        self.assertIs(selected[0], self.command)
        self.assertIs(selected[1], self.path)
        self.assertIs(selected[2], error)
        self.assertEqual(selected[3], .02)
        c.throw_release_lead_candidates.assert_not_called()
        evaluate.assert_not_called()
        c.log.assert_not_called()

    def test_v6_retains_failed_motion_and_indexed_timing(self):
        c = self.controller(ContactCascade)
        evaluate = Mock()
        selected = self.call(c, evaluate)
        self.assertIs(selected[0], self.command)
        self.assertIs(selected[1], self.path)
        self.assertIs(selected[2], self.error)
        self.assertEqual(selected[3], .02)
        evaluate.assert_not_called()

    def test_first_certified_timing_keeps_the_same_motion(self):
        for lead in (.02, -.02):
            with self.subTest(lead=lead):
                c = self.controller()
                path = SimpleNamespace(finger_release_lead_s=lead/2)
                error = np.array([.01, 0., 0., 0., 0., 0., 0., 0.])
                evaluate = Mock(return_value=(error, path))
                selected = self.call(c, evaluate, lead=lead)
                evaluate.assert_called_once()
                self.assertIs(evaluate.call_args.args[0], self.command)
                self.assertEqual(evaluate.call_args.kwargs, dict(release_lead_override=lead/2))
                self.assertIs(selected[0], self.command)
                self.assertIs(selected[1], path)
                self.assertIs(selected[2], error)
                self.assertEqual(selected[3], lead/2)
                self.assertTrue(c.log.call_args.kwargs['accepted'])

    def test_rejected_trials_return_original_objects_and_timing(self):
        c = self.controller()
        evaluate = Mock(side_effect=[ValueError('wrist limit'), (np.ones(8), object())])
        selected = self.call(c, evaluate)
        self.assertIs(selected[0], self.command)
        self.assertIs(selected[1], self.path)
        self.assertIs(selected[2], self.error)
        self.assertEqual(selected[3], .02)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual([call.kwargs['release_lead_override'] for call in evaluate.call_args_list], [.01, 0.])
        self.assertTrue(all(not call.kwargs['accepted'] for call in c.log.call_args_list))
        np.testing.assert_array_equal(self.command, [.15, -1.23, 4.5, -.04])

    def test_out_of_bounds_timing_never_reaches_physics(self):
        for candidate in (MAX_FINGER_RELEASE_LEAD_S+.001, -MAX_FINGER_RELEASE_LEAD_S-.001, float('nan'), float('inf')):
            with self.subTest(candidate=candidate):
                c = self.controller()
                c.throw_release_lead_candidates = Mock(return_value=(candidate,))
                evaluate = Mock()
                with self.assertRaisesRegex(ValueError, 'configured bound'):
                    self.call(c, evaluate)
                evaluate.assert_not_called()

    def test_zero_indexed_lead_does_not_add_redundant_trials(self):
        c = self.controller()
        evaluate = Mock()
        selected = self.call(c, evaluate, lead=0.)
        self.assertIs(selected[1], self.path)
        self.assertEqual(selected[3], 0.)
        evaluate.assert_not_called()


if __name__ == '__main__':
    unittest.main()

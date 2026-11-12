"""Recovery references preserve hard catch and safety requirements."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from jugglebot.humanoid.cascade import ContactCascade, MAX_RECEPTION_TILT_SCALE
from jugglebot.robustness.controller import ObservedCascade


class TestReceptionTilt(unittest.TestCase):
    def setUp(self):
        self.parameters = np.array([-.038, .036, .04, -.001, .044])
        self.ready = dict(centered=True, timely=True, captured=True, ready_for_next_throw=True,
                          collision_free=True, first_forbidden_contact=None)
        self.blocked = dict(self.ready, collision_free=False, first_forbidden_contact=dict(
            reason='POSE_LIMIT', part='left_wrist_pitch_joint', quantity='wrist_pitch_rad'))
        self.error = np.array([.004, .002, 0., .04, 0.])
        self.path = SimpleNamespace(catch_diagnostics=self.blocked)
        self.stop = object()
        self.axis = np.array([0., 0., 1.])

    def controller(self, cls=ObservedCascade):
        c = cls.__new__(cls)
        c.log = Mock()
        return c

    def call(self, c, evaluate, path=None):
        return c.try_reception_tilt('left', 0, self.parameters, self.error,
                                    self.path if path is None else path, self.stop, self.axis, evaluate)

    def assert_original(self, selected):
        for actual, expected in zip(selected, (self.error, self.path, self.stop, self.axis)):
            self.assertIs(actual, expected)

    def test_v6_keeps_existing_recovery_reference(self):
        c = self.controller(ContactCascade)
        evaluate = Mock()
        self.assert_original(self.call(c, evaluate))
        evaluate.assert_not_called()

    def test_certified_catch_does_not_enter_reference_search(self):
        c = self.controller()
        c.reception_tilt_scale_candidates = Mock()
        evaluate = Mock()
        path = SimpleNamespace(catch_diagnostics=self.ready)
        selected = self.call(c, evaluate, path=path)
        self.assertIs(selected[1], path)
        c.reception_tilt_scale_candidates.assert_not_called()
        evaluate.assert_not_called()

    def test_accepts_only_fully_certified_reference_with_the_same_target(self):
        c = self.controller()
        path = SimpleNamespace(catch_diagnostics=self.ready, reception_tilt_scale=1.5)
        stop, axis = object(), np.array([.01, 0., 1.])
        evaluate = Mock(side_effect=[(self.error, self.path, self.stop, self.axis),
                                     (np.zeros(5), path, stop, axis)])
        selected = self.call(c, evaluate)
        self.assertIs(selected[1], path)
        self.assertIs(selected[2], stop)
        self.assertIs(selected[3], axis)
        self.assertTrue(all(call.args[0] is self.parameters for call in evaluate.call_args_list))
        self.assertEqual([call.kwargs['tilt_scale_override'] for call in evaluate.call_args_list], [1.25, 1.5])
        self.assertEqual([call.kwargs['accepted'] for call in c.log.call_args_list], [False, True])
        np.testing.assert_array_equal(self.parameters, [-.038, .036, .04, -.001, .044])

    def test_zero_error_without_capture_and_nonzero_error_are_not_certificates(self):
        c = self.controller()
        invalid = SimpleNamespace(catch_diagnostics=dict(self.ready, captured=False))
        valid = SimpleNamespace(catch_diagnostics=self.ready)
        evaluate = Mock(side_effect=[(np.zeros(5), invalid, object(), self.axis),
                                     (np.full(5, .001), valid, object(), self.axis)])
        self.assert_original(self.call(c, evaluate))
        self.assertEqual(evaluate.call_count, 2)
        self.assertTrue(all(not call.kwargs['accepted'] for call in c.log.call_args_list))

    def test_physical_rejections_keep_the_original_reference(self):
        c = self.controller()
        evaluate = Mock(side_effect=ValueError('unreachable or unsafe'))
        self.assert_original(self.call(c, evaluate))
        self.assertEqual(evaluate.call_count, 2)

    def test_out_of_bounds_reference_never_reaches_physics(self):
        for scale in (.99, MAX_RECEPTION_TILT_SCALE+.01, float('nan'), float('inf')):
            with self.subTest(scale=scale):
                c = self.controller()
                c.reception_tilt_scale_candidates = Mock(return_value=(scale,))
                evaluate = Mock()
                with self.assertRaisesRegex(ValueError, 'configured bound'):
                    self.call(c, evaluate)
                evaluate.assert_not_called()

    def test_other_arm_limits_and_incomplete_catches_do_not_trigger_recovery(self):
        variants = [dict(self.blocked, first_forbidden_contact=dict(
            reason='POSE_LIMIT', part='right_wrist_pitch_joint', quantity='wrist_pitch_rad')),
                    dict(self.blocked, first_forbidden_contact=dict(
                        reason='POSE_LIMIT', part='world/lh_palm', quantity='self_penetration_m'))]
        variants += [dict(self.blocked, **{field: False}) for field in
                     ('centered', 'timely', 'captured', 'ready_for_next_throw')]
        c = self.controller()
        for diagnostics in variants:
            with self.subTest(diagnostics=diagnostics):
                self.assertEqual(c.reception_tilt_scale_candidates('left', diagnostics), ())

    def test_failed_retry_preserves_the_actual_catch_certificate(self):
        c = self.controller()
        c.silent = False
        c._use_execution_forecasts = False
        c.forecast_retry_count = 0
        c.hands = {'left': {'incoming': 0}}
        c.throw_count = 46
        c.execution_warm = {}
        c.events = []
        modes = []
        def fail(controller, side):
            modes.append(controller._use_execution_forecasts)
            controller.failure_event = dict(reason='NO_FEASIBLE_RECEPTION')
            controller.events.append(dict(kind='return_plan', side=side, ball=0,
                                          score=.04, parameters=self.parameters.tolist(), diagnostics=self.blocked))
            raise RuntimeError('no certified catch')
        with patch.object(ContactCascade, 'plan_reception', fail):
            with self.assertRaisesRegex(RuntimeError, 'no certified catch'):
                c.plan_reception('left')
        self.assertEqual(modes, [False, True, True])
        self.assertEqual(c.failure_event['reception_plan']['diagnostics'], self.blocked)
        self.assertEqual(c.failure_event['reason'], 'NO_FEASIBLE_RECEPTION')
        self.assertFalse(c._use_execution_forecasts)


if __name__ == '__main__':
    unittest.main()

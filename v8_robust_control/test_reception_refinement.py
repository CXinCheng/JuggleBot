"""A captured-ball refinement is bounded and leaves experiment inputs intact."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from jugglebot.humanoid.cascade import ContactCascade, RECEPTION_FINE_DIFFERENCE, RECEPTION_FINITE_DIFFERENCE
from jugglebot.robustness.controller import ObservedCascade


class TestReceptionRefinement(unittest.TestCase):
    def controller(self):
        c = ObservedCascade.__new__(ObservedCascade)
        c.silent = False
        c._use_execution_forecasts = False
        c._refining_reception = False
        c.forecast_retry_count = c.fine_reception_retry_count = 0
        c.throw_count = 46
        c.hands = {'left': {'incoming': 2}}
        c.events = []
        c.log = Mock()
        c.execution_warm = {'return:left:46': np.array([.02, .01, 0., 0., 0.])}
        c.data = SimpleNamespace(qpos=np.array([.1, .2, .8]), qvel=np.array([0., 0., -1.]))
        return c

    def event(self, captured=True):
        return dict(kind='return_plan', side='left', ball=2, score=.06,
                    parameters=[-.043, .03, .029, -.013, .08],
                    diagnostics=dict(captured=captured, timely=False, ready_for_next_throw=False,
                                     collision_free=False))

    def test_fine_steps_are_isolated_to_the_explicit_refinement(self):
        incomplete = dict(captured=True, timely=False, ready_for_next_throw=False)
        base = ContactCascade.__new__(ContactCascade)
        c = self.controller()
        np.testing.assert_array_equal(base.reception_difference_steps(incomplete), RECEPTION_FINITE_DIFFERENCE)
        np.testing.assert_array_equal(c.reception_difference_steps(incomplete), RECEPTION_FINITE_DIFFERENCE)
        c._refining_reception = True
        np.testing.assert_array_equal(c.reception_difference_steps(incomplete), RECEPTION_FINE_DIFFERENCE)

    def test_successful_refinement_restores_inputs_flags_and_physical_state(self):
        c = self.controller()
        inputs = c.execution_warm
        original = inputs['return:left:46'].copy()
        qpos, qvel = c.data.qpos.copy(), c.data.qvel.copy()
        modes = []
        def plan(controller, side):
            modes.append((controller._use_execution_forecasts, controller._refining_reception))
            if controller._refining_reception:
                np.testing.assert_array_equal(controller.execution_warm['return:left:46'], self.event()['parameters'])
                self.assertIsNone(controller.failure_event)
                return
            controller.events.append(self.event())
            controller.failure_event = dict(reason='NO_FEASIBLE_RECEPTION')
            raise RuntimeError('not ready')
        with patch.object(ContactCascade, 'plan_reception', plan):
            c.plan_reception('left')
        self.assertEqual(modes, [(False, False), (True, False), (True, True)])
        self.assertEqual(c.fine_reception_retry_count, 1)
        self.assertIs(c.execution_warm, inputs)
        np.testing.assert_array_equal(inputs['return:left:46'], original)
        np.testing.assert_array_equal(c.data.qpos, qpos)
        np.testing.assert_array_equal(c.data.qvel, qvel)
        self.assertFalse(c._use_execution_forecasts)
        self.assertFalse(c._refining_reception)

    def test_failed_refinement_is_not_repeated_and_keeps_failure_context(self):
        c = self.controller()
        inputs = c.execution_warm
        modes = []
        def plan(controller, side):
            modes.append((controller._use_execution_forecasts, controller._refining_reception))
            event = self.event()
            event['score'] = .07 if controller._refining_reception else .06
            controller.events.append(event)
            controller.failure_event = dict(reason='NO_FEASIBLE_RECEPTION')
            raise RuntimeError('still unsafe')
        with patch.object(ContactCascade, 'plan_reception', plan):
            with self.assertRaisesRegex(RuntimeError, 'still unsafe'):
                c.plan_reception('left')
        self.assertEqual(len(modes), 3)
        self.assertEqual(c.failure_event['reception_plan']['score'], .07)
        self.assertIs(c.execution_warm, inputs)
        self.assertFalse(c._use_execution_forecasts)
        self.assertFalse(c._refining_reception)

    def test_uncaptured_ball_does_not_enter_contact_refinement(self):
        c = self.controller()
        calls = []
        def plan(controller, side):
            calls.append(controller._refining_reception)
            controller.events.append(self.event(captured=False))
            controller.failure_event = dict(reason='NO_FEASIBLE_RECEPTION')
            raise RuntimeError('not captured')
        with patch.object(ContactCascade, 'plan_reception', plan):
            with self.assertRaisesRegex(RuntimeError, 'not captured'):
                c.plan_reception('left')
        self.assertEqual(calls, [False, False])
        self.assertEqual(c.fine_reception_retry_count, 0)

    def test_pose_guard_remains_a_failure_during_refinement(self):
        c = self.controller()
        inputs = c.execution_warm
        def plan(controller, side):
            if controller._refining_reception:
                controller.failure_event = dict(reason='POSE_LIMIT')
                raise RuntimeError('pose guard')
            controller.events.append(self.event())
            controller.failure_event = dict(reason='NO_FEASIBLE_RECEPTION')
            raise RuntimeError('not ready')
        with patch.object(ContactCascade, 'plan_reception', plan):
            with self.assertRaisesRegex(RuntimeError, 'pose guard'):
                c.plan_reception('left')
        self.assertEqual(c.failure_event['reason'], 'POSE_LIMIT')
        self.assertIs(c.execution_warm, inputs)
        self.assertFalse(c._use_execution_forecasts)
        self.assertFalse(c._refining_reception)


if __name__ == '__main__':
    unittest.main()

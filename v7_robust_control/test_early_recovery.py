"""Provisional empty-palm control remains separate from confirmed flight counting."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from jugglebot.humanoid.contact_control import DualArmDynamics
from jugglebot.humanoid.cascade import SUPPORTED_BALL_CENTER_OFFSET_M
from jugglebot.robustness.controller import ObservedCascade


class TestEarlyRecovery(unittest.TestCase):
    def controller(self):
        c = ObservedCascade.__new__(ObservedCascade)
        c.model = object()
        c.gravity_m_s2 = 9.81
        c.early_recovery_enabled = True
        c.early_recovery_active = False
        c.silent = False
        c.throw_count = 0
        c.targets = {1: 'right'}
        c.log = Mock()
        c.ball_contact = Mock(return_value=False)
        c.spatial = Mock(return_value=np.zeros(6))
        c.data = SimpleNamespace(time=2., qpos=np.zeros(7), qvel=np.zeros(7), site_xpos=np.array([[.33, -.14, .9]]))
        hand = dict(kin=SimpleNamespace(side='right', data=object(), qpos_adr=np.arange(7)),
                    seat=0, q=np.zeros(7), catch_p=np.array([.30, -.22, .86]),
                    held=0, mode='throw', capture_confirmed=True, release_pending=1.99,
                    contact_time=1.04, approach_joint=SimpleNamespace(t1=1.))
        c.hands = {'right': hand}
        duration = .15
        position = np.array([.3, -.2, .86+SUPPORTED_BALL_CENTER_OFFSET_M+duration+.5*9.81*duration**2])
        c.reception_ball_state = Mock(return_value=(position, np.array([0., .2, -1.])))
        return c, hand

    def call(self, c, hand, recorder=None):
        original_motion, target = object(), object()
        reference = SimpleNamespace(t0=c.data.time, t1=c.data.time+.15)
        calls = []
        def follow(controller, current_hand, data, motion, time, nominal):
            calls.append((current_hand['mode'], motion, time, nominal))
            if recorder is not None:
                recorder()
        with patch('jugglebot.robustness.controller.mujoco.mj_copyData'), \
                patch('jugglebot.robustness.controller.CartesianPalmMotion', return_value=reference) as factory, \
                patch.object(DualArmDynamics, 'follow_palm', follow):
            c.follow_palm(hand, c.data, original_motion, .17, target)
        return original_motion, reference, target, calls, factory

    def test_late_confirmed_recovery_enables_empty_motion_without_counting_a_release(self):
        c, hand = self.controller()
        before = c.data.qpos.copy(), c.data.qvel.copy(), dict(c.targets)
        original, reference, target, calls, factory = self.call(c, hand)
        self.assertEqual(calls, [('return', reference, 2., target)])
        self.assertTrue(c.early_recovery_active)
        self.assertEqual(hand['mode'], 'throw')
        self.assertEqual(hand['held'], 0)
        self.assertEqual(hand['release_pending'], 1.99)
        self.assertEqual(c.throw_count, 0)
        self.assertEqual(factory.call_count, 1)
        c.reception_ball_state.assert_called_once_with(1)
        np.testing.assert_array_equal(c.data.qpos, before[0])
        np.testing.assert_array_equal(c.data.qvel, before[1])
        self.assertEqual(c.targets, before[2])

    def test_timely_catch_and_unconfirmed_or_unfinished_recovery_do_not_activate(self):
        for fields in (dict(contact_time=1.02), dict(capture_confirmed=False), dict(mode='carry')):
            with self.subTest(fields=fields):
                c, hand = self.controller()
                hand.update(fields)
                original, _, target, calls, factory = self.call(c, hand)
                self.assertEqual(calls, [(hand['mode'], original, .17, target)])
                self.assertFalse(c.early_recovery_active)
                factory.assert_not_called()
                c.reception_ball_state.assert_not_called()

    def test_recontact_cancels_cached_empty_motion(self):
        c, hand = self.controller()
        c.early_recovery_active = True
        c.ball_contact.return_value = True
        hand['early_recovery'], hand['early_recovery_stamp'] = object(), 1.99
        original, _, target, calls, factory = self.call(c, hand)
        self.assertEqual(calls, [('throw', original, .17, target)])
        self.assertNotIn('early_recovery', hand)
        self.assertNotIn('early_recovery_stamp', hand)
        factory.assert_not_called()

    def test_private_trial_inherits_live_activation_without_using_hypothetical_lateness(self):
        c, hand = self.controller()
        c.silent = True
        original, _, target, calls, factory = self.call(c, hand)
        self.assertEqual(calls, [('throw', original, .17, target)])
        self.assertFalse(c.early_recovery_active)
        factory.assert_not_called()
        c.early_recovery_active = True
        _, reference, target, calls, factory = self.call(c, hand)
        self.assertEqual(calls, [('return', reference, 2., target)])
        self.assertTrue(c.early_recovery_active)
        self.assertEqual(factory.call_count, 1)

    def test_no_pending_release_or_disabled_feature_preserves_original_reference(self):
        for state in ('no_pending', 'released', 'disabled'):
            with self.subTest(state=state):
                c, hand = self.controller()
                if state == 'no_pending':
                    hand['release_pending'] = None
                elif state == 'released':
                    hand['held'] = None
                else:
                    c.early_recovery_enabled = False
                original, _, target, calls, factory = self.call(c, hand)
                self.assertEqual(calls, [('throw', original, .17, target)])
                factory.assert_not_called()
                c.reception_ball_state.assert_not_called()

    def test_missing_incoming_estimate_does_not_substitute_truth(self):
        c, hand = self.controller()
        c.reception_ball_state.return_value = None
        original, _, target, calls, factory = self.call(c, hand)
        self.assertEqual(calls, [('throw', original, .17, target)])
        self.assertNotIn('early_recovery', hand)
        factory.assert_not_called()

    def test_one_reference_per_separation_and_activation_remains_latched(self):
        c, hand = self.controller()
        first = SimpleNamespace(t1=2.15)
        second = SimpleNamespace(t1=2.16)
        calls = []
        def follow(controller, current_hand, data, motion, time, target):
            calls.append((motion,time,current_hand['mode']))
        with patch('jugglebot.robustness.controller.mujoco.mj_copyData'), \
                patch('jugglebot.robustness.controller.CartesianPalmMotion', side_effect=[first,second]) as factory, \
                patch.object(DualArmDynamics,'follow_palm',follow):
            c.follow_palm(hand,c.data,object(),.17,hand['q'])
            hand['contact_time'] = 1.01
            c.data.time = 2.001
            c.follow_palm(hand,c.data,object(),.171,hand['q'])
            hand['release_pending'] = 2.
            c.data.time = 2.002
            c.follow_palm(hand,c.data,object(),.172,hand['q'])
        self.assertEqual(calls,[(first,2.,'return'),(first,2.001,'return'),(second,2.002,'return')])
        self.assertEqual(factory.call_count,2)
        self.assertTrue(c.early_recovery_active)

    def test_control_exception_restores_bookkeeping_mode(self):
        c, hand = self.controller()
        def fail():
            raise RuntimeError('tracking failure')
        with self.assertRaisesRegex(RuntimeError,'tracking failure'):
            self.call(c,hand,fail)
        self.assertEqual(hand['mode'],'throw')
        self.assertEqual(hand['held'],0)
        self.assertEqual(c.throw_count,0)

    def test_other_hand_late_carry_waits_until_its_recovery_is_complete(self):
        c, hand = self.controller()
        hand['contact_time'] = 1.01
        other = dict(mode='carry',held=2,capture_confirmed=True,contact_time=1.05,approach_joint=SimpleNamespace(t1=1.))
        c.hands['left'] = other
        self.assertFalse(c._late_recovery_needed())
        other['mode'] = 'hold'
        self.assertTrue(c._late_recovery_needed())
        c.throw_count = 999
        _, reference, _, calls, _ = self.call(c,hand)
        self.assertIs(calls[0][1],reference)
        self.assertTrue(c.early_recovery_active)


if __name__ == '__main__':
    unittest.main()

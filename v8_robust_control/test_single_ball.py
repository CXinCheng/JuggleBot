"""Single-ball physical event acceptance and scheduling."""
from copy import deepcopy
import unittest
from unittest.mock import patch


class SingleBallTests(unittest.TestCase):
    def test_complete_single_ball_acceptance_requires_real_events_and_stable_hold(self):
        from .single_ball import validate_transfer
        value = dict(error=None, failure_event=None, ball_count=1, initial_hold_validated=True,
                     initial_hold_max_s=.05, final_hold_s=.2, throws=1, caught_release_ids=[1],
                     contact_only_checks=1000, airborne_physics_checks=500,
                     releases=[dict(count=1, ball=0, side='right', separation_time=.25)],
                     catches=[dict(release_id=1, ball=0, side='left', palm_touch_time=.75)],
                     final_evidence=dict(contact=True, relative_speed_m_s=.01, tangent_m=.01, normal_m=.022,
                                         ball_height_m=.85, forbidden_contact=False, pose_violation=False,
                                         external_ball_force=False))
        self.assertTrue(validate_transfer(value))
        for change in ('no_release', 'extra_release', 'wrong_hand', 'wrong_id', 'short_start', 'short_end', 'external_force'):
            altered = deepcopy(value)
            if change == 'no_release':
                altered['releases'] = []
            elif change == 'extra_release':
                altered['throws'] = 2
            elif change == 'wrong_hand':
                altered['catches'][0]['side'] = 'right'
            elif change == 'wrong_id':
                altered['caught_release_ids'] = [0]
            elif change == 'short_start':
                altered['initial_hold_max_s'] = .049
            elif change == 'short_end':
                altered['final_hold_s'] = .199
            else:
                altered['final_evidence']['external_ball_force'] = True
            self.assertFalse(validate_transfer(altered), change)

    def test_single_ball_reference_suppresses_only_live_extra_throws(self):
        from .single_ball import SingleBallTransfer
        from jugglebot.robustness.controller import ObservedCascade
        controller = object.__new__(SingleBallTransfer)
        controller.silent, controller.throw_count = False, 1
        with patch.object(ObservedCascade, 'ready_for_throw', return_value=True) as original:
            self.assertFalse(controller.ready_for_throw('left'))
            original.assert_not_called()
            controller.throw_count = 0
            self.assertTrue(controller.ready_for_throw('right'))
            controller.silent, controller.throw_count = True, 1
            self.assertTrue(controller.ready_for_throw('left'))
            self.assertEqual(original.call_count, 2)


if __name__ == '__main__':
    unittest.main()

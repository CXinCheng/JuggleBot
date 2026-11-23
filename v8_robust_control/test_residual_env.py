"""Physical reception and adversarial checks for residual-control experiments."""
from dataclasses import replace
import json
import unittest

import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

from .residual_env import (CatchAcceptance, CatchConfig, CatchEvidence,
                           ResidualCatchEnv, transition_reward)
from .train_residual import RandomizedReception


class RewardAcceptanceTests(unittest.TestCase):
    def evidence(self, **changes):
        return replace(CatchEvidence(True, .01, .01, .022, .86), **changes)

    def test_success_requires_continuous_support_and_is_not_repeatable(self):
        monitor = CatchAcceptance(.2)
        for _ in range(19):
            self.assertIsNone(monitor.update(self.evidence(), .01, .2)[1])
        monitor.update(self.evidence(contact=False), .01, .21)
        self.assertEqual(monitor.stable_s, 0.)
        for _ in range(20):
            success, reason = monitor.update(self.evidence(), .01, .4)
        self.assertTrue(success)
        self.assertEqual(reason, 'SUCCESS')
        self.assertFalse(monitor.update(self.evidence(), .01, .41)[0])

    def test_touch_bounce_and_side_contact_are_not_success(self):
        for changes in ({'relative_speed_m_s': .4}, {'tangent_m': .09},
                        {'normal_m': -.01}, {'contact': False}):
            monitor = CatchAcceptance()
            for _ in range(30):
                success, _ = monitor.update(self.evidence(**changes), .01, .2)
                self.assertFalse(success)

    def test_forbidden_events_take_precedence_over_stability(self):
        for changes, reason in (({'forbidden_contact': True}, 'FORBIDDEN_CONTACT'),
                                ({'pose_violation': True}, 'POSE_LIMIT'),
                                ({'external_ball_force': True}, 'EXTERNAL_BALL_FORCE'),
                                ({'ball_height_m': 1.31}, 'BALL_TOO_HIGH'),
                                ({'ball_height_m': .39}, 'BALL_DROPPED')):
            monitor = CatchAcceptance()
            monitor.stable_s = .199
            self.assertEqual(monitor.update(self.evidence(**changes), .001, .3), (False, reason))

    def test_airtime_and_survival_do_not_earn_reward(self):
        cfg = replace(CatchConfig(), shaping_weight=.25)
        zero = np.zeros(4)
        for n in (1, 20, 40):
            total = sum(transition_reward(cfg, .5, .5, .02, zero, zero)[0] for _ in range(n))
            self.assertLessEqual(total, 0.)
        for outcome in ('TASK_DEADLINE', 'BALL_TOO_HIGH', 'FORBIDDEN_CONTACT'):
            failure = transition_reward(cfg, .5, 1., .02, zero, zero, outcome)[0]
            success = transition_reward(cfg, .5, 0., .02, zero, zero, 'SUCCESS')[0]
            self.assertLess(failure, 0.)
            self.assertGreater(success, failure)

    def test_unsafe_contact_has_stronger_penalty_than_an_ordinary_miss(self):
        config = CatchConfig()
        action = np.zeros(4)
        missed = transition_reward(config, 0., 0., .02, action, action, 'BALL_DROPPED')[0]
        unsafe = transition_reward(config, 0., 0., .02, action, action, 'FORBIDDEN_CONTACT')[0]
        self.assertLess(unsafe, missed)
        self.assertLess(missed, 0.)
        with self.assertRaises(ValueError):
            replace(config, unsafe_failure_reward=-5.)

    def test_discounted_shaping_telescopes_for_all_terminal_outcomes(self):
        cfg = replace(CatchConfig(), shaping_weight=.25)
        for potentials in ((.2, 1., .1, .9), (.2, .1, .1, .1)):
            value = 0.
            for i in range(len(potentials)-1):
                _, terms = transition_reward(cfg, potentials[i], potentials[i+1], .02,
                    np.zeros(4), np.zeros(4), 'TASK_DEADLINE' if i == len(potentials)-2 else None)
                value += cfg.gamma**i * terms['shaping']
            self.assertAlmostEqual(value, -cfg.shaping_weight*potentials[0])


class PhysicalCatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ResidualCatchEnv()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_gym_contract(self):
        check_env(self.env, skip_render_check=True)

    def test_zero_residual_determinism_and_nominal_capture(self):
        records = []
        for _ in range(2):
            self.env.reset(seed=1)
            trajectory = []
            for _ in range(45):
                obs, reward, done, truncated, info = self.env.step(np.zeros(4))
                trajectory.append(np.r_[obs, reward])
                if done or truncated:
                    break
            self.assertTrue(info['is_success'], info)
            json.dumps(info, allow_nan=False)
            self.assertFalse(truncated)
            self.assertGreaterEqual(info['stable_contact_s'], .2-1e-9)
            records.append(np.asarray(trajectory))
        np.testing.assert_array_equal(*records)

    def test_ball_velocity_truth_cannot_change_policy_observation(self):
        self.env.reset(seed=9)
        for _ in range(3):
            self.env.step(np.zeros(4))
        before = self.env._observation()
        self.env.data.qvel[self.env.bv:self.env.bv+3] += [1., 2., 3.]
        mujoco.mj_forward(self.env.model, self.env.data)
        np.testing.assert_array_equal(before, self.env._observation())

    def test_action_bounds_rate_limits_and_invalid_requests(self):
        self.env.reset(seed=2)
        _, _, _, _, _ = self.env.step(np.full(4, 100.))
        limit = self.env.config.max_offset_speed_m_s/self.env.config.max_offset_m*self.env.config.control_period_s
        self.assertTrue(np.all(self.env.action[:3] <= limit+1e-9))
        self.assertLessEqual(self.env.action[3], .1+1e-9)
        self.assertLessEqual(self.env.max_offset, self.env.config.max_offset_m)
        for invalid in ([0., 0.], [np.nan, 0., 0., 0.]):
            with self.assertRaises(ValueError):
                self.env.step(invalid)

    def test_external_force_detected_without_reward(self):
        self.env.reset(seed=2)
        self.env.data.xfrc_applied[self.env.ball_body, 0] = 1.
        _, reward, done, _, info = self.env.step(np.zeros(4))
        self.assertTrue(done)
        self.assertEqual(info['reason'], 'EXTERNAL_BALL_FORCE')
        self.assertLess(reward, 0.)
        with self.assertRaises(RuntimeError):
            self.env.step(np.zeros(4))

    def test_nominal_feedback_moves_toward_measured_landing(self):
        self.env.reset(seed=1)
        initial = self.env.nominal_palm.copy()
        for _ in range(5):
            self.env.step(np.zeros(4))
        self.assertGreater(self.env.reference.updates, 0)
        self.assertGreater(np.linalg.norm(self.env.reference.last_intercept[:2]-initial[:2]), .001)

    def test_residual_authority_requires_hand_contact(self):
        obs, _ = self.env.reset(seed=1)
        np.testing.assert_array_equal(obs[-2:], np.zeros(2))
        for _ in range(30):
            permitted = self.env._residual_permitted()
            position_permitted = self.env._position_residual_permitted()
            obs, _, done, _, info = self.env.step(np.full(4, .1))
            if not permitted:
                np.testing.assert_array_equal(self.env.action, np.zeros(4))
            if not position_permitted:
                np.testing.assert_array_equal(self.env.action[:3], np.zeros(3))
            if permitted and np.any(self.env.action):
                self.assertGreater(info['residual_active_steps'], 0)
                return
            if done:
                break
        self.fail('Policy never received contact-phase action authority')

    def test_finger_contact_cannot_command_extra_closing_before_palm_support(self):
        self.env.reset(seed=1)
        self.env.last_evidence = CatchEvidence(True, .1, .03, .05, 1.)
        self.env.palm_contact = False
        self.env.step(np.full(4, .1))
        np.testing.assert_array_equal(self.env.action, np.zeros(4))
        self.env.reset(seed=1)
        self.env.last_evidence = CatchEvidence(True, .1, .03, .05, 1.)
        self.env.palm_contact = False
        self.env.step(np.array([0., 0., 0., -.05]))
        self.assertLess(self.env.action[3], 0.)

    def test_contact_phase_keeps_position_observation_available(self):
        self.env.reset(seed=1)
        available = False
        for _ in range(45):
            obs, _, done, _, info = self.env.step(np.zeros(4))
            if self.env.touch_time is not None and self.env.data.time-self.env.touch_time > .06:
                available |= bool(obs[6])
            if done:
                break
        self.assertTrue(available)
        self.assertGreater(info['contact_measurement_updates'], 2)

    def test_post_contact_velocity_truth_cannot_change_policy_observation(self):
        self.env.reset(seed=1)
        for _ in range(45):
            _, _, done, _, _ = self.env.step(np.zeros(4))
            if self.env.touch_time is not None and self.env.data.time-self.env.touch_time > .06:
                before = self.env._observation()
                self.assertTrue(before[6])
                self.env.data.qvel[self.env.bv:self.env.bv+3] += [1., 2., 3.]
                mujoco.mj_forward(self.env.model, self.env.data)
                np.testing.assert_array_equal(before, self.env._observation())
                return
            if done:
                break
        self.fail('No valid post-contact observation reached')

    def test_forward_cache_reuse_preserves_physical_trajectory(self):
        reference = ResidualCatchEnv(replace(CatchConfig(), reuse_forward_cache=False))
        try:
            reference.reset(seed=3)
            self.env.reset(seed=3)
            for _ in range(45):
                action = np.array([.02, -.03, .01, .02])
                a = reference.step(action)
                b = self.env.step(action)
                np.testing.assert_allclose(reference.data.qpos, self.env.data.qpos, atol=1e-12, rtol=0.)
                np.testing.assert_allclose(reference.data.qvel, self.env.data.qvel, atol=1e-10, rtol=0.)
                self.assertEqual(a[2:4], b[2:4])
                if a[2]:
                    self.assertEqual(a[4]['reason'], b[4]['reason'])
                    break
        finally:
            reference.close()


class PerceptionRandomizationTests(unittest.TestCase):
    def test_condition_rng_is_reproducible_and_does_not_change_initial_geometry(self):
        base = ResidualCatchEnv()
        try:
            base.reset(seed=73)
            qpos, qvel = base.data.qpos.copy(), base.data.qvel.copy()
            wrapped = RandomizedReception(base)
            first, info_a = wrapped.reset(seed=73)
            np.testing.assert_array_equal(qpos, base.data.qpos)
            np.testing.assert_array_equal(qvel, base.data.qvel)
            second, info_b = wrapped.reset(seed=73)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(info_a['perception_case'], info_b['perception_case'])
        finally:
            base.close()

    def test_randomization_covers_multiple_conditions(self):
        wrapped = RandomizedReception(ResidualCatchEnv())
        try:
            cases = {wrapped.reset(seed=seed)[1]['perception_case'] for seed in range(16)}
            self.assertGreater(len(cases), 1)
        finally:
            wrapped.close()


if __name__ == '__main__':
    unittest.main()

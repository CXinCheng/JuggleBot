"""Physical regressions for policy requests across contact transitions."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch

from .cascade_env import prepare_reference
from .feedback_env import FeedbackCascadeEnv, GuardedCascadeEnv
from .transfer_env import TransferConfig
from .train_cascade import make_model
from .audit_feedback import release_snapshot, summarize, native_trial


class ReleaseEvidenceTests(unittest.TestCase):
    def test_native_reset_uses_paired_displacements_and_keeps_timeout_as_failure(self):
        controller = SimpleNamespace(data=SimpleNamespace(time=0., qpos=np.zeros(21), qvel=np.zeros(18)),
            model=None, balls=[(0, 0), (7, 6), (14, 12)], ball_observer=Mock(), events=[],
            run_complete=Mock(return_value=dict(requested_throws=30, throws=2, catches=1,
                success=False, error='TIMEOUT', failure_event=None)))
        with patch('v8_robust_control.complete_task.CompleteCascade', return_value=controller), patch('mujoco.mj_forward'):
            row = native_trial((30, 2.2, 'seat_2mm', 410200))
        shifts = np.random.default_rng(410200).uniform(-.002, .002, (2, 2))
        np.testing.assert_array_equal(row['initial_displacement_m'], shifts[[0, 1, 0]])
        for ball, (q, _) in enumerate(controller.balls):
            np.testing.assert_array_equal(controller.data.qpos[q:q+2], shifts[0 if ball in (0, 2) else 1])
        controller.ball_observer.reset.assert_called_once_with()
        controller.run_complete.assert_called_once_with(timeout=2.2)
        record = dict(manifest=dict(controller='native', cases={'seat_2mm': {'seat_jitter_m': .002}},
            seeds=[410200], requested_throws=30), records=[row])
        summary = summarize(record)[0]
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['catches'], 1)
        row['result']['success'] = True
        with self.assertRaises(ValueError):
            summarize(record)

    def fixture(self, valid=True):
        position = np.array([1., 2., 3.])
        velocity = np.array([.1, .2, .3])
        estimate = SimpleNamespace(valid=valid, position_m=position+[.001, 0., 0.],
                                   velocity_m_s=velocity+[.01, 0., 0.])
        return SimpleNamespace(controller=SimpleNamespace(balls=[(0, 0)]),
            model=SimpleNamespace(nq=3), index=1,
            data=SimpleNamespace(time=.12, qpos=position, qvel=velocity),
            expected_states=np.array([[.99, 1.98, 3.03, .09, .22, .33]]),
            release_reference=[dict(separation_time=.1)],
            trackers=[SimpleNamespace(estimate=lambda now: estimate)],
            joint_offset=np.zeros((2, 7)), wrapping_offset=np.zeros((2, 16)))

    def test_release_evidence_separates_sample_time_and_estimation_error(self):
        env = self.fixture()
        before = np.r_[env.data.qpos, env.data.qvel].copy()
        evidence = release_snapshot(env, dict(id=1, ball=0, side=0, time=.105))
        self.assertEqual(evidence['diagnostic_time_s'], .12)
        self.assertEqual(evidence['separation_time_s'], .105)
        self.assertAlmostEqual(evidence['release_timing_error_s'], .005)
        np.testing.assert_allclose(evidence['velocity_error_m_s'], [.01, -.02, -.03])
        np.testing.assert_allclose(evidence['estimation_position_error_m'], [.001, 0., 0.], atol=1e-15)
        np.testing.assert_allclose(evidence['estimation_velocity_error_m_s'], [.01, 0., 0.], atol=1e-15)
        np.testing.assert_array_equal(np.r_[env.data.qpos, env.data.qvel], before)

    def test_invalid_estimate_is_not_reported_as_zero_estimation_error(self):
        evidence = release_snapshot(self.fixture(False), dict(id=1, ball=0, side=0, time=.105))
        self.assertFalse(evidence['estimate_valid'])
        self.assertIsNone(evidence['estimation_position_error_m'])
        self.assertIsNone(evidence['estimation_velocity_error_m_s'])

    def test_selected_motor_audit_scenes_and_request_semantics_are_validated(self):
        manifest = dict(controllers=['motor_residual'], cases={'clean': {}}, requests={'zero': [0.]*6},
                        seeds=[1], base_config={}, requested_throws=30)
        result = dict(environment_version=8, config={}, is_success=False, reason='NO_RELEASE',
                      requested_throws=30, catches=1, max_wrapping_offset_rad=0.)
        row = dict(controller='motor_residual', request='zero', case='clean', seed=1, result=result)
        record = dict(manifest=manifest, records=[row])
        self.assertEqual(summarize(record)[0]['catches'], 1)
        manifest['requests'] = {'disabled': [-1.]*6}
        with self.assertRaises(ValueError):
            summarize(record)


class LatchedFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.reference = Path(cls.temporary.name)/'two'
        prepare_reference(cls.reference, 'two')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_disable_request_survives_noisy_contact_transition_and_preserves_physics(self):
        env = FeedbackCascadeEnv(self.reference, TransferConfig(noise_std_m=.003))
        try:
            env.reset(seed=410000)
            saw_permission_transition = False
            previous = env.permissions()
            while True:
                _, _, done, _, info = env.step(-np.ones(6))
                current = env.permissions()
                saw_permission_transition |= bool(np.any(current != previous))
                previous = current
                np.testing.assert_array_equal(env.feedback_gains, np.zeros((2, 2)))
                np.testing.assert_array_equal(env.wrapping_gain, np.zeros(2))
                np.testing.assert_array_equal(np.r_[env.data.qpos, env.data.qvel],
                                              env.expected_states[env.index-1])
                if done:
                    break
            self.assertTrue(saw_permission_transition)
            self.assertTrue(info['is_success'])
            self.assertEqual((info['releases'], info['catches']), (2, 2))
            self.assertGreaterEqual(info['final_hold_s'], .2-1e-10)
        finally:
            env.close()

    def test_invalid_action_does_not_mutate_gain_or_physics(self):
        env = FeedbackCascadeEnv(self.reference)
        try:
            env.reset(seed=410001)
            state = np.r_[env.data.qpos, env.data.qvel].copy()
            for action in ([0.]*5, [float('nan')]*6):
                with self.assertRaises(ValueError):
                    env.step(action)
                np.testing.assert_array_equal(env.gain_request, np.zeros(6))
                np.testing.assert_array_equal(np.r_[env.data.qpos, env.data.qvel], state)
        finally:
            env.close()

    def test_policy_retains_disable_request_when_no_contact_permission_is_active(self):
        env = FeedbackCascadeEnv(self.reference)
        try:
            model, _ = make_model(env, 0)
            with torch.no_grad():
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.fill_(-3.)
            observation = np.zeros(191, dtype=np.float32)
            action, _ = model.predict(observation, deterministic=True)
            self.assertTrue(np.all(action < -.99))
        finally:
            env.close()


    def test_guarded_zero_request_is_neutral_with_delayed_noisy_measurements(self):
        env = GuardedCascadeEnv(self.reference, TransferConfig(noise_std_m=.003, delay_s=.02, dropout=.1))
        try:
            env.reset(seed=410002)
            while True:
                _, _, done, _, info = env.step(np.zeros(6))
                np.testing.assert_array_equal(env.feedback_gains, np.zeros((2, 2)))
                np.testing.assert_array_equal(env.wrapping_gain, np.zeros(2))
                np.testing.assert_array_equal(np.r_[env.data.qpos, env.data.qvel], env.expected_states[env.index-1])
                if done:
                    break
            self.assertTrue(info['is_success'])
        finally:
            env.close()


class EndingEvidenceTests(unittest.TestCase):
    def fixture(self):
        env = object.__new__(GuardedCascadeEnv)
        env.task, env.next_release, env.target_throws, env.count = 'complete', 30, 30, 3
        env.controller = SimpleNamespace(end=None)
        env.flight_ids = [28, 29, 30]
        env.data = SimpleNamespace(time=1.)
        env.estimate_valid = np.ones((1, 3), dtype=bool)
        env.estimate_reference = np.zeros((1, 3, 6))
        env._reference_index = lambda: 0
        env.ending_evidence_count, env.ending_evidence_stamp = 0, None
        estimate = SimpleNamespace(valid=True, sample_time_s=.9, position_m=(.1, 0., 0.))
        tracker = SimpleNamespace(estimate=lambda now: estimate, uncertainty=lambda now: (.003, .1))
        env.trackers = [tracker]*3
        return env, estimate

    def test_repeated_query_is_not_a_new_sensor_frame(self):
        env, estimate = self.fixture()
        with patch.object(FeedbackCascadeEnv, '_ending_control') as update:
            for _ in range(50):
                env._ending_control()
            self.assertEqual(env.ending_evidence_count, 1)
            update.assert_not_called()
            for stamp in (.92, .94):
                estimate.sample_time_s = stamp
                env._ending_control()
            update.assert_called_once()

    def test_invalid_measurement_breaks_the_consecutive_evidence_chain(self):
        env, estimate = self.fixture()
        with patch.object(FeedbackCascadeEnv, '_ending_control') as update:
            env._ending_control()
            estimate.sample_time_s = .92
            env._ending_control()
            estimate.valid = False
            env._ending_control()
            self.assertEqual(env.ending_evidence_count, 0)
            estimate.valid, estimate.sample_time_s = True, .99
            env._ending_control()
            self.assertEqual(env.ending_evidence_count, 1)
            update.assert_not_called()

    def test_motor_innovation_requires_distinct_consistent_position_frames(self):
        env, estimate = self.fixture()
        env.innovation_evidence = {}
        self.assertFalse(env._credible_innovation(2))
        for _ in range(20):
            self.assertFalse(env._credible_innovation(2))
        estimate.sample_time_s = .92
        self.assertFalse(env._credible_innovation(2))
        estimate.sample_time_s = .94
        self.assertTrue(env._credible_innovation(2))
        estimate.valid = False
        self.assertFalse(env._credible_innovation(2))
        estimate.valid, estimate.sample_time_s = True, .99
        self.assertFalse(env._credible_innovation(2))

    def test_direction_change_and_uncertain_measurement_reset_motor_evidence(self):
        env, estimate = self.fixture()
        env.innovation_evidence = {}
        for stamp in (.90, .92):
            estimate.sample_time_s = stamp
            self.assertFalse(env._credible_innovation(2))
        estimate.sample_time_s, estimate.position_m = .94, (-.1, 0., 0.)
        self.assertFalse(env._credible_innovation(2))
        estimate.sample_time_s, estimate.position_m = .96, (-.001, 0., 0.)
        self.assertFalse(env._credible_innovation(2))
        for stamp in (.98, 1.00):
            estimate.sample_time_s, estimate.position_m = stamp, (-.1, 0., 0.)
            self.assertFalse(env._credible_innovation(2))
        env.data.time = 1.03
        estimate.sample_time_s = 1.02
        self.assertTrue(env._credible_innovation(2))

if __name__ == '__main__':
    unittest.main()

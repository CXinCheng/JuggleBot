"""Physical multi-ball tasks and causal, bounded residual-control interfaces."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import shutil
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from gymnasium.utils.env_checker import check_env

from .cascade_env import CascadeEnv, prepare_reference
from .compare_residual import digest


class CascadeEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.references = {}
        for task in ('two', 'complete'):
            folder = Path(cls.temporary.name)/task
            prepare_reference(folder, task, 2 if task == 'two' else 4)
            cls.references[task] = folder

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_zero_residual_completes_exchange_and_three_ball_start_finish(self):
        for task, throws, hold in (('two', 2, .2), ('complete', 4, .8)):
            with self.subTest(task=task):
                env = CascadeEnv(self.references[task])
                try:
                    obs, _ = env.reset(seed=0)
                    self.assertEqual(obs.shape, (191,))
                    while True:
                        obs, _, done, truncated, info = env.step(np.zeros(6))
                        if done or truncated:
                            break
                    self.assertFalse(truncated)
                    self.assertTrue(info['is_success'], info)
                    self.assertEqual(info['reference_state_max_error'], 0.)
                    self.assertEqual(info['releases'], throws)
                    self.assertEqual(info['catches'], throws)
                    self.assertGreaterEqual(info['initial_hold_s'], .05-1e-10)
                    self.assertGreaterEqual(info['final_hold_s'], hold-1e-10)
                    self.assertAlmostEqual(info['episode_return'], 11.)
                    self.assertAlmostEqual(info['reward_sums']['progress'], 1.)
                    self.assertEqual(sorted(event['id'] for event in info['events'] if event['kind'] == 'catch'),
                                     list(range(1, throws+1)))
                    for ball, side in enumerate(env.final_sides):
                        self.assertTrue(env._supported(ball, side, task == 'complete'))
                    self.assertFalse(np.any(env.data.eq_active))
                    self.assertFalse(np.any(env.data.xfrc_applied))
                    self.assertEqual(info['adaptive_ending'], task == 'complete')
                    with self.assertRaises(RuntimeError):
                        env.step(np.zeros(6))
                finally:
                    env.close()

    def test_seeded_reset_preserves_pair_geometry_and_restores_dynamics(self):
        env = CascadeEnv(self.references['complete'])
        env.config = replace(env.config, seat_jitter_m=.002, noise_std_m=.003, delay_s=.02, dropout=.1)
        try:
            records = []
            for _ in range(2):
                obs, _ = env.reset(seed=71)
                np.testing.assert_array_equal(env.initial_displacement[0], env.initial_displacement[2])
                start = env.data.qpos.copy()
                for _ in range(8):
                    obs, reward, done, _, _ = env.step(np.full(6, .05))
                    if done:
                        break
                records.append((start, obs.copy(), env.data.qpos.copy(), reward))
            for first, second in zip(*records):
                np.testing.assert_array_equal(first, second)
        finally:
            env.close()

    def test_missing_target_estimate_does_not_borrow_another_balls_permission(self):
        env = CascadeEnv(self.references['two'])
        try:
            env.reset(seed=0)
            for _ in range(4):
                env.step(np.zeros(6))
            # Remove the initial-payload protection to isolate estimate routing.
            env.flight_ids = [1, 2]
            env.flight_sources = [1, 0]
            np.testing.assert_array_equal(env.permissions()[:4], True)
            env.trackers[1].max_age_s = 0.
            self.assertTrue(env.permissions()[0])
            self.assertFalse(env.permissions()[2])
        finally:
            env.close()

    def test_ball_truth_and_reference_arrays_do_not_enter_actor_observation(self):
        env = CascadeEnv(self.references['complete'])
        try:
            env.reset(seed=0)
            for _ in range(4):
                env.step(np.zeros(6))
            before = env._observation()
            for q, v in env.controller.balls:
                env.data.qpos[q:q+3] += [.01, .02, .03]
                env.data.qvel[v:v+3] += [1., 2., 3.]
            env.expected_states[:] = 1000.
            mujoco.mj_kinematics(env.model, env.data)
            mujoco.mj_comPos(env.model, env.data)
            mujoco.mj_comVel(env.model, env.data)
            np.testing.assert_array_equal(before, env._observation())
        finally:
            env.close()

    def test_cascade_gym_contract_and_motor_bounds(self):
        env = CascadeEnv(self.references['two'])
        try:
            check_env(env, skip_render_check=True)
            env.config = replace(env.config, noise_std_m=.001)
            env.reset(seed=0)
            previous = env.joint_offset.copy()
            for _ in range(15):
                _, _, done, _, _ = env.step(np.full(6, 10.))
                self.assertLessEqual(np.max(np.abs(env.joint_offset)), env.config.max_joint_offset_rad+1e-12)
                self.assertLessEqual(np.max(np.abs(env.joint_offset-previous)),
                                     env.config.max_joint_rate_rad_s*env.config.control_period_s+1e-12)
                self.assertLessEqual(np.max(np.abs(env.grip_offset)), env.config.max_grip_offset_rad+1e-12)
                self.assertLessEqual(np.max(np.abs(env.wrapping_offset)), env.MAX_WRAPPING_OFFSET_RAD+1e-12)
                previous = env.joint_offset.copy()
                if done:
                    break
        finally:
            env.close()

    def test_zero_error_initial_payload_is_invariant_to_requested_gains(self):
        env = CascadeEnv(self.references['two'])
        try:
            env.reset(seed=0)
            for _ in range(8):
                env.step(np.ones(6))
                np.testing.assert_array_equal(env.joint_offset, 0.)
                np.testing.assert_array_equal(env.wrapping_offset, 0.)
                np.testing.assert_array_equal(env.data.qpos, env.expected_states[env.index-1, :env.model.nq])
        finally:
            env.close()

    def test_wrapping_is_rate_limited_and_requires_real_palm_contact(self):
        env = CascadeEnv(self.references['two'])
        env.config = replace(env.config, noise_std_m=.001)
        try:
            env.reset(seed=0)
            touched = False
            for _ in range(100):
                _, _, done, _, _ = env.step([-1., -1., -1., -1., 0., 0.])
                if env._wrapping_ball(1) is not None:
                    touched = True
                    break
                if done:
                    break
            self.assertTrue(touched)
            before = env.wrapping_offset.copy()
            _, _, _, _, info = env.step([-1., -1., -1., -1., 0., .25])
            self.assertGreater(info['max_wrapping_offset_rad'], 0.)
            self.assertLessEqual(np.max(np.abs(env.wrapping_offset-before)),
                                 env.MAX_WRAPPING_RATE_RAD_S*env.config.control_period_s+1e-12)
            self.assertLessEqual(info['max_wrapping_offset_rad'], env.MAX_WRAPPING_OFFSET_RAD)
            np.testing.assert_array_equal(env.grip_offset, 0.)
            env.ball_loads[:] = 0.
            env.ball_contacts[:] = False
            self.assertIsNone(env._wrapping_ball(1))
            self.assertFalse(env.permissions()[5])
        finally:
            env.close()

    def test_incomplete_reference_is_not_a_training_environment(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'reference'
            shutil.copytree(self.references['two'], path)
            manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
            manifest['phase'] = 'preparing'
            (path/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                CascadeEnv(path)

    def test_reference_checksum_cannot_be_changed_silently(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'reference'
            shutil.copytree(self.references['two'], path)
            manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
            manifest['reference_sha256'] = 'different'
            (path/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                CascadeEnv(path)

    def test_encoder_initialization_resets_policy_heads_before_new_task(self):
        from .train_cascade import make_model
        torch.set_num_threads(1)
        env = CascadeEnv(self.references['two'])
        try:
            model, _ = make_model(env, 0)
            model.learn(256)
            with torch.no_grad():
                model.policy.action_net.bias.fill_(.5)
            with TemporaryDirectory() as folder:
                path = Path(folder)/'policy.zip'
                model.save(path)
                study = dict(manifest=dict(ball_count=2, source_sha256={}),
                             models=[dict(seed=0, checkpoint=path.name, checkpoint_sha256=digest(path), selected_step=256)])
                next_env = CascadeEnv(self.references['complete'])
                try:
                    initialized, provenance = make_model(next_env, 0, study, Path(folder))
                    observation, _ = next_env.reset(seed=0)
                    np.testing.assert_array_equal(initialized.predict(observation, deterministic=True)[0], 0.)
                    self.assertEqual(initialized.num_timesteps, 0)
                    self.assertEqual(provenance['checkpoint_sha256'], digest(path))
                finally:
                    next_env.close()
        finally:
            env.close()

    def test_longer_three_ball_task_retains_actor_and_resets_value_head(self):
        from .train_cascade import make_model
        torch.set_num_threads(1)
        env = CascadeEnv(self.references['complete'])
        try:
            model, _ = make_model(env, 0)
            # Synthetic checkpoint weights isolate transfer from task training.
            model.num_timesteps = 256
            with torch.no_grad():
                model.policy.action_net.bias.fill_(.2)
                model.policy.log_std.fill_(-5.25)
            with TemporaryDirectory() as folder:
                path = Path(folder)/'policy.zip'
                model.save(path)
                study = dict(manifest=dict(ball_count=3, task='complete', requested_throws=4, source_sha256={},
                                          observation_layout=CascadeEnv.OBSERVATION_LAYOUT, action_layout=CascadeEnv.ACTION_LAYOUT),
                             models=[dict(seed=0, checkpoint=path.name, checkpoint_sha256=digest(path), selected_step=256)])
                env.target_throws = 8
                initialized, provenance = make_model(env, 0, study, Path(folder))
                obs, _ = env.reset(seed=0)
                np.testing.assert_array_equal(initialized.predict(obs, deterministic=True)[0],
                                              model.predict(obs, deterministic=True)[0])
                self.assertTrue(provenance['actor_retained'])
                self.assertEqual(torch.count_nonzero(initialized.policy.value_net.weight).item(), 0)
                torch.testing.assert_close(initialized.policy.log_std, model.policy.log_std)
        finally:
            env.close()

    def test_random_gains_preserve_the_complete_nominal_trajectory(self):
        env = CascadeEnv(self.references['complete'])
        rng = np.random.default_rng(183)
        try:
            env.reset(seed=0)
            while True:
                _, _, done, _, info = env.step(rng.uniform(-1., 1., 6))
                np.testing.assert_array_equal(np.r_[env.data.qpos, env.data.qvel], env.expected_states[env.index-1])
                if done:
                    break
            self.assertTrue(info['is_success'])
            self.assertEqual(info['max_joint_offset_rad'], 0.)
            self.assertEqual(info['max_wrapping_offset_rad'], 0.)
        finally:
            env.close()

    def test_gain_actions_change_real_motors_when_measurements_deviate(self):
        env = CascadeEnv(self.references['two'])
        env.config = replace(env.config, noise_std_m=.001)
        try:
            positions, offsets = [], []
            for gain in (-1., 1.):
                env.reset(seed=91)
                for _ in range(8):
                    env.step([gain]*4+[0., 0.])
                positions.append(env.data.qpos.copy())
                offsets.append(float(np.max(np.abs(env.joint_offset))))
            self.assertEqual(offsets[0], 0.)
            self.assertGreater(offsets[1], 1e-6)
            self.assertGreater(np.max(np.abs(positions[0]-positions[1])), 1e-6)
        finally:
            env.close()

    def test_ending_trigger_uses_estimates_and_tactile_support_not_ball_truth(self):
        env = CascadeEnv(self.references['complete'])
        try:
            env.reset(seed=0)
            env.next_release = env.target_throws
            env.flight_ids = [1, 2, env.target_throws]
            env.flight_sources = [1, 0, 1]
            env.touch_times = [0., None, None]
            env.data.time = .2
            hand = env.hands[0]
            normal = env.data.site_xmat[hand['seat']].reshape(3, 3)[:, 2]
            estimate = SimpleNamespace(valid=True, position_m=env.data.site_xpos[hand['seat']]+.022*normal,
                                       velocity_m_s=env.controller.spatial(hand)[3:])
            env.ball_loads[0, 0, 0] = 1.
            with (patch.object(env.trackers[0], 'estimate', return_value=estimate),
                  patch.object(env.controller, 'supporting_digits', return_value=['rf', 'th']),
                  patch.object(env.controller, '_maybe_start_end') as start):
                for speed in (-1000., 1000.):
                    for _, v in env.controller.balls:
                        env.data.qvel[v:v+3] = speed
                    env._ending_control()
                    self.assertEqual(hand['held'], 0)
                self.assertEqual(start.call_count, 2)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()

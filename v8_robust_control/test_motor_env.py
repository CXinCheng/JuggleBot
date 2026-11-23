from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import mujoco
import torch

from .cascade_env import prepare_reference
from .motor_env import MotorResidualCascadeEnv, MotorCurriculum, training_curriculum
from .transfer_env import TransferConfig
from .train_cascade import make_model
from .compare_residual import digest


class PolicyFixture(gym.Env):
    def __init__(self, motor):
        self.CONTROLLER_NAME = 'motor_residual' if motor else 'reference'
        self.OBSERVATION_LAYOUT = MotorResidualCascadeEnv.OBSERVATION_LAYOUT if motor else 'legacy'
        self.ACTION_LAYOUT = MotorResidualCascadeEnv.ACTION_LAYOUT if motor else 'legacy'
        self.count = 2
        self.action_space = gym.spaces.Box(-1., 1., (6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (191,), dtype=np.float32)


class MotorPolicyTests(unittest.TestCase):
    def test_fresh_motor_exploration_accounts_for_all_gsde_latent_units(self):
        from .train_cascade import MOTOR_INITIAL_ACTION_STD
        model, _ = make_model(PolicyFixture(True), 0)
        observation = torch.randn((32, 191))*100.
        observation[:, 189:191] = 1.
        with torch.no_grad():
            distribution = model.policy.get_distribution(observation)
            deviations = distribution.distribution.scale
        self.assertLessEqual(float(deviations.max()), np.sqrt(MOTOR_INITIAL_ACTION_STD**2+1e-6)+1e-6)
        self.assertGreater(float(deviations.max()), 0.)

    def test_inactive_motor_requests_have_zero_actor_gradient_and_survive_reload(self):
        model, _ = make_model(PolicyFixture(True), 0)
        observation = torch.zeros((2, 191))
        with torch.no_grad():
            model.policy.action_net.bias.fill_(1.)
        actions, values, probabilities = model.policy(observation)
        torch.testing.assert_close(actions, torch.zeros_like(actions))
        torch.testing.assert_close(probabilities, torch.zeros_like(probabilities))
        _, probabilities, entropy = model.policy.evaluate_actions(observation, actions)
        loss = -probabilities.sum()
        if entropy is not None:
            loss -= entropy.sum()
        model.policy.zero_grad()
        loss.backward()
        self.assertTrue(torch.all(model.policy.action_net.bias.grad == 0.))
        observation[:, 189:191] = 1.
        with torch.no_grad():
            active, _, _ = model.policy(observation, deterministic=True)
        self.assertTrue(torch.all(active != 0.))
        with TemporaryDirectory() as directory:
            from stable_baselines3 import PPO
            path = Path(directory)/'masked.zip'
            model.save(path)
            loaded = PPO.load(path, device='cpu')
            np.testing.assert_array_equal(loaded.predict(np.zeros(191, dtype=np.float32), deterministic=True)[0], np.zeros(6))
        model.policy.zero_grad()
        _, values, _ = model.policy(torch.zeros((2, 191)))
        values.sum().backward()
        self.assertIsNotNone(model.policy.value_net.bias.grad)
        self.assertTrue(torch.any(model.policy.value_net.bias.grad != 0.))

    def test_warm_start_resets_changed_robot_features_and_motor_outputs(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            previous, _ = make_model(PolicyFixture(False), 0)
            previous.num_timesteps = 256
            with torch.no_grad():
                for network in (previous.policy.mlp_extractor.policy_net, previous.policy.mlp_extractor.value_net):
                    next(layer for layer in network if isinstance(layer, torch.nn.Linear)).weight.fill_(1.)
                previous.policy.action_net.weight.fill_(1.)
                previous.policy.action_net.bias.fill_(1.)
            path = folder/'previous.zip'
            previous.save(path)
            warm = dict(manifest=dict(observation_layout='legacy', action_layout='legacy', ball_count=2,
                source_sha256={}, controller='reference'), models=[dict(seed=0, checkpoint=path.name,
                checkpoint_sha256=digest(path), selected_step=256)])
            model, provenance = make_model(PolicyFixture(True), 0, warm, folder)
            for network in (model.policy.mlp_extractor.policy_net, model.policy.mlp_extractor.value_net):
                weights = next(layer for layer in network if isinstance(layer, torch.nn.Linear)).weight.detach().numpy()
                np.testing.assert_array_equal(weights[:, 42:146], 0.)
                np.testing.assert_array_equal(weights[:, 185:191], 0.)
            self.assertFalse(provenance['actor_retained'])
            observation = np.ones(191, dtype=np.float32)
            np.testing.assert_array_equal(model.predict(observation, deterministic=True)[0], np.zeros(6))


class MotorPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.reference = Path(cls.temporary.name)/'two'
        prepare_reference(cls.reference, 'two')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_noisy_ball_observations_cannot_open_motor_authority(self):
        env = MotorResidualCascadeEnv(self.reference, TransferConfig(noise_std_m=.01, delay_s=.04, dropout=.3))
        try:
            env.reset(seed=41)
            while True:
                observation, _, done, _, info = env.step(np.sin(env.data.time*7.+np.arange(6)))
                self.assertEqual(env.motor_authority, 0.)
                np.testing.assert_array_equal(np.r_[env.data.qpos, env.data.qvel], env.expected_states[env.index-1])
                np.testing.assert_array_equal(observation[42:146], np.zeros(104))
                if done:
                    break
            self.assertTrue(info['is_success'])
            self.assertEqual(info['max_joint_offset_rad'], 0.)
        finally:
            env.close()

    def test_motor_targets_preserve_magnitude_rate_and_cartesian_bounds(self):
        env = MotorResidualCascadeEnv(self.reference, TransferConfig(seat_jitter_m=.002))
        try:
            env.reset(seed=41)
            for _ in range(30):
                previous = env.joint_offset.copy()
                _, _, done, _, info = env.step(np.array([1., -1., 1., -1., 1., -1.]))
                self.assertLessEqual(np.max(np.abs(env.joint_offset)), env.config.max_joint_offset_rad+1e-12)
                self.assertLessEqual(np.max(np.abs(env.joint_offset-previous)),
                                     env.config.max_joint_rate_rad_s*env.config.control_period_s+1e-12)
                np.testing.assert_array_equal(env.joint_offset[:, [0, 2, 4, 6]], np.zeros((2, 4)))
                self.assertLessEqual(env.requested_translation_peak_m, env.config.max_cartesian_offset_m+1e-12)
                self.assertFalse(np.any(env.data.eq_active))
                self.assertFalse(np.any(env.data.xfrc_applied[env.controller.ball_body_ids]))
                if done:
                    break
            self.assertGreater(env.motor_authority, 0.)
            env.config = replace(env.config, seat_jitter_m=0.)
            env.reset(seed=41)
            self.assertEqual(env.motor_authority, 0.)
        finally:
            env.close()

    def test_actual_servo_commands_obey_rate_limit_at_one_khz(self):
        env = MotorResidualCascadeEnv(self.reference, TransferConfig(seat_jitter_m=.002))
        original_step = mujoco.mj_step2
        previous = np.zeros((2, 7))
        maximum = 0.

        def checked_step(model, data):
            nonlocal previous, maximum
            current = np.array([data.ctrl[hand['acts']]-env.controls[min(env.index, len(env.controls)-1), hand['acts']]
                                for hand in env.hands])
            delta = float(np.max(np.abs(current-previous)))
            maximum = max(maximum, delta)
            self.assertLessEqual(delta, env.config.max_joint_rate_rad_s*env.dt+1e-12)
            previous = current
            return original_step(model, data)

        try:
            env.reset(seed=41)
            with patch.object(mujoco, 'mj_step2', checked_step):
                for _ in range(30):
                    _, _, done, _, _ = env.step(np.ones(6))
                    if done:
                        break
            self.assertGreater(maximum, 0.)
        finally:
            env.close()

    def test_invalid_request_does_not_advance_physics(self):
        env = MotorResidualCascadeEnv(self.reference)
        try:
            env.reset(seed=0)
            state = np.r_[env.data.qpos, env.data.qvel].copy()
            for action in ([0.]*5, [float('nan')]*6):
                with self.assertRaises(ValueError):
                    env.step(action)
                np.testing.assert_array_equal(np.r_[env.data.qpos, env.data.qvel], state)
        finally:
            env.close()


class PrefixEnvironment(gym.Env):
    def __init__(self, unsafe=False):
        self.config = TransferConfig()
        self.target_throws = 30
        self.unsafe = unsafe

    def reset(self, **kwargs):
        return np.zeros(191), {}

    def step(self, action):
        return (np.zeros(191), -20. if self.unsafe else 0., self.unsafe, False,
                dict(catches=1, is_success=False, reason='FORBIDDEN_CONTACT' if self.unsafe else None))


class CurriculumAcceptanceTests(unittest.TestCase):
    def test_full_count_waits_for_the_underlying_final_hold(self):
        base = PrefixEnvironment()
        base.target_throws = 2
        base.step = lambda action: (np.zeros(191), 0., False, False,
            dict(catches=2, is_success=False, reason=None))
        schedule = [dict(min_steps=0, seat_jitter_m=.002, catch_goal=2, noise_std_m=[0.])]
        env = MotorCurriculum(base, schedule)
        env.reset(seed=0)
        with patch.object(env, '_potential', return_value=0.):
            _, reward, terminated, truncated, info = env.step(np.zeros(6))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 0.)
        self.assertFalse(info['curriculum_goal_reached'])
        self.assertEqual(info['curriculum_terminal_bonus'], 0.)
        self.assertTrue(env.active)
        base.step = lambda action: (np.zeros(191), 10., True, False,
            dict(catches=2, is_success=True, reason='SUCCESS'))
        _, reward, terminated, _, info = env.step(np.zeros(6))
        self.assertTrue(terminated)
        self.assertTrue(info['is_success'])
        self.assertEqual(reward, 10.)
        self.assertEqual(info['curriculum_terminal_bonus'], 0.)

    def test_time_limit_preserves_truncation_without_prefix_bonus(self):
        base = PrefixEnvironment()
        original = base.step

        def truncated_step(action):
            observation, reward, terminated, _, info = original(action)
            return observation, reward, terminated, True, info

        base.step = truncated_step
        env = MotorCurriculum(base, training_curriculum())
        env.reset(seed=0)
        with patch.object(env, '_potential', return_value=.4):
            _, reward, terminated, truncated, info = env.step(np.zeros(6))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(info['curriculum_goal_reached'])
        self.assertEqual(info['curriculum_terminal_bonus'], 0.)
        self.assertAlmostEqual(reward, .5*.99*.4)
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(6))

    def test_elapsed_steps_do_not_advance_an_unlearned_prefix(self):
        env = MotorCurriculum(PrefixEnvironment(), training_curriculum())
        env.steps = 10000
        env.outcomes.extend([False]*32)
        env.reset(seed=0)
        self.assertEqual(env.stage_index, 0)
        env.outcomes.clear()
        env.outcomes.extend([True]*32)
        env.reset(seed=0)
        self.assertEqual(env.stage_index, 1)
        self.assertEqual(env.stage_start_step, env.steps)

    def test_prefix_bonus_is_once_and_not_a_full_task_success(self):
        env = MotorCurriculum(PrefixEnvironment(), training_curriculum())
        env.reset(seed=0)
        _, reward, done, _, info = env.step(np.zeros(6))
        self.assertTrue(done)
        self.assertEqual(reward, 10.)
        self.assertTrue(info['curriculum_goal_reached'])
        self.assertFalse(info['is_success'])
        self.assertIsNone(info['reason'])
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(6))

    def test_prefix_cannot_override_a_physical_failure(self):
        env = MotorCurriculum(PrefixEnvironment(unsafe=True), training_curriculum())
        env.reset(seed=0)
        _, reward, done, _, info = env.step(np.zeros(6))
        self.assertTrue(done)
        self.assertEqual(reward, -20.)
        self.assertFalse(info['curriculum_goal_reached'])
        self.assertEqual(info['curriculum_terminal_bonus'], 0.)
        self.assertEqual(info['reason'], 'FORBIDDEN_CONTACT')


if __name__ == '__main__':
    unittest.main()

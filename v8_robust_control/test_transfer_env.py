"""Complete physical transfer, causal actor input and residual bounds."""
from dataclasses import asdict, replace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import unittest

from gymnasium.utils.env_checker import check_env
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO

from .transfer_env import TransferEnv, TransferConfig, prepare_reference


class TransferEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.reference = Path(cls.temporary.name)/'reference'
        prepare_reference(cls.reference)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def make_env(self, **settings):
        return TransferEnv(self.reference, TransferConfig(**settings))

    def test_zero_residual_reproduces_nominal_physics_and_completes_one_flight(self):
        env = self.make_env()
        try:
            observation, _ = env.reset(seed=1)
            self.assertEqual(observation.shape, (191,))
            rewards = []
            while True:
                _, reward, done, truncated, info = env.step(np.zeros(6))
                rewards.append(reward)
                if done or truncated:
                    break
            self.assertFalse(truncated)
            self.assertTrue(info['is_success'])
            self.assertEqual(info['reference_state_max_error'], 0.)
            self.assertGreaterEqual(info['initial_hold_max_s'], .05)
            self.assertGreaterEqual(info['final_hold_s'], .2-1e-10)
            self.assertGreaterEqual(info['airborne_s'], .3)
            self.assertTrue(all(value <= 0. for value in rewards[:-1]))
            self.assertEqual(sum(rewards), 10.)
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(6))
        finally:
            env.close()

    def test_reset_restores_contact_caches_and_reproduces_seeded_initial_variation(self):
        env = self.make_env(seat_jitter_m=.002, noise_std_m=.003, dropout=.1)
        try:
            values = []
            for _ in range(2):
                obs, info = env.reset(seed=91)
                start = (obs.copy(), env.data.qpos.copy(), info['initial_displacement_m'])
                for _ in range(6):
                    obs, reward, done, _, _ = env.step(np.zeros(6))
                    if done:
                        break
                values.append((*start, obs.copy(), env.data.qpos.copy(), reward))
            for first, second in zip(*values):
                np.testing.assert_array_equal(first, second)
        finally:
            env.close()

    def test_ball_truth_and_reference_verification_states_do_not_enter_policy_input(self):
        env = self.make_env()
        try:
            env.reset(seed=1)
            for _ in range(3):
                env.step(np.zeros(6))
            before = env._observation()
            env.data.qpos[env.q:env.q+3] += [.01, .02, .03]
            env.data.qvel[env.v:env.v+3] += [1., 2., 3.]
            mujoco.mj_kinematics(env.model, env.data)
            mujoco.mj_comPos(env.model, env.data)
            mujoco.mj_comVel(env.model, env.data)
            env.expected_states[:] = 1000.
            np.testing.assert_array_equal(before, env._observation())
        finally:
            env.close()

    def test_missing_and_delayed_frames_do_not_create_position_permissions(self):
        for settings in (dict(dropout=1.), dict(delay_s=.04)):
            env = self.make_env(**settings)
            try:
                obs, _ = env.reset(seed=1)
                for _ in range(2):
                    np.testing.assert_array_equal(obs[-6:-2], 0.)
                    obs, _, _, _, _ = env.step([1., -1., 1., -1., 0., 0.])
                    np.testing.assert_array_equal(env.joint_offset, 0.)
            finally:
                env.close()

    def test_joint_and_finger_residuals_remain_bounded_and_slew_limited(self):
        env = self.make_env()
        try:
            env.reset(seed=4)
            joints, grips = env.joint_offset.copy(), env.grip_offset.copy()
            for index in range(30):
                action = np.full(6, 10. if index % 2 else -10.)
                _, _, done, _, info = env.step(action)
                self.assertLessEqual(np.max(np.abs(env.joint_offset)), env.config.max_joint_offset_rad+1e-12)
                self.assertLessEqual(np.max(np.abs(env.grip_offset)), env.config.max_grip_offset_rad+1e-12)
                self.assertLessEqual(np.max(np.abs(env.joint_offset-joints)),
                                     env.config.max_joint_rate_rad_s*env.config.control_period_s+1e-12)
                self.assertLessEqual(np.max(np.abs(env.grip_offset-grips)),
                                     env.config.max_grip_rate_rad_s*env.config.control_period_s+1e-12)
                self.assertAlmostEqual(sum(info['reward_sums'].values()), info['episode_return'])
                self.assertFalse(np.any(env.data.eq_active))
                self.assertFalse(np.any(env.data.xfrc_applied))
                joints, grips = env.joint_offset.copy(), env.grip_offset.copy()
                if done:
                    break
        finally:
            env.close()

    def test_nonfinite_actions_cannot_mutate_physics_and_external_forces_fail(self):
        env = self.make_env()
        try:
            env.reset(seed=1)
            before = env.data.qpos.copy()
            for action in ([0., 0.], [0., 0., 0., 0., 0., float('nan')]):
                with self.assertRaises(ValueError):
                    env.step(action)
                np.testing.assert_array_equal(before, env.data.qpos)
            env.data.xfrc_applied[env.controller.ball_body_ids[0], 2] = .01
            _, reward, done, truncated, info = env.step(np.zeros(6))
            self.assertTrue(done)
            self.assertFalse(truncated)
            self.assertFalse(info['is_success'])
            self.assertEqual(info['reason'], 'EXTERNAL_BALL_FORCE')
            self.assertEqual(reward, -20.)
        finally:
            env.close()

    def test_gym_reset_step_contract(self):
        env = self.make_env()
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()

    def test_source_contact_after_confirmed_release_cannot_start_another_throw(self):
        env = self.make_env()
        try:
            env.reset(seed=1)
            for _ in range(4):
                _, _, done, _, _ = env.step(np.zeros(6))
                self.assertFalse(done)
            self.assertTrue(env.contacts[0])
            # Keep the real, loaded source-hand contact, but present it after
            # the flight detector has confirmed separation. It must fail,
            # rather than allowing a new source impulse to earn catch credit.
            env.released = True
            env.release_time = float(env.data.time)-.02
            _, reward, done, truncated, info = env.step(np.zeros(6))
            self.assertTrue(done)
            self.assertFalse(truncated)
            self.assertEqual(info['reason'], 'SOURCE_RECONTACT')
            self.assertFalse(info['is_success'])
            self.assertLess(reward, 0.)
        finally:
            env.close()

    def test_masked_ppo_starts_at_zero_and_saved_policy_roundtrips(self):
        from .train_transfer import create_model
        torch.set_num_threads(1)
        env = self.make_env()
        try:
            actor = create_model(env, 0)
            obs, _ = env.reset(seed=1)
            np.testing.assert_array_equal(actor.predict(obs, deterministic=True)[0], 0.)
            batch = torch.zeros((4, 191))
            actions = torch.full((4, 6), .1)
            _, log_prob, _ = actor.policy.evaluate_actions(batch, actions)
            log_prob.sum().backward()
            np.testing.assert_array_equal(actor.policy.action_net.bias.grad.numpy(), 0.)
            actor.policy.zero_grad()
            batch[:, -6] = 1.
            _, log_prob, _ = actor.policy.evaluate_actions(batch, actions)
            log_prob.sum().backward()
            self.assertNotEqual(actor.policy.action_net.bias.grad[0].item(), 0.)
            np.testing.assert_array_equal(actor.policy.action_net.bias.grad[1:].numpy(), 0.)
            actor.learn(128)
            before = actor.predict(obs, deterministic=True)[0]
            with TemporaryDirectory() as directory:
                path = Path(directory)/'policy.zip'
                actor.save(path)
                loaded = PPO.load(path, device='cpu')
                np.testing.assert_array_equal(before, loaded.predict(obs, deterministic=True)[0])
                self.assertTrue(np.all(np.isfinite(before)))
        finally:
            env.close()

    def test_fixed_and_timed_controls_do_not_consume_ball_truth(self):
        from .train_transfer import Fixed, Frozen, Timed
        fixed = Fixed([.1]*6)
        observation = np.zeros(191)
        first = fixed.predict(observation)[0]
        first[:] = 0.
        np.testing.assert_array_equal(fixed.predict(np.ones(191))[0], .1)
        class Actor:
            def predict(self, obs, deterministic=True):
                return obs[:6].copy(), None
        frozen = Frozen(Actor())
        observation[:6] = .2
        np.testing.assert_array_equal(frozen.predict(observation)[0], 0.)
        observation[12] = 1.
        np.testing.assert_array_equal(frozen.predict(observation)[0], .2)
        observation[:6] = .8
        np.testing.assert_array_equal(frozen.predict(observation)[0], .2)
        frozen.reset()
        np.testing.assert_array_equal(frozen.predict(observation)[0], .8)
        schedule = np.arange(80*6, dtype=float).reshape(80, 6)/1000.
        timed = Timed(schedule)
        observation[168] = .04/1.6
        np.testing.assert_array_equal(timed.predict(observation)[0], schedule[2])

    def test_module_entry_saves_policy_loadable_from_another_process(self):
        # Exercise the -m entry dispatch without running a full training study.
        # Loading here must not require aliases on this process's __main__.
        code = '''
import runpy
import sys
import warnings
from pathlib import Path
import torch
import v8_robust_control.train_transfer as training
from v8_robust_control.transfer_env import TransferEnv
def smoke():
    torch.set_num_threads(1)
    env = TransferEnv(Path(sys.argv[1]))
    try:
        training.create_model(env, 0).save(Path(sys.argv[2]))
    finally:
        env.close()
training.main = smoke
warnings.filterwarnings('ignore', category=RuntimeWarning)
runpy.run_module('v8_robust_control.train_transfer', run_name='__main__', alter_sys=True)
'''
        with TemporaryDirectory() as directory:
            path = Path(directory)/'policy.zip'
            result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code,
                                     str(self.reference), str(path)],
                                    cwd=Path(__file__).resolve().parents[1],
                                    capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            loaded = PPO.load(path, device='cpu')
            self.assertEqual(type(loaded.policy).__module__, 'v8_robust_control.train_transfer')
            self.assertEqual(type(loaded.policy.features_extractor).__module__,
                             'v8_robust_control.train_transfer')

    def test_validation_cannot_accept_tied_controls_or_modified_evidence(self):
        from .train_transfer import CASES, Fixed, evaluate, fixed_candidates, counts, score, summarize
        trials = evaluate(Fixed(), self.reference, [380000])
        controls = [dict(**candidate, trials=deepcopy(trials)) for candidate in fixed_candidates()]
        point = dict(step=128, checkpoint_sha256='fixture', trials=trials,
                     score=list(score(trials, counts(trials))))
        models = [dict(seed=seed, training_steps=128, selected_step=128, checkpoint_sha256='fixture',
                       save_load_verified=True, curve=[deepcopy(point)], trials=deepcopy(trials),
                       frozen_trials=deepcopy(trials), timed_trials=deepcopy(trials),
                       timed_actions=np.zeros((80, 6)).tolist(),
                       timed_fit=dict(actions=np.zeros((80, 6)).tolist(), sums=np.zeros((80, 6)).tolist(),
                                      counts=np.ones(80, dtype=int).tolist())) for seed in (0, 1, 2)]
        record = dict(manifest=dict(cases=CASES, training_seeds=[0, 1, 2], validation_seeds=[380000],
                                    observation_size=191, action_size=6, task_config=asdict(TransferConfig()),
                                    checkpoints=[128]), references=controls, models=models)
        self.assertFalse(summarize(record)['all_candidates_passed'])
        for change in ('missing', 'fit', 'budget', 'overlap', 'case'):
            altered = deepcopy(record)
            if change == 'missing':
                altered['references'].pop()
            elif change == 'fit':
                altered['models'][0]['timed_fit']['sums'][0][0] = 1.
            elif change == 'budget':
                altered['models'][0]['training_steps'] = 256
            elif change == 'overlap':
                altered['manifest']['validation_seeds'] = [0]
            else:
                altered['models'][0]['trials'][0]['config']['seat_jitter_m'] = .01
            with self.assertRaises(ValueError):
                summarize(altered)


if __name__ == '__main__':
    unittest.main()

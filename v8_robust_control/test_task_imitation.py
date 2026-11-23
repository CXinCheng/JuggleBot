from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO

from .task_env import TaskConfig, observation
from .test_task_env import request
from .task_imitation import (NormalizedMeasurements, initialize_model, tensors,
                            observation_statistics, fit_actor, loss_value, validate_episode_splits,
                            select_checkpoints, evaluate_training_cases)


class MeasurementEnv(gym.Env):
    def __init__(self, reference):
        self.reference_path = reference
        self.obs = observation(request(), TaskConfig())
        self.observation_space = spaces.Dict({k:spaces.Box(-10., 10., v.shape, np.float32)
                                               for k,v in self.obs.items()})
        self.action_space = spaces.Box(-1., 1., (6,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {k:v.copy() for k,v in self.obs.items()}, {}

    def step(self, action):
        if not np.all(np.isfinite(action)) or np.any(np.abs(action)>1.):
            raise ValueError('Unbounded policy action')
        return {k:v.copy() for k,v in self.obs.items()}, 1.-float(np.mean(action**2)), True, False, {}


class TaskImitationTests(unittest.TestCase):
    def test_validation_and_test_splits_are_by_whole_episodes(self):
        validate_episode_splits([12], [20, 21], {10, 11})
        with self.assertRaises(ValueError):
            validate_episode_splits([12], [13], {10, 11, 12, 13})
        with self.assertRaises(ValueError):
            validate_episode_splits([12], [12], {10, 11})

    def test_loss_weights_episodes_equally_and_rejects_extra_inputs(self):
        with TemporaryDirectory() as tmp:
            env = MeasurementEnv(Path(tmp)/'reference.json')
            obs = {k:v.tolist() for k,v in env.obs.items()}
            item = dict(observation=obs, action=[0.]*6)
            batch = tensors([dict(demonstrations=[item]), dict(demonstrations=[item]*4)], env.observation_space)
            self.assertAlmostEqual(float(batch[2][0]), float(batch[2][1:].sum()))
            bad = dict(observation=dict(obs, future_outcome=[1.]), action=[0.]*6)
            with self.assertRaises(ValueError):
                tensors([dict(demonstrations=[bad])], env.observation_space)

    def test_normalization_is_frozen_and_rejects_invalid_scales(self):
        with TemporaryDirectory() as tmp:
            env = MeasurementEnv(Path(tmp)/'reference.json')
            obs = {k:torch.as_tensor(v[None]) for k,v in env.obs.items()}
            center, scale = observation_statistics(obs, tuple(env.observation_space.spaces))
            extractor = NormalizedMeasurements(env.observation_space, center, scale)
            before = extractor.center.clone()
            changed = {k:v+1. for k,v in obs.items()}
            extractor(changed)
            torch.testing.assert_close(extractor.center, before)
            self.assertFalse(extractor.center.requires_grad)
            with self.assertRaises(ValueError):
                NormalizedMeasurements(env.observation_space, center, [0.]*len(scale))

    def test_actor_fits_supervision_preserves_critic_and_can_continue_ppo(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as tmp:
            reference = Path(tmp)/'reference.json'
            reference.write_text('{}', encoding='utf-8')
            env = MeasurementEnv(reference)
            observations = {k:torch.as_tensor(np.repeat(v[None], 8, axis=0)) for k,v in env.obs.items()}
            observations['errors'][:, 0, 0] = torch.tensor([-1., 1.]*4)
            actions = torch.zeros((8, 6))
            actions[:, 0] = torch.tensor([-.1, .1]*4)
            batch = observations, actions, torch.ones(8)
            center, scale = observation_statistics(observations, tuple(env.observation_space.spaces))
            model = initialize_model(env, 0, center, scale, 6)
            critic_before = {k:v.clone() for k,v in model.policy.mlp_extractor.value_net.state_dict().items()}
            initial_loss = float(loss_value(model.policy, batch).detach())
            fit_actor(model, batch, 150)
            self.assertLess(float(loss_value(model.policy, batch).detach()), initial_loss*.1)
            for k,v in model.policy.mlp_extractor.value_net.state_dict().items():
                torch.testing.assert_close(v, critic_before[k], rtol=0., atol=0.)
            outside = {k:v.clone() for k,v in observations.items()}
            outside['context'][:, 5] = 6./30
            torch.testing.assert_close(model.policy._predict(outside, deterministic=True), torch.zeros((8, 6)))
            model.policy.zero_grad()
            _, log_probability, _ = model.policy.evaluate_actions(outside, torch.zeros((8, 6)))
            log_probability.sum().backward()
            torch.testing.assert_close(model.policy.action_net.weight.grad,
                                       torch.zeros_like(model.policy.action_net.weight.grad))
            path = Path(tmp)/'ppo_bc.zip'
            model.save(path)
            loaded = PPO.load(path, env=env, device='cpu')
            torch.testing.assert_close(loaded.policy._predict(observations, deterministic=True),
                                       model.policy._predict(observations, deterministic=True), rtol=0., atol=0.)
            loaded.learn(64)
            self.assertEqual(loaded.num_timesteps, 64)
            self.assertTrue(all(torch.all(torch.isfinite(p)) for p in loaded.policy.parameters()))

    def test_selection_uses_validation_safety_and_recovery_not_training_loss(self):
        models = [dict(seed=0, epoch=i, checkpoint=str(i), training_mse=1./(i+1)) for i in (0, 10, 20)]
        def episode(reason, recovered):
            return dict(result=dict(reason=reason, training_prefix=reason=='TRAINING_PREFIX', recovered_ids=recovered))
        validation = [dict(checkpoint='0', validation_rollouts=[episode('DROPPED', [])]),
                      dict(checkpoint='10', validation_rollouts=[episode('TRAINING_PREFIX', [1,2,3])]),
                      dict(checkpoint='20', validation_rollouts=[episode('POSE_LIMIT', [1,2])])]
        self.assertEqual(select_checkpoints(models, validation)[0]['epoch'], 10)

    def test_training_replays_keep_each_demonstrated_domain_and_restore_evaluation_domain(self):
        env = SimpleNamespace(schedule=[dict(seat_jitter_m=.0005)])
        seen = []
        def run(current, actor, seed, mode):
            jitter = current.schedule[0]['seat_jitter_m']
            seen.append((seed, jitter))
            return dict(result=dict(config=dict(seat_jitter_m=jitter)))
        cases = [dict(seed=410402, jitter_m=.0005), dict(seed=410200, jitter_m=.002)]
        with patch('v8_robust_control.task_imitation.rollout', side_effect=run):
            evaluate_training_cases(env, None, cases)
        self.assertEqual(seen, [(410402, .0005), (410200, .002)])
        self.assertEqual(env.schedule[0]['seat_jitter_m'], .0005)


if __name__ == '__main__':
    unittest.main()

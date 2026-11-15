"""PPO defaults, source compatibility and checkpoint continuity."""
import argparse
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
import random
import torch

from . import train_residual
from .evaluate_residual import compatible_training_sources, select_training_rows


class ResidualCliTests(unittest.TestCase):
    def parse_training(self, arguments):
        parsed = []
        original = argparse.ArgumentParser.parse_args
        class ParsedOnly(Exception):
            pass
        def intercept(parser, *args, **kwargs):
            parsed.append(original(parser, ['--output-dir', 'unused', *arguments]))
            raise ParsedOnly
        with patch.object(argparse.ArgumentParser, 'parse_args', intercept):
            with self.assertRaises(ParsedOnly):
                train_residual.main()
        return parsed[0]

    def test_training_defaults_to_ppo(self):
        self.assertEqual(self.parse_training([]).algorithms, ['ppo'])

    def test_sac_and_comparison_are_explicit_options(self):
        self.assertEqual(self.parse_training(['--algorithms', 'sac']).algorithms, ['sac'])
        self.assertEqual(self.parse_training(['--algorithms', 'sac', 'ppo']).algorithms, ['sac', 'ppo'])

    def test_evaluation_selects_only_recorded_algorithms(self):
        ppo = dict(kind='trained', algorithm='ppo')
        sac = dict(kind='trained', algorithm='sac')
        self.assertEqual(select_training_rows([ppo]), [ppo])
        self.assertEqual(select_training_rows([ppo, sac]), [ppo, sac])
        self.assertEqual(select_training_rows([ppo, sac], ['sac']), [sac])
        with self.assertRaises(ValueError):
            select_training_rows([ppo], ['sac'])
        with self.assertRaises(ValueError):
            select_training_rows([dict(kind='trained', algorithm='unknown')])

    def test_old_observation_runtime_is_not_silently_reused(self):
        current = train_residual.source_hashes()
        self.assertTrue(compatible_training_sources(current))
        self.assertFalse(compatible_training_sources({'legacy_runtime.py': '0'*64}))
        for name in ('residual_env.py', 'train_residual.py'):
            changed = deepcopy(current)
            changed['v8_robust_control/'+name] = '0'*64
            self.assertFalse(compatible_training_sources(changed))

    def test_checkpoint_validation_does_not_reset_ongoing_training(self):
        class Policy:
            def __init__(self):
                self.num_timesteps = 0
                self._last_obs = np.ones((1, 73))
                self.logger = SimpleNamespace(name_to_value={})
                self.calls = []

            def learn(self, total_timesteps, callback, reset_num_timesteps):
                self.calls.append((total_timesteps, reset_num_timesteps))
                self.num_timesteps += total_timesteps

            def predict(self, obs, deterministic=True):
                np.testing.assert_array_equal(obs, np.ones((1, 73)))
                return np.zeros((1, 4)), None

            def save(self, path):
                path.write_text(str(self.num_timesteps))

            @classmethod
            def load(cls, path, device):
                value = cls()
                value.num_timesteps = int(path.read_text())
                return value

        class Environment:
            def reset(self, **kwargs):
                raise AssertionError('Training must not reset between chunks')

            def close(self):
                pass

        policy = Policy()
        trials = [dict(case=c, is_success=True, reason='SUCCESS', residual_rms=0., policy='learned')
                  for c in train_residual.EVALUATION_CASES]
        with TemporaryDirectory() as directory, \
                patch.object(train_residual, 'ResidualCatchEnv'), \
                patch.object(train_residual, 'RandomizedReception', return_value=Environment()), \
                patch.object(train_residual, 'create_model', return_value=policy), \
                patch.object(train_residual, 'evaluate', return_value=trials):
            row = train_residual.train_run(('ppo', 0, .0003, 0., 256,
                train_residual.CatchConfig(), [60000], Path(directory), [128, 256],
                {c: 1 for c in train_residual.EVALUATION_CASES}))
        self.assertEqual(policy.calls, [(128, False), (128, False)])
        self.assertEqual(row['training_steps'], 256)
        self.assertEqual(row['selected_training_steps'], 128)

    def test_model_evaluation_cannot_reseed_training_randomness(self):
        with train_residual.isolated_evaluation_rng():
            def seed(value):
                random.seed(value)
                np.random.seed(value)
                torch.manual_seed(value)
            def draw():
                return [random.random(), float(np.random.random()), float(torch.rand(()))]
            seed(42)
            expected = draw()
            seed(42)
            with train_residual.isolated_evaluation_rng():
                seed(99)
                draw()
            self.assertEqual(draw(), expected)


if __name__ == '__main__':
    unittest.main()

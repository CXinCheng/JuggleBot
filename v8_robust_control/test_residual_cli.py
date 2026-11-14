"""PPO defaults, metadata-driven selection and narrow source compatibility."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

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

    def test_recorded_run_survives_only_the_cli_default_change(self):
        root = Path(__file__).parent
        recorded = json.loads((root/'results/reception_training.json').read_text(encoding='utf-8'))['manifest']['source_sha256']
        self.assertTrue(compatible_training_sources(recorded))
        changed = deepcopy(recorded)
        changed['v8_robust_control/residual_env.py'] = '0'*64
        self.assertFalse(compatible_training_sources(changed))
        changed = deepcopy(recorded)
        changed['v8_robust_control/train_residual.py'] = '0'*64
        self.assertFalse(compatible_training_sources(changed))


if __name__ == '__main__':
    unittest.main()

"""Checkpoint archival and preservation of the continuous physical state."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from jugglebot.robustness.controller import ObservedCascade
from .verify_baselines import main


class TestBaselineCheckpoints(unittest.TestCase):
    def test_one_controller_records_both_checkpoints_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = Mock()
            shared = {'count': 0}

            def run(count, timeout, guard):
                shared['count'] = count
                return dict(success=True, validated_flights=count, metrics=shared)

            controller.run_continuous.side_effect = run
            with (patch.object(sys, 'argv', ['verify', '--observation-mode', 'sampled', '--output-dir', directory]),
                  patch('v7_robust_control.verify_baselines.ObservedCascade', return_value=controller) as construct,
                  contextlib.redirect_stdout(io.StringIO())):
                main()
            construct.assert_called_once_with(observation_mode='sampled')
            self.assertEqual(controller.run_continuous.call_args_list,
                             [unittest.mock.call(30, 30., guard=True), unittest.mock.call(50, 35., guard=True)])
            records = [json.loads((Path(directory)/f'sampled_baseline{count}.json').read_text()) for count in (30,50)]
            self.assertEqual([record['result']['metrics']['count'] for record in records], [30,50])
            self.assertEqual(records[0]['source_sha256'], records[1]['source_sha256'])
            self.assertIn('v7_robust_control/verify_baselines.py', records[0]['source_sha256'])
            self.assertEqual(records[0]['continuous_checkpoints'], [30,50])

    def test_existing_checkpoint_rejects_the_whole_run_before_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory)/'truth_baseline50.json'
            existing.write_text('original', encoding='utf-8')
            with (patch.object(sys, 'argv', ['verify', '--output-dir', directory]),
                  patch('v7_robust_control.verify_baselines.ObservedCascade') as construct,
                  contextlib.redirect_stderr(io.StringIO())):
                with self.assertRaises(SystemExit) as stopped:
                    main()
            self.assertEqual(stopped.exception.code, 2)
            construct.assert_not_called()
            self.assertEqual(existing.read_text(), 'original')
            self.assertFalse((Path(directory)/'truth_baseline30.json').exists())

    def test_failed_checkpoint_is_saved_without_a_fabricated_later_result(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = Mock()
            result = dict(success=False, validated_flights=23, failure_event=dict(reason='NO_FEASIBLE_RECEPTION'))
            controller.run_continuous.return_value = result
            with (patch.object(sys, 'argv', ['verify', '--output-dir', directory]),
                  patch('v7_robust_control.verify_baselines.ObservedCascade', return_value=controller),
                  contextlib.redirect_stdout(io.StringIO())):
                with self.assertRaises(SystemExit) as stopped:
                    main()
            self.assertEqual(stopped.exception.code, 1)
            controller.run_continuous.assert_called_once_with(30, 30., guard=True)
            record = json.loads((Path(directory)/'truth_baseline30.json').read_text())
            self.assertEqual(record['result'], result)
            self.assertFalse((Path(directory)/'truth_baseline50.json').exists())

    def test_checkpoint_summary_does_not_change_future_physics(self):
        continuous = ObservedCascade()
        checkpointed = ObservedCascade()
        continuous.planning = checkpointed.planning = False
        for index in range(700):
            continuous.step()
            checkpointed.step()
            if index == 399:
                checkpointed.run_continuous(1, float(checkpointed.data.time), guard=True)
            np.testing.assert_array_equal(continuous.data.qpos, checkpointed.data.qpos)
            np.testing.assert_array_equal(continuous.data.qvel, checkpointed.data.qvel)
            np.testing.assert_array_equal(continuous.data.ctrl, checkpointed.data.ctrl)
        self.assertEqual(continuous.events, checkpointed.events)


if __name__ == '__main__':
    unittest.main()

"""Experiment archival and failure-reporting checks without physical searches."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

from v8_robust_control.juggle import main


class TestExperimentOutput(unittest.TestCase):
    def run_case(self, destination, success):
        result = dict(success=success, validated_flights=30 if success else 4,
                      throws=32 if success else 6, time=10.21 if success else 1.92,
                      drops=0, ball_contacts=0,
                      error=None if success else 'NO_FEASIBLE_RECEPTION',
                      failure_event=None if success else dict(reason='NO_FEASIBLE_RECEPTION'),
                      conditions=dict(ball_observation=dict(mode='sampled')))
        controller = Mock()
        controller.run_continuous.return_value = result
        argv = ['juggle', '--throws', '30', '--observation-mode', 'sampled', '--output', str(destination)]
        with (patch.object(sys, 'argv', argv),
              patch('v8_robust_control.juggle.ObservedCascade', return_value=controller),
              contextlib.redirect_stdout(io.StringIO())):
            if success:
                main()
            else:
                with self.assertRaises(SystemExit) as stopped:
                    main()
                self.assertEqual(stopped.exception.code, 1)
        record = json.loads(destination.read_text(encoding='utf-8'))
        self.assertEqual(record['result'], result)
        self.assertEqual(record['requested_flights'], 30)
        self.assertEqual(record['simulated_timeout_s'], 30.)
        self.assertEqual(len(record['optimizer_inputs_sha256']), 64)
        self.assertIn('src/jugglebot/robustness/observation.py', record['source_sha256'])
        controller.run_continuous.assert_called_once_with(30, 30., guard=True)

    def test_successful_baseline_is_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_case(Path(directory) / 'nested' / 'baseline.json', True)

    def test_failed_baseline_is_archived_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_case(Path(directory) / 'baseline.json', False)

    def test_existing_archive_is_rejected_before_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'baseline.json'
            destination.write_text('original', encoding='utf-8')
            with (patch.object(sys, 'argv', ['juggle', '--output', str(destination)]),
                  patch('v8_robust_control.juggle.ObservedCascade') as constructor,
                  contextlib.redirect_stderr(io.StringIO())):
                with self.assertRaises(SystemExit) as stopped:
                    main()
                self.assertEqual(stopped.exception.code, 2)
                constructor.assert_not_called()
            self.assertEqual(destination.read_text(encoding='utf-8'), 'original')


if __name__ == '__main__':
    unittest.main()

"""Fast checks for passive metrics and experiment bookkeeping."""
import unittest
import copy
import csv
import io
import contextlib
import sys
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from jugglebot.humanoid.metrics import ExecutionMetrics
from jugglebot.humanoid.cascade import ContactCascade
from jugglebot.humanoid.logging import tee_output
from v6_humanoid_hands.evaluate_noise import failure_category, summarize
from v6_humanoid_hands.summarize_noise import (
    export_report, load_experiment, render_report, render_trials_csv, render_execution_csv,
)


class TestEvaluation(unittest.TestCase):
    def test_metrics_measure_signed_limits_and_gear_without_writes(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <worldbody><body><joint name="hinge"/><geom size=".1"/></body></worldbody>
          <actuator><motor name="motor" joint="hinge" gear="2"
             forcelimited="true" forcerange="-1 1"/></actuator></mujoco>''')
        data = mujoco.MjData(model)
        metrics = ExecutionMetrics(model)
        for speed, force in [(-3., .99), (4., -1.)]:
            data.qvel[0] = speed
            data.actuator_force[0] = force
            before = [array.copy() for array in (data.qpos, data.qvel, data.ctrl, data.actuator_force)]
            metrics.record(data)
            for actual, expected in zip((data.qpos, data.qvel, data.ctrl, data.actuator_force), before):
                np.testing.assert_array_equal(actual, expected)
        result = metrics.summary()
        row = result['actuators'][0]
        self.assertEqual(result['samples'], 2)
        self.assertEqual(row['peak_speed_rad_s'], 4.)
        self.assertEqual(row['peak_actuator_torque_nm'], 2.)
        self.assertAlmostEqual(row['rms_actuator_torque_nm'], np.sqrt((1.98**2+2**2)/2))
        self.assertEqual(row['configured_torque_range_nm'], [-2., 2.])
        self.assertEqual(row['near_force_limit_fraction'], 1.)
        self.assertFalse(result['hardware_limits_verified'])

    def test_unlimited_actuator_has_no_limit_fraction(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody>
          <actuator><motor joint="j"/></actuator></mujoco>''')
        metrics = ExecutionMetrics(model)
        metrics.record(mujoco.MjData(model))
        row = metrics.summary()['actuators'][0]
        self.assertIsNone(row['near_force_limit_fraction'])
        self.assertIsNone(row['configured_torque_range_nm'])

    def test_seed_wiring_and_passive_live_sampling(self):
        a = ContactCascade(seed=17)
        b = ContactCascade(seed=17)
        np.testing.assert_array_equal(a.rng.normal(size=12), b.rng.normal(size=12))
        self.assertEqual(a.conditions_summary()['observation_noise_seed'], 17)
        a.planning = b.planning = False
        b.execution_metrics = Mock()
        for _ in range(20):
            a.step()
            b.step()
        for actual, expected in zip((a.data.qpos, a.data.qvel, a.data.ctrl),
                                    (b.data.qpos, b.data.qvel, b.data.ctrl)):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(a.execution_metrics.count, 20)
        a.silent = True
        a.step()
        self.assertEqual(a.execution_metrics.count, 20, 'Private planning steps must not enter metrics')

    def test_failure_categories_and_summary_do_not_count_initial_ball(self):
        base = dict(success=False, failure_event=None, error='TIMEOUT', drops=0,
                    ball_contacts=0, validated_flights=2)
        self.assertEqual(failure_category(base), 'timeout')
        self.assertEqual(failure_category(dict(base, error='NO_FEASIBLE_RECEPTION: ...')),
                         'reception_planning_rejected')
        self.assertEqual(failure_category(dict(base, drops=1)), 'drop')
        self.assertEqual(failure_category(dict(base, ball_contacts=1)), 'ball_ball_collision')
        good = dict(base, success=True, error=None, validated_flights=3)
        rows = [dict(noise_std_mm=5., result=r, outcome=failure_category(r)) for r in (base, good)]
        summary = summarize(rows)[0]
        self.assertEqual(summary['success_rate'], .5)
        self.assertEqual(summary['completed_requested_flights'], 5)


class TestNoiseReports(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / 'experiment.jsonl'
        header = dict(kind='experiment', schema_version=1, noise_std_mm=[0., 5.], seeds=[0],
                      requested_flights=2, simulated_timeout_s=5., jobs=1,
                      optimizer_inputs_sha256='a'*64, source_sha256={'cascade.py': 'b'*64})
        trials = []
        for noise, completed, dropped in ((0., 2, 0), (5., 1, 1)):
            conditions = dict(mujoco_version='test', timestep_s=.001, ball_diameter_m=.085,
                              ball_mass_kg=.12, throw_interval_s=.32, flight_target_s=.53,
                              observation_noise_seed=0, partial_prediction_noise_m=noise/1000.)
            result = dict(success=not bool(dropped), requested_throws=2, validated_flights=completed,
                          drops=dropped, ball_contacts=0, conditions=conditions,
                          time=.02, throws=completed+2,
                          execution=dict(samples=20, sample_period_s=.001, hardware_limits_verified=False,
                                         actuators=[dict(actuator='motor', joint='j', peak_speed_rad_s=2.+noise,
                                                         peak_actuator_torque_nm=.5,
                                                         rms_actuator_torque_nm=.2,
                                                         configured_torque_range_nm=[-.5, .5],
                                                         near_force_limit_fraction=.25)]),
                          error='DROPPED' if dropped else None,
                          failure_event={'reason': 'DROPPED', 'height': .39} if dropped else None)
            trials.append(dict(kind='trial', noise_std_mm=noise, seed=0,
                               outcome=failure_category(result), result=result))
        self.records = [header, *trials, dict(kind='summary', groups=summarize(trials))]

    def write_source(self, records):
        self.source.write_text(''.join(json.dumps(row)+'\n' for row in records), encoding='utf-8')

    def test_report_preserves_outcomes_and_ignores_completion_order(self):
        self.write_source(self.records)
        header, trials = load_experiment(self.source)
        text = render_report(header, trials)
        self.assertEqual(text, render_report(header, list(reversed(trials))))
        self.assertIn('Completed trials: 2', text)
        self.assertIn('1/1 completed the requested task', text)
        self.assertIn('0/1 completed the requested task', text)
        self.assertIn('DROPPED', text)
        self.assertIn('"height": 0.39', text)
        self.assertIn('a'*64, text)
        self.assertIn('--input v6_humanoid_hands/results/noise_sensitivity.jsonl', text)
        self.assertIn('--output v6_humanoid_hands/results/noise_sensitivity.jsonl', text)
        self.assertNotIn('.rerun', text)
        self.assertNotIn('build/v6_humanoid_hands/noise_sensitivity.jsonl', text)

    def test_incomplete_duplicate_and_inconsistent_data_are_rejected(self):
        bad_summary = copy.deepcopy(self.records)
        bad_summary[-1]['groups'][0]['successes'] = 0
        bad_outcome = copy.deepcopy(self.records)
        bad_outcome[2]['outcome'] = 'success'
        bad_condition = copy.deepcopy(self.records)
        bad_condition[2]['result']['conditions']['observation_noise_seed'] = 1
        bad_success = copy.deepcopy(self.records)
        bad_success[1]['result']['drops'] = 1
        bad_header = copy.deepcopy(self.records)
        bad_header[0]['seeds'] = [0, 0]
        variants = [self.records[:-1], [self.records[0], self.records[1], self.records[1], self.records[-1]],
                    bad_summary, bad_outcome, bad_condition, bad_success, bad_header]
        for records in variants:
            with self.subTest(records=records):
                self.write_source(records)
                with self.assertRaises(ValueError):
                    load_experiment(self.source)

    def test_existing_report_is_never_overwritten(self):
        self.write_source(self.records)
        output = Path(self.directory.name) / 'report.txt'
        export_report(self.source, output)
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            export_report(self.source, output)
        self.assertEqual(output.read_bytes(), before)

    def test_explicit_overwrite_validates_input_before_replacing_report(self):
        self.write_source(self.records)
        raw = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, 'different files'):
            export_report(self.source, self.source, overwrite=True)
        self.assertEqual(self.source.read_bytes(), raw)
        output = Path(self.directory.name) / 'report.txt'
        output.write_text('previous report', encoding='utf-8')
        export_report(self.source, output, overwrite=True)
        self.assertIn('Completed trials: 2', output.read_text(encoding='utf-8'))
        before = output.read_bytes()
        self.write_source(self.records[:-1])
        with self.assertRaises(ValueError):
            export_report(self.source, output, overwrite=True)
        self.assertEqual(output.read_bytes(), before)

    def test_trial_csv_allows_independent_counting(self):
        self.write_source(self.records)
        header, trials = load_experiment(self.source)
        text = render_trials_csv(header, trials)
        self.assertEqual(text, render_trials_csv(header, list(reversed(trials))))
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual([float(row['noise_std_mm']) for row in rows], [0., 5.])
        self.assertEqual([int(row['completed_flights']) for row in rows], [2, 1])
        self.assertEqual(sum(int(row['success']) for row in rows), 1)
        self.assertEqual(sum(int(row['drops']) for row in rows), 1)
        self.assertEqual(rows[1]['failure_height_m'], '0.39')
        self.assertEqual({row['optimizer_inputs_sha256'] for row in rows}, {'a'*64})

    def test_execution_csv_selects_zero_noise_and_preserves_units(self):
        self.write_source(self.records)
        header, trials = load_experiment(self.source)
        rows = list(csv.DictReader(io.StringIO(render_execution_csv(header, list(reversed(trials))))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['peak_speed_rad_s'], '2.0')
        self.assertEqual(rows[0]['rms_actuator_torque_nm'], '0.2')
        self.assertEqual(rows[0]['configured_torque_min_nm'], '-0.5')
        self.assertEqual(rows[0]['live_samples'], '20')
        self.assertEqual(rows[0]['hardware_limits_verified'], '0')
        trials[0]['result']['execution']['actuators'][0]['configured_torque_range_nm'] = None
        trials[0]['result']['execution']['actuators'][0]['near_force_limit_fraction'] = None
        row = next(csv.DictReader(io.StringIO(render_execution_csv(header, trials))))
        self.assertEqual(row['configured_torque_min_nm'], '')
        self.assertEqual(row['near_force_limit_fraction'], '')
        with self.assertRaisesRegex(ValueError, 'zero-noise'):
            render_execution_csv(header, trials[1:])
        noisy_header = dict(header, noise_std_mm=[5.])
        self.assertNotIn('--execution-output', render_report(noisy_header, trials[1:]))

    def test_csv_conflicts_do_not_partially_replace_reports(self):
        self.write_source(self.records)
        output = Path(self.directory.name) / 'report.txt'
        csv_path = Path(self.directory.name) / 'trials.csv'
        csv_path.write_text('keep', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            export_report(self.source, output, csv_output=csv_path)
        self.assertFalse(output.exists())
        self.assertEqual(csv_path.read_text(encoding='utf-8'), 'keep')
        with self.assertRaisesRegex(ValueError, 'different files'):
            export_report(self.source, output, csv_output=output, overwrite=True)
        with self.assertRaisesRegex(ValueError, 'different files'):
            export_report(self.source, output, execution_output=self.source, overwrite=True)
        broken = copy.deepcopy(self.records)
        del broken[1]['result']['execution']
        self.write_source(broken)
        with self.assertRaises(KeyError):
            export_report(self.source, output, csv_output=csv_path,
                          execution_output=Path(self.directory.name)/'execution.csv', overwrite=True)
        self.assertFalse(output.exists())
        self.assertEqual(csv_path.read_text(encoding='utf-8'), 'keep')


class TestVerificationLogging(unittest.TestCase):
    def test_utf8_tee_restores_streams_and_preserves_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'nested' / 'verification.txt'
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    with tee_output(output):
                        print('验证记录')
                        print('错误信息', file=sys.stderr)
                        raise SystemExit(3)
                self.assertEqual(raised.exception.code, 3)
                self.assertIs(sys.stdout, stdout)
                self.assertIs(sys.stderr, stderr)
                with tee_output():
                    print('ordinary output')
            self.assertEqual(output.read_text(encoding='utf-8'), '验证记录\n错误信息\n')
            self.assertEqual(stdout.getvalue(), '验证记录\nordinary output\n')
            self.assertEqual(stderr.getvalue(), '错误信息\n')

    def test_assembly_failure_is_logged_and_returns_nonzero(self):
        from v6_humanoid_hands import verify_g1_hands
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'failure.txt'
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(sys, 'argv', ['verify_g1_hands.py', '--log', str(output)]), \
                    patch.object(verify_g1_hands, 'verify_assembly', side_effect=RuntimeError('装配检查失败')), \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(verify_g1_hands.main(), 1)
            log = output.read_text(encoding='utf-8')
            self.assertIn('Upper-Body G1', log)
            self.assertIn('Traceback', log)
            self.assertIn('RuntimeError: 装配检查失败', log)
            self.assertIn('RuntimeError: 装配检查失败', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()

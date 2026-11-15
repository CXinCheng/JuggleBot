"""Validate fixed-action selection, seed separation and comparable RL evidence."""
from copy import deepcopy
import hashlib
import json
import unittest

import numpy as np

from .compare_residual import FixedResidual, comparison_report, paired_counts, validate_selection
from . import test_residual_reporting as reporting_tests
from .train_residual import EVALUATION_CASES, validation_score


def comparison_fixture():
    manifest, rows = reporting_tests.ResidualReportTests().learning_records()
    manifest.update(checkpoints=[128, 512], source_sha256={'fixture': 'same-runtime'},
                    evaluation_cases=EVALUATION_CASES, training_perception_cases=EVALUATION_CASES,
                    training_case_probabilities=[1/len(EVALUATION_CASES)]*len(EVALUATION_CASES),
                    versions={'fixture': '1'}, control_policies=['zero', 'static'])
    rows[0]['trials'] = [t for t in rows[0]['trials'] if t['policy'] != 'random']
    evidence = []
    for algorithm in ('ppo', 'sac'):
        m = deepcopy(manifest)
        m['algorithms'] = [algorithm]
        records = deepcopy([rows[0]]+[row for row in rows[1:] if row['algorithm'] == algorithm])
        raw = ''.join(json.dumps(r)+'\n' for r in records).encode()
        training = dict(manifest=m, records=records, trials_sha256=hashlib.sha256(raw).hexdigest())
        test_rows = deepcopy(records)
        for row in test_rows:
            row.pop('validation_curve', None)
            for trial in row['trials']:
                trial['seed'] = 30000
        test = dict(environment_version=m['environment_version'], config=m['config'],
                    source_sha256=m['source_sha256'], training_manifest=m,
                    evaluation_cases=EVALUATION_CASES, evaluation_seeds=[30000],
                    algorithms=[algorithm], records=test_rows)
        evidence.extend([training, test])
    ppo = evidence[0]['manifest']
    controls = [t for t in rows[0]['trials'] if t['policy'] == 'zero']
    fixed = [dict(t, policy='fixed') for t in controls]
    nominal = {case: 1 for case in EVALUATION_CASES}
    nominal['unsafe'] = 0
    score = list(validation_score(fixed, nominal))
    candidates = [dict(id='zero', method='zero', action=[0., 0., 0., 0.],
                       validation_score=score, trials=deepcopy(fixed))]
    for seed in ppo['training_seeds']:
        for method in ('initial', 'contact_mean'):
            candidate = dict(id=f'ppo_seed{seed}_{method}', method=method,
                             action=[0., 0., 0., 0.], validation_score=score.copy(), trials=deepcopy(fixed))
            if method == 'contact_mean':
                candidate.update(action_sums=[0., 0., 0., 0.], eligible_counts=[1, 1, 1, 1])
            candidates.append(candidate)
    selection = dict(training_manifest=ppo, environment_version=ppo['environment_version'], config=ppo['config'],
                     source_sha256=ppo['source_sha256'], evaluation_cases=EVALUATION_CASES,
                     evaluation_seeds=ppo['evaluation_seeds'], nominal_trials=controls,
                     candidates=candidates, selected_candidate_id='zero')
    fixed_test = dict(environment_version=ppo['environment_version'], config=ppo['config'],
                      source_sha256=ppo['source_sha256'], evaluation_cases=EVALUATION_CASES,
                      evaluation_seeds=[30000], selection_sha256='pinned-selection',
                      selected_candidate_id='zero', action=[0., 0., 0., 0.],
                      trials=[dict(t, seed=30000) for t in fixed])
    return [*evidence, selection, fixed_test]


class ResidualComparisonTests(unittest.TestCase):
    def report(self, evidence):
        return comparison_report(*evidence, selection_sha256='pinned-selection')

    def test_all_algorithms_and_training_seeds_are_reported(self):
        text = self.report(comparison_fixture())
        for algorithm in ('PPO', 'SAC'):
            for seed in (0, 1, 2):
                self.assertIn(f'{algorithm} / seed {seed}', text)
        self.assertIn('固定残差', text)
        self.assertIn('不表示任一算法已收敛', text)

    def test_fixed_action_does_not_depend_on_observation_or_mutate(self):
        model = FixedResidual([.1, -.2, 0., .3])
        first, _ = model.predict(np.zeros(73))
        first[:] = 0.
        np.testing.assert_array_equal(model.predict(np.ones(73))[0], [.1, -.2, 0., .3])

    def test_invalid_constant_is_rejected(self):
        for action in ([0., 0.], [0., 0., 0., 1.1], [0., np.nan, 0., 0.]):
            with self.assertRaises(ValueError):
                FixedResidual(action)

    def test_selection_cannot_use_test_scenes(self):
        selection = comparison_fixture()[-2]
        selection['evaluation_seeds'] = [30000]
        with self.assertRaisesRegex(ValueError, 'validation scenes'):
            validate_selection(selection)

    def test_zero_is_preferred_on_a_complete_tie(self):
        selection = comparison_fixture()[-2]
        self.assertEqual(validate_selection(selection)['id'], 'zero')
        selection['selected_candidate_id'] = 'ppo_seed0_initial'
        with self.assertRaisesRegex(ValueError, 'validation score'):
            validate_selection(selection)

    def test_missing_fixed_candidate_or_modified_fit_is_rejected(self):
        selection = comparison_fixture()[-2]
        missing = deepcopy(selection)
        missing['candidates'].pop()
        with self.assertRaisesRegex(ValueError, 'candidate coverage'):
            validate_selection(missing)
        selection['candidates'][-1]['action_sums'][0] = .5
        with self.assertRaisesRegex(ValueError, 'validation fit'):
            validate_selection(selection)

    def test_changed_test_action_or_selection_digest_is_rejected(self):
        for field, value in (('action', [.1, 0., 0., 0.]), ('selection_sha256', 'different')):
            evidence = comparison_fixture()
            evidence[-1][field] = value
            with self.assertRaisesRegex(ValueError, 'validation-selected'):
                self.report(evidence)

    def test_selection_digest_accepts_git_newline_conversion(self):
        evidence = comparison_fixture()
        raw = (json.dumps(evidence[-2], indent=2, allow_nan=False)+'\n').encode('utf-8')
        lf = hashlib.sha256(raw).hexdigest()
        crlf = hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest()
        for actual, recorded in ((lf, crlf), (crlf, lf)):
            evidence[-1]['selection_sha256'] = recorded
            self.assertIn('固定残差', comparison_report(*evidence, selection_sha256=actual))

    def test_newline_compatibility_cannot_hide_changed_selection_content(self):
        evidence = comparison_fixture()
        raw = (json.dumps(evidence[-2], indent=2, allow_nan=False)+'\n').encode('utf-8')
        evidence[-1]['selection_sha256'] = hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest()
        evidence[-2]['method'] = 'changed after evaluation'
        changed = (json.dumps(evidence[-2], indent=2, allow_nan=False)+'\n').encode('utf-8')
        with self.assertRaisesRegex(ValueError, 'validation-selected'):
            comparison_report(*evidence, selection_sha256=hashlib.sha256(changed).hexdigest())

    def test_incomplete_fixed_test_is_rejected(self):
        evidence = comparison_fixture()
        evidence[-1]['trials'].pop()
        with self.assertRaisesRegex(ValueError, 'comparison trials'):
            self.report(evidence)

    def test_changed_evaluation_implementation_is_rejected(self):
        for record, field in ((3, 'evaluation_source_sha256'), (5, 'comparison_source_sha256')):
            evidence = comparison_fixture()
            evidence[record][field] = 'different-evaluator'
            with self.assertRaisesRegex(ValueError, 'implementations differ'):
                self.report(evidence)

    def test_different_runtime_or_budget_is_not_a_fair_algorithm_comparison(self):
        for field, value in (('steps', 1024), ('source_sha256', {'fixture': 'other-runtime'})):
            evidence = comparison_fixture()
            evidence[2]['manifest'][field] = value
            with self.assertRaises(ValueError):
                self.report(evidence)

    def test_nominal_regression_cannot_be_hidden_by_repeated_controls(self):
        evidence = comparison_fixture()
        controls = evidence[3]['records'][0]['trials']
        controls[0]['max_ball_height_m'] = 1.1
        with self.assertRaisesRegex(ValueError, 'nominal physical evidence'):
            self.report(evidence)

    def test_paired_counts_and_duplicate_scene_rejection(self):
        reference = [dict(case='baseline', seed=s, is_success=bool(s)) for s in (0, 1)]
        changed = [dict(t, is_success=not t['is_success']) for t in reference]
        self.assertEqual(paired_counts(changed, reference), dict(wins=1, losses=1))
        with self.assertRaises(ValueError):
            paired_counts(changed+[changed[0]], reference)


if __name__ == '__main__':
    unittest.main()

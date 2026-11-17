"""Full-task completion must not imply robustness or feedback advantage."""
from copy import deepcopy
import unittest

from .evaluate_cascade import assess
from .train_cascade import runtime_compatible, selection_score


class CascadeAssessmentTests(unittest.TestCase):
    def values(self):
        baseline = dict(clean=5, sensor_only=2, perturbed=3, unsafe=8)
        names = ['zero', 'timed']+[f'{kind}_seed{seed}' for seed in (0, 1, 2) for kind in ('live', 'frozen')]
        return {name: deepcopy(baseline) for name in names}

    def test_completed_nominal_task_can_still_regress_under_noise(self):
        values = self.values()
        values['live_seed0']['sensor_only'] = 1
        result = assess(values, 5)
        self.assertTrue(result['all_nominal_tasks_completed'])
        self.assertFalse(result['robustness_nondegradation']['0'])
        self.assertFalse(any(result['feedback_advantage'].values()))

    def test_unsafe_regression_is_reported_without_hiding_completed_tasks(self):
        values = self.values()
        values['live_seed1']['unsafe'] += 1
        result = assess(values, 5)
        self.assertTrue(result['nominal_task_gates']['1'])
        self.assertFalse(result['robustness_nondegradation']['1'])
        values['live_seed1']['clean'] -= 1
        self.assertFalse(assess(values, 5)['all_nominal_tasks_completed'])

    def test_controls_cannot_be_omitted_or_a_tie_reported_as_feedback_gain(self):
        values = self.values()
        result = assess(values, 5)
        self.assertTrue(all(result['robustness_nondegradation'].values()))
        self.assertFalse(any(result['feedback_advantage'].values()))
        values.pop('frozen_seed2')
        with self.assertRaises(ValueError):
            assess(values, 5)

    def test_weights_require_identical_environment_and_policy_but_not_trainer_cli(self):
        driver = 'v8_robust_control/train_cascade.py'
        previous = {driver: 'old', 'environment': 'same', 'policy': 'same'}
        current = dict(previous, **{driver: 'new'})
        self.assertTrue(runtime_compatible(previous, current))
        self.assertFalse(runtime_compatible(previous, dict(current, environment='changed')))
        self.assertFalse(runtime_compatible(previous, dict(current, policy='changed')))

    def test_selection_keeps_completed_nominal_tasks_ahead_of_noise_gain(self):
        nominal = dict(clean=3, sensor_only=1, perturbed=0, unsafe=0)
        def trials(clean, sensor):
            return [dict(case=case, is_success=index < successes, reason='SUCCESS' if index < successes else 'BALL_DROPPED')
                    for case, successes in [('clean', clean), ('sensor_only_3mm', sensor)] for index in range(3)]
        manifest = dict(selection_protocol='nominal_task_then_robustness')
        self.assertGreater(selection_score(trials(3, 0), nominal, manifest),
                           selection_score(trials(0, 3), nominal, manifest))
        self.assertGreater(selection_score(trials(3, 2), nominal, manifest),
                           selection_score(trials(3, 1), nominal, manifest))


if __name__ == '__main__':
    unittest.main()

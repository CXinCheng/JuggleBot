"""Reject incomplete pilots and rewards unsupported by physical evidence."""
from copy import deepcopy
from dataclasses import asdict
import unittest

from .train_residual import EVALUATION_CASES
from .residual_env import CatchConfig
from .summarize_residual import summarize, validate, archived_trials, confirmation_report, screening_decisions
from .audit_residual_feedback import FeedbackAblation
import numpy as np
import hashlib
import json


class ResidualReportTests(unittest.TestCase):
    def records(self):
        manifest = dict(algorithms=['sac', 'ppo'], training_seeds=[0, 1, 2],
                        environment_version=2,
                        learning_rates=[.0003], shaping_weights=[.25],
                        evaluation_seeds=[10000], steps=512, config=asdict(CatchConfig()))
        def trials(policies):
            return [dict(policy=p, case=c, seed=10000, is_success=True, reason='SUCCESS',
                         environment_version=2, nominal_feedback=p != 'static',
                         stable_contact_s=.2, max_residual_offset_m=.01,
                         episode_return=9.9, reward_sums={'event': 10., 'shaping': -.1},
                         final_evidence=dict(contact=True, relative_speed_m_s=.01, tangent_m=.01,
                             normal_m=.022, ball_height_m=.88, forbidden_contact=False,
                             pose_violation=False, external_ball_force=False),
                         residual_rms=.1, residual_change_rms=.01, max_ball_height_m=1.)
                    for p in policies for c in EVALUATION_CASES]
        rows = [dict(kind='controls', trials=trials(['zero', 'static', 'random']))]
        for algorithm in manifest['algorithms']:
            for seed in manifest['training_seeds']:
                rows.append(dict(kind='trained', algorithm=algorithm, seed=seed,
                                 learning_rate=.0003, shaping_weight=.25, training_steps=512,
                                 save_load_verified=True, training_wall_time_s=1., steps_per_second=512.,
                                 trials=trials(['learned'])))
        return manifest, rows

    def test_complete_three_seed_report(self):
        manifest, rows = self.records()
        aggregates, text = summarize(manifest, rows)
        self.assertEqual(len(aggregates), 5)
        self.assertEqual(aggregates[3]['cases']['baseline']['episodes'], 3)
        self.assertIn('不能据此断言', text)

    def test_archive_checksum_supports_both_platform_newlines(self):
        _, rows = self.records()
        for newline in ('\n', '\r\n'):
            raw = ''.join(json.dumps(row)+newline for row in rows).encode()
            archive = dict(records=rows, trials_sha256=hashlib.sha256(raw).hexdigest())
            self.assertEqual(archived_trials(archive)[1], raw)
            changed = deepcopy(archive)
            changed['records'][0]['trials'][0]['episode_return'] = 999.
            with self.assertRaises(ValueError):
                archived_trials(changed)

    def test_missing_or_duplicate_runs_rejected(self):
        manifest, rows = self.records()
        for changed in (rows[:-1], rows+[deepcopy(rows[-1])]):
            with self.assertRaises(ValueError):
                validate(manifest, changed)

    def test_training_and_evaluation_seeds_must_be_disjoint(self):
        manifest, rows = self.records()
        manifest['evaluation_seeds'] = [0]
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            validate(manifest, rows)

    def test_incomplete_or_overlapping_evaluation_rejected(self):
        manifest, rows = self.records()
        for mutate in ('delete', 'duplicate', 'training_seed'):
            changed = deepcopy(rows)
            if mutate == 'delete':
                changed[-1]['trials'].pop()
            elif mutate == 'duplicate':
                changed[-1]['trials'].append(deepcopy(changed[-1]['trials'][0]))
            else:
                changed[-1]['trials'][0]['seed'] = 0
            with self.assertRaises(ValueError):
                validate(manifest, changed)


    def test_false_success_and_reward_tampering_rejected(self):
        manifest, rows = self.records()
        for path, value in ((('reason',), 'TASK_DEADLINE'), (('stable_contact_s',), .1),
                            (('episode_return',), 100.), (('max_residual_offset_m',), .03),
                            (('final_evidence', 'contact'), False),
                            (('final_evidence', 'relative_speed_m_s'), 1.),
                            (('final_evidence', 'forbidden_contact'), True)):
            changed = deepcopy(rows)
            target = changed[-1]['trials'][0]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError):
                validate(manifest, changed)


class FeedbackAblationTests(unittest.TestCase):
    class EchoPolicy:
        def predict(self, observation, deterministic=True):
            return observation[:4].copy(), None

    def test_frozen_action_is_constant_until_reset(self):
        policy = FeedbackAblation(self.EchoPolicy(), 'frozen_initial')
        observation = np.zeros(63)
        observation[0] = .1
        first, _ = policy.predict(observation)
        observation[0], observation[58] = .9, .1
        np.testing.assert_array_equal(policy.predict(observation)[0], first)
        observation[58] = 0.
        self.assertEqual(policy.predict(observation)[0][0], .9)

    def test_masked_ball_state_is_not_forwarded_to_policy(self):
        observation = np.ones(63)
        result, _ = FeedbackAblation(self.EchoPolicy(), 'masked_ball').predict(observation)
        np.testing.assert_array_equal(result, np.zeros(4))


class ConfirmationReportTests(unittest.TestCase):
    def receipt(self):
        manifest, rows = ResidualReportTests().records()
        rows[0]['trials'] = [t for t in rows[0]['trials'] if t['policy'] != 'random']
        for row in rows:
            for trial in row['trials']:
                trial['seed'] = 30000
        return dict(environment_version=2, config=manifest['config'],
                    evaluation_cases=EVALUATION_CASES, evaluation_seeds=[30000],
                    algorithms=manifest['algorithms'], training_manifest=manifest, records=rows)

    def test_all_seeds_and_algorithms_are_reported(self):
        text = confirmation_report(self.receipt())
        self.assertIn('| sac | 3/3', text)
        self.assertIn('| ppo | 3/3', text)

    def test_reusing_validation_seeds_is_rejected(self):
        receipt = self.receipt()
        receipt['evaluation_seeds'] = [10000]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            confirmation_report(receipt)

    def test_missing_policy_and_false_success_are_rejected(self):
        receipt = self.receipt()
        changed = deepcopy(receipt)
        changed['records'].pop()
        with self.assertRaises(ValueError):
            confirmation_report(changed)
        receipt['records'][-1]['trials'][0]['final_evidence']['contact'] = False
        with self.assertRaises(ValueError):
            confirmation_report(receipt)

    def test_screening_requires_gain_and_no_baseline_regression(self):
        receipt = self.receipt()
        self.assertFalse(screening_decisions(receipt)['sac']['candidate_gate'])
        def fail(trial):
            trial.update(is_success=False, reason='BALL_DROPPED', stable_contact_s=0.,
                         episode_return=-10.1, reward_sums=dict(event=-10., shaping=-.1))
            trial['final_evidence'].update(contact=False, ball_height_m=.39)
        for trial in receipt['records'][0]['trials']:
            if trial['policy'] == 'zero' and trial['case'] != 'baseline':
                fail(trial)
        self.assertTrue(screening_decisions(receipt)['sac']['candidate_gate'])
        fail(receipt['records'][1]['trials'][0])
        self.assertFalse(screening_decisions(receipt)['sac']['candidate_gate'])

    def test_failed_episode_cannot_receive_success_reward(self):
        receipt = self.receipt()
        trial = receipt['records'][-1]['trials'][0]
        trial.update(is_success=False, reason='TASK_DEADLINE')
        with self.assertRaisesRegex(ValueError, 'Terminal reward'):
            confirmation_report(receipt)


if __name__ == '__main__':
    unittest.main()

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import mujoco

from .audit_task_learning import (PROFILES, LOCAL_INDICES, OpeningActor, check_splits, features,
                                  fit_candidates, matrices, outcome, score, select_model, load_training_data)
from .task_env import TaskConfig, observation
from .task_env import TaskResidualEnv, result_reason
from .task_control import TaskSpaceCascade
from .train_task import TaskCurriculum
from jugglebot.robustness.perception import PerceptionConfig
from .test_task_env import request


class DirectPrefix(TaskSpaceCascade):
    """Observe the recovery criterion without the Gym worker or action decoder."""
    def plan_throw(self, side):
        hand = self.hands[side]
        if (hand.get('candidate_flight') == 1 and 1 in self.caught_events
                and hand.get('capture_confirmed', False) and hand.get('held') is not None
                and np.linalg.norm(self.data.site_xpos[hand['seat']]-hand['p']) <= .03
                and np.linalg.norm(self.spatial(hand)[3:]) <= .30):
            raise RuntimeError('OBSERVED_RECOVERY')
        return super().plan_throw(side)


class TaskLearningTests(unittest.TestCase):
    def test_zero_baseline_matches_direct_physics_at_nominal_and_perturbed_states(self):
        for seed, jitter in ((0, 0.), (410301, .0005)):
            with self.subTest(seed=seed, jitter=jitter):
                rng = np.random.default_rng(seed)
                shifts = rng.uniform(-jitter, jitter, (2, 2))[[0, 1, 0]]
                direct = DirectPrefix(throws=30,
                    perception=PerceptionConfig(seed=int(rng.integers(0, 2**31))))
                for (q, _), shift in zip(direct.balls, shifts):
                    direct.data.qpos[q:q+2] += shift
                mujoco.mj_forward(direct.model, direct.data)
                direct.ball_observer.reset()
                direct.ball_observer.update(direct.data)
                result = direct.run_complete(timeout=12.)
                env = TaskCurriculum(TaskResidualEnv(TaskConfig()),
                                     [dict(catch_goal=1, seat_jitter_m=jitter)])
                try:
                    obs, _ = env.reset(seed=seed)
                    while True:
                        obs, _, done, truncated, info = env.step(np.zeros(6))
                        if done or truncated:
                            break
                    expected = ('TRAINING_PREFIX' if result['error'] == 'OBSERVED_RECOVERY'
                                else result_reason(result))
                    self.assertEqual(info['reason'], expected)
                    self.assertEqual(info['catches'], result['catches'])
                    self.assertEqual(info['releases'], result['throws'])
                    self.assertEqual(info['elapsed_s'], result['time'])
                    np.testing.assert_array_equal(env.unwrapped.controller.data.qpos, direct.data.qpos)
                    np.testing.assert_array_equal(env.unwrapped.controller.data.qvel, direct.data.qvel)
                finally:
                    env.close()

    def test_selector_can_learn_a_state_dependent_safe_correction(self):
        x = np.zeros((20, 306))
        x[:, 0] = np.tile([-1., 1.], 10)
        y = np.zeros((20, len(PROFILES), 2))
        y[x[:, 0] < 0, 1, 0] = 1.
        y[x[:, 0] > 0, 2, 0] = 1.
        model, _ = select_model(x[:12], y[:12], x[12:16], y[12:16])
        self.assertEqual(score(model, x[16:], y[16:])['recovered'], 4)
        # An action succeeding only in the opposite state cannot be selected
        # from the seed, since no seed enters either fitting or prediction.
        self.assertEqual(score(model, x[16:], y[16:])['choices'],
                         ['forward', 'upward', 'forward', 'upward'])

    def test_features_and_actor_use_only_detached_current_measurements(self):
        obs = observation(request(), TaskConfig())
        self.assertEqual(features(obs).shape, (306,))
        actor = OpeningActor(dict(kind='constant', index=1))
        np.testing.assert_array_equal(actor.predict(obs)[0], PROFILES['forward'])
        obs['context'][5] = 1./30
        np.testing.assert_array_equal(actor.predict(obs)[0], np.zeros(6))
        obs['context'][5] = 0.
        obs['context'][0] = 0.
        np.testing.assert_array_equal(actor.predict(obs)[0], np.zeros(6))

    def test_fit_does_not_mutate_training_or_validation_data(self):
        rng = np.random.default_rng(5)
        x, y = rng.normal(size=(8, 306)), rng.integers(0, 2, (8, 5, 2)).astype(float)
        original_x, original_y = x.copy(), y.copy()
        self.assertEqual(len(fit_candidates(x, y)), 6)
        np.testing.assert_array_equal(x, original_x)
        np.testing.assert_array_equal(y, original_y)

    def test_serialized_selector_preserves_choices_and_nominal_mask(self):
        rng = np.random.default_rng(5)
        x, y = rng.normal(size=(8, 306)), rng.integers(0, 2, (8, 5, 2)).astype(float)
        for model in fit_candidates(x, y):
            restored = json.loads(json.dumps(model))
            self.assertEqual(score(model, x, y), score(restored, x, y))
        current = request()
        obs = observation(current, TaskConfig(), asdict(current))
        actor = OpeningActor(dict(kind='constant', index=1))
        np.testing.assert_array_equal(actor.predict(obs)[0], np.zeros(6))

    def test_local_model_ignores_unrelated_joints_and_keeps_physical_measurements(self):
        self.assertEqual(len(LOCAL_INDICES), 40)
        current = request()
        obs = observation(current, TaskConfig(), asdict(current))
        x = np.tile(features(obs), (12, 1))
        x[:, LOCAL_INDICES[0]] = np.tile([-1., 1.], 6)
        y = np.zeros((12, 5, 2))
        y[x[:, LOCAL_INDICES[0]] < 0, 1, 0] = 1.
        y[x[:, LOCAL_INDICES[0]] > 0, 2, 0] = 1.
        for model in fit_candidates(x, y, 'local'):
            before = score(model, x, y)
            changed = x.copy()
            changed[:, 60] += 10.  # An arm joint, outside the local feature view.
            self.assertEqual(before, score(model, changed, y))
            self.assertGreaterEqual(before['recovered'], 6)

    def test_reused_old_test_seeds_cannot_remain_independent(self):
        with self.assertRaises(ValueError):
            check_splits([[10, 11, 12, 20], [21], [12]])
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'protocol.json').write_text(json.dumps(dict(source_sha256={})))
            (folder/'report.json').write_text(json.dumps(dict(protocol={})))
            with self.assertRaises(ValueError):
                load_training_data(folder, {}, 'different-reference', .5)

    def test_recovery_needs_real_flight_id_and_cannot_be_complete_success(self):
        row = dict(reason='TRAINING_PREFIX', training_prefix=True, recovered_ids=[1], is_success=False)
        self.assertEqual(outcome(row), [1., 0.])
        with self.assertRaises(ValueError):
            outcome(dict(row, recovered_ids=[]))
        with self.assertRaises(ValueError):
            outcome(dict(row, is_success=True))
        self.assertEqual(outcome(dict(row, reason='POSE_LIMIT', training_prefix=False)), [0., 1.])

    def test_seed_groups_are_disjoint_and_counterfactual_states_match(self):
        check_splits([[0, 1], [2], [3]])
        for groups in ([[0], [0], [1]], [[0, 0], [1], [2]], [[], [1], [2]]):
            with self.assertRaises(ValueError):
                check_splits(groups)
        rows = [dict(seed=0, profile=name, features=[0.]*306, reason='DROPPED',
                     training_prefix=False, recovered_ids=[], is_success=False) for name in PROFILES]
        self.assertEqual(matrices(rows)[1].shape, (1, 5, 2))
        rows[-1]['features'][0] = 1.
        with self.assertRaises(ValueError):
            matrices(rows)


if __name__ == '__main__':
    unittest.main()

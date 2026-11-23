"""Task qualification, feedback attribution and evaluation archive integrity."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from collections import deque
import unittest

import numpy as np

from .cascade_env import CascadeEnv, prepare_reference
from .train_transfer import Fixed
from .train_cascade import (CASES, Frozen, PhaseTimed, counts, evaluate, qualifies,
                            score, summarize, validate_trials)


class CascadeStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.reference = Path(cls.temporary.name)/'reference'
        prepare_reference(cls.reference, 'two')
        env = CascadeEnv(cls.reference)
        config = asdict(env.config)
        env.close()
        trials, fit = evaluate(Fixed(), cls.reference, [380000], fit=True)
        point = dict(step=256, checkpoint_sha256='fixture', trials=deepcopy(trials),
                     score=list(score(trials, counts(trials))))
        cls.fixture = dict(manifest=dict(task='two', requested_throws=2, ball_count=2,
                task_config=config, training_seeds=[0, 1, 2], validation_seeds=[380000],
                checkpoints=[256], observation_size=191, action_size=6), baseline=trials,
                models=[dict(seed=seed, training_steps=256, selected_step=256,
                    checkpoint_sha256='fixture', save_load_verified=True,
                    trials=deepcopy(trials), frozen_trials=deepcopy(trials), timed_trials=deepcopy(trials),
                    curve=[deepcopy(point)], timed_fit=deepcopy(fit)) for seed in (0, 1, 2)])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_task_qualification_does_not_claim_feedback_gain_from_a_tie(self):
        summary = summarize(self.fixture)
        self.assertTrue(summary['all_candidates_passed'])
        self.assertFalse(any(summary['feedback_advantage'].values()))

    def test_task_qualification_rejects_safety_and_completion_regressions(self):
        nominal = dict(clean=3, sensor_only=3, perturbed=5, unsafe=2)
        self.assertTrue(qualifies(nominal, nominal))
        for field, value in (('clean', 2), ('sensor_only', 2), ('perturbed', 4), ('unsafe', 3)):
            self.assertFalse(qualifies(dict(nominal, **{field: value}), nominal))

    def test_archive_rejects_missing_scenes_changed_physics_and_fitted_controls(self):
        for change in ('scene', 'case', 'fit', 'budget', 'seed', 'catch'):
            with self.subTest(change=change):
                altered = deepcopy(self.fixture)
                if change == 'scene':
                    altered['baseline'].pop()
                elif change == 'case':
                    altered['models'][0]['trials'][0]['config']['noise_std_m'] = .02
                elif change == 'fit':
                    altered['models'][0]['timed_fit']['sums'][0][0] = 1.
                elif change == 'budget':
                    altered['models'][0]['training_steps'] = 512
                elif change == 'seed':
                    altered['models'][1]['seed'] = 0
                else:
                    event = next(event for event in altered['models'][0]['trials'][0]['events'] if event['kind'] == 'catch')
                    event['supporting_digits'] = ['if']
                with self.assertRaises(ValueError):
                    summarize(altered)

    def test_frozen_control_waits_for_all_active_balls_then_stays_causal(self):
        class Model:
            juggling_ball_count = 2
            def predict(self, observation, deterministic=True):
                return observation[:6].copy(), None
        actor = Frozen(Model())
        observation = np.zeros(191)
        observation[:6], observation[12] = .2, 1.
        np.testing.assert_array_equal(actor.predict(observation)[0], 0.)
        observation[26] = 1.
        np.testing.assert_array_equal(actor.predict(observation)[0], .2)
        observation[:6] = .9
        np.testing.assert_array_equal(actor.predict(observation)[0], .2)
        actor.reset()
        np.testing.assert_array_equal(actor.predict(observation)[0], .9)

    def test_time_sequence_uses_physical_time_across_task_horizons(self):
        for horizon in (2., 12.):
            actions = np.arange((int(np.ceil(horizon/.02))+1)*6, dtype=float).reshape(-1, 6)
            actor = PhaseTimed(actions, horizon)
            observation = np.zeros(191)
            observation[168] = .3/CascadeEnv.TIME_SCALE_S
            np.testing.assert_array_equal(actor.predict(observation)[0], actions[15])

    def test_success_requires_paired_release_ids_and_supported_finish(self):
        for change in ('release', 'catch', 'hold'):
            trials = deepcopy(self.fixture['baseline'])
            trial = trials[0]
            self.assertTrue(trial['is_success'])
            if change == 'release':
                next(event for event in trial['events'] if event['kind'] == 'release')['vertical_speed_m_s'] = .5
            elif change == 'catch':
                next(event for event in trial['events'] if event['kind'] == 'catch')['id'] = 3
            else:
                trial['final_hold_s'] = .01
            with self.assertRaises(ValueError):
                validate_trials(trials, self.fixture['manifest'], [380000])


class EndingEstimationTests(unittest.TestCase):
    def fixture(self):
        gravity = np.array([0., 0., -9.81])
        dt, start = .001, 1.
        position, velocity = np.array([.3, .2, 1.]), np.array([.1, -.5, 2.])
        frames = [SimpleNamespace(sample_time_s=.99, position_m=[99., 99., 99.])]
        for elapsed in (.01, .027, .043, .06):
            measured = position+elapsed*velocity+.5*(elapsed**2+dt*elapsed)*gravity
            frames.append(SimpleNamespace(sample_time_s=start+elapsed, position_m=measured,
                                          velocity_m_s=[999., 999., 999.]))
        tracker = SimpleNamespace(history=deque(frames), estimate=lambda now: SimpleNamespace(valid=True))
        env = SimpleNamespace(trackers=[tracker], release_times=[start], dt=dt,
                              data=SimpleNamespace(time=1.067), model=SimpleNamespace(opt=SimpleNamespace(gravity=gravity)))
        return env, position, velocity, gravity

    def test_ending_fit_excludes_support_history_and_measurement_velocity(self):
        env, position, velocity, gravity = self.fixture()
        result = CascadeEnv._ending_ball_state(env, 0)
        elapsed = env.data.time-env.release_times[0]
        np.testing.assert_allclose(result[0], position+elapsed*velocity+.5*(elapsed**2+env.dt*elapsed)*gravity, atol=1e-13)
        np.testing.assert_allclose(result[1], velocity+elapsed*gravity, atol=1e-13)

    def test_ending_waits_for_three_delivered_flight_frames(self):
        env, *_ = self.fixture()
        env.trackers[0].history = deque(list(env.trackers[0].history)[:3])
        self.assertIsNone(CascadeEnv._ending_ball_state(env, 0))

    def test_ending_rejects_invalid_estimates_and_unreleased_payload(self):
        env, *_ = self.fixture()
        env.release_times[0] = None
        self.assertIsNone(CascadeEnv._ending_ball_state(env, 0))
        env.release_times[0] = 1.
        env.trackers[0].estimate = lambda now: None
        self.assertIsNone(CascadeEnv._ending_ball_state(env, 0))


if __name__ == '__main__':
    unittest.main()

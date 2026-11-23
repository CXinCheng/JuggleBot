from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO, SAC

from jugglebot.robustness.observation import BallEstimate
from .task_control import TaskRequest, decode_action
from .task_env import TaskConfig, TaskResidualEnv, TaskPolicy, observation, action_authority, context_key
from .train_task import (make_model, screening, validate_trial, evaluate, evaluate_prefix, TaskCurriculum,
                         measurement_potential, REWARD_DISCOUNT, POTENTIAL_BOUND, TrainingRewardScale)


def request():
    state = BallEstimate(0, .3, .29, .29, (.33, -.14, .9), (0., 0., 0.), (0., 0., 0.), True)
    return TaskRequest('throw', 'right', 0, .3, 0, 0, .035, (0., 0., 3., 0., 0.),
        (state, None, None), ((0.,)*7,)*2, ((0.,)*7,)*2, ((0.,)*16,)*2, ((0.,)*16,)*2,
        ((.33, -.14, .82), (.33, .14, .82)), ((0.,)*3,)*2, ((0.,)*5,)*2, ('hold', 'hold'))


class TaskObservationTests(unittest.TestCase):
    def test_action_audit_rejects_mismatched_initial_states_and_distinguishes_no_authority(self):
        from copy import deepcopy
        from .train_task import audit_pairs
        zero=dict(seed=1,profile='zero',initial_ball_states=[{'position_m':[0.,0.,1.]}],
            decision_diagnostics=[],catches=0,recovered_ids=[],reason='TIMEOUT')
        probe=dict(deepcopy(zero),profile='forward_first_throw_small',reason='POSE_LIMIT',catches=1,
            decision_diagnostics=[dict(time_s=.3,executed_action=[0.]*6)])
        pair=audit_pairs([zero,probe])[0]
        self.assertEqual(pair['executed_nonzero_decisions'],0)
        self.assertIsNone(pair['first_changed_decision_s'])
        self.assertTrue(pair['safety_regression'])
        probe['decision_diagnostics'][0]['executed_action'][0]=.1
        self.assertEqual(audit_pairs([zero,probe])[0]['first_changed_decision_s'],.3)
        probe['initial_ball_states'][0]['position_m'][0]=.001
        with self.assertRaisesRegex(ValueError,'identical physical initial states'):
            audit_pairs([zero,probe])

    def test_action_audit_truth_is_read_only_and_cannot_supply_missing_policy_estimates(self):
        from dataclasses import replace
        from .train_task import audit_decision
        env=TaskResidualEnv(TaskConfig(horizon_s=.4),estimator='filtered')
        try:
            obs,_=env.reset(seed=418000)
            env._request=replace(env._request,balls=(None,None,None))
            before=[env.controller.data.qpos.copy(),env.controller.data.qvel.copy()]
            observation_before={key:value.copy() for key,value in obs.items()}
            row=audit_decision(env,obs,np.zeros(6))
            self.assertTrue(all(not state['usable'] for state in row['estimates']))
            self.assertTrue(all(state['measurement'] is None for state in row['estimates']))
            self.assertEqual(len(row['physics_diagnostic']['balls']),3)
            for expected,actual in zip(before,[env.controller.data.qpos,env.controller.data.qvel]):
                np.testing.assert_array_equal(actual,expected)
            for key,value in observation_before.items():
                np.testing.assert_array_equal(obs[key],value)
        finally:
            env.close()

    def test_dropping_early_does_not_receive_a_smaller_failure_penalty(self):
        self.assertEqual(TaskResidualEnv.terminal_reward('DROPPED'), -20.)
        self.assertEqual(TaskResidualEnv.terminal_reward('TIMEOUT'), -20.)
        self.assertEqual(TaskResidualEnv.terminal_reward('POSE_LIMIT'), -20.)
        self.assertEqual(TaskResidualEnv.terminal_reward('FORBIDDEN_CONTACT'), -20.)
        self.assertGreater(TaskResidualEnv.terminal_reward('SUCCESS'), 0.)

    def test_authority_uses_measured_innovation_and_agrees_with_the_policy_mask(self):
        current = request()
        nominal = asdict(current)
        obs = observation(current, TaskConfig(), nominal)
        self.assertEqual(action_authority(obs), 0.)
        tensors = {name: torch.as_tensor(value[None]) for name, value in obs.items()}
        self.assertEqual(TaskPolicy.active(tensors).item(), 0.)
        obs['errors'][0, 0] = .5
        self.assertEqual(action_authority(obs), 1.)
        tensors = {name: torch.as_tensor(value[None]) for name, value in obs.items()}
        self.assertEqual(TaskPolicy.active(tensors).item(), 1.)
        obs['errors'][0, 0] = 1.4
        self.assertEqual(action_authority(obs), 1.)
        self.assertEqual(action_authority(observation(current, TaskConfig())), 1.)

    def test_other_arm_and_finger_motion_do_not_open_action_authority(self):
        current = request()
        obs = observation(current, TaskConfig(), asdict(current))
        obs['robot_error'][7:14] = 10.
        obs['robot_error'][44:60] = 10.
        obs['robot_error'][76:92] = 10.
        self.assertEqual(action_authority(obs), 0.)
        tensors = {name: torch.as_tensor(value[None]) for name, value in obs.items()}
        self.assertEqual(TaskPolicy.active(tensors).item(), 0.)
        obs['robot_error'][0] = .25
        self.assertEqual(action_authority(obs), 1.)
        tensors = {name: torch.as_tensor(value[None]) for name, value in obs.items()}
        self.assertEqual(TaskPolicy.active(tensors).item(), 1.)

    def test_other_ball_errors_allow_coordination_and_local_fingers_allow_release_correction(self):
        current = request()
        for key, index in (('errors', (1, 0)), ('robot_error', 28)):
            obs = observation(current, TaskConfig(), asdict(current))
            obs[key][index] = .01
            self.assertEqual(action_authority(obs), 1.)
            tensors = {name: torch.as_tensor(value[None]) for name, value in obs.items()}
            self.assertEqual(TaskPolicy.active(tensors).item(), 1.)

    def test_observations_use_sampled_context_and_invalidate_missing_estimates(self):
        current = request()
        nominal = asdict(current)
        nominal['balls'][0]['position_m'] = (.329, -.14, .9)
        obs = observation(current, TaskConfig(), nominal)
        self.assertAlmostEqual(float(obs['errors'][0, 0]), .2, places=5)
        self.assertEqual(obs['errors'][0, 6], 1.)
        np.testing.assert_array_equal(obs['balls'][1, :10], 0.)
        self.assertEqual(obs['balls'][1, 10], 10.)
        np.testing.assert_array_equal(obs['errors'][1], 0.)
        self.assertEqual(obs['robot'].shape, (104,))
        self.assertEqual(obs['context'].shape, (26,))
        self.assertTrue(all(np.all(np.isfinite(value)) for value in obs.values()))

    def test_partial_progress_cannot_pass_complete_task_screening(self):
        def row(success, catches=0, reason='TIMEOUT', case='seat_2mm', seed=0):
            return dict(is_success=success, catches=catches, reason=reason, case=case, seed=seed)
        baseline = [row(True, 30, 'SUCCESS', 'clean'), row(False)]
        partial = [row(True, 30, 'SUCCESS', 'clean'), row(False, 29)]
        self.assertFalse(screening(partial, {'zero': baseline})['passed'])
        baseline.append(row(False, case='combined'))
        unsafe_gain = [row(True, 30, 'SUCCESS', 'clean'), row(True, 30, 'SUCCESS'), row(False, reason='POSE_LIMIT', case='combined')]
        self.assertFalse(screening(unsafe_gain, {'zero': baseline})['passed'])
        with self.assertRaises(ValueError):
            screening(unsafe_gain, {'zero': baseline[:-1]})
        with self.assertRaises(ValueError):
            screening(unsafe_gain, {})


class TaskEnvironmentTests(unittest.TestCase):
    def test_small_observed_error_does_not_remove_a_physically_successful_opening(self):
        base = TaskResidualEnv(TaskConfig(horizon_s=1.3))
        env = TaskCurriculum(base, [dict(catch_goal=1, seat_jitter_m=.0005)])
        try:
            env.reset(seed=410202)
            current = base._request
            nominal = asdict(current)
            nominal['balls'][2]['position_m'] = (np.asarray(current.balls[2].position_m)-[.0003, 0., 0.]).tolist()
            base.reference[context_key(current)] = nominal
            base._observation = observation(current, base.config, nominal)
            action = np.array([.1/.75, 0., 0., 0., 0., 0.])
            while True:
                _, _, done, truncated, info = env.step(action)
                action = np.zeros(6)
                if done or truncated:
                    break
            first = base.controller.decisions[0]
            self.assertAlmostEqual(first['parameters'][0]-first['nominal_parameters'][0], .1)
            self.assertTrue(info['training_prefix'], info['reason'])
            self.assertIn(1, info['recovered_ids'])
            self.assertFalse(info['is_success'])
        finally:
            env.close()

    def test_sac_replay_training_and_save_load_share_the_bounded_task_interface(self):
        torch.set_num_threads(1)
        env = TaskResidualEnv(TaskConfig(horizon_s=.301))
        try:
            obs, _ = env.reset(seed=0)
            model = make_model(env, 0, 'sac')
            ppo = make_model(env, 0)
            np.testing.assert_allclose(model.actor.log_std.detach().numpy(),
                                       ppo.policy.log_std.detach().numpy())
            bounds = np.exp(model.actor.log_std.detach().numpy()[0])*8
            self.assertTrue(np.all((bounds > 0.) & (bounds < 1.)))
            self.assertGreater(bounds[0], np.max(bounds[1:]))
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], 0.)
            self.assertTrue(model.use_sde_at_warmup)
            model.learning_starts = 0
            model.batch_size = 4
            model.learn(8)
            self.assertGreater(model._n_updates, 0)
            with TemporaryDirectory() as directory:
                path = Path(directory)/'sac.zip'
                model.save(path)
                loaded = SAC.load(path)
                np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                              loaded.predict(obs, deterministic=True)[0])
                self.assertTrue(env.action_space.contains(loaded.predict(obs, deterministic=False)[0]))
        finally:
            env.close()

    def test_invalid_action_cannot_advance_the_paused_physics(self):
        env = TaskResidualEnv(TaskConfig(horizon_s=.7))
        try:
            obs, _ = env.reset(seed=0)
            self.assertTrue(env.observation_space.contains(obs))
            before = env.controller.data.qpos.copy()
            clock = env.controller.data.time
            for action in ([0.]*5, [float('nan')]*6, [1.1]*6):
                with self.assertRaises(ValueError):
                    env.step(action)
                self.assertEqual(env.controller.data.time, clock)
                np.testing.assert_array_equal(env.controller.data.qpos, before)
            env.step(np.zeros(6))
            self.assertGreater(env.controller.data.time, clock)
        finally:
            env.close()

    def test_reset_reproduces_transport_and_does_not_share_episode_data(self):
        env = TaskResidualEnv(TaskConfig(seat_jitter_m=.002, noise_std_m=.003, delay_s=.02, dropout=.1))
        try:
            first, _ = env.reset(seed=71)
            previous, worker = env.controller, env._thread
            clock = previous.data.time
            second, _ = env.reset(seed=71)
            self.assertFalse(worker.is_alive())
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
            self.assertIs(previous.model, env.controller.model)
            self.assertIsNot(previous.data, env.controller.data)
            self.assertIsNot(previous.hands['left']['kin'].data, env.controller.hands['left']['kin'].data)
            env.step(np.zeros(6))
            self.assertEqual(previous.data.time, clock)
        finally:
            env.close()

    def test_decision_budget_is_a_truncation_without_a_success_bonus(self):
        env = TaskResidualEnv(TaskConfig(max_decisions=1))
        try:
            env.reset(seed=0)
            obs, reward, terminated, truncated, info = env.step(np.zeros(6))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info['reason'], 'BUDGET')
            self.assertFalse(info['is_success'])
            self.assertEqual(info['reward_sums']['event'], 0.)
            self.assertTrue(np.any(obs['context'][:2]))
            validate_trial(info)
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(6))
        finally:
            env.close()

    def test_task_deadline_is_a_failure_even_before_the_first_decision(self):
        env = TaskResidualEnv(TaskConfig(horizon_s=.01))
        try:
            obs, info = env.reset(seed=0)
            self.assertTrue(info['initial_terminal'])
            np.testing.assert_array_equal(obs['context'], 0.)
            _, reward, terminated, truncated, info = env.step(np.zeros(6))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info['reason'], 'TIMEOUT')
            self.assertEqual(reward, -20.)
            validate_trial(info)
        finally:
            env.close()

    def test_policy_actions_change_the_executed_throw_parameters_and_motion(self):
        env = TaskResidualEnv(TaskConfig(horizon_s=.7))
        try:
            env.reset(seed=0)
            baseline, *_ = env.step(np.zeros(6))
            env.reset(seed=0)
            changed, *_ = env.step(np.array([.1, 0., 0., 0., 0., 0.]))
            decision = env.controller.decisions[0]
            self.assertAlmostEqual(decision['parameters'][0]-decision['nominal_parameters'][0], .075)
            self.assertGreater(np.linalg.norm(changed['balls']-baseline['balls']), 1e-5)
            self.assertFalse(env.controller.conditions_summary()['task_decisions']['online_physics_rollouts'])
        finally:
            env.close()

    def test_ppo_starts_at_zero_and_save_load_preserves_actions(self):
        torch.set_num_threads(1)
        env = TaskResidualEnv(TaskConfig(horizon_s=.31))
        try:
            obs, _ = env.reset(seed=0)
            model = make_model(env, 0)
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], 0.)
            for _ in range(16):
                action = model.predict(obs, deterministic=False)[0]
                self.assertTrue(env.action_space.contains(action))
                self.assertLessEqual(decode_action(env._request, action).defer_s, .025)
            empty = env._empty_observation()
            tensor, _ = model.policy.obs_to_tensor(empty)
            model.policy.zero_grad()
            _, probability, _ = model.policy.evaluate_actions(tensor, torch.zeros((1, 6)))
            probability.sum().backward()
            np.testing.assert_array_equal(model.policy.action_net.weight.grad.numpy(), 0.)
            with TemporaryDirectory() as directory:
                path = Path(directory)/'model.zip'
                model.save(path)
                loaded = PPO.load(path)
                np.testing.assert_array_equal(loaded.predict(obs, deterministic=True)[0], model.predict(obs, deterministic=True)[0])
                loaded.task_action_layout = 'feedback_gain_actions'
                with self.assertRaisesRegex(ValueError, 'semantics'):
                    evaluate(loaded, Path(directory)/'absent.json', [0], ['clean'])
        finally:
            env.close()

    def test_zero_residual_completes_the_actual_task_and_reward_ledger(self):
        env = TaskResidualEnv()
        total = 0.
        try:
            env.reset(seed=0)
            while True:
                _, reward, terminated, truncated, info = env.step(np.zeros(6))
                total += reward
                if terminated or truncated:
                    break
            self.assertTrue(info['is_success'], info['reason'])
            self.assertEqual((info['releases'], info['catches']), (30, 30))
            self.assertGreaterEqual(info['final_hold_s'], .8-1e-10)
            self.assertEqual(total, 40.)
            self.assertEqual(info['reward_sums'], dict(progress=30., action=0., rate=0., event=10.))
            self.assertTrue(set(range(1, 28)).issubset(info['recovered_ids']))
            self.assertNotIn(0, info['recovered_ids'])
            validate_trial(info)
        finally:
            env.close()


class CurriculumFixture(gym.Env):
    """Supply event evidence without weakening the physical environment."""
    observation_space = gym.spaces.Box(-1., 1., (1,))
    action_space = gym.spaces.Box(-1., 1., (6,))

    def __init__(self):
        self.config = TaskConfig()
        self.done = self.truncated = self.closed = False
        self.ids = []

    def reset(self, *, seed=None, options=None):
        self.closed = False
        return np.zeros(1), {}

    def step(self, action):
        return np.zeros(1), 1., self.done, self.truncated, dict(
            recovered_ids=self.ids, completed_ids=[1, 2, 3], is_success=False,
            episode_return=1., reward_sums=dict(progress=1.), reason='RUNNING')

    def close(self):
        self.closed = True


class TaskEvaluationTests(unittest.TestCase):
    @staticmethod
    def fixture(config=None, reference=None):
        env = CurriculumFixture()
        env.config = config or TaskConfig()
        def obs():
            current = request()
            value = observation(current, env.config, asdict(current))
            if env.config.seat_jitter_m:
                value['errors'][0, 0] = .5
            return value
        env.reset = lambda **kwargs: (obs(), {})
        env.step = lambda action: (obs(), -20., True, False, dict(
            environment_version=TaskResidualEnv.ENVIRONMENT_VERSION, reason='TIMEOUT', is_success=False,
            requested_throws=30, releases=0, catches=0, recovered_ids=[], decisions=1,
            reward_sums=dict(progress=0., action=0., rate=0., event=-20.), episode_return=-20.))
        return env

    class Recorder:
        def __init__(self):
            self.contexts = []

        def predict(self, obs, deterministic=True):
            self.contexts.append(obs['context'].copy())
            return np.full(6, .25), None

    def test_prefix_and_training_use_the_same_time_feature(self):
        actor = self.Recorder()
        with patch('v8_robust_control.train_task.TaskResidualEnv', side_effect=self.fixture):
            evaluate_prefix(actor, None, [71])
        self.assertAlmostEqual(float(actor.contexts[0][4]), request().time_s/TaskConfig().horizon_s)

    def test_fixed_controls_are_fitted_only_to_active_decisions(self):
        actor = self.Recorder()
        # Preserve the constants used by outcome validation while replacing
        # the simulator with two contrasting measurement states.
        with patch('v8_robust_control.train_task.TaskResidualEnv', wraps=TaskResidualEnv,
                   side_effect=self.fixture) as constructor:
            for key in ('ENVIRONMENT_VERSION', 'OBSERVATION_LAYOUT', 'ACTION_LAYOUT', 'ACTION_COST', 'RATE_COST'):
                setattr(constructor, key, getattr(TaskResidualEnv, key))
            constructor.terminal_reward = TaskResidualEnv.terminal_reward
            _, samples = evaluate(actor, None, [71], ['clean', 'seat_2mm'], collect=True)
        self.assertEqual(len(samples), 1)
        self.assertEqual(action_authority(samples[0][0]), 1.)
        np.testing.assert_array_equal(samples[0][1], .25)


class TaskCurriculumTests(unittest.TestCase):
    def test_waiting_cannot_discount_a_failure_penalty(self):
        penalty = TaskResidualEnv.terminal_reward('POSE_LIMIT')
        for waits in (0, 1, 7, 30, 100):
            delayed = sum(REWARD_DISCOUNT**i*r for i,r in enumerate([0.]*waits+[penalty]))
            self.assertEqual(delayed, penalty)

    def test_training_scale_keeps_raw_event_evidence_and_failure_ordering(self):
        base = CurriculumFixture()
        env = TrainingRewardScale(base, .01)
        env.reset()
        _, reward, _, _, info = env.step(np.zeros(6))
        self.assertEqual(reward, .01)
        self.assertEqual(info['episode_return'], 1.)
        self.assertEqual(info['reward_sums'], dict(progress=1.))
        self.assertLess(env.reward(-20.), env.reward(3.))
        for invalid in (0., -1., 2., float('nan')):
            with self.assertRaises(ValueError):
                TrainingRewardScale(base, invalid)

    def test_potential_reward_telescopes_and_cannot_pay_for_early_abort(self):
        current = request()
        initial = observation(current, TaskConfig(), asdict(current))
        initial['errors'][0, 0] = 1.
        phi0 = measurement_potential(initial)
        self.assertLess(phi0, 0.)
        for deviations in ([0.], [8., .5, 4.], [1.]*20):
            phi, total = phi0, 0.
            for step, deviation in enumerate(deviations):
                obs = {k:v.copy() for k,v in initial.items()}
                obs['errors'][0, 0] = deviation
                next_phi = 0. if step == len(deviations)-1 else measurement_potential(obs)
                total += REWARD_DISCOUNT**step*(REWARD_DISCOUNT*next_phi-phi)
                phi = next_phi
            self.assertAlmostEqual(total, -phi0)
        initial['balls'][0, 9] = 0.
        self.assertEqual(measurement_potential(initial), -POTENTIAL_BOUND)

    def test_shaped_prefix_closes_its_potential_and_preserves_the_reward_ledger(self):
        base = CurriculumFixture()
        current = request()
        obs = observation(current, TaskConfig(), asdict(current))
        obs['errors'][0, 0] = 2.
        base.reset = lambda **kwargs: (obs, {})
        base.step = lambda action: (obs, 1., False, False, dict(recovered_ids=[1],
            is_success=False, episode_return=1., reward_sums=dict(progress=1.), reason='RUNNING'))
        env = TaskCurriculum(base, shaping=True)
        env.reset()
        _, reward, done, truncated, info = env.step(np.zeros(6))
        self.assertTrue(done and not truncated)
        self.assertFalse(info['is_success'])
        self.assertAlmostEqual(reward, 3.-measurement_potential(obs))
        self.assertAlmostEqual(sum(info['reward_sums'].values()), info['episode_return'])

    def test_catch_alone_and_gapped_recovery_cannot_finish_a_prefix(self):
        base = CurriculumFixture()
        env = TaskCurriculum(base, [dict(catch_goal=3, seat_jitter_m=.002)])
        env.reset()
        for ids in ([], [1, 2], [1, 2, 4]):
            base.ids = ids
            _, _, done, truncated, info = env.step(np.zeros(6))
            self.assertFalse(done or truncated)
            self.assertFalse(info['training_prefix'])
        base.ids = [1, 2, 3]
        _, reward, done, truncated, info = env.step(np.zeros(6))
        self.assertTrue(done and base.closed)
        self.assertFalse(truncated or info['is_success'])
        self.assertEqual(info['reason'], 'TRAINING_PREFIX')
        self.assertEqual(reward, 3.)
        self.assertEqual(info['episode_return'], 3.)
        self.assertEqual(info['reward_sums'], dict(progress=1., curriculum=2.))

    def test_physical_failure_or_budget_takes_precedence_over_prefix(self):
        for terminated, truncated in ((True, False), (False, True)):
            base = CurriculumFixture()
            base.ids, base.done, base.truncated = [1], terminated, truncated
            env = TaskCurriculum(base)
            env.reset()
            _, reward, done, trunc, info = env.step(np.zeros(6))
            self.assertEqual((done, trunc), (terminated, truncated))
            self.assertEqual(reward, 1.)
            self.assertFalse(info['training_prefix'])
            self.assertEqual(list(env.outcomes), [False])

    def test_final_stage_waits_for_original_full_task_termination(self):
        base = CurriculumFixture()
        base.ids = list(range(1, 31))
        env = TaskCurriculum(base, [dict(catch_goal=30, seat_jitter_m=.002)])
        env.reset()
        _, reward, done, trunc, info = env.step(np.zeros(6))
        self.assertFalse(done or trunc or base.closed)
        self.assertEqual(reward, 1.)
        self.assertFalse(info['is_success'] or info['training_prefix'])

    def test_advancement_requires_a_full_window_of_real_successes(self):
        base = CurriculumFixture()
        env = TaskCurriculum(base)
        env.steps = 100000
        env.reset()
        self.assertEqual(env.stage, 0)
        env.outcomes.extend([True]*23+[False]*9)
        env.reset()
        self.assertEqual(env.stage, 0)
        env.outcomes.clear()
        env.outcomes.extend([True]*24+[False]*8)
        env.reset()
        self.assertEqual(env.stage, 1)
        self.assertEqual(base.config.seat_jitter_m, .001)
        self.assertEqual(len(env.outcomes), 0)


if __name__ == '__main__':
    unittest.main()

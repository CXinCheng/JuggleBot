"""Causal observation/estimation and controller integration checks."""
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from jugglebot.robustness.observation import (
    BallEstimate, BallObservation, BallObserver, MuJoCoBallSource, PositionEstimator,
)
from jugglebot.robustness.controller import ObservedCascade as ContactCascade
from jugglebot.humanoid.cascade import ContactCascade as BaselineCascade


def packet(t, position, ball_id=0, received=None, valid=True):
    return BallObservation(ball_id, t, t if received is None else received, position, valid)


class TestPositionEstimator(unittest.TestCase):
    def test_linear_velocity_and_prediction_use_sample_not_receipt_time(self):
        estimator = PositionEstimator(0, max_age_s=.2)
        estimator.update(packet(0., [1., 2., 3.], received=.02))
        self.assertFalse(estimator.estimate(.02).valid)
        estimator.update(packet(.04, [1.08, 1.96, 3.02], received=.09))
        state = estimator.estimate(.1)
        np.testing.assert_allclose(state.velocity_m_s, [2., -1., .5])
        np.testing.assert_allclose(state.position_m, [1.2, 1.9, 3.05])
        self.assertEqual(state.sample_time_s, .04)
        self.assertEqual(state.received_time_s, .09)
        self.assertAlmostEqual(state.age_s, .06)
        self.assertTrue(state.valid)

    def test_irregular_samples_recover_ballistic_velocity_at_control_time(self):
        estimator = PositionEstimator(0, max_age_s=.2)
        p0, v0, gravity = np.array([.2, .1, 1.]), np.array([.3, -.7, 2.]), np.array([0., 0., -9.81])
        for t in (0., .017, .033):
            estimator.update(packet(t, p0 + t*v0 + .5*t*t*gravity))
        state = estimator.estimate(.04)
        np.testing.assert_allclose(state.position_m, p0+.04*v0+.5*.04**2*gravity, atol=1e-12)
        np.testing.assert_allclose(state.velocity_m_s, v0+.04*gravity, atol=1e-11)
        np.testing.assert_allclose(state.acceleration_m_s2, gravity, atol=1e-10)

    def test_duplicate_reordered_and_future_packets(self):
        estimator = PositionEstimator(0)
        estimator.update(packet(.02, [0., 0., 1.], received=.03))
        before = estimator.estimate(.03)
        self.assertFalse(estimator.update(packet(.02, [99., 0., 1.], received=.04)))
        self.assertFalse(estimator.update(packet(.01, [99., 0., 1.], received=.04)))
        self.assertFalse(estimator.update(packet(.025, [99., 0., 1.], received=.025)))
        self.assertEqual(before, estimator.estimate(.03))
        with self.assertRaises(ValueError):
            estimator.estimate(.025)
        with self.assertRaises(ValueError):
            estimator.update(packet(.04, [0., 0., 1.], ball_id=1))

    def test_staleness_invalid_measurement_gap_and_reset(self):
        estimator = PositionEstimator(0, max_age_s=.05)
        self.assertIsNone(estimator.estimate(0.))
        for t in (0., .01):
            estimator.update(packet(t, [t, 0., 1.]))
        self.assertTrue(estimator.estimate(.02).valid)
        self.assertFalse(estimator.estimate(.061).valid)
        estimator.update(packet(.02, [0., 0., 0.], valid=False))
        self.assertFalse(estimator.estimate(.02).valid)
        estimator.update(packet(.03, [.03, 0., 1.]))
        self.assertIsNone(estimator.estimate(.03).velocity_m_s)
        estimator.update(packet(.04, [.04, 0., 1.]))
        self.assertTrue(estimator.estimate(.04).valid)
        estimator.update(packet(.2, [.2, 0., 1.]))
        self.assertFalse(estimator.estimate(.2).valid)
        estimator.reset()
        self.assertIsNone(estimator.estimate(0.))

    def test_observations_are_validated_detached_and_immutable(self):
        source = np.array([1., 2., 3.])
        observation = packet(0., source)
        source[:] = 99.
        self.assertEqual(observation.position_m, (1., 2., 3.))
        with self.assertRaises(FrozenInstanceError):
            observation.valid = False
        for t, received, position in ((1., .9, [0., 0., 0.]), (float('nan'), 1., [0., 0., 0.]),
                                      (0., 0., [float('nan'), 0., 0.]), (0., 0., [0., 0.])):
            with self.assertRaises(ValueError):
                packet(t, position, received=received)


class TestBallObserver(unittest.TestCase):
    def test_sampled_mode_never_reads_truth_velocity(self):
        class PositionOnlyData:
            time = 0.
            qpos = np.array([0., 0., 1.])

            @property
            def qvel(self):
                raise AssertionError('sampled observer read truth velocity')

            @property
            def qacc(self):
                raise AssertionError('sampled observer read truth acceleration')

        data = PositionOnlyData()
        observer = BallObserver([(0, 0)], mode='sampled', sample_period_s=.01, position_step_s=.001)
        observer.update(data)
        self.assertFalse(observer.estimate(0, 0.).valid)
        data.time, data.qpos = .01, np.array([.02, 0., 1.])
        observer.update(data)
        np.testing.assert_allclose(observer.estimate(0, .01).velocity_m_s, [2., 0., 0.])
        data.time, data.qpos = .02, np.array([.04, 0., 1.])
        observer.update(data)
        np.testing.assert_allclose(observer.estimate(0, .025).velocity_m_s, [2., 0., 0.])
        self.assertIsNone(observer.observations[0].velocity_m_s)

    def test_sampled_free_flight_aligns_mujoco_position_and_velocity(self):
        for integrator in ('Euler', 'implicit', 'implicitfast', 'RK4'):
            for timestep, gravity in ((.001, '0 0 -9.81'), (.002, '.4 -.3 -4.5')):
                with self.subTest(integrator=integrator, timestep=timestep, gravity=gravity):
                    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
                      <option timestep="{timestep}" integrator="{integrator}" gravity="{gravity}"/>
                      <worldbody><body pos="0 0 3"><freejoint/><geom size=".04" mass=".12"/></body></worldbody>
                    </mujoco>''')
                    data = mujoco.MjData(model)
                    data.qvel[:3] = [.2, -.6, 1.2]
                    observer = BallObserver([(0, 0)], mode='sampled',
                                            position_step_s=0. if integrator == 'RK4' else model.opt.timestep)
                    derivative_observer = BallObserver([(0, 0)], mode='sampled')
                    for _ in range(501):
                        observer.update(data)
                        derivative_observer.update(data)
                        state = observer.estimate(0, float(data.time))
                        if data.time > .05:
                            self.assertTrue(state.valid)
                            np.testing.assert_allclose(state.position_m, data.qpos[:3], atol=1e-10)
                            np.testing.assert_allclose(state.velocity_m_s, data.qvel[:3], atol=1e-9)
                            derivative = derivative_observer.estimate(0, float(data.time))
                            expected_bias = np.zeros(3) if integrator == 'RK4' else .5*model.opt.timestep*model.opt.gravity
                            np.testing.assert_allclose(np.asarray(derivative.velocity_m_s)-data.qvel[:3],
                                                       expected_bias, atol=1e-9)
                            raw_fit = observer.estimators[0].estimate(float(data.time))
                            np.testing.assert_array_equal(state.position_m, raw_fit.position_m)
                        mujoco.mj_step(model, data)

    def test_physics_tick_intervals_avoid_large_epoch_clock_drift(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001" gravity="0 0 -9.81"/>
          <worldbody><body pos="0 0 3"><freejoint/><geom size=".04" mass=".12"/></body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        data.time = 1e6
        data.qvel[:3] = [.2, -.6, 1.2]
        observer = BallObserver([(0, 0)], mode='sampled', position_step_s=model.opt.timestep)
        recorded_intervals = PositionEstimator(0)
        tick_error = recorded_error = 0.
        for step in range(501):
            if observer.update(data):
                packet = observer.observations[0]
                self.assertEqual(packet.sample_time_s, float(data.time))
                recorded_intervals.update(packet)
            if step > 50:
                state = observer.estimate(0, float(data.time))
                recorded = recorded_intervals.estimate(float(data.time))
                recorded_velocity = (np.asarray(recorded.velocity_m_s)
                                     - .5*model.opt.timestep*np.asarray(recorded.acceleration_m_s2))
                tick_error = max(tick_error, float(np.max(np.abs(np.asarray(state.velocity_m_s)-data.qvel[:3]))))
                recorded_error = max(recorded_error, float(np.max(np.abs(recorded_velocity-data.qvel[:3]))))
            mujoco.mj_step(model, data)
        self.assertLess(tick_error, 1e-10)
        self.assertGreater(recorded_error, 1e-8)
        for quantum in (-.001, float('nan')):
            with self.assertRaises(ValueError):
                PositionEstimator(0, time_quantum_s=quantum)

    def test_sampling_quantization_duplicates_and_no_backfill(self):
        observer = BallObserver([(0, 0)], mode='sampled')
        data = SimpleNamespace(time=0., qpos=np.array([0., 0., 1.]))
        stamps = []
        for tick in range(51):
            data.time = tick*.001
            data.qpos[0] = data.time
            if observer.update(data):
                stamps.append(data.time)
            self.assertFalse(observer.update(data))
        np.testing.assert_allclose(stamps, [0., .017, .034, .05])
        state = observer.estimate(0, .05)
        np.testing.assert_allclose(state.velocity_m_s, [1., 0., 0.], atol=1e-12)
        data.time = .5
        observer.update(data)
        self.assertEqual(observer.observations[0].sample_time_s, .5)
        self.assertFalse(observer.estimate(0, .5).valid)

    def test_reset_clears_each_ball_and_requires_explicit_clock_reset(self):
        observer = BallObserver([(0, 0), (3, 3)], mode='sampled', sample_period_s=.01)
        data = SimpleNamespace(time=1., qpos=np.array([0., 0., 1., 4., 0., 1.]))
        observer.update(data)
        data.time = 1.01
        data.qpos[[0, 3]] += [.01, -.02]
        observer.update(data)
        np.testing.assert_allclose(observer.estimate(0, 1.01).velocity_m_s, [1., 0., 0.])
        np.testing.assert_allclose(observer.estimate(1, 1.01).velocity_m_s, [-2., 0., 0.])
        data.time = 0.
        with self.assertRaises(ValueError):
            observer.update(data)
        observer.reset()
        observer.update(data)
        for ball in (0, 1):
            self.assertFalse(observer.estimate(ball, 0.).valid)

    def test_truth_source_is_passive_and_exact(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".05"/></body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        data.qpos[:3], data.qvel[:3] = [.1, -.2, 1.], [.4, .5, -.6]
        before = [x.copy() for x in (data.qpos, data.qvel, data.ctrl, data.qfrc_applied)]
        observer = BallObserver([(0, 0)])
        observer.update(data)
        state = observer.estimate(0, 0.)
        np.testing.assert_array_equal(state.position_m, data.qpos[:3])
        np.testing.assert_array_equal(state.velocity_m_s, data.qvel[:3])
        self.assertEqual(state, MuJoCoBallSource([(0, 0)]).truth(data, 0))
        for actual, expected in zip((data.qpos, data.qvel, data.ctrl, data.qfrc_applied), before):
            np.testing.assert_array_equal(actual, expected)

    def test_configuration_validation(self):
        for kwargs in (dict(mode='invalid'), dict(sample_period_s=0), dict(sample_period_s=float('nan')),
                       dict(max_age_s=.001), dict(max_age_s=float('inf')),
                       dict(position_step_s=-.001), dict(position_step_s=float('nan'))):
            with self.assertRaises(ValueError):
                BallObserver([(0, 0)], **kwargs)


class TestControllerObservation(unittest.TestCase):
    def test_throw_history_contains_only_executed_same_hand_motion_guesses(self):
        controller = ContactCascade()
        events = []
        for index in range(10):
            events.extend([
                dict(kind='plan', side='right', time=float(index), ball=index%3,
                     command=[float(index), .7, 4., .1, .01]),
                dict(kind='stroke', side='right', time=float(index), ball=index%3)])
        # A rejected plan at the same instant precedes the executed plan.
        events.extend([
            dict(kind='plan', side='right', time=10., ball=0, command=[999., 0., 0., 0.]),
            dict(kind='plan', side='right', time=10., ball=0, command=[9., .7, 4., .1]),
            dict(kind='stroke', side='right', time=10., ball=0),
            dict(kind='plan', side='left', time=11., ball=1, command=[998., 0., 0., 0.]),
            dict(kind='stroke', side='left', time=11., ball=1),
            dict(kind='plan', side='right', time=12., ball=2, command=[997., 0., 0., 0.])])
        controller.events = events
        before = controller.data.qpos.copy(), controller.data.qvel.copy()
        guesses = controller.additional_throw_guesses('right')
        self.assertEqual(len(guesses), 8)
        self.assertEqual([value[0] for value in guesses], list(range(9, 1, -1)))
        self.assertTrue(all(value.shape == (4,) for value in guesses))
        guesses[0][0] = -100.
        self.assertEqual(events[-5]['command'][0], 9.)
        np.testing.assert_array_equal(controller.data.qpos, before[0])
        np.testing.assert_array_equal(controller.data.qvel, before[1])
        controller.events = [events[-1]]
        self.assertEqual(controller.additional_throw_guesses('right'), ())

    def test_reception_precision_is_common_bounded_and_does_not_change_physics(self):
        controller = ContactCascade(observation_mode='sampled', normalize_reception_input=True)
        estimate = BallEstimate(2, .1, .1, .1, (.312345678901, -.212345678901, 1.),
                                (.123456789012, -.234567890123, -1.345678901234),
                                (0., 0., -9.81), True)
        before = controller.data.qpos.copy(), controller.data.qvel.copy()
        expected = tuple(np.round(vector, 8) for vector in (estimate.position_m, estimate.velocity_m_s))
        with patch.object(controller, 'observed_ball', return_value=estimate):
            for mode, silent in [('sampled', False), ('truth', False), ('sampled', True)]:
                controller.ball_observer.mode, controller.silent = mode, silent
                state = controller.reception_ball_state(2)
                for result, reference, raw in zip(state, expected, (estimate.position_m, estimate.velocity_m_s)):
                    np.testing.assert_array_equal(result, reference)
                    self.assertLessEqual(float(np.max(np.abs(result-raw))), .5e-8+1e-15)
            controller.normalize_reception_input = False
            state = controller.reception_ball_state(2)
            np.testing.assert_array_equal(state[0], estimate.position_m)
            np.testing.assert_array_equal(state[1], estimate.velocity_m_s)
        np.testing.assert_array_equal(controller.data.qpos, before[0])
        np.testing.assert_array_equal(controller.data.qvel, before[1])

    def test_normalized_truth_and_sampled_have_identical_physical_motion(self):
        truth = ContactCascade(observation_mode='truth', normalize_reception_input=True)
        sampled = ContactCascade(observation_mode='sampled', normalize_reception_input=True)
        truth.planning = sampled.planning = False
        # Two live receptions, with no optimizer search or oracle injection.
        for _ in range(700):
            truth.step()
            sampled.step()
            np.testing.assert_array_equal(truth.data.qpos, sampled.data.qpos)
            np.testing.assert_array_equal(truth.data.qvel, sampled.data.qvel)
            np.testing.assert_array_equal(truth.data.ctrl, sampled.data.ctrl)
        self.assertEqual(truth.events, sampled.events)

    def test_sampled_reception_waits_for_acceleration_model(self):
        controller = ContactCascade(observation_mode='sampled')
        linear = BallEstimate(2, .017, .017, .017, (.3, -.2, 1.),
                              (0., 0., -1.), None, True)
        complete = BallEstimate(2, .034, .034, .034, (.3, -.2, 1.),
                                (0., 0., -1.), (0., 0., -9.81), True)
        with patch.object(controller, 'observed_ball', return_value=linear):
            self.assertIsNone(controller.reception_ball_state(2))
            controller.silent = True
            state = controller.reception_ball_state(2)
            np.testing.assert_array_equal(state[1], linear.velocity_m_s)
        controller.silent = False
        with patch.object(controller, 'observed_ball', return_value=complete):
            state = controller.reception_ball_state(2)
            np.testing.assert_array_equal(state[0], complete.position_m)
            np.testing.assert_array_equal(state[1], complete.velocity_m_s)
        self.assertTrue(controller.conditions_summary()['ball_observation']['reception_requires_acceleration'])
        controller.ball_observer.mode = 'truth'
        with patch.object(controller, 'observed_ball', return_value=linear):
            self.assertIsNotNone(controller.reception_ball_state(2))
        self.assertFalse(controller.conditions_summary()['ball_observation']['reception_requires_acceleration'])

    def test_forecast_retry_preserves_explicit_inputs_and_restores_commands(self):
        controller = ContactCascade()
        controller.silent = True
        controller._use_execution_forecasts = True
        commands = controller.commands
        original = {side: value.copy() for side, value in commands.items()}
        controller.warm = {'throw:right:0': [2., 2., 2., 2.]}
        controller.execution_warm = {'throw:right:0': [3., 3., 3., 3.],
                                     'throw:left:0': [4., 4., 4., 4.]}

        def forecast(_):
            np.testing.assert_array_equal(controller.warm.get('throw:right:0', controller.commands['right']), [2.]*4)
            np.testing.assert_array_equal(controller.warm.get('throw:left:0', controller.commands['left']), [4.]*4)
            raise ValueError('stop private rollout')

        with patch.object(BaselineCascade, 'step', forecast):
            with self.assertRaisesRegex(ValueError, 'stop private rollout'):
                controller.step()
        self.assertIs(controller.commands, commands)
        for side in commands:
            np.testing.assert_array_equal(commands[side], original[side])

    def test_reception_retry_recertifies_without_changing_live_ball_state(self):
        controller = ContactCascade()
        controller.hands['right']['incoming'] = 2
        before = controller.data.qpos.copy(), controller.data.qvel.copy()
        modes = []

        def plan(_, side):
            modes.append(controller._use_execution_forecasts)
            if not controller._use_execution_forecasts:
                controller.failure_event = dict(reason='NO_FEASIBLE_RECEPTION')
                raise RuntimeError('no feasible reception')
            self.assertIsNone(controller.failure_event)

        with patch.object(BaselineCascade, 'plan_reception', plan):
            controller.plan_reception('right')
        self.assertEqual(modes, [False, True])
        self.assertFalse(controller._use_execution_forecasts)
        self.assertEqual(controller.forecast_retry_count, 1)
        np.testing.assert_array_equal(controller.data.qpos, before[0])
        np.testing.assert_array_equal(controller.data.qvel, before[1])

    def test_reception_retry_is_bounded_and_does_not_hide_failure(self):
        controller = ContactCascade()
        controller.hands['right']['incoming'] = 2
        calls = []

        def fail(_, side):
            calls.append(controller._use_execution_forecasts)
            controller.failure_event = dict(reason='NO_FEASIBLE_RECEPTION')
            raise RuntimeError('still infeasible')

        with patch.object(BaselineCascade, 'plan_reception', fail):
            with self.assertRaisesRegex(RuntimeError, 'still infeasible'):
                controller.plan_reception('right')
        self.assertEqual(controller.forecast_retry_count, 1)
        self.assertEqual(calls, [False, True])
        self.assertEqual(controller.failure_event['reason'], 'NO_FEASIBLE_RECEPTION')
        self.assertFalse(controller._use_execution_forecasts)

    def test_pose_failure_is_not_retried_or_cleared(self):
        controller = ContactCascade()

        def fail(_, side):
            controller.failure_event = dict(reason='POSE_LIMIT')
            raise RuntimeError('pose failure')

        with patch.object(BaselineCascade, 'plan_reception', fail):
            with self.assertRaisesRegex(RuntimeError, 'pose failure'):
                controller.plan_reception('right')
        self.assertEqual(controller.forecast_retry_count, 0)
        self.assertEqual(controller.failure_event['reason'], 'POSE_LIMIT')

    def test_v6_stays_independent_and_v7_truth_matches_moving_baseline(self):
        baseline = BaselineCascade()
        observed = ContactCascade(normalize_reception_input=False)
        self.assertFalse(hasattr(baseline, 'ball_observer'))
        self.assertEqual(baseline.additional_throw_guesses('right'), ())
        self.assertNotIn('ball_observation', baseline.conditions_summary())
        baseline.planning = observed.planning = False
        # Exercise real arm motion, release and reception without slow searches.
        for _ in range(400):
            baseline.step()
            observed.step()
            np.testing.assert_array_equal(baseline.data.qpos, observed.data.qpos)
            np.testing.assert_array_equal(baseline.data.qvel, observed.data.qvel)
            np.testing.assert_array_equal(baseline.data.ctrl, observed.data.ctrl)
        self.assertEqual(baseline.events, observed.events)

    def test_truth_and_sampled_input_and_rollout_isolation(self):
        controller = ContactCascade(observation_mode='sampled')
        controller.planning = False
        observer = controller.ball_observer
        self.assertFalse(controller.observed_ball(2).valid)
        before = observer.observations
        original = controller.data
        trial = mujoco.MjData(controller.model)
        mujoco.mj_copyData(trial, controller.model, original)
        trial.time = 1.
        trial.qpos[controller.balls[2][0]] += .5
        controller.data, controller.silent = trial, True
        controller.step()
        trial_estimate = controller.observed_ball(2)
        self.assertTrue(trial_estimate.valid)
        np.testing.assert_array_equal(trial_estimate.position_m, trial.qpos[controller.balls[2][0]:controller.balls[2][0]+3])
        self.assertIs(observer.observations, before)
        controller.data, controller.silent = original, False
        self.assertFalse(controller.observed_ball(2).valid)
        self.assertEqual(controller.conditions_summary()['ball_observation']['mode'], 'sampled')

    def test_reception_dispatches_through_observer_and_skips_invalid_input(self):
        invalid = BallEstimate(2, 0., 0., 0., (0., 0., 1.), None, None, False)
        for estimate in (None, invalid):
            with self.subTest(estimate=estimate):
                controller = ContactCascade()
                controller.planning = False
                controller.hands['right'].update(mode='wait', held=None)
                for hand in controller.hands.values():
                    hand['throw_time'] = 10.
                with patch.object(controller, 'observed_ball', return_value=estimate) as observe:
                    controller.step()
                observe.assert_called_once_with(2)
                self.assertIsNone(controller.hands['right']['reception'])
                self.assertEqual(controller.conditions_summary()['ball_observation']['mode'], 'truth')

    def test_reception_uses_supplied_estimate_instead_of_truth(self):
        from jugglebot.humanoid.cascade import SUPPORTED_BALL_CENTER_OFFSET_M
        controller = ContactCascade()
        controller.planning = False
        hand = controller.hands['right']
        hand.update(mode='wait', held=None)
        for h in controller.hands.values():
            h['throw_time'] = 10.
        impact_delay = .2
        # Deliberately differ from the actual airborne ball's state.
        position = (.32, -.10, hand['catch_p'][2] + SUPPORTED_BALL_CENTER_OFFSET_M
                    + .5*controller.gravity_m_s2*impact_delay**2)
        estimate = BallEstimate(2, 0., 0., 0., position, (0., 0., 0.), None, True)
        with patch.object(controller, 'observed_ball', return_value=estimate):
            controller.step()
        prepare = next(event for event in controller.events if event['kind'] == 'prepare')
        self.assertEqual(prepare['point'][:2], [.32, -.10])
        self.assertEqual(prepare['impact'], impact_delay)


if __name__ == '__main__':
    unittest.main()

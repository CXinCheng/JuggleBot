from dataclasses import FrozenInstanceError, replace
from contextlib import redirect_stderr
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from jugglebot.robustness.observation import BallEstimate
from .task_control import (OracleDecisions, RecordedReceptionRejected, TaskCommand, TaskRequest,
                           TaskSpaceCascade, decode_action, validate_command)


def request(kind='throw', remaining=.035):
    return TaskRequest(kind, 'right', 2, .3, 0, 0, remaining,
        (0., 0., 3., 0., 0.) if kind == 'throw' else (0.,)*5,
        (), (), (), (), (), (), (), (), ())


class TaskDecisionContractTests(unittest.TestCase):
    def test_uncaught_braking_is_torque_limited_and_scoped_to_a_missed_catch_tail(self):
        from types import SimpleNamespace
        c = TaskSpaceCascade.__new__(TaskSpaceCascade)
        c.uncaught_braking,c.silent = True,False
        c.uncaught_braking_steps,c.uncaught_braking_first = 0,None
        c.ball_geom_ids,c.hand_geoms = [3],{'left':{4}}
        c._receive_clearance_geoms = {'left':(4,)}
        c._stand_separation = lambda *a,**k:(.02,np.array([-1.,0.,0.,0.,0.,0.,0.]),None,None,None)
        stop = SimpleNamespace(t0=1.)
        hand = dict(mode='receive',held=None,incoming=0,reception=(None,stop),
            kin=SimpleNamespace(side='left',dof_adr=np.arange(7)),force_limits=np.arange(1.,8.))
        d = SimpleNamespace(time=.99,qvel=np.ones(7),contact=[],qpos=np.zeros(7),
                            qfrc_applied=np.zeros(7),xfrc_applied=np.zeros((1,6)))
        torque,response,free = np.zeros(7),np.eye(7),np.zeros(7)
        def brake():
            return c._brake_uncaught_reception(hand,d,torque,response,free)
        np.testing.assert_array_equal(brake(),torque)
        d.time=1.01
        arrays=[d.qpos.copy(),d.qvel.copy(),d.qfrc_applied.copy(),d.xfrc_applied.copy()]
        np.testing.assert_array_equal(brake(),-hand['force_limits'])
        for before,after in zip(arrays,[d.qpos,d.qvel,d.qfrc_applied,d.xfrc_applied]):
            np.testing.assert_array_equal(before,after)
        self.assertEqual(c.uncaught_braking_steps,1)
        # A latched brake does not chatter as velocity approaches zero.
        c._stand_separation = lambda *a,**k:None
        self.assertTrue(np.any(brake()))
        d.contact=[SimpleNamespace(geom1=3,geom2=4,dist=0.,efc_address=0)]
        np.testing.assert_array_equal(brake(),torque)
        d.contact=[]
        hand['held']=0
        np.testing.assert_array_equal(brake(),torque)
        hand['held']=None
        hand['reception']=(None,SimpleNamespace(t0=1.005))
        np.testing.assert_array_equal(brake(),torque)
        hand['reception']=(None,stop)
        c.uncaught_braking=False
        np.testing.assert_array_equal(brake(),torque)

    def test_fast_capture_braking_is_continuous_and_preserves_the_recovery_endpoint(self):
        from types import SimpleNamespace
        from jugglebot.humanoid.planning import Quintic
        controller = TaskSpaceCascade.__new__(TaskSpaceCascade)
        controller.model = SimpleNamespace(opt=SimpleNamespace(timestep=.001))
        controller.adaptive_braking = True
        controller.silent = False
        controller.fast_capture_brakes = 0
        start, velocity, goal = np.array([.3, .14, .86]), np.array([.1, .2, -2.]), np.array([.33, .14, .82])
        zero = np.zeros(3)
        stopped = start+.5*.08*velocity
        initial_acceleration = np.array([.3, -.4, -1.])
        original = SimpleNamespace(t0=0., duration=.25, t1=.25, initial_linear_velocity=velocity,
            capture_cushion=(Quintic(0., .08, start, velocity, initial_acceleration, stopped, zero, zero),
                             Quintic(.08, .17, stopped, zero, zero, goal, zero, zero)))
        hand = dict(mode='carry', carry=original)
        adjusted = controller._capture_recovery_reference(hand, SimpleNamespace(time=0.), original)
        self.assertIsNot(adjusted, original)
        self.assertIs(hand['carry'], adjusted)
        self.assertEqual(original.capture_cushion[0].duration, .08)
        first, second = adjusted.capture_cushion
        self.assertEqual(first.duration, .045)
        for actual, expected in zip(first.sample(0.), original.capture_cushion[0].sample(0.)):
            np.testing.assert_allclose(actual, expected, atol=1e-9)
        for actual, expected in zip(first.sample(first.t1), second.sample(second.t0)):
            np.testing.assert_allclose(actual, expected, atol=1e-9)
        for actual, expected in zip(second.sample(.25), original.capture_cushion[1].sample(.25)):
            np.testing.assert_allclose(actual, expected, atol=1e-9)
        self.assertGreater(first.sample(first.t1)[0][2], stopped[2])
        self.assertEqual(controller.fast_capture_brakes, 1)
        self.assertIs(controller._capture_recovery_reference(hand, SimpleNamespace(time=0.), adjusted), adjusted)
        for mode, time_s, speed in (('throw', 0., -2.), ('carry', .02, -2.), ('carry', 0., -1.7)):
            hand['mode'] = mode
            original.initial_linear_velocity = np.array([0., 0., speed])
            self.assertIs(controller._capture_recovery_reference(hand, SimpleNamespace(time=time_s), original), original)
        controller.silent = True
        original.initial_linear_velocity = velocity
        hand['mode'] = 'carry'
        controller._capture_recovery_reference(hand, SimpleNamespace(time=0.), original)
        self.assertEqual(controller.fast_capture_brakes, 1)

    def test_torque_limits_preserve_the_nullspace_correction_direction(self):
        task = np.array([[1., 1., 0.], [0., 0., 1.]])
        torque = np.array([.8, -.2, .1])
        correction = np.array([1., -1., 0.])
        result = TaskSpaceCascade._bounded_torque_change(torque, correction, np.ones(3))
        self.assertTrue(np.all(np.abs(result) <= 1.))
        np.testing.assert_allclose(task@(result-torque), 0., atol=1e-12)
        np.testing.assert_allclose(result, [1., -.4, .1])
        np.testing.assert_array_equal(TaskSpaceCascade._bounded_torque_change(torque, np.zeros(3), np.ones(3)), torque)

    def test_missing_oracle_explains_the_prerequisite_without_initializing_a_simulator(self):
        from .task_control import main
        with TemporaryDirectory() as directory:
            output = Path(directory)/'result.json'
            missing = Path(directory)/'missing.json'
            message = io.StringIO()
            with patch('sys.argv', ['task_control', '--oracle-input', str(missing), '--output', str(output)]), \
                    patch('v8_robust_control.task_control.TaskSpaceCascade') as constructor, redirect_stderr(message):
                with self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn('audit_feedback --native-timeout', message.getvalue())
            constructor.assert_not_called()
            self.assertFalse(output.exists())

    def test_command_detaches_policy_arrays_and_is_json_serializable(self):
        import json
        from dataclasses import asdict
        values = np.array([0., 0., 3., 0., 0.], dtype=np.float32)
        command = TaskCommand(values, defer_s=np.float64(.005))
        values[:] = 123.
        self.assertEqual(command.parameters, (0., 0., 3., 0., 0.))
        json.dumps(asdict(command), allow_nan=False)

    def test_normalized_actions_respect_native_bounds_and_cannot_request_unlimited_waiting(self):
        rng = np.random.default_rng(13)
        for kind in ('throw', 'reception'):
            current = request(kind)
            for action in rng.uniform(-1., 1., (32, 6)):
                command = decode_action(current, action)
                bounds = TaskSpaceCascade.reception_bounds('right') if kind == 'reception' else None
                validate_command(current, command, bounds)
        exhausted = replace(request(), remaining_defer_s=0., attempt=7)
        self.assertEqual(decode_action(exhausted, np.ones(6)).defer_s, 0.)
        tiny = np.r_[np.zeros(5), 1e-9]
        self.assertEqual(decode_action(request(), tiny).defer_s, 0.)
        self.assertEqual(decode_action(request(), np.r_[np.zeros(5), .2]).defer_s, .005)
        self.assertAlmostEqual(decode_action(request(), np.r_[np.zeros(5), .23]).defer_s, .006)
        for action in ([0.]*5, [2.]*6, [float('nan')]*6):
            with self.assertRaises(ValueError):
                decode_action(request(), action)

    def test_invalid_and_out_of_bounds_commands_fail_before_execution(self):
        current = request()
        for command in (TaskCommand((0.,)*4), TaskCommand((float('nan'), 0., 3., 0., 0.)),
                        TaskCommand((.751, 0., 3., 0., 0.)),
                        TaskCommand(current.nominal_parameters, defer_s=.026),
                        TaskCommand(current.nominal_parameters, defer_s=.001),
                        TaskCommand(current.nominal_parameters, defer_s=.0055),
                        TaskCommand(current.nominal_parameters, tilt_scale=1.1)):
            with self.subTest(command=command), self.assertRaises(ValueError):
                validate_command(current, command)
        with self.assertRaises(ValueError):
            validate_command(replace(current, nominal_parameters=(float('nan'),)*5), TaskCommand((0.,)*5))
        with self.assertRaises(ValueError):
            validate_command(request(remaining=.005), TaskCommand(current.nominal_parameters, defer_s=.01))
        with self.assertRaises(TypeError):
            validate_command(current, np.zeros(6))

    def test_reception_keeps_existing_geometric_and_tilt_bounds(self):
        current = request('reception')
        bounds = (-np.ones(5)*.12, np.ones(5)*.12)
        validate_command(current, TaskCommand((0.,)*5, tilt_scale=1.5), bounds)
        for command in (TaskCommand((.061, 0., 0., 0., 0.)),
                        TaskCommand((0., 0., 0., .121, 0.)),
                        TaskCommand((0.,)*5, tilt_scale=1.51), TaskCommand((0.,)*5, defer_s=.005)):
            with self.assertRaises(ValueError):
                validate_command(current, command, bounds)

    def test_oracle_replays_rejected_decisions_and_rejects_missing_or_wrong_ball(self):
        events = [dict(kind='plan', side='right', ball=2, time=.3, score=.03, command=[0., 0., 3., 0.]),
                  dict(kind='plan', side='right', ball=2, time=.305, score=.01, command=[.1, 0., 3., 0.])]
        oracle = OracleDecisions(events)
        self.assertEqual(oracle(request()).defer_s, .005)
        self.assertEqual(oracle(replace(request(), time_s=.305)).parameters, (.1, 0., 3., 0., 0.))
        for current in (replace(request(), time_s=.31), replace(request(), ball_id=0)):
            with self.assertRaises(RuntimeError):
                oracle(current)

    def test_oracle_preserves_reception_rejections_instead_of_executing_failed_trials(self):
        event = dict(kind='return_plan', side='right', ball=2, time=.3, parameters=[0.]*5,
                     diagnostics=dict(captured=True, collision_free=False, ready_for_next_throw=True))
        with self.assertRaises(RecordedReceptionRejected):
            OracleDecisions([event])(request('reception'))
        event['diagnostics']['collision_free'] = True
        self.assertEqual(OracleDecisions([event])(request('reception')).parameters, (0.,)*5)


class TaskContactExecutionTests(unittest.TestCase):
    def test_nominal_event_commands_are_distinct_from_native_optimizer_guesses(self):
        import json
        from .task_control import DEFAULT_EVENT_SEED_PLANS
        from .complete_task import DEFAULT_SEED_PLANS
        executed = json.loads(DEFAULT_EVENT_SEED_PLANS.read_text(encoding='utf-8'))
        guesses = json.loads(DEFAULT_SEED_PLANS.read_text(encoding='utf-8'))
        c = TaskSpaceCascade()
        self.assertEqual(c._nominal_inputs['throw:left:47'],executed['execution']['throw:left:47'])
        self.assertNotEqual(c._nominal_inputs['throw:left:47'],guesses['execution']['throw:left:47'])
        self.assertEqual(executed['forecast'],guesses['forecast'])
        # The exported changes are after throw 30; the short nominal task keeps
        # its established commands. The native controller retains its guesses.
        for key,value in guesses['execution'].items():
            if int(key.rsplit(':',1)[1]) <= 30:
                self.assertEqual(c._nominal_inputs[key],value)

    def test_default_launch_uses_native_contact_readiness_without_extra_world_frame_delay(self):
        from .complete_task import CompleteCascade
        controller = TaskSpaceCascade()
        hand = controller.hands['left']
        hand.update(mode='hold',contact_time=0.)
        hand['p'] = controller.data.site_xpos[hand['seat']] + [.034,0.,0.]
        self.assertFalse(controller.recovery_gate)
        with patch.object(CompleteCascade,'ready_for_throw',return_value=True):
            self.assertTrue(controller.ready_for_throw('left'))
        with patch.object(CompleteCascade,'ready_for_throw',return_value=False):
            self.assertFalse(controller.ready_for_throw('left'))
        # This changes launch scheduling only; physical task acceptance remains
        # the complete controller's original contact, pose and support checks.
        self.assertEqual(controller.ELBOW_OUT_END_M,CompleteCascade.ELBOW_OUT_END_M)
        self.assertEqual(controller.FINAL_SETTLE_S,CompleteCascade.FINAL_SETTLE_S)

    def test_uncaught_braking_prevents_stand_collisions_without_crediting_a_catch(self):
        import mujoco
        def decide(request):
            parameters=list(request.nominal_parameters)
            if request.kind=='throw' and request.released==2:
                parameters[2]-=.2
            return TaskCommand(tuple(parameters))
        for seed,jitter in [(0,0.),(413000,.0005)]:
            for enabled in (False,True):
                with self.subTest(seed=seed,braking=enabled):
                    # Keep the reproduced launch schedule fixed while testing
                    # the optional response to this deliberately missed catch.
                    c=TaskSpaceCascade(decide,uncaught_braking=enabled,recovery_gate=True)
                    shifts=np.random.default_rng(seed).uniform(-jitter,jitter,(2,2))[[0,1,0]]
                    for (q,_),shift in zip(c.balls,shifts):
                        c.data.qpos[q:q+2]+=shift
                    mujoco.mj_forward(c.model,c.data)
                    c.ball_observer.reset()
                    c.ball_observer.update(c.data)
                    result=c.run_complete(timeout=2.)
                    self.assertFalse(result['success'])
                    self.assertEqual(result['catches'],2)
                    if enabled:
                        self.assertGreater(c.uncaught_braking_steps,0)
                        self.assertEqual(result['failure_event']['reason'],'DROPPED')
                        self.assertLessEqual(result['maximum_self_penetration'],.004)
                    else:
                        self.assertEqual(result['failure_event']['quantity'],'self_penetration_m')
                    self.assertFalse(result['grasp_assistance'])
                    self.assertEqual(result['grasp_constraints'],0)

    def test_elbow_clearance_guard_avoids_pose_violation_without_counting_a_miss_as_success(self):
        import mujoco
        for enabled in (False, True):
            controller = TaskSpaceCascade(posture_guard=enabled)
            shifts = np.random.default_rng(410202).uniform(-.0005, .0005, (2, 2))[[0, 1, 0]]
            for (q, _), shift in zip(controller.balls, shifts):
                controller.data.qpos[q:q+2] += shift
            mujoco.mj_forward(controller.model, controller.data)
            controller.ball_observer.reset()
            controller.ball_observer.update(controller.data)
            result = controller.run_complete(timeout=1.24)
            self.assertFalse(result['success'])
            self.assertEqual(result['catches'], 0)
            self.assertFalse(result['grasp_assistance'])
            if enabled:
                self.assertEqual(result['failure_event']['reason'], 'DROPPED')
                self.assertGreater(controller.posture_guard_steps, 0)
                self.assertGreaterEqual(result['minimum_shoulder_above_elbow'], .08)
            else:
                self.assertEqual(result['failure_event']['quantity'], 'shoulder_elbow_clearance_m')

    def test_fast_capture_recovery_avoids_the_reproduced_next_throw_pose_failure(self):
        import mujoco
        # A fixed three-release opening, followed by nominal commands. These
        # stress-test parameters include a rejected native reception candidate;
        # neither controller is being evaluated as a learned policy here.
        opening = {
            ('throw', 'left', 1): (.09123129035600325, -1.1705849001438007, 4.206128920892808, -.22022123095123053, 0.),
            ('reception', 'left', 2): (-.04052217798437438, .03231141385506146, -.039, .025713541474890484, -.011047006366822248),
            ('throw', 'right', 2): (-.4244629333229177, .6657385241179398, 3.4224400869648695, -.2845386287692097, 0.),
            ('reception', 'right', 3): (-.0047515504699095126, -.03938213595589183, .004784713980515666, .019929363838883156, .01856365888886853),
        }
        def decide(current):
            return TaskCommand(opening.get((current.kind, current.side, current.released), current.nominal_parameters))

        for enabled in (False, True):
            # Isolate braking with the settling gate used by this reproduced
            # failure, rather than comparing two different launch schedules.
            controller = TaskSpaceCascade(decide, adaptive_braking=enabled, posture_guard=False,recovery_gate=True)
            original_protect = controller.protect_arm
            peak_torque_ratio = 0.
            def monitored_protect(hand, data, torque, response=None, free_acceleration=None):
                nonlocal peak_torque_ratio
                actual = original_protect(hand, data, torque, response, free_acceleration)
                peak_torque_ratio = max(peak_torque_ratio, float(np.max(np.abs(actual)/hand['force_limits'])))
                return actual
            controller.protect_arm = monitored_protect
            shifts = np.random.default_rng(410202).uniform(-.002, .002, (2, 2))[[0, 1, 0]]
            for (q, _), shift in zip(controller.balls, shifts):
                controller.data.qpos[q:q+2] += shift
            mujoco.mj_forward(controller.model, controller.data)
            controller.ball_observer.reset()
            controller.ball_observer.update(controller.data)
            result = controller.run_complete(timeout=1.8)
            self.assertFalse(result['success'])
            self.assertLessEqual(peak_torque_ratio, 1.+1e-12)
            if enabled:
                self.assertEqual(result['error'], 'TIMEOUT')
                self.assertIsNone(result['failure_event'])
                self.assertGreaterEqual(result['throws'], 5)
                self.assertGreaterEqual(result['catches'], 2)
                self.assertGreater(controller.fast_capture_brakes, 0)
                self.assertEqual(result['ball_contacts'], 0)
                self.assertFalse(result['grasp_assistance'])
            else:
                self.assertEqual(result['failure_event']['reason'], 'POSE_LIMIT')
                self.assertEqual(result['failure_event']['quantity'], 'elbow_out_m')

    def test_expired_recovery_reference_does_not_authorize_an_unrecovered_launch(self):
        from types import SimpleNamespace
        from .complete_task import CompleteCascade
        controller = TaskSpaceCascade(recovery_gate=True)
        hand = controller.hands['left']
        controller.data.time = .3
        hand.update(mode='hold', contact_time=.05, carry=SimpleNamespace(t1=.28))
        palm = controller.data.site_xpos[hand['seat']].copy()
        hand['p'] = palm + [.13, 0., 0.]
        with patch.object(CompleteCascade, 'ready_for_throw', return_value=True), \
                patch.object(controller, 'spatial', return_value=np.r_[np.zeros(3), .45, 0., .43]):
            self.assertFalse(controller.ready_for_throw('left'))
        self.assertEqual(controller.recovery_rejections, 1)
        self.assertAlmostEqual(controller.last_recovery_rejection['position_error_m'], .13)
        self.assertGreater(controller.last_recovery_rejection['speed_m_s'], .6)
        hand['p'] = palm + [.02, 0., 0.]
        with patch.object(CompleteCascade, 'ready_for_throw', return_value=True), \
                patch.object(controller, 'spatial', return_value=np.r_[np.zeros(3), .2, 0., 0.]):
            self.assertTrue(controller.ready_for_throw('left'))
        with patch.object(CompleteCascade, 'ready_for_throw', return_value=False):
            self.assertFalse(controller.ready_for_throw('left'))

    def test_private_recovery_checks_do_not_change_live_diagnostics(self):
        from .complete_task import CompleteCascade
        controller = TaskSpaceCascade(recovery_gate=True)
        hand = controller.hands['left']
        hand['contact_time'] = 0.
        hand['p'] = controller.data.site_xpos[hand['seat']] + [.1, 0., 0.]
        controller.silent = True
        with patch.object(CompleteCascade, 'ready_for_throw', return_value=True):
            self.assertFalse(controller.ready_for_throw('left'))
            self.assertEqual(controller.recovery_rejections, 0)
            self.assertIsNone(controller.last_recovery_rejection)
            controller.recovery_gate = False
            self.assertTrue(controller.ready_for_throw('left'))

    def test_wrist_clearance_prevents_the_reproduced_support_stand_collision(self):
        import mujoco
        # Fixed commands from native seed 410202, including its rejected final
        # reception candidate. Deliberately executing that candidate exercises
        # the physical guard; this is not a policy or complete-task evaluation.
        commands = {
            ('throw', 1): (.09123129035600325, -1.1705849001438007, 4.206128920892808, -.22022123095123053, 0.),
            ('reception', 2): (-.04511438609989413, .02655645060860765, -.03847186517137235, .028290743473229508, -.011489714807236363),
            ('throw', 2): (-.3699323407026835, .632988785096872, 3.3948007066834722, -.2821213751163812, 0.),
            ('reception', 3): (-.010793099791504257, -.03131864658254477, .0056295905771770426, .009088003509874959, -.0588267249167682),
        }

        def decide(current):
            return TaskCommand(commands.get((current.kind, current.released), current.nominal_parameters))

        results = {}
        for enabled in (False, True):
            controller = TaskSpaceCascade(decide, stand_guard=enabled, adaptive_braking=False)
            shifts = np.random.default_rng(410202).uniform(-.002, .002, (2, 2))[[0, 1, 0]]
            for (q, _), shift in zip(controller.balls, shifts):
                controller.data.qpos[q:q+2] += shift
            mujoco.mj_forward(controller.model, controller.data)
            controller.ball_observer.reset()
            controller.ball_observer.update(controller.data)
            result = controller.run_complete(timeout=1.25)
            results[enabled] = result
            if enabled:
                self.assertEqual(result['error'], 'TIMEOUT')
                self.assertEqual(result['catches'], 1)
                self.assertGreater(controller.stand_guard_steps, 0)
                self.assertLess(result['maximum_self_penetration'], .004)
                self.assertFalse(result['success'])
            else:
                self.assertEqual(result['failure_event']['quantity'], 'self_penetration_m')
                self.assertGreater(result['failure_event']['actual'], .004)
        self.assertEqual(results[False]['catches'], 0)

    def test_signed_stand_distance_jacobian_matches_actual_joint_motion(self):
        import mujoco
        controller = TaskSpaceCascade()
        data = controller.data
        hand = controller.hands['left']
        geom = controller._wrist_geoms['left'][0]
        initial = controller._stand_separation(data, geom, max_distance=.5)
        self.assertIsNotNone(initial)
        distance, gradient, normal, _, _ = initial
        self.assertGreater(distance, 0.)
        self.assertAlmostEqual(np.linalg.norm(normal), 1., places=6)
        direction = np.zeros(controller.model.nv)
        direction[hand['kin'].dof_adr] = np.random.default_rng(24).normal(size=7)
        mujoco.mj_integratePos(controller.model, data.qpos, direction, 1e-6)
        mujoco.mj_forward(controller.model, data)
        shifted = controller._stand_separation(data, geom, max_distance=.5)[0]
        self.assertAlmostEqual((shifted-distance)/1e-6, gradient@direction, delta=2e-5)

    def test_callback_inputs_are_detached_and_use_estimates_instead_of_ball_truth(self):
        controller = TaskSpaceCascade()
        estimate = BallEstimate(0, 0., 0., 0., (.5, .6, .7), (.1, .2, .3), (0., 0., -9.81), True)
        qvel = controller.data.qvel.copy()
        for _, v in controller.balls:
            controller.data.qvel[v:v+3] = 12345.
        try:
            with patch.object(controller, 'observed_ball', return_value=estimate):
                current = controller._request('throw', 'right', 2)
            self.assertEqual(current.balls, (estimate,)*3)
            np.testing.assert_array_equal(current.contact_normal_force_n, controller._contact_force)
            self.assertLessEqual(current.contact_time_s, current.time_s)
            self.assertEqual(np.asarray(current.contact_normal_force_n).shape, (2, 5))
            self.assertFalse(any(12345. in vector for vector in current.arm_velocity_rad_s+current.finger_velocity_rad_s))
            position = current.arm_position_rad
            controller.data.qpos[controller.hands['right']['kin'].qpos_adr] += .01
            self.assertEqual(current.arm_position_rad, position)
            with self.assertRaises(FrozenInstanceError):
                current.time_s = 1.
            self.assertEqual(controller.data.time, 0.)
        finally:
            controller.data.qvel[:] = qvel

    def test_deferrals_exhaust_a_fixed_budget_without_advancing_physics(self):
        controller = TaskSpaceCascade(lambda current: TaskCommand(current.nominal_parameters, defer_s=.005))
        initial = np.r_[controller.data.qpos, controller.data.qvel].copy()
        for _ in range(controller.MAX_THROW_PLAN_ATTEMPTS-1):
            self.assertIsNone(controller.plan_throw('right'))
        with self.assertRaisesRegex(RuntimeError, 'retries exhausted'):
            controller.plan_throw('right')
        self.assertEqual(controller.failure_event['reason'], 'NO_FEASIBLE_THROW')
        np.testing.assert_array_equal(np.r_[controller.data.qpos, controller.data.qvel], initial)
        self.assertEqual(controller.data.time, 0.)

    def test_native_retry_consumes_real_physics_time(self):
        def decide(current):
            delay = .005 if current.kind == 'throw' and current.released == 0 and current.attempt == 0 else 0.
            return TaskCommand(current.nominal_parameters, defer_s=delay)
        controller = TaskSpaceCascade(decide)
        result = controller.run_complete(timeout=.65)
        first = [row for row in controller.decisions if row['kind'] == 'throw' and row['released'] == 0]
        self.assertEqual(len(first), 2)
        self.assertAlmostEqual(first[1]['time_s']-first[0]['time_s'], .005)
        self.assertEqual(first[1]['attempt'], 1)
        self.assertEqual(result['throws'], 1)
        self.assertFalse(result['success'])

    def test_zero_task_residual_completes_the_original_contact_task(self):
        controller = TaskSpaceCascade()
        protect = controller.protect_arm
        peak_ratio = 0.

        def bounded_torque(hand, data, torque, response=None, free_acceleration=None):
            nonlocal peak_ratio
            final = protect(hand, data, torque, response, free_acceleration)
            peak_ratio = max(peak_ratio, float(np.max(np.abs(final)/hand['force_limits'])))
            return final

        controller.protect_arm = bounded_torque
        result = controller.run_complete(timeout=12.)
        self.assertTrue(result['success'], result['error'])
        self.assertEqual((result['throws'], result['catches']), (30, 30))
        self.assertGreaterEqual(result['final_hold']['continuous_support_s'], .8-1e-10)
        self.assertFalse(result['grasp_assistance'])
        self.assertEqual(result['grasp_constraints'], 0)
        self.assertEqual(result['ball_contacts'], 0)
        self.assertLess(result['max_height'], 1.3)
        self.assertFalse(result['conditions']['task_decisions']['online_physics_rollouts'])
        self.assertLessEqual(peak_ratio, 1.+1e-12)
        self.assertEqual(controller.stand_guard_steps, 0)


if __name__ == '__main__':
    unittest.main()

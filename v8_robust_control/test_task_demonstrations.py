from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .task_control import TaskCommand, decode_action
from .task_demonstrations import (encode_command, fork_controller, query_planner,
                                  live_fingerprint, PlannerAdvisor, summarize, PlanningBudgetExceeded,
                                  fit_event_tables, EventStudent, student_key,
                                  restore_planner_memory, planner_memory, candidate_events)
from .task_env import observation, TaskConfig
from .test_task_env import request


def fake_controller():
    data = SimpleNamespace(time=.3, **{name:np.zeros(6) for name in
        ('qpos', 'qvel', 'qacc', 'ctrl', 'qfrc_applied', 'xfrc_applied')})
    return SimpleNamespace(model=object(), data=data, decision=lambda _: None,
                           cancel=Event(), ball_observer=[['history']], failure_event=None)


class TaskDemonstrationTests(unittest.TestCase):
    def test_query_diagnostics_count_rollouts_and_forbidden_tail_without_changing_live_state(self):
        from types import MethodType
        original = fake_controller()
        original.step = MethodType(lambda c:setattr(c.data,'time',c.data.time+.001),original)
        original.reception_reference = MethodType(lambda c,*a,**kw:None,original)
        def forbidden(c):
            c.failure_event = dict(reason='BALL_COLLISION',time=c.data.time)
            raise RuntimeError('collision')
        original.check_forbidden_contacts = MethodType(forbidden,original)
        current = replace(request(),kind='reception',nominal_parameters=(0.,)*5)
        before = live_fingerprint(original)
        def predict(fork,current):
            fork.reception_reference('right',np.zeros(5))
            fork.step()
            fork.reception_reference('left',np.ones(5))
            try:
                fork.check_forbidden_contacts()
            except RuntimeError:
                pass
            fork.step()
            return TaskCommand(current.nominal_parameters)
        with patch('v8_robust_control.task_demonstrations.planner_command',side_effect=predict):
            row = query_planner(original,current)
        self.assertEqual(row['status'],'accepted')
        diagnostics = row['computation']
        self.assertEqual(diagnostics['rollout_steps'],2)
        self.assertEqual(len(diagnostics['candidates']),1)
        self.assertEqual(diagnostics['auxiliary_references'],1)
        self.assertEqual(diagnostics['candidates'][0]['rollout_steps'],2)
        self.assertEqual(diagnostics['candidates'][0]['steps_after_forbidden_contact'],1)
        self.assertEqual(diagnostics['candidates'][0]['first_forbidden_contact']['reason'],'BALL_COLLISION')
        self.assertGreaterEqual(diagnostics['rollout_wall_s'],0.)
        self.assertEqual(live_fingerprint(original),before)
        self.assertIsNone(original.failure_event)
        json.dumps(row,allow_nan=False)

    def test_labels_round_trip_without_changing_the_planner_command(self):
        current = request()
        for command in (TaskCommand((.1, 0., 3.1, .02, -.003)),
                        TaskCommand(current.nominal_parameters, defer_s=.005)):
            action = encode_command(current, command)
            decoded = decode_action(current, action)
            np.testing.assert_allclose(decoded.parameters, command.parameters, atol=1e-12)
            self.assertEqual(decoded.defer_s, command.defer_s)
        current = replace(current, kind='reception', nominal_parameters=(0.,)*5)
        command = TaskCommand((.01, -.02, -.01, .03, .02), tilt_scale=1.25)
        self.assertEqual(decode_action(current, encode_command(current, command)), command)

    def test_out_of_range_or_unquantized_labels_are_not_clipped(self):
        current = request()
        for command in (TaskCommand((.751, 0., 3., 0., 0.)),
                        TaskCommand(current.nominal_parameters, defer_s=.0055)):
            with self.assertRaises(ValueError):
                encode_command(current, command)

    def test_fork_detaches_live_worker_locks_and_preserves_observation_history(self):
        original = fake_controller()
        fork = fork_controller(original)
        self.assertIs(fork.model, original.model)
        self.assertIsNone(fork.cancel)
        self.assertIsNone(fork.decision)
        fork.ball_observer[0].append('future')
        fork.data.qpos[0] = 1.
        self.assertEqual(original.ball_observer, [['history']])
        self.assertEqual(original.data.qpos[0], 0.)
        original.ball_observer = SimpleNamespace(transport=SimpleNamespace(_digest=hashlib.sha256(b'past')))
        fork = fork_controller(original)
        self.assertEqual(fork.ball_observer.transport._digest.hexdigest(),
                         original.ball_observer.transport._digest.hexdigest())
        fork.ball_observer.transport._digest.update(b'future')
        self.assertNotEqual(fork.ball_observer.transport._digest.hexdigest(),
                            original.ball_observer.transport._digest.hexdigest())

    def test_teacher_prediction_cannot_advance_or_rewrite_live_physics(self):
        original, current = fake_controller(), request()
        before = live_fingerprint(original)
        def predict(fork, _):
            fork.data.qpos[:] = 123.
            fork.data.time += 1.
            return TaskCommand(current.nominal_parameters)
        with patch('v8_robust_control.task_demonstrations.planner_command', side_effect=predict):
            row = query_planner(original, current)
        self.assertEqual(row['status'], 'accepted')
        self.assertEqual(live_fingerprint(original), before)
        with patch('v8_robust_control.task_demonstrations.planner_command',
                   return_value=TaskCommand((.76, 0., 3., 0., 0.))):
            row = query_planner(original, current)
        self.assertEqual(row['status'], 'unrepresentable')
        self.assertNotIn('action', row)

    def test_joint_advisor_covers_reception_while_first_advisor_does_not(self):
        current = replace(request(), kind='reception', released=1, nominal_parameters=(0.,)*5)
        base = SimpleNamespace(_request=current, controller=fake_controller())
        env = SimpleNamespace(unwrapped=base)
        obs = observation(current, TaskConfig())
        answer = dict(status='accepted', action=[.1, 0., 0., 0., 0., 0.])
        with patch('v8_robust_control.task_demonstrations.query_planner', return_value=answer) as query:
            np.testing.assert_array_equal(PlannerAdvisor(env, 'planner_first', 4).predict(obs)[0], np.zeros(6))
            self.assertFalse(query.called)
            actor = PlannerAdvisor(env, 'planner_joint', 4)
            self.assertEqual(actor.predict(obs)[0][0], .1)
            self.assertEqual(len(actor.records), 1)
            obs['robot'][0] += 1.
            self.assertNotEqual(actor.records[0]['observation']['robot'][0], obs['robot'][0])
        base._request = replace(current, released=4)
        np.testing.assert_array_equal(actor.predict(obs)[0], np.zeros(6))

    def test_rejections_are_reported_and_never_become_guessed_actions(self):
        current = request()
        env = SimpleNamespace(unwrapped=SimpleNamespace(_request=current, controller=fake_controller()))
        actor = PlannerAdvisor(env, 'planner_joint', 4)
        with patch('v8_robust_control.task_demonstrations.query_planner',
                   return_value=dict(status='planner_rejected', error='no feasible reception')):
            with self.assertRaisesRegex(RuntimeError, 'TEACHER_PLANNER_REJECTED'):
                actor.predict(observation(current, TaskConfig()))
        self.assertEqual(actor.records[0]['status'], 'planner_rejected')
        row = dict(mode='planner_joint', result=dict(reason='TEACHER_PLANNER_REJECTED', training_prefix=False),
                   demonstrations=actor.records)
        self.assertEqual(summarize([row])['planner_joint']['rejected'], 1)
        self.assertEqual(summarize([row])['planner_joint']['recovered'], 0)

    def test_computation_limit_is_not_a_physical_rejection(self):
        original = fake_controller()
        before = live_fingerprint(original)
        with patch('v8_robust_control.task_demonstrations.planner_command',
                   side_effect=PlanningBudgetExceeded('budget')):
            result = query_planner(original, request())
        self.assertEqual(result['status'], 'resource_limit')
        self.assertNotIn('action', result)
        self.assertEqual(live_fingerprint(original), before)
        row = dict(mode='planner_joint', result=dict(reason='TEACHER_RESOURCE_LIMIT', training_prefix=False),
                   demonstrations=[result])
        summary = summarize([row])['planner_joint']
        self.assertEqual(summary['resource_limited'], 1)
        self.assertEqual(summary['rejected'], 0)
        self.assertEqual(summary['unsafe'], 0)

    def test_retry_cache_and_executed_history_survive_without_touching_live_state(self):
        original = fake_controller()
        original.hands = dict(right={}, left={})
        original.decisions = [dict(kind='throw', side='right', ball_id=2, time_s=.3,
                                   defer_s=0., parameters=[.1, .2, 3., 0., 0.])]
        original.events = [dict(kind='stroke', time=.3, side='right', ball=2)]
        state = dict(right=dict(key='throw:right:1', command=[.2, .1, 3.1, 0.]))
        observed = []
        def predict(fork, current):
            observed.append(fork.hands['right']['retry_throw_command'].copy())
            self.assertEqual([r['kind'] for r in fork.events], ['plan', 'stroke'])
            fork.hands['right']['retry_throw_command'][0] = .3
            return TaskCommand(current.nominal_parameters, defer_s=.005)
        with patch('v8_robust_control.task_demonstrations.planner_command', side_effect=predict):
            first = query_planner(original, request(), optimizer_state=state)
            second = query_planner(original, request(), optimizer_state=first['optimizer_state'])
        self.assertEqual(observed[0][0], .2)
        self.assertEqual(observed[1][0], .3)
        self.assertEqual(original.hands['right'], {})
        self.assertEqual([r['kind'] for r in original.events], ['stroke'])
        self.assertEqual(state['right']['command'][0], .2)
        self.assertEqual(second['planning_outcome'], 'defer')

    def test_candidate_import_checks_reset_and_does_not_trust_old_success(self):
        shifts = np.random.default_rng(17).uniform(-.002, .002, (2, 2))
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'candidate.json'
            # A candidate can have failed its old full task; only re-execution
            # in the current environment can establish the requested prefix.
            old = dict(events=[dict(kind='plan')], shifts_m=shifts.tolist(), result=dict(success=False))
            path.write_text(json.dumps(old), encoding='utf-8')
            self.assertEqual(candidate_events(path, 17, .002), old['events'])
            with self.assertRaises(ValueError):
                candidate_events(path, 18, .002)
            audit = dict(manifest=dict(controller='native', requested_throws=30), records=[
                dict(seed=17, settings=dict(seat_jitter_m=.002),
                     initial_displacement_m=shifts[[0,1,0]].tolist(), events=old['events'])])
            path.write_text(json.dumps(audit), encoding='utf-8')
            self.assertEqual(candidate_events(path, 17, .002), old['events'])
            audit['records'][0]['settings']['noise_std_m'] = .003
            path.write_text(json.dumps(audit), encoding='utf-8')
            with self.assertRaises(ValueError):
                candidate_events(path, 17, .002)

    def test_observation_only_student_switches_labels_and_survives_serialization(self):
        import json
        current = request()
        obs = observation(current, TaskConfig())
        key = student_key(obs)
        rows = []
        for x, action in ((-1., .1), (1., -.1)):
            state = {name:value.copy() for name,value in obs.items()}
            state['balls'][0, 0] = x
            from dataclasses import asdict
            command = TaskCommand((.75*action, 0., 3., 0., 0.))
            rows.append(dict(mode='planner_joint', goal=1,
                result=dict(training_prefix=True, recovered_ids=[1], catches=1, is_success=False),
                demonstrations=[dict(status='accepted', observation={k:v.tolist() for k,v in state.items()},
                    request=asdict(current), command=asdict(command), action=[action, 0., 0., 0., 0., 0.])]))
        tables = json.loads(json.dumps(fit_event_tables(rows)))
        self.assertIn(key, tables)
        student = EventStudent(tables, 'nearest_event', 4)
        fixed = EventStudent(tables, 'fixed_event', 4)
        obs['balls'][0, 0] = -1.
        self.assertAlmostEqual(student.predict(obs)[0][0], .1)
        np.testing.assert_allclose(fixed.predict(obs)[0], 0., atol=1e-12)
        obs['balls'][0, 0] = 1.
        self.assertAlmostEqual(student.predict(obs)[0][0], -.1)
        obs['context'][5] = 4./30
        np.testing.assert_array_equal(student.predict(obs)[0], np.zeros(6))
        rows[0]['result']['recovered_ids'] = []
        with self.assertRaises(ValueError):
            fit_event_tables(rows)


if __name__ == '__main__':
    unittest.main()

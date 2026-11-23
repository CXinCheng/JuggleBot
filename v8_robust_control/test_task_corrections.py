import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tempfile import TemporaryDirectory

from .task_control import decode_action
from .task_corrections import (correction_rows, validate_new_seeds, collect_trial,
                               fit_retained_head, apply_head, retention_selection, load_corrections,
                               fit_partial_actor, local_event_tables, command_difference,
                               compare_decision_traces, diagnostic_rollout, PartialActorConfig)
from .task_demonstrations import EventStudent
from .compare_residual import digest
from .task_imitation import validate_episode_splits
from .task_imitation import initialize_model
from .test_task_imitation import MeasurementEnv
from .test_task_env import request


class TaskCorrectionTests(unittest.TestCase):
    def test_partial_config_rejects_invalid_optimizer_settings(self):
        for settings in [dict(learning_rate=0.),dict(final_learning_rate=.1),
                         dict(anchor_weight=-1.),dict(anchor_weight=float('nan'))]:
            with self.assertRaises(ValueError):
                PartialActorConfig(**settings)
        self.assertEqual(PartialActorConfig(anchor_weight=0.).anchor_weight,0.)

    def test_command_comparison_uses_decoded_effect_not_raw_output(self):
        req = request()
        a, b = np.zeros(6), np.zeros(6)
        b[5] = .01
        self.assertFalse(command_difference(asdict(decode_action(req,a)),asdict(decode_action(req,b)))['changed'])
        b[0] = .01
        delta = command_difference(asdict(decode_action(req,a)),asdict(decode_action(req,b)))
        self.assertTrue(delta['changed'])
        self.assertAlmostEqual(delta['parameters'][0], .0075)

    def test_trace_comparison_stops_at_callback_or_clock_change(self):
        req = request()
        obs = MeasurementEnv(Path('reference')).obs
        row = dict(key='throw:right:0:2:0',time_s=.3,request=asdict(req),
                   command=asdict(decode_action(req,np.zeros(6))),
                   observation={k:v.tolist() for k,v in obs.items()})
        original = dict(trace=[copy.deepcopy(row),copy.deepcopy(row)])
        candidate = copy.deepcopy(original)
        candidate['trace'][0]['command']['parameters'] = list(candidate['trace'][0]['command']['parameters'])
        candidate['trace'][0]['command']['parameters'][0] += .01
        candidate['trace'][1]['observation']['balls'][0][0] += .02
        compared = compare_decision_traces(original,candidate)
        self.assertEqual(compared['first_command_change']['index'],0)
        self.assertEqual(compared['first_observation_change']['index'],1)
        candidate['trace'][1]['time_s'] += .001
        compared = compare_decision_traces(original,candidate)
        self.assertEqual(compared['alignment_lost']['index'],1)
        self.assertIsNone(compared['first_observation_change'])
        self.assertEqual(len(compared['paired_decisions']),1)

    def test_single_event_and_gain_interventions_only_change_normal_actions(self):
        class Env:
            def __init__(self):
                self.unwrapped,self._request,self.actions = self,request(),[]
            def reset(self,seed):
                return dict(measured=np.array([1.])),{}
            def step(self,action):
                self.actions.append(action.copy())
                return dict(measured=np.array([1.])),0.,len(self.actions)==2,False,dict(
                    reason='TRAINING_PREFIX',config=dict(seat_jitter_m=.0005))
        parent = SimpleNamespace(predict=lambda obs,deterministic:(np.zeros(6),None))
        candidate = SimpleNamespace(predict=lambda obs,deterministic:(np.full(6,.2),None))
        for options,expected in [(dict(single=1),[0.,.2]),(dict(prefix=1),[.2,0.]),(dict(gain=.25),[.05,.05])]:
            env = Env()
            with patch('v8_robust_control.task_corrections.action_authority',return_value=1.):
                trace = diagnostic_rollout(env,parent,candidate,10,'test',**options)
            np.testing.assert_allclose(env.actions,np.repeat(np.asarray(expected)[:,None],6,axis=1))
            self.assertEqual(len(trace['trace']),2)
        with self.assertRaises(ValueError):
            diagnostic_rollout(Env(),parent,candidate,10,'invalid',gain=2.)
        env = Env()
        env.controller = SimpleNamespace(events=[dict(kind='release')])
        with patch('v8_robust_control.task_corrections.action_authority',return_value=1.):
            trace = diagnostic_rollout(env,parent,candidate,10,'axes',single=0,components=(0,2))
        np.testing.assert_allclose(env.actions,[[.2,0.,.2,0.,0.,0.],[0.]*6])
        trace['physical_events'][0]['kind'] = 'changed'
        self.assertEqual(env.controller.events[0]['kind'],'release')

    def test_partial_actor_fits_only_last_hidden_layer_and_head(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as tmp:
            reference = Path(tmp)/'reference.json'
            reference.write_text('{}', encoding='utf-8')
            env = MeasurementEnv(reference)
            size = sum(int(np.prod(s.shape)) for s in env.observation_space.spaces.values())
            model = initialize_model(env, 0, [0.]*size, [1.]*size, 6)
            def batch(values):
                obs = {k:torch.as_tensor(np.repeat(v[None], 2, axis=0)) for k,v in env.obs.items()}
                obs['errors'][:,0,0] = torch.tensor(values)
                return obs, torch.full((2,6), .1), torch.ones(2)
            anchors, corrections = batch([.2,.4]), batch([2.,4.])
            old = {k:v.clone() for k,v in model.policy.state_dict().items()}
            checkpoints = []
            config = PartialActorConfig(learning_rate=2e-4,anchor_weight=50.)
            fit = fit_partial_actor(model, anchors, corrections, 500,
                                    lambda epoch,*_:checkpoints.append(epoch),config=config)
            self.assertEqual(fit['anchor_weight'],50.)
            self.assertEqual(fit['learning_rate'],2e-4)
            self.assertEqual(checkpoints, [0,125,250,500])
            self.assertLess(fit['history'][-1]['correction_mse'], fit['history'][0]['correction_mse'])
            allowed = ('mlp_extractor.policy_net.2.', 'action_net.')
            changed = []
            for key,value in old.items():
                if not torch.equal(value, model.policy.state_dict()[key]):
                    changed.append(key)
                    self.assertTrue(key.startswith(allowed), key)
            self.assertTrue(any(k.startswith(allowed[0]) for k in changed))
            self.assertTrue(any(k.startswith(allowed[1]) for k in changed))
            self.assertTrue(all(p.requires_grad for p in model.policy.parameters()))
            self.assertEqual(model.num_timesteps, 0)

    def test_fixed_event_uses_all_local_training_labels_without_success_claim(self):
        with TemporaryDirectory() as tmp:
            env = MeasurementEnv(Path(tmp)/'reference.json')
            obs = {k:v.copy() for k,v in env.obs.items()}
            obs['errors'][0,0] = 1.
            rows = [dict(demonstrations=[dict(observation={k:v.tolist() for k,v in obs.items()},
                                             action=[value,0.,0.,0.,0.,0.])]) for value in (.1,.3)]
            tables = local_event_tables(rows)
            action = EventStudent(tables, 'fixed_event', 6).predict(obs)[0]
            np.testing.assert_allclose(action, [.2,0.,0.,0.,0.,0.])
            self.assertNotIn('result', rows[0])

    def test_reused_labels_keep_archived_collector_identity_and_old_holdouts(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            reference, training, archive = (folder/'reference.json', folder/'training.json', folder/'driver_source.py')
            reference.write_text('{}', encoding='utf-8')
            training.write_text('{}', encoding='utf-8')
            archive.write_text('# historical collector', encoding='utf-8')
            protocol = dict(source_sha256={}, parent_checkpoint='parent.zip',
                parent_training_sha256=digest(training), goal=3, control_releases=6,
                input_sha256={(folder/'task_corrections.py').as_posix():digest(archive),
                              reference.as_posix():digest(reference)},
                train_seeds=[10], validation_seeds=[20], test_seeds=[30],
                reserved_previous_seeds=[1,2,3])
            data = dict(protocol=protocol, records=[dict(seed=10, corrections=[])])
            (folder/'corrections.json').write_text(json.dumps(data), encoding='utf-8')
            with patch('v8_robust_control.task_corrections.runtime_sources', return_value={}):
                rows, inputs, reserved = load_corrections([folder], reference, folder, 'parent.zip', 3, 6)
                self.assertEqual(reserved, {1,2,3,10,20,30})
                self.assertEqual(rows[0]['seed'], 10)
                self.assertIn(archive.as_posix(), inputs)
                with self.assertRaises(ValueError):
                    validate_episode_splits([40], [20], reserved)
                archive.write_text('# modified', encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_corrections([folder], reference, folder, 'parent.zip', 3, 6)

    def test_retained_fit_freezes_trunk_and_preserves_anchor_outputs(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as tmp:
            reference = Path(tmp)/'reference.json'
            reference.write_text('{}', encoding='utf-8')
            env = MeasurementEnv(reference)
            size = sum(int(np.prod(s.shape)) for s in env.observation_space.spaces.values())
            model = initialize_model(env, 0, [0.]*size, [1.]*size, 6)
            def batch(values):
                obs = {k:torch.as_tensor(np.repeat(v[None], 2, axis=0)) for k,v in env.obs.items()}
                obs['errors'][:,0,0] = torch.tensor(values)
                return obs, torch.full((2,6), .1), torch.ones(2)
            anchors, corrections = batch([.2,.4]), batch([.6,.8])
            state = {k:v.clone() for k,v in model.policy.state_dict().items()}
            old_actions = model.policy._predict(anchors[0], deterministic=True).detach().clone()
            before, delta, fitting = fit_retained_head(model, anchors, corrections)
            self.assertLess(fitting['anchor_linear_drift'], 1e-12)
            self.assertLessEqual(fitting['update_norm'], .5+1e-12)
            self.assertGreater(fitting['free_features'], 0)
            apply_head(model, before, delta, 1.)
            for key,value in state.items():
                if not key.startswith('action_net.'):
                    self.assertTrue(torch.equal(value, model.policy.state_dict()[key]), key)
            torch.testing.assert_close(old_actions, model.policy._predict(anchors[0], deterministic=True),
                                       rtol=0., atol=1e-7)
            apply_head(model, before, delta, 0.)
            for key,value in state.items():
                self.assertTrue(torch.equal(value, model.policy.state_dict()[key]), key)

    def test_anchor_regression_cannot_win_with_better_validation(self):
        def row(seed, success, reason='TRAINING_PREFIX'):
            return dict(seed=seed, result=dict(reason=reason, training_prefix=success,
                        recovered_ids=[1,2,3] if success else []))
        original = [row(1, True), row(2, True)]
        fallback = dict(fraction=0., anchor_rollouts=copy.deepcopy(original),
                        validation_rollouts=[row(20, False, 'POSE_LIMIT')])
        regression = dict(fraction=1., anchor_rollouts=[row(1, True), row(2, False, 'FORBIDDEN_CONTACT')],
                          validation_rollouts=[row(20, True)])
        self.assertIs(retention_selection([fallback, regression], original, 3), fallback)
        self.assertFalse(regression['anchors_retained'])
        retained = dict(fraction=.25, anchor_rollouts=copy.deepcopy(original),
                        validation_rollouts=[row(20, True)])
        self.assertIs(retention_selection([fallback, retained], original, 3), retained)

    def test_new_collection_excludes_all_old_holdouts(self):
        parent = dict(demonstration_attempt_seeds=[1], validation_episode_seeds=[2], test_episode_seeds=[3])
        validate_new_seeds([10, 11], [20], [30], parent)
        validate_new_seeds([], [20], [30], parent, allow_empty=True)
        with self.assertRaises(ValueError):
            validate_new_seeds([], [20], [30], parent)
        for train, val, test in [([2], [20], [30]), ([3], [20], [30]),
                                 ([10], [1], [30]), ([10], [20], [3]),
                                 ([10], [20], [20]), ([10, 10], [20], [30])]:
            with self.assertRaises(ValueError):
                validate_new_seeds(train, val, test, parent)

    def test_failed_assisted_episode_retains_only_accepted_local_labels(self):
        req = request()
        expert, learner = np.array([.1, .1, .1, .1, .1, 0.]), np.zeros(6)
        item = dict(status='accepted', request=asdict(req),
                    command=asdict(decode_action(req, expert)), action=expert.tolist(),
                    learner_action=learner.tolist(), executed_by='learner', executed_action=learner.tolist())
        row = dict(seed=10, jitter_m=.0005, result=dict(reason='BALL_HEIGHT', training_prefix=False),
                   corrections=[item, dict(status='resource_limit')])
        batch = correction_rows([row])
        self.assertEqual(len(batch[0]['demonstrations']), 1)
        self.assertEqual(row['result']['reason'], 'BALL_HEIGHT')
        bad = copy.deepcopy(row)
        bad['corrections'][0]['executed_action'] = expert.tolist()
        with self.assertRaises(AssertionError):
            correction_rows([bad])
        bad = copy.deepcopy(row)
        bad['corrections'][0]['action'][0] = 1.
        with self.assertRaises(AssertionError):
            correction_rows([bad])

    def test_learner_prefix_runs_before_teacher_takeover(self):
        req = request()
        learner, expert = np.zeros(6), np.array([.1, .1, .1, .1, .1, 0.])
        class FakeEnv:
            def __init__(self):
                self.unwrapped = self
                self._request, self.controller, self.executed, self.closed = req, object(), [], False
            def reset(self, seed):
                return dict(measured=np.array([1.], np.float32)), {}
            def step(self, action):
                self.executed.append(action.copy())
                n = len(self.executed)
                return dict(measured=np.array([1.+n], np.float32)), 0., n==3, False, dict(reason='TRAINING_PREFIX')
            def close(self):
                self.closed = True
        actor = SimpleNamespace(predict=lambda obs, deterministic: (learner.copy(), None))
        label = dict(status='accepted', command=asdict(decode_action(req, expert)),
                     action=expert.tolist(), optimizer_state={})
        for prefix in (0, 2):
            env = FakeEnv()
            with self.subTest(prefix=prefix), \
                 patch('v8_robust_control.task_corrections.load_parent', return_value=(actor, {})), \
                 patch('v8_robust_control.task_corrections.TaskResidualEnv'), \
                 patch('v8_robust_control.task_corrections.TaskCurriculum', return_value=env), \
                 patch('v8_robust_control.task_corrections.action_authority', return_value=1.), \
                 patch('v8_robust_control.task_corrections.query_planner', side_effect=lambda *a:copy.deepcopy(label)):
                row = collect_trial((Path('.'), 'parent.zip', Path('ref'), 10, .0005, 3, 6, prefix, 1., 5.))
            self.assertTrue(env.closed)
            np.testing.assert_array_equal(env.executed, [learner]*prefix+[expert]*(3-prefix))
            self.assertEqual([r['executed_by'] for r in row['corrections']],
                             ['learner']*prefix+['teacher']*(3-prefix))
            self.assertEqual(row['corrections'][2]['observation']['measured'], [3.])
            self.assertFalse(row['rl_gain_verified'])
            if not prefix:
                self.assertEqual(row['role'], 'privileged teacher from the initial state')


if __name__ == '__main__':
    unittest.main()

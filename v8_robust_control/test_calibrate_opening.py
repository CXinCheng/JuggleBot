from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .calibrate_opening import (OpeningProfile, ClearanceCascade, OpeningEnv, opening_action,
                                reception_residual, rank, trial, summarize_results)
from .calibrate_opening import joint_profiles,retained_height_profiles
from .calibrate_opening import ReceptionPlanningCascade
from .task_control import TaskCommand
from .task_env import TaskConfig, TaskResidualEnv
from .train_task import TaskCurriculum


class OpeningCalibrationTests(unittest.TestCase):
    def test_preview_audit_preserves_live_physics_and_original_motion(self):
        torch.set_num_threads(1)
        states = []
        previews = []
        original = ReceptionPlanningCascade.preview_reception
        def checked_preview(controller, side, reference):
            arrays = [controller.data.qpos,controller.data.qvel,controller.data.ctrl,
                      controller.data.qfrc_applied,controller.data.xfrc_applied,
                      controller.hands[side]['kin'].data.qpos]
            before = [a.copy() for a in arrays]
            result = original(controller,side,reference)
            for a,b in zip(arrays,before):
                np.testing.assert_array_equal(a,b)
            previews.append(result)
            return result
        with patch.object(ReceptionPlanningCascade,'preview_reception',checked_preview):
            for mode in ('off','audit'):
                base = OpeningEnv(TaskConfig(),None,OpeningProfile(reception_planner=mode))
                env = TaskCurriculum(base,[dict(catch_goal=1,seat_jitter_m=0.)])
                try:
                    env.reset(seed=0)
                    while True:
                        _,_,done,truncated,info = env.step(np.zeros(6))
                        if done or truncated:
                            break
                    states.append((info['reason'],base.controller.data.qpos.copy(),base.controller.data.qvel.copy()))
                finally:
                    env.close()
        self.assertTrue(previews)
        for preview in previews:
            for phase,key in [('approach','approach_feasible'),('uncaught_cushion','uncaught_cushion_feasible')]:
                samples = [s for s in preview['samples'] if s['phase']==phase]
                self.assertTrue(samples)
                self.assertEqual(preview[key],all(s['violation']<=1e-9 for s in samples))
        self.assertEqual(states[0][0],states[1][0])
        for a,b in zip(states[0][1:],states[1][1:]):
            np.testing.assert_array_equal(a,b)

    def test_infeasible_preview_falls_back_to_original_reference(self):
        parameters = np.zeros(5)
        reference = (object(),object(),object())
        applied = []
        request = SimpleNamespace(ball_id=0,released=4,time_s=1.)
        c = SimpleNamespace(hands={'left':{'incoming':0}},reception_planner='project',reception_plans=[],
            _request=lambda *a:request,_command=lambda r:(TaskCommand(tuple(parameters)),parameters),
            reception_reference=lambda *a,**kw:reference,
            preview_reception=lambda *a:dict(feasible=False,violation=1.,samples=[]),
            reception_bounds=lambda s:(np.full(5,-1.),np.full(5,1.)),
            apply_reception_reference=lambda h,*ref:applied.append(ref))
        with patch('v8_robust_control.calibrate_opening.validate_command') as validate:
            ReceptionPlanningCascade.plan_reception(c,'left')
        self.assertEqual(validate.call_count,9)
        self.assertEqual(applied,[reference])
        self.assertTrue(c.reception_plans[0]['fallback'])
        self.assertFalse(c.reception_plans[0]['changed'])
        np.testing.assert_array_equal(c.hands['left']['return_parameters'],parameters)

    def test_retained_height_search_keeps_first_two_launches_and_receptions(self):
        profiles = retained_height_profiles()
        self.assertEqual(len(profiles),10)
        for values in profiles[1:]:
            profile = OpeningProfile(**values)
            obs = dict(context=np.zeros(26,np.float32))
            obs['context'][0] = 1.
            for released in (0,1):
                obs['context'][5] = released/30
                np.testing.assert_array_equal(opening_action(obs,profile,[0.]*3),np.zeros(6))
            obs['context'][5] = 2/30
            self.assertLess(opening_action(obs,profile,[0.]*3)[2],0.)
            obs['context'][0],obs['context'][1] = 0.,1.
            for released in (2,3,6):
                obs['context'][5] = released/30
                np.testing.assert_array_equal(opening_action(obs,profile,[0.]*3),np.zeros(6))
            obs['context'][5] = 4/30
            np.testing.assert_allclose(opening_action(obs,profile,[0.]*3)[:3],np.asarray(profile.reception_offset_m)/.06)
        with self.assertRaises(ValueError):
            OpeningProfile(reception_start_release=-1)

    def test_joint_offsets_follow_release_count_and_end_after_opening(self):
        profiles = joint_profiles()
        self.assertEqual(len(profiles),13)
        self.assertEqual(OpeningProfile(**profiles[0]),OpeningProfile())
        p = OpeningProfile(launch_velocity_m_s=((.01,0.,0.),(-.01,0.,0.),(.01,0.,-.01)),
                           reception_offset_m=(.005,0.,0.),release_lead_s=.001)
        obs = dict(context=np.zeros(26,np.float32))
        obs['context'][0] = 1.
        for released,x,z in [(0,.01,0.),(1,-.01,0.),(2,.01,-.01),(3,0.,0.)]:
            obs['context'][5] = released/30
            a = opening_action(obs,p,[0.]*3)
            self.assertAlmostEqual(a[0],x/.75)
            self.assertAlmostEqual(a[2],z/1.25)
            self.assertAlmostEqual(a[4],.001/.015 if released<3 else 0.)
            self.assertLessEqual(float(np.max(np.abs(a))),1.)
        obs['context'][0],obs['context'][1] = 0.,1.
        for released in (2,5,6):
            obs['context'][5] = released/30
            expected = [.005/.06,0.,0.,0.,0.,0.] if released<6 else [0.]*6
            np.testing.assert_allclose(opening_action(obs,p,[0.]*3),expected)
        with self.assertRaises(ValueError):
            OpeningProfile(reception_offset_m=(.03,0.,0.))
        with self.assertRaises(ValueError):
            OpeningProfile(release_lead_s=.004)

    def test_robot_margin_reads_contacts_and_capped_proximity_without_state_changes(self):
        bodies = {0:SimpleNamespace(name='world'),1:SimpleNamespace(name='lh_palm'),
                  2:SimpleNamespace(name='ball0'),'left_elbow_link':SimpleNamespace(id=3),
                  'left_shoulder_pitch_link':SimpleNamespace(id=4)}
        xpos = np.zeros((5,3)); xpos[3,1],xpos[4,1] = .08,.04
        m = SimpleNamespace(body=lambda i:bodies[i],geom_bodyid=[0,1,2],
                            geom=lambda i:SimpleNamespace(name=['stand','palm','ball'][i]),opt=SimpleNamespace(timestep=.001))
        d = SimpleNamespace(time=.01,xpos=xpos,contact=[SimpleNamespace(geom1=0,geom2=1,dist=-.003),
                                                       SimpleNamespace(geom1=1,geom2=2,dist=-.02)])
        c = SimpleNamespace(model=m,data=d,hands={'left':{}},_clearance_geoms=(1,),_stand_geom=0,
            _clearance_tags=(0,2,-1),_clearance_elbows=((3,4),),
            clearance=dict(max_robot_penetration_m=0.,robot_contact_pair=None,min_elbow_margin_m=None,
                           min_hand_stand_gap_m=None,hand_stand_samples=0,hand_stand_time_s=None))
        before = xpos.copy()
        with patch('v8_robust_control.calibrate_opening.mujoco.mj_geomDistance',return_value=-.001):
            ClearanceCascade.measure_robot_clearance(c)
        np.testing.assert_array_equal(xpos,before)
        self.assertEqual(c.clearance['robot_contact_pair'],['stand','palm'])
        self.assertAlmostEqual(c.clearance['max_robot_penetration_m'],.003)
        self.assertAlmostEqual(c.clearance['min_hand_stand_gap_m'],-.001)
        self.assertAlmostEqual(c.clearance['min_elbow_margin_m'],.06)

    def test_phase_summary_compares_only_matching_center_trials(self):
        profile = dict(settle_s=0.,forward_up_m_s=0.)
        def row(success,probe):
            return dict(profile=profile,probe_m_s=probe,
                        result=dict(training_prefix=success,is_success=False,reason='TRAINING_PREFIX' if success else 'DROPPED'),
                        clearance=dict(min_gap_m=.01))
        center,probe = row(False,[0.]*3),row(True,[.001,0.,0.])
        summary = summarize_results([profile],0,[center,probe],[center],[])
        self.assertEqual(summary['test']['0']['episodes'],2)
        self.assertEqual(summary['phase_baseline']['episodes'],1)
        self.assertEqual(summary['phase_baseline']['completed'],0)
        self.assertFalse(summary['calibration_accepted'])
        other = dict(settle_s=0.,forward_up_m_s=.01)
        better = row(True,[0.]*3)
        better['profile'] = other
        self.assertFalse(summarize_results([profile,other],1,[center,better],[],[])['calibration_accepted'])
        clean = [dict(profile=p,result=dict(is_success=True,reason='SUCCESS')) for p in (profile,other)]
        self.assertTrue(summarize_results([profile,other],1,[center,better],[],clean)['calibration_accepted'])

    def test_nominal_calibration_bypasses_reference_gate_but_policy_keeps_reference(self):
        obs = dict(context=np.zeros(26,np.float32))
        obs['context'][0] = 1.
        class FakeEnv:
            def __init__(self):
                self.unwrapped = self
                self.controller = SimpleNamespace(clearance={})
                self.reference = {'nominal':'measurement'}
            def reset(self,seed):
                return obs,{}
            def step(self,action):
                gates.append(bool(self.reference))
                return obs,0.,True,False,dict(is_success=False,reason='TRAINING_PREFIX')
            def close(self):
                pass
        references = []
        gates = []
        def factory(config,reference,profile):
            references.append(reference)
            return FakeEnv()
        actor = SimpleNamespace(predict=lambda obs,deterministic:(np.zeros(6),None))
        with patch('v8_robust_control.calibrate_opening.OpeningEnv',side_effect=factory), \
             patch('v8_robust_control.calibrate_opening.TaskCurriculum',side_effect=lambda env,*a:env), \
             patch('v8_robust_control.calibrate_opening.PPO.load',return_value=actor):
            for candidate in (None,'policy.zip'):
                trial(('reference.json',dict(settle_s=0.,forward_up_m_s=.1),0,0.,[0.]*3,3,candidate,.002))
        self.assertEqual(references,['reference.json','reference.json'])
        self.assertEqual(gates,[False,True])

    def test_calibration_template_keeps_the_reference_estimator_for_both_preview_modes(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import mujoco
        from .task_env import measured_sources
        for mode in ('off','audit'):
            record=dict(source_sha256=measured_sources(),mujoco_version=mujoco.__version__,
                requested_throws=30,event_estimator='filtered',requests=[],result={})
            with TemporaryDirectory() as directory:
                path=Path(directory)/'reference.json'
                path.write_text(json.dumps(record),encoding='utf-8')
                with patch('v8_robust_control.task_env.result_reason',return_value='SUCCESS'):
                    env=OpeningEnv(TaskConfig(),path,OpeningProfile(reception_planner=mode))
                try:
                    self.assertEqual(env.estimator,'filtered')
                    self.assertEqual(env._template.estimator,'filtered')
                    self.assertEqual(env._template.ball_observer.conditions()['estimator'],
                        'contact_support_and_ballistic_fit')
                    env.reset(seed=0)
                    self.assertEqual(env.controller.estimator,'filtered')
                finally:
                    env.close()

    def test_clearance_excludes_held_balls_and_never_writes_physics(self):
        qpos = np.array([0.,0.,1., .1,0.,1., .01,0.,1.])
        c = SimpleNamespace(clearance=dict(min_gap_m=None,pair=None,time_s=None,samples=0,peak_ball_height_m=0.),
            balls=[(0,0),(3,3),(6,6)],airborne={0:1,1:2,2:3},hands={'right':{},'left':{}},
            data=SimpleNamespace(qpos=qpos,time=.5),ball_geom_ids=[0,1,2],
            model=SimpleNamespace(geom_size=np.full((3,1),.0425)),ball_contact=lambda i,s:i==2)
        before = qpos.copy()
        ClearanceCascade.measure_clearance(c)
        np.testing.assert_array_equal(qpos,before)
        self.assertAlmostEqual(c.clearance['min_gap_m'],.015)
        self.assertEqual(c.clearance['pair'],[0,1])
        self.assertEqual(c.clearance['samples'],1)
        c.data.qpos[3] = .08
        ClearanceCascade.measure_clearance(c)
        self.assertAlmostEqual(c.clearance['min_gap_m'],-.005)

    def test_opening_and_reception_authority_are_separate(self):
        obs = dict(context=np.zeros(26,np.float32))
        profile = OpeningProfile(.1,.1)
        obs['context'][0] = 1.
        np.testing.assert_allclose(opening_action(obs,profile,[.001,0.,0.]),[.101/.75,0.,.1/1.25,0.,0.,0.])
        local = OpeningProfile(velocity_bias_m_s=(-.01,0.,.01))
        np.testing.assert_allclose(opening_action(obs,local,[0.]*3),[-.01/.75,0.,.01/1.25,0.,0.,0.])
        np.testing.assert_array_equal(reception_residual(obs,np.ones(6)),np.zeros(6))
        obs['context'][0],obs['context'][1],obs['context'][5] = 0.,1.,1/30
        np.testing.assert_array_equal(reception_residual(obs,np.ones(6)),np.zeros(6))
        obs['context'][5] = 2/30
        np.testing.assert_allclose(reception_residual(obs,np.ones(6)),[.002/.06]*3+[0.]*3)
        np.testing.assert_array_equal(opening_action(obs,profile,[0.]*3),np.zeros(6))
        with self.assertRaises(ValueError):
            reception_residual(obs,np.ones(6),cap_m=.1)
        with self.assertRaises(ValueError):
            OpeningProfile(settle_s=.3)
        with self.assertRaises(ValueError):
            OpeningProfile(velocity_bias_m_s=(.3,0.,0.))

    def test_missing_airborne_samples_cannot_outscore_completion(self):
        def row(completed,gap):
            return dict(result=dict(training_prefix=completed,is_success=False,reason='TRAINING_PREFIX' if completed else 'DROPPED'),
                        clearance=dict(min_gap_m=gap))
        self.assertGreater(rank([row(True,.001)]),rank([row(False,1.)]))
        self.assertGreater(rank([row(True,.001)]),rank([row(True,None)]))

    def test_zero_profile_monitor_matches_original_physics(self):
        torch.set_num_threads(1)
        states = []
        for monitored in (False,True):
            base = OpeningEnv(TaskConfig(),None,OpeningProfile()) if monitored else TaskResidualEnv(TaskConfig())
            env = TaskCurriculum(base,[dict(catch_goal=1,seat_jitter_m=0.)])
            try:
                env.reset(seed=0)
                while True:
                    _,_,done,truncated,info = env.step(np.zeros(6))
                    if done or truncated:
                        break
                states.append((info,base.controller.data.qpos.copy(),base.controller.data.qvel.copy()))
            finally:
                env.close()
        self.assertEqual(states[0][0],states[1][0])
        np.testing.assert_array_equal(states[0][1],states[1][1])
        np.testing.assert_array_equal(states[0][2],states[1][2])


if __name__=='__main__':
    unittest.main()

"""Position-only event estimation across contact, transport and release."""
import unittest
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import mujoco
import subprocess
import sys
import numpy as np
from jugglebot.robustness.observation import BallObservation,PositionEstimator
from .event_observation import EventPositionEstimator


class EventEstimatorTests(unittest.TestCase):
    def test_cli_records_the_filtered_estimator_source_manifest(self):
        with TemporaryDirectory() as folder:
            output=Path(folder)/'short_task.json'
            completed=subprocess.run([sys.executable,'-X','utf8','-m','v8_robust_control.task_control',
                '--estimator','filtered','--timeout','.01','--seat-jitter-mm','0','--output',str(output)],
                cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=60)
            self.assertEqual(completed.returncode,1,completed.stderr)
            self.assertTrue(output.is_file(),completed.stderr)
            record=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(record['result']['error'],'TIMEOUT')
            self.assertIn('v8_robust_control/event_observation.py',record['source_sha256'])
            self.assertEqual(record['result']['conditions']['ball_observation']['estimator'],
                             'contact_support_and_ballistic_fit')

    def test_reference_selects_its_estimator_and_rejects_a_different_override(self):
        from .task_env import TaskResidualEnv,measured_sources
        record=dict(source_sha256=measured_sources(),mujoco_version=mujoco.__version__,
                    requested_throws=30,event_estimator='filtered',requests=[],result={})
        with TemporaryDirectory() as folder:
            path=Path(folder)/'reference.json'
            path.write_text(json.dumps(record),encoding='utf-8')
            # This checks metadata routing, independently of the physical
            # reference acceptance exercised by full-task runs.
            with patch('v8_robust_control.task_env.result_reason',return_value='SUCCESS'), \
                 patch('v8_robust_control.task_env._EpisodeCascade') as factory:
                env=TaskResidualEnv(reference=path)
                self.assertEqual(env.estimator,'filtered')
                factory.assert_called_once_with(throws=30,estimator='filtered')
                env.close()
                with self.assertRaisesRegex(ValueError,'estimators differ'):
                    TaskResidualEnv(reference=path,estimator='difference')

    def test_release_excludes_support_and_delayed_pre_release_positions(self):
        g = np.array([0.,0.,-9.81])
        for delay in (0.,.02):
            tracker = EventPositionEstimator(2,g,.001,.05)
            pending = []
            valid = 0
            for tick in range(201):
                now=tick*.001
                tracker.notify_support(now,np.zeros((2,3)),np.zeros((2,3)),0 if now<.05 else None)
                dt=now-.05
                position=np.array([.1,.2,1.]) if now<.05 else np.array([.1,.2,1.])+dt*np.array([.3,-.2,2.])+.5*(dt*dt+.001*dt)*g
                if tick%17==0:
                    pending.append(BallObservation(2,now,now+delay,position,velocity_m_s=(999.,999.,999.)))
                while pending and pending[0].received_time_s<=now+1e-12:
                    tracker.update(replace(pending.pop(0),received_time_s=now))
                state=tracker.estimate(now)
                if .05<=now<.085+delay-1e-12:
                    self.assertIsNone(state)
                if now>=.085+delay:
                    self.assertIsNotNone(state)
                    self.assertEqual(state.ball_id,2)
                    np.testing.assert_allclose(state.position_m,position,atol=1e-12)
                    np.testing.assert_allclose(state.velocity_m_s,np.array([.3,-.2,2.])+dt*g,atol=1e-12)
                    valid+=1
            self.assertGreater(valid,50)

    def test_contact_does_not_assume_free_fall_or_invent_position_frames(self):
        tracker = EventPositionEstimator(0,[0.,0.,-9.81],.001,.05)
        acceleration=np.array([2.,-1.,3.])
        for tick in range(151):
            t=tick*.001
            palms=np.array([.5*(t*t+.001*t)*acceleration,np.zeros(3)])
            tracker.notify_support(t,palms,np.array([t*acceleration,np.zeros(3)]),0)
            if tick in (0,17,34):
                tracker.update(BallObservation(0,t,t,palms[0]+[0.,0.,.022]))
            state=tracker.estimate(t)
            if 35<=tick<=84:
                self.assertIsNotNone(state)
                np.testing.assert_allclose(state.position_m,palms[0]+[0.,0.,.022],atol=1e-12)
            if tick>84:
                self.assertIsNone(state)

    def test_noise_is_filtered_without_observation_velocity_or_launch_prior(self):
        rng=np.random.default_rng(416100)
        errors=[[],[]]
        for trial in range(200):
            filtered=EventPositionEstimator(0,[0.,0.,-9.81],.001,.05)
            raw=PositionEstimator(0)
            for i in range(8):
                t=i/60
                filtered.notify_support(t,np.zeros((2,3)),np.zeros((2,3)),None)
                p=np.array([.3,.2,1.])+t*np.array([.2,.3,2.])+.5*(t*t+.001*t)*np.array([0.,0.,-9.81])
                frame=BallObservation(0,t,t,p+rng.normal(0.,.003,3),velocity_m_s=(999.,999.,999.))
                filtered.update(frame); raw.update(frame)
            actual=np.array([.2,.3,2.])+t*np.array([0.,0.,-9.81])
            for j,tracker in enumerate((filtered,raw)):
                errors[j].append(np.asarray(tracker.estimate(t).velocity_m_s)-actual)
        self.assertLess(np.std(errors[0]),.08)
        self.assertLess(np.std(errors[0]),.25*np.std(errors[1]))

    def test_rejected_frames_and_reset_cannot_preserve_stale_history(self):
        tracker=EventPositionEstimator(0,[0.,0.,-9.81],.001,.05)
        for i in range(3):
            t=i*.017
            tracker.notify_support(t,np.zeros((2,3)),np.zeros((2,3)),None)
            self.assertTrue(tracker.update(BallObservation(0,t,t,(0.,0.,1.))))
        before=tracker.estimate(.034)
        self.assertFalse(tracker.update(BallObservation(0,.017,.034,(9.,9.,9.))))
        self.assertEqual(before,tracker.estimate(.034))
        with self.assertRaises(ValueError):
            tracker.update(BallObservation(1,.05,.05,(0.,0.,1.)))
        tracker.notify_support(.051,np.zeros((2,3)),np.zeros((2,3)),None)
        tracker.update(BallObservation(0,.051,.051,(0.,0.,1.),valid=False))
        self.assertIsNone(tracker.estimate(.051))
        tracker.reset()
        self.assertIsNone(tracker.estimate(0.))

    def test_gap_requires_new_frames_and_queries_cannot_use_future_proprioception(self):
        tracker=EventPositionEstimator(0,[0.,0.,-9.81],.001,.05)
        for t in (0.,.017,.034,.2,.217,.234,.235):
            tracker.notify_support(t,np.zeros((2,3)),np.zeros((2,3)),0)
            if t!=.235:
                tracker.update(BallObservation(0,t,t,(0.,0.,.022)))
            if t in (.2,.217):
                self.assertIsNone(tracker.estimate(t))
        np.testing.assert_allclose(tracker.estimate(.235).position_m,[0.,0.,.022],atol=1e-12)
        with self.assertRaises(ValueError):
            tracker.estimate(.234)
        with self.assertRaises(ValueError):
            tracker.estimate(float('nan'))

    def test_filtered_observer_survives_environment_clone_and_reset(self):
        from .task_env import TaskConfig,TaskResidualEnv
        env=TaskResidualEnv(TaskConfig(max_decisions=1),estimator='filtered')
        try:
            first=[]
            trackers=[]
            for _ in range(2):
                obs,info=env.reset(seed=416000)
                first.append(obs)
                trackers.append(env.controller.ball_observer.estimators[0])
                self.assertEqual(env.controller.estimator,'filtered')
                self.assertEqual(env.controller.ball_observer.conditions()['estimator'],'contact_support_and_ballistic_fit')
                self.assertEqual(tuple(s.ball_id for s in env._request.balls),(0,1,2))
            self.assertIsNot(trackers[0],trackers[1])
            for name in first[0]:
                np.testing.assert_array_equal(first[0][name],first[1][name])
        finally:
            env.close()


if __name__=='__main__':
    unittest.main()

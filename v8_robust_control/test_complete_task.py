"""Observation routing and actual-contact completion of the 2+1 task."""
import unittest
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import mujoco

from jugglebot.humanoid import cascade
from .complete_task import CompleteCascade, DEFAULT_SEED_PLANS, complete_record
from v7_three_ball_hold.end21 import EndingCascade
from v7_three_ball_hold.cascade21 import TwoPlusOneCascade
from .summarize_perception import reports, validate


class TestCompleteTask(unittest.TestCase):
    def test_default_full_task_seed_file_is_consumed(self):
        inputs=json.loads(DEFAULT_SEED_PLANS.read_text(encoding='utf-8'))
        c=CompleteCascade(throws=4)
        self.assertEqual(c.execution_warm['throw:left:47'],inputs['execution']['throw:left:47'])
        self.assertEqual(c.warm['throw:left:17'],inputs['forecast']['throw:left:17'])

    def test_record_identifies_the_selected_optimizer_input_file(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'candidate.json'
            path.write_text('{"forecast":{},"execution":{}}',encoding='utf-8')
            record=complete_record(None,30,40.,path)
            self.assertEqual(record['optimizer_inputs_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertIn('base_optimizer_inputs_sha256',record)
            self.assertEqual(record['optimizer_inputs_file'],path.as_posix())

    def test_planning_precision_is_shared_without_changing_raw_states(self):
        c=CompleteCascade(throws=4)
        self.assertTrue(c.normalize_reception_input)
        position=(.3123456781,-.2123456781,1.0123456781)
        velocity=(.123456781,.612345671,-1.123456781)
        raw=SimpleNamespace(valid=True,position_m=position,velocity_m_s=velocity,acceleration_m_s2=(0.,0.,-9.81))
        near=SimpleNamespace(valid=True,position_m=tuple(v+1e-14 for v in position),
                             velocity_m_s=tuple(v+1e-14 for v in velocity),acceleration_m_s2=(0.,0.,-9.81))
        with patch.object(c,'observed_ball',return_value=raw):
            first=c.reception_ball_state(0)
        with patch.object(c,'observed_ball',return_value=near):
            second=c.reception_ball_state(0)
        for a,b in zip(first,second): np.testing.assert_array_equal(a,b)
        self.assertEqual(raw.position_m,position)
        self.assertEqual(raw.velocity_m_s,velocity)
        self.assertEqual(c.conditions_summary()['reception_input_precision']['position_resolution_m'],1e-8)

    def test_reset_samples_match_the_two_plus_one_initial_state(self):
        c = CompleteCascade(throws=4, observation_mode='sampled')
        self.assertFalse(c.airborne)
        for i, (q, _) in enumerate(c.balls):
            state = c.ball_observer.estimate(i, c.data.time)
            np.testing.assert_allclose(state.position_m, c.data.qpos[q:q+3])
        self.assertIsNone(c.initial_support)
        self.assertFalse(c.ready_for_throw('right'))

    def test_startup_establishes_real_continuous_three_ball_support(self):
        c = CompleteCascade(throws=4)
        for _ in range(250):
            c.step()
        self.assertIsNotNone(c.initial_support)
        self.assertGreaterEqual(c.initial_support['continuous_support_s'], .05)
        self.assertTrue(all(b['contact'] for b in c.initial_support['balls'].values()))
        self.assertEqual(c.throw_count, 0)

    def test_ending_waits_for_valid_observation(self):
        c = CompleteCascade(throws=4, observation_mode='sampled')
        c.targets = {2:'right'}
        with patch.object(c, 'reception_ball_state', return_value=None) as read:
            c._maybe_start_end()
        read.assert_called_once_with(2)
        self.assertIsNone(c.end)

    def test_late_observation_cannot_build_a_negative_duration_ending(self):
        c = CompleteCascade(throws=4)
        c.targets = {2:'right'}
        off=c._goal_rotation() @ c.catch_target_local
        height=c.hands['right']['catch_p'][2]+off[2]
        state=(np.array([.3,-.1,height+.5*c.gravity_m_s2*.08**2]),np.zeros(3))
        with (patch.object(c,'reception_ball_state',return_value=state),
              patch('v7_three_ball_hold.end21.CartesianPalmMotion') as motion):
            c._maybe_start_end()
        motion.assert_not_called()
        self.assertIsNone(c.end)

    def test_v8_approach_margin_does_not_change_the_v7_positive_duration_rule(self):
        for factory, expected, starts in ((EndingCascade, .10, True),
                                          (CompleteCascade, .125, False)):
            c=factory(stop_after=4) if factory is EndingCascade else factory(throws=4)
            self.assertAlmostEqual(c.minimum_ending_horizon_s(),expected)
            if factory is CompleteCascade:
                self.assertAlmostEqual(c.conditions_summary()['complete_task']['ending_minimum_horizon_s'],expected)
            c.targets={1:'right'}
            height=c.hands['right']['catch_p'][2]+(c._goal_rotation() @ c.catch_target_local)[2]
            state=(np.array([.3,-.1,height+.5*c.gravity_m_s2*.11**2]),np.zeros(3))
            with (patch.object(c,'reception_ball_state',return_value=state),
                  patch('v7_three_ball_hold.end21.CartesianPalmMotion') as motion):
                c._maybe_start_end()
            self.assertEqual(motion.called,starts)
            self.assertEqual(c.end is not None,starts)

    def test_ending_limit_does_not_modify_other_instances(self):
        c = CompleteCascade(throws=4)
        original = cascade.MAX_ELBOW_OUT_M
        c.end = {}
        self.assertEqual(c.elbow_out_limit('right'), .10)
        self.assertEqual(cascade.MAX_ELBOW_OUT_M, original)
        other=EndingCascade(stop_after=4)
        other.end={}
        try:
            with (patch.object(other,'_end_control'), patch.object(other,'_end_monitor'),
                  patch.object(TwoPlusOneCascade,'step')):
                other.step()
            self.assertEqual(other.elbow_out_limit('right'), .12)
            self.assertEqual(cascade.MAX_ELBOW_OUT_M,original)
            self.assertEqual(c.elbow_out_limit('right'), .10)
        finally:
            cascade.MAX_ELBOW_OUT_M=original

    def test_ending_posture_bias_preserves_palm_task_to_first_order(self):
        c = CompleteCascade(throws=4)
        c.end = dict(impact=1., t_start=-1.)
        c.ENDING_ELBOW_SOFT_LIMIT_M=.01
        h=c.hands['right']; dof=h['kin'].dof_adr
        original=c.data.qpos.copy(), c.data.qvel.copy()
        target=c.data.qpos[h['kin'].qpos_adr].copy()
        biased=c.ending_posture_target(h,c.data,target)
        delta=biased-target
        jp,jr,je,js=(np.zeros((3,c.model.nv)) for _ in range(4))
        mujoco.mj_jacSite(c.model,c.data,jp,jr,h['seat'])
        elbow=c.model.body('right_elbow_link').id
        shoulder=c.model.body('right_shoulder_pitch_link').id
        mujoco.mj_jacBody(c.model,c.data,je,None,elbow)
        mujoco.mj_jacBody(c.model,c.data,js,None,shoulder)
        gradient=(np.sign(c.data.xpos[elbow,1])*je[1,dof]
                  -np.sign(c.data.xpos[shoulder,1])*js[1,dof])
        self.assertLess(gradient@delta,0.)
        np.testing.assert_allclose(np.vstack([jp[:,dof],jr[:,dof]])@delta,0.,atol=1e-8)
        self.assertLessEqual(np.linalg.norm(delta),c.ENDING_POSTURE_MAX_OFFSET_RAD+1e-12)
        np.testing.assert_array_equal(c.data.qpos,original[0])
        np.testing.assert_array_equal(c.data.qvel,original[1])

    def test_ending_posture_blends_from_the_start_of_the_approach(self):
        c = CompleteCascade(throws=4)
        c.end=dict(impact=1., t_start=0.)
        c.ENDING_ELBOW_SOFT_LIMIT_M=.01
        target=c.data.qpos[c.hands['right']['kin'].qpos_adr].copy()
        np.testing.assert_array_equal(c.ending_posture_target(c.hands['right'],c.data,target),target)
        c.data.time=.15
        biased=c.ending_posture_target(c.hands['right'],c.data,target)
        self.assertGreater(np.linalg.norm(biased-target),0.)
        self.assertLessEqual(np.linalg.norm(biased-target),.5*c.ENDING_POSTURE_MAX_OFFSET_RAD+1e-12)

    def test_pair_exemption_ends_at_first_confirmed_release(self):
        c = CompleteCascade(throws=4)
        for _ in range(200):
            c.step()
        self.assertTrue(c._pair_exempt())
        c.hands['right']['two_phase'] = False
        c.hands['right']['t_release'] = c.data.time
        self.assertFalse(c._pair_exempt())

    def test_final_hold_requires_uninterrupted_real_support(self):
        c = CompleteCascade(throws=4)
        c.end = dict(a=2, c=0, rel_first=dict(t=0., speed=2.))
        c.airborne = {2:4}
        c.targets = {2:'right'}
        c.catch_count = 3
        with patch.object(c, 'support_state', return_value=dict(ok=True, balls={})) as support:
            for _ in range(799):
                c._end_monitor()
            self.assertIsNone(c.end_result)
            self.assertNotIn(4, c.caught_events)
            support.return_value = dict(ok=False, balls={})
            c._end_monitor()
            self.assertEqual(c.final_stable_age, 0.)
            support.return_value = dict(ok=True, balls={})
            for _ in range(800):
                c._end_monitor()
            self.assertTrue(c.end_result['ok'])
            self.assertIn(4, c.caught_events)
            self.assertEqual(c.catch_count, 4)
            self.assertFalse(c.airborne)
            self.assertFalse(c.targets)

    def test_left_ball_uses_measured_relative_velocity(self):
        c = CompleteCascade(throws=4)
        _, v = c.balls[1]
        c.data.qvel[v] = 1.
        state = c.support_state({1:'left'})
        self.assertFalse(state['ok'])
        self.assertGreater(state['balls']['1']['relative_speed_m_s'], .9)

    def test_complete_record_fingerprints_consumed_task_assets(self):
        record = complete_record(None, 30, 40.)
        self.assertIn('v7_three_ball_hold/end21.py', record['source_sha256'])
        self.assertEqual(len(record['task_inputs_sha256']), 7)
        self.assertEqual(record['task_inputs_sha256']['v8_robust_control/complete_seed_plans.json'],
                         record['optimizer_inputs_sha256'])

    def test_invalid_end_parity_rejected(self):
        for throws in (0, 2, 5, 31):
            with self.assertRaises(ValueError):
                CompleteCascade(throws=throws)

    def test_infeasible_throw_stops_after_bounded_attempts(self):
        c = CompleteCascade(throws=4)
        c.last_plan_error = .03
        with patch.object(EndingCascade, 'plan_throw', return_value=None):
            for _ in range(c.MAX_THROW_PLAN_ATTEMPTS-1):
                c.plan_throw('right')
            with self.assertRaisesRegex(RuntimeError, 'NO_FEASIBLE_THROW'):
                c.plan_throw('right')
        self.assertEqual(c.failure_event['attempts'], 8)
        self.assertEqual(c.failure_event['reason'], 'NO_FEASIBLE_THROW')

    def complete_rows(self):
        from .test_perception_evaluation import TestPerceptionEvaluation
        rows = TestPerceptionEvaluation().records()
        rows[0]['task'] = 'complete'
        r = rows[1]['result']
        balls={str(i):dict(contact=True, relative_speed_m_s=.01, local_m=[0.,0.,.03],
                          height_m=.85, hand='left' if i==1 else 'right') for i in range(3)}
        r.update(task='three_ball_start_cascade_finish', throws=30, airborne_at_end=[],
                 initial_hold=dict(ok=True, continuous_support_s=.05, balls=deepcopy(balls)),
                 final_hold=dict(ok=True, continuous_support_s=.8, balls=balls))
        return rows

    def test_complete_report_requires_actual_final_support_evidence(self):
        rows = self.complete_rows()
        self.assertIn('三球连续 0.8 s', reports(rows)[0])
        for field, value in (('final_hold', None), ('airborne_at_end', [2]), ('throws', 31)):
            bad = deepcopy(rows)
            bad[1]['result'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(bad)
        for field, value in (('continuous_support_s', .79), ('balls', {})):
            bad = deepcopy(rows)
            bad[1]['result']['final_hold'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(bad)


if __name__ == '__main__':
    unittest.main()

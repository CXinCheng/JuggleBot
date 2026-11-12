"""Evaluation isolation, report integrity and command-line validation."""
import contextlib
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from jugglebot.robustness.controller import ObservedCascade
from jugglebot.robustness.evaluation import EvaluatedCascade
from jugglebot.robustness.perception import PerceptionConfig
from .evaluate_perception import cases
from .juggle import main
from .summarize_perception import reports, validate


class TestPerceptionEvaluation(unittest.TestCase):
    def test_telemetry_does_not_change_physics_or_sampling(self):
        config = PerceptionConfig(.001,.02,.1,5)
        plain = ObservedCascade(observation_mode='sampled',perception=config)
        measured = EvaluatedCascade(observation_mode='sampled',perception=config)
        plain.planning = measured.planning = False
        for _ in range(150):
            plain.step()
            measured.step()
            np.testing.assert_array_equal(plain.data.qpos,measured.data.qpos)
            np.testing.assert_array_equal(plain.data.qvel,measured.data.qvel)
            np.testing.assert_array_equal(plain.data.ctrl,measured.data.ctrl)
        self.assertEqual(plain.ball_observer.conditions(),measured.ball_observer.conditions())

    def test_private_rollouts_do_not_consume_transport_or_telemetry(self):
        c = EvaluatedCascade(observation_mode='sampled',perception=PerceptionConfig(.001,.02,.1,4))
        before = deepcopy(c.ball_observer.conditions()), c.estimation_summary()
        c.planning = False
        c.silent = True
        for _ in range(40):
            c.step()
        self.assertEqual((c.ball_observer.conditions(),c.estimation_summary()),before)

    def test_truth_video_telemetry_does_not_query_stale_truth_estimate(self):
        c = EvaluatedCascade()
        c.planning = False
        for _ in range(20):
            c.step()
        self.assertEqual(c.error_count,0)

    def test_all_dropped_frames_cannot_plan_a_truth_based_catch(self):
        c = EvaluatedCascade(observation_mode='sampled',perception=PerceptionConfig(dropout_probability=1.))
        result = c.run_continuous(1,1.,guard=True)
        self.assertFalse(result['success'])
        self.assertEqual(result['validated_flights'],0)
        self.assertEqual(c.ball_observer.transport.delivered_frames,0)
        self.assertFalse(any(e['kind']=='prepare' for e in c.events))
        self.assertEqual(c.error_count,0)

    def test_cli_invalid_disturbances_rejected_before_construction(self):
        for flags in (['--noise-mm','-1'],['--observation-mode','truth','--delay-ms','20'],
                      ['--dropout','nan'],['--seed','-1']):
            with self.subTest(flags=flags), patch.object(sys,'argv',['juggle',*flags]), \
                    patch('v7_robust_control.juggle.ObservedCascade') as constructor, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code,2)
                constructor.assert_not_called()

    def records(self):
        case = cases()[0]
        cfg = asdict(PerceptionConfig())
        transport = dict(cfg,sampled_frames=3,dropped_frames=0,delivered_frames=3,pending_frames=0,trace_sha256='a'*64)
        result = dict(success=True,guard=True,validated_flights=30,requested_throws=30,drops=0,ball_contacts=0,
                      grasp_assistance=False,grasp_constraints=0,
                      conditions=dict(ball_observation=dict(mode='sampled',sample_period_s=1/60,max_age_s=.05,perception=transport)))
        return [dict(kind='experiment',cases=[case],seeds=[0],requested_flights=30,observation_period_s=1/60,observation_max_age_s=.05),
                dict(kind='trial',case='baseline',seed=0,perception=cfg,result=result,
                     estimation=dict(position_rmse_m=0.,velocity_rmse_m_s=0.,invalid_samples=0)),
                dict(kind='complete',trials=1)]

    def test_reports_are_deterministic(self):
        rows = self.records()
        self.assertEqual(reports(rows),reports(deepcopy(rows)))
        self.assertIn('| baseline | 0 | 0 | 0 | 1/1 | 30.0 | 0 | 0 |',reports(rows)[0])

    def test_parallel_completion_order_does_not_change_report(self):
        original = self.records()
        original[0]['seeds'] = [0,1]
        original[-1]['trials'] = 2
        second = deepcopy(original[1])
        second['seed'] = second['perception']['seed'] = 1
        second['result']['conditions']['ball_observation']['perception']['seed'] = 1
        ordered = [original[0],original[1],second,original[-1]]
        shuffled = [original[0],second,original[1],original[-1]]
        self.assertEqual(reports(ordered),reports(shuffled))

    def test_reject_partial_duplicate_and_inconsistent_reports(self):
        original = self.records()
        variants = [original[:-1], [original[0],original[1],original[1],original[2]]]
        for field,value in (('guard',False),('validated_flights',29),('grasp_assistance',True)):
            rows = deepcopy(original)
            rows[1]['result'][field] = value
            variants.append(rows)
        rows = deepcopy(original)
        rows[1]['result']['conditions']['ball_observation']['perception']['delivered_frames'] = 2
        variants.append(rows)
        for rows in variants:
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                validate(rows)


if __name__ == '__main__':
    unittest.main()

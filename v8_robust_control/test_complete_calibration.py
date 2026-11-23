"""Calibration exports only controls that reached actual execution."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from .calibrate_complete import collect
from .complete_task import CompleteCascade


class TestCompleteCalibration(unittest.TestCase):
    def test_input_override_keeps_forecasts_and_live_guesses_separate(self):
        inputs=dict(forecast={'throw:left:1':[.2,-.8,3.2,0.]},
                    execution={'throw:right:2':[.1,.8,3.1,0.,.005]})
        c=CompleteCascade(throws=4,optimizer_inputs=inputs)
        self.assertEqual(c.warm['throw:left:1'],inputs['forecast']['throw:left:1'])
        self.assertEqual(c.execution_warm['throw:right:2'],inputs['execution']['throw:right:2'])
        self.assertIn('throw:right:0',c.execution_warm)
        self.assertNotEqual(c.execution_warm['throw:left:1'],inputs['forecast']['throw:left:1'])

    def test_only_executed_launches_and_prepared_zero_error_receptions_are_cached(self):
        inputs=dict(forecast={'throw:right:0':[0.,1.,3.,0.]},execution={})
        before=deepcopy(inputs)
        events=[
            dict(kind='plan',side='right',ball=2,time=.3,command=[.1,.8,3.5,0.]),
            dict(kind='stroke',side='right',ball=2,time=.3),
            dict(kind='release',side='right',ball=2,time=.48,count=1),
            dict(kind='plan',side='left',ball=1,time=.6,command=[.2,-.8,3.5,0.]),
            dict(kind='plan_rejected',side='left',ball=1,time=.605),
            dict(kind='return_plan',side='left',ball=2,time=.8,score=0.,parameters=[1.,2.,3.,4.,5.]),
            dict(kind='prepare',side='left',ball=2,time=.8),
            dict(kind='return_plan',side='right',ball=1,time=.9,score=.1,parameters=[5.,4.,3.,2.,1.]),
            dict(kind='prepare',side='right',ball=1,time=.9),
            dict(kind='return_plan',side='right',ball=1,time=1.,score=0.,tilt_scale=1.25,parameters=[5.,4.,3.,2.,1.]),
            dict(kind='prepare',side='right',ball=1,time=1.)]
        result=collect(SimpleNamespace(events=events),inputs)
        self.assertEqual(result['execution'],{'throw:right:0':[.1,.8,3.5,0.],
                                              'return:left:1':[1.,2.,3.,4.,5.]})
        self.assertEqual(result['forecast'],before['forecast'])
        self.assertEqual(inputs,before)


if __name__=='__main__':
    unittest.main()

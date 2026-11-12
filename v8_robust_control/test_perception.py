"""Causal frame transport, seeded perturbations and observer integration."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from jugglebot.robustness.observation import BallObservation, BallObserver
from jugglebot.robustness.perception import PerceptionConfig, FrameTransport


def frame(time):
    return tuple(BallObservation(i,time,time,(i+time,0.,1.)) for i in range(3))


class TestPerception(unittest.TestCase):
    def test_invalid_configuration(self):
        for kw in (dict(noise_std_m=-1),dict(delay_s=float('nan')),dict(dropout_probability=1.1),
                   dict(seed=-1),dict(seed=1.2)):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                PerceptionConfig(**kw)

    def test_fifo_and_actual_receipt_time(self):
        t = FrameTransport(PerceptionConfig(delay_s=.0205))
        t.submit(frame(0.))
        self.assertEqual(t.receive(.020), ())
        arrived = t.receive(.021)[0]
        self.assertEqual([o.sample_time_s for o in arrived], [0.]*3)
        self.assertEqual([o.received_time_s for o in arrived], [.021]*3)
        t.submit(frame(.03))
        t.submit(frame(.04))
        arrivals = t.receive(.1)
        self.assertEqual([f[0].sample_time_s for f in arrivals],[.03,.04])
        self.assertFalse(t.queue)

    def test_deliver_between_sample_ticks_and_expire(self):
        observer = BallObserver([(0,0)],mode='sampled',sample_period_s=.01,max_age_s=.02,
                                perception=PerceptionConfig(delay_s=.005))
        data = SimpleNamespace(time=0.,qpos=np.array([0.,0.,1.]))
        self.assertFalse(observer.update(data))
        self.assertIsNone(observer.estimate(0,0.))
        data.time = .005
        self.assertTrue(observer.update(data))
        self.assertEqual(observer.observations[0].received_time_s,.005)
        self.assertEqual(observer.transport.sampled_frames,1)
        for stamp in (.01,.015,.02,.025):
            data.time = stamp
            data.qpos[0] = stamp
            observer.update(data)
        self.assertTrue(observer.estimate(0,.025).valid)
        self.assertFalse(observer.estimate(0,.05).valid)

    def test_seeded_noise_and_dropout_replay(self):
        cfg = PerceptionConfig(.003,.02,.25,7)
        def run(seed):
            t = FrameTransport(replace(cfg,seed=seed))
            result = []
            for i in range(100):
                stamp = i*.017
                t.submit(frame(stamp))
                result.extend(t.receive(stamp))
            result.extend(t.receive(2.))
            return result,t.conditions()
        self.assertEqual(run(7),run(7))
        self.assertNotEqual(run(7),run(8))
        rows, stats = run(7)
        self.assertEqual(stats['sampled_frames'], stats['delivered_frames']+stats['dropped_frames'])
        self.assertTrue(0 < stats['dropped_frames'] < 100)
        self.assertTrue(all(len(f)==3 and all(o.velocity_m_s is None for o in f) for f in rows))

    def test_reset_replays_stream_and_clears_pending(self):
        t = FrameTransport(PerceptionConfig(.01,.1,.2,2))
        t.submit(frame(0.))
        first = t.conditions()
        t.reset()
        t.submit(frame(0.))
        self.assertEqual(first,t.conditions())
        t.reset()
        self.assertEqual(t.receive(0.),())
        self.assertEqual(t.conditions()['sampled_frames'],0)

    def test_no_disturbance_is_identity(self):
        t = FrameTransport(PerceptionConfig())
        original = frame(.1)
        t.submit(original)
        self.assertEqual(t.receive(.1),(original,))

    def test_complete_dropout_never_creates_measurements(self):
        t = FrameTransport(PerceptionConfig(dropout_probability=1.))
        for i in range(5):
            t.submit(frame(float(i)))
            self.assertEqual(t.receive(float(i)),())
        self.assertEqual(t.conditions()['dropped_frames'],5)

    def test_rejects_noncausal_input(self):
        t = FrameTransport(PerceptionConfig())
        t.submit(frame(.1))
        t.receive(.2)
        with self.assertRaises(ValueError):
            t.submit(frame(.15))
        with self.assertRaises(ValueError):
            t.receive(.1)
        with self.assertRaises(ValueError):
            t.submit((replace(frame(.3)[0],velocity_m_s=(1.,0.,0.)),))

    def test_truth_rejects_disturbance(self):
        with self.assertRaises(ValueError):
            BallObserver([(0,0)],perception=PerceptionConfig(noise_std_m=.001))

    def test_observer_reset_replays_noise_without_truth_velocity(self):
        o = BallObserver([(0,0)],mode='sampled',perception=PerceptionConfig(.001,0.,0.,3))
        data = SimpleNamespace(time=0.,qpos=np.array([0.,0.,1.]))
        o.update(data)
        first = o.observations
        o.reset()
        o.update(data)
        self.assertEqual(first,o.observations)

    def test_gaussian_scale_and_dropout_have_independent_streams(self):
        full = FrameTransport(PerceptionConfig(.003,0.,0.,11))
        lossy = FrameTransport(PerceptionConfig(.003,0.,.2,11))
        errors = []
        for i in range(1000):
            stamp = float(i)
            original = frame(stamp)
            full.submit(original)
            lossy.submit(original)
            reference = full.receive(stamp)[0]
            arrived = lossy.receive(stamp)
            if arrived:
                self.assertEqual(arrived[0],reference)
            errors.extend(np.asarray(reference[j].position_m)-original[j].position_m for j in range(3))
        error = np.asarray(errors)
        np.testing.assert_allclose(error.std(axis=0),.003,rtol=.08)
        self.assertTrue(np.all(np.abs(error.mean(axis=0)) < .0002))


if __name__ == '__main__':
    unittest.main()

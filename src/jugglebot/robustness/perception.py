"""Seeded position measurement noise and causal frame transport."""
from collections import deque
from dataclasses import dataclass, replace
import hashlib
import json
import math

import numpy as np


@dataclass(frozen=True)
class PerceptionConfig:
    noise_std_m: float = 0.
    delay_s: float = 0.
    dropout_probability: float = 0.
    seed: int = 0

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in (self.noise_std_m, self.delay_s)):
            raise ValueError('Noise standard deviation and delay must be finite and nonnegative')
        if not math.isfinite(self.dropout_probability) or not 0 <= self.dropout_probability <= 1:
            raise ValueError('Dropout probability must be between zero and one')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('Perception seed must be a nonnegative integer')

    @property
    def enabled(self):
        return bool(self.noise_std_m or self.delay_s or self.dropout_probability)


class FrameTransport:
    """Whole-frame independent dropout, per-axis Gaussian noise and fixed FIFO delay.

    Receipt is the actual polling time at delivery, rounded up by physics ticks.
    Separate random streams keep noise draws independent of dropout settings.
    """
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        noise_seed, drop_seed = np.random.SeedSequence(self.config.seed).spawn(2)
        self.noise_rng = np.random.default_rng(noise_seed)
        self.drop_rng = np.random.default_rng(drop_seed)
        self.queue = deque()
        self.last_sample = self.last_poll = None
        self.sampled_frames = self.dropped_frames = self.delivered_frames = 0
        self.max_queue_depth = 0
        self._digest = hashlib.sha256()

    def _record(self, event):
        self._digest.update((json.dumps(event, separators=(',', ':'), allow_nan=False)+'\n').encode())

    def submit(self, observations):
        if not observations:
            raise ValueError('A frame must contain observations')
        stamp = observations[0].sample_time_s
        if any(o.sample_time_s != stamp or o.velocity_m_s is not None for o in observations):
            raise ValueError('Transport accepts position-only observations from one frame')
        if self.last_sample is not None and stamp <= self.last_sample:
            raise ValueError('Frame sample times must strictly increase')
        if self.last_poll is not None and stamp < self.last_poll:
            raise ValueError('Cannot submit a historical frame')
        self.last_sample = stamp
        self.sampled_frames += 1
        # Draw for every frame, including dropped frames, to pair noise streams.
        noise = self.noise_rng.normal(0., self.config.noise_std_m, (len(observations), 3))
        dropped = self.drop_rng.random() < self.config.dropout_probability
        frame = tuple(replace(o, position_m=tuple(np.asarray(o.position_m)+n))
                      for o, n in zip(observations, noise)) if self.config.noise_std_m else tuple(observations)
        self._record(['sample', stamp, dropped, [o.position_m for o in frame]])
        if dropped:
            self.dropped_frames += 1
        else:
            self.queue.append((stamp+self.config.delay_s, frame))
            self.max_queue_depth = max(self.max_queue_depth, len(self.queue))

    def receive(self, now_s):
        if not math.isfinite(now_s) or now_s < 0 or (self.last_poll is not None and now_s < self.last_poll):
            raise ValueError('Transport polling time must be finite and monotonic')
        self.last_poll = now_s
        delivered = []
        while self.queue and self.queue[0][0] <= now_s:
            _, frame = self.queue.popleft()
            delivered.append(tuple(replace(o, received_time_s=now_s) for o in frame))
            self.delivered_frames += 1
            self._record(['receive', frame[0].sample_time_s, now_s])
        return tuple(delivered)

    def conditions(self):
        return dict(noise_std_m=self.config.noise_std_m, delay_s=self.config.delay_s,
                    dropout_probability=self.config.dropout_probability, seed=self.config.seed,
                    noise_model='independent_gaussian_per_axis_per_frame',
                    dropout_model='independent_whole_frame', delay_model='fixed_fifo',
                    sampled_frames=self.sampled_frames, dropped_frames=self.dropped_frames,
                    delivered_frames=self.delivered_frames, pending_frames=len(self.queue),
                    max_queue_depth=self.max_queue_depth, trace_sha256=self._digest.hexdigest())

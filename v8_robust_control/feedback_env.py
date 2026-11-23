"""Latched gain requests with contact permissions evaluated at servo rate."""
from pathlib import Path

import numpy as np

from .cascade_env import CascadeEnv, runtime_sources as reference_sources
from .compare_residual import digest


def runtime_sources():
    return dict(reference_sources(), **{'v8_robust_control/feedback_env.py': digest(Path(__file__))})


class FeedbackCascadeEnv(CascadeEnv):
    """Keep requested gains across a contact transition between policy ticks.

    The reference environment masks requests at 50 Hz. A masked zero means
    default gain, not disabled feedback. When a contact becomes eligible at
    1 kHz, that default can override the policy's last disable request.
    Latched requests preserve the policy choice; the inherited target methods
    still check valid estimates, actual palm contact and release permissions.
    """
    ENVIRONMENT_VERSION = 6
    CONTROLLER_NAME = 'latched_feedback'

    def reset(self, *, seed=None, options=None):
        self.gain_request = np.zeros(6)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if not self.active:
            raise RuntimeError('Reset before stepping or after termination')
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.all(np.isfinite(action)):
            raise ValueError('Expected six finite gain requests')
        self.gain_request = np.clip(action, -1., 1.).copy()
        return super().step(action)

    def _targets(self, action):
        # Contact permissions gate motor effects, not the requested gain value.
        super()._targets(self.gain_request)

    def _info(self, reason):
        return dict(super()._info(reason), controller=self.CONTROLLER_NAME,
                    gain_request=self.gain_request.tolist())


class GuardedCascadeEnv(FeedbackCascadeEnv):
    """Zero-neutral gains and evidence-gated departure from the ending prior."""
    ENVIRONMENT_VERSION = 7
    CONTROLLER_NAME = 'guarded_feedback'
    ACTION_LAYOUT = 'positive squared gains; zero disables; initial grasp centering'
    ENDING_SIGMA_MULTIPLIER = 3.
    ENDING_EVIDENCE_FRAMES = 3

    def reset(self, *, seed=None, options=None):
        self.ending_evidence_count = 0
        self.ending_evidence_stamp = None
        self.innovation_evidence = {}
        return super().reset(seed=seed, options=options)

    def _credible_innovation(self, ball, side=None):
        key = (ball, side, self.flight_ids[ball] if hasattr(self, 'flight_ids') else None)
        tracker = self.trackers[ball]
        estimate = tracker.estimate(float(self.data.time))
        index = self._reference_index()
        if estimate is None or not estimate.valid or not self.estimate_valid[index, ball]:
            self.innovation_evidence.pop(key, None)
            return False
        error = np.asarray(estimate.position_m)-self.estimate_reference[index, ball, :3]
        if side is not None:
            error = error-(self.data.site_xpos[self.hands[side]['seat']]-self.palm_reference[index, side, :3])
        stamp, direction, count = self.innovation_evidence.get(key, (None, None, 0))
        if stamp != estimate.sample_time_s:
            sigma = float(tracker.uncertainty(float(self.data.time))[0])
            axis = int(np.argmax(np.abs(error)))
            current = (axis, float(np.sign(error[axis]))) if abs(error[axis]) > 3*sigma else None
            count = count+1 if current is not None and current == direction else int(current is not None)
            self.innovation_evidence[key] = (estimate.sample_time_s, current, count)
        return count >= 3

    def _xy_innovation(self, side):
        value = super()._xy_innovation(side)
        if value is None:
            return None
        held = [ball for ball in range(self.count) if self.ball_contacts[ball, side]]
        candidates = held or [ball for ball in range(self.count)
                             if hasattr(self, 'flight_ids') and self.flight_ids[ball] is not None
                             and 1-self.flight_sources[ball] == side]
        return value if any(self._credible_innovation(ball, side if held else None) for ball in candidates) else None

    def _wrapping_ball(self, side):
        selected = super()._wrapping_ball(side)
        return selected if selected is not None and self._credible_innovation(selected[0], side) else None

    def _targets(self, action):
        positive = np.maximum(self.gain_request, 0.)**2
        # The quadratic map provides a continuous quiet region around zero.
        # Motor permissions and original magnitude/rate bounds remain active.
        CascadeEnv._targets(self, np.r_[2*positive[:4]-1., 2*positive[4:]-1.])

    def _wrapping_target(self):
        target = super()._wrapping_target()
        if not hasattr(self, 'flight_sources'):
            return target
        index = self._reference_index()
        for side, hand in enumerate(self.hands):
            if side in self.flight_sources or self._releasing(side):
                continue
            errors = []
            for ball, tracker in enumerate(self.trackers):
                if self.flight_ids[ball] is not None or not self.ball_contacts[ball, side]:
                    continue
                estimate = tracker.estimate(float(self.data.time))
                if (estimate is None or not estimate.valid or not self.estimate_valid[index, ball]
                        or not self._credible_innovation(ball, side)):
                    continue
                nominal = self.estimate_reference[index, ball, :3]-self.palm_reference[index, side, :3]
                measured = np.asarray(estimate.position_m)-self.data.site_xpos[hand['seat']]
                errors.append(measured-nominal)
            if errors:
                rotation = self.data.site_xmat[hand['seat']].reshape(3, 3)
                correction = hand['fingers'].wrap_map@(rotation.T@np.mean(errors, axis=0))
                target[side] = np.clip(-.5*self.wrapping_gain[side]*correction,
                                       -self.MAX_WRAPPING_OFFSET_RAD, self.MAX_WRAPPING_OFFSET_RAD)
        return target

    def permissions(self):
        allowed = super().permissions()
        if hasattr(self, 'flight_sources'):
            for side in (0, 1):
                if side not in self.flight_sources and not self._releasing(side):
                    allowed[4+side] |= any(self.ball_contacts[ball, side] and self._credible_innovation(ball, side)
                                          for ball in range(self.count))
        return allowed

    def _ending_control(self):
        if self.task != 'complete' or self.next_release != self.target_throws:
            return
        if self.controller.end is None:
            ball = next((ball for ball in range(self.count)
                         if self.flight_ids[ball] == self.target_throws), None)
            if ball is None:
                return
            tracker = self.trackers[ball]
            estimate = tracker.estimate(float(self.data.time))
            index = self._reference_index()
            if estimate is None or not estimate.valid or not self.estimate_valid[index, ball]:
                self.ending_evidence_count = 0
                return
            if estimate.sample_time_s != self.ending_evidence_stamp:
                self.ending_evidence_stamp = estimate.sample_time_s
                sigma = float(tracker.uncertainty(float(self.data.time))[0])
                error = np.linalg.norm(np.asarray(estimate.position_m)-self.estimate_reference[index, ball, :3])
                threshold = self.ENDING_SIGMA_MULTIPLIER*np.sqrt(3.)*sigma
                self.ending_evidence_count = self.ending_evidence_count+1 if error > threshold else 0
            if self.ending_evidence_count < self.ENDING_EVIDENCE_FRAMES:
                return
        super()._ending_control()

    def _info(self, reason):
        return dict(super()._info(reason), ending_evidence_frames=self.ending_evidence_count,
                    gain_mapping='positive squared gain requests; zero disables feedback')

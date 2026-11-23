"""Contact-aware position-only observations for event task decisions."""
from dataclasses import replace

import mujoco
import numpy as np

from jugglebot.robustness.observation import BallObserver
from .reception_control import BallisticPositionTracker
from .transfer_observation import SupportPositionTracker


class EventPositionEstimator:
    """Support tracking during contact; fresh ballistic fits after release.

    Contact and palm kinematics are sensor inputs, not ball velocity priors.
    Delayed positions retain their sampling timestamps and phase membership.
    """
    def __init__(self, ball_id, gravity, physics_dt, max_age_s):
        self.ball_id = ball_id
        self.gravity = np.asarray(gravity).copy()
        self.physics_dt,self.max_age_s = physics_dt,max_age_s
        self.reset()

    def reset(self):
        self.support = SupportPositionTracker(self.gravity,self.physics_dt,max_age_s=self.max_age_s)
        self.flight = BallisticPositionTracker(self.gravity,self.physics_dt,max_age_s=self.max_age_s)
        self._last = None
        self._phase_start = 0.
        self._side = None
        self._support_time = None

    def notify_support(self, now_s, positions, velocities, side):
        if side!=self._side:
            self._phase_start = now_s
            self.flight.history.clear()
        self._side,self._support_time = side,now_s
        self.support.notify_support(now_s,positions,velocities,side)

    def update(self, observation):
        if observation.ball_id!=self.ball_id:
            raise ValueError('Observation belongs to another ball')
        if self._last is not None and (observation.sample_time_s<=self._last.sample_time_s
                or observation.received_time_s<self._last.received_time_s):
            return False
        if (not observation.valid or self._last is not None
                and observation.sample_time_s-self._last.sample_time_s>self.max_age_s):
            # Reacquisition needs a new fit; proprioception cannot refresh age.
            previous = self.support
            self.support = SupportPositionTracker(self.gravity,self.physics_dt,max_age_s=self.max_age_s)
            self.support.hand_times = previous.hand_times.copy()
            self.support.hand_samples = previous.hand_samples.copy()
            self.support.events = previous.events.copy()
            self.support.in_contact = previous.in_contact
            self.flight.history.clear()
            self._phase_start = self._support_time if self._support_time is not None else observation.sample_time_s
        self._last = observation
        if not observation.valid:
            return True
        self.support.update(observation)
        if self._side is None and observation.sample_time_s>=self._phase_start-1e-12:
            self.flight.update(observation)
        return True

    def estimate(self, now_s):
        if not np.isfinite(now_s):
            raise ValueError('Query time must be finite')
        if now_s<0. or self._support_time is not None and now_s<self._support_time:
            raise ValueError('Cannot query before observed proprioception')
        if self._last is not None and now_s<self._last.received_time_s:
            raise ValueError('Cannot use an observation before receipt')
        tracker = self.flight if self._side is None else self.support
        state = tracker.estimate(now_s)
        return None if state is None else replace(state,ball_id=self.ball_id)


class EventBallObserver(BallObserver):
    """Keep the existing sampling/transport clock and replace only estimation."""
    def __init__(self, ball_addresses, *, model, seat_sites, ball_geoms, hand_geoms, **kwargs):
        self.model = model
        self.seat_sites = tuple(seat_sites)
        self.ball_geoms = tuple(ball_geoms)
        self.hand_geoms = tuple(frozenset(group) for group in hand_geoms)
        super().__init__(ball_addresses,mode='sampled',**kwargs)
        self.estimators = tuple(EventPositionEstimator(i,model.opt.gravity,self.position_step_s,self.max_age_s)
                                for i in range(len(self.source.ball_addresses)))

    def update(self, data):
        now = float(data.time)
        if self._last_update_s is None or now>self._last_update_s:
            positions = np.array([data.site_xpos[s] for s in self.seat_sites])
            velocities = []
            for site in self.seat_sites:
                spatial = np.zeros(6)
                mujoco.mj_objectVelocity(self.model,data,mujoco.mjtObj.mjOBJ_SITE,site,spatial,0)
                velocities.append(spatial[3:].copy())
            support = [None]*len(self.ball_geoms)
            for contact in data.contact:
                if contact.dist>1e-5 or contact.efc_address<0:
                    continue
                for ball,geom in enumerate(self.ball_geoms):
                    other = contact.geom2 if contact.geom1==geom else contact.geom1 if contact.geom2==geom else -1
                    for side,group in enumerate(self.hand_geoms):
                        if other in group:
                            support[ball] = side
            for estimator,side in zip(self.estimators,support):
                estimator.notify_support(now,positions,velocities,side)
        return super().update(data)

    def estimate(self, ball_id, now_s):
        if not 0<=ball_id<len(self.estimators):
            raise IndexError(ball_id)
        # Both reused trackers already implement semi-implicit endpoint velocity.
        # Applying BallObserver's quadratic half-step correction again is wrong.
        return self.estimators[ball_id].estimate(now_s)

    def conditions(self):
        result = super().conditions()
        result.update(estimator='contact_support_and_ballistic_fit',history_positions=8,
            contact_measurement_std_floor_m=.003,
            release_history='positions sampled after last measured hand contact',
            support_input='hand contact and palm proprioception; no ball velocity')
        return result


def make_event_observer(controller, estimator, *, perception=None, sample_period_s=1/60, max_age_s=.05):
    kwargs = dict(position_step_s=float(controller.model.opt.timestep),perception=perception,
                  sample_period_s=sample_period_s,max_age_s=max_age_s)
    if estimator=='difference':
        return BallObserver(controller.balls,mode='sampled',**kwargs)
    if estimator!='filtered':
        raise ValueError('Estimator must be difference or filtered')
    sides = ('right','left')
    return EventBallObserver(controller.balls,model=controller.model,
        seat_sites=[controller.hands[s]['seat'] for s in sides],ball_geoms=controller.ball_geom_ids,
        hand_geoms=[controller.hand_geoms[s] for s in sides],**kwargs)

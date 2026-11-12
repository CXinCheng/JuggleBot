"""v7 observation adapter over the v6 contact-physics reference controller."""
import numpy as np
import mujoco

from jugglebot.humanoid.cascade import (
    ContactCascade, MAX_RECEPTION_TILT_SCALE, RECEPTION_FINE_DIFFERENCE,
    RECEPTION_TIMELINESS_WINDOW_S, RECEPTION_MIN_APPROACH_S, RECEPTION_MAX_APPROACH_S,
    SUPPORTED_BALL_CENTER_OFFSET_M, RECEIVE_TANGENTIAL_VELOCITY_RATIO,
    RECEIVE_VELOCITY_RATIO, RECEIVE_INWARD_BANK, RECEIVE_PALM_BACK_TILT,
)
from jugglebot.humanoid.contact_control import EMPTY_RETURN_TILT_DEG, CartesianPalmMotion
from .observation import BallObserver, DEFAULT_OBSERVATION_PERIOD_S, DEFAULT_OBSERVATION_MAX_AGE_S

RECEPTION_INPUT_DECIMALS = 8
MAX_THROW_HISTORY_GUESSES = 8


class ObservedCascade(ContactCascade):
    def __init__(self, *args, observation_mode='truth',
                 observation_period_s=DEFAULT_OBSERVATION_PERIOD_S,
                 observation_max_age_s=DEFAULT_OBSERVATION_MAX_AGE_S,
                  normalize_reception_input=False, early_recovery=True, perception=None, **kwargs):
        super().__init__(*args, **kwargs)
        # The inherited split step is used only with one-step integrators.
        # A generic observer can use position_step_s=0 with full RK4 stepping.
        integrator = self.model.opt.integrator
        if integrator not in (mujoco.mjtIntegrator.mjINT_EULER, mujoco.mjtIntegrator.mjINT_IMPLICIT,
                              mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
            raise ValueError('The split-step contact controller requires a one-step integrator')
        position_step_s = float(self.model.opt.timestep)
        self.ball_observer = BallObserver(self.balls, mode=observation_mode,
                                         sample_period_s=observation_period_s,
                                         max_age_s=observation_max_age_s,
                                          position_step_s=position_step_s, perception=perception)
        self.ball_observer.update(self.data)
        self.normalize_reception_input = normalize_reception_input
        self._use_execution_forecasts = False
        self.forecast_retry_count = 0
        self._refining_reception = False
        self.fine_reception_retry_count = 0
        self.early_recovery_enabled = bool(early_recovery)
        self.early_recovery_active = False

    def observed_ball(self, ball_id):
        """Private forecasts never advance live observation history."""
        if self.silent:
            return self.ball_observer.source.truth(self.data, ball_id)
        self.ball_observer.update(self.data)
        return self.ball_observer.estimate(ball_id, float(self.data.time))

    def additional_throw_guesses(self, side):
        """Reuse executed motion guesses; every guess is re-certified in physics."""
        latest_plan = None
        executed = []
        for event in self.events:
            if event['side'] != side:
                continue
            if event['kind'] == 'plan':
                latest_plan = event
            elif (event['kind'] == 'stroke' and latest_plan is not None
                  and latest_plan['time'] == event['time']
                  and latest_plan['ball'] == event['ball']):
                executed.append(tuple(latest_plan['command'][:4]))
        guesses = []
        seen = set()
        for command in reversed(executed):
            if command in seen:
                continue
            seen.add(command)
            guesses.append(np.asarray(command, dtype=float))
            if len(guesses) == MAX_THROW_HISTORY_GUESSES:
                break
        return tuple(guesses)

    def reception_ball_state(self, ball_id):
        estimate = self.observed_ball(ball_id)
        if estimate is None or not estimate.valid:
            return None
        # A two-position secant is a valid linear estimate, but cannot supply
        # the endpoint velocity convention of an accelerating free ball.
        # Wait for the complete position-only model before live reception.
        if (not self.silent and self.ball_observer.mode == 'sampled'
                and estimate.acceleration_m_s2 is None):
            return None
        state = np.asarray(estimate.position_m), np.asarray(estimate.velocity_m_s)
        if self.normalize_reception_input:
            # The position-only reconstruction and a direct truth read can
            # differ at floating-point roundoff. Normalize both live and trial
            # inputs well below physical precision, before trajectory search.
            # Measurements, estimates and simulator state remain unrounded.
            return tuple(np.round(vector, RECEPTION_INPUT_DECIMALS) for vector in state)
        return state

    def throw_release_lead_candidates(self, side, release_lead):
        # A calibrated early opening can fail after a different physical catch.
        # Try moving it toward the stroke endpoint, retaining the best motion.
        return (.5*release_lead, 0.) if release_lead else ()

    def _late_recovery_needed(self):
        for hand in self.hands.values():
            approach = hand.get('approach_joint')
            if (hand.get('capture_confirmed', False) and hand.get('held') is not None
                    and hand.get('mode') in ('hold', 'throw', 'return') and approach is not None
                    and hand.get('contact_time', -1.) >= 0.
                    and hand['contact_time']-approach.t1 >= RECEPTION_TIMELINESS_WINDOW_S):
                return True
        return False

    def _early_recovery_reference(self, hand, data):
        side = hand['kin'].side
        for ball, receiver in self.targets.items():
            if receiver != side:
                continue
            state = self.reception_ball_state(ball)
            if state is None:
                continue
            position, velocity = state
            disc = velocity[2]**2 + 2*self.gravity_m_s2*(
                position[2]-hand['catch_p'][2]-SUPPORTED_BALL_CENTER_OFFSET_M)
            duration = (velocity[2]+np.sqrt(max(0., disc)))/self.gravity_m_s2
            if not RECEPTION_MIN_APPROACH_S < duration < RECEPTION_MAX_APPROACH_S:
                continue
            point = hand['catch_p'].copy()
            point[:2] = position[:2]+velocity[:2]*duration
            end_velocity = velocity.copy()*RECEIVE_TANGENTIAL_VELOCITY_RATIO
            end_velocity[2] = (velocity[2]-self.gravity_m_s2*duration)*RECEIVE_VELOCITY_RATIO
            axis = np.array([-RECEIVE_PALM_BACK_TILT,
                             RECEIVE_INWARD_BANK if side == 'right' else -RECEIVE_INWARD_BANK, 1.])
            axis /= np.linalg.norm(axis)
            mujoco.mj_copyData(hand['kin'].data, self.model, data)
            motion = CartesianPalmMotion(
                hand['kin'], float(data.time), duration, data.site_xpos[hand['seat']].copy(),
                self.spatial(hand)[3:].copy(), data.qpos[hand['kin'].qpos_adr].copy(),
                point, end_velocity, axis=axis, nominal=hand['q'], empty_arc=True)
            return motion, ball
        return None

    def follow_palm(self, hand, data, motion, time, target):
        pending = hand.get('release_pending')
        eligible = (getattr(self, 'early_recovery_enabled', False) and pending is not None
                    and hand.get('held') is not None and hand.get('mode') in ('throw', 'return')
                    and not self.ball_contact(hand['held'], hand['kin'].side))
        if not eligible:
            hand.pop('early_recovery', None)
            hand.pop('early_recovery_stamp', None)
            return super().follow_palm(hand, data, motion, time, target)
        if not self.early_recovery_active:
            # A trial inherits the live recovery policy. Hypothetical late
            # catches must not switch policy within a candidate rollout.
            if self.silent or not self._late_recovery_needed():
                return super().follow_palm(hand, data, motion, time, target)
            # Latch until a new controller is created; do not switch the motion
            # phase back and forth after each catch. Flight counting is unchanged.
            self.early_recovery_active = True
            self.log('early_recovery_enabled', hand['kin'].side, hand['held'], reason='late_confirmed_catch')
        if hand.get('early_recovery_stamp') != pending:
            reference = self._early_recovery_reference(hand, data)
            if reference is None:
                return super().follow_palm(hand, data, motion, time, target)
            hand['early_recovery'], incoming = reference
            hand['early_recovery_stamp'] = pending
            self.log('early_recovery', hand['kin'].side, incoming, outgoing_ball=hand['held'],
                     separation_time=float(pending), impact_time=float(hand['early_recovery'].t1))
        previous_mode = hand['mode']
        hand['mode'] = 'return'
        try:
            return super().follow_palm(hand, data, hand['early_recovery'], float(data.time), target)
        finally:
            hand['mode'] = previous_mode

    def reception_tilt_scale_candidates(self, side, diagnostics):
        failure = diagnostics.get('first_forbidden_contact') or {}
        if (not all(diagnostics.get(field, False) for field in
                    ('centered', 'timely', 'captured', 'ready_for_next_throw'))
                or failure.get('reason') != 'POSE_LIMIT'
                or failure.get('part') != side+'_wrist_pitch_joint'
                or failure.get('quantity') != 'wrist_pitch_rad'):
            return ()
        return (1.25, MAX_RECEPTION_TILT_SCALE)

    def reception_difference_steps(self, diagnostics):
        if getattr(self, '_refining_reception', False):
            return RECEPTION_FINE_DIFFERENCE
        return super().reception_difference_steps(diagnostics)

    def _latest_reception_plan(self, side):
        return next((event for event in reversed(self.events)
                     if event['kind'] == 'return_plan' and event['side'] == side
                     and event['ball'] == self.hands[side]['incoming']), None)

    def _refine_reception(self, side, event):
        original_inputs = self.execution_warm
        previous = getattr(self, '_refining_reception', False)
        self.execution_warm = dict(original_inputs)
        self.execution_warm[f'return:{side}:{self.throw_count}'] = np.asarray(event['parameters'], dtype=float).copy()
        self._refining_reception = True
        self.fine_reception_retry_count = getattr(self, 'fine_reception_retry_count', 0) + 1
        self.failure_event = None
        self.log('reception_refinement', side, self.hands[side]['incoming'], finite_difference='fine')
        try:
            return super().plan_reception(side)
        finally:
            self.execution_warm = original_inputs
            self._refining_reception = previous

    def step(self):
        if not self.silent:
            self.ball_observer.update(self.data)
        if not (self.silent and getattr(self, '_use_execution_forecasts', False)):
            return super().step()
        # Base forecasts retain explicit forecast inputs. When one is absent,
        # use the same indexed initial guess that live throw planning uses,
        # before falling back to the previous (possibly unrelated) launch.
        commands = self.commands
        self.commands = {side: self.execution_warm.get(f'throw:{side}:{self.throw_count}', command)
                         for side, command in commands.items()}
        try:
            return super().step()
        finally:
            self.commands = commands

    def plan_reception(self, side):
        try:
            return super().plan_reception(side)
        except RuntimeError:
            failure = self.failure_event
            if (self.silent or failure is None
                    or failure.get('reason') != 'NO_FEASIBLE_RECEPTION'):
                raise
        # A receiver can be caught and ready, yet fail certification because
        # the OTHER hand's stale forecast violates a pose limit. Re-certify
        # once with indexed throw guesses; no safety/settling limit changes.
        self.forecast_retry_count = getattr(self, 'forecast_retry_count', 0) + 1
        self.log('forecast_retry', side, self.hands[side]['incoming'],
                 policy='indexed_execution_fallback')
        previous = getattr(self, '_use_execution_forecasts', False)
        self._use_execution_forecasts = True
        self.failure_event = None
        try:
            try:
                return super().plan_reception(side)
            except RuntimeError:
                event = self._latest_reception_plan(side)
                if (self.failure_event is not None and self.failure_event.get('reason') == 'NO_FEASIBLE_RECEPTION'
                        and event is not None and event['diagnostics'].get('captured', False)):
                    return self._refine_reception(side, event)
                raise
        except RuntimeError:
            if self.failure_event is not None and self.failure_event.get('reason') == 'NO_FEASIBLE_RECEPTION':
                event = self._latest_reception_plan(side)
                if event is not None:
                    self.failure_event['reception_plan'] = {
                        key: event[key] for key in ('score', 'parameters', 'diagnostics')}
            raise
        finally:
            self._use_execution_forecasts = previous

    def conditions_summary(self):
        conditions = super().conditions_summary()
        conditions['ball_observation'] = self.ball_observer.conditions()
        conditions['ball_observation']['reception_requires_acceleration'] = self.ball_observer.mode == 'sampled'
        resolution = 10.**-RECEPTION_INPUT_DECIMALS if self.normalize_reception_input else None
        conditions['reception_input_precision'] = dict(position_resolution_m=resolution,
                                                       velocity_resolution_m_s=resolution)
        conditions['additional_throw_guesses'] = dict(source='recent_executed_same_hand_launches',
                                                     max_guesses=MAX_THROW_HISTORY_GUESSES,
                                                     release_timing='current_indexed_input',
                                                     acceptance='unchanged_full_physics_certificate')
        conditions['reception_forecast'] = dict(
            initial='explicit_forecast_or_previous_command',
            retry='explicit_forecast_or_indexed_execution_or_previous_command',
            max_retries=1, attempted_retries=getattr(self, 'forecast_retry_count', 0))
        timing_trials = [event for event in self.events if event['kind'] == 'release_timing_trial']
        conditions['release_timing_search'] = dict(
            candidates='half_indexed_lead_then_zero', motion='unchanged_best_four_parameter_command',
            max_trials_per_plan=2, attempted_trials=len(timing_trials),
            accepted_trials=sum(event['accepted'] for event in timing_trials),
            acceptance='unchanged_full_physics_objective_and_safety_limits')
        recovery_trials = [event for event in self.events if event['kind'] == 'reception_tilt_trial']
        conditions['reception_tilt_search'] = dict(
            trigger='receiver_wrist_pitch_limit_with_centered_timely_captured_ready_ball',
            default_amplitude_deg=EMPTY_RETURN_TILT_DEG,
            max_amplitude_deg=EMPTY_RETURN_TILT_DEG*MAX_RECEPTION_TILT_SCALE,
            max_trials_per_plan=2, attempted_trials=len(recovery_trials),
            accepted_trials=sum(event['accepted'] for event in recovery_trials),
            acceptance='zero_error_full_physics_certificate_and_unchanged_safety_limits')
        conditions['reception_refinement'] = dict(
            trigger='captured_ball_after_failed_indexed_forecast_retry', max_retries=1,
            attempted_retries=getattr(self, 'fine_reception_retry_count', 0),
            finite_difference=RECEPTION_FINE_DIFFERENCE.tolist(),
            acceptance='unchanged_reception_acceptance_and_safety_limits')
        conditions['early_recovery'] = dict(
            enabled=getattr(self, 'early_recovery_enabled', False),
            active=getattr(self, 'early_recovery_active', False),
            activation='late_confirmed_catch_after_loaded_recovery',
            lateness_trigger_s=RECEPTION_TIMELINESS_WINDOW_S,
            intercept_window_s=[RECEPTION_MIN_APPROACH_S, RECEPTION_MAX_APPROACH_S],
            starts=sum(event['kind'] == 'early_recovery' for event in self.events),
            finger_control='existing_release_curve_until_confirmation',
            flight_counting='unchanged_contact_separation_confirmation')
        if self.ball_observer.mode == 'sampled':
            conditions['observation'] = (
                'Sampled position estimates for live reception; truth for contact, safety and rollouts')
        return conditions

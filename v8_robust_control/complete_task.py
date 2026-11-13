"""Three-ball 2+1 start, cascade and continuously supported 2+1 finish."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import mujoco

from jugglebot.humanoid import cascade as C
from jugglebot.robustness.controller import ObservedCascade
from jugglebot.robustness.evaluation import EvaluationTelemetry
from jugglebot.robustness.perception import PerceptionConfig
from v7_three_ball_hold.end21 import EndingCascade
from .juggle import experiment_record

DEFAULT_SEED_PLANS = Path(__file__).with_name('complete_seed_plans.json')


class CompleteCascade(EndingCascade, ObservedCascade):
    # Keep the ordinary v6 comfort bound, including the ending maneuver.
    ELBOW_OUT_END_M = .10
    INITIAL_SETTLE_S = .05
    FINAL_SETTLE_S = .80
    SUPPORT_SPEED_M_S = .15
    SUPPORT_RADIUS_M = .16
    # One existing timeliness window at the nominal 5 ms replan cadence.
    MAX_THROW_PLAN_ATTEMPTS = 1 + round(C.RECEPTION_TIMELINESS_WINDOW_S / C.THROW_REPLAN_INTERVAL_S)
    ENDING_ELBOW_SOFT_LIMIT_M = .085
    ENDING_POSTURE_MAX_OFFSET_RAD = .10
    ENDING_POSTURE_BLEND_S = .30

    def __init__(self, throws=30, optimizer_inputs=None, **kwargs):
        if throws < 4 or throws % 2:
            raise ValueError('Complete task requires an even number of throws, at least four')
        kwargs.setdefault('normalize_reception_input', True)
        super().__init__(stop_after=throws, **kwargs)
        if optimizer_inputs is None:
            optimizer_inputs = json.loads(DEFAULT_SEED_PLANS.read_text(encoding='utf-8'))
        self.set_optimizer_inputs(dict(
            forecast=dict(self.warm, **optimizer_inputs.get('forecast', {})),
            execution=dict(self.execution_warm, **optimizer_inputs.get('execution', {}))))
        # TwoPlusOneCascade places its initial balls after ObservedCascade's
        # constructor. No samples from the earlier reset may survive.
        self.ball_observer.reset()
        self.ball_observer.update(self.data)
        self.initial_stable_age = 0.
        self.initial_support = None
        self.final_stable_age = 0.
        self.final_longest_stable_age = 0.
        self.throw_plan_attempts = {}

    def minimum_ending_horizon_s(self):
        """Reserve the minimum reception approach time for sampled control."""
        return max(super().minimum_ending_horizon_s(),
                   self.ARRIVE_EARLY_S + C.RECEPTION_MIN_APPROACH_S)

    def ending_posture_target(self, hand, data, target):
        """Bias redundant arm posture inward while preserving the palm task."""
        if self.end is None:
            return target
        model = self.model
        dof = hand['kin'].dof_adr
        elbow = model.body(hand['kin'].side+'_elbow_link').id
        shoulder = model.body(hand['kin'].side+'_shoulder_pitch_link').id
        outward = abs(data.xpos[elbow,1])-abs(data.xpos[shoulder,1])
        if outward <= self.ENDING_ELBOW_SOFT_LIMIT_M:
            return target
        jp, jr, je, js = (np.zeros((3,model.nv)) for _ in range(4))
        mujoco.mj_jacSite(model,data,jp,jr,hand['seat'])
        mujoco.mj_jacBody(model,data,je,None,elbow)
        mujoco.mj_jacBody(model,data,js,None,shoulder)
        task = np.vstack([jp[:,dof],jr[:,dof]])
        null = np.eye(7)-np.linalg.pinv(task,rcond=.001)@task
        gradient = null @ (np.sign(data.xpos[elbow,1])*je[1,dof]
                           -np.sign(data.xpos[shoulder,1])*js[1,dof])
        norm = np.linalg.norm(gradient)
        if norm < 1e-8:
            return target
        phase = np.clip((data.time-self.end['t_start'])
                        /self.ENDING_POSTURE_BLEND_S,0.,1.)
        blend = phase**3*(10.-15.*phase+6.*phase**2)
        weight = np.clip((outward-self.ENDING_ELBOW_SOFT_LIMIT_M)
                         /(self.ELBOW_OUT_END_M-self.ENDING_ELBOW_SOFT_LIMIT_M),0.,1.)
        return np.asarray(target)-self.ENDING_POSTURE_MAX_OFFSET_RAD*blend*weight*gradient/norm

    def return_servo(self, hand, data, target, **kwargs):
        if hand.get('mode') == 'ending':
            target = self.ending_posture_target(hand,data,target)
        return super().return_servo(hand,data,target,**kwargs)

    def plan_throw(self, side):
        live = not self.silent
        path = super().plan_throw(side)
        if live and (path is None or self.last_plan_error > .02):
            key = (side, self.throw_count)
            attempts = self.throw_plan_attempts.get(key, 0) + 1
            self.throw_plan_attempts = dict(self.throw_plan_attempts)
            self.throw_plan_attempts[key] = attempts
            if attempts >= self.MAX_THROW_PLAN_ATTEMPTS:
                self.failure_event = dict(reason='NO_FEASIBLE_THROW', side=side,
                                          ball=self.hands[side]['held'], time=float(self.data.time),
                                          attempts=attempts, objective=float(self.last_plan_error)
                                          if math.isfinite(self.last_plan_error) else None)
                raise RuntimeError('NO_FEASIBLE_THROW: exhausted complete-task planning attempts')
        return path

    def support_state(self, assignments):
        out = {}
        for ball, side in assignments.items():
            h = self.hands[side]
            q, v = self.balls[ball]
            rotation = self.data.site_xmat[h['seat']].reshape(3, 3)
            local = rotation.T @ (self.data.qpos[q:q+3]-self.data.site_xpos[h['seat']])
            relative = self.data.qvel[v:v+3]-self.spatial(h)[3:]
            out[str(ball)] = dict(hand=side, contact=bool(self.ball_contact(ball, side)),
                                 local_m=local.tolist(), relative_speed_m_s=float(np.linalg.norm(relative)),
                                 height_m=float(self.data.qpos[q+2]))
        ok = all(o['contact'] and o['relative_speed_m_s'] < self.SUPPORT_SPEED_M_S
                 and np.linalg.norm(o['local_m']) < self.SUPPORT_RADIUS_M
                 and o['height_m'] > .40 for o in out.values())
        return dict(ok=bool(ok), balls=out)

    def ready_for_throw(self, side):
        if (not self.silent and self.throw_count == 0
                and self.initial_support is None):
            return False
        return super().ready_for_throw(side)

    def _pair_exempt(self):
        # The reset pair can transmit load through the retained ball while
        # fingers open. The exemption ends at the confirmed first release.
        h = self.hands['right']
        if self.throw_count or not h.get('two_phase') or not self.ball_contact(0, 'right'):
            return False
        return all(np.linalg.norm(self.data.qpos[q:q+3]-self.data.site_xpos[h['seat']])
                   < self.SUPPORT_RADIUS_M for q, _ in (self.balls[0], self.balls[2]))

    def step(self):
        live = not self.silent
        if live and self.throw_count == 0 and self.initial_support is None:
            state = self.support_state({0:'right', 2:'right', 1:'left'})
            self.initial_stable_age = (self.initial_stable_age + self.model.opt.timestep
                                       if state['ok'] else 0.)
            if self.initial_stable_age + 1e-12 >= self.INITIAL_SETTLE_S:
                self.initial_support = dict(**state, time_s=float(self.data.time),
                                            continuous_support_s=self.initial_stable_age)
                self.log('initial_hold', 'right', 2, continuous_support_s=self.initial_stable_age)
        super().step()

    def _end_monitor(self):
        e = self.end
        incoming = e['a']
        if e['rel_first'] is None and self.ball_contact(incoming, 'right'):
            q, v = self.balls[incoming]
            relative = self.data.qvel[v:v+3]-self.spatial(self.hands['right'])[3:]
            e['rel_first'] = dict(t=float(self.data.time), speed=float(np.linalg.norm(relative)))
        assignments = {i: ('right' if i in (incoming, e['c']) else 'left') for i in range(3)}
        state = self.support_state(assignments)
        self.final_stable_age = self.final_stable_age + self.model.opt.timestep if state['ok'] else 0.
        self.final_longest_stable_age = max(self.final_longest_stable_age, self.final_stable_age)
        if self.final_stable_age + 1e-12 < self.FINAL_SETTLE_S or self.end_result is not None:
            return
        self.end_result = dict(**state, continuous_support_s=self.final_stable_age,
                               time_s=float(self.data.time), first_contact=e['rel_first'])
        # The fingertip ending has two supported balls in one hand. Its final
        # flight is confirmed here, separately from the one-ball cradle FSM.
        if incoming in self.airborne:
            release_id = self.airborne.pop(incoming)
            self.targets.pop(incoming)
            self.caught_events.add(release_id)
            self.catch_count += int(release_id > 0)
            self.log('catch', 'right', incoming, release_id=release_id,
                     palm_touch_time=e['rel_first']['t'],
                     contact_settle_time=float(self.data.time-e['rel_first']['t']),
                     ending=True, continuous_support_s=self.final_stable_age)

    def conditions_summary(self):
        result = super().conditions_summary()
        result['complete_task'] = dict(
            start='two balls right, one left; no initially airborne ball',
            initial_continuous_support_s=self.INITIAL_SETTLE_S,
            final_continuous_support_s=self.FINAL_SETTLE_S,
            ending_minimum_horizon_s=self.minimum_ending_horizon_s(),
            support_relative_speed_m_s=self.SUPPORT_SPEED_M_S,
            support_local_radius_m=self.SUPPORT_RADIUS_M,
            elbow_out_limit_m=self.ELBOW_OUT_END_M,
            max_throw_plan_attempts=self.MAX_THROW_PLAN_ATTEMPTS,
            ending_elbow_soft_limit_m=self.ENDING_ELBOW_SOFT_LIMIT_M,
            ending_posture_max_offset_rad=self.ENDING_POSTURE_MAX_OFFSET_RAD,
            ending_posture_blend_s=self.ENDING_POSTURE_BLEND_S,
            ending_posture_start='beginning of the ending approach',
            ball_collision_policy='reset pair 0/2 only before confirmed first release, retained ball touching right hand and both centers within the support radius; no post-release exemption',
            control='pure contact; ending prediction uses the selected ball observation mode')
        return result

    def run_complete(self, timeout=35.):
        result = super().run_continuous(self.stop_after, timeout, guard=True)
        complete = (self.initial_support is not None and self.end_result is not None
                    and self.end_result['ok'] and self.throw_count == self.stop_after
                    and not self.airborne and not self.targets)
        result['success'] = bool(result['success'] and complete)
        if not result['success'] and result['error'] is None:
            result['error'] = 'INCOMPLETE_TASK'
        result['task'] = 'three_ball_start_cascade_finish'
        result['initial_hold'] = self.initial_support
        result['final_hold'] = self.end_result
        result['longest_final_support_s'] = self.final_longest_stable_age
        result['supported_pair_contact_steps'] = getattr(self, 'pair_contact_steps', 0)
        result['supported_pair_force_max_n'] = self.hands['right'].get('pair_force_max', 0.)
        result['airborne_at_end'] = sorted(self.airborne)
        return result


class EvaluatedCompleteCascade(EvaluationTelemetry, CompleteCascade):
    pass


def complete_record(result, throws, timeout, optimizer_file=DEFAULT_SEED_PLANS):
    record = experiment_record(result, throws, timeout)
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), root/'v7_three_ball_hold/cascade21.py', root/'v7_three_ball_hold/end21.py']
    for path in files:
        record['source_sha256'][path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = ('hold2f_right_s20.json', 'first_throw_calibration.json', 'second_throw_calibration.json',
              'seeds_2plus1.json', 'catchtipM_az0_el90.json', 'pinky_A.json')
    record['task_inputs_sha256'] = {
        'v7_three_ball_hold/results/'+name: hashlib.sha256((root/'v7_three_ball_hold/results'/name).read_bytes()).hexdigest()
        for name in inputs}
    if optimizer_file is not None:
        path = Path(optimizer_file)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record['base_optimizer_inputs_sha256'] = record['optimizer_inputs_sha256']
        record['optimizer_inputs_sha256'] = digest
        try:
            label = path.resolve().relative_to(root).as_posix()
        except ValueError:
            label = path.resolve().as_posix()
        record['optimizer_inputs_file'] = label
        record['task_inputs_sha256'][label] = digest
    record['experiment'] = 'v8 complete three-ball task'
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--throws', type=int, default=30)
    parser.add_argument('--timeout', type=float, default=40.)
    parser.add_argument('--observation-mode', choices=('truth','sampled'), default='truth')
    parser.add_argument('--noise-mm', type=float, default=0.)
    parser.add_argument('--delay-ms', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--seed-plans', type=Path, default=DEFAULT_SEED_PLANS,
                        help='Full-task forecast and execution guesses from calibrate_complete.')
    args = parser.parse_args()
    if args.throws < 4 or args.throws % 2 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Use an even --throws >= 4 and a finite positive --timeout')
    if args.output.exists() or (args.video and args.video.exists()):
        parser.error('Output and video paths must be new')
    if args.video and args.output.resolve() == args.video.resolve():
        parser.error('Output JSON and video paths must be distinct')
    try:
        config = PerceptionConfig(args.noise_mm/1000., args.delay_ms/1000., args.dropout, args.seed)
        if args.observation_mode == 'truth' and config.enabled:
            raise ValueError('Perception disturbances require sampled mode')
    except ValueError as error:
        parser.error(str(error))
    recorder_factory = None
    if args.video:
        from jugglebot.video import VideoRecorder
        from v6_humanoid_hands.juggle import cascade_camera
        recorder_factory = lambda model: VideoRecorder(model, args.video, camera=cascade_camera())
    inputs = None
    if args.seed_plans:
        inputs = json.loads(args.seed_plans.read_text(encoding='utf-8'))
    record = complete_record(None, args.throws, args.timeout, args.seed_plans)
    controller = EvaluatedCompleteCascade(throws=args.throws, observation_mode=args.observation_mode,
                                          perception=config, recorder_factory=recorder_factory,
                                          optimizer_inputs=inputs)
    try:
        result = controller.run_complete(args.timeout)
    finally:
        if controller.recorder:
            controller.recorder.close()
    record['result'] = result
    record['estimation'] = controller.estimation_summary()
    if controller.recorder:
        record['video'] = dict(path=args.video.as_posix(), frames=controller.recorder.frames, fps=60, speed=1.)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('success','validated_flights','throws','error','time','final_hold')}, indent=2), flush=True)
    if not result['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

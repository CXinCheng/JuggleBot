"""Physical one-ball hold, release and opposite-hand catch reference.

This validates the complete nominal transfer before adding a training API.
It does not expose residual actions or claim learned-policy performance.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from jugglebot.robustness.controller import ObservedCascade
from .juggle import experiment_record
from .residual_env import CatchEvidence


INITIAL_HOLD_S = .05
FINAL_HOLD_S = .2


class SingleBallTransfer(ObservedCascade):
    def __init__(self, observation_mode='sampled'):
        super().__init__(n_balls=1, observation_mode=observation_mode, obs_noise=0.)

    def ready_for_throw(self, side):
        # Forecast rollouts retain the original controller. Live execution
        # stops scheduling throws after the one real release.
        if not self.silent and self.throw_count >= 1:
            return False
        return super().ready_for_throw(side)


def hand_evidence(controller, side):
    model, data = controller.model, controller.data
    hand = controller.hands[side]
    q, v = controller.balls[0]
    ball = controller.ball_geom_ids[0]
    contact = forbidden = False
    allowed = controller.hand_geoms['right'] | controller.hand_geoms['left']
    for index, item in enumerate(data.contact):
        other = item.geom2 if item.geom1 == ball else item.geom1 if item.geom2 == ball else -1
        if other < 0 or item.efc_address < 0 or item.dist > 1e-5:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        contact |= other in controller.hand_geoms[side] and force[0] > .02
        forbidden |= other not in allowed
    rotation = data.site_xmat[hand['seat']].reshape(3, 3)
    center = rotation.T@(data.qpos[q:q+3]-data.site_xpos[hand['seat']])
    speed = np.linalg.norm(data.qvel[v:v+3]-controller.spatial(hand)[3:])
    external = bool(np.any(data.eq_active) or np.any(data.qfrc_applied[v:v+6])
                    or np.any(data.xfrc_applied[controller.ball_body_ids]))
    return CatchEvidence(bool(contact), float(speed), float(np.linalg.norm(center[:2])),
                         float(center[2]), float(data.qpos[q+2]), bool(forbidden), False, external)


def stable(evidence):
    return (evidence.contact and evidence.relative_speed_m_s < .15 and evidence.tangent_m < .04
            and .01 < evidence.normal_m < .06 and not evidence.forbidden_contact
            and not evidence.pose_violation and not evidence.external_ball_force)


def validate_transfer(result):
    releases, catches = result['releases'], result['catches']
    evidence = CatchEvidence(**result['final_evidence'])
    return (result['error'] is None and result['failure_event'] is None and result['ball_count'] == 1
            and result['contact_only_checks'] > 0 and result['airborne_physics_checks'] > 0
            and result['initial_hold_validated'] and result['initial_hold_max_s']+1e-10 >= INITIAL_HOLD_S
            and result['final_hold_s']+1e-10 >= FINAL_HOLD_S and stable(evidence)
            and .4 < evidence.ball_height_m < 1.3
            and result['throws'] == 1 and result['caught_release_ids'] == [1]
            and len(releases) == 1 and releases[0]['count'] == 1 and releases[0]['ball'] == 0
            and releases[0]['side'] == 'right' and releases[0]['separation_time'] > 0.
            and len(catches) == 1 and catches[0]['release_id'] == 1 and catches[0]['ball'] == 0
            and catches[0]['side'] == 'left' and catches[0]['palm_touch_time'] > releases[0]['separation_time'])


def run_transfer(observation_mode='sampled', timeout=5.):
    if observation_mode not in ('truth', 'sampled') or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('Require truth/sampled observation and a positive finite timeout')
    record = experiment_record(None, 1, timeout)
    source_key = 'v8_robust_control/single_ball.py'
    record['source_sha256'][source_key] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    controller = SingleBallTransfer(observation_mode)
    initial_time = initial_max = final_time = 0.
    initial_validated = False
    error = None
    evidence = hand_evidence(controller, 'left')
    dt = float(controller.model.opt.timestep)
    try:
        while controller.data.time < timeout:
            controller.step()
            if controller.throw_count == 0:
                start_evidence = hand_evidence(controller, 'right')
                initial_time = initial_time+dt if stable(start_evidence) else 0.
                initial_max = max(initial_max, initial_time)
                initial_validated |= initial_time+1e-10 >= INITIAL_HOLD_S
            evidence = hand_evidence(controller, 'left')
            if evidence.external_ball_force or evidence.forbidden_contact or not np.isfinite(evidence.ball_height_m):
                raise RuntimeError('INVALID_BALL_PHYSICS')
            if not .4 < evidence.ball_height_m < 1.3:
                raise RuntimeError('BALL_HEIGHT_LIMIT')
            if controller.throw_count > 1:
                raise RuntimeError('EXTRA_RELEASE')
            if controller.throw_count and not initial_validated:
                raise RuntimeError('UNSTABLE_INITIAL_HOLD')
            final_time = final_time+dt if 1 in controller.caught_events and stable(evidence) else 0.
            if final_time+1e-10 >= FINAL_HOLD_S:
                break
        else:
            error = 'TASK_DEADLINE'
    except RuntimeError as failure:
        error = str(failure)
    releases = [event for event in controller.events if event['kind'] == 'release']
    catches = [event for event in controller.events if event['kind'] == 'catch']
    result = dict(ball_count=len(controller.balls), throws=controller.throw_count,
                  caught_release_ids=sorted(controller.caught_events), releases=releases, catches=catches,
                  initial_hold_validated=bool(initial_validated), initial_hold_max_s=initial_max,
                  final_hold_s=final_time, final_evidence=asdict(evidence), error=error,
                  time_s=float(controller.data.time), failure_event=controller.failure_event,
                  contact_only_checks=controller.contact_only_checks,
                  airborne_physics_checks=controller.airborne_physics_checks,
                  conditions=controller.conditions_summary(), style=controller.style_summary())
    result['success'] = bool(validate_transfer(result))
    if not result['success'] and result['error'] is None:
        result['error'] = 'TRANSFER_ACCEPTANCE_FAILED'
    current = experiment_record(None, 1, timeout)
    current['source_sha256'][source_key] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (current['source_sha256'] != record['source_sha256']
            or current['optimizer_inputs_sha256'] != record['optimizer_inputs_sha256']):
        raise RuntimeError('Single-ball runtime changed during execution')
    record['result'] = result
    record.update(experiment='single-ball nominal transfer', task_scope='right hold, one release, left stable catch',
                  observation_mode=observation_mode, initial_hold_required_s=INITIAL_HOLD_S,
                  final_hold_required_s=FINAL_HOLD_S, residual_control=False, training_api=False)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation-mode', choices=['truth', 'sampled'], default='sampled')
    parser.add_argument('--timeout', type=float, default=5.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Use a new output and a positive finite timeout')
    record = run_transfer(args.observation_mode, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: record['result'][key] for key in
                      ('success', 'throws', 'initial_hold_max_s', 'final_hold_s', 'time_s', 'error')}, indent=2), flush=True)
    if not record['result']['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

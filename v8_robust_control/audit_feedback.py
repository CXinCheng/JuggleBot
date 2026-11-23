"""Paired physical checks of gain requests across contact-permission changes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .compare_residual import digest
from .residual_env import UNSAFE_REASONS
from .train_cascade import MOTOR_CONTROLLERS, environment_type, sources


CASES = {
    'noise_3mm': dict(noise_std_m=.003),
    'perception_combined': dict(noise_std_m=.003, delay_s=.02, dropout=.1),
    'seat_2mm': dict(seat_jitter_m=.002),
}
CONTROLLERS = ('reference', 'latched_feedback')
REQUESTS = {'default': [0.]*6, 'disabled': [-1.]*6}
CASE_OPTIONS = dict(CASES, clean={}, perception_extreme=dict(noise_std_m=.01, delay_s=.04, dropout=.3))
CONTROLLER_OPTIONS = CONTROLLERS+('guarded_feedback',)+MOTOR_CONTROLLERS
REQUEST_OPTIONS = dict(REQUESTS, zero=[0.]*6, negative=[-1.]*6, positive=[1.]*6)


def native_trial(job):
    """Apply the same reset perturbation to the original planning controller."""
    from .complete_task import CompleteCascade
    from jugglebot.robustness.perception import PerceptionConfig
    throws, timeout, case, seed = job
    settings = CASE_OPTIONS[case]
    rng = np.random.default_rng(seed)
    jitter = settings.get('seat_jitter_m', 0.)
    shifts = rng.uniform(-jitter, jitter, (2, 2))
    perception_seed = int(rng.integers(0, 2**31))
    perception = PerceptionConfig(settings.get('noise_std_m', 0.), settings.get('delay_s', 0.),
                                  settings.get('dropout', 0.), perception_seed)
    started = time.perf_counter()
    controller = CompleteCascade(throws=throws, observation_mode='sampled', perception=perception)
    # Initialization only: no ball state is rewritten after physics starts.
    if controller.data.time != 0.:
        raise ValueError('Native perturbation requires an unstepped controller')
    for ball, (q, _) in enumerate(controller.balls):
        controller.data.qpos[q:q+2] += shifts[0 if ball in (0, 2) else 1]
    mujoco.mj_forward(controller.model, controller.data)
    controller.ball_observer.reset()
    controller.ball_observer.update(controller.data)
    controller.initial_ball_states = [dict(position_m=controller.data.qpos[q:q+3].tolist(),
        velocity_m_s=controller.data.qvel[v:v+3].tolist()) for q, v in controller.balls]
    result = controller.run_complete(timeout=timeout)
    print('native', case, seed, result['throws'], result['catches'], result['error'], flush=True)
    return dict(case=case, seed=seed, settings=settings, initial_displacement_m=shifts[[0, 1, 0]].tolist(),
                wall_time_s=time.perf_counter()-started, events=controller.events, result=result)


def summarize_native(record):
    manifest, rows = record['manifest'], record['records']
    if (not manifest['cases'] or any(name not in CASE_OPTIONS or settings != CASE_OPTIONS[name]
            for name, settings in manifest['cases'].items())
            or not manifest['seeds'] or len(set(manifest['seeds'])) != len(manifest['seeds'])):
        raise ValueError('Unknown native cases or duplicate seeds')
    expected = {(case, seed) for case in manifest['cases'] for seed in manifest['seeds']}
    if len(rows) != len(expected) or {(row['case'], row['seed']) for row in rows} != expected:
        raise ValueError('Missing or duplicate native diagnostic scenes')
    summary = []
    for row in rows:
        result = row['result']
        jitter = row['settings'].get('seat_jitter_m', 0.)
        expected_shift = np.random.default_rng(row['seed']).uniform(-jitter, jitter, (2, 2))[[0, 1, 0]]
        if (row['settings'] != manifest['cases'][row['case']]
                or not np.array_equal(np.asarray(row['initial_displacement_m']), expected_shift)
                or result['requested_throws'] != manifest['requested_throws']
                or result['catches'] > result['throws'] or result['throws'] > manifest['requested_throws']
                or result['success'] and (result['catches'] != manifest['requested_throws']
                    or result['error'] is not None or result['failure_event'] is not None
                    or result['final_hold'] is None or result['final_hold']['continuous_support_s'] < .8-1e-10
                    or result['max_height'] >= 1.3 or result['drops'] or result['ball_contacts']
                    or result['grasp_assistance'] or result['grasp_constraints'])):
            raise ValueError('Different native task or unconfirmed completion')
        summary.append(dict(case=row['case'], seed=row['seed'], complete=result['success'],
            releases=result['throws'], catches=result['catches'], error=result['error'],
            failure_event=result['failure_event']))
    return summary


def release_snapshot(env, event):
    """Read physical release evidence for diagnosis, outside policy inputs."""
    ball = event['ball']
    q, v = env.controller.balls[ball]
    reference = env.expected_states[env.index-1]
    nominal = env.release_reference[event['id']-1]
    position = env.data.qpos[q:q+3].copy()
    velocity = env.data.qvel[v:v+3].copy()
    reference_position = reference[q:q+3]
    reference_velocity = reference[env.model.nq+v:env.model.nq+v+3]
    estimate = env.trackers[ball].estimate(float(env.data.time))
    return dict(id=event['id'], ball=ball, source_side=event['side'],
        separation_time_s=float(event['time']), diagnostic_time_s=float(env.data.time),
        nominal_separation_time_s=nominal['separation_time'],
        release_timing_error_s=float(event['time']-nominal['separation_time']),
        position_m=position.tolist(), velocity_m_s=velocity.tolist(),
        nominal_position_m=reference_position.tolist(), nominal_velocity_m_s=reference_velocity.tolist(),
        position_error_m=(position-reference_position).tolist(),
        velocity_error_m_s=(velocity-reference_velocity).tolist(),
        estimate_valid=bool(estimate is not None and estimate.valid),
        estimation_position_error_m=None if estimate is None or not estimate.valid else
            (np.asarray(estimate.position_m)-position).tolist(),
        estimation_velocity_error_m_s=None if estimate is None or not estimate.valid else
            (np.asarray(estimate.velocity_m_s)-velocity).tolist(),
        joint_offset_rad=env.joint_offset.tolist(),
        wrapping_offset_rad=env.wrapping_offset.tolist())


def trial(job):
    reference, controller, request, case, seed = job
    env = environment_type(controller)(reference)
    env.config = replace(env.config, **CASE_OPTIONS[case])
    first_wrapping = None
    releases = []
    previous_events = 0
    first_permission = None
    requested_ticks = permitted_ticks = 0
    original = env._wrapping_target
    original_monitor = env._monitor

    def measured_target():
        nonlocal first_wrapping
        value = original()
        if first_wrapping is None and np.any(value != 0.):
            first_wrapping = float(env.data.time)
        return value

    env._wrapping_target = measured_target

    def measured_monitor():
        nonlocal previous_events
        try:
            return original_monitor()
        finally:
            for event in env.events[previous_events:]:
                if event['kind'] == 'release':
                    releases.append(release_snapshot(env, event))
            previous_events = len(env.events)

    env._monitor = measured_monitor
    try:
        env.reset(seed=seed)
        while True:
            permissions = env.permissions()
            requested = np.asarray(REQUEST_OPTIONS[request]) != 0.
            requested_ticks += int(np.any(requested))
            permitted_ticks += int(np.any(requested & permissions))
            if first_permission is None and np.any(permissions):
                first_permission = float(env.data.time)
            _, _, done, truncated, info = env.step(REQUEST_OPTIONS[request])
            if done or truncated:
                break
        row = dict(controller=controller, request=request, case=case, seed=seed,
            first_wrapping_request_s=first_wrapping, first_control_permission_s=first_permission,
            requested_control_ticks=requested_ticks, permitted_control_ticks=permitted_ticks,
            releases=releases, terminal_balls=[dict(ball=ball,
                position_m=env.data.qpos[q:q+3].tolist(), velocity_m_s=env.data.qvel[v:v+3].tolist())
                for ball, (q, v) in enumerate(env.controller.balls)], result=info)
        print(controller, request, case, seed, info['reason'],
              f"{info['catches']}/{info['requested_throws']}", flush=True)
        return row
    finally:
        env.close()


def summarize(record):
    manifest = record['manifest']
    if manifest.get('controller') == 'native':
        return summarize_native(record)
    controllers = manifest.get('controllers', CONTROLLERS)
    cases, requests = manifest['cases'], manifest['requests']
    if (not controllers or len(set(controllers)) != len(controllers)
            or any(name not in CONTROLLER_OPTIONS for name in controllers)
            or not cases or any(name not in CASE_OPTIONS or value != CASE_OPTIONS[name] for name, value in cases.items())
            or not requests or any(name not in REQUEST_OPTIONS or value != REQUEST_OPTIONS[name] for name, value in requests.items())
            or any(name in MOTOR_CONTROLLERS for name in controllers) and 'disabled' in requests):
        raise ValueError('Unknown diagnostic controller, perturbation or request')
    expected = {(controller, request, case, seed) for controller in controllers
                for request in requests for case in cases for seed in manifest['seeds']}
    rows = record['records']
    if len(rows) != len(expected) or {(row['controller'], row['request'], row['case'], row['seed'])
                                     for row in rows} != expected:
        raise ValueError('Missing or duplicate paired feedback cases')
    for row in rows:
        result = row['result']
        if (result['environment_version'] != environment_type(row['controller']).ENVIRONMENT_VERSION
                or result['config'] != dict(manifest['base_config'], **cases[row['case']])
                or result['is_success'] != (result['reason'] == 'SUCCESS')
                or result['requested_throws'] != manifest['requested_throws']):
            raise ValueError('Different physical task or perturbation configuration')
        if row['controller'] == 'latched_feedback' and row['request'] == 'disabled':
            if row['first_wrapping_request_s'] is not None or result['max_wrapping_offset_rad'] != 0.:
                raise ValueError('Disabled feedback was applied after a contact transition')
    values = []
    for controller in controllers:
        for request in requests:
            for case in cases:
                selected = [row['result'] for row in rows if
                            (row['controller'], row['request'], row['case']) == (controller, request, case)]
                values.append(dict(controller=controller, request=request, case=case,
                    episodes=len(selected), successes=sum(row['is_success'] for row in selected),
                    unsafe=sum(row['reason'] in UNSAFE_REASONS for row in selected),
                    catches=sum(row['catches'] for row in selected),
                    max_wrapping_offset_rad=max(row['max_wrapping_offset_rad'] for row in selected)))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[410000, 410001])
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controllers', nargs='+', choices=CONTROLLER_OPTIONS, default=list(CONTROLLERS))
    parser.add_argument('--cases', nargs='+', choices=CASE_OPTIONS, default=list(CASES))
    parser.add_argument('--requests', nargs='+', choices=REQUEST_OPTIONS, default=list(REQUESTS))
    parser.add_argument('--native-timeout', type=float,
        help='Diagnose the original planner instead of replay controllers, up to this simulation time')
    args = parser.parse_args()
    if (args.output.exists() or args.jobs < 1 or not args.seeds
            or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds)
            or any(len(set(values)) != len(values) for values in (args.controllers, args.cases, args.requests))):
        parser.error('Use a new output, distinct nonnegative seeds and positive jobs')
    if args.native_timeout is None and any(name in MOTOR_CONTROLLERS for name in args.controllers) and 'disabled' in args.requests:
        parser.error('Use zero, negative or positive requests for motor residuals; negative is not disabled')
    if args.native_timeout is not None and (not np.isfinite(args.native_timeout) or args.native_timeout <= 0):
        parser.error('Native timeout must be finite and positive')
    env = environment_type('reference')(args.reference_dir)
    try:
        from dataclasses import asdict
        config, throws, task = asdict(env.config), env.target_throws, env.task
    finally:
        env.close()
    if args.native_timeout is not None:
        if task != 'complete':
            parser.error('Native planning diagnosis requires a complete-task reference')
        fingerprints = sources('reference')
        manifest = dict(schema_version=2, controller='native', seeds=args.seeds,
            cases={name: CASE_OPTIONS[name] for name in args.cases}, requested_throws=throws,
            timeout_s=args.native_timeout, reference_dir=args.reference_dir.as_posix(),
            reference_sha256=digest(args.reference_dir/'reference.npz'), source_sha256=fingerprints,
            audit_sha256=digest(Path(__file__)),
            scope='native planning diagnosis; a prefix timeout is not complete-task success')
        jobs = [(throws, args.native_timeout, case, seed) for case in args.cases for seed in args.seeds]
        with ProcessPoolExecutor(args.jobs) as pool:
            rows = list(pool.map(native_trial, jobs))
        if sources('reference') != fingerprints or digest(Path(__file__)) != manifest['audit_sha256']:
            raise ValueError('Native diagnostic sources changed during execution')
        record = dict(manifest=manifest, records=rows)
        record['summary'] = summarize(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
        print(json.dumps(record['summary']), flush=True)
        return
    fingerprints = {name: sources(name) for name in args.controllers}
    manifest = dict(schema_version=2, controllers=args.controllers, seeds=args.seeds,
        cases={name: CASE_OPTIONS[name] for name in args.cases},
        requests={name: REQUEST_OPTIONS[name] for name in args.requests}, base_config=config,
        requested_throws=throws, reference_dir=args.reference_dir.as_posix(),
        reference_sha256=digest(args.reference_dir/'reference.npz'), source_sha256=fingerprints,
        audit_sha256=digest(Path(__file__)), scope='development diagnosis; no learned-policy qualification')
    jobs = [(args.reference_dir, controller, request, case, seed) for controller in args.controllers
            for request in args.requests for case in args.cases for seed in args.seeds]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(trial, jobs))
    if fingerprints != {name: sources(name) for name in args.controllers} or digest(Path(__file__)) != manifest['audit_sha256']:
        raise ValueError('Audit controller sources changed during execution')
    record = dict(manifest=manifest, records=rows)
    record['summary'] = summarize(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
    print(json.dumps(record['summary']), flush=True)


if __name__ == '__main__':
    main()

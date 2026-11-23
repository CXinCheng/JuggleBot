"""Collect planner supervision through the bounded event-action interface."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from jugglebot.humanoid import cascade as C
from .compare_residual import digest
from .complete_task import CompleteCascade
from .task_control import (TaskCommand, THROW_RESIDUAL_LIMITS, RECEPTION_RESIDUAL_LIMITS,
                           MAX_DEFER_S, TaskRequest, OracleDecisions, decode_action, validate_command)
from .task_env import TaskConfig, TaskResidualEnv, action_authority, context_key, runtime_sources
from .train_task import TaskCurriculum, write_new
from .audit_task_learning import features, LOCAL_INDICES


MODES = ('zero', 'fixed_first', 'planner_first', 'planner_joint')
VERIFIED_DEMONSTRATION_MODES = ('planner_joint', 'recorded_joint')


class PlanningBudgetExceeded(TimeoutError):
    """Computation stopped; this is not a physical task failure."""


def live_fingerprint(controller):
    """Only an isolation check; never a student feature."""
    data = controller.data
    arrays = (data.qpos, data.qvel, data.qacc, data.ctrl, data.qfrc_applied, data.xfrc_applied)
    value = b''.join(np.asarray(array).tobytes() for array in arrays)
    return hashlib.sha256(value+np.float64(data.time).tobytes()).hexdigest()


def fork_controller(controller):
    # Live callbacks refer to the Gym worker and its queues/locks. They must
    # neither be cloned nor invoked by a hypothetical planning computation.
    memo = {id(controller.model): controller.model}
    for name in ('decision', 'cancel'):
        item = getattr(controller, name, None)
        if item is not None:
            memo[id(item)] = None
    transport = getattr(controller.ball_observer, 'transport', None)
    if transport is not None:
        memo[id(transport._digest)] = transport._digest.copy()
    return copy.deepcopy(controller, memo)


def encode_command(request, command):
    """Reject unexpressible labels instead of clipping planner decisions."""
    bounds = CompleteCascade.reception_bounds(request.side) if request.kind == 'reception' else None
    parameters = validate_command(request, command, bounds)
    limits = THROW_RESIDUAL_LIMITS if request.kind == 'throw' else RECEPTION_RESIDUAL_LIMITS
    action = np.r_[(parameters-request.nominal_parameters)/limits,
                   command.defer_s/MAX_DEFER_S if request.kind == 'throw' else 2.*(command.tilt_scale-1.)]
    if np.any(np.abs(action) > 1.+1e-12):
        raise ValueError('Planner decision is outside the normalized action space')
    action = np.clip(action, -1., 1.)  # Numerical tolerance only, after validation.
    decoded = decode_action(request, action)
    if (not np.allclose(decoded.parameters, command.parameters, rtol=0., atol=1e-12)
            or abs(decoded.defer_s-command.defer_s) > 1e-12
            or abs(decoded.tilt_scale-command.tilt_scale) > 1e-12):
        raise ValueError('Planner decision cannot round-trip through the action decoder')
    return action


def planner_command(fork, request):
    """Original planning algorithm, with the same low-level guards as the student."""
    if request.kind == 'throw':
        path = CompleteCascade.plan_throw(fork, request.side)
        if path is None or fork.last_plan_error > .02:
            return TaskCommand(request.nominal_parameters, defer_s=C.THROW_REPLAN_INTERVAL_S)
        parameters = tuple(fork.commands[request.side][:4])+(float(getattr(path, 'finger_release_lead_s', 0.)),)
        return TaskCommand(parameters)
    CompleteCascade.plan_reception(fork, request.side)
    hand = fork.hands[request.side]
    return TaskCommand(tuple(hand['return_parameters']),
        tilt_scale=float(getattr(hand['approach_joint'], 'reception_tilt_scale', 1.)))


def restore_planner_memory(fork, state):
    """Restore optimizer guesses, never a live actuator or ball state."""
    for side, item in (state or {}).items():
        command = np.asarray(item['command'], dtype=float)
        if side not in fork.hands or command.shape != (4,) or not np.all(np.isfinite(command)):
            raise ValueError('Invalid teacher optimizer memory')
        fork.hands[side]['retry_throw_key'] = item['key']
        fork.hands[side]['retry_throw_command'] = command.copy()
    # The bounded executor records commands separately from native plan events.
    # Reconstruct only physically started strokes so the native planner retains
    # its ordinary previous-launch guesses on subsequent events.
    accepted = {(row['time_s'], row['side'], row['ball_id']): row
                for row in getattr(fork, 'decisions', [])
                if row['kind'] == 'throw' and row['defer_s'] == 0.}
    history = []
    for event in getattr(fork, 'events', []):
        key = (event['time'], event['side'], event['ball'])
        if event['kind'] == 'stroke' and key in accepted:
            history.append(dict(kind='plan', time=event['time'], side=event['side'],
                                ball=event['ball'], command=list(accepted[key]['parameters'])))
        history.append(event)
    if hasattr(fork, 'events'):
        fork.events = history


def planner_memory(fork):
    return {side:dict(key=hand['retry_throw_key'], command=hand['retry_throw_command'].tolist())
            for side,hand in getattr(fork, 'hands', {}).items()
            if hand.get('retry_throw_key') is not None and hand.get('retry_throw_command') is not None}


def query_planner(controller, request, budget_s=120., trial_deadline=float('inf'), optimizer_state=None):
    before = live_fingerprint(controller)
    started = time.perf_counter()
    deadline = min(started+budget_s, trial_deadline)
    computation = dict(setup_wall_s=0.,rollout_wall_s=0.,reference_wall_s=0.,
                       rollout_steps=0,auxiliary_references=0,candidates=[],budget_phase=None)
    candidate_paths = []
    def outcome(**values):
        for row,path in zip(computation['candidates'],candidate_paths):
            row['catch_diagnostics'] = copy.deepcopy(getattr(path,'catch_diagnostics',None))
        return dict(values,query_wall_s=time.perf_counter()-started,computation=computation)
    try:
        fork = fork_controller(controller)
        if fork.data is controller.data or fork.ball_observer is controller.ball_observer:
            raise AssertionError('Planner fork shares mutable live state')
        restore_planner_memory(fork, optimizer_state)
        computation['setup_wall_s'] = time.perf_counter()-started
        if request.kind=='reception' and hasattr(fork,'reception_reference'):
            original_reference = fork.reception_reference
            def measured_reference(side,parameters,**kwargs):
                index = None
                if side==request.side:
                    index = len(computation['candidates'])
                    computation['candidates'].append(dict(side=side,parameters=np.asarray(parameters).tolist(),
                        tilt_scale=kwargs.get('tilt_scale',1.),rollout_steps=0,
                        steps_after_forbidden_contact=0,first_forbidden_contact=None))
                    candidate_paths.append(None)
                else:
                    # The other arm can construct its calibrated forecast
                    # during a rollout; it is not a new optimization candidate.
                    computation['auxiliary_references'] += 1
                begin = time.perf_counter()
                try:
                    reference = original_reference(side,parameters,**kwargs)
                    if index is not None and reference is not None:
                        candidate_paths[index] = reference[0]
                    return reference
                finally:
                    computation['reference_wall_s'] += time.perf_counter()-begin
            fork.reception_reference = measured_reference
        if hasattr(fork,'check_forbidden_contacts'):
            original_check = fork.check_forbidden_contacts
            def measured_check():
                try:
                    return original_check()
                except RuntimeError:
                    if computation['candidates']:
                        row = computation['candidates'][-1]
                        if row['first_forbidden_contact'] is None:
                            row['first_forbidden_contact'] = copy.deepcopy(fork.failure_event)
                    raise
            fork.check_forbidden_contacts = measured_check
        if hasattr(fork, 'step'):
            original_step = fork.step
            def bounded_step():
                if time.perf_counter() >= deadline:
                    computation['budget_phase'] = 'rollout'
                    raise PlanningBudgetExceeded('Planner computation budget exhausted')
                begin = time.perf_counter()
                try:
                    return original_step()
                finally:
                    computation['rollout_wall_s'] += time.perf_counter()-begin
                    computation['rollout_steps'] += 1
                    if computation['candidates']:
                        row = computation['candidates'][-1]
                        row['rollout_steps'] += 1
                        row['steps_after_forbidden_contact'] += int(row['first_forbidden_contact'] is not None)
            fork.step = bounded_step
        try:
            if time.perf_counter() >= deadline:
                computation['budget_phase'] = 'setup'
                raise PlanningBudgetExceeded('Planner computation budget exhausted')
            command = planner_command(fork, request)
        except PlanningBudgetExceeded as error:
            return outcome(status='resource_limit', error=str(error))
        except (RuntimeError, ValueError) as error:
            return outcome(status='planner_rejected', error=str(error),
                           failure_event=copy.deepcopy(fork.failure_event))
        try:
            action = encode_command(request, command)
        except ValueError as error:
            return outcome(status='unrepresentable', command=asdict(command), error=str(error))
        return outcome(status='accepted', command=asdict(command), action=action.tolist(),
                    optimizer_state=planner_memory(fork),
                    planning_outcome=('defer' if command.defer_s else request.kind))
    finally:
        if live_fingerprint(controller) != before:
            raise AssertionError('Teacher query changed live physics')


class PlannerAdvisor:
    """Privileged label generator, not a learned or deployable policy."""
    def __init__(self, env, mode, control_releases, planner_budget_s=120., trial_budget_s=300.):
        self.env, self.mode, self.control_releases = env, mode, control_releases
        self.records = []
        self.optimizer_state = {}
        self.label = mode
        self.planner_budget_s = planner_budget_s
        self.trial_deadline = time.perf_counter()+trial_budget_s

    def predict(self, obs, deterministic=True):
        request = self.env.unwrapped._request
        if request is None:
            return np.zeros(6), None
        eligible = (request.released < self.control_releases and action_authority(obs)
                    and (self.mode == 'planner_joint' or request.kind == 'throw' and request.released == 0))
        if self.mode == 'zero' or not eligible:
            return np.zeros(6), None
        if self.mode == 'fixed_first':
            return np.array([.1/.75, 0., .1/1.25, 0., 0., 0.]), None
        print('[Teacher query]', self.label, context_key(request), f't={request.time_s:.3f}', flush=True)
        row = dict(key=context_key(request), request=asdict(request),
                   observation={name:value.tolist() for name,value in obs.items()},
                   **query_planner(self.env.unwrapped.controller, request,
                                   self.planner_budget_s, self.trial_deadline, self.optimizer_state))
        self.records.append(row)
        print('[Teacher decision]', self.label, row['key'], row['status'], f"{row.get('query_wall_s', 0.):.1f}s", flush=True)
        if row['status'] != 'accepted':
            raise RuntimeError('TEACHER_'+row['status'].upper()+': '+row['error'])
        self.optimizer_state = copy.deepcopy(row.get('optimizer_state', {}))
        return np.asarray(row['action']), None


def candidate_events(path, seed, jitter):
    """Read candidate numbers; old success claims never replace new physics checks."""
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    shifts = np.random.default_rng(seed).uniform(-jitter, jitter, (2, 2))
    if record.get('manifest', {}).get('controller') == 'native':
        if record['manifest']['requested_throws'] != TaskConfig().throws:
            raise ValueError('Candidate task length differs')
        rows = [row for row in record['records'] if row['seed']==seed
                and row['settings'].get('seat_jitter_m', 0.)==jitter
                and not any(row['settings'].get(name,0.) for name in ('noise_std_m','delay_s','dropout'))]
        if len(rows)!=1 or not np.array_equal(rows[0]['initial_displacement_m'], shifts[[0,1,0]]):
            raise ValueError('Require one candidate with matching reset and perception settings')
        return rows[0]['events']
    if 'events' in record and 'shifts_m' in record and np.array_equal(record['shifts_m'], shifts):
        return record['events']
    raise ValueError('Candidate trace is missing or its initial displacement differs')


class RecordedAdvisor:
    """Revalidate an offline plan; this is an oracle capability test, not a policy."""
    def __init__(self, env, events, control_releases):
        self.env, self.control_releases = env, control_releases
        self.oracle = OracleDecisions(events)
        self.records = []

    def predict(self, obs, deterministic=True):
        request = self.env.unwrapped._request
        if request is None or request.released >= self.control_releases or not action_authority(obs):
            return np.zeros(6), None
        try:
            command = self.oracle(request)
            action = encode_command(request, command)
        except (RuntimeError, ValueError) as error:
            self.records.append(dict(key=context_key(request), status='candidate_mismatch', error=str(error)))
            raise RuntimeError('TEACHER_CANDIDATE_MISMATCH: '+str(error)) from error
        self.records.append(dict(key=context_key(request), request=asdict(request), command=asdict(command),
            observation={name:value.tolist() for name,value in obs.items()}, action=action.tolist(),
            status='accepted', origin='recorded_candidate', query_wall_s=0.))
        return action, None


def run_trial(job):
    reference, seed, jitter, mode, goal, control_releases, planner_budget, trial_budget, *extra = job
    torch.set_num_threads(1)
    env = TaskCurriculum(TaskResidualEnv(TaskConfig(), reference),
                         [dict(catch_goal=goal, seat_jitter_m=jitter)])
    actor = (RecordedAdvisor(env, candidate_events(extra[0], seed, jitter), control_releases)
             if mode=='recorded_joint' else PlannerAdvisor(env, mode, control_releases, planner_budget, trial_budget))
    actor.label = f'{mode}/seed{seed}'
    started = time.perf_counter()
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            try:
                action = actor.predict(obs)[0]
            except RuntimeError as error:
                if not str(error).startswith('TEACHER_'):
                    raise
                # The expert declined or cannot be expressed. End the diagnostic
                # without advancing physics or substituting a guessed command.
                info = dict(env.unwrapped._info(str(error).split(':', 1)[0]),
                            training_prefix=False, is_success=False, diagnostic_abort=True,
                            terminated=False, truncated=True)
                break
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        if info.get('training_prefix') and not set(range(1, goal+1)).issubset(info['recovered_ids']):
            raise ValueError('Missing physical recovered-flight evidence')
        print('[Joint event audit]', seed, jitter, mode, info['reason'],
              'queries', len(actor.records), f'{time.perf_counter()-started:.1f}s', flush=True)
        return dict(seed=seed, jitter_m=jitter, mode=mode, goal=goal,
            control_releases=control_releases, wall_s=time.perf_counter()-started,
            result=info, demonstrations=actor.records,
            advisor_uses_privileged_simulator=mode.startswith('planner_'),
            advisor_uses_recorded_future_plan=mode=='recorded_joint',
            complete_task_verified=False, rl_gain_verified=False)
    finally:
        env.close()


def summarize(rows):
    return {mode:dict(episodes=len(group),
        recovered=sum(bool(row['result'].get('training_prefix')) for row in group),
        unsafe=sum(row['result']['reason'] in TaskResidualEnv.UNSAFE for row in group),
        rejected=sum(row['result']['reason'] == 'TEACHER_PLANNER_REJECTED' for row in group),
        unrepresentable=sum(row['result']['reason'] == 'TEACHER_UNREPRESENTABLE' for row in group),
        resource_limited=sum(row['result']['reason'] == 'TEACHER_RESOURCE_LIMIT' for row in group),
        candidate_mismatch=sum(row['result']['reason'] == 'TEACHER_CANDIDATE_MISMATCH' for row in group),
        labels=sum(item['status'] == 'accepted' for row in group for item in row['demonstrations']))
        for mode in MODES+('recorded_joint',) if (group := [row for row in rows if row['mode'] == mode])}


STUDENT_MODES = ('zero', 'fixed_first', 'fixed_event', 'first_only', 'nearest_event')


def student_key(obs):
    context = obs['context']
    kind = 'throw' if context[0] > .5 else 'reception'
    side = 'left' if context[3] > .5 else 'right'
    return f'{kind}:{side}:{round(float(context[5])*30)}:{round(float(context[8])*2)}:{round(float(context[7])*8)}'


def fit_event_tables(records):
    groups = {}
    for row in records:
        if row['mode'] not in VERIFIED_DEMONSTRATION_MODES or not row['result'].get('training_prefix'):
            continue
        if (row['result']['is_success'] or not set(range(1, row['goal']+1)).issubset(row['result']['recovered_ids'])
                or row['result']['catches'] < row['goal']):
            raise ValueError('Demonstration lacks physical recovered-flight evidence')
        for item in row['demonstrations']:
            if item['status'] != 'accepted':
                raise ValueError('A successful demonstration contains an unexecuted decision')
            obs = {name:np.asarray(value, dtype=np.float32) for name,value in item['observation'].items()}
            np.testing.assert_allclose(encode_command(TaskRequest(**item['request']),
                TaskCommand(**item['command'])), item['action'], atol=1e-12, rtol=0.)
            key = student_key(obs)
            groups.setdefault(key, []).append((features(obs)[LOCAL_INDICES], item['action']))
    if not groups:
        raise ValueError('No physically successful joint demonstration is available')
    tables = {}
    for key, pairs in groups.items():
        x, actions = np.asarray([v[0] for v in pairs]), np.asarray([v[1] for v in pairs])
        center, scale = x.mean(0), x.std(0)
        scale[scale < 1e-5] = 1.
        tables[key] = dict(center=center.tolist(), scale=scale.tolist(),
                           x=((x-center)/scale).tolist(), actions=actions.tolist())
    return tables


class EventStudent:
    """Small fixed-model pilot using only current actor observations."""
    def __init__(self, tables, mode, control_releases):
        self.tables, self.mode, self.control_releases = tables, mode, control_releases

    def predict(self, obs, deterministic=True):
        context = obs['context']
        opening = context[0] > .5 and context[5] == 0.
        if (self.mode == 'zero' or not np.any(context[:2]) or not action_authority(obs)
                or round(float(context[5])*30) >= self.control_releases):
            return np.zeros(6), None
        if self.mode == 'fixed_first':
            return (np.array([.1/.75, 0., .1/1.25, 0., 0., 0.]) if opening else np.zeros(6)), None
        if self.mode == 'first_only' and not opening:
            return np.zeros(6), None
        table = self.tables.get(student_key(obs))
        if table is None:
            return np.zeros(6), None
        if self.mode == 'fixed_event':
            action = np.asarray(table['actions']).mean(0)
        else:
            x = (features(obs)[LOCAL_INDICES]-table['center'])/np.asarray(table['scale'])
            index = int(np.argmin(np.mean((np.asarray(table['x'])-x)**2, axis=1)))
            action = np.asarray(table['actions'][index])
        return np.clip(action, -1., 1.), None


def student_trial(job):
    reference, seed, jitter, goal, control_releases, tables = job
    torch.set_num_threads(1)
    env = TaskCurriculum(TaskResidualEnv(TaskConfig(), reference),
                         [dict(catch_goal=goal, seat_jitter_m=jitter)])
    rows = []
    try:
        for mode in STUDENT_MODES:
            actor = EventStudent(tables, mode, control_releases)
            obs, _ = env.reset(seed=seed)
            while True:
                obs, _, done, truncated, info = env.step(actor.predict(obs)[0])
                if done or truncated:
                    break
            rows.append(dict(mode=mode, seed=seed, result=info))
            print('[Observation-only student]', mode, seed, info['reason'], flush=True)
        return rows
    finally:
        env.close()


def replay_trial(job):
    reference, source = job
    torch.set_num_threads(1)
    env = TaskCurriculum(TaskResidualEnv(TaskConfig(), reference),
                         [dict(catch_goal=source['goal'], seat_jitter_m=source['jitter_m'])])
    index = 0
    try:
        obs, _ = env.reset(seed=source['seed'])
        while True:
            current = env.unwrapped._request
            action = np.zeros(6)
            if current is not None and current.released < source['control_releases'] and action_authority(obs):
                if index >= len(source['demonstrations']):
                    raise ValueError('Replay exhausted before the physical recovery goal')
                item = source['demonstrations'][index]
                if item['key'] != context_key(current) or abs(item['request']['time_s']-current.time_s) > 1e-10:
                    raise ValueError('Replay callback differs from the recorded demonstration')
                for name, value in obs.items():
                    np.testing.assert_array_equal(value, np.asarray(item['observation'][name], np.float32))
                action = np.asarray(item['action'])
                index += 1
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        if (not info.get('training_prefix') or index != len(source['demonstrations'])
                or info['recovered_ids'] != source['result']['recovered_ids']):
            raise ValueError('Recorded commands did not reproduce the physically recovered prefix')
        print('[Demonstration replay]', source['seed'], index, info['reason'], flush=True)
        return dict(seed=source['seed'], decisions=index, observation_match=True, result=info,
                    scope='recorded-action capability replay, not a causal policy')
    finally:
        env.close()


def evaluate_demonstrations(args):
    folder = args.evaluate
    record_path, protocol_path = folder/'report.json', folder/'protocol.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    protocol = record['protocol']
    archive = folder/'driver_source.py'
    teacher_source = archive if archive.exists() else Path(__file__)
    if (protocol != json.loads(protocol_path.read_text(encoding='utf-8'))
            or protocol['source_sha256'] != runtime_sources()
            or protocol['reference_sha256'] != digest(args.reference)
            or protocol['driver_sha256'] != digest(teacher_source)
            or set(record['summary']) != set(summarize(record['records']))
            or any(record['summary'][mode].get(key, 0 if key=='candidate_mismatch' else None) != value
                   for mode,summary in summarize(record['records']).items() for key,value in summary.items())
            or protocol['goal'] != args.goal or protocol['control_releases'] != args.control_releases
            or protocol['jitter_mm'] != args.jitter_mm or set(protocol['seeds']) & set(args.seeds)):
        raise ValueError('Use matching teacher data and settings, with fresh independent test seeds')
    successful = [row for row in record['records']
                  if row['mode'] in VERIFIED_DEMONSTRATION_MODES and row['result'].get('training_prefix')]
    tables = fit_event_tables(successful)
    sources, driver_hash = runtime_sources(), digest(Path(__file__))
    feature_path = Path(__file__).with_name('audit_task_learning.py')
    inputs = {path.as_posix():digest(path) for path in (record_path, protocol_path, teacher_source,
                                                      args.reference, feature_path)}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'driver_source.py').open('xb').write(Path(__file__).read_bytes())
    model_path = args.output_dir/'event_tables.json'
    write_new(model_path, dict(tables=tables, feature_indices=LOCAL_INDICES, goal=args.goal,
        control_releases=args.control_releases, source_sha256=sources, input_sha256=inputs,
        training_seeds=[row['seed'] for row in successful],
        models='fixed event means and nearest measured demonstration; no hyperparameter tuning'))
    model_hash = digest(model_path)
    # Reload the frozen artifact. No observations or labels from the new test
    # scenes participate in fitting, scaling, model selection or calibration.
    tables = json.loads(model_path.read_text(encoding='utf-8'))['tables']
    with ProcessPoolExecutor(args.jobs) as pool:
        replay = list(pool.map(replay_trial, [(args.reference.as_posix(), row) for row in successful]))
        jobs = [(args.reference.as_posix(), seed, args.jitter_mm/1000., args.goal,
                 args.control_releases, tables) for seed in args.seeds]
        blocks = list(pool.map(student_trial, jobs))
    rows = [row for block in blocks for row in block]
    summary = {mode:dict(episodes=len(group), recovered=sum(bool(row['result'].get('training_prefix')) for row in group),
        unsafe=sum(row['result']['reason'] in TaskResidualEnv.UNSAFE for row in group))
        for mode in STUDENT_MODES if (group := [row for row in rows if row['mode'] == mode])}
    if (runtime_sources() != sources or digest(Path(__file__)) != driver_hash or digest(model_path) != model_hash
            or any(digest(Path(path)) != value for path,value in inputs.items())):
        raise ValueError('Frozen evaluation inputs changed')
    write_new(args.output_dir/'report.json', dict(scope='small independent multi-event imitation pilot; no RL acceptance',
        source_sha256=sources, driver_sha256=driver_hash, input_sha256=inputs, model_sha256=model_hash,
        test_seeds=args.seeds, training_seeds=[row['seed'] for row in successful],
        teacher_replay=replay, records=rows, summary=summary,
        complete_task_verified=False, rl_gain_verified=False))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--evaluate', type=Path, help='Fit fixed/nearest event selectors and evaluate on fresh seeds')
    source.add_argument('--oracle-input', type=Path, help='Revalidate an offline candidate trace in current physics')
    parser.add_argument('--seeds', type=int, nargs='+', default=[410400, 410401, 410402])
    parser.add_argument('--jitter-mm', type=float, default=.5)
    parser.add_argument('--goal', type=int, default=1)
    parser.add_argument('--control-releases', type=int, default=4)
    parser.add_argument('--modes', nargs='+', choices=MODES+('recorded_joint',), default=list(MODES))
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--planner-budget-s', type=float, default=120.)
    parser.add_argument('--trial-budget-s', type=float, default=300.)
    args = parser.parse_args()
    if (args.output_dir.exists() or not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or min(args.seeds) < 0 or args.jobs < 1 or len(set(args.modes)) != len(args.modes)
            or not 0. < args.jitter_mm <= 10. or not 1 <= args.goal < 30
            or not 1 <= args.control_releases <= 30
            or not np.isfinite(args.planner_budget_s) or args.planner_budget_s <= 0.
            or not np.isfinite(args.trial_budget_s) or args.trial_budget_s <= 0.):
        parser.error('Use new outputs, distinct seeds/modes and positive bounded settings')
    if args.evaluate:
        evaluate_demonstrations(args)
        return
    if bool(args.oracle_input) != (args.modes==['recorded_joint']):
        parser.error('Use --oracle-input with exactly --modes recorded_joint')
    if 'recorded_joint' in args.modes and not args.oracle_input:
        parser.error('Recorded candidate mode requires an oracle input')
    sources, driver_hash, reference_hash = runtime_sources(), digest(Path(__file__)), digest(args.reference)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'driver_source.py').open('xb').write(Path(__file__).read_bytes())
    protocol = dict(scope='multi-event prefix capability and privileged supervision; no RL acceptance',
        source_sha256=sources, driver_sha256=driver_hash, reference=args.reference.as_posix(),
        reference_sha256=reference_hash, task_config=asdict(TaskConfig()),
        seeds=args.seeds, jitter_mm=args.jitter_mm, goal=args.goal, modes=args.modes,
        control_releases=args.control_releases,
        planner_budget_s=args.planner_budget_s, trial_budget_s=args.trial_budget_s,
        optimizer_memory=('unused for recorded plans' if args.oracle_input else
                          'native per-throw retry guesses and physically executed launch history'),
        oracle_input=args.oracle_input.as_posix() if args.oracle_input else None,
        oracle_sha256=digest(args.oracle_input) if args.oracle_input else None,
        teacher=('recorded candidate plan revalidated in current physics; not a causal policy' if args.oracle_input else
                 'original planner on isolated state copies with matched low-level guards'),
        student_input='sampled estimates, proprioception, tactile measurements and event context',
        declined_queries='record and stop; never substitute clipped or unchecked commands')
    write_new(args.output_dir/'protocol.json', protocol)
    jobs = [(args.reference.as_posix(), seed, args.jitter_mm/1000., mode, args.goal, args.control_releases,
             args.planner_budget_s, args.trial_budget_s, str(args.oracle_input) if args.oracle_input else None)
            for seed in args.seeds for mode in args.modes]
    # Save each finished trial so expensive teacher computations remain reviewable.
    with ProcessPoolExecutor(args.jobs) as pool:
        completed = {}
        futures = {pool.submit(run_trial, job):index for index,job in enumerate(jobs)}
        for future in as_completed(futures):
            index, row = futures[future], future.result()
            completed[index] = row
            write_new(args.output_dir/f'trial_{index:03d}.json', row)
        rows = [completed[index] for index in sorted(completed)]
    if (sources != runtime_sources() or driver_hash != digest(Path(__file__))
            or reference_hash != digest(args.reference)
            or args.oracle_input and digest(args.oracle_input)!=protocol['oracle_sha256']):
        raise ValueError('Demonstration sources or reference changed')
    write_new(args.output_dir/'report.json', dict(protocol=protocol, records=rows, summary=summarize(rows),
        complete_task_verified=False, rl_gain_verified=False))
    print(json.dumps(summarize(rows)), flush=True)


if __name__ == '__main__':
    main()

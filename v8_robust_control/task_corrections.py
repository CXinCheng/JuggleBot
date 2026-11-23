"""Collect learner-visited correction labels and run a fixed-budget actor pilot."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from .compare_residual import digest
from .task_control import TaskRequest, TaskCommand, decode_action
from .task_demonstrations import (query_planner, encode_command, EventStudent,
                                  student_key)
from .audit_task_learning import features, LOCAL_INDICES
from .task_env import TaskConfig, TaskResidualEnv, runtime_sources, action_authority, context_key
from .task_imitation import (load_demonstrations, validate_episode_splits, tensors,
                             fit_actor, loss_value, rollout, rollout_metrics, evaluate_training_cases)
from .train_task import TaskCurriculum, write_new


@dataclass(frozen=True)
class PartialActorConfig:
    learning_rate: float = 3e-4
    final_learning_rate: float = 1e-6
    anchor_weight: float = 100.

    def __post_init__(self):
        if (not np.isfinite([self.learning_rate,self.final_learning_rate,self.anchor_weight]).all()
                or not 0. < self.final_learning_rate <= self.learning_rate
                or self.anchor_weight < 0.):
            raise ValueError('Require finite nonnegative anchor weight and 0 < final rate <= initial rate')


def load_parent(folder, checkpoint, reference):
    """Verify the frozen initialization and its generator, without relabeling it."""
    manifest = json.loads((folder/'training.json').read_text(encoding='utf-8'))
    protocol = manifest['protocol']
    archive = folder/'driver_source.py'
    model = PPO.load(folder/checkpoint, device='cpu')
    entries = [row for row in manifest['models'] if row['checkpoint'] == checkpoint]
    if (len(entries) != 1 or entries[0]['checkpoint_sha256'] != digest(folder/checkpoint)
            or entries[0]['epoch'] != protocol['epochs']
            or model.task_runtime_sha256 != runtime_sources()
            or model.task_reference_sha256 != digest(reference)
            or model.task_imitation_sha256 != digest(archive)
            or model.policy.control_releases != protocol['control_releases']):
        raise ValueError('Require the verified full-budget initialization and matching runtime/reference')
    return model, protocol


def correction_rows(records):
    """Accepted local labels remain useful even if the assisted episode later fails."""
    rows = []
    for row in records:
        accepted = []
        for item in row['corrections']:
            if item['status'] != 'accepted':
                continue
            np.testing.assert_allclose(encode_command(TaskRequest(**item['request']),
                TaskCommand(**item['command'])), item['action'], atol=1e-12, rtol=0.)
            if item['executed_by'] not in ('learner', 'teacher'):
                raise ValueError('Correction lacks action provenance')
            target = item['learner_action'] if item['executed_by']=='learner' else item['action']
            np.testing.assert_allclose(item['executed_action'], target, atol=0., rtol=0.)
            accepted.append(item)
        if accepted:
            rows.append(dict(seed=row['seed'], jitter_m=row['jitter_m'], demonstrations=accepted))
    return rows


def collect_trial(job):
    folder, checkpoint, reference, seed, jitter, goal, window, learner_steps, query_s, trial_s = job
    torch.set_num_threads(1)
    actor, _ = load_parent(folder, checkpoint, reference)
    env = TaskCurriculum(TaskResidualEnv(TaskConfig(), reference),
                         [dict(catch_goal=goal, seat_jitter_m=jitter)])
    started, memory, records, active_steps = time.perf_counter(), {}, [], 0
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            request = env.unwrapped._request
            action = np.asarray(actor.predict(obs, deterministic=True)[0], dtype=float)
            if request is not None and request.released < window and action_authority(obs):
                print('[Correction query]', seed, context_key(request), active_steps, flush=True)
                item = dict(key=context_key(request), request=asdict(request),
                    observation={k:v.tolist() for k,v in obs.items()}, learner_action=action.tolist(),
                    learner_prefix=active_steps < learner_steps,
                    **query_planner(env.unwrapped.controller, request, query_s,
                                    started+trial_s, memory))
                records.append(item)
                print('[Correction label]', seed, item['status'], flush=True)
                if item['status'] != 'accepted':
                    info = dict(env.unwrapped._info('TEACHER_'+item['status'].upper()),
                                training_prefix=False, is_success=False, diagnostic_abort=True,
                                terminated=False, truncated=True)
                    break
                memory = copy.deepcopy(item.get('optimizer_state', {}))
                item['executed_by'] = 'learner' if active_steps < learner_steps else 'teacher'
                if item['executed_by'] == 'teacher':
                    action = np.asarray(item['action'], dtype=float)
                item['executed_action'] = action.tolist()
                item['label_action_error'] = float(np.linalg.norm(action-np.asarray(item['action'])))
                active_steps += 1
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        return dict(seed=seed, jitter_m=jitter, result=info, corrections=records,
                    wall_s=time.perf_counter()-started,
                    role=('learner prefix followed by privileged teacher' if learner_steps
                          else 'privileged teacher from the initial state'),
                    complete_task_verified=False, rl_gain_verified=False)
    finally:
        env.close()


def evaluate_trial(job):
    parent_folder, checkpoint, corrected, reference, seed, jitter, goal, tables = job
    torch.set_num_threads(1)
    parent, protocol = load_parent(parent_folder, checkpoint, reference)
    model = PPO.load(corrected, device='cpu')
    env = TaskCurriculum(TaskResidualEnv(TaskConfig(), reference),
                         [dict(catch_goal=goal, seat_jitter_m=jitter)])
    try:
        actors = [('parent', parent), ('corrected', model),
                  ('zero', EventStudent(tables, 'zero', protocol['control_releases'])),
                  ('fixed_first', EventStudent(tables, 'fixed_first', protocol['control_releases'])),
                  ('fixed_event', EventStudent(tables, 'fixed_event', protocol['control_releases']))]
        return [rollout(env, actor, seed, name) for name,actor in actors]
    finally:
        env.close()


def run_jobs(function, jobs, workers, output, prefix):
    rows = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, job):i for i,job in enumerate(jobs)}
        for future in as_completed(futures):
            i = futures[future]
            rows[i] = future.result()
            write_new(output/f'{prefix}_{i:03d}.json', rows[i])
    return [rows[i] for i in range(len(jobs))]


def validate_new_seeds(train, validation, test, parent_protocol, allow_empty=False):
    forbidden = set(parent_protocol['demonstration_attempt_seeds']+
                    parent_protocol['validation_episode_seeds']+parent_protocol['test_episode_seeds'])
    if ((not train and not allow_empty) or len(set(train)) != len(train)
            or any(seed<0 for seed in train) or set(train)&forbidden):
        raise ValueError('Correction training must use fresh whole initial states')
    validate_episode_splits(validation, test, forbidden|set(train))


def load_corrections(folders, reference, parent_folder, checkpoint, goal, window):
    """Reuse training labels while keeping every earlier holdout reserved."""
    records, inputs, reserved = [], {}, set()
    for folder in folders:
        path, archive = folder/'corrections.json', folder/'driver_source.py'
        data = json.loads(path.read_text(encoding='utf-8'))
        protocol = data['protocol']
        driver_hashes = [v for k,v in protocol['input_sha256'].items()
                         if Path(k).name == 'task_corrections.py']
        if (protocol['source_sha256'] != runtime_sources()
                or protocol['parent_checkpoint'] != checkpoint
                or protocol['parent_training_sha256'] != digest(parent_folder/'training.json')
                or protocol['goal'] != goal or protocol['control_releases'] != window
                or driver_hashes != [digest(archive)]):
            raise ValueError('Correction archive is incompatible with the frozen parent or runtime')
        # The archived driver identifies the historical collector. All its other
        # inputs must still match; never pretend an old record used this driver.
        for name, expected in protocol['input_sha256'].items():
            source = archive if Path(name).name == 'task_corrections.py' else Path(name)
            if digest(source) != expected:
                raise ValueError('Correction archive input changed: '+name)
        if digest(reference) != protocol['input_sha256'].get(reference.as_posix()):
            raise ValueError('Correction reference differs')
        seeds = [row['seed'] for row in data['records']]
        if (sorted(seeds) != sorted(protocol['train_seeds']+protocol.get('reused_training_seeds', []))
                or set(seeds)&reserved):
            raise ValueError('Duplicate or unrecorded correction initial states')
        records.extend(data['records'])
        reserved.update(protocol['train_seeds']+protocol['validation_seeds']+protocol['test_seeds'])
        reserved.update(protocol.get('reserved_previous_seeds', []))
        for item in (path, archive):
            inputs[item.as_posix()] = digest(item)
    correction_rows(records)
    return records, inputs, reserved


def fit_retained_head(model, anchors, corrections, ridge=.01, max_update=.5):
    """Ridge fit in the anchor feature nullspace; no trunk/critic/statistics update."""
    policy = model.policy
    policy.set_training_mode(False)
    def features(batch):
        with torch.no_grad():
            values = policy.extract_features(batch[0])
            if isinstance(values, tuple):
                values = values[0]
            latent = policy.mlp_extractor.forward_actor(values)
        return np.column_stack([latent.cpu().numpy(), np.ones(len(latent))]).astype(np.float64)
    a, x = features(anchors), features(corrections)
    _, singular, vh = np.linalg.svd(a, full_matrices=True)
    rank = int(np.sum(singular > max(a.shape)*np.finfo(np.float64).eps*singular[0]))
    null = vh[rank:].T
    head = policy.action_net
    before = np.column_stack([head.weight.detach().cpu().numpy(),
                              head.bias.detach().cpu().numpy()]).astype(np.float64)
    # Tanh-normal actions have finite inverse targets only inside the bounds.
    target = np.arctanh(np.clip(corrections[1].numpy(), -.999999, .999999))
    weight = np.sqrt(corrections[2].numpy())[:, None]
    design = (x@null)*weight
    delta = np.linalg.solve(design.T@design+ridge*np.eye(null.shape[1]),
                            design.T@((target-x@before.T)*weight)).T@null.T
    norm = float(np.linalg.norm(delta))
    delta *= min(1., max_update/max(norm, 1e-12))
    return before, delta, dict(method='anchor-nullspace ridge output-head fit', anchor_rank=rank,
        free_features=null.shape[1], ridge=ridge, max_update=max_update,
        raw_update_norm=norm, update_norm=float(np.linalg.norm(delta)),
        anchor_linear_drift=float(np.max(np.abs(a@delta.T))))


def apply_head(model, before, delta, fraction):
    with torch.no_grad():
        values = torch.as_tensor(before+fraction*delta, dtype=model.policy.action_net.weight.dtype)
        model.policy.action_net.weight.copy_(values[:, :-1])
        model.policy.action_net.bias.copy_(values[:, -1])


def fit_partial_actor(model, anchors, corrections, epochs, on_checkpoint=None, config=None):
    """Train only the final actor hidden layer and output, with parent distillation."""
    policy = model.policy
    config = config or PartialActorConfig()
    hidden = [layer for layer in policy.mlp_extractor.policy_net if isinstance(layer, torch.nn.Linear)][-1]
    parameters = list(hidden.parameters())+list(policy.action_net.parameters())
    selected = {id(p) for p in parameters}
    flags = [(p, p.requires_grad) for p in policy.parameters()]
    frozen = {name:p.detach().clone() for name,p in policy.named_parameters() if id(p) not in selected}
    with torch.no_grad():
        targets = policy._predict(anchors[0], deterministic=True).detach().clone()
    distillation = (anchors[0], targets, anchors[2])
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.final_learning_rate)
    checkpoints = {0, max(1,epochs//4), max(1,epochs//2), epochs}
    history = []
    try:
        for parameter, _ in flags:
            parameter.requires_grad_(id(parameter) in selected)
        policy.set_training_mode(True)
        for epoch in range(epochs+1):
            if epoch % 100 == 0 or epoch in checkpoints:
                with torch.no_grad():
                    row = dict(epoch=epoch, correction_mse=float(loss_value(policy, corrections)),
                               anchor_distillation_mse=float(loss_value(policy, distillation)))
                history.append(row)
                if epoch in checkpoints and on_checkpoint:
                    on_checkpoint(epoch, model, row)
            if epoch == epochs:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = loss_value(policy, corrections)+config.anchor_weight*loss_value(policy, distillation)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite partial actor loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            scheduler.step()
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        policy.set_training_mode(False)
    for name, original in frozen.items():
        if not torch.equal(original, dict(policy.named_parameters())[name]):
            raise AssertionError('Partial training changed a frozen parameter: '+name)
    return dict(method='last actor hidden layer and output; parent action distillation', epochs=epochs,
        **asdict(config),
        trainable_parameters=sum(p.numel() for p in parameters), history=history)


def local_event_tables(rows):
    """A fixed-event control receives the same locally accepted training labels."""
    groups = {}
    for row in rows:
        for item in row['demonstrations']:
            obs = {k:np.asarray(v, np.float32) for k,v in item['observation'].items()}
            groups.setdefault(student_key(obs), []).append((features(obs)[LOCAL_INDICES], item['action']))
    tables = {}
    for key, pairs in groups.items():
        x, actions = np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])
        if actions.shape[1:] != (6,) or not np.isfinite(actions).all() or np.any(np.abs(actions)>1.):
            raise ValueError('Invalid local baseline targets')
        center, scale = x.mean(0), x.std(0)
        scale[scale<1e-5] = 1.
        tables[key] = dict(center=center.tolist(), scale=scale.tolist(),
                           x=((x-center)/scale).tolist(), actions=actions.tolist())
    return tables


def retention_selection(candidates, parent_rows, goal):
    """Preserve each anchor's physical recovery and safety before ranking validation."""
    original = {r['seed']:r['result'] for r in parent_rows}
    eligible = []
    for candidate in candidates:
        rows = candidate['anchor_rollouts']
        valid = {r['seed'] for r in rows} == set(original) and len(rows)==len(original)
        for row in rows:
            baseline, result = original[row['seed']], row['result']
            if baseline.get('training_prefix'):
                valid &= bool(result.get('training_prefix')) and set(range(1,goal+1)).issubset(result['recovered_ids'])
            valid &= not (result['reason'] in TaskResidualEnv.UNSAFE and baseline['reason'] not in TaskResidualEnv.UNSAFE)
        candidate['anchors_retained'] = bool(valid)
        if valid:
            eligible.append(candidate)
    if not eligible:
        raise ValueError('The unchanged parent must remain an eligible fallback')
    def key(candidate):
        rows = candidate['validation_rollouts']
        metrics = rollout_metrics(rows)
        return (-metrics['unsafe'], metrics['recovered'],
                sum(len(r['result']['recovered_ids']) for r in rows), -candidate['fraction'])
    return max(eligible, key=key)


def retention_trial(job):
    checkpoint, reference, goal, jitter, anchors, validation = job
    torch.set_num_threads(1)
    actor = PPO.load(checkpoint, device='cpu')
    env = TaskCurriculum(TaskResidualEnv(TaskConfig(), reference),
                         [dict(catch_goal=goal, seat_jitter_m=jitter)])
    try:
        return dict(checkpoint=Path(checkpoint).name,
            anchor_rollouts=evaluate_training_cases(env, actor, anchors),
            validation_rollouts=[rollout(env, actor, s, 'retention_validation') for s in validation])
    finally:
        env.close()


def command_difference(a, b):
    delta = np.asarray(b['parameters'])-np.asarray(a['parameters'])
    delay, tilt = b['defer_s']-a['defer_s'], b['tilt_scale']-a['tilt_scale']
    return dict(parameters=delta.tolist(), defer_s=float(delay), tilt_scale=float(tilt),
                changed=bool(np.max(np.abs(delta))>1e-12 or abs(delay)>1e-12 or abs(tilt)>1e-12))


def diagnostic_rollout(env, parent, candidate, seed, label, gain=1., prefix=None, single=None, components=None):
    """Intervene only through normal bounded actions; both actors see current measurements."""
    obs, _ = env.reset(seed=seed)
    trace = []
    while True:
        request = env.unwrapped._request
        a = np.asarray(parent.predict(obs, deterministic=True)[0], dtype=float)
        b = np.asarray(candidate.predict(obs, deterministic=True)[0], dtype=float)
        index = len(trace)
        active = (prefix is None or index<prefix) and (single is None or index==single)
        weight = gain if active else 0.
        delta = b-a
        if components is not None:
            filtered = np.zeros(6)
            filtered[list(components)] = delta[list(components)]
            delta = filtered
        action = a+weight*delta
        if not 0. <= gain <= 1. or np.any(np.abs(action)>1.+1e-12):
            raise ValueError('Diagnostic gain must preserve the original bounded action space')
        action = np.clip(action, -1., 1.)  # floating-point roundoff only
        if request is not None:
            authority = action_authority(obs)
            trace.append(dict(key=f'{context_key(request)}:{request.attempt}',
                time_s=request.time_s, request=asdict(request),
                observation={k:v.tolist() for k,v in obs.items()}, parent_action=a.tolist(),
                candidate_action=b.tolist(), executed_action=action.tolist(), gain=weight,
                command=asdict(decode_action(request, action*authority))))
        obs, _, done, truncated, info = env.step(action)
        if done or truncated:
            break
    print('[Closed-loop diagnostic]', seed, label, info['reason'], flush=True)
    events = copy.deepcopy(getattr(getattr(env.unwrapped,'controller',None),'events',[]))
    return dict(label=label, seed=seed, jitter_m=info['config']['seat_jitter_m'], trace=trace, result=info,
                physical_events=events)


def compare_decision_traces(parent, candidate):
    """Stop pairing when callback identity or simulation clocks differ."""
    pairs, first_command, first_observation, alignment_lost = [], None, None, None
    for index,(a,b) in enumerate(zip(parent['trace'], candidate['trace'])):
        if a['key']!=b['key'] or abs(a['time_s']-b['time_s'])>1e-9:
            alignment_lost = dict(index=index, parent_key=a['key'], candidate_key=b['key'],
                                  time_difference_s=b['time_s']-a['time_s'])
            break
        delta = command_difference(a['command'],b['command'])
        error = float(np.max(np.abs(features(a['observation'])-features(b['observation']))))
        if first_command is None and delta['changed']:
            first_command = dict(index=index, key=a['key'], time_s=a['time_s'], delta=delta)
        if first_observation is None and error>0.:
            first_observation = dict(index=index, key=a['key'], time_s=a['time_s'], normalized_linf=error)
        ar, br = a['request'], b['request']
        palm = 1000.*np.max(np.linalg.norm(np.asarray(ar['palm_position_m'])-br['palm_position_m'],axis=1))
        balls = []
        for x,y in zip(ar['balls'], br['balls']):
            if (x and y and x['valid'] and y['valid']
                    and abs(x['sample_time_s']-y['sample_time_s'])<1e-9):
                balls.append(float(1000.*np.linalg.norm(np.asarray(x['position_m'])-y['position_m'])))
        pairs.append(dict(index=index, key=a['key'], time_s=a['time_s'], observation_linf=error,
                          palm_position_delta_mm=float(palm),
                          sampled_ball_position_delta_mm=max(balls) if balls else None))
    if alignment_lost is None and len(parent['trace'])!=len(candidate['trace']):
        alignment_lost = dict(index=len(pairs), reason='different callback counts')
    return dict(first_command_change=first_command, first_observation_change=first_observation,
                alignment_lost=alignment_lost, paired_decisions=pairs)


def diagnose_seed(job):
    parent_folder, parent_checkpoint, candidate_path, reference, source = job
    torch.set_num_threads(1)
    parent, _ = load_parent(parent_folder,parent_checkpoint,reference)
    candidate = PPO.load(candidate_path,device='cpu')
    env = TaskCurriculum(TaskResidualEnv(TaskConfig(),reference),
                         [dict(catch_goal=source['goal'],seat_jitter_m=source['jitter_m'])])
    seed = source['seed']
    try:
        baseline = diagnostic_rollout(env,parent,candidate,seed,'parent',gain=0.)
        changed = diagnostic_rollout(env,parent,candidate,seed,'candidate')
        paired = compare_decision_traces(baseline,changed)
        first = paired['first_command_change']
        interventions = []
        if first:
            interventions.append(diagnostic_rollout(env,parent,candidate,seed,'single_first_change',single=first['index']))
            for label, axes in [('first_velocity',(0,1,2)),('first_normal',(3,)),
                                ('first_finger_lead',(4,)),('first_wait',(5,))]:
                interventions.append(diagnostic_rollout(env,parent,candidate,seed,label,
                                                       single=first['index'],components=axes))
        for label, options in [('candidate_prefix2',dict(prefix=2)),('candidate_prefix4',dict(prefix=4)),
                               ('gain_1of16',dict(gain=.0625)),('gain_quarter',dict(gain=.25))]:
            interventions.append(diagnostic_rollout(env,parent,candidate,seed,label,**options))
        return dict(seed=seed,jitter_m=source['jitter_m'],parent=baseline,candidate=changed,
                    comparison=paired,interventions=interventions)
    finally:
        env.close()


def diagnosis_main(argv):
    parser = argparse.ArgumentParser(description='Paired training-state traces and bounded action interventions')
    parser.add_argument('--parent-dir',type=Path,required=True)
    parser.add_argument('--parent-checkpoint',default='ppo_bc_seed0_epoch2000.zip')
    parser.add_argument('--candidate-dir',type=Path,required=True)
    parser.add_argument('--candidate',default='partial_epoch6000.zip')
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--demonstrations',type=Path,nargs='+',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=1)
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.jobs<1:
        parser.error('Use a new output directory and positive jobs')
    parent, parent_protocol = load_parent(args.parent_dir,args.parent_checkpoint,args.reference)
    goal, window = parent_protocol['goal'],parent_protocol['control_releases']
    anchors, inputs, _ = load_demonstrations(args.demonstrations,args.reference,goal,window)
    _, archives, _ = load_corrections([args.candidate_dir],args.reference,args.parent_dir,
                                    args.parent_checkpoint,goal,window)
    inputs.update(archives)
    selection_path = args.candidate_dir/'selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    candidate_path = args.candidate_dir/args.candidate
    entries = [r for r in selection['candidates'] if r['checkpoint']==args.candidate]
    if len(entries)!=1 or digest(candidate_path)!=entries[0]['checkpoint_sha256']:
        parser.error('Require an unchanged recorded candidate checkpoint')
    for path in (args.parent_dir/'training.json',args.parent_dir/args.parent_checkpoint,
                 candidate_path,selection_path,args.reference,Path(__file__)):
        inputs[path.as_posix()] = digest(path)
    protocol = dict(scope='training-state closed-loop diagnosis; no model selection or independent-test claim',
        input_sha256=inputs,source_sha256=runtime_sources(),seeds=[r['seed'] for r in anchors],
        candidate=args.candidate,goal=goal,control_releases=window,
        intervention_rule='Only bounded policy actions; no live ball-state or force modifications',
        physical_events='Read-only simulator event diagnostics; excluded from policy inputs',
        labels_used_for_training=False)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'driver_source.py').write_bytes(Path(__file__).read_bytes())
    write_new(args.output_dir/'protocol.json',protocol)
    rows = run_jobs(diagnose_seed,[(args.parent_dir,args.parent_checkpoint,candidate_path,args.reference,r)
                                 for r in anchors],args.jobs,args.output_dir,'trace')
    for path,expected in inputs.items():
        if digest(Path(path))!=expected:
            raise ValueError('Diagnostic input changed: '+path)
    if runtime_sources()!=protocol['source_sha256']:
        raise ValueError('Diagnostic runtime changed')
    summary = [dict(seed=r['seed'],jitter_m=r['jitter_m'],first_command_change=r['comparison']['first_command_change'],
        first_observation_change=r['comparison']['first_observation_change'],alignment_lost=r['comparison']['alignment_lost'],
        outcomes={b['label']:dict(reason=b['result']['reason'],recovered_ids=b['result']['recovered_ids'])
                  for b in [r['parent'],r['candidate']]+r['interventions']}) for r in rows]
    write_new(args.output_dir/'report.json',dict(protocol=protocol,records=rows,summary=summary,
        complete_task_verified=False,rl_gain_verified=False))
    print(json.dumps(summary),flush=True)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='diagnose':
        diagnosis_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', default='ppo_bc_seed0_epoch2000.zip')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--demonstrations', type=Path, nargs='+', required=True)
    parser.add_argument('--train-seeds', type=int, nargs='+', default=[])
    parser.add_argument('--validation-seeds', type=int, nargs='+', required=True)
    parser.add_argument('--test-seeds', type=int, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--correction-input', type=Path, nargs='*', default=[])
    parser.add_argument('--retain-anchors', action='store_true')
    parser.add_argument('--partial-actor', action='store_true')
    parser.add_argument('--actor-learning-rate', type=float, default=3e-4)
    parser.add_argument('--actor-final-learning-rate', type=float, default=1e-6)
    parser.add_argument('--anchor-weight', type=float, default=100.)
    parser.add_argument('--learner-steps', type=int, default=2)
    parser.add_argument('--jitter-mm', type=float, default=.5)
    parser.add_argument('--planner-budget-s', type=float, default=120.)
    parser.add_argument('--trial-budget-s', type=float, default=360.)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if (args.output_dir.exists() or not 0 <= args.learner_steps <= 8 or args.epochs < 25
            or args.jobs < 1 or not 0. < args.jitter_mm <= 10.
            or not np.isfinite(args.planner_budget_s) or not 0. < args.planner_budget_s <= 600.
            or not np.isfinite(args.trial_budget_s) or not 0. < args.trial_budget_s <= 1800.):
        parser.error('Use new outputs and finite, positive, bounded pilot settings')
    if args.partial_actor and not args.retain_anchors:
        parser.error('--partial-actor requires physical anchor regression via --retain-anchors')
    try:
        partial_config = PartialActorConfig(args.actor_learning_rate,args.actor_final_learning_rate,args.anchor_weight)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(1)
    model, parent_protocol = load_parent(args.parent_dir, args.checkpoint, args.reference)
    validate_new_seeds(args.train_seeds, args.validation_seeds, args.test_seeds, parent_protocol,
                       allow_empty=bool(args.correction_input))
    goal, window = parent_protocol['goal'], parent_protocol['control_releases']
    anchors, inputs, source_seeds = load_demonstrations(args.demonstrations, args.reference, goal, window)
    previous, reused_inputs, reserved = load_corrections(args.correction_input, args.reference,
        args.parent_dir, args.checkpoint, goal, window)
    inputs.update(reused_inputs)
    validate_episode_splits(args.validation_seeds, args.test_seeds, reserved|source_seeds|set(args.train_seeds))
    if set(args.train_seeds)&reserved:
        parser.error('New correction collection must not reuse earlier training or holdout states')
    validate_episode_splits(args.validation_seeds, args.test_seeds, source_seeds|set(args.train_seeds))
    if set(args.train_seeds)&source_seeds or not anchors:
        parser.error('Use fresh correction seeds and verified anchor episodes')
    for path in (Path(__file__), Path(__file__).with_name('task_imitation.py'),
                 Path(__file__).with_name('task_demonstrations.py'), args.reference,
                 args.parent_dir/'training.json', args.parent_dir/'driver_source.py',
                 args.parent_dir/args.checkpoint):
        inputs[path.as_posix()] = digest(path)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'driver_source.py').write_bytes(Path(__file__).read_bytes())
    protocol = dict(scope='Correction-label supervised pilot; not RL or full-task acceptance',
        source_sha256=runtime_sources(), input_sha256=inputs, parent_checkpoint=args.checkpoint,
        parent_training_sha256=digest(args.parent_dir/'training.json'), goal=goal, control_releases=window,
        train_seeds=args.train_seeds, validation_seeds=args.validation_seeds, test_seeds=args.test_seeds,
        jitter_m=args.jitter_mm/1000., learner_steps=args.learner_steps,
        planner_budget_s=args.planner_budget_s, trial_budget_s=args.trial_budget_s, epochs=args.epochs,
        label_rule='Accepted bounded local planner labels, including failed assisted episodes; no success claim',
        fit_rule='Equal episode weighting, anchors plus corrections; frozen parent normalization/critic',
        evaluation_rule='Fixed final epoch, no checkpoint or hyperparameter selection from validation/test',
        task_config=asdict(TaskConfig()))
    protocol.update(reused_training_seeds=[r['seed'] for r in previous],
                    reserved_previous_seeds=sorted(reserved), retain_anchors=args.retain_anchors,
                    partial_actor=args.partial_actor,
                    fixed_event_data='Same anchors and locally accepted correction labels as the actor')
    if args.retain_anchors:
        protocol.update(fit_rule='Frozen trunk/critic/statistics; ridge=0.01, output-head delta norm<=0.5 in anchor nullspace',
            evaluation_rule='Retain physical anchor recoveries; rank fixed fractions by validation safety/recovery/progress; test once',
            checkpoint_fractions=[0., .0625, .25, 1.])
    if args.partial_actor:
        protocol.pop('checkpoint_fractions', None)
        protocol.update(fit_rule='Final actor hidden layer and output; cosine Adam schedule with parent distillation',
                        partial_actor_config=asdict(partial_config),
                        checkpoint_epochs=sorted({0,max(1,args.epochs//4),max(1,args.epochs//2),args.epochs}))
    write_new(args.output_dir/'protocol.json', protocol)
    jobs = [(args.parent_dir,args.checkpoint,args.reference,seed,args.jitter_mm/1000.,goal,window,
             args.learner_steps,args.planner_budget_s,args.trial_budget_s) for seed in args.train_seeds]
    collected = run_jobs(collect_trial, jobs, args.jobs, args.output_dir, 'collection')
    corrections = correction_rows(previous+collected)
    write_new(args.output_dir/'corrections.json', dict(protocol=protocol, records=previous+collected))
    if not corrections:
        write_new(args.output_dir/'report.json', dict(protocol=protocol, records=collected,
            status='no_accepted_corrections', complete_task_verified=False, rl_gain_verified=False))
        return
    tables = local_event_tables(anchors+corrections)
    write_new(args.output_dir/'fixed_event.json', tables)
    env = TaskCurriculum(TaskResidualEnv(TaskConfig(), args.reference),
                         [dict(catch_goal=goal, seat_jitter_m=args.jitter_mm/1000.)])
    try:
        before_critic = {k:v.clone() for k,v in model.policy.state_dict().items()
                         if 'value_net' in k or 'features_extractor' in k}
        if args.retain_anchors:
            candidates = []
            if args.partial_actor:
                def save_epoch(epoch, actor, losses):
                    candidate = args.output_dir/f'partial_epoch{epoch}.zip'
                    actor.save(candidate)
                    candidates.append(dict(fraction=epoch/args.epochs, epoch=epoch, checkpoint=candidate.name,
                                           checkpoint_sha256=digest(candidate), losses=losses))
                fit = fit_partial_actor(model, tensors(anchors, env.observation_space),
                                        tensors(corrections, env.observation_space), args.epochs, save_epoch,
                                        config=partial_config)
            else:
                before, delta, fit = fit_retained_head(model, tensors(anchors, env.observation_space),
                                                      tensors(corrections, env.observation_space))
                for fraction in protocol['checkpoint_fractions']:
                    apply_head(model, before, delta, fraction)
                    candidate = args.output_dir/f'head_{fraction:g}.zip'
                    model.save(candidate)
                    candidates.append(dict(fraction=fraction, checkpoint=candidate.name,
                                           checkpoint_sha256=digest(candidate)))
            anchor_cases = [dict(seed=r['seed'],jitter_m=r['jitter_m']) for r in anchors]
            jobs = [(args.output_dir/c['checkpoint'], args.reference, goal, args.jitter_mm/1000.,
                     anchor_cases, args.validation_seeds) for c in candidates]
            physical = run_jobs(retention_trial, jobs, args.jobs, args.output_dir, 'retention')
            for candidate, result in zip(candidates, physical):
                candidate.update(result)
            chosen = retention_selection(candidates, candidates[0]['anchor_rollouts'], goal)
            write_new(args.output_dir/'selection.json', dict(candidates=candidates, selected=chosen,
                scope='anchor regression and validation only; independent test not read'))
            model = PPO.load(args.output_dir/chosen['checkpoint'], device='cpu')
            fit.update(selected_fraction=chosen['fraction'], anchors_retained=chosen['anchors_retained'])
            if args.partial_actor:
                fit.update(selected_epoch=chosen['epoch'])
        else:
            batch = tensors(anchors+corrections, env.observation_space)
            fit = fit_actor(model, batch, args.epochs)
        for key,value in before_critic.items():
            if not torch.equal(value, model.policy.state_dict()[key]):
                raise AssertionError('Fine-tuning changed critic or frozen normalization')
        model.task_corrections_sha256 = digest(Path(__file__))
        model.task_parent_sha256 = digest(args.parent_dir/args.checkpoint)
        path = args.output_dir/'corrected.zip'
        model.save(path)
        write_new(args.output_dir/'fitting.json', dict(protocol=protocol, fitting=fit,
            checkpoint_sha256=digest(path), accepted_labels=sum(len(r['demonstrations']) for r in corrections)))
        training_rows = evaluate_training_cases(env, model,
            [dict(seed=r['seed'],jitter_m=r['jitter_m']) for r in anchors+corrections])
    finally:
        env.close()
    evaluated = {}
    for split, seeds in [('validation',args.validation_seeds),('test',args.test_seeds)]:
        jobs = [(args.parent_dir,args.checkpoint,path,args.reference,seed,args.jitter_mm/1000.,goal,tables)
                for seed in seeds]
        evaluated[split] = [r for rows in run_jobs(evaluate_trial, jobs, args.jobs, args.output_dir, split) for r in rows]
    for name,expected in inputs.items():
        if digest(Path(name)) != expected:
            raise ValueError('Experiment input changed during the pilot')
    if runtime_sources() != protocol['source_sha256']:
        raise ValueError('Runtime changed during the pilot')
    write_new(args.output_dir/'report.json', dict(protocol=protocol, records=collected,
        fitting=fit, training_rollouts=training_rows, evaluated=evaluated,
        summary={split:{mode:rollout_metrics([r for r in rows if r['mode']==mode])
                        for mode in ('parent','corrected','zero','fixed_first','fixed_event')}
                 for split,rows in evaluated.items()},
        complete_task_verified=False, rl_gain_verified=False))


if __name__ == '__main__':
    main()

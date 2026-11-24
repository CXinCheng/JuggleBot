"""Independent complete-task evaluation, with robustness reported separately."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from .compare_residual import digest, read_json
from .train_cascade import (sources, summarize, counts, qualifies, dominates,
                            test_one, validate_trials, runtime_compatible, CASES)


PERCEPTION_CASES = {
    'clean': {},
    'sensor_only_3mm': dict(noise_std_m=.003),
    'noise_1mm': dict(noise_std_m=.001),
    'noise_5mm': dict(noise_std_m=.005),
    'noise_10mm': dict(noise_std_m=.01),
    'delay_20ms': dict(delay_s=.02),
    'delay_40ms': dict(delay_s=.04),
    'dropout_10pct': dict(dropout=.1),
    'dropout_30pct': dict(dropout=.3),
    'perception_combined': dict(noise_std_m=.003, delay_s=.02, dropout=.1),
}


def evaluation_cases(protocol):
    suite = protocol.get('suite', 'reference')
    if suite not in ('reference', 'perception'):
        raise ValueError('Unknown independent evaluation suite')
    cases = PERCEPTION_CASES if suite == 'perception' else CASES
    if protocol.get('evaluation_cases', cases) != cases:
        raise ValueError('Independent evaluation cases differ from the frozen suite')
    return cases


def assess(values, episodes):
    expected = ['zero', 'timed']+[f'{kind}_seed{seed}' for seed in (0, 1, 2) for kind in ('live', 'frozen')]
    if list(values) != expected or episodes < 1:
        raise ValueError('Require every selected policy and control')
    nominal = {str(seed): values[f'live_seed{seed}']['clean'] == episodes for seed in (0, 1, 2)}
    nondegradation = {str(seed): qualifies(values[f'live_seed{seed}'], values['zero']) for seed in (0, 1, 2)}
    advantage = {str(seed): all(dominates(values[f'live_seed{seed}'], values[name])
        for name in ('zero', 'timed', f'frozen_seed{seed}')) for seed in (0, 1, 2)}
    return dict(counts=values, nominal_task_gates=nominal, all_nominal_tasks_completed=all(nominal.values()),
                robustness_nondegradation=nondegradation, feedback_advantage=advantage)


def validate_record(record, training):
    protocol = record['protocol']
    cases = evaluation_cases(protocol)
    raw = (json.dumps(training, allow_nan=False, separators=(',', ':'))+'\n').encode('utf-8')
    hashes = {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest()}
    checked = summarize(training)
    manifest, seeds = training['manifest'], protocol['seeds']
    selected = [{key: row[key] for key in ('seed', 'selected_step', 'checkpoint', 'checkpoint_sha256')}
                for row in training['models']]
    if (checked != training['summary'] or protocol['training_sha256'] not in hashes
            or protocol['source_sha256'] != manifest['source_sha256']
            or not runtime_compatible(protocol['source_sha256'], protocol['evaluation_source_sha256'])
            or protocol['selected_checkpoints'] != selected
            or protocol['validation_robustness_passed'] != checked['all_candidates_passed']
            or protocol['task'] != manifest['task'] or protocol['requested_throws'] != manifest['requested_throws']
            or protocol.get('controller', 'reference') != manifest.get('controller', 'reference')
            or not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 390000
            or set(seeds) & set(manifest['training_seeds']+manifest['validation_seeds'])):
        raise ValueError('Different selected candidates, task or independent seeds')
    timed = next(row for row in training['models'] if row['seed'] == checked['best_timed_seed'])
    candidates = [dict(id='zero', kind='zero'), dict(id='timed', kind='timed',
        actions=timed['timed_fit']['actions'], horizon_s=timed['timed_fit']['horizon_s'])]
    candidates += [dict(id=f'{kind}_seed{seed}', kind=kind, seed=seed)
                   for seed in (0, 1, 2) for kind in ('live', 'frozen')]
    if protocol['candidates'] != candidates or len(record['records']) != len(candidates):
        raise ValueError('Missing or different policy and control records')
    values = {}
    for candidate, row in zip(candidates, record['records']):
        if any(row[key] != value for key, value in candidate.items()):
            raise ValueError('Independent controller differs from the frozen protocol')
        validate_trials(row['trials'], manifest, seeds, cases=cases)
        values[row['id']] = counts(row['trials'])
    if record['summary'] != assess(values, len(seeds)):
        raise ValueError('Independent assessment differs from physical trials')


def report(record):
    protocol, summary = record['protocol'], record['summary']
    n = len(protocol['seeds'])
    cases = evaluation_cases(protocol)
    perception = protocol.get('suite') == 'perception'
    other = 'Other perception disturbances' if perception else 'Physical seat jitter & combined'
    lines = [f"# {protocol['requested_throws']}-Throw Complete-Task Independent Evaluation", '',
        'Complete-task capability, disturbance non-degradation, and strict feedback gain are verified separately. Every successful episode must pass release, catch, continuous support, collision, and posture checks.', '',
        'At zero error, feedback gains preserve the nominal trajectory; clean counts verify compatibility, while disturbed counts and baselines test learning gain.', '',
        f'| Controller | Clean | 3 mm noise only | {other} | Safety failures |',
        '| --- | --- | --- | --- | --- |']
    if 'controller' in protocol:
        lines[2:2] = [f"Controller: `{protocol['controller']}`; suite: `{protocol.get('suite', 'reference')}`.", '']
    for name, value in summary['counts'].items():
        lines.append(f"| {name} | {value['clean']}/{n} | {value['sensor_only']}/{n} | {value['perturbed']}/{(len(cases)-2)*n} | {value['unsafe']}/{len(cases)*n} |")
    lines += ['', '| PPO training seed | Clean complete task | Disturbance non-degradation | Strict feedback gain |', '| --- | --- | --- | --- |']
    for seed in ('0', '1', '2'):
        labels = ['PASS' if summary[field][seed] else 'FAIL' for field in
                  ('nominal_task_gates', 'robustness_nondegradation', 'feedback_advantage')]
        lines.append(f"| {seed} | {' | '.join(labels)} |")
    if perception:
        lines += ['', 'This suite varies only position measurement noise, transmission delay, and frame dropout, with no physical initial-state randomization.', '',
                  '| Condition | Zero request | Timed gain | PPO 0 | PPO 1 | PPO 2 |', '| --- | --- | --- | --- | --- | --- |']
        indexed = {row['id']: row['trials'] for row in record['records']}
        for case in cases:
            values = [sum(row['is_success'] for row in indexed[name] if row['case'] == case)
                      for name in ('zero', 'timed', 'live_seed0', 'live_seed1', 'live_seed2')]
            lines.append(f"| {case} | "+' | '.join(f'{value}/{n}' for value in values)+' |')
    lines += ['', 'Clean runs repeat the same physical initial state to verify complete-task and reset consistency; repeated seeds do not represent independent physical initial states.', '',
              ('Disturbance non-degradation requires non-decreasing completions on perception disturbances with non-increasing safety failures. Strict feedback gain additionally requires exceeding zero-request, timed, and own-frozen baselines on other perception disturbances.' if perception else
               'Disturbance non-degradation requires non-decreasing completions on perception noise and physical seat jitter with non-increasing safety failures. Strict feedback gain additionally requires exceeding zero-residual, timed, and own-frozen baselines on physical seat jitter.'), '',
              'Candidate checkpoints and timed baselines were fixed on the validation set; independent evaluation does not re-select policies. Configuration, model checksums, source hashes, and per-episode physical evidence are stored in the protocol and raw JSON.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--training', type=Path, required=True)
    run.add_argument('--input-dir', type=Path, required=True)
    run.add_argument('--seed-start', type=int, default=390000)
    run.add_argument('--episodes', type=int, default=5)
    run.add_argument('--jobs', type=int, default=3)
    run.add_argument('--suite', choices=('reference', 'perception'), default='reference')
    run.add_argument('--protocol-output', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--report', type=Path)
    export = commands.add_parser('report')
    export.add_argument('--input', type=Path, required=True)
    export.add_argument('--training', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'report':
        if args.output.exists():
            parser.error('Use a new report path')
        record, training = read_json(args.input), read_json(args.training)
        validate_record(record, training)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report(record), encoding='utf-8', newline='\n')
        return
    paths = [args.output, args.protocol_output]+([] if args.report is None else [args.report])
    if (any(path.exists() for path in paths) or len({path.resolve() for path in paths}) != len(paths)
            or args.seed_start < 390000 or args.episodes < 1 or args.jobs < 1):
        parser.error('Use distinct new outputs, positive counts and independent seeds')
    training = read_json(args.training)
    summary = summarize(training)
    fingerprints = sources(training['manifest'].get('controller', 'reference'))
    if summary != training['summary'] or not runtime_compatible(training['manifest']['source_sha256'], fingerprints):
        raise ValueError('Require the unchanged training archive and runtime')
    manifest = training['manifest']
    cases = PERCEPTION_CASES if args.suite == 'perception' else CASES
    seeds = list(range(args.seed_start, args.seed_start+args.episodes))
    if set(seeds) & set(manifest['training_seeds']+manifest['validation_seeds']):
        raise ValueError('Independent seeds overlap training or validation')
    reference = Path(manifest['reference_dir'])
    if (digest(reference/'reference.npz') != manifest['reference_sha256']
            or digest(reference/'manifest.json') != manifest['reference_manifest_sha256']):
        raise ValueError('Nominal reference changed after training')
    for model in training['models']:
        path = args.input_dir/model['checkpoint']
        if not path.resolve().is_relative_to(args.input_dir.resolve()) or digest(path) != model['checkpoint_sha256']:
            raise ValueError('Selected model differs from the validation archive')
    timed = next(row for row in training['models'] if row['seed'] == summary['best_timed_seed'])
    candidates = [dict(id='zero', kind='zero'), dict(id='timed', kind='timed',
        actions=timed['timed_fit']['actions'], horizon_s=timed['timed_fit']['horizon_s'])]
    candidates += [dict(id=f'{kind}_seed{seed}', kind=kind, seed=seed)
                   for seed in (0, 1, 2) for kind in ('live', 'frozen')]
    evaluator = digest(Path(__file__))
    protocol = dict(training_sha256=digest(args.training), source_sha256=manifest['source_sha256'],
        evaluation_source_sha256=fingerprints,
        evaluator_sha256=evaluator, task=manifest['task'], requested_throws=manifest['requested_throws'],
        controller=manifest.get('controller', 'reference'),
        suite=args.suite, evaluation_cases=cases,
        seeds=seeds, selected_checkpoints=[{key: row[key] for key in
            ('seed', 'selected_step', 'checkpoint', 'checkpoint_sha256')} for row in training['models']],
        validation_robustness_passed=summary['all_candidates_passed'], candidates=candidates,
        scope='nominal complete task; robustness and feedback benefit reported independently')
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.protocol_output.write_text(json.dumps(protocol, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    with ProcessPoolExecutor(args.jobs) as pool:
        records = list(pool.map(test_one, [(row, training, args.input_dir, seeds, cases) for row in candidates]))
    if (sources(manifest.get('controller', 'reference')) != fingerprints or digest(Path(__file__)) != evaluator
            or digest(args.training) != protocol['training_sha256']):
        raise ValueError('Evaluation sources or selected training archive changed')
    values = {}
    for row in records:
        validate_trials(row['trials'], manifest, seeds, cases=cases)
        values[row['id']] = counts(row['trials'])
    result = dict(protocol=protocol, records=records, summary=assess(values, args.episodes))
    validate_record(result, training)
    args.output.write_text(json.dumps(result, allow_nan=False, separators=(',', ':'))+'\n', encoding='utf-8', newline='\n')
    if args.report is not None:
        args.report.write_text(report(result), encoding='utf-8', newline='\n')
    print(json.dumps(result['summary'], ensure_ascii=False), flush=True)
    if not result['summary']['all_nominal_tasks_completed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

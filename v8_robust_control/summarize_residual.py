"""Validate and summarize reception training, nominal checks and independent tests."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

from .train_residual import EVALUATION_CASES, validation_score
from .evaluate_residual import compatible_training_sources
from .residual_env import UNSAFE_REASONS


def recorded_validation_score(manifest, trials, nominal):
    cases = manifest.get('evaluation_cases', EVALUATION_CASES)
    if manifest['environment_version'] >= 4:
        return validation_score(trials, nominal, cases)
    counts = {c: sum(t['is_success'] for t in trials if t['case'] == c) for c in cases}
    unsafe = sum(t['reason'] in ('FORBIDDEN_CONTACT', 'POSE_LIMIT', 'EXTERNAL_BALL_FORCE') for t in trials)
    return (int(counts['baseline'] >= nominal['baseline']), min(counts.values()), sum(counts.values()),
            -unsafe, -float(np.mean([t['residual_rms'] for t in trials])))


def archived_trials(archive):
    rows = archive['records']
    raw = ''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows).encode()
    # Text-mode JSONL uses platform newlines; preserve the original raw digest.
    for candidate in (raw, raw.replace(b'\n', b'\r\n')):
        if hashlib.sha256(candidate).hexdigest() == archive['trials_sha256']:
            return rows, candidate
    raise ValueError('Archived trial checksum mismatch')


def validate_trial(config, trial, environment_version=None):
    environment_version = trial.get('environment_version') if environment_version is None else environment_version
    if environment_version not in (2, 3, 4, 5, 6):
        raise ValueError('Unknown reception environment version')
    if trial.get('environment_version') != environment_version:
        raise ValueError('Trial environment version mismatch')
    if trial.get('nominal_feedback') != (trial['policy'] != 'static'):
        raise ValueError('Nominal feedback mode mismatch')
    if trial['is_success'] != (trial['reason'] == 'SUCCESS'):
        raise ValueError('Success/reason mismatch')
    expected_event = config['success_reward'] if trial['is_success'] else config['failure_reward']
    if environment_version >= 4 and trial['reason'] in UNSAFE_REASONS:
        expected_event = config['unsafe_failure_reward']
    if trial['reward_sums']['event'] != expected_event:
        raise ValueError('Terminal reward does not match the task outcome')
    if trial['is_success'] and trial['stable_contact_s'] < config['stable_duration_s']-1e-9:
        raise ValueError('Success without continuous stable support')
    if trial['is_success']:
        evidence = trial['final_evidence']
        if (not evidence['contact'] or evidence['relative_speed_m_s'] >= .15
                or evidence['tangent_m'] >= .04 or not .01 < evidence['normal_m'] < .06
                or evidence['forbidden_contact'] or evidence['pose_violation']
                or evidence['external_ball_force'] or not .4 < evidence['ball_height_m'] < 1.3):
            raise ValueError('Successful trial has invalid final physical evidence')
    if trial['max_residual_offset_m'] > config['max_offset_m']+1e-9:
        raise ValueError('Action bound exceeded')
    if trial['is_success'] and trial['max_ball_height_m'] >= 1.3:
        raise ValueError('Success despite a ball-height violation')
    if abs(sum(trial['reward_sums'].values())-trial['episode_return']) > 1e-8:
        raise ValueError('Reward accounting mismatch')


def validate(manifest, rows):
    if manifest.get('environment_version') not in (2, 3, 4, 5, 6):
        raise ValueError('Unknown reception environment version')
    if (set(manifest['training_seeds']) & set(manifest['evaluation_seeds'])
            or len(set(manifest['training_seeds'])) != len(manifest['training_seeds'])
            or len(set(manifest['evaluation_seeds'])) != len(manifest['evaluation_seeds'])):
        raise ValueError('Training and evaluation seeds must be distinct and disjoint')
    expected = {(a, s, lr, w) for a in manifest['algorithms']
                for s in manifest['training_seeds'] for lr in manifest['learning_rates']
                for w in manifest['shaping_weights']}
    trained = [r for r in rows if r['kind'] == 'trained']
    actual = [(r['algorithm'], r['seed'], r['learning_rate'], r['shaping_weight']) for r in trained]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Missing, extra or duplicate training runs')
    controls = [r for r in rows if r['kind'] == 'controls']
    if len(controls) != 1:
        raise ValueError('Expected one paired control evaluation')
    cases = manifest.get('evaluation_cases', EVALUATION_CASES)
    for row in rows:
        policies = manifest.get('control_policies', ['zero', 'static', 'random']) if row['kind'] == 'controls' else ('learned',)
        expected_eval = {(p, c, s) for p in policies for c in cases
                         for s in manifest['evaluation_seeds']}
        keys = [(t['policy'], t['case'], t['seed']) for t in row['trials']]
        if set(keys) != expected_eval or len(keys) != len(expected_eval):
            raise ValueError('Evaluation seed/case coverage mismatch')
        if row['kind'] == 'trained' and (row['training_steps'] != manifest['steps']
                                        or not row['save_load_verified']):
            raise ValueError('Training budget or serialization mismatch')
        if row['kind'] == 'trained' and 'checkpoints' in manifest:
            points = row['validation_curve']
            if [p['training_steps'] for p in points] != manifest['checkpoints']:
                raise ValueError('Learning curve budget coverage mismatch')
            selected = next((p for p in points if p['training_steps'] == row['selected_training_steps']), None)
            if (selected is None or selected['checkpoint_sha256'] != row['checkpoint_sha256']
                    or selected['trials'] != row['trials']):
                raise ValueError('Selected checkpoint does not match validation evidence')
            nominal = {case: sum(t['is_success'] for t in controls[0]['trials']
                               if t['policy'] == 'zero' and t['case'] == case) for case in cases}
            nominal['unsafe'] = sum(t['reason'] in UNSAFE_REASONS for t in controls[0]['trials'] if t['policy'] == 'zero')
            for point in points:
                keys = [(t['case'], t['seed']) for t in point['trials']]
                expected_keys = {(c, s) for c in cases for s in manifest['evaluation_seeds']}
                if (set(keys) != expected_keys or len(keys) != len(expected_keys)
                        or any(t['policy'] != 'learned' for t in point['trials'])):
                    raise ValueError('Learning curve evaluation coverage mismatch')
                if tuple(point['validation_score']) != recorded_validation_score(manifest, point['trials'], nominal):
                    raise ValueError('Learning curve selection score mismatch')
                for trial in point['trials']:
                    validate_trial(manifest['config'], trial, manifest['environment_version'])
            best = max(points, key=lambda p: (tuple(p['validation_score']), -p['training_steps']))
            if best['training_steps'] != row['selected_training_steps']:
                raise ValueError('Checkpoint was not selected by the declared validation rule')
        for trial in row['trials']:
            validate_trial(manifest['config'], trial, manifest['environment_version'])
    return trained, controls[0]


def summarize(manifest, rows):
    trained, controls = validate(manifest, rows)
    cases = manifest.get('evaluation_cases', EVALUATION_CASES)
    records = []
    for policy in manifest.get('control_policies', ['zero', 'static', 'random']):
        records.append((policy, None, None, None,
                        [t for t in controls['trials'] if t['policy'] == policy], []))
    for algorithm in manifest['algorithms']:
        for lr in manifest['learning_rates']:
            for weight in manifest['shaping_weights']:
                runs = [r for r in trained if r['algorithm'] == algorithm
                        and r['learning_rate'] == lr and r['shaping_weight'] == weight]
                records.append((algorithm, lr, weight, len(runs),
                                [t for r in runs for t in r['trials']], runs))
    aggregates = []
    for policy, lr, weight, n, trials, runs in records:
        aggregate = dict(policy=policy, learning_rate=lr, shaping_weight=weight, training_seeds=n,
                         training_wall_time_s=sum(r['training_wall_time_s'] for r in runs), cases={})
        if runs:
            aggregate['median_steps_per_second'] = statistics.median(r['steps_per_second'] for r in runs)
        for case in cases:
            values = [t for t in trials if t['case'] == case]
            aggregate['cases'][case] = dict(successes=sum(t['is_success'] for t in values),
                episodes=len(values), failures=dict(Counter(t['reason'] for t in values if not t['is_success'])),
                mean_residual_rms=statistics.mean(t['residual_rms'] for t in values),
                mean_residual_change_rms=statistics.mean(t['residual_change_rms'] for t in values),
                max_ball_height_m=max(t['max_ball_height_m'] for t in values))
            if runs:
                aggregate['cases'][case]['successes_per_training_seed'] = {
                    str(r['seed']): sum(t['is_success'] for t in r['trials'] if t['case'] == case)
                    for r in runs}
        aggregates.append(aggregate)
    lines = ['# Single-Ball Reception Residual Control Validation', '',
             'The task is catching a pre-launched descending ball, excluding active throws or the complete three-ball task.', '',
             'zero is the classical feedback controller; static disables its landing prediction while keeping identical finger control and contact cushioning.', '',
             'These results are on the validation set; final conclusions require independent tests and feedback ablations and cannot alone establish closed-loop noise robustness.', '',
             f"Training budget: {manifest['steps']} control steps per seed; training seeds: {manifest['training_seeds']}.",
             f"Each perception condition uses {len(manifest['evaluation_seeds'])} independent evaluation seeds with exploration disabled.", '',
             *(['Trained policy rows come from each seed\'s selected checkpoint; all training seeds ran the full budget.', '']
               if 'checkpoints' in manifest else []),
             '| Policy | Learning rate | Shaping weight | '+' | '.join(cases)+' | Training time (s) |',
             '| --- | --- | --- | '+' | '.join('---' for _ in cases)+' | --- |']
    for r in aggregates:
        counts = [f"{r['cases'][c]['successes']}/{r['cases'][c]['episodes']}" for c in cases]
        lines.append(f"| {r['policy']} | {r['learning_rate'] if r['learning_rate'] is not None else '—'} | "
                     f"{r['shaping_weight'] if r['shaping_weight'] is not None else '—'} | "
                     + ' | '.join(counts) + f" | {r['training_wall_time_s']:.2f} |")
    lines += ['', 'Each control policy is evaluated once; trained-policy denominators pool all training seeds.',
              'The shaping-weight column reflects training setup; evaluation returns use the unified reward config in the manifest, and policies are judged by independent verification metrics.',
              'Different training seeds share the same evaluation scenes; repeated scenes are not independent samples.',
              'This budget verifies the training pipeline and screens configurations; it does not establish convergence or general superiority of SAC or PPO.',
              'The validation set is used for this screening; confirmation experiments must use fresh seeds. Wall-clock time reflects local concurrency load.',
              'Nominal reception predicts landing from position samples and yields on contact; this subtask is not equivalent to the complete three-ball task.', '']
    if 'checkpoints' in manifest:
        lines += ['## Learning Curve', '',
                  'Budgets advance continuously along the same training run; checkpoints are selected on the validation set only, excluding independent test scenes.', '',
                  'Training time sums actual learn calls excluding checkpoint evaluation; summing across seeds does not equal concurrent wall-clock duration.', '',
                  '| Algorithm/Seed | Control steps | '+' | '.join(cases)+' | Selected |',
                  '| --- | --- | '+' | '.join('---' for _ in cases)+' | --- |']
        for run in trained:
            for point in run['validation_curve']:
                counts = [f"{sum(t['is_success'] for t in point['trials'] if t['case'] == c)}/{len(manifest['evaluation_seeds'])}"
                          for c in cases]
                lines.append(f"| {run['algorithm']}/{run['seed']} | {point['training_steps']} | "
                             + ' | '.join(counts) + (' | Yes |' if point['training_steps'] == run['selected_training_steps'] else ' | — |'))
        lines += ['', 'Learning curves use the same validation scenes; shaded bands show range across training seeds, not confidence intervals on success rate.',
                  'Larger budgets do not guarantee improvement, and checkpoint selection does not imply convergence.', '']
    return aggregates, '\n'.join(lines)


def plot_learning_curve(manifest, rows, output):
    """Export validation curves from archived evidence, without loading policies."""
    trained, controls = validate(manifest, rows)
    if 'checkpoints' not in manifest:
        raise ValueError('Archive contains no learning curve')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    cases = manifest.get('evaluation_cases', EVALUATION_CASES)
    nrows = (len(cases)+1)//2
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 3.5*nrows), sharex=True, sharey=True, squeeze=False)
    denominator = len(manifest['evaluation_seeds'])
    for axis, case in zip(axes.flat, cases):
        baseline = sum(t['is_success'] for t in controls['trials'] if t['policy'] == 'zero' and t['case'] == case)/denominator
        axis.axhline(baseline, color='black', linestyle='--', label='Nominal feedback')
        groups = dict.fromkeys((r['algorithm'], r['learning_rate'], r['shaping_weight']) for r in trained)
        for algorithm, lr, weight in groups:
            runs = [r for r in trained if (r['algorithm'], r['learning_rate'], r['shaping_weight']) == (algorithm, lr, weight)]
            rates = np.array([[sum(t['is_success'] for t in p['trials'] if t['case'] == case)/denominator
                               for p in r['validation_curve']] for r in runs])
            steps = manifest['checkpoints']
            label = algorithm.upper() if len(groups) == 1 else f'{algorithm.upper()} lr={lr:g}, shape={weight:g}'
            axis.plot(steps, rates.mean(axis=0), marker='o', label=label)
            axis.fill_between(steps, rates.min(axis=0), rates.max(axis=0), alpha=.2)
        axis.set(title=case, ylim=(-.03, 1.03), xlabel='Training control steps', ylabel='Validation success fraction')
        axis.grid(alpha=.25)
    for axis in list(axes.flat)[len(cases):]:
        axis.set_visible(False)
    axes.flat[0].legend()
    fig.suptitle('Single-ball reception: mean and range across training seeds')
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def confirmation_report(receipt):
    if receipt.get('environment_version') not in (2, 3, 4, 5, 6):
        raise ValueError('Unknown reception environment version')
    rows = receipt['records']
    cases = list(receipt['evaluation_cases'])
    seeds = receipt['evaluation_seeds']
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate evaluation seeds')
    training = receipt.get('training_manifest')
    trained = [r for r in rows if r['kind'] == 'trained']
    if training:
        if set(seeds) & (set(training['training_seeds']) | set(training['evaluation_seeds'])):
            raise ValueError('Confirmation seeds overlap training or validation')
        expected = {(a, s, lr, w) for a in receipt['algorithms'] for s in training['training_seeds']
                    for lr in training['learning_rates'] for w in training['shaping_weights']}
        actual = [(r['algorithm'], r['seed'], r['learning_rate'], r['shaping_weight']) for r in trained]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError('Missing, extra or duplicate confirmation policies')
    elif trained:
        raise ValueError('Policy evaluation without a training manifest')
    controls = [r for r in rows if r['kind'] == 'controls']
    if len(controls) != 1:
        raise ValueError('Expected paired nominal and static controls')
    for row in rows:
        policies = ('zero', 'static') if row['kind'] == 'controls' else ('learned',)
        expected = {(p, c, s) for p in policies for c in cases for s in seeds}
        actual = [(t['policy'], t['case'], t['seed']) for t in row['trials']]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError('Confirmation coverage mismatch')
        for trial in row['trials']:
            validate_trial(receipt['config'], trial, receipt['environment_version'])
    title = 'Single-Ball Reception Independent Test' if training else 'Single-Ball Nominal Controller Verification'
    lines = ['# '+title, '', 'The task is single-ball reception only, excluding active throws or the complete three-ball cycle.', '',
             f'Each perception condition uses {len(seeds)} scene seeds: {seeds}.', '',
             '| Policy | '+' | '.join(cases)+' |',
             '| --- | '+' | '.join('---' for _ in cases)+' |']
    groups = [(p, [t for t in controls[0]['trials'] if t['policy'] == p]) for p in ('zero', 'static')]
    if training:
        for algorithm in receipt['algorithms']:
            groups.append((algorithm, [t for r in trained if r['algorithm'] == algorithm for t in r['trials']]))
    for label, trials in groups:
        counts = []
        for case in cases:
            values = [t for t in trials if t['case'] == case]
            counts.append(f"{sum(t['is_success'] for t in values)}/{len(values)}")
        lines.append('| '+label+' | '+' | '.join(counts)+' |')
    lines += ['', 'zero is the nominal reception feedback controller; static disables landing prediction while keeping identical grasping and cushioning.',
              'Learned policies are pooled across all training seeds; training seeds share the same test scenes and are not independent scene samples.',
              'Policy evaluation disables exploration; failure reasons, catch evidence, and action metrics are stored in the raw JSON.', '']
    return '\n'.join(lines)


def screening_decisions(receipt):
    """A predeclared screening gate, not automatic policy adoption."""
    confirmation_report(receipt)
    rows = receipt['records']
    controls = next(r['trials'] for r in rows if r['kind'] == 'controls')
    nominal_baseline = sum(t['is_success'] for t in controls if t['policy'] == 'zero' and t['case'] == 'baseline')
    nominal_perturbed = sum(t['is_success'] for t in controls if t['policy'] == 'zero' and t['case'] != 'baseline')
    decisions = {}
    for algorithm in receipt['algorithms']:
        runs = [r for r in rows if r['kind'] == 'trained' and r['algorithm'] == algorithm]
        baseline_counts = [sum(t['is_success'] for t in r['trials'] if t['case'] == 'baseline') for r in runs]
        perturbed = sum(t['is_success'] for r in runs for t in r['trials'] if t['case'] != 'baseline')
        gain = perturbed-len(runs)*nominal_perturbed
        preserves = all(count >= nominal_baseline for count in baseline_counts)
        unsafe_reasons = {'FORBIDDEN_CONTACT', 'POSE_LIMIT', 'EXTERNAL_BALL_FORCE', 'BALL_TOO_HIGH', 'NONFINITE_STATE'}
        nominal_unsafe = sum(t['reason'] in unsafe_reasons for t in controls if t['policy'] == 'zero')
        unsafe_counts = [sum(t['reason'] in unsafe_reasons for t in r['trials']) for r in runs]
        safety_preserved = all(count <= nominal_unsafe for count in unsafe_counts)
        decisions[algorithm] = dict(baseline_per_seed=baseline_counts, nominal_baseline=nominal_baseline,
                                     paired_perturbed_gain=gain, baseline_preserved=preserves,
                                     candidate_gate=preserves and gain > 0)
        if receipt['environment_version'] >= 3:
            decisions[algorithm].update(nominal_unsafe=nominal_unsafe, unsafe_per_seed=unsafe_counts,
                                       safety_preserved=safety_preserved,
                                       candidate_gate=preserves and gain > 0 and safety_preserved)
        if receipt['environment_version'] >= 4:
            decisions[algorithm].update(nominal_perturbed=nominal_perturbed,
                perturbed_episodes_per_seed=sum(t['policy'] == 'zero' and t['case'] != 'baseline' for t in controls),
                perturbed_per_seed=[sum(t['is_success'] for t in r['trials'] if t['case'] != 'baseline') for r in runs])
    return decisions


def delivery_report(test, training=None, nominal=None, feedback=None):
    sections = [confirmation_report(test)]
    if training is not None:
        rows, _ = archived_trials(training)
        _, text = summarize(training['manifest'], rows)
        if not compatible_training_sources(training['manifest']['source_sha256'], test['source_sha256']):
            raise ValueError('Training and confirmation revisions differ')
        if training['manifest']['config'] != test['config']:
            raise ValueError('Training and confirmation task configurations differ')
        sections.append(text.replace('# Single-Ball Reception Residual Control Validation', '## Training and Validation', 1))
    if nominal is not None:
        if not compatible_training_sources(nominal['source_sha256'], test['source_sha256']):
            raise ValueError('Nominal verification revision differs')
        if nominal['config'] != test['config']:
            raise ValueError('Nominal verification task configuration differs')
        sections.append(confirmation_report(nominal).replace('# Single-Ball Nominal Controller Verification', '## Nominal Controller Verification', 1))
    if feedback is not None:
        if (feedback['environment_version'] != test['environment_version']
                or not compatible_training_sources(feedback['training_manifest']['source_sha256'], test['source_sha256'])
                or feedback['config'] != test['config']):
            raise ValueError('Feedback diagnostic revision differs')
        diagnostic_lines = ['## Feedback Diagnostics', '', '| Policy/Mode | Case | Success |', '| --- | --- | --- |']
        for row in feedback['records']:
            trials = row['trials']
            policies = list(dict.fromkeys(t['policy'] for t in trials))
            for policy in policies:
                for case in dict.fromkeys(t['case'] for t in trials):
                    selected = [t for t in trials if t['policy'] == policy and t['case'] == case]
                    expected_seeds = set(feedback['evaluation_seeds'])
                    if {t['seed'] for t in selected} != expected_seeds or len(selected) != len(expected_seeds):
                        raise ValueError('Feedback diagnostic coverage mismatch')
                    for trial in selected:
                        validate_trial(feedback['config'], trial, feedback['environment_version'])
                    label = row.get('checkpoint', policy).split('_lr')[0]+' / '+row['mode']
                    diagnostic_lines.append(f"| {label} | {case} | {sum(t['is_success'] for t in selected)}/{len(selected)} |")
        sections.append('\n'.join(diagnostic_lines))
        sections.append('\nMasking policy ball state affects direct input fields only (nominal reference still carries ball info); frozen policy output still retains the classical controller.\n')
        if test['environment_version'] >= 3:
            checkpoints = dict.fromkeys(r['checkpoint'] for r in feedback['records'] if 'checkpoint' in r)
            for checkpoint in checkpoints:
                modes = {r['mode']: r['trials'] for r in feedback['records'] if r.get('checkpoint') == checkpoint}
                normal, frozen = modes.get('unmodified', []), modes.get('frozen_initial', [])
                if normal and len(normal) == len(frozen):
                    count, fixed = sum(t['is_success'] for t in normal), sum(t['is_success'] for t in frozen)
                    sections.append(f"{checkpoint.split('_lr')[0]}: normal policy {count}/{len(normal)}, "
                                    f"frozen initial output {fixed}/{len(frozen)}; both retain identical nominal feedback. "
                                    + ('This small sample shows no additional gain from online policy feedback.' if fixed >= count else
                                       'Differences in this small sample require more independent scenes to confirm.'))
    if test.get('training_manifest'):
        sections.append('## Screening Gate\n\nEvery training seed must preserve zero-disturbance catches and paired disturbed totals must exceed the classical controller. This gate does not trigger automatic replacement.\n')
        for algorithm, decision in screening_decisions(test).items():
            verdict = 'passed candidate screening' if decision['candidate_gate'] else 'failed candidate screening'
            sections.append(f"- {algorithm}: {verdict}; zero-disturbance per seed {decision['baseline_per_seed']}, "
                            f"classical controller {decision['nominal_baseline']}; paired disturbed delta {decision['paired_perturbed_gain']:+d}.")
            if test['environment_version'] >= 3:
                sections.append(f"  Safety failures: classical controller {decision['nominal_unsafe']}; "
                                f"per training seed {decision['unsafe_per_seed']}. Any seed increasing safety failures fails.")
            if test['environment_version'] >= 4:
                episodes = decision['perturbed_episodes_per_seed']
                repeats = len(decision['perturbed_per_seed'])
                sections.append(f"  Disturbed baseline completions {decision['nominal_perturbed']}/{episodes}; "
                                f"per-seed policy completions {decision['perturbed_per_seed']} (out of {episodes} each). "
                                f"Paired pooling repeats the same baseline {repeats} times and does not add independent scenes.")
        sections.append('\nThis budget and test set cannot establish general algorithm superiority; per-condition degradation and collisions must be checked jointly. The three-ball task is not yet wired to learned policies.\n')
    if test['environment_version'] >= 3:
        lines = ['## Post-Contact Estimation', '',
                 'Errors are computed over valid estimation steps only; ground truth is diagnostic and never fed to policies. Episodes without valid estimates are not counted as zero error.', '',
                 '| Policy | Case | Valid estimate steps | Position RMSE (mm) | Velocity RMSE (m/s) |',
                 '| --- | --- | --- | --- | --- |']
        groups = [('zero', [t for r in test['records'] if r['kind'] == 'controls' for t in r['trials'] if t['policy'] == 'zero'])]
        for algorithm in test.get('algorithms', []):
            groups.append((algorithm, [t for r in test['records'] if r['kind'] == 'trained' and r['algorithm'] == algorithm for t in r['trials']]))
        safety = ['## Safety and Action Metrics', '',
                  '| Policy | Case | Episodes | Forbidden contact | Pose limit | Ball height | Other safety | Task deadline | Mean residual RMS | Max axis offset (mm) |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for policy, trials in groups:
            for case in test['evaluation_cases']:
                case_trials = [t for t in trials if t['case'] == case]
                reasons = Counter(t['reason'] for t in case_trials)
                counts = [reasons[key] for key in ('FORBIDDEN_CONTACT', 'POSE_LIMIT', 'BALL_TOO_HIGH')]
                counts += [reasons['EXTERNAL_BALL_FORCE']+reasons['NONFINITE_STATE'], reasons['TASK_DEADLINE']]
                safety.append(f'| {policy} | {case} | {len(case_trials)} | '+ ' | '.join(map(str, counts))
                              + f" | {statistics.mean(t['residual_rms'] for t in case_trials):.4f}"
                              + f" | {1000*max(t['max_residual_offset_m'] for t in case_trials):.3f} |")
                values = [t for t in trials if t['case'] == case and t.get('post_contact_estimates', 0) > 0]
                count = sum(t['post_contact_estimates'] for t in values)
                errors = []
                for field, scale in (('post_contact_position_rmse_m', 1000), ('post_contact_velocity_rmse_m_s', 1)):
                    errors.append(f"{scale*(sum(t['post_contact_estimates']*t[field]**2 for t in values)/count)**.5:.3f}" if count else '—')
                lines.append(f'| {policy} | {case} | {count} | '+ ' | '.join(errors)+' |')
        sections.append('\n'.join(safety)+'\n')
        sections.append('\n'.join(lines)+'\n')
    return '\n\n'.join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-dir', type=Path)
    source.add_argument('--input', type=Path, help='Archived result JSON, without checkpoint files')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--training', type=Path, help='Archived training JSON for a combined report')
    parser.add_argument('--nominal', type=Path, help='Nominal verification JSON for a combined report')
    parser.add_argument('--feedback', type=Path, help='Feedback diagnostic JSON for a combined report')
    parser.add_argument('--curve-output', type=Path, help='Validation learning curve PNG from a training archive')
    args = parser.parse_args()
    if args.curve_output is not None and args.curve_output.exists():
        parser.error('Curve output must be a new file')
    outputs = [p.resolve() for p in (args.output, args.report, args.curve_output) if p is not None]
    if len(set(outputs)) != len(outputs):
        parser.error('All output paths must be distinct')
    if (args.report.exists() or (args.output is not None and
            (args.output.exists() or args.output.resolve() == args.report.resolve()))):
        parser.error('Output and report must be distinct new files')
    if args.input_dir is not None:
        if args.output is None:
            parser.error('--input-dir requires --output for the archived result')
        if not json.loads((args.input_dir/'complete.json').read_text())['complete']:
            parser.error('Pilot is incomplete')
        manifest = json.loads((args.input_dir/'manifest.json').read_text())
        raw = (args.input_dir/'trials.jsonl').read_bytes()
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        for row in rows:
            if row['kind'] == 'trained':
                checkpoint = args.input_dir/row['checkpoint']
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row['checkpoint_sha256']:
                    raise ValueError('Checkpoint checksum mismatch')
                for point in row.get('validation_curve', []):
                    if hashlib.sha256((args.input_dir/point['checkpoint']).read_bytes()).hexdigest() != point['checkpoint_sha256']:
                        raise ValueError('Learning curve checkpoint checksum mismatch')
    else:
        archive = json.loads(args.input.read_text(encoding='utf-8'))
        if 'scope' in archive:
            if args.curve_output is not None:
                parser.error('--curve-output requires training input')
            if args.output is not None:
                parser.error('Independent evaluation is already archived; specify --report only')
            optional = {name: json.loads(getattr(args, name).read_text(encoding='utf-8'))
                        if getattr(args, name) is not None else None
                        for name in ('training', 'nominal', 'feedback')}
            report = delivery_report(archive, **optional)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report, encoding='utf-8', newline='\n')
            print(report)
            return
        manifest = archive['manifest']
        rows, raw = archived_trials(archive)
    aggregates, report = summarize(manifest, rows)
    if args.curve_output is not None:
        plot_learning_curve(manifest, rows, args.curve_output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(manifest=manifest, trials_sha256=hashlib.sha256(raw).hexdigest(),
                                             aggregates=aggregates, records=rows), indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    args.report.write_text(report, encoding='utf-8', newline='\n')
    print(report)


if __name__ == '__main__':
    main()

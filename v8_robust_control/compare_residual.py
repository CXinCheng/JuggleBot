"""Select fixed residuals on validation scenes and compare independent tests."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from .residual_env import CatchConfig, ResidualCatchEnv, UNSAFE_REASONS
from .train_residual import evaluate, source_hashes, validation_score
from .summarize_residual import archived_trials, validate, validate_trial, confirmation_report, screening_decisions


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')


def selection_hash_matches(selection, actual_hash, recorded_hash):
    """Accept Git's LF/CRLF conversion of the original JSON serialization."""
    if actual_hash == recorded_hash:
        return True
    raw = (json.dumps(selection, indent=2, allow_nan=False)+'\n').encode('utf-8')
    variants = {hashlib.sha256(raw).hexdigest(),
                hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest()}
    return actual_hash in variants and recorded_hash in variants


class FixedResidual:
    """A constant normalized motor request; all environment guards still apply."""
    def __init__(self, action):
        self.action = np.asarray(action, dtype=float)
        if self.action.shape != (4,) or not np.all(np.isfinite(self.action)) or np.max(np.abs(self.action)) > 1:
            raise ValueError('Require four finite bounded residuals')

    def predict(self, observation, deterministic=True):
        return self.action.copy(), None


class ActionMean:
    """Fit a constant using actor requests at eligible validation timesteps."""
    def __init__(self, model):
        self.model = model
        self.sums = np.zeros(4)
        self.counts = np.zeros(4, dtype=int)

    def predict(self, observation, deterministic=True):
        action, state = self.model.predict(observation, deterministic=True)
        action = np.clip(action, -1., 1.)
        if observation[57] > .5:  # Current measured hand support.
            eligible = np.r_[np.full(3, observation[-1] > .5),
                             observation[-2] > .5 or action[3] <= 0.]
            self.sums[eligible] += action[eligible]
            self.counts[eligible] += 1
        return action, state

    def action(self):
        return np.divide(self.sums, self.counts, out=np.zeros(4), where=self.counts > 0)


def trial_counts(trials):
    return dict(baseline=sum(t['is_success'] for t in trials if t['case'] == 'baseline'),
                perturbed=sum(t['is_success'] for t in trials if t['case'] != 'baseline'),
                unsafe=sum(t['reason'] in UNSAFE_REASONS for t in trials))


def check_coverage(trials, cases, seeds, policy, config, version):
    keys = [(t['case'], t['seed'], t['policy']) for t in trials]
    expected = {(c, s, policy) for c in cases for s in seeds}
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError('Missing, duplicate or unexpected comparison trials')
    for trial in trials:
        validate_trial(config, trial, version)


def select_candidate(candidates):
    if not candidates or len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('Require distinct fixed candidates')
    return max(candidates, key=lambda candidate: tuple(candidate['validation_score']))


def fixed_candidates(arguments):
    folder, row, config, seeds, cases = arguments
    torch.set_num_threads(1)
    path = folder/row['checkpoint']
    if digest(path) != row['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    model = PPO.load(path, device='cpu')
    if model.num_timesteps != row['selected_training_steps'] or model.observation_space.shape != (ResidualCatchEnv.OBSERVATION_SIZE,):
        raise ValueError('Checkpoint budget or observation version mismatch')
    env = ResidualCatchEnv(config)
    try:
        observation, _ = env.reset(seed=seeds[0])
        initial, _ = model.predict(observation, deterministic=True)
    finally:
        env.close()
    fitted = ActionMean(model)
    fit_trials = evaluate(fitted, config, seeds, policies=('learned',), cases=cases)
    # Fitting must reproduce the previously selected policy's validation outcomes.
    if [(t['case'], t['seed'], t['reason']) for t in fit_trials] != [
            (t['case'], t['seed'], t['reason']) for t in row['trials']]:
        raise ValueError('Policy validation replay changed')
    candidates = []
    for method, action in (('initial', initial), ('contact_mean', fitted.action())):
        trials = evaluate(FixedResidual(action), config, seeds, policies=('fixed',), cases=cases)
        candidate = dict(id=f"ppo_seed{row['seed']}_{method}", action=np.asarray(action).tolist(),
                         method=method, checkpoint=row['checkpoint'], checkpoint_sha256=row['checkpoint_sha256'],
                         trials=trials)
        if method == 'contact_mean':
            candidate.update(action_sums=fitted.sums.tolist(), eligible_counts=fitted.counts.tolist())
        candidates.append(candidate)
        print('[Fixed validation]', candidate['id'], trial_counts(trials), flush=True)
    return candidates


def validate_selection(selection):
    manifest = selection['training_manifest']
    if selection['environment_version'] != manifest['environment_version'] or selection['config'] != manifest['config']:
        raise ValueError('Fixed selection environment mismatch')
    if selection['source_sha256'] != manifest['source_sha256']:
        raise ValueError('Fixed selection source mismatch')
    if (selection['evaluation_seeds'] != manifest['evaluation_seeds']
            or selection['evaluation_cases'] != manifest['evaluation_cases']):
        raise ValueError('Fixed selection must use validation scenes only')
    controls = selection['nominal_trials']
    check_coverage(controls, selection['evaluation_cases'], selection['evaluation_seeds'], 'zero',
                   selection['config'], selection['environment_version'])
    nominal = {c: sum(t['is_success'] for t in controls if t['case'] == c) for c in selection['evaluation_cases']}
    nominal['unsafe'] = trial_counts(controls)['unsafe']
    expected = {'zero'} | {f'ppo_seed{s}_{method}' for s in manifest['training_seeds']
                          for method in ('initial', 'contact_mean')}
    if {c['id'] for c in selection['candidates']} != expected:
        raise ValueError('Fixed candidate coverage mismatch')
    for candidate in selection['candidates']:
        FixedResidual(candidate['action'])
        check_coverage(candidate['trials'], selection['evaluation_cases'], selection['evaluation_seeds'], 'fixed',
                       selection['config'], selection['environment_version'])
        score = validation_score(candidate['trials'], nominal, selection['evaluation_cases'])
        if tuple(candidate['validation_score']) != score:
            raise ValueError('Fixed candidate selection score mismatch')
        if candidate.get('method') == 'contact_mean':
            sums, counts = np.asarray(candidate['action_sums']), np.asarray(candidate['eligible_counts'])
            if sums.shape != (4,) or counts.shape != (4,) or np.any(counts < 0) or np.any(counts != np.floor(counts)):
                raise ValueError('Invalid fixed residual fit statistics')
            fitted = np.divide(sums, counts, out=np.zeros(4), where=counts > 0)
            if not np.allclose(candidate['action'], fitted, rtol=0., atol=1e-12):
                raise ValueError('Fixed action does not match its validation fit')
    if selection['candidates'][0]['id'] != 'zero' or selection['candidates'][0]['action'] != [0., 0., 0., 0.]:
        raise ValueError('Require the zero-residual fallback first')
    selected = select_candidate(selection['candidates'])
    if selected['id'] != selection['selected_candidate_id']:
        raise ValueError('Fixed residual was not selected by validation score')
    return selected


def select_fixed(args):
    fingerprints = source_hashes()
    manifest = read_json(args.input_dir/'manifest.json')
    if (not read_json(args.input_dir/'complete.json')['complete']
            or manifest['source_sha256'] != fingerprints
            or manifest['environment_version'] != ResidualCatchEnv.ENVIRONMENT_VERSION
            or manifest['algorithms'] != ['ppo']):
        raise ValueError('Require a complete current-environment PPO run')
    rows = [json.loads(line) for line in (args.input_dir/'trials.jsonl').read_text(encoding='utf-8').splitlines()]
    trained, control = validate(manifest, rows)
    controls = [t for t in control['trials'] if t['policy'] == 'zero']
    nominal = {c: sum(t['is_success'] for t in controls if t['case'] == c) for c in manifest['evaluation_cases']}
    nominal['unsafe'] = trial_counts(controls)['unsafe']
    candidates = [dict(id='zero', method='zero', action=[0., 0., 0., 0.],
                       trials=[dict(t, policy='fixed') for t in controls])]
    tasks = [(args.input_dir, row, CatchConfig(**manifest['config']), manifest['evaluation_seeds'],
              manifest['evaluation_cases']) for row in trained]
    with ProcessPoolExecutor(args.jobs) as pool:
        for group in pool.map(fixed_candidates, tasks):
            candidates.extend(group)
    for candidate in candidates:
        candidate['validation_score'] = list(validation_score(candidate['trials'], nominal, manifest['evaluation_cases']))
    selection = dict(scope='fixed_residual_selection', environment_version=manifest['environment_version'],
                     config=manifest['config'], source_sha256=fingerprints, training_manifest=manifest,
                     evaluation_cases=manifest['evaluation_cases'], evaluation_seeds=manifest['evaluation_seeds'],
                     nominal_trials=controls, candidates=candidates,
                     selected_candidate_id=select_candidate(candidates)['id'],
                     selection_rule='safety-first validation_score; earlier candidate on ties; zero fallback first',
                     method='zero plus initial and eligible-contact mean requests from each validation-selected PPO',
                     comparison_source_sha256=digest(Path(__file__)))
    validate_selection(selection)
    if source_hashes() != fingerprints:
        raise ValueError('Environment changed during fixed selection')
    write_json(args.output, selection)
    print('[Selected fixed]', selection['selected_candidate_id'], flush=True)


def test_fixed(args):
    selection = read_json(args.selection)
    candidate = validate_selection(selection)
    reference = read_json(args.reference)
    confirmation_report(reference)
    if (reference['config'] != selection['config'] or reference['source_sha256'] != selection['source_sha256']
            or source_hashes() != selection['source_sha256']
            or reference['environment_version'] != selection['environment_version']):
        raise ValueError('Require the same current environment for the independent test')
    training = selection['training_manifest']
    if set(reference['evaluation_seeds']) & (set(training['evaluation_seeds']) | set(training['training_seeds'])):
        raise ValueError('Fixed test overlaps training or validation')
    torch.set_num_threads(1)
    trials = evaluate(FixedResidual(candidate['action']), CatchConfig(**selection['config']),
                      reference['evaluation_seeds'], policies=('fixed',), cases=reference['evaluation_cases'])
    if source_hashes() != selection['source_sha256']:
        raise ValueError('Environment changed during fixed test')
    receipt = dict(scope='fixed_residual_confirmation', environment_version=selection['environment_version'],
                   config=selection['config'], source_sha256=selection['source_sha256'],
                   selection_sha256=digest(args.selection), selected_candidate_id=candidate['id'], action=candidate['action'],
                   evaluation_seeds=reference['evaluation_seeds'], evaluation_cases=reference['evaluation_cases'],
                   comparison_source_sha256=digest(Path(__file__)), trials=trials)
    write_json(args.output, receipt)
    print('[Fixed independent test]', trial_counts(trials), flush=True)


def paired_counts(trials, reference):
    left = {(t['case'], t['seed']): t for t in trials}
    right = {(t['case'], t['seed']): t for t in reference}
    if len(left) != len(trials) or len(right) != len(reference) or left.keys() != right.keys():
        raise ValueError('Paired scenes differ or contain duplicates')
    return dict(wins=sum(left[k]['is_success'] and not right[k]['is_success'] for k in left),
                losses=sum(right[k]['is_success'] and not left[k]['is_success'] for k in left))


def comparison_report(ppo_training, ppo_test, sac_training, sac_test, selection, fixed_test, selection_sha256):
    tests = [('ppo', ppo_training, ppo_test), ('sac', sac_training, sac_test)]
    for algorithm, training, test in tests:
        rows, _ = archived_trials(training)
        validate(training['manifest'], rows)
        confirmation_report(test)
        if test['training_manifest'] != training['manifest'] or test['algorithms'] != [algorithm]:
            raise ValueError('Algorithm test does not match its training archive')
    ppo, sac = ppo_training['manifest'], sac_training['manifest']
    fields = ('environment_version', 'config', 'source_sha256', 'steps', 'checkpoints', 'training_seeds',
              'evaluation_seeds', 'evaluation_cases', 'learning_rates', 'shaping_weights',
              'training_case_probabilities', 'training_perception_cases', 'versions')
    if any(ppo[key] != sac[key] for key in fields):
        raise ValueError('PPO and SAC training configurations are not comparable')
    if ppo_test.get('evaluation_source_sha256') != sac_test.get('evaluation_source_sha256'):
        raise ValueError('PPO and SAC evaluation implementations differ')
    for key in ('evaluation_cases', 'evaluation_seeds', 'config', 'source_sha256', 'environment_version'):
        if ppo_test[key] != sac_test[key] or fixed_test[key] != ppo_test[key]:
            raise ValueError('Independent test scenarios or environments differ')
    selected = validate_selection(selection)
    if (selection['training_manifest'] != ppo
            or not selection_hash_matches(selection, selection_sha256, fixed_test['selection_sha256'])
            or fixed_test['selected_candidate_id'] != selected['id'] or fixed_test['action'] != selected['action']):
        raise ValueError('Fixed test does not match the validation-selected constant')
    if selection.get('comparison_source_sha256') != fixed_test.get('comparison_source_sha256'):
        raise ValueError('Fixed selection and test implementations differ')
    check_coverage(fixed_test['trials'], fixed_test['evaluation_cases'], fixed_test['evaluation_seeds'],
                   'fixed', fixed_test['config'], fixed_test['environment_version'])
    def nominal(test):
        return [t for r in test['records'] if r['kind'] == 'controls' for t in r['trials'] if t['policy'] == 'zero']
    controls = nominal(ppo_test)
    other = nominal(sac_test)
    paired_counts(controls, other)
    if any({k: v for k, v in a.items() if k != 'wall_time_s'} !=
           {k: v for k, v in b.items() if k != 'wall_time_s'} for a, b in zip(controls, other, strict=True)):
        raise ValueError('Repeated nominal physical evidence changed')
    baseline, fixed = trial_counts(controls), trial_counts(fixed_test['trials'])
    episodes = len(ppo_test['evaluation_seeds'])
    perturbations = (len(ppo_test['evaluation_cases'])-1)*episodes
    lines = ['# Single-Ball Reception Policy Comparison', '',
             f"PPO and SAC each use three training seeds at {ppo['steps']:,} control steps per seed; checkpoints are selected on the validation set only.",
             f"Validation seeds: {ppo['evaluation_seeds'][0]}–{ppo['evaluation_seeds'][-1]}; test seeds: "
             f"{ppo_test['evaluation_seeds'][0]}–{ppo_test['evaluation_seeds'][-1]}.", '',
             'The fixed residual requests the same 4-D vector across all episodes; nominal feedback, contact authority, clipping, and rate limits remain active.',
             'Candidates are zero residual plus each of the three PPO models\' initial outputs and mean validation contact-phase requests; test outcomes are excluded from fitting and selection.', '',
             'Constant selection uses the safety-first validation score, breaking ties toward earlier candidates with zero residual listed first. Because the fixed control reuses PPO training assets, parameter-search cost is not compared separately.', '',
             f"Selected constant: `{selected['id']}`; normalized request: `{selected['action']}`.", '',
             '| Policy | Zero-disturbance catches | Disturbed catches | Safety failures | Wins/Losses vs. fixed residual |',
             '| --- | --- | --- | --- | --- |']
    groups = [('Classical controller', controls), ('Fixed residual', fixed_test['trials'])]
    for algorithm, _, test in tests:
        groups.extend((f"{algorithm.upper()} / seed {r['seed']}", r['trials']) for r in test['records'] if r['kind'] == 'trained')
    for label, trials in groups:
        counts = trial_counts(trials)
        paired = paired_counts(trials, fixed_test['trials'])
        lines.append(f"| {label} | {counts['baseline']}/{episodes} | {counts['perturbed']}/{perturbations} | "
                     f"{counts['unsafe']}/{episodes+perturbations} | {paired['wins']}/{paired['losses']} |")
    lines += ['', 'Safety failures include forbidden contacts, posture or ball-height violations, external forces, and non-finite states; they are counted separately from drops and deadline timeouts.',
              'All policies share the same test scenes; pooling three training seeds does not create new independent test scenes.', '',
              '## Catches by Condition', '', '| Policy | '+' | '.join(ppo_test['evaluation_cases'])+' |',
              '| --- | '+' | '.join('---' for _ in ppo_test['evaluation_cases'])+' |']
    pooled = [('Classical controller', controls), ('Fixed residual', fixed_test['trials'])]
    for algorithm, _, test in tests:
        pooled.append((algorithm.upper(), [t for r in test['records'] if r['kind'] == 'trained' for t in r['trials']]))
    for label, trials in pooled:
        counts = []
        for case in ppo_test['evaluation_cases']:
            values = [t for t in trials if t['case'] == case]
            counts.append(f"{sum(t['is_success'] for t in values)}/{len(values)}")
        lines.append('| '+label+' | '+' | '.join(counts)+' |')
    lines += ['', '## Candidate Screening', '']
    for algorithm, _, test in tests:
        decision = screening_decisions(test)[algorithm]
        lines.append(f"- {algorithm.upper()}: {'PASS' if decision['candidate_gate'] else 'FAIL'}; "
                     f"paired disturbed delta {decision['paired_perturbed_gain']:+d}, safety failures per seed {decision['unsafe_per_seed']}.")
    fixed_pass = (fixed['baseline'] >= baseline['baseline'] and fixed['perturbed'] > baseline['perturbed']
                  and fixed['unsafe'] <= baseline['unsafe'])
    lines += [f"- Fixed residual: {'PASS' if fixed_pass else 'FAIL'} under the same success and safety gate.", '',
              '## Training Budget and Wall-Clock Time', '', '| Algorithm/Seed | Training steps | Selected step | Training wall time (s) |',
              '| --- | --- | --- | --- |']
    for algorithm, training, _ in tests:
        for row in training['records']:
            if row['kind'] == 'trained':
                lines.append(f"| {algorithm.upper()}/{row['seed']} | {row['training_steps']} | "
                             f"{row['selected_training_steps']} | {row['training_wall_time_s']:.2f} |")
    lines += ['', 'Wall-clock times come from actual learn calls excluding validation; affected by machine load and concurrency, they are not a strict algorithm speed ranking.',
              'Equal interaction budgets do not imply equal gradient update counts or convergence of either algorithm.', '', '## Conclusion and Scope', '',
              'Results for the classical controller, fixed residual, and both RL algorithms inform whether further training is warranted; candidate screening does not trigger controller replacement.']
    consistent = []
    for algorithm, _, test in tests:
        trained = [trial_counts(r['trials']) for r in test['records'] if r['kind'] == 'trained']
        if (screening_decisions(test)[algorithm]['candidate_gate']
                and all(c['perturbed'] > fixed['perturbed'] and c['baseline'] >= fixed['baseline']
                        and c['unsafe'] <= fixed['unsafe'] for c in trained)):
            consistent.append(algorithm.upper())
    if consistent:
        lines.append('In this test, '+', '.join(consistent)+' exceeded the fixed residual across all training seeds and remain candidates for further training; differences still require more scenes to confirm.')
    else:
        lines.append('This test did not show either algorithm consistently exceeding the fixed residual across all training seeds. Retain the classical controller and fixed residual as baselines and do not scale to three-ball RL on this basis.')
    lines += ['This comparison covers one finite-budget setting and cannot establish general superiority of SAC or PPO.',
              'The fixed residual searches only the finite candidate set above; any additional benefit from online policy feedback must be judged alongside feedback ablations.',
              'The task covers catching a descending single ball only, excluding active throws, two-ball exchange, or the complete three-ball task.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    select = commands.add_parser('select-fixed', help='Fit and select constants on validation scenes only')
    select.add_argument('--input-dir', type=Path, required=True)
    select.add_argument('--jobs', type=int, default=1)
    select.add_argument('--output', type=Path, required=True)
    test = commands.add_parser('test-fixed', help='Test a pinned selection on the reference test scenes')
    test.add_argument('--selection', type=Path, required=True)
    test.add_argument('--reference', type=Path, required=True)
    test.add_argument('--output', type=Path, required=True)
    report = commands.add_parser('report', help='Regenerate the comparison from archived JSON')
    for field in ('ppo-training', 'ppo-test', 'sac-training', 'sac-test', 'fixed-selection', 'fixed-test'):
        report.add_argument('--'+field, type=Path, required=True)
    report.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or getattr(args, 'jobs', 1) < 1:
        parser.error('Use a new output path and positive job count')
    torch.set_num_threads(1)
    if args.command == 'select-fixed':
        select_fixed(args)
    elif args.command == 'test-fixed':
        test_fixed(args)
    else:
        text = comparison_report(*(read_json(getattr(args, name)) for name in
                                   ('ppo_training', 'ppo_test', 'sac_training', 'sac_test', 'fixed_selection', 'fixed_test')),
                                 selection_sha256=digest(args.fixed_selection))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        print(text)


if __name__ == '__main__':
    main()

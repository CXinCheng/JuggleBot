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
    lines = ['# 单球接球策略对照', '',
             f"PPO 与 SAC 各使用三个训练种子、每种子 {ppo['steps']:,} 个控制步；仅按验证集选择检查点。",
             f"验证种子为 {ppo['evaluation_seeds'][0]}–{ppo['evaluation_seeds'][-1]}；测试种子为 "
             f"{ppo_test['evaluation_seeds'][0]}–{ppo_test['evaluation_seeds'][-1]}。", '',
             '固定残差在全部回合请求同一个四维向量；名义反馈、接触权限、限幅和变化率限制仍生效。',
             '候选为零残差，以及三个 PPO 各自的初始输出和验证接触阶段请求均值；参数与选择均不使用测试结果。', '',
             '常数沿用安全优先的验证评分，同分时优先较早候选；零残差列在首位。固定对照复用 PPO 训练资产，不单独比较参数搜索成本。', '',
             f"选定常数：`{selected['id']}`；归一化请求为 `{selected['action']}`。", '',
             '| 策略 | 零扰动接稳 | 扰动接稳 | 安全失败 | 相对固定残差新增成功/丢失成功 |',
             '| --- | --- | --- | --- | --- |']
    groups = [('经典控制器', controls), ('固定残差', fixed_test['trials'])]
    for algorithm, _, test in tests:
        groups.extend((f"{algorithm.upper()} / seed {r['seed']}", r['trials']) for r in test['records'] if r['kind'] == 'trained')
    for label, trials in groups:
        counts = trial_counts(trials)
        paired = paired_counts(trials, fixed_test['trials'])
        lines.append(f"| {label} | {counts['baseline']}/{episodes} | {counts['perturbed']}/{perturbations} | "
                     f"{counts['unsafe']}/{episodes+perturbations} | {paired['wins']}/{paired['losses']} |")
    lines += ['', '安全失败包含禁止接触、姿态或球高越限、外力和非有限状态；与实际掉球、截止超时分别统计。',
              '所有策略复用相同场景；三个训练种子的汇总不代表新增独立测试场景。', '',
              '## 各配置接稳数', '', '| 策略 | '+' | '.join(ppo_test['evaluation_cases'])+' |',
              '| --- | '+' | '.join('---' for _ in ppo_test['evaluation_cases'])+' |']
    pooled = [('经典控制器', controls), ('固定残差', fixed_test['trials'])]
    for algorithm, _, test in tests:
        pooled.append((algorithm.upper(), [t for r in test['records'] if r['kind'] == 'trained' for t in r['trials']]))
    for label, trials in pooled:
        counts = []
        for case in ppo_test['evaluation_cases']:
            values = [t for t in trials if t['case'] == case]
            counts.append(f"{sum(t['is_success'] for t in values)}/{len(values)}")
        lines.append('| '+label+' | '+' | '.join(counts)+' |')
    lines += ['', '## 候选筛选', '']
    for algorithm, _, test in tests:
        decision = screening_decisions(test)[algorithm]
        lines.append(f"- {algorithm.upper()}：{'通过' if decision['candidate_gate'] else '未通过'}；"
                     f"扰动配对完成数差 {decision['paired_perturbed_gain']:+d}，安全失败数为 {decision['unsafe_per_seed']}。")
    fixed_pass = (fixed['baseline'] >= baseline['baseline'] and fixed['perturbed'] > baseline['perturbed']
                  and fixed['unsafe'] <= baseline['unsafe'])
    lines += [f"- 固定残差：{'通过' if fixed_pass else '未通过'}相同的成功与安全门槛。", '',
              '## 训练预算与耗时', '', '| 算法/种子 | 实际训练步数 | 选中步数 | 训练耗时（s） |',
              '| --- | --- | --- | --- |']
    for algorithm, training, _ in tests:
        for row in training['records']:
            if row['kind'] == 'trained':
                lines.append(f"| {algorithm.upper()}/{row['seed']} | {row['training_steps']} | "
                             f"{row['selected_training_steps']} | {row['training_wall_time_s']:.2f} |")
    lines += ['', '耗时来自实际 learn 调用，不含验证；受机器负载和并发影响，不作为严格的算法速度排名。',
              '相同交互预算不等于相同梯度更新次数，也不表示任一算法已收敛。', '', '## 结论与范围', '',
              '经典控制器、固定残差及两个 RL 算法的结果用于判断是否值得继续训练；候选筛选不触发控制器替换。']
    consistent = []
    for algorithm, _, test in tests:
        trained = [trial_counts(r['trials']) for r in test['records'] if r['kind'] == 'trained']
        if (screening_decisions(test)[algorithm]['candidate_gate']
                and all(c['perturbed'] > fixed['perturbed'] and c['baseline'] >= fixed['baseline']
                        and c['unsafe'] <= fixed['unsafe'] for c in trained)):
            consistent.append(algorithm.upper())
    if consistent:
        lines.append('本测试中，'+ '、'.join(consistent)+' 各训练种子均超过固定残差，作为后续训练候选；差异仍需更多场景确认。')
    else:
        lines.append('本测试未显示任一算法在全部训练种子上稳定超过固定残差。后续保留经典控制器和固定残差对照，暂不据此扩展三球 RL。')
    lines += ['本比较仅覆盖一种有限预算配置，不能判定 SAC 或 PPO 的普遍优劣。',
              '固定残差仅搜索上述有限候选；策略实时反馈的额外收益仍需结合反馈消融判断。',
              '任务仅为下降单球的接稳，不包含主动抛球、双球或完整三球任务。', '']
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
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text)
        print(text)


if __name__ == '__main__':
    main()

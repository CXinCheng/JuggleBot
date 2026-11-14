"""Validate and summarize reception training, nominal checks and independent tests."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from .train_residual import EVALUATION_CASES
from .evaluate_residual import compatible_training_sources


def archived_trials(archive):
    rows = archive['records']
    raw = ''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows).encode()
    # Text-mode JSONL uses platform newlines; preserve the original raw digest.
    for candidate in (raw, raw.replace(b'\n', b'\r\n')):
        if hashlib.sha256(candidate).hexdigest() == archive['trials_sha256']:
            return rows, candidate
    raise ValueError('Archived trial checksum mismatch')


def validate_trial(config, trial):
    if trial.get('environment_version') != 2:
        raise ValueError('Trial environment version mismatch')
    if trial.get('nominal_feedback') != (trial['policy'] != 'static'):
        raise ValueError('Nominal feedback mode mismatch')
    if trial['is_success'] != (trial['reason'] == 'SUCCESS'):
        raise ValueError('Success/reason mismatch')
    expected_event = config['success_reward'] if trial['is_success'] else config['failure_reward']
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
    if manifest.get('environment_version') != 2:
        raise ValueError('Expected the feedback reception environment, version 2')
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
    for row in rows:
        policies = ('zero', 'static', 'random') if row['kind'] == 'controls' else ('learned',)
        expected_eval = {(p, c, s) for p in policies for c in EVALUATION_CASES
                         for s in manifest['evaluation_seeds']}
        keys = [(t['policy'], t['case'], t['seed']) for t in row['trials']]
        if set(keys) != expected_eval or len(keys) != len(expected_eval):
            raise ValueError('Evaluation seed/case coverage mismatch')
        if row['kind'] == 'trained' and (row['training_steps'] != manifest['steps']
                                        or not row['save_load_verified']):
            raise ValueError('Training budget or serialization mismatch')
        for trial in row['trials']:
            validate_trial(manifest['config'], trial)
    return trained, controls[0]


def summarize(manifest, rows):
    trained, controls = validate(manifest, rows)
    records = []
    for policy in ('zero', 'static', 'random'):
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
        for case in EVALUATION_CASES:
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
    lines = ['# 单球接球残差控制验证', '',
             '任务为预置下降球的接稳，不包含主动抛球或完整三球任务。', '',
             'zero 为经典反馈控制器；static 关闭其落点预测，保留相同的手指控制和接触缓冲。', '',
             '结果属于验证集，最终结论需结合独立测试和反馈消融，不能单独证明闭环抗噪能力。', '',
             f"每次训练 {manifest['steps']} 个控制步；训练种子：{manifest['training_seeds']}。",
             f"每个感知配置使用 {len(manifest['evaluation_seeds'])} 个独立评估种子，关闭策略探索。", '',
             '| 策略 | 学习率 | 训练引导权重 | 零扰动成功 | 1 mm 成功 | 3 mm 成功 | 组合扰动成功 | 训练耗时（s） |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in aggregates:
        counts = [f"{r['cases'][c]['successes']}/{r['cases'][c]['episodes']}" for c in EVALUATION_CASES]
        lines.append(f"| {r['policy']} | {r['learning_rate'] if r['learning_rate'] is not None else '—'} | "
                     f"{r['shaping_weight'] if r['shaping_weight'] is not None else '—'} | "
                     + ' | '.join(counts) + f" | {r['training_wall_time_s']:.2f} |")
    lines += ['', '各对照策略评估一次；训练策略的汇总分母包含全部训练种子。',
              '引导权重列指训练设置；评估回报使用清单中的统一奖励配置，策略优劣以独立验收指标衡量。',
              '不同训练种子共享评估场景，不能把这些重复场景视为独立样本。',
              '本预算用于验证训练链路和筛选配置，不能据此断言 SAC 或 PPO 已收敛或普遍更优。',
              '评估集用于本次筛选；后续确认实验须另选种子。耗时包含本机并发负载影响。',
              '名义接球根据位置采样预测落点，接触后下沉缓冲；本子任务不等同于完整三球任务。', '']
    return aggregates, '\n'.join(lines)


def confirmation_report(receipt):
    if receipt.get('environment_version') != 2:
        raise ValueError('Expected environment version 2')
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
            validate_trial(receipt['config'], trial)
    title = '单球接球独立测试' if training else '单球名义控制器验收'
    lines = ['# '+title, '', '任务为单球接球，不包含主动抛球或完整三球循环。', '',
             f'每个感知配置使用 {len(seeds)} 个场景种子：{seeds}。', '',
             '| 策略 | '+' | '.join(cases)+' |',
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
    lines += ['', 'zero 为完整经典反馈控制器，static 仅关闭落点预测，保留相同的抓握与缓冲。',
              '学习策略按全部训练种子汇总；各训练种子使用相同测试场景，不视为独立场景样本。',
              '策略评估关闭探索；失败原因、接稳证据和动作指标保存在原始 JSON 中。', '']
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
        decisions[algorithm] = dict(baseline_per_seed=baseline_counts, nominal_baseline=nominal_baseline,
                                     paired_perturbed_gain=gain, baseline_preserved=preserves,
                                     candidate_gate=preserves and gain > 0)
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
        sections.append(text.replace('# 单球接球残差控制验证', '## 训练与验证', 1))
    if nominal is not None:
        if not compatible_training_sources(nominal['source_sha256'], test['source_sha256']):
            raise ValueError('Nominal verification revision differs')
        if nominal['config'] != test['config']:
            raise ValueError('Nominal verification task configuration differs')
        sections.append(confirmation_report(nominal).replace('# 单球名义控制器验收', '## 名义控制器验收', 1))
    if feedback is not None:
        if (feedback['environment_version'] != 2
                or not compatible_training_sources(feedback['training_manifest']['source_sha256'], test['source_sha256'])
                or feedback['config'] != test['config']):
            raise ValueError('Feedback diagnostic revision differs')
        diagnostic_lines = ['## 反馈诊断', '', '| 策略/模式 | 配置 | 成功 |', '| --- | --- | --- |']
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
                        validate_trial(feedback['config'], trial)
                    label = row.get('checkpoint', policy).split('_lr')[0]+' / '+row['mode']
                    diagnostic_lines.append(f"| {label} | {case} | {sum(t['is_success'] for t in selected)}/{len(selected)} |")
        sections.append('\n'.join(diagnostic_lines))
        sections.append('\n策略球状态遮盖仅作用直接输入字段，名义参考仍含球信息；冻结策略输出仍保留经典控制器。\n')
    if test.get('training_manifest'):
        sections.append('## 筛选门槛\n\n各训练种子均须保持零扰动成功数，且扰动场景的配对汇总须超过经典控制器。此门槛不触发自动替换。\n')
        for algorithm, decision in screening_decisions(test).items():
            verdict = '通过候选筛选' if decision['candidate_gate'] else '未通过候选筛选'
            sections.append(f"- {algorithm}：{verdict}；零扰动各种子 {decision['baseline_per_seed']}，"
                            f"经典控制器 {decision['nominal_baseline']}；扰动配对完成数差 {decision['paired_perturbed_gain']:+d}。")
        sections.append('\n本预算与测试集不能证明算法普遍优劣；须同时检查各配置的退化与实际碰撞。三球任务尚未接入学习策略。\n')
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
    args = parser.parse_args()
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
    else:
        archive = json.loads(args.input.read_text(encoding='utf-8'))
        if 'scope' in archive:
            if args.output is not None:
                parser.error('Independent evaluation is already archived; specify --report only')
            optional = {name: json.loads(getattr(args, name).read_text(encoding='utf-8'))
                        if getattr(args, name) is not None else None
                        for name in ('training', 'nominal', 'feedback')}
            report = delivery_report(archive, **optional)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report, encoding='utf-8')
            print(report)
            return
        manifest = archive['manifest']
        rows, raw = archived_trials(archive)
    aggregates, report = summarize(manifest, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(manifest=manifest, trials_sha256=hashlib.sha256(raw).hexdigest(),
                                             aggregates=aggregates, records=rows), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    args.report.write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()

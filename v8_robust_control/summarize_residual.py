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
    lines = ['# 单球接球残差控制验证', '',
             '任务为预置下降球的接稳，不包含主动抛球或完整三球任务。', '',
             'zero 为经典反馈控制器；static 关闭其落点预测，保留相同的手指控制和接触缓冲。', '',
             '结果属于验证集，最终结论需结合独立测试和反馈消融，不能单独证明闭环抗噪能力。', '',
             f"每次训练 {manifest['steps']} 个控制步；训练种子：{manifest['training_seeds']}。",
             f"每个感知配置使用 {len(manifest['evaluation_seeds'])} 个独立评估种子，关闭策略探索。", '',
             *(['表中训练策略的结果来自各自选中的检查点；各训练种子均执行完整预算。', '']
               if 'checkpoints' in manifest else []),
             '| 策略 | 学习率 | 训练引导权重 | '+' | '.join(cases)+' | 训练耗时（s） |',
             '| --- | --- | --- | '+' | '.join('---' for _ in cases)+' | --- |']
    for r in aggregates:
        counts = [f"{r['cases'][c]['successes']}/{r['cases'][c]['episodes']}" for c in cases]
        lines.append(f"| {r['policy']} | {r['learning_rate'] if r['learning_rate'] is not None else '—'} | "
                     f"{r['shaping_weight'] if r['shaping_weight'] is not None else '—'} | "
                     + ' | '.join(counts) + f" | {r['training_wall_time_s']:.2f} |")
    lines += ['', '各对照策略评估一次；训练策略的汇总分母包含全部训练种子。',
              '引导权重列指训练设置；评估回报使用清单中的统一奖励配置，策略优劣以独立验收指标衡量。',
              '不同训练种子共享评估场景，不能把这些重复场景视为独立样本。',
              '本预算用于验证训练链路和筛选配置，不能据此断言 SAC 或 PPO 已收敛或普遍更优。',
              '评估集用于本次筛选；后续确认实验须另选种子。耗时包含本机并发负载影响。',
              '名义接球根据位置采样预测落点，接触后下沉缓冲；本子任务不等同于完整三球任务。', '']
    if 'checkpoints' in manifest:
        lines += ['## 学习曲线', '',
                  '各预算沿同一次训练连续推进；检查点仅按验证集选择，独立测试不参与选择。', '',
                  '训练耗时为各次 learn 调用的耗时之和，不含检查点评估；多个种子的耗时相加不等于并发任务的总历时。', '',
                  '| 算法/训练种子 | 控制步 | '+' | '.join(cases)+' | 选用 |',
                  '| --- | --- | '+' | '.join('---' for _ in cases)+' | --- |']
        for run in trained:
            for point in run['validation_curve']:
                counts = [f"{sum(t['is_success'] for t in point['trials'] if t['case'] == c)}/{len(manifest['evaluation_seeds'])}"
                          for c in cases]
                lines.append(f"| {run['algorithm']}/{run['seed']} | {point['training_steps']} | "
                             + ' | '.join(counts) + (' | 是 |' if point['training_steps'] == run['selected_training_steps'] else ' | — |'))
        lines += ['', '学习曲线使用同一组验证场景；误差带表示训练种子间范围，不是成功率置信区间。',
                  '较大训练预算不保证改善；检查点被选中也不代表训练已收敛。', '']
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
    lines += ['', 'zero 为名义接球反馈控制器，static 仅关闭落点预测，保留相同的抓握与缓冲。',
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
        sections.append(text.replace('# 单球接球残差控制验证', '## 训练与验证', 1))
    if nominal is not None:
        if not compatible_training_sources(nominal['source_sha256'], test['source_sha256']):
            raise ValueError('Nominal verification revision differs')
        if nominal['config'] != test['config']:
            raise ValueError('Nominal verification task configuration differs')
        sections.append(confirmation_report(nominal).replace('# 单球名义控制器验收', '## 名义控制器验收', 1))
    if feedback is not None:
        if (feedback['environment_version'] != test['environment_version']
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
                        validate_trial(feedback['config'], trial, feedback['environment_version'])
                    label = row.get('checkpoint', policy).split('_lr')[0]+' / '+row['mode']
                    diagnostic_lines.append(f"| {label} | {case} | {sum(t['is_success'] for t in selected)}/{len(selected)} |")
        sections.append('\n'.join(diagnostic_lines))
        sections.append('\n策略球状态遮盖仅作用直接输入字段，名义参考仍含球信息；冻结策略输出仍保留经典控制器。\n')
        if test['environment_version'] >= 3:
            checkpoints = dict.fromkeys(r['checkpoint'] for r in feedback['records'] if 'checkpoint' in r)
            for checkpoint in checkpoints:
                modes = {r['mode']: r['trials'] for r in feedback['records'] if r.get('checkpoint') == checkpoint}
                normal, frozen = modes.get('unmodified', []), modes.get('frozen_initial', [])
                if normal and len(normal) == len(frozen):
                    count, fixed = sum(t['is_success'] for t in normal), sum(t['is_success'] for t in frozen)
                    sections.append(f"{checkpoint.split('_lr')[0]}：正常策略 {count}/{len(normal)}，"
                                    f"冻结初始输出 {fixed}/{len(frozen)}；两者保留相同的名义反馈。"
                                    + ('本组小样本未显示实时策略反馈的额外收益。' if fixed >= count else
                                       '本组小样本中的差异仍需更多独立场景确认。'))
    if test.get('training_manifest'):
        sections.append('## 筛选门槛\n\n各训练种子均须保持零扰动成功数，且扰动场景的配对汇总须超过经典控制器。此门槛不触发自动替换。\n')
        for algorithm, decision in screening_decisions(test).items():
            verdict = '通过候选筛选' if decision['candidate_gate'] else '未通过候选筛选'
            sections.append(f"- {algorithm}：{verdict}；零扰动各种子 {decision['baseline_per_seed']}，"
                            f"经典控制器 {decision['nominal_baseline']}；扰动配对完成数差 {decision['paired_perturbed_gain']:+d}。")
            if test['environment_version'] >= 3:
                sections.append(f"  安全失败数：经典控制器 {decision['nominal_unsafe']}；"
                                f"各训练种子 {decision['unsafe_per_seed']}。任一种子增加安全失败数均不通过。")
            if test['environment_version'] >= 4:
                episodes = decision['perturbed_episodes_per_seed']
                repeats = len(decision['perturbed_per_seed'])
                sections.append(f"  扰动对照实际完成 {decision['nominal_perturbed']}/{episodes}，"
                                f"各策略完成数 {decision['perturbed_per_seed']}（各 {episodes} 次）。"
                                f"配对汇总将同一对照重复 {repeats} 次，不代表新增独立场景。")
        sections.append('\n本预算与测试集不能证明算法普遍优劣；须同时检查各配置的退化与实际碰撞。三球任务尚未接入学习策略。\n')
    if test['environment_version'] >= 3:
        lines = ['## 首次接触后的估计', '',
                 '误差仅使用有效估计时刻统计；真值用于诊断，不输入策略。无有效估计的回合不计作零误差。', '',
                 '| 策略 | 配置 | 有效估计时刻数 | 位置 RMSE（mm） | 速度 RMSE（m/s） |',
                 '| --- | --- | --- | --- | --- |']
        groups = [('zero', [t for r in test['records'] if r['kind'] == 'controls' for t in r['trials'] if t['policy'] == 'zero'])]
        for algorithm in test.get('algorithms', []):
            groups.append((algorithm, [t for r in test['records'] if r['kind'] == 'trained' and r['algorithm'] == algorithm for t in r['trials']]))
        safety = ['## 安全与动作指标', '',
                  '| 策略 | 配置 | 评估次数 | 禁止接触 | 姿态越限 | 球高越限 | 其他安全失败 | 截止超时 | 平均残差 RMS | 最大单轴偏移（mm） |',
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
            args.report.write_text(report, encoding='utf-8')
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
                                             aggregates=aggregates, records=rows), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    args.report.write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()

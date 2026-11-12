"""Validate perception trial records and generate a deterministic degradation report."""
import argparse
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
from statistics import mean

from jugglebot.robustness.perception import PerceptionConfig


def validate(rows):
    if not rows or rows[0].get('kind') != 'experiment' or rows[-1].get('kind') != 'complete':
        raise ValueError('A complete experiment manifest and completion marker are required')
    manifest = rows[0]
    trials = rows[1:-1]
    expected = {(case['name'],seed) for case in manifest['cases'] for seed in manifest['seeds']}
    keys = [(r.get('case'),r.get('seed')) for r in trials]
    if len(keys) != len(set(keys)) or set(keys) != expected or rows[-1]['trials'] != len(expected):
        raise ValueError('Missing, duplicate or unexpected trials')
    for row in trials:
        if row.get('kind') != 'trial':
            raise ValueError('Unexpected record type')
        case = next(c for c in manifest['cases'] if c['name'] == row['case'])
        cfg = asdict(PerceptionConfig(**{k:case[k] for k in ('noise_std_m','delay_s','dropout_probability')},seed=row['seed']))
        result = row['result']
        settings = result['conditions']['ball_observation']
        transport = settings['perception']
        if row['perception'] != cfg or any(transport[k] != v for k,v in cfg.items()):
            raise ValueError('Trial disturbance configuration differs from manifest')
        if settings['mode'] != 'sampled' or settings['sample_period_s'] != manifest['observation_period_s'] or settings['max_age_s'] != manifest['observation_max_age_s']:
            raise ValueError('Observation settings differ from manifest')
        count = result['validated_flights']
        if result['requested_throws'] != manifest['requested_flights'] or not 0 <= count <= manifest['requested_flights']:
            raise ValueError('Invalid flight counts')
        if not result['guard'] or result['grasp_assistance'] or result['grasp_constraints']:
            raise ValueError('Guarded pure-contact trials required')
        if result['success'] and (count != manifest['requested_flights'] or result['drops'] or result['ball_contacts']):
            raise ValueError('Inconsistent successful result')
        if transport['sampled_frames'] != transport['dropped_frames']+transport['delivered_frames']+transport['pending_frames']:
            raise ValueError('Frame accounting mismatch')
        if len(transport['trace_sha256']) != 64:
            raise ValueError('Missing perception trace digest')
    return manifest,sorted(trials,key=lambda r:(r['case'],r['seed']))


def reports(rows):
    manifest,trials = validate(rows)
    text = ['# v7 感知扰动基线','',
            f"每次请求 {manifest['requested_flights']} 次完整飞行；测试种子：{', '.join(map(str,manifest['seeds']))}。",
            '控制器参数固定，使用位置采样与三点状态估计。噪声为各坐标轴独立高斯噪声，延迟为固定 FIFO 交付延迟，丢帧按整帧独立采样。',
            '接触、安全检查和私有规划试算使用仿真真值。以下结果属于仿真敏感性评估，扰动范围未经真实传感器标定。','',
            '| 配置 | 噪声 σ (mm) | 延迟 (ms) | 丢帧概率 | 成功/试验 | 平均完成飞行 | 掉球总数 | 球球碰撞总数 |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for case in manifest['cases']:
        selected = [r for r in trials if r['case']==case['name']]
        success = sum(r['result']['success'] for r in selected)
        text.append(f"| {case['name']} | {case['noise_std_m']*1000:g} | {case['delay_s']*1000:g} | {case['dropout_probability']:g} | {success}/{len(selected)} | {mean(r['result']['validated_flights'] for r in selected):.1f} | {sum(r['result']['drops'] for r in selected)} | {sum(r['result']['ball_contacts'] for r in selected)} |")
    text += ['', '各条件的试验次数有限；成功次数用于描述本次实验，不作为总体成功概率的精确估计。',
             '规划拒绝、姿态越限和数据过期造成的失败均保留在逐次结果中。估计误差仅统计实时空中球的有效估计，按物理步采样；无效估计单独计数。','']
    stream = io.StringIO(newline='')
    writer = csv.writer(stream,lineterminator='\n')
    writer.writerow(['case','seed','noise_std_mm','delay_ms','dropout_probability','success','complete_flights','requested_flights','drops','ball_collisions','failure','position_rmse_m','velocity_rmse_m_s','invalid_estimates','perception_trace_sha256'])
    for row in trials:
        r,e,c = row['result'],row['estimation'],row['perception']
        writer.writerow([row['case'],row['seed'],c['noise_std_m']*1000,c['delay_s']*1000,c['dropout_probability'],r['success'],r['validated_flights'],manifest['requested_flights'],r['drops'],r['ball_contacts'],(r.get('failure_event') or {}).get('reason') or r.get('error') or '',e['position_rmse_m'],e['velocity_rmse_m_s'],e['invalid_samples'],r['conditions']['ball_observation']['perception']['trace_sha256']])
    return '\n'.join(text),stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--csv-output',type=Path,required=True)
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    paths = [p.resolve() for p in (args.input,args.output,args.csv_output)]
    if len(set(paths)) != 3:
        parser.error('Input and output paths must be distinct')
    if not args.overwrite and any(p.exists() for p in (args.output,args.csv_output)):
        parser.error('Reports already exist; use --overwrite to regenerate')
    try:
        text,csv_text = reports([json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()])
    except (ValueError,KeyError,TypeError) as error:
        parser.error(str(error))
    for path,payload in ((args.output,text),(args.csv_output,csv_text)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(payload,encoding='utf-8',newline='')


if __name__ == '__main__':
    main()

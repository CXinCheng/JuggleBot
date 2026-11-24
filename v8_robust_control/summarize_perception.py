"""Validate perception trial records and generate a deterministic degradation report."""
import argparse
import csv
from dataclasses import asdict
import io
import json
import math
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
        if manifest.get('task') == 'complete':
            if result.get('task') != 'three_ball_start_cascade_finish':
                raise ValueError('Complete task result required')
            if result['success']:
                initial, final = result.get('initial_hold'), result.get('final_hold')
                if (not initial or not initial['ok'] or not final or not final['ok']
                        or initial['continuous_support_s'] < .05 - 1e-12
                        or len(initial['balls']) != 3
                        or final['continuous_support_s'] < .8 - 1e-12
                        or len(final['balls']) != 3
                        or not all(b['contact'] and b['relative_speed_m_s'] < .15 for b in final['balls'].values())
                        or result['airborne_at_end'] or result['throws'] != manifest['requested_flights']):
                    raise ValueError('Successful complete task lacks continuous three-ball support')
                for support in (initial, final):
                    balls = support['balls']
                    if (set(balls) != {'0','1','2'}
                            or sorted(b['hand'] for b in balls.values()) != ['left','right','right']
                            or not all(b['contact'] and b['relative_speed_m_s'] < .15
                                       and b['height_m'] > .4
                                       and math.sqrt(sum(x*x for x in b['local_m'])) < .16
                                       for b in balls.values())):
                        raise ValueError('Invalid three-ball support geometry or speed')
        if transport['sampled_frames'] != transport['dropped_frames']+transport['delivered_frames']+transport['pending_frames']:
            raise ValueError('Frame accounting mismatch')
        if len(transport['trace_sha256']) != 64:
            raise ValueError('Missing perception trace digest')
    return manifest,sorted(trials,key=lambda r:(r['case'],r['seed']))


def _estimator_note(trials):
    def estimator_of(row):
        return row['result']['conditions']['ball_observation'].get('estimator', 'three_position_backward_difference')

    def is_undisturbed(row):
        p = row['perception']
        return p['noise_std_m'] == 0 and p['delay_s'] == 0 and p['dropout_probability'] == 0

    estimators = {estimator_of(r) for r in trials}
    undisturbed = {estimator_of(r) for r in trials if is_undisturbed(r)}
    disturbed = {estimator_of(r) for r in trials if not is_undisturbed(r)}
    if estimators == {'three_position_backward_difference'}:
        desc = 'position sampling and three-point state estimation'
    elif estimators == {'contact_support_and_ballistic_fit'}:
        desc = 'position sampling and filtered (contact-support and ballistic-fit) state estimation'
    elif (undisturbed == {'three_position_backward_difference'}
          and disturbed == {'contact_support_and_ballistic_fit'}):
        desc = 'position sampling (three-point state estimation when undisturbed; filtered contact-support and ballistic-fit estimation under perception disturbances)'
    else:
        desc = 'position sampling with recorded estimators: ' + ', '.join(sorted(map(str, estimators)))
    return f'Controller parameters are fixed, using {desc}. Noise is independent per-axis Gaussian noise, delay is a fixed FIFO delivery delay, and dropout drops whole frames independently.'


def reports(rows):
    manifest,trials = validate(rows)
    estimator_note = _estimator_note(trials)
    text = ['# Perception Disturbance Evaluation Results','',
            f"Requested {manifest['requested_flights']} complete flights per trial; test seeds: {', '.join(map(str,manifest['seeds']))}.",
            estimator_note,
            'Contacts, safety checks, and private planning rollouts use simulation ground truth. These results are a simulation sensitivity study; disturbance ranges are not calibrated to real hardware sensors.','',
            '| Case | Noise sigma (mm) | Delay (ms) | Dropout prob | Success/Trials | Mean completed flights | Total drops | Total ball-ball collisions |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for case in manifest['cases']:
        selected = [r for r in trials if r['case']==case['name']]
        success = sum(r['result']['success'] for r in selected)
        text.append(f"| {case['name']} | {case['noise_std_m']*1000:g} | {case['delay_s']*1000:g} | {case['dropout_probability']:g} | {success}/{len(selected)} | {mean(r['result']['validated_flights'] for r in selected):.1f} | {sum(r['result']['drops'] for r in selected)} | {sum(r['result']['ball_contacts'] for r in selected)} |")
    text += ['', 'Trial counts per condition are limited; success counts summarize this experiment and are not an accurate estimate of population success probability.',
             'Failures from planning rejections, posture limit violations, and expired measurements are retained in per-trial records. Estimation error is computed only over valid online airborne-ball estimates sampled per physics step; invalid estimates are counted separately.','']
    if manifest.get('task') == 'complete':
        text[2:2] = ['The task includes the three-ball 2+1 opening, cascade, and 2+1 ending. Success requires completing all target flights plus 0.8 s continuous three-ball hand contact with relative palm speed < 0.15 m/s; initial hold requires >= 0.05 s continuous stability.', '']
        text += ['The ball-ball collisions column counts forbidden contacts only. Permitted opening pair support contact before the first confirmed release has its step count and peak force listed separately in the per-trial CSV and raw records.', '']
    stream = io.StringIO(newline='')
    writer = csv.writer(stream,lineterminator='\n')
    complete_task = manifest.get('task') == 'complete'
    extra_headers = ['initial_support_s','final_support_s','supported_pair_contact_steps','supported_pair_force_max_n'] if complete_task else []
    writer.writerow(['case','seed','noise_std_mm','delay_ms','dropout_probability','success','complete_flights','requested_flights','drops','ball_collisions','failure','position_rmse_m','velocity_rmse_m_s','invalid_estimates','perception_trace_sha256']+extra_headers)
    for row in trials:
        r,e,c = row['result'],row['estimation'],row['perception']
        extra = [(r.get('initial_hold') or {}).get('continuous_support_s'),
                 (r.get('final_hold') or {}).get('continuous_support_s'),
                 r.get('supported_pair_contact_steps'),r.get('supported_pair_force_max_n')] if complete_task else []
        writer.writerow([row['case'],row['seed'],c['noise_std_m']*1000,c['delay_s']*1000,c['dropout_probability'],r['success'],r['validated_flights'],manifest['requested_flights'],r['drops'],r['ball_contacts'],(r.get('failure_event') or {}).get('reason') or r.get('error') or '',e['position_rmse_m'],e['velocity_rmse_m_s'],e['invalid_samples'],r['conditions']['ball_observation']['perception']['trace_sha256']]+extra)
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

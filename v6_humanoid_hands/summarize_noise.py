"""Validate a completed noise-evaluation JSONL and export TXT/CSV reports.

This command does not run a simulation. Replacing a report requires --overwrite.
"""
import argparse
import csv
import io
import json
from pathlib import Path

from v6_humanoid_hands.evaluate_noise import failure_category, summarize


def load_experiment(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if (len(records) < 3 or records[0].get('kind') != 'experiment'
            or records[0].get('schema_version') != 1 or records[-1].get('kind') != 'summary'):
        raise ValueError('Expected a complete version-1 experiment ending with a summary record')
    header, trials = records[0], records[1:-1]
    if (len(set(header['noise_std_mm'])) != len(header['noise_std_mm'])
            or len(set(header['seeds'])) != len(header['seeds'])):
        raise ValueError('Duplicate noise levels or seeds in the experiment definition')
    expected = {(noise, seed) for noise in header['noise_std_mm'] for seed in header['seeds']}
    actual = [(row['noise_std_mm'], row['seed']) for row in trials]
    if (not expected or any(row.get('kind') != 'trial' for row in trials)
            or len(actual) != len(expected) or set(actual) != expected):
        raise ValueError('Missing, duplicate or unexpected trial configuration')
    static_fields = ('mujoco_version', 'timestep_s', 'ball_diameter_m', 'ball_mass_kg',
                     'throw_interval_s', 'flight_target_s')
    conditions = trials[0]['result']['conditions']
    for row in trials:
        result = row['result']
        current = result['conditions']
        if (not 0 <= result['validated_flights'] <= result['requested_throws']
                or result['drops'] < 0 or result['ball_contacts'] < 0
                or (result['success'] and (result['validated_flights'] != result['requested_throws']
                                          or result['drops'] or result['ball_contacts'] or result['error']))):
            raise ValueError('Inconsistent completion or failure counters')
        if (result['requested_throws'] != header['requested_flights']
                or current['observation_noise_seed'] != row['seed']
                or current['partial_prediction_noise_m'] != row['noise_std_mm']/1000.
                or any(current[key] != conditions[key] for key in static_fields)):
            raise ValueError('Trial conditions disagree with the experiment configuration')
        if row['outcome'] != failure_category(result):
            raise ValueError('Trial outcome disagrees with its recorded result')
    if summarize(trials) != records[-1]['groups']:
        raise ValueError('Stored summary disagrees with per-trial results')
    return header, sorted(trials, key=lambda row: (row['noise_std_mm'], row['seed']))


def render_report(header, trials):
    conditions = trials[0]['result']['conditions']
    noise = ' '.join(f'{value:g}' for value in header['noise_std_mm'])
    seeds = ' '.join(str(value) for value in header['seeds'])
    export_command = ('uv run python -m v6_humanoid_hands.summarize_noise '
                      '--input v6_humanoid_hands/results/noise_sensitivity.jsonl '
                      '--output v6_humanoid_hands/results/noise_sensitivity.txt '
                      '--csv-output v6_humanoid_hands/results/noise_trials.csv')
    baseline = sorted((row for row in trials if row['noise_std_mm'] == 0), key=lambda row: row['seed'])
    if (baseline and baseline[0]['result']['success']
            and baseline[0]['result'].get('execution', {}).get('samples')
            and baseline[0]['result']['execution'].get('actuators')):
        export_command += ' --execution-output v6_humanoid_hands/results/execution_demand.csv'
    export_command += ' --overwrite'
    lines = [
        'v6 partial receive-position prediction noise: saved experiment summary',
        'Generated from a completed JSONL by v6_humanoid_hands.summarize_noise.',
        'This is an experiment summary, not a unit-test transcript or hardware validation.',
        '', 'Validate the archived experiment and regenerate derived reports from the repository root:',
        export_command,
        '', 'To regenerate the experiment, first back up or remove the existing JSONL:',
        f"uv run python -m v6_humanoid_hands.evaluate_noise --noise-mm {noise} --seeds {seeds} "
        f"--throws {header['requested_flights']} --timeout {header['simulated_timeout_s']:g} "
        f"--jobs {header['jobs']} --output v6_humanoid_hands/results/noise_sensitivity.jsonl",
        'The archived JSONL is retained with its derived TXT/CSV files in results/.',
        'Evaluation refuses an existing JSONL. After evaluation, regenerate the derived reports with the command above.',
        'To compare experiments, choose distinct JSONL filenames and matching report input/output paths.',
        '', f"Completed trials: {len(trials)}",
        f"Requested released-and-caught flights per trial: {header['requested_flights']}",
        f"Simulated timeout per trial: {header['simulated_timeout_s']} s",
        f"Independent worker processes: {header['jobs']}",
        f"MuJoCo version: {conditions['mujoco_version']}",
        f"Timestep: {conditions['timestep_s']} s",
        f"Ball diameter / mass: {conditions['ball_diameter_m']} m / {conditions['ball_mass_kg']} kg",
        f"Alternating throw interval / target flight: {conditions['throw_interval_s']} s / {conditions['flight_target_s']} s",
        'Noise values are Gaussian position standard deviations in mm, not bounded errors.',
        'Only part of receive-position prediction is perturbed. Velocity and rollout initialization use truth.',
        'No sensor sampling, latency, dropout or actuator-error model is included.',
        '', 'Per-trial results (completed flights exclude the initially airborne ball):',
        'std_mm  seed  completed/requested  outcome                       drops  ball_contacts',
    ]
    for row in sorted(trials, key=lambda row: (row['noise_std_mm'], row['seed'])):
        result = row['result']
        lines.append(f"{row['noise_std_mm']:6g}  {row['seed']:4d}  "
                     f"{result['validated_flights']:9d}/{result['requested_throws']:<9d}  "
                     f"{row['outcome']:<28s}  {result['drops']:5d}  {result['ball_contacts']:13d}")
    lines += ['', 'Group summaries:']
    for group in summarize(trials):
        lines.append(f"std={group['noise_std_mm']:g} mm: "
                     f"{group['successes']}/{group['trials']} completed the requested task; "
                     f"outcomes={json.dumps(group['outcomes'], sort_keys=True)}")
    lines += ['', 'Stop details:']
    for row in trials:
        result = row['result']
        if not result['success']:
            lines.append(f"std={row['noise_std_mm']:g} mm, seed={row['seed']}: {result['error']}")
            lines.append('  ' + json.dumps(result['failure_event'], sort_keys=True))
    lines += [
        '', 'Interpretation:',
        '- Planning rejection, simulated timeout, actual drop and ball-ball collision are distinct outcomes.',
        '- Empirical fractions from a small seed set do not establish population success probabilities.',
        '- Zero-noise seeds do not create distinct perturbation conditions.',
        '- This experiment does not replace the independent zero-noise 30/50-flight regression log.',
        '', 'Recorded optimizer-input SHA-256:', header['optimizer_inputs_sha256'],
        '', 'Recorded source SHA-256 at experiment start:',
    ]
    lines += [f'{name}: {digest}' for name, digest in sorted(header['source_sha256'].items())]
    return '\n'.join(lines) + '\n'


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def render_trials_csv(header, trials):
    rows = []
    for trial in sorted(trials, key=lambda row: (row['noise_std_mm'], row['seed'])):
        result = trial['result']
        event = result['failure_event'] or {}
        conditions = result['conditions']
        rows.append(dict(
            noise_std_mm=trial['noise_std_mm'], seed=trial['seed'],
            requested_flights=result['requested_throws'], completed_flights=result['validated_flights'],
            success=int(result['success']), outcome=trial['outcome'],
            drops=result['drops'], ball_contacts=result['ball_contacts'],
            simulated_time_s=result['time'], actual_releases=result['throws'],
            failure_reason=event.get('reason', result['error']) or '',
            failure_ball=event.get('ball', ''), failure_time_s=event.get('time', ''),
            failure_height_m=event.get('height', ''),
            timestep_s=conditions['timestep_s'], ball_diameter_m=conditions['ball_diameter_m'],
            ball_mass_kg=conditions['ball_mass_kg'], throw_interval_s=conditions['throw_interval_s'],
            flight_target_s=conditions['flight_target_s'], mujoco_version=conditions['mujoco_version'],
            optimizer_inputs_sha256=header['optimizer_inputs_sha256']))
    return csv_text(rows)


def render_execution_csv(header, trials):
    baseline = sorted((row for row in trials if row['noise_std_mm'] == 0), key=lambda row: row['seed'])
    if not baseline or not baseline[0]['result']['success']:
        raise ValueError('Execution CSV requires a successful zero-noise baseline (lowest recorded seed)')
    trial = baseline[0]
    result = trial['result']
    execution = result['execution']
    if not execution['samples'] or not execution['actuators']:
        raise ValueError('Execution CSV requires live actuator samples')
    rows = []
    for actuator in sorted(execution['actuators'], key=lambda row: row['actuator']):
        limits = actuator['configured_torque_range_nm']
        rows.append(dict(
            noise_std_mm=trial['noise_std_mm'], seed=trial['seed'],
            requested_flights=result['requested_throws'], completed_flights=result['validated_flights'],
            live_samples=execution['samples'], sample_period_s=execution['sample_period_s'],
            hardware_limits_verified=int(execution['hardware_limits_verified']),
            actuator=actuator['actuator'], joint=actuator['joint'],
            peak_speed_rad_s=actuator['peak_speed_rad_s'],
            peak_actuator_torque_nm=actuator['peak_actuator_torque_nm'],
            rms_actuator_torque_nm=actuator['rms_actuator_torque_nm'],
            configured_torque_min_nm=limits[0] if limits is not None else '',
            configured_torque_max_nm=limits[1] if limits is not None else '',
            near_force_limit_fraction=actuator['near_force_limit_fraction'],
            optimizer_inputs_sha256=header['optimizer_inputs_sha256']))
    return csv_text(rows)


def export_report(input_path, output_path, overwrite=False, csv_output=None, execution_output=None):
    targets = [Path(path) for path in (output_path, csv_output, execution_output) if path is not None]
    resolved = [path.resolve() for path in targets]
    if Path(input_path).resolve() in resolved or len(set(resolved)) != len(resolved):
        raise ValueError('Input JSONL and all output reports must be different files')
    header, trials = load_experiment(input_path)
    outputs = [(Path(output_path), render_report(header, trials))]
    if csv_output is not None:
        outputs.append((Path(csv_output), render_trials_csv(header, trials)))
    if execution_output is not None:
        outputs.append((Path(execution_output), render_execution_csv(header, trials)))
    # Validate/render every requested artifact and preflight existing paths
    # before writing any report. Never replace raw input data.
    if not overwrite:
        for target in targets:
            if target.exists():
                raise FileExistsError(f'Report already exists: {target}')
    for target, text in outputs:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w' if overwrite else 'x', encoding='utf-8', newline='') as stream:
            stream.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Completed evaluation JSONL.')
    parser.add_argument('--output', type=Path, required=True, help='TXT report path.')
    parser.add_argument('--csv-output', type=Path, help='Optional per-trial CSV for independent statistics.')
    parser.add_argument('--execution-output', type=Path,
                        help='Optional per-actuator CSV: zero-noise trial with lowest recorded seed.')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace existing output reports.')
    args = parser.parse_args()
    try:
        export_report(args.input, args.output, overwrite=args.overwrite,
                      csv_output=args.csv_output, execution_output=args.execution_output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f'Saved verified experiment summary: {args.output}')


if __name__ == '__main__':
    main()

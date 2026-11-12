"""Run the v7 timestamped-observation contact-controller experiment."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from jugglebot.robustness.controller import ObservedCascade
from jugglebot.robustness.observation import DEFAULT_OBSERVATION_PERIOD_S, DEFAULT_OBSERVATION_MAX_AGE_S


def experiment_record(result, requested_flights, simulated_timeout_s):
    """Archive the actual result and runtime-source fingerprints."""
    root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__)]
    for package in ('humanoid', 'robustness'):
        source_files.extend((root / 'src' / 'jugglebot' / package).glob('*.py'))
    seed_file = root / 'src/jugglebot/humanoid/leap_cascade_seed_plans.json'
    return dict(schema_version=1, experiment='v7 zero-perturbation observation baseline',
                requested_flights=requested_flights, simulated_timeout_s=simulated_timeout_s,
                source_sha256={path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(source_files)},
                optimizer_inputs_sha256=hashlib.sha256(seed_file.read_bytes()).hexdigest(),
                result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--throws', type=int, default=1, help='Released-and-caught flights to complete.')
    parser.add_argument('--timeout', type=float, help='Simulated time limit in seconds, not wall time.')
    parser.add_argument('--observation-mode', choices=('truth', 'sampled'), default='truth')
    parser.add_argument('--observation-period', type=float, default=DEFAULT_OBSERVATION_PERIOD_S)
    parser.add_argument('--observation-max-age', type=float, default=DEFAULT_OBSERVATION_MAX_AGE_S)
    parser.add_argument('--output', type=Path, help='Optional new JSON path for the complete experiment record.')
    args = parser.parse_args()
    if args.throws < 1:
        parser.error('--throws must be positive')
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error('--timeout must be finite and positive')
    if (not math.isfinite(args.observation_period) or args.observation_period <= 0
            or not math.isfinite(args.observation_max_age)
            or args.observation_max_age < args.observation_period):
        parser.error('--observation-period must be positive and finite; --observation-max-age must be finite and at least the period')
    if args.output is not None and args.output.exists():
        parser.error('--output must name a new file')
    timeout = args.timeout if args.timeout is not None else max(30., .6*args.throws+5.)
    record = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        record = experiment_record(None, args.throws, timeout)
    controller = ObservedCascade(observation_mode=args.observation_mode,
                                 observation_period_s=args.observation_period,
                                 observation_max_age_s=args.observation_max_age)
    controller.progress_interval = 10
    print(f'[Setup] v7 | observation: {args.observation_mode} | requested flights: {args.throws}', flush=True)
    result = controller.run_continuous(args.throws, timeout, guard=True)
    summary = {key: result[key] for key in (
        'success', 'validated_flights', 'throws', 'time', 'drops', 'ball_contacts', 'error', 'failure_event')}
    summary['ball_observation'] = result['conditions']['ball_observation']
    print(json.dumps(summary, indent=2), flush=True)
    if args.output is not None:
        record['result'] = result
        with args.output.open('x', encoding='utf-8', newline='') as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
        print(f'Saved experiment: {args.output}', flush=True)
    if not result['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

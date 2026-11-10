"""Explicit, reproducible partial receive-position noise experiments.

Writes one JSONL file only when --output is provided. Each run is flushed so
an interrupted experiment retains its completed trials. No seed/model updates.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np

from jugglebot.humanoid.cascade import ContactCascade
from v6_humanoid_hands.juggle import benchmark_timeout


def failure_category(result):
    if result['success']:
        return 'success'
    event = result.get('failure_event') or {}
    reason = event.get('reason') or result.get('error') or 'UNKNOWN'
    if result.get('drops'):
        return 'drop'
    if result.get('ball_contacts'):
        return 'ball_ball_collision'
    if 'NO_FEASIBLE_RECEPTION' in reason:
        return 'reception_planning_rejected'
    if reason == 'TIMEOUT':
        return 'timeout'
    return reason


def summarize(runs):
    groups = []
    for noise in sorted({row['noise_std_mm'] for row in runs}):
        selected = [row for row in runs if row['noise_std_mm'] == noise]
        successes = sum(row['result']['success'] for row in selected)
        groups.append(dict(noise_std_mm=noise, trials=len(selected), successes=successes,
                           success_rate=successes / len(selected),
                           completed_requested_flights=sum(row['result']['validated_flights'] for row in selected),
                           outcomes=dict(Counter(row['outcome'] for row in selected))))
    return groups


def run_trial(noise, seed, throws, timeout):
    controller = ContactCascade(obs_noise=noise / 1000., seed=seed)
    result = controller.run_continuous(throws, timeout, guard=True)
    return dict(kind='trial', noise_std_mm=noise, seed=seed,
                outcome=failure_category(result), result=result)


def trial_results(args, timeout):
    cases = [(noise, seed) for noise in args.noise_mm for seed in args.seeds]
    if args.jobs == 1:
        for noise, seed in cases:
            print(f'[Trial] noise std {noise:g} mm | seed {seed} | requested flights {args.throws}', flush=True)
            yield run_trial(noise, seed, args.throws, timeout)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(run_trial, noise, seed, args.throws, timeout) for noise, seed in cases]
            for future in as_completed(futures):
                yield future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--noise-mm', nargs='+', type=float, default=[0., 1., 3., 5., 10.])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--throws', type=int, default=30)
    parser.add_argument('--jobs', type=int, default=1, help='Independent worker processes; default 1 limits memory use.')
    parser.add_argument('--timeout', type=float, help='Simulated seconds per trial, not a wall-clock limit.')
    parser.add_argument('--output', type=Path, required=True, help='New JSONL path; existing files are never overwritten.')
    args = parser.parse_args()
    if args.throws < 1 or args.jobs < 1 or any(seed < 0 for seed in args.seeds):
        parser.error('throws and jobs must be positive and seeds nonnegative')
    if any(not np.isfinite(noise) or noise < 0 for noise in args.noise_mm):
        parser.error('noise standard deviations must be finite and nonnegative')
    if args.timeout is not None and (not np.isfinite(args.timeout) or args.timeout <= 0):
        parser.error('timeout must be finite and positive')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.noise_mm)) != len(args.noise_mm):
        parser.error('duplicate noise levels or seeds would repeat the same trial')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seed_file = Path(__file__).resolve().parents[1] / 'src/jugglebot/humanoid/leap_cascade_seed_plans.json'
    timeout = args.timeout if args.timeout is not None else benchmark_timeout(args.throws)
    source_files = [Path(__file__), Path(__file__).with_name('juggle.py')]
    source_files += list(seed_file.parent.glob('*.py'))
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(source_files)}
    runs = []
    with args.output.open('x', encoding='utf-8') as stream:
        def write(record):
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()

        write(dict(kind='experiment', schema_version=1,
                   experiment='partial receive-position prediction noise sensitivity',
                   limitations='velocity and rollout initialization use truth; no sensor sampling, latency or dropout model',
                   noise_std_mm=args.noise_mm, seeds=args.seeds, requested_flights=args.throws,
                   jobs=args.jobs,
                   simulated_timeout_s=timeout,
                   source_sha256=source_hashes,
                   optimizer_inputs_sha256=hashlib.sha256(seed_file.read_bytes()).hexdigest()))
        for row in trial_results(args, timeout):
            result = row['result']
            write(row)
            runs.append(row)
            print(f"[Result] noise std {row['noise_std_mm']:g} mm | seed {row['seed']} | "
                  f"{row['outcome']} | complete {result['validated_flights']}/{args.throws} | "
                  f"drops {result['drops']} | ball contacts {result['ball_contacts']}", flush=True)
        summary = summarize(runs)
        write(dict(kind='summary', groups=summary))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Compare cache reuse with identical physics, servo and task rules."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from .residual_env import CatchConfig, ResidualCatchEnv
from .train_residual import evaluate, EVALUATION_CASES, source_hashes, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--seed-start', type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists() or args.episodes < 1 or args.seed_start < 0:
        parser.error('Use a new output and valid seed/episode counts')
    records = []
    fingerprints = source_hashes()
    for reuse in (False, True):
        config = replace(CatchConfig(), reuse_forward_cache=reuse)
        started = time.perf_counter()
        trials = evaluate(None, config, list(range(args.seed_start, args.seed_start+args.episodes)),
                          policies=('zero',), cases={c: EVALUATION_CASES[c] for c in ('baseline', 'combined')})
        wall = sum(t['wall_time_s'] for t in trials)
        steps = sum(t['control_steps'] for t in trials)
        record = dict(config=asdict(config), trials=trials, rollout_wall_time_s=wall,
                      total_wall_time_s=time.perf_counter()-started, control_steps=steps,
                      control_steps_per_second=steps/wall)
        records.append(record)
        print('reuse', reuse, 'steps/s', round(steps/wall, 2), json.dumps(summary(trials)), flush=True)
    if source_hashes() != fingerprints:
        raise RuntimeError('Benchmark source changed during execution')
    receipt = dict(environment_version=ResidualCatchEnv.ENVIRONMENT_VERSION, source_sha256=fingerprints,
                   seed_start=args.seed_start, episodes=args.episodes, records=records,
                   interpretation='same integration, servo and per-step safety checks; reuse completed forward caches')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()

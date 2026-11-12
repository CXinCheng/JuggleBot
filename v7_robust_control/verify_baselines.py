"""Record the 30- and 50-flight checkpoints of one zero-perturbation run."""
import argparse
import hashlib
import json
from pathlib import Path

from jugglebot.robustness.controller import ObservedCascade
from .juggle import experiment_record

CHECKPOINTS = (30, 50)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation-mode', choices=('truth', 'sampled'), default='truth')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    destinations = {count: args.output_dir/f'{args.observation_mode}_baseline{count}.json'
                    for count in CHECKPOINTS}
    if any(path.exists() for path in destinations.values()):
        parser.error('baseline output files must be new')
    entry = Path(__file__).resolve()
    entry_name = entry.relative_to(entry.parents[1]).as_posix()
    entry_hash = hashlib.sha256(entry.read_bytes()).hexdigest()
    records = {}
    for count in CHECKPOINTS:
        record = experiment_record(None, count, max(30., .6*count+5.))
        record['source_sha256'][entry_name] = entry_hash
        record['continuous_checkpoints'] = list(CHECKPOINTS)
        records[count] = record
    args.output_dir.mkdir(parents=True, exist_ok=True)
    controller = ObservedCascade(observation_mode=args.observation_mode)
    controller.progress_interval = 10
    print(f'[Setup] v7 | observation: {args.observation_mode} | checkpoints: 30, 50', flush=True)
    for count in CHECKPOINTS:
        record = records[count]
        result = controller.run_continuous(count, record['simulated_timeout_s'], guard=True)
        record['result'] = result
        # Serialize before continuing: some result fields reference live metrics.
        payload = json.dumps(record, indent=2, allow_nan=False) + '\n'
        with destinations[count].open('x', encoding='utf-8', newline='') as stream:
            stream.write(payload)
        print(f"[Checkpoint] {result['validated_flights']}/{count} flights | "
              f"success: {result['success']} | {destinations[count]}", flush=True)
        if not result['success']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Independent reception evaluation with paired nominal and static controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from stable_baselines3 import SAC, PPO

from .residual_env import CatchConfig, ResidualCatchEnv
from .train_residual import EVALUATION_CASES, evaluate, source_hashes, summary

TEST_CASES = dict(EVALUATION_CASES,
                  noise_10mm={'noise_std_m': .010},
                  delay_40ms={'delay_s': .040},
                  dropout_30pct={'dropout': .30})


def compatible_training_sources(recorded, current=None):
    """A changed observation/controller runtime requires a new training run."""
    current = source_hashes() if current is None else current
    return recorded == current


def select_training_rows(rows, algorithms=None):
    """Use the algorithms present in the run, unless explicitly filtered."""
    trained = [row for row in rows if row['kind'] == 'trained']
    available = {row['algorithm'] for row in trained}
    if available - {'sac', 'ppo'}:
        raise ValueError('Unknown algorithm in the training run')
    if algorithms is None:
        return trained
    if not set(algorithms).issubset(available):
        raise ValueError('Requested algorithm is absent from the training run')
    return [row for row in trained if row['algorithm'] in algorithms]


def evaluate_checkpoint(arguments):
    path, row, config, seeds, cases = arguments
    torch.set_num_threads(1)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    model_type = SAC if row['algorithm'] == 'sac' else PPO
    model = model_type.load(path, device='cpu')
    if model.num_timesteps != row.get('selected_training_steps', row.get('training_steps', model.num_timesteps)):
        raise ValueError('Checkpoint training budget does not match the record')
    if model.observation_space.shape != (ResidualCatchEnv.OBSERVATION_SIZE,):
        raise ValueError('Policy belongs to a different environment version')
    records = evaluate(model, config, seeds, policies=('learned',), cases=cases)
    print(path.name, json.dumps(summary(records)), flush=True)
    return dict(kind='trained', algorithm=row['algorithm'], seed=row['seed'],
                learning_rate=row['learning_rate'], shaping_weight=row['shaping_weight'],
                selected_training_steps=row.get('selected_training_steps', row.get('training_steps')),
                checkpoint=path.name, checkpoint_sha256=digest, trials=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed-start', type=int, default=30000)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--cases', nargs='+', choices=list(TEST_CASES), default=list(EVALUATION_CASES))
    parser.add_argument('--algorithms', nargs='+', choices=['sac', 'ppo'],
                        help='Filter algorithms; by default evaluate those recorded in the run')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() or args.episodes < 1 or args.seed_start < 0 or args.jobs < 1:
        parser.error('Output must be new; episode count, jobs and seeds must be valid')
    torch.set_num_threads(1)
    seeds = list(range(args.seed_start, args.seed_start+args.episodes))
    cases = {name: TEST_CASES[name] for name in args.cases}
    training = None
    config = CatchConfig()
    tasks = []
    fingerprints = source_hashes()
    if args.input_dir:
        if not json.loads((args.input_dir/'complete.json').read_text())['complete']:
            raise ValueError('Training run is incomplete')
        training = json.loads((args.input_dir/'manifest.json').read_text())
        if training['environment_version'] != ResidualCatchEnv.ENVIRONMENT_VERSION:
            raise ValueError('Environment version mismatch')
        if not compatible_training_sources(training['source_sha256'], fingerprints):
            raise ValueError('Training source changed; use its recorded revision for evaluation')
        if set(seeds) & (set(training['training_seeds']) | set(training['evaluation_seeds'])):
            raise ValueError('Confirmation seeds overlap training or validation')
        config = CatchConfig(**training['config'])
        rows = [json.loads(line) for line in (args.input_dir/'trials.jsonl').read_text().splitlines()]
        for row in select_training_rows(rows, args.algorithms):
            tasks.append((args.input_dir/row['checkpoint'], row, config, seeds, cases))
        if not tasks:
            raise ValueError('No matching policies')
    controls = evaluate(None, config, seeds, policies=('zero', 'static'), cases=cases)
    records = [dict(kind='controls', trials=controls)]
    print('Controls', json.dumps(summary(controls)), flush=True)
    if args.jobs > 1 and tasks:
        with ProcessPoolExecutor(args.jobs) as pool:
            records.extend(pool.map(evaluate_checkpoint, tasks))
    else:
        records.extend(map(evaluate_checkpoint, tasks))
    if fingerprints != source_hashes():
        raise ValueError('Evaluation source changed during execution')
    receipt = dict(task='single_ball_reception', environment_version=ResidualCatchEnv.ENVIRONMENT_VERSION,
                   config=asdict(config), evaluation_seeds=seeds, evaluation_cases=cases,
                   training_manifest=training, source_sha256=fingerprints, records=records,
                   algorithms=sorted({task[1]['algorithm'] for task in tasks}),
                   evaluation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   screening_rule=('every training seed must preserve baseline success; '
                                   'pooled perturbed success must exceed paired nominal; '
                                   'no training seed may increase unsafe failures over nominal; '
                                   'no automatic deployment'),
                   scope='confirmation' if training else 'nominal_controller_verification')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()

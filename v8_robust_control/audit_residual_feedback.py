"""Separate nominal sensor feedback from learned residual feedback."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC

from .train_residual import EVALUATION_CASES, evaluate, summary
from .residual_env import CatchConfig, ResidualCatchEnv
from .evaluate_residual import compatible_training_sources, select_training_rows


class FeedbackAblation:
    """Ablate the actor only; nominal sensor feedback remains active."""
    def __init__(self, model, mode):
        self.model, self.mode = model, mode
        self.frozen = None

    def predict(self, observation, deterministic=True):
        observation = observation.copy()
        if self.mode == 'masked_ball':
            observation[:8] = [0., 0., 0., 0., 0., 0., 0., 2.]
        if self.mode == 'frozen_initial':
            if self.frozen is None or observation[ResidualCatchEnv.TIME_INDEX] == 0.:
                self.frozen, _ = self.model.predict(observation, deterministic=True)
            return self.frozen.copy(), None
        return self.model.predict(observation, deterministic=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--seed-start', type=int, default=40000)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--cases', nargs='+', choices=list(EVALUATION_CASES), default=['baseline', 'combined'])
    args = parser.parse_args()
    if args.output.exists() or args.seed_start < 0 or args.episodes < 1:
        parser.error('Output must be new and evaluation seeds must be valid')
    manifest = json.loads((args.input_dir/'manifest.json').read_text())
    if manifest['environment_version'] != ResidualCatchEnv.ENVIRONMENT_VERSION:
        raise ValueError('Environment version mismatch')
    if not compatible_training_sources(manifest['source_sha256']):
        raise ValueError('Use the training revision for feedback diagnostics')
    config = CatchConfig(**manifest['config'])
    evaluation_seeds = list(range(args.seed_start, args.seed_start+args.episodes))
    if set(evaluation_seeds) & (set(manifest['training_seeds']) | set(manifest['evaluation_seeds'])):
        raise ValueError('Diagnostic seeds overlap training or validation')
    cases = {name: EVALUATION_CASES[name] for name in args.cases}
    training_rows = [json.loads(line) for line in (args.input_dir/'trials.jsonl').read_text().splitlines()]
    by_checkpoint = {r['checkpoint']: r for r in select_training_rows(training_rows)}
    torch.set_num_threads(1)
    records = []
    controls = evaluate(None, config, evaluation_seeds, policies=('zero', 'static'), cases=cases)
    records.append(dict(mode='nominal_controls', trials=controls))
    print('Nominal controls', json.dumps(summary(controls)), flush=True)
    blackout = evaluate(None, config, evaluation_seeds, policies=('zero',),
                        cases={'sensor_blackout': {'dropout': 1.}})
    records.append(dict(mode='nominal_sensor_blackout', trials=blackout))
    for name in args.checkpoints:
        path = args.input_dir/name
        row = by_checkpoint[name]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row['checkpoint_sha256']:
            raise ValueError('Checkpoint checksum mismatch')
        model_type = SAC if row['algorithm'] == 'sac' else PPO
        model = model_type.load(path, device='cpu')
        for mode in ('unmodified', 'masked_ball', 'frozen_initial'):
            rows = evaluate(FeedbackAblation(model, mode), config,
                            evaluation_seeds, policies=('learned',), cases=cases)
            records.append(dict(checkpoint=name,
                                checkpoint_sha256=digest,
                                mode=mode, trials=rows))
            print(name, mode, json.dumps(summary(rows)), flush=True)
    result = dict(task='single_ball_reception', environment_version=ResidualCatchEnv.ENVIRONMENT_VERSION,
                  scope='feedback_diagnostic', config=asdict(config), training_manifest=manifest, records=records,
                  evaluation_cases=cases, evaluation_seeds=evaluation_seeds,
                  interpretation=('masked_ball removes direct actor ball fields but retains nominal references; '
                                  'frozen_initial fixes actor output; static/blackout ablate nominal sensor feedback'),
                  diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()

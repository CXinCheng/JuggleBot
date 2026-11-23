"""Test whether causal opening measurements predict useful bounded corrections."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .compare_residual import digest
from .task_env import TaskConfig, TaskResidualEnv, action_authority, runtime_sources
from .train_task import TaskCurriculum, write_new


# Normalized commands, decoded by the unchanged task environment.
PROFILES = dict(zero=[0., 0., 0., 0., 0., 0.],
    forward=[.1/.75, 0., 0., 0., 0., 0.],
    upward=[0., 0., .1/1.25, 0., 0., 0.],
    lead_minus3ms=[0., 0., 0., 0., -.003/.015, 0.],
    forward_upward=[.1/.75, 0., .1/1.25, 0., 0., 0.])
FEATURES = ('balls', 'errors', 'robot', 'robot_error', 'nominal', 'tactile', 'context')
# Three balls relative to the active palm (position/velocity), both palm
# position/velocity innovations, and measured palm/finger forces: 40 values.
LOCAL_INDICES = ([12*ball+axis for ball in range(3) for axis in range(6)]
                 + list(range(253, 265))+list(range(270, 280)))


def features(obs):
    """No seed, reset offset, outcome, simulator state or future observations."""
    value = np.concatenate([np.asarray(obs[key], dtype=float).ravel() for key in FEATURES])
    if value.shape != (306,) or not np.all(np.isfinite(value)):
        raise ValueError('Require the finite sampled task observation')
    return value


def is_opening(obs):
    return obs['context'][0] > .5 and obs['context'][5] == 0.


def outcome(info):
    recovered = (info['reason'] == 'TRAINING_PREFIX' and info['training_prefix']
                 and 1 in info['recovered_ids'] and not info['is_success'])
    if info['training_prefix'] != recovered:
        raise ValueError('Require physical first-flight recovery evidence')
    return [float(recovered), float(info['reason'] in TaskResidualEnv.UNSAFE)]


def episode(env, seed, actor):
    obs, _ = env.reset(seed=seed)
    initial = features(obs).tolist()
    if not is_opening(obs):
        raise ValueError('Expected the first throw callback')
    started = time.perf_counter()
    while True:
        obs, _, done, truncated, info = env.step(actor.predict(obs)[0])
        if done or truncated:
            break
    outcome(info)
    return dict(seed=seed, features=initial, wall_s=time.perf_counter()-started, **info)


class OpeningActor:
    def __init__(self, model):
        self.model = model

    def predict(self, obs, deterministic=True):
        if not is_opening(obs) or not action_authority(obs):
            return np.zeros(6), None
        index = choose(self.model, features(obs))
        return np.asarray(list(PROFILES.values())[index]), None


def collect_seed(job):
    reference, seed, jitter = job
    torch.set_num_threads(1)
    env = TaskCurriculum(TaskResidualEnv(TaskConfig(), reference),
                         [dict(catch_goal=1, seat_jitter_m=jitter)])
    rows = []
    try:
        for index, name in enumerate(PROFILES):
            row = episode(env, seed, OpeningActor(dict(kind='constant', index=index)))
            if rows:
                np.testing.assert_array_equal(row['features'], rows[0]['features'])
            rows.append(dict(profile=name, **row))
            print('[Opening capability]', seed, name, row['reason'], flush=True)
        return rows
    finally:
        env.close()


def matrices(rows):
    grouped = {}
    for row in rows:
        key = (row['seed'], row['profile'])
        if key in grouped:
            raise ValueError('Duplicate seed/profile')
        grouped[key] = row
    seeds = sorted({row['seed'] for row in rows})
    x, y = [], []
    for seed in seeds:
        states = [grouped[seed, name] for name in PROFILES]
        if any(row['features'] != states[0]['features'] for row in states):
            raise ValueError('Profile counterfactuals must share the initial observation')
        x.append(states[0]['features'])
        y.append([outcome(row) for row in states])
    return np.asarray(x), np.asarray(y)


def choose(model, x):
    if model['kind'] == 'constant':
        return model['index']
    x = np.asarray(x)
    if model.get('view') == 'local':
        x = x[LOCAL_INDICES]
    z = (x-model['center'])/np.asarray(model['scale'])
    if model['kind'] == 'ridge':
        prediction = np.r_[1., z] @ np.asarray(model['weights'])
        prediction = np.clip(prediction.reshape(len(PROFILES), 2), 0., 1.)
    else:
        distance = np.mean((np.asarray(model['x'])-z)**2, axis=1)
        prediction = np.asarray(model['y'])[np.argsort(distance, kind='stable')[:model['neighbors']]].mean(0)
    # A safe failed prefix has utility 0, recovery +1, unsafe failure -1.
    scores = prediction[:, 0]-prediction[:, 1]
    # Prefer zero / the smaller command when estimated utility is tied.
    costs = np.linalg.norm(np.asarray(list(PROFILES.values())), axis=1)
    return int(min(range(len(PROFILES)), key=lambda i: (-round(float(scores[i]), 10), costs[i], i)))


def fit_candidates(x, y, view='full'):
    if view not in ('full', 'local'):
        raise ValueError('Unknown observation feature view')
    if view == 'local':
        x = x[:, LOCAL_INDICES]
    center = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-5] = 1.
    z = (x-center)/scale
    common = dict(view=view, center=center.tolist(), scale=scale.tolist())
    result = []
    a = np.column_stack([np.ones(len(z)), z])
    # Dual ridge keeps the solve proportional to the small number of episodes.
    for penalty in (.1, 1., 10.):
        weights = a.T @ np.linalg.solve(a @ a.T+penalty*np.eye(len(a)), y.reshape(len(y), -1))
        result.append(dict(kind='ridge', penalty=penalty, weights=weights.tolist(), **common))
    for neighbors in (1, 3, 5):
        result.append(dict(kind='nearest', neighbors=min(neighbors, len(z)),
                           x=z.tolist(), y=y.tolist(), **common))
    return result


def score(model, x, y):
    indices = [choose(model, state) for state in x]
    selected = y[np.arange(len(y)), indices]
    return dict(recovered=int(selected[:, 0].sum()), unsafe=int(selected[:, 1].sum()),
                choices=[list(PROFILES)[i] for i in indices])


def select_model(x, y, vx, vy, view='full'):
    candidates = fit_candidates(x, y, view)
    # Validation only; the test set does not exist at selection time.
    scores = [score(model, vx, vy) for model in candidates]
    index = max(range(len(scores)), key=lambda i: (scores[i]['recovered']-scores[i]['unsafe'],
                                                  -scores[i]['unsafe'], -i))
    return candidates[index], scores


def fixed_model(y):
    values = y[:, :, 0].sum(0)-y[:, :, 1].sum(0)
    return dict(kind='constant', index=int(np.argmax(values)))


def load_training_data(folder, sources, reference_hash, jitter_mm):
    """Reuse every old split as training only; require fresh new holdouts."""
    protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    source = folder/'audit_source.py'
    if not source.exists():
        source = Path(__file__)
    if (protocol != report['protocol'] or protocol['source_sha256'] != sources
            or protocol['reference_sha256'] != reference_hash or protocol['jitter_mm'] != jitter_mm
            or protocol['profiles'] != PROFILES or protocol['audit_sha256'] != digest(source)
            or report['selectors_sha256'] != digest(folder/'selectors.json')
            or protocol['environment_version'] != TaskResidualEnv.ENVIRONMENT_VERSION):
        raise ValueError('Reused data must have matching physics, measurements, actions and provenance')
    rows, hashes = [], {}
    for name, group in (('train.json', 'train'), ('validation.json', 'validation'),
                        ('test_library.json', 'test')):
        path = folder/name
        block = json.loads(path.read_text(encoding='utf-8'))
        matrices(block)
        if {row['seed'] for row in block} != set(protocol['seed_groups'][group]):
            raise ValueError('Reused data seed grid differs from its protocol')
        rows.extend(block)
        hashes[path.as_posix()] = digest(path)
    matrices(rows)
    hashes[(folder/'protocol.json').as_posix()] = digest(folder/'protocol.json')
    hashes[(folder/'report.json').as_posix()] = digest(folder/'report.json')
    return rows, hashes


def replay_seed(job):
    reference, seed, jitter, models = job
    torch.set_num_threads(1)
    env = TaskCurriculum(TaskResidualEnv(TaskConfig(), reference),
                         [dict(catch_goal=1, seat_jitter_m=jitter)])
    try:
        rows = {name: episode(env, seed, OpeningActor(model)) for name, model in models.items()}
        print('[Frozen replay]', seed, {name: row['reason'] for name, row in rows.items()}, flush=True)
        return rows
    finally:
        env.close()


def collect(reference, seeds, jitter, jobs):
    tasks = [(str(reference), seed, jitter) for seed in seeds]
    if jobs == 1:
        blocks = [collect_seed(task) for task in tasks]
    else:
        with ProcessPoolExecutor(jobs) as pool:
            blocks = list(pool.map(collect_seed, tasks))
    return [row for block in blocks for row in block]


def check_splits(groups):
    if (any(not group or len(set(group)) != len(group) or min(group) < 0 for group in groups)
            or len(set().union(*map(set, groups))) != sum(map(len, groups))):
        raise ValueError('Require nonempty disjoint nonnegative seed sets')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--training-data', type=Path,
                        help='Reuse all splits of a verified earlier audit as training data')
    parser.add_argument('--train-seeds', type=int, nargs='+', default=list(range(410210, 410222)))
    parser.add_argument('--validation-seeds', type=int, nargs='+', default=list(range(410230, 410236)))
    parser.add_argument('--test-seeds', type=int, nargs='+', default=list(range(410250, 410256)))
    parser.add_argument('--jitter-mm', type=float, default=.5)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists() or not 0. < args.jitter_mm <= 10. or args.jobs < 1:
        parser.error('Use a new output directory, positive jitter <= 10 mm and positive jobs')
    sources, reference_hash = runtime_sources(), digest(args.reference)
    driver_hash = digest(Path(__file__))
    reused, reused_hashes = (load_training_data(args.training_data, sources, reference_hash, args.jitter_mm)
                            if args.training_data else ([], {}))
    train_seeds = sorted({row['seed'] for row in reused})+args.train_seeds
    check_splits([train_seeds, args.validation_seeds, args.test_seeds])
    args.output_dir.mkdir(parents=True)
    protocol = dict(scope='first-flight ready recovery; not complete-task or RL acceptance',
        source_sha256=sources, audit_sha256=driver_hash, reference=str(args.reference),
        environment_version=TaskResidualEnv.ENVIRONMENT_VERSION,
        reference_sha256=reference_hash, task_config=asdict(TaskConfig()),
        seed_groups=dict(train=train_seeds, validation=args.validation_seeds, test=args.test_seeds),
        reused_as_training_sha256=reused_hashes,
        profiles=PROFILES, feature_keys=FEATURES, jitter_mm=args.jitter_mm,
        local_feature_indices=LOCAL_INDICES,
        utility='confirmed recovery minus unsafe failure', label_shuffle_seeds=[0, 1, 2],
        baseline_scope='matched controller without online planning; not the original planning controller')
    write_new(args.output_dir/'protocol.json', protocol)
    train = reused+collect(args.reference, args.train_seeds, args.jitter_mm/1000., args.jobs)
    write_new(args.output_dir/'train.json', train)
    validation = collect(args.reference, args.validation_seeds, args.jitter_mm/1000., args.jobs)
    write_new(args.output_dir/'validation.json', validation)
    x, y = matrices(train)
    vx, vy = matrices(validation)
    learned, candidates = select_model(x, y, vx, vy)
    local, local_candidates = select_model(x, y, vx, vy, 'local')
    models = dict(zero=dict(kind='constant', index=0), fixed=fixed_model(y),
                  fixed_validated=fixed_model(vy), learned=learned, local=local)
    shuffle_candidates = {}
    for seed in range(3):
        shuffled, scores = select_model(x, y[np.random.default_rng(seed).permutation(len(y))], vx, vy, 'local')
        models[f'shuffled{seed}'] = shuffled
        shuffle_candidates[str(seed)] = scores
    frozen_path = args.output_dir/'selectors.json'
    write_new(frozen_path, dict(models=models, validation_scores=candidates,
        local_validation_scores=local_candidates,
        shuffled_validation_scores=shuffle_candidates, protocol_sha256=digest(args.output_dir/'protocol.json')))
    frozen_hash = digest(frozen_path)
    # Collecting all test actions measures library coverage, never selects a model.
    test = collect(args.reference, args.test_seeds, args.jitter_mm/1000., args.jobs)
    write_new(args.output_dir/'test_library.json', test)
    tx, ty = matrices(test)
    results = {name: score(model, tx, ty) for name, model in models.items()}
    # Replay the selected actions through the actual observation-only actor.
    tasks = [(str(args.reference), seed, args.jitter_mm/1000., models) for seed in sorted(args.test_seeds)]
    if args.jobs == 1:
        blocks = [replay_seed(task) for task in tasks]
    else:
        with ProcessPoolExecutor(args.jobs) as pool:
            blocks = list(pool.map(replay_seed, tasks))
    replay = {name: [block[name] for block in blocks] for name in models}
    for name, rows in replay.items():
        actual = np.asarray([outcome(row) for row in rows])
        chosen = [list(PROFILES).index(value) for value in results[name]['choices']]
        np.testing.assert_array_equal(actual, ty[np.arange(len(ty)), chosen])
    if (sources != runtime_sources() or reference_hash != digest(args.reference)
            or driver_hash != digest(Path(__file__)) or frozen_hash != digest(frozen_path)
            or any(digest(Path(path)) != value for path, value in reused_hashes.items())):
        raise ValueError('Audit inputs changed')
    coverage = int(np.any(ty[:, :, 0] > .5, axis=1).sum())
    write_new(args.output_dir/'report.json', dict(protocol=protocol, selectors_sha256=frozen_hash,
        train_episodes=len(x), validation_episodes=len(vx),
        train_scores={name:score(model, x, y) for name, model in models.items()},
        validation_scores={name:score(model, vx, vy) for name, model in models.items()},
        train_library_coverage=int(np.any(y[:, :, 0] > .5, axis=1).sum()),
        test_library_coverage=coverage, test_episodes=len(tx), results=results, replay=replay,
        complete_task_verified=False, rl_improvement_verified=False))
    print('[Learning audit]', json.dumps(results), 'library coverage', coverage, '/', len(tx), flush=True)


if __name__ == '__main__':
    main()

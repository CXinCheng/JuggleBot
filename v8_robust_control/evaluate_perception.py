"""Evaluate the fixed v8 controller under position noise, delay and frame dropout."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing
from pathlib import Path

from jugglebot.robustness.evaluation import EvaluatedCascade
from jugglebot.robustness.perception import PerceptionConfig
from .juggle import experiment_record


def cases():
    """One-factor sweeps followed by a combined condition; SI units internally."""
    rows = [dict(name='baseline', group='baseline', noise_std_m=0., delay_s=0., dropout_probability=0.)]
    for mm in (1,3,5,10):
        rows.append(dict(name=f'noise_{mm}mm',group='noise',noise_std_m=mm/1000.,delay_s=0.,dropout_probability=0.))
    for ms in (10,20,40):
        rows.append(dict(name=f'delay_{ms}ms',group='delay',noise_std_m=0.,delay_s=ms/1000.,dropout_probability=0.))
    for probability in (.05,.15,.30):
        rows.append(dict(name=f'dropout_{round(100*probability)}pct',group='dropout',noise_std_m=0.,delay_s=0.,dropout_probability=probability))
    rows.append(dict(name='combined',group='combined',noise_std_m=.003,delay_s=.020,dropout_probability=.10))
    return rows


def run_trial(case, seed, throws, timeout, sample_period_s, max_age_s, video_path=None, task='cascade'):
    config = PerceptionConfig(**{key:case[key] for key in ('noise_std_m','delay_s','dropout_probability')},seed=seed)
    recorder_factory = None
    if video_path:
        from jugglebot.video import VideoRecorder
        from v6_humanoid_hands.juggle import cascade_camera
        recorder_factory = lambda model: VideoRecorder(model, Path(video_path), camera=cascade_camera())
    factory = EvaluatedCascade
    extra = {}
    if task == 'complete':
        from .complete_task import EvaluatedCompleteCascade
        factory = EvaluatedCompleteCascade
        extra['throws'] = throws
    controller = factory(observation_mode='sampled',perception=config,
                                 observation_period_s=sample_period_s, observation_max_age_s=max_age_s,
                                 recorder_factory=recorder_factory, **extra)
    try:
        result = (controller.run_complete(timeout) if task == 'complete'
                  else controller.run_continuous(throws,timeout,guard=True))
    finally:
        if controller.recorder:
            controller.recorder.close()
    row = dict(kind='trial',case=case['name'],seed=seed,perception=asdict(config),result=result,
               estimation=controller.estimation_summary())
    if controller.recorder:
        row['video'] = dict(path=Path(video_path).as_posix(),fps=60,speed=1.,frames=controller.recorder.frames)
    return row


def source_fingerprints(task='cascade'):
    root = Path(__file__).resolve().parents[1]
    if task == 'complete':
        from .complete_task import complete_record
        record = complete_record(None, 30, 40.)
        # Include every task calibration file in the during-run change check.
        record['source_sha256'].update(record['task_inputs_sha256'])
    else:
        record = experiment_record(None,1,30.)
    for name in ('evaluate_perception.py','summarize_perception.py'):
        path = Path(__file__).with_name(name)
        record['source_sha256'][path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record['source_sha256'],record['optimizer_inputs_sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('cascade','complete'),default='cascade')
    parser.add_argument('--cases',nargs='+',choices=[c['name'] for c in cases()],default=[c['name'] for c in cases()])
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1])
    parser.add_argument('--throws',type=int,default=30)
    parser.add_argument('--timeout',type=float)
    parser.add_argument('--jobs',type=int,default=1)
    parser.add_argument('--observation-period',type=float,default=1/60)
    parser.add_argument('--observation-max-age',type=float,default=.05)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--video-cases',nargs='*',choices=[c['name'] for c in cases()],default=[])
    parser.add_argument('--video-dir',type=Path,default=Path('videos'))
    args = parser.parse_args()
    if args.throws < 1 or args.jobs < 1 or any(s < 0 for s in args.seeds):
        parser.error('throws and jobs must be positive; seeds must be nonnegative')
    if args.task == 'complete' and (args.throws < 4 or args.throws % 2):
        parser.error('Complete task requires an even --throws >= 4')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.cases)) != len(args.cases):
        parser.error('Duplicate seeds or cases')
    if not set(args.video_cases).issubset(args.cases):
        parser.error('Video cases must be included in the evaluation cases')
    if not math.isfinite(args.observation_period) or args.observation_period <= 0 or not math.isfinite(args.observation_max_age) or args.observation_max_age < args.observation_period:
        parser.error('Invalid observation period or maximum age')
    timeout = args.timeout if args.timeout is not None else max(30., .6*args.throws+5.)
    if not math.isfinite(timeout) or timeout <= 0:
        parser.error('timeout must be finite and positive')
    if args.output.exists():
        parser.error('Output JSONL must be a new file')
    chosen = [c for c in cases() if c['name'] in args.cases]
    work = []
    for case in chosen:
        for seed in args.seeds:
            prefix = 'v8_complete_before_rl' if args.task == 'complete' else 'v8_before_rl'
            video = args.video_dir/f'{prefix}_{case["name"]}_seed{seed}.mp4' if case['name'] in args.video_cases and seed == args.seeds[0] else None
            if video and video.exists():
                parser.error(f'Video already exists: {video}')
            work.append((case,seed,args.throws,timeout,args.observation_period,args.observation_max_age,str(video) if video else None,args.task))
    hashes,seed_hash = source_fingerprints(args.task)
    manifest = dict(kind='experiment',schema_version=1,experiment='v8 perception sensitivity',
                    controller='fixed_v8_classical_contact',cases=chosen,seeds=args.seeds,
                    task=args.task,
                    requested_flights=args.throws,simulated_timeout_s=timeout,jobs=args.jobs,
                    observation_period_s=args.observation_period,observation_max_age_s=args.observation_max_age,
                    source_sha256=hashes,optimizer_inputs_sha256=seed_hash,
                    scope='position perception only; contact, safety and private rollouts use truth; no policy training')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='') as stream:
        def write(row):
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            stream.flush()
        write(manifest)
        if args.jobs == 1:
            results = map(lambda task: run_trial(*task),work)
            for row in results:
                write(row)
                print(f"[Result] {row['case']} seed {row['seed']}: {row['result']['validated_flights']}/{args.throws}, success={row['result']['success']}",flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(run_trial,*task) for task in work]
                for future in as_completed(futures):
                    row = future.result()
                    write(row)
                    print(f"[Result] {row['case']} seed {row['seed']}: {row['result']['validated_flights']}/{args.throws}, success={row['result']['success']}",flush=True)
        after,_ = source_fingerprints(args.task)
        if after != hashes:
            raise RuntimeError('Runtime sources changed during evaluation; no completion marker written')
        write(dict(kind='complete',trials=len(work)))


if __name__ == '__main__':
    main()

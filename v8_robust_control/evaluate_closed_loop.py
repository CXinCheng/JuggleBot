"""Held-out 7-gradient evaluation for the native ClosedLoopBvpPolicy & ClosedLoopTaskEnv.

Evaluates the frozen closed-loop BVP + torque-CBF architecture on the exact
10 held-out test seeds (440000..440009) across all 7 seat-jitter gradients
(0.00, 0.02, 0.10, 0.25, 0.50, 1.00, 2.00 mm) using the official filtered
reference trajectory, strictly matching the leak-free ablation protocol.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .closed_loop_bvp import ClosedLoopBvpPolicy, ClosedLoopConfig, ClosedLoopTaskEnv
from .task_env import TaskConfig, TaskResidualEnv, measured_sources

TEST_SEEDS = tuple(440000 + i for i in range(10))
JITTER_GRADIENTS_MM = (0.00, 0.02, 0.10, 0.25, 0.50, 1.00, 2.00)
_CANDIDATE_REFS = [
    Path('v8_robust_control/results/ablation_baselines/task_filtered_reference_final.json'),
    Path('build/v8_task_filtered_reference_final.json'),
]
DEFAULT_REFERENCE = next((p for p in _CANDIDATE_REFS if p.exists()), _CANDIDATE_REFS[0])


def _eval_episode(job: tuple[int, float, str, dict]) -> dict:
    seed, jitter_mm, reference_str, cfg_dict = job
    cfg = ClosedLoopConfig(**cfg_dict)
    policy = ClosedLoopBvpPolicy(cfg)
    env = ClosedLoopTaskEnv(
        TaskConfig(throws=30, horizon_s=12.0, seat_jitter_m=jitter_mm * 1e-3),
        reference=reference_str,
        closed_loop_cfg=cfg,
    )
    try:
        obs, info = env.reset(seed=seed)
        done = trunc = False
        while not (done or trunc):
            act = policy.predict(obs, env)
            obs, _, done, trunc, info = env.step(act)
        cbf_elbow = int(getattr(env.controller, 'stiff_elbow_cbf_steps', 0))
        cbf_wrist = int(getattr(env.controller, 'wrist_cbf_steps', 0))
        uncaught_steps = int(getattr(env.controller, 'uncaught_braking_steps', 0))
        perturbed = bool(getattr(env.controller, 'perturbed_active', False))
    finally:
        env.close()
    return dict(
        seed=seed,
        jitter_mm=jitter_mm,
        catches=int(info['catches']),
        releases=int(info['releases']),
        elapsed_s=round(float(info['elapsed_s']), 3),
        reason=str(info['reason']),
        is_success=bool(info['is_success']),
        is_unsafe=bool(info['reason'] in TaskResidualEnv.UNSAFE),
        max_ball_height_m=round(float(info['max_ball_height_m'] or 0.0), 4),
        max_executed_action=round(float(info['max_executed_action']), 4),
        perturbed_active=perturbed,
        stiff_elbow_cbf_steps=cbf_elbow,
        wrist_cbf_steps=cbf_wrist,
        uncaught_braking_steps=uncaught_steps,
        failure_event=info.get('failure_event'),
    )


def evaluate_suite(
    seeds=TEST_SEEDS,
    jitters_mm=JITTER_GRADIENTS_MM,
    reference: Path = DEFAULT_REFERENCE,
    cfg: ClosedLoopConfig | None = None,
    workers: int = 4,
) -> dict:
    cfg = cfg or ClosedLoopConfig()
    cfg_dict = asdict(cfg)
    jobs = [(int(s), float(j), str(reference), cfg_dict) for j in jitters_mm for s in seeds]
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for row in pool.map(_eval_episode, jobs):
            rows.append(row)
    summary = {}
    for j in jitters_mm:
        sub = [r for r in rows if abs(r['jitter_mm'] - j) < 1e-9]
        catches = [r['catches'] for r in sub]
        summary[f'{j:.2f}mm'] = dict(
            jitter_mm=float(j),
            episodes=len(sub),
            successes=sum(int(r['is_success']) for r in sub),
            mean_catches=round(float(np.mean(catches)), 2),
            max_catches=int(max(catches)),
            unsafe_count=sum(int(r['is_unsafe']) for r in sub),
            unsafe_rate=round(float(np.mean([r['is_unsafe'] for r in sub])), 3),
            catches_per_seed=catches,
            reasons=[r['reason'] for r in sub],
        )
    return dict(
        schema_version=1,
        policy='ClosedLoopBvpPolicy+ClosedLoopBvpCascade',
        config=cfg_dict,
        reference=str(reference),
        source_sha256=measured_sources(),
        seeds=list(seeds),
        wall_s=round(time.perf_counter() - t0, 2),
        summary=summary,
        episodes=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(TEST_SEEDS))
    parser.add_argument('--jitters', type=float, nargs='+', default=list(JITTER_GRADIENTS_MM))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('build/v8_closed_loop_7gradient_test.json'))
    args = parser.parse_args()
    report = evaluate_suite(
        seeds=tuple(args.seeds),
        jitters_mm=tuple(args.jitters),
        reference=args.reference,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"\n=== Native Closed-Loop BVP Held-Out Test Summary (wall={report['wall_s']}s) ===", flush=True)
    for key, s in report['summary'].items():
        print(
            f"  {key:<7} succ={s['successes']:2d}/{s['episodes']} "
            f"mean={s['mean_catches']:5.2f} max={s['max_catches']:2d} "
            f"unsafe={s['unsafe_count']:2d}/{s['episodes']} catches={s['catches_per_seed']}",
            flush=True,
        )


if __name__ == '__main__':
    main()

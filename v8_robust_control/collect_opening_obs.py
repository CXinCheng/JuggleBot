"""Collect k=0 opening observations (zero residual, truncated after a few decisions) for
paired helps/hurts diagnosis. No training, no 450000..450019.

k=0 observation is identical across candidates for a given (seed, jitter, perception), so it is a
clean pre-intervention feature vector.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from v8_robust_control.closed_loop_bvp import ClosedLoopConfig, ClosedLoopResidualEnv
from v8_robust_control.task_env import TaskConfig

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "v8_robust_control/results/ablation_baselines/task_filtered_reference_final.json"
JITTERS = (0.02, 0.10, 0.25, 0.50, 1.00, 2.00)
CONDS = {
    "clean": dict(noise_std_m=0.0, delay_s=0.0, dropout=0.0),
    "combined": dict(noise_std_m=0.001, delay_s=0.020, dropout=0.10),
}


def _worker(job):
    pairs, pk, n_dec = job
    env = ClosedLoopResidualEnv(
        TaskConfig(throws=30, horizon_s=12.0, **pk), reference=str(REF),
        closed_loop_cfg=ClosedLoopConfig(), residual_scale=0.25,
        active_max_decisions=5, active_max_released=3,
        active_mask=(1., 1., 1., 0., 0., 0.))
    out = []
    try:
        for seed, j in pairs:
            env.config = replace(env.config, seat_jitter_m=j * 1e-3, **pk)
            obs, _ = env.reset(seed=seed)
            rec = dict(seed=seed, jitter_mm=j, obs=[])
            for k in range(n_dec):
                rec["obs"].append({n: v.astype(float).tolist() for n, v in obs.items()})
                obs, _, done, trunc, _ = env.step(np.zeros(6))
                if done or trunc:
                    break
            out.append(rec)
    finally:
        env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(440000, 440010)))
    ap.add_argument("--cond", choices=list(CONDS), default="clean")
    ap.add_argument("--n-dec", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert not any(450000 <= s <= 450019 for s in a.seeds), "final test seeds are frozen"
    pairs = [(s, j) for j in JITTERS for s in a.seeds]
    jobs = [(pairs[i::a.workers], CONDS[a.cond], a.n_dec) for i in range(a.workers)]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = [r for b in pool.map(_worker, jobs) for r in b]
    a.out.write_text(json.dumps(dict(cond=a.cond, rows=rows)), encoding="utf-8")
    print("saved", a.out, len(rows))


if __name__ == "__main__":
    main()

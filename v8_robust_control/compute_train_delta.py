"""Compute empirical mean opening deltas of iter=2 on TRAIN_SEEDS."""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from stable_baselines3 import PPO

from v8_robust_control.closed_loop_bvp import ClosedLoopConfig, ClosedLoopResidualEnv
from v8_robust_control.task_env import TaskConfig

ROOT = Path(__file__).resolve().parents[1]
REF_PATH = ROOT / "v8_robust_control" / "results" / "ablation_baselines" / "task_filtered_reference_final.json"
MODEL_PATH = ROOT / "v8_robust_control" / "models" / "ppo_cl_opening_pilot.zip"

TRAIN_SEEDS = tuple(435000 + i for i in range(16))
PERTURBED_JITTERS_MM = (0.02, 0.10, 0.25, 0.50, 1.00, 2.00)

from concurrent.futures import ProcessPoolExecutor

def _worker_batch(pairs):
    model = PPO.load(str(MODEL_PATH), device="cpu")
    env = ClosedLoopResidualEnv(
        TaskConfig(throws=30, horizon_s=12.0),
        reference=str(REF_PATH),
        closed_loop_cfg=ClosedLoopConfig(),
        residual_scale=0.25,
        active_max_decisions=5,
        active_max_released=3,
        active_mask=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        record_action_log=True,
    )
    logs = []
    try:
        for s, j in pairs:
            env.config = replace(env.config, seat_jitter_m=j * 1e-3)
            obs, info = env.reset(seed=s)
            done = trunc = False
            while not (done or trunc):
                act, _ = model.predict(obs, deterministic=True)
                obs, _, done, trunc, info = env.step(act)
            logs.append([(e["k"], e["requested_delta"], e["learned_delta"]) for e in env.action_log])
    finally:
        env.close()
    return logs

def main():
    pairs = [(s, j) for j in PERTURBED_JITTERS_MM for s in TRAIN_SEEDS]
    n = len(pairs)
    batches = [pairs[:n//2], pairs[n//2:]]
    with ProcessPoolExecutor(2) as pool:
        results = [r for batch in pool.map(_worker_batch, batches) for r in batch]

    deltas_by_k_req = {}
    deltas_by_k_lrn = {}
    for ep in results:
        for k, req, lrn in ep:
            deltas_by_k_req.setdefault(k, []).append(req)
            deltas_by_k_lrn.setdefault(k, []).append(lrn)

    result = {}
    print(f"=== Empirical Mean Opening Deltas on TRAIN_SEEDS (16 seeds x 6 jitters = 96 eps) ===")
    for k in sorted(deltas_by_k_req.keys()):
        req_arr = np.array(deltas_by_k_req[k])
        lrn_arr = np.array(deltas_by_k_lrn[k])
        req_m = req_arr.mean(axis=0).round(6).tolist()
        req_s = req_arr.std(axis=0).round(6).tolist()
        lrn_m = lrn_arr.mean(axis=0).round(6).tolist()
        lrn_s = lrn_arr.std(axis=0).round(6).tolist()
        result[k] = {
            "count": len(req_arr),
            "requested_mean": req_m,
            "requested_std": req_s,
            "learned_mean": lrn_m,
            "learned_std": lrn_s,
        }
        print(f"k={k} (N={len(req_arr)}):")
        print(f"  requested mean={req_m[:3]} std={req_s[:3]}")
        print(f"  learned   mean={lrn_m[:3]} std={lrn_s[:3]}")

    out_file = ROOT / "v8_robust_control" / "results" / "closed_loop_bvp" / "train_mean_deltas_iter2.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()

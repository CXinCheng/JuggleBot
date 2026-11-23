"""Train and evaluate official ImitationPolicy (SB3 PPO with NormalizedMeasurements)
and 7D-k0 Active-Subspace LCB Policy on strictly disjoint TRAIN / VAL / TEST splits.

Uses official `ImitationPolicy` and `NormalizedMeasurements` from `v8_robust_control.task_imitation`.
Writes the fitted checkpoint and evaluation report to `build/v8_distilled_imitation_ppo/`.
"""
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from v8_robust_control.task_env import TaskResidualEnv, TaskConfig, action_authority
from v8_robust_control.task_imitation import (
    ImitationPolicy, NormalizedMeasurements, observation_statistics, initialize_model, fit_actor
)
from v8_robust_control.train_stagewise_imitation_ppo import (
    REFERENCE, TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS, JITTER_MM,
    K0_CANDIDATES, REC_CANDIDATES, APEX_TRIM_ACT, extract_stage_features
)

_REP_CANDIDATES = [
    ROOT / "v8_robust_control" / "results" / "ablation_baselines" / "stagewise_report.json",
    ROOT / "build" / "v8_stagewise_imitation_ppo" / "report.json",
]
REPORT_IN = next((p for p in _REP_CANDIDATES if p.exists()), _REP_CANDIDATES[0])
OUT_DIR = ROOT / "build" / "v8_distilled_imitation_ppo"
OUT_JSON = OUT_DIR / "ppo_and_lcb_evaluation.json"

# Include retained ball 0 (indices 0,1), left ball 1 (indices 6,7), and thrown ball 2 (indices 12,13,14) at k=0
STAGE_ACTIVE_INDICES_V2 = {
    "k0": [0, 1, 6, 7, 12, 13, 14],
    "k2": [12, 13, 14, 15, 16, 17],
    "k4": [6, 7, 8, 9, 10, 11],
}


def predict_lcb_v2(table, stage_name, feat_vec, k_neighbors=3, beta=0.30, rec_margin=0.45):
    cols = STAGE_ACTIVE_INDICES_V2[stage_name]
    z_full = (np.asarray(feat_vec, dtype=np.float64) - np.asarray(table["center"])) / np.asarray(table["scale"])
    Z_full = np.asarray(table["Z"], dtype=np.float64)
    z = z_full[cols]
    Z = Z_full[:, cols]
    Y = np.asarray(table["Y"], dtype=np.float64)

    dists = np.mean((Z - z) ** 2, axis=1)
    idx = np.argsort(dists, kind="stable")[:k_neighbors]
    weights = 1.0 / (dists[idx] + 0.05)
    weights /= weights.sum()

    sub_Y = Y[idx]
    mean_u = (sub_Y * weights[:, None]).sum(axis=0)
    var_u = ((sub_Y - mean_u) ** 2 * weights[:, None]).sum(axis=0)
    std_u = np.sqrt(np.maximum(var_u, 0.0))

    scores = mean_u - beta * std_u
    if stage_name in ("k2", "k4"):
        for i in range(1, len(scores)):
            scores[i] -= rec_margin
    best_idx = int(max(range(len(scores)), key=lambda i: (round(float(scores[i]), 6), i == 0)))
    return table["candidates"][best_idx]


def evaluate_ppo_or_lcb_seed(job):
    seed, jitter_mm, mode, payload = job
    torch.set_num_threads(1)
    env = TaskResidualEnv(
        TaskConfig(throws=30, horizon_s=12.0, seat_jitter_m=jitter_mm * 1e-3),
        reference=REFERENCE,
    )
    ppo_model = None
    if mode == "ppo_bc":
        ppo_model = PPO.load(payload["checkpoint"], device="cpu")
    try:
        obs, _ = env.reset(seed=seed)
        done, k = False, 0
        while not done:
            req = env._request
            act = np.zeros(6, dtype=np.float32)
            if req is not None and action_authority(obs):
                if mode == "ppo_bc":
                    act = np.asarray(ppo_model.predict(obs, deterministic=True)[0], dtype=np.float32)
                    if req.kind == "throw" and req.released >= 3:
                        act = np.asarray(APEX_TRIM_ACT, dtype=np.float32)
                elif mode == "lcb_v2":
                    tables, cfg = payload["tables"], payload["cfg"]
                    if k == 0:
                        name = predict_lcb_v2(tables["k0"], "k0", extract_stage_features(obs, 0), **cfg)
                        act = np.asarray(K0_CANDIDATES[name], dtype=np.float32)
                    elif k == 2 and req.kind == "reception":
                        name = predict_lcb_v2(tables["k2"], "k2", extract_stage_features(obs, 2), **cfg)
                        act = np.asarray(REC_CANDIDATES[name], dtype=np.float32)
                    elif k == 4 and req.kind == "reception":
                        name = predict_lcb_v2(tables["k4"], "k4", extract_stage_features(obs, 4), **cfg)
                        act = np.asarray(REC_CANDIDATES[name], dtype=np.float32)
                    elif req.kind == "throw" and req.released >= 2:
                        act = np.asarray(APEX_TRIM_ACT, dtype=np.float32)
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
            k += 1
        return dict(
            seed=seed, jitter_mm=jitter_mm, mode=mode,
            catches=int(info["catches"]), releases=int(env.controller.throw_count),
            elapsed_s=float(info["elapsed_s"]), reason=info["reason"],
            is_success=bool(info["is_success"]),
            unsafe=bool(info["reason"] in TaskResidualEnv.UNSAFE),
        )
    finally:
        env.close()


def main():
    torch.set_num_threads(1)
    data = json.loads(REPORT_IN.read_text(encoding="utf-8"))
    tables = data["tables"]
    best_train = data["best_train_actions"]

    # 1. Collect demonstration tensors on 16 TRAIN_SEEDS for official ImitationPolicy (control_releases=3)
    print("1. Collecting training tensors on 16 TRAIN_SEEDS for ImitationPolicy (control_releases=3)...", flush=True)
    env = TaskResidualEnv(TaskConfig(throws=30, horizon_s=12.0, seat_jitter_m=JITTER_MM * 1e-3), reference=REFERENCE)
    keys = tuple(env.observation_space.spaces)
    obs_lists = {k: [] for k in keys}
    act_list, weight_list = [], []

    for s in TRAIN_SEEDS:
        acts_s = best_train[str(s)]
        obs, _ = env.reset(seed=s)
        done, k = False, 0
        ep_items = 0
        while not done and k <= 4:
            req = env._request
            act = np.zeros(6, dtype=np.float32)
            if req is not None and action_authority(obs) and req.released < 3:
                if k == 0:
                    act = np.asarray(K0_CANDIDATES[acts_s["a0"]], dtype=np.float32)
                elif k == 2 and req.kind == "reception":
                    act = np.asarray(REC_CANDIDATES[acts_s["a2"]], dtype=np.float32)
                elif k == 4 and req.kind == "reception":
                    act = np.asarray(REC_CANDIDATES[acts_s["a4"]], dtype=np.float32)
                elif req.kind == "throw" and req.released == 2:
                    act = np.asarray(APEX_TRIM_ACT, dtype=np.float32)
                for key in keys:
                    obs_lists[key].append(obs[key].copy())
                act_list.append(act.copy())
                ep_items += 1
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
            k += 1
        for _ in range(ep_items):
            weight_list.append(1.0 / max(1, ep_items))

    train_obs = {k: torch.as_tensor(np.asarray(v, dtype=np.float32)) for k, v in obs_lists.items()}
    train_act = torch.as_tensor(np.asarray(act_list, dtype=np.float32))
    train_w = torch.as_tensor(np.asarray(weight_list, dtype=np.float32))
    train_batch = (train_obs, train_act, train_w)

    center, scale = observation_statistics(train_obs, keys)
    ppo_model = initialize_model(env, seed=0, center=center, scale=scale, control_releases=3)
    fit_info = fit_actor(ppo_model, train_batch, epochs=2000)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ppo_ckpt = OUT_DIR / "ppo_official_imitation_seed0_epoch2000.zip"
    ppo_model.save(str(ppo_ckpt))
    env.close()
    print(f"  Official ImitationPolicy trained (samples={len(act_list)}, final_mse={fit_info['final_training_mse']:.6f})", flush=True)

    workers = min(6, os.cpu_count() or 1)
    lcb_cfg = dict(k_neighbors=3, beta=0.30, rec_margin=0.45)

    with ProcessPoolExecutor(workers) as pool:
        print("\n2. Evaluating on 8 VAL_SEEDS (0.25 mm)...", flush=True)
        for mode, payload in (
            ("lcb_v2", dict(tables=tables, cfg=lcb_cfg)),
            ("ppo_bc", dict(checkpoint=str(ppo_ckpt))),
        ):
            v_rows = list(pool.map(evaluate_ppo_or_lcb_seed, [(s, JITTER_MM, mode, payload) for s in VAL_SEEDS]))
            c = [r["catches"] for r in v_rows]
            print(f"  [VAL] {mode:10s}: mean_catches={np.mean(c):.2f} unsafe={sum(r['unsafe'] for r in v_rows)}/8 catches={c}", flush=True)

        print("\n3. Evaluating both policies on 10 HELD-OUT TEST_SEEDS across jitter levels...", flush=True)
        test_out = {"lcb_v2": {}, "ppo_bc": {}}
        for mode, payload in (
            ("lcb_v2", dict(tables=tables, cfg=lcb_cfg)),
            ("ppo_bc", dict(checkpoint=str(ppo_ckpt))),
        ):
            print(f"\n--- TEST SET: {mode} ---", flush=True)
            for j_mm in (0.0, 0.02, 0.10, 0.25, 0.50, 1.00, 2.00):
                t_rows = list(pool.map(evaluate_ppo_or_lcb_seed, [(s, j_mm, mode, payload) for s in TEST_SEEDS]))
                c = [r["catches"] for r in t_rows]
                u = sum(r["unsafe"] for r in t_rows)
                el = [r["elapsed_s"] for r in t_rows]
                test_out[mode][str(j_mm)] = dict(
                    jitter_mm=j_mm,
                    runs=len(t_rows),
                    successes=sum(r["is_success"] for r in t_rows),
                    mean_catches=float(np.mean(c)),
                    min_catches=int(min(c)),
                    max_catches=int(max(c)),
                    mean_elapsed_s=float(np.mean(el)),
                    unsafe=u,
                    reasons=dict(Counter(r["reason"] for r in t_rows)),
                    catches_per_seed=c,
                )
                print(f"  jitter={j_mm:5.2f}mm | succ={test_out[mode][str(j_mm)]['successes']:2d}/10 | "
                      f"mean_catches={np.mean(c):5.2f} (min={min(c)}, max={max(c)}) | "
                      f"mean_t={np.mean(el):5.2f}s | unsafe={u:2d}/10 | catches={c} | {test_out[mode][str(j_mm)]['reasons']}",
                      flush=True)

    OUT_JSON.write_text(json.dumps(dict(
        ppo_checkpoint=str(ppo_ckpt),
        ppo_fitting=fit_info,
        lcb_config=lcb_cfg,
        test_evaluation=test_out,
    ), indent=2), encoding="utf-8")
    print(f"\nSaved complete evaluation to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

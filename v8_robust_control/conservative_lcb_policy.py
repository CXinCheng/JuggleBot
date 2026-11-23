"""Conservative Lower-Confidence-Bound (LCB) Offline RL / Contextual Bandit Policy
in the 6D Active Ball-Error Subspace, with SB3 PPO (TaskPolicy) Distillation.

Reuses the 16-seed counterfactual tables from build/v8_stagewise_imitation_ppo/report.json
(TRAIN_SEEDS = 435000..435015, strictly disjoint from VAL_SEEDS = 436000..436007 and TEST_SEEDS = 440000..440009).

Key innovation over naive 58D KNN:
1. Active physical subspace projection:
   - k=0 (throw:right:0): indices [6,7,8, 12,13,14] (initial 3D position errors of ball 1 and ball 2)
   - k=2 (reception:left:2, incoming ball 2): indices [12,13,14,15,16,17] (6D pos/vel error of ball 2)
   - k=4 (reception:right:3, incoming ball 1): indices [6,7,8,9,10,11] (6D pos/vel error of ball 1)
2. Pessimistic Offline RL (LCB):
   Q_LCB(s, a) = mean_k(Y) - beta * std_k(Y) - margin * I(a != zero)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from v8_robust_control.task_env import TaskResidualEnv, TaskConfig, action_authority
from v8_robust_control.audit_task_learning import features, LOCAL_INDICES
from v8_robust_control.train_task import make_model
from v8_robust_control.train_stagewise_imitation_ppo import (
    REFERENCE, TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS, JITTER_MM,
    K0_CANDIDATES, REC_CANDIDATES, APEX_TRIM_ACT, extract_stage_features
)

_REP_CANDIDATES = [
    ROOT / "v8_robust_control" / "results" / "ablation_baselines" / "stagewise_report.json",
    ROOT / "build" / "v8_stagewise_imitation_ppo" / "report.json",
]
REPORT_IN = next((p for p in _REP_CANDIDATES if p.exists()), _REP_CANDIDATES[0])
OUT_FILE = ROOT / "v8_robust_control" / "results" / "ablation_baselines" / "lcb_report.json"

STAGE_ACTIVE_INDICES = {
    "k0": [6, 7, 8, 12, 13, 14],
    "k2": [12, 13, 14, 15, 16, 17],
    "k4": [6, 7, 8, 9, 10, 11],
}


def predict_lcb_action(table, stage_name, feat_vec, k_neighbors=4, beta=0.35, rec_margin=0.40, k0_mode="lcb"):
    if stage_name == "k0" and k0_mode != "lcb":
        return k0_mode
    cols = STAGE_ACTIVE_INDICES[stage_name]
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
        # Require non-zero reception residual to beat 'zero' by rec_margin in LCB utility
        for i in range(1, len(scores)):
            scores[i] -= rec_margin
    best_idx = int(max(range(len(scores)), key=lambda i: (round(float(scores[i]), 6), i == 0)))
    return table["candidates"][best_idx]


def evaluate_lcb_seed(job):
    seed, jitter_mm, tables, cfg = job
    env = TaskResidualEnv(
        TaskConfig(throws=30, horizon_s=12.0, seat_jitter_m=jitter_mm * 1e-3),
        reference=REFERENCE,
    )
    choices = {}
    try:
        obs, _ = env.reset(seed=seed)
        done, k = False, 0
        while not done:
            req = env._request
            act = np.zeros(6, dtype=np.float32)
            if req is not None and action_authority(obs):
                if k == 0:
                    feat = extract_stage_features(obs, 0)
                    name = predict_lcb_action(tables["k0"], "k0", feat, **cfg)
                    act = np.asarray(K0_CANDIDATES[name], dtype=np.float32)
                    choices["k0"] = name
                elif k == 2 and req.kind == "reception":
                    feat = extract_stage_features(obs, 2)
                    name = predict_lcb_action(tables["k2"], "k2", feat, **cfg)
                    act = np.asarray(REC_CANDIDATES[name], dtype=np.float32)
                    choices["k2"] = name
                elif k == 4 and req.kind == "reception":
                    feat = extract_stage_features(obs, 4)
                    name = predict_lcb_action(tables["k4"], "k4", feat, **cfg)
                    act = np.asarray(REC_CANDIDATES[name], dtype=np.float32)
                    choices["k4"] = name
                elif req.kind == "throw" and req.released >= 2 and cfg.get("use_apex_trim", True):
                    act = np.asarray(APEX_TRIM_ACT, dtype=np.float32)
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
            k += 1
        return dict(
            seed=seed, jitter_mm=jitter_mm, choices=choices,
            catches=int(info["catches"]), releases=int(env.controller.throw_count),
            elapsed_s=float(info["elapsed_s"]), reason=info["reason"],
            is_success=bool(info["is_success"]),
            unsafe=bool(info["reason"] in TaskResidualEnv.UNSAFE),
            failure=info.get("failure_event"),
        )
    finally:
        env.close()


def main():
    data = json.loads(REPORT_IN.read_text(encoding="utf-8"))
    tables = data["tables"]
    workers = min(6, os.cpu_count() or 1)

    configs = [
        ("lcb_6d_k4_b03_m03", dict(k_neighbors=4, beta=0.30, rec_margin=0.30, k0_mode="lcb")),
        ("lcb_6d_k5_b04_m05", dict(k_neighbors=5, beta=0.40, rec_margin=0.50, k0_mode="lcb")),
        ("lcb_fwdup_med_rec", dict(k_neighbors=4, beta=0.35, rec_margin=0.35, k0_mode="fwd_up_med")),
        ("lcb_fwdup_small_rec", dict(k_neighbors=4, beta=0.35, rec_margin=0.35, k0_mode="fwd_up_small")),
    ]

    with ProcessPoolExecutor(workers) as pool:
        print("=== VALIDATION SELECTION ON 8 VAL_SEEDS (0.25 mm) ===", flush=True)
        val_records = {}
        best_name, best_cfg, best_score = None, None, (-1.0, -99)
        for name, cfg in configs:
            v_rows = list(pool.map(evaluate_lcb_seed, [(s, JITTER_MM, tables, cfg) for s in VAL_SEEDS]))
            c = [r["catches"] for r in v_rows]
            u = sum(r["unsafe"] for r in v_rows)
            mean_c = float(np.mean(c))
            val_records[name] = dict(mean_catches=mean_c, unsafe=u, catches=c,
                                     choices=[r["choices"] for r in v_rows])
            print(f"  {name:22s}: val_mean={mean_c:5.2f} unsafe={u}/8 catches={c}", flush=True)
            if (mean_c, -u) > best_score:
                best_score = (mean_c, -u)
                best_name, best_cfg = name, cfg

        print(f"\nSelected on VAL_SEEDS: {best_name} (val_mean={best_score[0]:.2f})", flush=True)

        print("\n=== INDEPENDENT EVALUATION ON 10 HELD-OUT TEST_SEEDS ===", flush=True)
        test_results = {}
        for j_mm in (0.0, 0.02, 0.10, 0.25, 0.50, 1.00, 2.00):
            t_rows = list(pool.map(evaluate_lcb_seed, [(s, j_mm, tables, best_cfg) for s in TEST_SEEDS]))
            c = [r["catches"] for r in t_rows]
            u = sum(r["unsafe"] for r in t_rows)
            t_el = [r["elapsed_s"] for r in t_rows]
            test_results[str(j_mm)] = dict(
                jitter_mm=j_mm,
                runs=len(t_rows),
                successes=sum(r["is_success"] for r in t_rows),
                mean_catches=float(np.mean(c)),
                min_catches=int(min(c)),
                max_catches=int(max(c)),
                mean_elapsed_s=float(np.mean(t_el)),
                unsafe=u,
                reasons=dict(Counter(r["reason"] for r in t_rows)),
                catches_per_seed=c,
                rows=t_rows,
            )
            print(f"  jitter={j_mm:5.2f}mm | succ={test_results[str(j_mm)]['successes']:2d}/10 | "
                  f"mean_catches={np.mean(c):5.2f} (min={min(c)}, max={max(c)}) | "
                  f"mean_t={np.mean(t_el):5.2f}s | unsafe={u:2d}/10 | catches={c} | {test_results[str(j_mm)]['reasons']}",
                  flush=True)

    out_data = dict(
        selected_model=best_name,
        selected_config=best_cfg,
        validation_results=val_records,
        test_results=test_results,
    )
    OUT_FILE.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"\nSaved LCB evaluation report to {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()

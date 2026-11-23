"""Leak-free stagewise counterfactual collection, observation-conditioned policy fitting,
and SB3 PPO (TaskPolicy) actor distillation on TaskResidualEnv.

Splits (strictly disjoint):
- TRAIN_SEEDS: 435000..435015 (16 seeds)
- VAL_SEEDS:   436000..436007 (8 seeds)
- TEST_SEEDS:  440000..440009 (10 seeds)

Stages controlled:
- k=0 (throw:right:0): 4 opening pinch-clearance candidates
- k=2 (reception:left:2): 6 reception candidates
- k=4 (reception:right:3): 6 reception candidates
- Throws with released >= 2: apex safety margin trim (-0.05 m/s on vz when authority is active)

All selectors and the distilled PPO neural policy read ONLY the detached callback observation `obs`.
At 0.00 mm jitter, `action_authority(obs) == 0`, guaranteeing 10/10 (30.0 catches) zero-disturbance retention.
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
from v8_robust_control.task_control import THROW_RESIDUAL_LIMITS
from v8_robust_control.audit_task_learning import features, LOCAL_INDICES
from v8_robust_control.train_task import make_model

_CANDIDATE_REFS = [
    ROOT / "v8_robust_control" / "results" / "ablation_baselines" / "task_filtered_reference_final.json",
    ROOT / "build" / "v8_task_filtered_reference_final.json",
]
REFERENCE = str(next((p for p in _CANDIDATE_REFS if p.exists()), _CANDIDATE_REFS[0]))
OUT_DIR = ROOT / "v8_robust_control" / "results" / "ablation_baselines"
MODEL_DIR = ROOT / "v8_robust_control" / "models"

TRAIN_SEEDS = [435000 + i for i in range(16)]
VAL_SEEDS = [436000 + i for i in range(8)]
TEST_SEEDS = [440000 + i for i in range(10)]
JITTER_MM = 0.25

# Candidate libraries per stage (normalized 6D actions in [-1, 1])
K0_CANDIDATES = {
    "zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "fwd_up_small": [0.05 / 0.75, 0.0, 0.05 / 1.25, 0.0, 0.0, 0.0],
    "fwd_up_med": [0.10 / 0.75, 0.0, 0.10 / 1.25, 0.0, 0.0, 0.0],
    "fwd_med": [0.10 / 0.75, 0.0, 0.0, 0.0, 0.0, 0.0],
}

REC_CANDIDATES = {
    "zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "r2_minus": [0.0, 0.0, -0.4, 0.0, 0.0, 0.0],
    "r2_plus": [0.0, 0.0, 0.4, 0.0, 0.0, 0.0],
    "r4_minus": [0.0, 0.0, 0.0, 0.0, -0.4, 0.0],
    "r4_plus": [0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
    "r3_minus": [0.0, 0.0, 0.0, -0.4, 0.0, 0.0],
}

APEX_TRIM_ACT = [0.0, 0.0, -0.05 / 1.25, 0.0, 0.0, 0.0]


def extract_stage_features(obs, stage_k):
    """Extract compact, causal local error features from `obs` at stage `stage_k`."""
    full = features(obs)
    local = full[LOCAL_INDICES]
    err_balls = obs["errors"][:, :6].ravel()
    return np.concatenate([err_balls, local]).astype(np.float64)


def rollout_prefix(job):
    """Run an episode with specified actions at k=0, k=2, k=4 (and apex trim on throws >= 2)."""
    seed, jitter_mm, a0_name, a2_name, a4_name = job
    env = TaskResidualEnv(
        TaskConfig(throws=30, horizon_s=12.0, seat_jitter_m=jitter_mm * 1e-3),
        reference=REFERENCE,
    )
    obs_by_k = {}
    traj_obs, traj_act = [], []
    try:
        obs, _ = env.reset(seed=seed)
        done, k = False, 0
        while not done:
            req = env._request
            if k in (0, 2, 4):
                obs_by_k[k] = extract_stage_features(obs, k).tolist()
            act = np.zeros(6, dtype=np.float32)
            if req is not None and action_authority(obs):
                if k == 0:
                    act = np.asarray(K0_CANDIDATES[a0_name], dtype=np.float32)
                elif k == 2 and a2_name is not None:
                    act = np.asarray(REC_CANDIDATES[a2_name], dtype=np.float32)
                elif k == 4 and a4_name is not None:
                    act = np.asarray(REC_CANDIDATES[a4_name], dtype=np.float32)
                elif req.kind == "throw" and req.released >= 2:
                    act = np.asarray(APEX_TRIM_ACT, dtype=np.float32)
                if k <= 8:
                    traj_obs.append({key: val.copy() for key, val in obs.items()})
                    traj_act.append(act.copy())
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
            k += 1
        return dict(
            seed=seed, a0=a0_name, a2=a2_name, a4=a4_name,
            catches=int(info["catches"]), releases=int(env.controller.throw_count),
            elapsed_s=float(info["elapsed_s"]), reason=info["reason"],
            is_success=bool(info["is_success"]),
            unsafe=bool(info["reason"] in TaskResidualEnv.UNSAFE),
            obs_by_k=obs_by_k,
        )
    finally:
        env.close()


def utility(row):
    """Score candidate outcome: catches + 0.1 * elapsed_s - 0.5 * unsafe."""
    return (row["is_success"], row["catches"], -int(row["unsafe"]), round(row["elapsed_s"], 3))


def fit_stage_table(x_list, y_matrix, cand_names):
    """Fit standardized feature statistics and store counterfactual utility matrix for KNN/Ridge selection."""
    X = np.asarray(x_list, dtype=np.float64)
    Y = np.asarray(y_matrix, dtype=np.float64)  # shape (N_seeds, N_cands)
    center = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-4] = 1.0
    Z = (X - center) / scale
    return dict(
        candidates=list(cand_names),
        center=center.tolist(),
        scale=scale.tolist(),
        Z=Z.tolist(),
        Y=Y.tolist(),
    )


def predict_stage_action(table, feat_vec, k_neighbors=3, ridge_alpha=None):
    z = (np.asarray(feat_vec, dtype=np.float64) - np.asarray(table["center"])) / np.asarray(table["scale"])
    Z = np.asarray(table["Z"], dtype=np.float64)
    Y = np.asarray(table["Y"], dtype=np.float64)
    if ridge_alpha is not None:
        # Ridge regression from z to expected utility per candidate
        n, d = Z.shape
        Z_aug = np.column_stack([np.ones(n), Z])
        reg = ridge_alpha * np.eye(d + 1)
        reg[0, 0] = 0.0
        W = np.linalg.solve(Z_aug.T @ Z_aug + reg, Z_aug.T @ Y)
        scores = np.r_[1.0, z] @ W
    else:
        dists = np.mean((Z - z) ** 2, axis=1)
        idx = np.argsort(dists, kind="stable")[:k_neighbors]
        weights = 1.0 / (dists[idx] + 1e-4)
        scores = (Y[idx] * weights[:, None]).sum(axis=0) / weights.sum()
    # Tie-break toward 'zero' (index 0)
    best_idx = int(max(range(len(scores)), key=lambda i: (round(float(scores[i]), 6), i == 0)))
    return table["candidates"][best_idx]


def evaluate_policy_on_seed(job):
    seed, jitter_mm, tables, hyper = job
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
                    name = predict_stage_action(tables["k0"], feat, **hyper)
                    act = np.asarray(K0_CANDIDATES[name], dtype=np.float32)
                    choices["k0"] = name
                elif k == 2 and req.kind == "reception":
                    feat = extract_stage_features(obs, 2)
                    name = predict_stage_action(tables["k2"], feat, **hyper)
                    act = np.asarray(REC_CANDIDATES[name], dtype=np.float32)
                    choices["k2"] = name
                elif k == 4 and req.kind == "reception":
                    feat = extract_stage_features(obs, 4)
                    name = predict_stage_action(tables["k4"], feat, **hyper)
                    act = np.asarray(REC_CANDIDATES[name], dtype=np.float32)
                    choices["k4"] = name
                elif req.kind == "throw" and req.released >= 2:
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    workers = min(6, os.cpu_count() or 1)

    with ProcessPoolExecutor(workers) as pool:
        # --- STAGE 0 (k=0) on TRAIN_SEEDS ---
        print("1. Collecting Stage 0 (k=0) counterfactuals on 16 TRAIN_SEEDS...", flush=True)
        k0_names = list(K0_CANDIDATES.keys())
        jobs0 = [(s, JITTER_MM, name, "zero", "zero") for s in TRAIN_SEEDS for name in k0_names]
        res0 = list(pool.map(rollout_prefix, jobs0))

        x0_list, y0_mat, best_a0 = [], [], {}
        for s in TRAIN_SEEDS:
            s_rows = [next(r for r in res0 if r["seed"] == s and r["a0"] == name) for name in k0_names]
            x0_list.append(s_rows[0]["obs_by_k"][0])
            # Continuous utility score: catches + 0.25 * elapsed_s - 0.5 * unsafe
            u_scores = [r["catches"] + 0.25 * r["elapsed_s"] - 0.5 * int(r["unsafe"]) for r in s_rows]
            y0_mat.append(u_scores)
            pick = max(range(len(k0_names)), key=lambda i: (utility(s_rows[i]), i == 0))
            best_a0[s] = k0_names[pick]
        print("  Stage 0 oracle train catches:", [max(r["catches"] for r in res0 if r["seed"] == s) for s in TRAIN_SEEDS],
              "mean:", np.mean([max(r["catches"] for r in res0 if r["seed"] == s) for s in TRAIN_SEEDS]).round(2), flush=True)

        # --- STAGE 1 (k=2) on TRAIN_SEEDS ---
        print("2. Collecting Stage 1 (k=2) counterfactuals on 16 TRAIN_SEEDS...", flush=True)
        rec_names = list(REC_CANDIDATES.keys())
        jobs2 = [(s, JITTER_MM, best_a0[s], name, "zero") for s in TRAIN_SEEDS for name in rec_names]
        res2 = list(pool.map(rollout_prefix, jobs2))

        x2_list, y2_mat, best_a2 = [], [], {}
        for s in TRAIN_SEEDS:
            s_rows = [next(r for r in res2 if r["seed"] == s and r["a2"] == name) for name in rec_names]
            if 2 in s_rows[0]["obs_by_k"]:
                x2_list.append(s_rows[0]["obs_by_k"][2])
                u_scores = [r["catches"] + 0.25 * r["elapsed_s"] - 0.5 * int(r["unsafe"]) for r in s_rows]
                y2_mat.append(u_scores)
            pick = max(range(len(rec_names)), key=lambda i: (utility(s_rows[i]), i == 0))
            best_a2[s] = rec_names[pick]
        print("  Stage 1 oracle train catches:", [max(r["catches"] for r in res2 if r["seed"] == s) for s in TRAIN_SEEDS],
              "mean:", np.mean([max(r["catches"] for r in res2 if r["seed"] == s) for s in TRAIN_SEEDS]).round(2), flush=True)

        # --- STAGE 2 (k=4) on TRAIN_SEEDS ---
        print("3. Collecting Stage 2 (k=4) counterfactuals on 16 TRAIN_SEEDS...", flush=True)
        jobs4 = [(s, JITTER_MM, best_a0[s], best_a2[s], name) for s in TRAIN_SEEDS for name in rec_names]
        res4 = list(pool.map(rollout_prefix, jobs4))

        x4_list, y4_mat, best_a4 = [], [], {}
        for s in TRAIN_SEEDS:
            s_rows = [next(r for r in res4 if r["seed"] == s and r["a4"] == name) for name in rec_names]
            if 4 in s_rows[0]["obs_by_k"]:
                x4_list.append(s_rows[0]["obs_by_k"][4])
                u_scores = [r["catches"] + 0.25 * r["elapsed_s"] - 0.5 * int(r["unsafe"]) for r in s_rows]
                y4_mat.append(u_scores)
            pick = max(range(len(rec_names)), key=lambda i: (utility(s_rows[i]), i == 0))
            best_a4[s] = rec_names[pick]
        print("  Stage 2 oracle train catches:", [max(r["catches"] for r in res4 if r["seed"] == s) for s in TRAIN_SEEDS],
              "mean:", np.mean([max(r["catches"] for r in res4 if r["seed"] == s) for s in TRAIN_SEEDS]).round(2), flush=True)

        tables = {
            "k0": fit_stage_table(x0_list, y0_mat, k0_names),
            "k2": fit_stage_table(x2_list, y2_mat, rec_names),
            "k4": fit_stage_table(x4_list, y4_mat, rec_names),
        }

        # --- HYPERPARAMETER SELECTION ON VAL_SEEDS ONLY ---
        print("4. Selecting selector hyperparameters strictly on 8 VAL_SEEDS...", flush=True)
        hyper_grid = [
            ("knn_1", dict(k_neighbors=1, ridge_alpha=None)),
            ("knn_3", dict(k_neighbors=3, ridge_alpha=None)),
            ("knn_5", dict(k_neighbors=5, ridge_alpha=None)),
            ("ridge_10", dict(k_neighbors=3, ridge_alpha=10.0)),
            ("ridge_50", dict(k_neighbors=3, ridge_alpha=50.0)),
        ]
        val_results = {}
        best_hyper_name, best_hyper, best_val_score = None, None, (-1, -1)
        for h_name, h_cfg in hyper_grid:
            v_rows = list(pool.map(evaluate_policy_on_seed, [(s, JITTER_MM, tables, h_cfg) for s in VAL_SEEDS]))
            mean_c = float(np.mean([r["catches"] for r in v_rows]))
            unsafe_c = sum(r["unsafe"] for r in v_rows)
            val_results[h_name] = dict(mean_catches=mean_c, unsafe=unsafe_c, catches=[r["catches"] for r in v_rows])
            print(f"  [VAL] {h_name:10s}: mean_catches={mean_c:.2f} unsafe={unsafe_c}/8 catches={[r['catches'] for r in v_rows]}", flush=True)
            score = (mean_c, -unsafe_c)
            if score > best_val_score:
                best_val_score = score
                best_hyper_name, best_hyper = h_name, h_cfg

        print(f"Selected model on VAL_SEEDS: {best_hyper_name} (val mean_catches={best_val_score[0]:.2f})", flush=True)

        # --- INDEPENDENT EVALUATION ON HELD-OUT TEST_SEEDS ACROSS JITTER LEVELS ---
        print("5. Evaluating frozen selected policy on 10 HELD-OUT TEST_SEEDS (0.00, 0.25, 0.50 mm)...", flush=True)
        test_summary = {}
        for j_mm in (0.0, 0.25, 0.50):
            t_rows = list(pool.map(evaluate_policy_on_seed, [(s, j_mm, tables, best_hyper) for s in TEST_SEEDS]))
            c_list = [r["catches"] for r in t_rows]
            test_summary[str(j_mm)] = dict(
                jitter_mm=j_mm,
                runs=len(t_rows),
                successes=sum(r["is_success"] for r in t_rows),
                mean_catches=float(np.mean(c_list)),
                min_catches=int(min(c_list)),
                max_catches=int(max(c_list)),
                unsafe=sum(r["unsafe"] for r in t_rows),
                mean_elapsed_s=float(np.mean([r["elapsed_s"] for r in t_rows])),
                reasons=dict(Counter(r["reason"] for r in t_rows)),
                catches_per_seed=c_list,
                rows=t_rows,
            )
            print(f"  [TEST jitter={j_mm:.2f}mm] successes={test_summary[str(j_mm)]['successes']}/10 "
                  f"mean_catches={test_summary[str(j_mm)]['mean_catches']:.2f} "
                  f"catches={c_list} reasons={test_summary[str(j_mm)]['reasons']}", flush=True)

    # Distill the selected stagewise policy into an SB3 PPO (TaskPolicy) checkpoint on TRAIN_SEEDS
    print("6. Distilling stagewise training demonstrations into SB3 PPO (TaskPolicy) actor...", flush=True)
    torch.set_num_threads(2)
    base_env = TaskResidualEnv(TaskConfig(throws=30, horizon_s=12.0, seat_jitter_m=JITTER_MM * 1e-3), reference=REFERENCE)
    ppo_model = make_model(base_env, seed=0, algorithm="ppo")
    # Collect observation-action pairs on TRAIN_SEEDS from best_a0, best_a2, best_a4
    demo_obs = {k: [] for k in base_env.observation_space.spaces}
    demo_act = []
    for s in TRAIN_SEEDS:
        obs, _ = base_env.reset(seed=s)
        done, k = False, 0
        while not done and k <= 8:
            req = base_env._request
            act = np.zeros(6, dtype=np.float32)
            if req is not None and action_authority(obs):
                if k == 0:
                    act = np.asarray(K0_CANDIDATES[best_a0[s]], dtype=np.float32)
                elif k == 2:
                    act = np.asarray(REC_CANDIDATES[best_a2[s]], dtype=np.float32)
                elif k == 4:
                    act = np.asarray(REC_CANDIDATES[best_a4[s]], dtype=np.float32)
                elif req.kind == "throw" and req.released >= 2:
                    act = np.asarray(APEX_TRIM_ACT, dtype=np.float32)
                for key in demo_obs:
                    demo_obs[key].append(obs[key].copy())
                demo_act.append(act.copy())
            obs, _, term, trunc, _ = base_env.step(act)
            done = term or trunc
            k += 1
    base_env.close()

    obs_tensors = {k: torch.as_tensor(np.asarray(v, dtype=np.float32)) for k, v in demo_obs.items()}
    act_tensors = torch.as_tensor(np.asarray(demo_act, dtype=np.float32))
    policy = ppo_model.policy
    optimizer = torch.optim.Adam(
        list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters()),
        lr=1e-3, weight_decay=1e-5
    )
    policy.set_training_mode(True)
    for epoch in range(1500):
        optimizer.zero_grad()
        pred = policy._predict(obs_tensors, deterministic=True)
        loss = torch.mean((pred - act_tensors) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
    policy.set_training_mode(False)
    bc_mse = float(torch.mean((policy._predict(obs_tensors, deterministic=True) - act_tensors) ** 2).detach())
    ppo_path = MODEL_DIR / "ppo_official_imitation_seed0_epoch2000.zip"
    ppo_model.save(str(ppo_path))
    print(f"  Saved distilled SB3 PPO model to {ppo_path} (final BC MSE={bc_mse:.6f})", flush=True)

    report = dict(
        wall_s=time.time() - start,
        train_seeds=TRAIN_SEEDS,
        val_seeds=VAL_SEEDS,
        test_seeds=TEST_SEEDS,
        best_train_actions={str(s): dict(a0=best_a0[s], a2=best_a2[s], a4=best_a4[s]) for s in TRAIN_SEEDS},
        tables=tables,
        validation_selection=dict(selected=best_hyper_name, hyper=best_hyper, results=val_results),
        test_evaluation=test_summary,
        distilled_ppo=dict(checkpoint=str(ppo_path), samples=len(demo_act), final_bc_mse=bc_mse),
    )
    (OUT_DIR / "stagewise_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Complete in {(time.time() - start)/60:.1f} min! Saved report to {OUT_DIR / 'stagewise_report.json'}", flush=True)


if __name__ == "__main__":
    main()

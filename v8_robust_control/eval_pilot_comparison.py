"""Evaluate 4-way comparison on DEV_SEEDS under clean and perception perturbation conditions:
1. Zero-Residual Base (Frozen ClosedLoopConfig)
2. Fixed-Bias (Empirical mean opening delta sequence from TRAIN_SEEDS)
3. Biased PPO (ppo_cl_opening_iter2)
4. Error-Diff PPO (f(s, e) - f(s, 0))

Zero-leakage protocol:
- Dev seeds: 440000..440009
- 450000..450019 strictly untouched
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO

from v8_robust_control.closed_loop_bvp import ClosedLoopConfig, ClosedLoopResidualEnv
from v8_robust_control.task_env import TaskConfig

ROOT = Path(__file__).resolve().parents[1]
REF_PATH = ROOT / "v8_robust_control" / "results" / "ablation_baselines" / "task_filtered_reference_final.json"
MODEL_PATH = ROOT / "v8_robust_control" / "models" / "ppo_cl_opening_pilot.zip"
TRAIN_DELTAS_PATH = ROOT / "v8_robust_control" / "results" / "closed_loop_bvp" / "train_mean_deltas_iter2.json"

DEV_SEEDS = tuple(440000 + i for i in range(10))
PERTURBED_JITTERS_MM = (0.02, 0.10, 0.25, 0.50, 1.00, 2.00)
UNSAFE_RAW = {
    "FORBIDDEN_CONTACT", "HIT_ARM", "BALL_COLLISION", "ARM_COLLISION",
    "POSE_LIMIT", "NONFINITE_STATE", "EXTERNAL_BALL_FORCE",
}

PILOT_RESIDUAL_SCALE = 0.25
PILOT_MAX_DECISIONS = 5
PILOT_MAX_RELEASED = 3
PILOT_ACTIVE_MASK = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)


def parse_episode_outcome(info: dict) -> tuple[int, int, float, str, bool, bool]:
    fe = info.get("failure_event")
    catches = int(info.get("catches", 0))
    releases = int(info.get("releases", 0))
    elapsed_s = float(info.get("elapsed_s", 0.0))
    is_success = bool(catches == 30 and fe is None)
    reason = "SUCCESS" if is_success else (fe.get("reason") if isinstance(fe, dict) else "TIMEOUT")
    is_unsafe = reason in UNSAFE_RAW
    return catches, releases, elapsed_s, reason, is_success, is_unsafe


def _eval_worker(job: tuple[str, list[tuple[int, float]], dict]) -> list[dict]:
    mode, pairs, kwargs = job
    ref_str = str(REF_PATH)
    noise_std_m = kwargs.get("noise_std_m", 0.0)
    delay_s = kwargs.get("delay_s", 0.0)
    dropout = kwargs.get("dropout", 0.0)

    cfg = ClosedLoopConfig()
    env = ClosedLoopResidualEnv(
        TaskConfig(
            throws=30,
            horizon_s=12.0,
            seat_jitter_m=0.0,
            noise_std_m=noise_std_m,
            delay_s=delay_s,
            dropout=dropout,
        ),
        reference=ref_str,
        closed_loop_cfg=cfg,
        residual_scale=PILOT_RESIDUAL_SCALE,
        active_max_decisions=PILOT_MAX_DECISIONS,
        active_max_released=PILOT_MAX_RELEASED,
        active_mask=PILOT_ACTIVE_MASK,
        record_action_log=False,
    )

    fixed_deltas = {}
    if mode == "fixed_bias":
        raw = json.loads(TRAIN_DELTAS_PATH.read_text(encoding="utf-8"))
        for k_str, val in raw.items():
            fixed_deltas[int(k_str)] = np.array(val["requested_mean"], dtype=np.float32)

    model = None
    policy = None
    if mode in ("biased_ppo", "error_diff_ppo"):
        model = PPO.load(str(MODEL_PATH), device="cpu")
        policy = model.policy

    def get_action(obs, k_step):
        if mode == "zero_base":
            return np.zeros(6, dtype=np.float32)
        elif mode == "fixed_bias":
            return fixed_deltas.get(k_step, np.zeros(6, dtype=np.float32))
        elif mode == "biased_ppo":
            act, _ = model.predict(obs, deterministic=True)
            return act
        elif mode == "error_diff_ppo":
            obs_tensors = {k: torch.as_tensor(v[None], dtype=torch.float32) for k, v in obs.items()}
            feat = policy.extract_features(obs_tensors)
            raw_obs = policy.action_net(policy.mlp_extractor.forward_actor(feat)).detach().numpy().squeeze(0)

            obs0 = {k: v.copy() for k, v in obs.items()}
            obs0["errors"][:] = 0.0
            obs0["robot_error"][:] = 0.0
            obs0_tensors = {k: torch.as_tensor(v[None], dtype=torch.float32) for k, v in obs0.items()}
            feat0 = policy.extract_features(obs0_tensors)
            raw_zero = policy.action_net(policy.mlp_extractor.forward_actor(feat0)).detach().numpy().squeeze(0)

            return raw_obs - raw_zero
        else:
            raise ValueError(f"Unknown mode {mode}")

    rows = []
    try:
        for seed, jitter_mm in pairs:
            env.config = replace(
                env.config,
                seat_jitter_m=jitter_mm * 1e-3,
                noise_std_m=noise_std_m,
                delay_s=delay_s,
                dropout=dropout,
            )
            obs, info = env.reset(seed=seed)
            done = trunc = False
            k = 0
            while not (done or trunc):
                act = get_action(obs, k)
                obs, _, done, trunc, info = env.step(act)
                k += 1
            catches, releases, elapsed_s, reason, is_success, is_unsafe = parse_episode_outcome(info)
            rows.append(dict(
                mode=mode,
                seed=seed,
                jitter_mm=jitter_mm,
                catches=catches,
                releases=releases,
                elapsed_s=round(elapsed_s, 4),
                reason=reason,
                is_success=is_success,
                is_unsafe=is_unsafe,
            ))
    finally:
        env.close()
    return rows


def summarize_run(rows: list[dict], base_map: dict[tuple[int, float], dict], jitters: tuple[float, ...]) -> dict:
    per_jitter = {}
    pert_rows = [r for r in rows if r["jitter_mm"] > 0.0]
    for j in jitters:
        sub = [r for r in rows if abs(r["jitter_mm"] - j) < 1e-6]
        if not sub:
            continue
        reg = [r for r in sub if r["is_unsafe"] and not base_map[(r["seed"], round(j, 4))]["is_unsafe"]]
        u2s = [r for r in sub if (not r["is_unsafe"]) and base_map[(r["seed"], round(j, 4))]["is_unsafe"]]
        earlier = sum(1 for r in u2s if r["elapsed_s"] < base_map[(r["seed"], round(j, 4))]["elapsed_s"] - 0.02)
        per_jitter[f"{j:.2f}mm"] = dict(
            episodes=len(sub),
            success=sum(r["is_success"] for r in sub),
            mean_catches=round(float(np.mean([r["catches"] for r in sub])), 3),
            max_catches=max(r["catches"] for r in sub),
            mean_elapsed=round(float(np.mean([r["elapsed_s"] for r in sub])), 3),
            unsafe=sum(r["is_unsafe"] for r in sub),
            paired_reg_vs_base=len(reg),
            paired_u2s_vs_base=len(u2s),
            paired_u2s_earlier=earlier,
            catches_per_seed=[r["catches"] for r in sorted(sub, key=lambda x: x["seed"])],
        )
    all_reg = [r for r in pert_rows if r["is_unsafe"] and not base_map[(r["seed"], round(r["jitter_mm"], 4))]["is_unsafe"]]
    all_u2s = [r for r in pert_rows if (not r["is_unsafe"]) and base_map[(r["seed"], round(r["jitter_mm"], 4))]["is_unsafe"]]
    all_earlier = sum(1 for r in all_u2s if r["elapsed_s"] < base_map[(r["seed"], round(r["jitter_mm"], 4))]["elapsed_s"] - 0.02)
    return dict(
        per_jitter=per_jitter,
        perturbed_total=dict(
            episodes=len(pert_rows),
            success=sum(r["is_success"] for r in pert_rows),
            mean_catches=round(float(np.mean([r["catches"] for r in pert_rows])), 3),
            mean_elapsed=round(float(np.mean([r["elapsed_s"] for r in pert_rows])), 3),
            unsafe=sum(r["is_unsafe"] for r in pert_rows),
            paired_reg_vs_base=len(all_reg),
            paired_reg_cases=[(r["seed"], r["jitter_mm"], r["reason"]) for r in all_reg],
            paired_u2s_vs_base=len(all_u2s),
            paired_u2s_earlier=all_earlier,
        ),
    )


def run_candidate(mode: str, pairs: list[tuple[int, float]], base_map: dict, workers: int = 2, **kwargs) -> tuple[dict, list[dict]]:
    n = len(pairs)
    batches = [(mode, pairs[i::workers], kwargs) for i in range(workers)]
    with ProcessPoolExecutor(workers) as pool:
        rows = [r for batch in pool.map(_eval_worker, batches) for r in batch]
    summary = summarize_run(rows, base_map, PERTURBED_JITTERS_MM)
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval-perception", action="store_true", help="Also evaluate perception perturbation conditions")
    args = parser.parse_args()

    t0 = time.perf_counter()
    dev_base_raw = json.loads((ROOT / "v8_robust_control/results/closed_loop_bvp/heldout_7gradient.json").read_text(encoding="utf-8"))
    dev_base_map = {(int(r["seed"]), round(float(r["jitter_mm"]), 4)): r for r in dev_base_raw["episodes"]}

    dev_pairs = [(s, j) for j in PERTURBED_JITTERS_MM for s in DEV_SEEDS]
    modes = ["zero_base", "fixed_bias", "biased_ppo", "error_diff_ppo"]

    # 1. Clean Condition Evaluation
    results_clean = {}
    print(f"=== [Phase 1] 4-Way Clean Evaluation on DEV_SEEDS ({len(DEV_SEEDS)} seeds x 6 jitters = 60 eps each) ===", flush=True)
    for mode in modes:
        t_m = time.perf_counter()
        summary, rows = run_candidate(mode, dev_pairs, dev_base_map, workers=args.workers)
        results_clean[mode] = dict(summary=summary, rows=rows)
        pt = summary["perturbed_total"]
        print(
            f"  [{mode:<14}] 60-ep: mean_c={pt['mean_catches']:.3f} unsafe={pt['unsafe']:2d}/60 "
            f"mean_t={pt['mean_elapsed']:.3f}s reg={pt['paired_reg_vs_base']:2d} u2s={pt['paired_u2s_vs_base']:2d} "
            f"(earlier={pt['paired_u2s_earlier']}) wall={time.perf_counter()-t_m:.1f}s",
            flush=True,
        )

    # Print Clean Condition Table
    print("\n" + "="*95, flush=True)
    print(f"{'Jitter':<8} | {'Zero-Base':<17} | {'Fixed-Bias':<17} | {'Biased-PPO':<17} | {'Error-Diff-PPO':<17}", flush=True)
    print(f"{'':<8} | {'mean_c / unsafe':<17} | {'mean_c / unsafe':<17} | {'mean_c / unsafe':<17} | {'mean_c / unsafe':<17}", flush=True)
    print("-"*95, flush=True)
    for j in PERTURBED_JITTERS_MM:
        jk = f"{j:.2f}mm"
        z = results_clean["zero_base"]["summary"]["per_jitter"][jk]
        f = results_clean["fixed_bias"]["summary"]["per_jitter"][jk]
        b = results_clean["biased_ppo"]["summary"]["per_jitter"][jk]
        e = results_clean["error_diff_ppo"]["summary"]["per_jitter"][jk]
        print(
            f"{jk:<8} | {z['mean_catches']:5.2f} / {z['unsafe']:2d} (reg={z['paired_reg_vs_base']}) "
            f"| {f['mean_catches']:5.2f} / {f['unsafe']:2d} (reg={f['paired_reg_vs_base']}) "
            f"| {b['mean_catches']:5.2f} / {b['unsafe']:2d} (reg={b['paired_reg_vs_base']}) "
            f"| {e['mean_catches']:5.2f} / {e['unsafe']:2d} (reg={e['paired_reg_vs_base']})",
            flush=True,
        )
    print("-"*95, flush=True)
    zt = results_clean["zero_base"]["summary"]["perturbed_total"]
    ft = results_clean["fixed_bias"]["summary"]["perturbed_total"]
    bt = results_clean["biased_ppo"]["summary"]["perturbed_total"]
    et = results_clean["error_diff_ppo"]["summary"]["perturbed_total"]
    print(
        f"{'60-EP':<8} | {zt['mean_catches']:5.2f} / {zt['unsafe']:2d} (reg={zt['paired_reg_vs_base']:2d}) "
        f"| {ft['mean_catches']:5.2f} / {ft['unsafe']:2d} (reg={ft['paired_reg_vs_base']:2d}) "
        f"| {bt['mean_catches']:5.2f} / {bt['unsafe']:2d} (reg={bt['paired_reg_vs_base']:2d}) "
        f"| {et['mean_catches']:5.2f} / {et['unsafe']:2d} (reg={et['paired_reg_vs_base']:2d})",
        flush=True,
    )
    print("="*95, flush=True)

    # 2. Perception Perturbation Suite (if requested)
    perception_results = {}
    if args.eval_perception:
        cases = {
            "noise_1mm": dict(noise_std_m=0.001, delay_s=0.0, dropout=0.0),
            "delay_20ms": dict(noise_std_m=0.0, delay_s=0.020, dropout=0.0),
            "dropout_10pct": dict(noise_std_m=0.0, delay_s=0.0, dropout=0.10),
            "combined": dict(noise_std_m=0.001, delay_s=0.020, dropout=0.10),
        }
        print(f"\n=== [Phase 2] Perception Perturbation Evaluation on DEV_SEEDS (4 conditions x 60 eps each) ===", flush=True)
        for case_name, p_kwargs in cases.items():
            print(f"\n--- Condition: {case_name} ({p_kwargs}) ---", flush=True)
            perception_results[case_name] = {}
            for mode in modes:
                t_m = time.perf_counter()
                summary, rows = run_candidate(mode, dev_pairs, dev_base_map, workers=args.workers, **p_kwargs)
                perception_results[case_name][mode] = dict(summary=summary, rows=rows)
                pt = summary["perturbed_total"]
                print(
                    f"  [{mode:<14}] {case_name:<13}: mean_c={pt['mean_catches']:.3f} unsafe={pt['unsafe']:2d}/60 "
                    f"mean_t={pt['mean_elapsed']:.3f}s reg={pt['paired_reg_vs_base']:2d} u2s={pt['paired_u2s_vs_base']:2d} "
                    f"(earlier={pt['paired_u2s_earlier']}) wall={time.perf_counter()-t_m:.1f}s",
                    flush=True,
                )

    out_file = ROOT / "v8_robust_control" / "results" / "closed_loop_bvp" / "step2b_four_way_dev_comparison.json"
    out_file.write_text(json.dumps({
        "wall_s": round(time.perf_counter() - t0, 1),
        "clean_summaries": {m: results_clean[m]["summary"] for m in modes},
        "clean_episodes": {m: results_clean[m]["rows"] for m in modes},
        "perception_summaries": {
            c: {m: perception_results[c][m]["summary"] for m in modes}
            for c in perception_results
        },
    }, indent=2), encoding="utf-8")
    print(f"\nSaved comprehensive comparison to {out_file} in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

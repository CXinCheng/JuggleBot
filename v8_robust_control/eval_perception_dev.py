"""Evaluate perception perturbations (noise, delay, dropout, combined) on DEV_SEEDS.
Comparing Zero-Base, Fixed-Bias, and Biased-PPO under 4 perception conditions across 6 jitters.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
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
    mode, pairs, p_kwargs = job
    ref_str = str(REF_PATH)
    noise_std_m = p_kwargs.get("noise_std_m", 0.0)
    delay_s = p_kwargs.get("delay_s", 0.0)
    dropout = p_kwargs.get("dropout", 0.0)

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
    if mode == "biased_ppo":
        model = PPO.load(str(MODEL_PATH), device="cpu")

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
                if mode == "zero_base":
                    act = np.zeros(6, dtype=np.float32)
                elif mode == "fixed_bias":
                    act = fixed_deltas.get(k, np.zeros(6, dtype=np.float32))
                elif mode == "biased_ppo":
                    act, _ = model.predict(obs, deterministic=True)
                else:
                    raise ValueError(mode)
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


def summarize(rows: list[dict], base_map: dict) -> dict:
    per_jitter = {}
    for j in PERTURBED_JITTERS_MM:
        sub = [r for r in rows if abs(r["jitter_mm"] - j) < 1e-6]
        reg = [r for r in sub if r["is_unsafe"] and not base_map[(r["seed"], round(j, 4))]["is_unsafe"]]
        u2s = [r for r in sub if (not r["is_unsafe"]) and base_map[(r["seed"], round(j, 4))]["is_unsafe"]]
        earlier = sum(1 for r in u2s if r["elapsed_s"] < base_map[(r["seed"], round(j, 4))]["elapsed_s"] - 0.02)
        per_jitter[f"{j:.2f}mm"] = dict(
            mean_catches=round(float(np.mean([r["catches"] for r in sub])), 3),
            mean_elapsed=round(float(np.mean([r["elapsed_s"] for r in sub])), 3),
            unsafe=sum(r["is_unsafe"] for r in sub),
            reg=len(reg),
            u2s=len(u2s),
            earlier=earlier,
        )
    all_reg = [r for r in rows if r["is_unsafe"] and not base_map[(r["seed"], round(r["jitter_mm"], 4))]["is_unsafe"]]
    all_u2s = [r for r in rows if (not r["is_unsafe"]) and base_map[(r["seed"], round(r["jitter_mm"], 4))]["is_unsafe"]]
    all_earlier = sum(1 for r in all_u2s if r["elapsed_s"] < base_map[(r["seed"], round(r["jitter_mm"], 4))]["elapsed_s"] - 0.02)
    return dict(
        per_jitter=per_jitter,
        total=dict(
            episodes=len(rows),
            mean_catches=round(float(np.mean([r["catches"] for r in rows])), 3),
            mean_elapsed=round(float(np.mean([r["elapsed_s"] for r in rows])), 3),
            unsafe=sum(r["is_unsafe"] for r in rows),
            reg=len(all_reg),
            u2s=len(all_u2s),
            earlier=all_earlier,
        )
    )


def main():
    t0 = time.perf_counter()
    dev_base_raw = json.loads((ROOT / "v8_robust_control/results/closed_loop_bvp/heldout_7gradient.json").read_text(encoding="utf-8"))
    dev_base_map = {(int(r["seed"]), round(float(r["jitter_mm"]), 4)): r for r in dev_base_raw["episodes"]}

    dev_pairs = [(s, j) for j in PERTURBED_JITTERS_MM for s in DEV_SEEDS]
    workers = 4

    conditions = {
        "noise_1mm": dict(noise_std_m=0.001, delay_s=0.0, dropout=0.0),
        "delay_20ms": dict(noise_std_m=0.0, delay_s=0.020, dropout=0.0),
        "dropout_10pct": dict(noise_std_m=0.0, delay_s=0.0, dropout=0.10),
        "combined": dict(noise_std_m=0.001, delay_s=0.020, dropout=0.10),
    }
    modes = ["zero_base", "fixed_bias", "biased_ppo"]

    results = {}
    print(f"=== Running Perception Perturbation Evaluation (DEV_SEEDS, 4 conditions x 3 modes x 60 eps, workers={workers}) ===", flush=True)

    for c_name, c_kwargs in conditions.items():
        results[c_name] = {}
        print(f"\n--- Condition: {c_name} ({c_kwargs}) ---", flush=True)
        for mode in modes:
            t_m = time.perf_counter()
            batches = [(mode, dev_pairs[i::workers], c_kwargs) for i in range(workers)]
            with ProcessPoolExecutor(workers) as pool:
                rows = [r for batch in pool.map(_eval_worker, batches) for r in batch]
            summ = summarize(rows, dev_base_map)
            results[c_name][mode] = dict(summary=summ, rows=rows)
            tot = summ["total"]
            print(
                f"  [{mode:<11}] 60-ep: mean_c={tot['mean_catches']:.3f} unsafe={tot['unsafe']:2d}/60 "
                f"mean_t={tot['mean_elapsed']:.3f}s reg={tot['reg']:2d} u2s={tot['u2s']:2d} (earlier={tot['earlier']}) "
                f"wall={time.perf_counter()-t_m:.1f}s",
                flush=True,
            )

    print("\n" + "="*85, flush=True)
    print(f"{'Condition':<14} | {'Zero-Base':<20} | {'Fixed-Bias':<20} | {'Biased-PPO':<20}", flush=True)
    print(f"{'':<14} | {'mean_c / unsafe (reg)':<20} | {'mean_c / unsafe (reg)':<20} | {'mean_c / unsafe (reg)':<20}", flush=True)
    print("-"*85, flush=True)
    for c_name in conditions:
        z = results[c_name]["zero_base"]["summary"]["total"]
        f = results[c_name]["fixed_bias"]["summary"]["total"]
        b = results[c_name]["biased_ppo"]["summary"]["total"]
        print(
            f"{c_name:<14} | {z['mean_catches']:5.2f} / {z['unsafe']:2d} (reg={z['reg']:2d}) "
            f"| {f['mean_catches']:5.2f} / {f['unsafe']:2d} (reg={f['reg']:2d}) "
            f"| {b['mean_catches']:5.2f} / {b['unsafe']:2d} (reg={b['reg']:2d})",
            flush=True,
        )
    print("="*85, flush=True)

    out_file = ROOT / "v8_robust_control" / "results" / "closed_loop_bvp" / "perception_dev_comparison.json"
    out_file.write_text(json.dumps({
        "wall_s": round(time.perf_counter() - t0, 1),
        "summaries": {c: {m: results[c][m]["summary"] for m in modes} for c in conditions},
        "episodes": {c: {m: results[c][m]["rows"] for m in modes} for c in conditions},
    }, indent=2), encoding="utf-8")
    print(f"\nSaved perception report to {out_file}", flush=True)


if __name__ == "__main__":
    main()

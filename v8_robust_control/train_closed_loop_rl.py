"""Step 2B small-scale opening-increment RL pilot on the frozen ClosedLoopResidualEnv base.

Design & experimental discipline:
- Frozen rule base: ``ClosedLoopConfig()`` (``apex_trim_dvz=-0.15`` + wrist-path uncaught_braking guard).
- Bounded opening window: explicitly gated by BOTH ``active_max_decisions=5`` (0-indexed ``k < 5``)
  AND ``active_max_released=3`` (``req.released < 3``) so deferrals cannot widen the window.
- Bounded amplitude & channels: ``residual_scale=0.25``, ``active_mask=(1, 1, 1, 0, 0, 0)``
  (3D spatial/velocity opening increment only; orientation & deferral channels masked to 0).
- Zero-leakage splits:
  - ``TRAIN_SEEDS``: ``435000..435015`` (16 seeds x 6 jitter gradients)
  - ``VAL_SEEDS``:   ``436000..436007`` (8 seeds x 6 jitter gradients, safety-gated model selection)
  - ``DEV_SEEDS``:   ``440000..440009`` (10 seeds x 7 gradients, paired comparison vs zero-residual base)
  - ``450000..450019``: strictly UNTOUCHED and reserved for final frozen test.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from .closed_loop_bvp import ClosedLoopConfig, ClosedLoopResidualEnv
from .task_env import TaskConfig, TaskPolicy, TaskResidualEnv, action_authority, measured_sources
from .task_imitation import NormalizedMeasurements, observation_statistics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF = ROOT / "v8_robust_control" / "results" / "ablation_baselines" / "task_filtered_reference_final.json"
DEFAULT_MODEL_OUT = ROOT / "v8_robust_control" / "models" / "ppo_cl_opening_pilot.zip"
DEFAULT_REPORT_OUT = ROOT / "v8_robust_control" / "results" / "closed_loop_bvp" / "step2b_rl_pilot.json"

TRAIN_SEEDS = tuple(435000 + i for i in range(16))
VAL_SEEDS = tuple(436000 + i for i in range(8))
DEV_SEEDS = tuple(440000 + i for i in range(10))
PERTURBED_JITTERS_MM = (0.02, 0.10, 0.25, 0.50, 1.00, 2.00)
ALL_JITTERS_MM = (0.00,) + PERTURBED_JITTERS_MM
UNSAFE_RAW = {
    "FORBIDDEN_CONTACT", "HIT_ARM", "BALL_COLLISION", "ARM_COLLISION",
    "POSE_LIMIT", "NONFINITE_STATE", "EXTERNAL_BALL_FORCE",
}

# Bounded Step 2B opening pilot specification
PILOT_RESIDUAL_SCALE = 0.25
PILOT_MAX_DECISIONS = 5
PILOT_MAX_RELEASED = 3
PILOT_ACTIVE_MASK = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_episode_outcome(info: dict) -> tuple[int, int, float, str, bool, bool]:
    fe = info.get("failure_event")
    catches = int(info.get("catches", 0))
    releases = int(info.get("releases", 0))
    elapsed_s = float(info.get("elapsed_s", 0.0))
    is_success = bool(catches == 30 and fe is None)
    reason = "SUCCESS" if is_success else (fe.get("reason") if isinstance(fe, dict) else "TIMEOUT")
    is_unsafe = reason in UNSAFE_RAW
    return catches, releases, elapsed_s, reason, is_success, is_unsafe


def _rollout_base_batch(job: tuple[list[tuple[int, float]], str, bool]) -> list[dict]:
    pairs, ref_str, collect_obs = job
    cfg = ClosedLoopConfig()
    env = ClosedLoopResidualEnv(
        TaskConfig(throws=30, horizon_s=12.0, seat_jitter_m=0.0),
        reference=ref_str,
        closed_loop_cfg=cfg,
        residual_scale=PILOT_RESIDUAL_SCALE,
        active_max_decisions=PILOT_MAX_DECISIONS,
        active_max_released=PILOT_MAX_RELEASED,
        active_mask=PILOT_ACTIVE_MASK,
    )
    out = []
    try:
        for seed, jitter_mm in pairs:
            env.config = replace(env.config, seat_jitter_m=jitter_mm * 1e-3)
            obs_samples = []
            obs, info = env.reset(seed=seed)
            done = trunc = False
            while not (done or trunc):
                if collect_obs and env.decision_index < PILOT_MAX_DECISIONS:
                    req = env._request
                    if req is not None and req.released < PILOT_MAX_RELEASED:
                        obs_samples.append({k: v.copy() for k, v in obs.items()})
                obs, _, done, trunc, info = env.step(np.zeros(6))
            catches, releases, elapsed_s, reason, is_success, is_unsafe = parse_episode_outcome(info)
            out.append(dict(
                seed=seed,
                jitter_mm=jitter_mm,
                catches=catches,
                releases=releases,
                elapsed_s=round(elapsed_s, 4),
                reason=reason,
                is_success=is_success,
                is_unsafe=is_unsafe,
                obs_samples=obs_samples,
            ))
    finally:
        env.close()
    return out


class PairedSafeOpeningEnv(gym.Env):
    """Training wrapper over a SINGLE persistent frozen ``ClosedLoopResidualEnv``:
    - Samples ``(seed, jitter_mm)`` across ``TRAIN_SEEDS x PERTURBED_JITTERS_MM`` via ``replace(env.config, ...)``.
    - Exposes ONLY active opening steps (``k < 5`` and ``released < 3``) to PPO; once
      the opening window ends, fast-forwards the remainder under the frozen rule base
      and returns the true full-episode paired-baseline safety-constrained reward.
    """
    def __init__(
        self,
        ref_str: str,
        base_table: dict[tuple[int, float], dict],
        seeds: tuple[int, ...],
        jitters_mm: tuple[float, ...],
        worker_seed: int = 0,
    ):
        super().__init__()
        self.ref_str = ref_str
        self.base_table = base_table
        self.seeds = tuple(seeds)
        self.jitters_mm = tuple(jitters_mm)
        self.rng = np.random.default_rng(worker_seed)
        self.cl_cfg = ClosedLoopConfig()
        self._env = ClosedLoopResidualEnv(
            TaskConfig(throws=30, horizon_s=12.0, seat_jitter_m=self.jitters_mm[0] * 1e-3),
            reference=ref_str,
            closed_loop_cfg=self.cl_cfg,
            residual_scale=PILOT_RESIDUAL_SCALE,
            active_max_decisions=PILOT_MAX_DECISIONS,
            active_max_released=PILOT_MAX_RELEASED,
            active_mask=PILOT_ACTIVE_MASK,
        )
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._cur_key: tuple[int, float] | None = None
        self._delta_sq_sum: float = 0.0

    def _in_opening_window(self, env: ClosedLoopResidualEnv) -> bool:
        if env.decision_index >= PILOT_MAX_DECISIONS:
            return False
        req = env._request
        return req is not None and req.released < PILOT_MAX_RELEASED

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        s = int(self.seeds[int(self.rng.integers(0, len(self.seeds)))])
        j = float(self.jitters_mm[int(self.rng.integers(0, len(self.jitters_mm)))])
        self._env.config = replace(self._env.config, seat_jitter_m=j * 1e-3)
        self._cur_key = (s, round(j, 4))
        self._delta_sq_sum = 0.0
        obs, info = self._env.reset(seed=s)
        return obs, info

    def step(self, action):
        env = self._env
        assert self._cur_key is not None
        obs, _, done, trunc, info = env.step(action)
        d = env.last_learned_delta
        step_reg = -0.04 * float(np.dot(d[:3], d[:3]))
        self._delta_sq_sum += float(np.dot(d[:3], d[:3]))

        # Fast-forward remaining frozen-rule episode once active opening window finishes
        while not (done or trunc) and not self._in_opening_window(env):
            obs, _, done, trunc, info = env.step(np.zeros(6))

        if not (done or trunc):
            return obs, step_reg, False, False, info

        catches, releases, elapsed_s, reason, is_success, is_unsafe = parse_episode_outcome(info)
        base = self.base_table[self._cur_key]
        dc = float(catches - base["catches"])
        dt = float(elapsed_s - base["elapsed_s"])
        u_now = 1 if is_unsafe else 0
        u_base = 1 if base["is_unsafe"] else 0
        du = float(u_now - u_base)
        reg_pen = 2.0 if (u_now == 1 and u_base == 0) else 0.0
        early_drop_pen = 1.5 * max(0.0, -dt) if (u_now == 0 and u_base == 1 and dt < -0.05) else 0.0
        term_rew = 1.2 * dc + 0.4 * float(np.clip(dt, -1.5, 2.0)) - 3.0 * du - reg_pen - early_drop_pen
        return obs, step_reg + term_rew, True, False, info

    def close(self):
        self._env.close()


def _eval_ckpt_batch(job: tuple[list[tuple[int, float]], str, str, bool]) -> list[dict]:
    pairs, ref_str, ckpt_str, keep_log = job
    model = PPO.load(ckpt_str, device="cpu")
    cfg = ClosedLoopConfig()
    env = ClosedLoopResidualEnv(
        TaskConfig(throws=30, horizon_s=12.0, seat_jitter_m=0.0),
        reference=ref_str,
        closed_loop_cfg=cfg,
        residual_scale=PILOT_RESIDUAL_SCALE,
        active_max_decisions=PILOT_MAX_DECISIONS,
        active_max_released=PILOT_MAX_RELEASED,
        active_mask=PILOT_ACTIVE_MASK,
        record_action_log=keep_log,
    )
    out = []
    try:
        for seed, jitter_mm in pairs:
            env.config = replace(env.config, seat_jitter_m=jitter_mm * 1e-3)
            obs, info = env.reset(seed=seed)
            done = trunc = False
            while not (done or trunc):
                act, _ = model.predict(obs, deterministic=True)
                obs, _, done, trunc, info = env.step(act)
            log = env.action_log[:8] if keep_log else []
            max_exec = float(info.get("max_executed_action", 0.0))
            catches, releases, elapsed_s, reason, is_success, is_unsafe = parse_episode_outcome(info)
            out.append(dict(
                seed=seed,
                jitter_mm=jitter_mm,
                catches=catches,
                releases=releases,
                elapsed_s=round(elapsed_s, 4),
                reason=reason,
                is_success=is_success,
                is_unsafe=is_unsafe,
                max_executed_action=round(max_exec, 4),
                action_log=log,
            ))
    finally:
        env.close()
    return out


def _chunk_pairs(pairs: list[tuple[int, float]], n_chunks: int) -> list[list[tuple[int, float]]]:
    n = max(1, min(n_chunks, len(pairs)))
    return [pairs[i::n] for i in range(n)]


def summarize_against_base(rows: list[dict], base_map: dict[tuple[int, float], dict], jitters: tuple[float, ...]) -> dict:
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


def make_ppo_model(vec_env, center, scale, *, n_steps: int = 128, seed: int = 0) -> PPO:
    model = PPO(
        TaskPolicy,
        vec_env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.0,
        target_kl=0.02,
        seed=seed,
        device="cpu",
        verbose=0,
        policy_kwargs=dict(
            features_extractor_class=NormalizedMeasurements,
            features_extractor_kwargs=dict(center=center, scale=scale),
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            ortho_init=False,
        ),
    )
    # Zero-initialize actor output head so iteration 0 starts identically at the frozen rule base!
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.log_std.fill_(-1.4)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REF)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--iterations", type=int, default=6, help="Number of PPO iterations (256 active steps each)")
    parser.add_argument("--workers", type=int, default=2, help="Parallel worker processes (kept lean for MuJoCo RAM)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    ref_str = str(args.reference)
    work_dir = ROOT / "build" / "v8_step2b_pilot_ckpts"
    work_dir.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Precompute (or load cached) frozen zero-residual base on TRAIN_SEEDS and VAL_SEEDS
    cache_json = work_dir / "base_cache.json"
    cache_npz = work_dir / "obs_stats.npz"
    if cache_json.is_file() and cache_npz.is_file():
        cached = json.loads(cache_json.read_text(encoding="utf-8"))
        train_base_rows = cached["train_base_rows"]
        val_base_rows = cached["val_base_rows"]
        npz = np.load(cache_npz)
        center, scale = npz["center"], npz["scale"]
        print("[1/4] Loaded cached frozen zero-residual TRAIN/VAL base and obs stats.", flush=True)
    else:
        print(f"[1/4] Computing frozen zero-residual base on TRAIN ({len(TRAIN_SEEDS)}x6=96) and VAL ({len(VAL_SEEDS)}x6=48)...", flush=True)
        train_pairs = [(s, j) for j in PERTURBED_JITTERS_MM for s in TRAIN_SEEDS]
        val_pairs = [(s, j) for j in PERTURBED_JITTERS_MM for s in VAL_SEEDS]
        train_batches = [(ch, ref_str, True) for ch in _chunk_pairs(train_pairs, args.workers)]
        val_batches = [(ch, ref_str, False) for ch in _chunk_pairs(val_pairs, args.workers)]
        with ProcessPoolExecutor(args.workers) as pool:
            train_base_rows = [r for batch in pool.map(_rollout_base_batch, train_batches) for r in batch]
            val_base_rows = [r for batch in pool.map(_rollout_base_batch, val_batches) for r in batch]
        all_obs = [o for r in train_base_rows for o in r["obs_samples"]]
        keys = tuple(all_obs[0].keys())
        obs_tensors = {k: torch.as_tensor(np.asarray([o[k] for o in all_obs], dtype=np.float32)) for k in keys}
        center, scale = observation_statistics(obs_tensors, keys)
        print(f"  Collected {len(all_obs)} TRAIN opening obs for frozen NormalizedMeasurements.", flush=True)
        # Strip obs_samples before caching to disk
        train_clean = [{k: v for k, v in r.items() if k != "obs_samples"} for r in train_base_rows]
        val_clean = [{k: v for k, v in r.items() if k != "obs_samples"} for r in val_base_rows]
        cache_json.write_text(json.dumps(dict(train_base_rows=train_clean, val_base_rows=val_clean)), encoding="utf-8")
        np.savez(cache_npz, center=center, scale=scale)
        train_base_rows, val_base_rows = train_clean, val_clean
        gc.collect()

    train_base_map = {(r["seed"], round(r["jitter_mm"], 4)): r for r in train_base_rows}
    val_base_map = {(r["seed"], round(r["jitter_mm"], 4)): r for r in val_base_rows}

    val_base_unsafe = sum(r["is_unsafe"] for r in val_base_rows)
    val_base_catches = float(np.mean([r["catches"] for r in val_base_rows]))
    val_base_elapsed = float(np.mean([r["elapsed_s"] for r in val_base_rows]))
    print(f"  TRAIN base: mean_c={np.mean([r['catches'] for r in train_base_rows]):.3f}, unsafe={sum(r['is_unsafe'] for r in train_base_rows)}/96", flush=True)
    print(f"  VAL   base: mean_c={val_base_catches:.3f}, mean_t={val_base_elapsed:.3f}s, unsafe={val_base_unsafe}/48", flush=True)

    # 2. Small-scale PPO pilot training on PairedSafeOpeningEnv (lean 2 workers x 128 steps = 256 active steps/iter)
    n_envs = max(1, min(args.workers, 2))
    n_steps = 256 // n_envs
    print(f"[2/4] Training bounded opening-increment PPO ({args.iterations} iters x {n_steps*n_envs} active steps = {args.iterations*n_steps*n_envs} steps, n_envs={n_envs})...", flush=True)

    def _make_env(rank: int):
        def _init():
            return PairedSafeOpeningEnv(
                ref_str=ref_str,
                base_table=train_base_map,
                seeds=TRAIN_SEEDS,
                jitters_mm=PERTURBED_JITTERS_MM,
                worker_seed=1000 * args.seed + rank,
            )
        return _init

    vec_env = None
    model = None
    val_pairs = [(s, j) for j in PERTURBED_JITTERS_MM for s in VAL_SEEDS]
    ckpt_history = []
    best_ckpt = None
    best_val_score = (-1e9, -1e9, -1e9)

    try:
        for it in range(1, args.iterations + 1):
            t_it = time.perf_counter()
            ckpt_path = work_dir / f"ppo_cl_opening_iter{it}.zip"
            if ckpt_path.is_file():
                # Reuse already-trained checkpoint from this pilot run
                pass
            else:
                if vec_env is None:
                    vec_env = SubprocVecEnv([_make_env(i) for i in range(n_envs)])
                    prev_ckpt = work_dir / f"ppo_cl_opening_iter{it - 1}.zip"
                    if it > 1 and prev_ckpt.is_file():
                        model = PPO.load(str(prev_ckpt), env=vec_env, device="cpu")
                    else:
                        model = make_ppo_model(vec_env, center, scale, n_steps=n_steps, seed=args.seed)
                model.learn(total_timesteps=n_steps * n_envs, reset_num_timesteps=False, progress_bar=False)
                model.save(str(ckpt_path))

            # Evaluate checkpoint deterministically on VAL_SEEDS using persistent batched workers (low RAM!)
            eval_batches = [(ch, ref_str, str(ckpt_path), False) for ch in _chunk_pairs(val_pairs, n_envs)]
            with ProcessPoolExecutor(n_envs) as pool:
                val_rows = [r for batch in pool.map(_eval_ckpt_batch, eval_batches) for r in batch]
            val_sum = summarize_against_base(val_rows, val_base_map, PERTURBED_JITTERS_MM)
            vt = val_sum["perturbed_total"]
            passes_gate = (
                vt["unsafe"] <= val_base_unsafe
                and vt["mean_elapsed"] >= val_base_elapsed - 0.02
                and vt["mean_catches"] > val_base_catches
            )
            ckpt_rec = dict(
                iteration=it,
                active_steps=it * n_steps * n_envs,
                iter_wall_s=round(time.perf_counter() - t_it, 1),
                ckpt=str(ckpt_path),
                val_total=vt,
                passes_val_gate=passes_gate,
            )
            ckpt_history.append(ckpt_rec)
            print(
                f"  iter={it} ({it*n_steps*n_envs:4d} steps, {ckpt_rec['iter_wall_s']:5.1f}s) | "
                f"VAL mean_c={vt['mean_catches']:.3f} (base {val_base_catches:.3f}) "
                f"unsafe={vt['unsafe']:2d}/48 (base {val_base_unsafe:2d}/48) "
                f"mean_t={vt['mean_elapsed']:.3f}s reg={vt['paired_reg_vs_base']} "
                f"gate={'PASS' if passes_gate else 'FAIL'}",
                flush=True,
            )
            if passes_gate:
                # Jointly value catch gain AND safety reduction on VAL_SEEDS
                score = (
                    (vt["mean_catches"] - val_base_catches) + 0.15 * (val_base_unsafe - vt["unsafe"]),
                    -(vt["unsafe"] - val_base_unsafe),
                    vt["mean_elapsed"],
                )
                if score > best_val_score:
                    best_val_score = score
                    best_ckpt = ckpt_path
    finally:
        if vec_env is not None:
            vec_env.close()
        gc.collect()

    # Fallback selection if none beat base strictly on all 3 criteria: pick lowest-unsafe highest-catch for diagnostic audit
    selected_passed_gate = best_ckpt is not None
    if best_ckpt is None:
        best_rec = max(
            ckpt_history,
            key=lambda r: (-(r["val_total"]["unsafe"] - val_base_unsafe), r["val_total"]["mean_catches"]),
        )
        best_ckpt = Path(best_rec["ckpt"])
        print(f"[3/4] WARNING: No checkpoint passed strict VAL gate; archiving best diagnostic ckpt {best_ckpt.name}", flush=True)
    else:
        print(f"[3/4] Selected VAL-passing checkpoint: {best_ckpt.name}", flush=True)

    # Copy selected checkpoint to formal model path
    args.model_out.write_bytes(best_ckpt.read_bytes())

    # 4. Evaluate selected checkpoint vs EXACT same-base zero-residual control on DEV_SEEDS (440000..440009 x 7 jitters)
    print(f"[4/4] Evaluating selected checkpoint vs zero-residual base on DEV ({len(DEV_SEEDS)}x7=70 episodes)...", flush=True)
    dev_base_raw = json.loads((ROOT / "v8_robust_control/results/closed_loop_bvp/heldout_7gradient.json").read_text(encoding="utf-8"))
    dev_base_map = {(int(r["seed"]), round(float(r["jitter_mm"]), 4)): r for r in dev_base_raw["episodes"]}

    dev_pairs = [(s, j) for j in ALL_JITTERS_MM for s in DEV_SEEDS]
    dev_batches = [(ch, ref_str, str(args.model_out), True) for ch in _chunk_pairs(dev_pairs, args.workers)]
    with ProcessPoolExecutor(args.workers) as pool:
        dev_rows = [r for batch in pool.map(_eval_ckpt_batch, dev_batches) for r in batch]
    dev_summary = summarize_against_base(dev_rows, dev_base_map, ALL_JITTERS_MM)

    print("\n=== Step 2B DEV Comparison (440000..440009): Frozen Rule Base + Learned Opening Increment vs Zero-Residual Base ===", flush=True)
    for j in ALL_JITTERS_MM:
        jk = f"{j:.2f}mm"
        s_rl = dev_summary["per_jitter"][jk]
        s_bl = dev_base_raw["summary"][jk]
        print(
            f"  {jk:<7} | RL+CL: succ={s_rl['success']:2d}/10 mean_c={s_rl['mean_catches']:5.2f} unsafe={s_rl['unsafe']:2d}/10 mean_t={s_rl['mean_elapsed']:5.3f}s "
            f"(reg={s_rl['paired_reg_vs_base']}, u2s={s_rl['paired_u2s_vs_base']}[ear={s_rl['paired_u2s_earlier']}]) "
            f"|| Zero-Base: mean_c={s_bl['mean_catches']:5.2f} unsafe={s_bl['unsafe_count']:2d}/10",
            flush=True,
        )
    dt = dev_summary["perturbed_total"]
    base_pert = [r for r in dev_base_raw["episodes"] if r["jitter_mm"] > 0.0]
    print(
        f"  60-EP   | RL+CL: mean_c={dt['mean_catches']:.3f} unsafe={dt['unsafe']}/60 mean_t={dt['mean_elapsed']:.3f}s "
        f"reg={dt['paired_reg_vs_base']} u2s={dt['paired_u2s_vs_base']}(ear={dt['paired_u2s_earlier']}) "
        f"|| Zero-Base: mean_c={np.mean([r['catches'] for r in base_pert]):.3f} unsafe={sum(r['is_unsafe'] for r in base_pert)}/60 "
        f"mean_t={np.mean([r['elapsed_s'] for r in base_pert]):.3f}s",
        flush=True,
    )

    report = dict(
        schema_version=1,
        stage="Step 2B small-scale opening-increment RL pilot (DEV comparison only; 450000..450019 untouched)",
        pilot_spec=dict(
            frozen_closed_loop_cfg=asdict(ClosedLoopConfig()),
            residual_scale=PILOT_RESIDUAL_SCALE,
            active_max_decisions=PILOT_MAX_DECISIONS,
            active_max_released=PILOT_MAX_RELEASED,
            active_mask=list(PILOT_ACTIVE_MASK),
            window_note="Explicitly gated by BOTH decision_index < 5 (first 5 callbacks) AND req.released < 3.",
            train_seeds=list(TRAIN_SEEDS),
            val_seeds=list(VAL_SEEDS),
            dev_seeds=list(DEV_SEEDS),
            iterations=args.iterations,
            active_steps_per_iter=n_steps * n_envs,
            seed=args.seed,
        ),
        digests=dict(
            model_sha256=sha256_file(args.model_out),
            reference_sha256=sha256_file(args.reference),
            closed_loop_bvp_sha256=sha256_file(ROOT / "v8_robust_control" / "closed_loop_bvp.py"),
            trainer_sha256=sha256_file(Path(__file__)),
            measured_sources=measured_sources(),
        ),
        wall_s=round(time.perf_counter() - t0, 1),
        val_base=dict(mean_catches=round(val_base_catches, 3), mean_elapsed=round(val_base_elapsed, 3), unsafe=val_base_unsafe),
        selected_passed_val_gate=selected_passed_gate,
        selected_source_ckpt=str(best_ckpt),
        model_path=str(args.model_out.relative_to(ROOT).as_posix()),
        val_history=ckpt_history,
        dev_summary=dev_summary,
        dev_episodes=dev_rows,
    )
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved Step 2B pilot report to {args.report_out} in {report['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()

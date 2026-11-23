"""Four-arm diagnostic ablation on seeds 440000..440002 under 1mm noise and 0.00mm initial jitter:
Arm 1 (Rule-Action ON, Underlying-Intervention ON): baseline frozen closed-loop.
Arm 2 (Rule-Action OFF, Underlying-Intervention ON): rule policy yields zeros, but controller.perturbed_active is True (CBF, tight posture, thumb clamp active).
Arm 3 (Rule-Action ON, Underlying-Intervention OFF): rule policy yields action, but controller.perturbed_active is forced False.
Arm 4 (Rule-Action OFF, Underlying-Intervention OFF): pure nominal trajectory, both rule actions and underlying interventions disabled.
Tracks:
  - First command divergence (time, k, kind, commanded action)
  - Actual physical release velocities (especially ball 0 at t~1.1s)
  - First failure mode & time
  - Complete 30-throw outcome
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from v8_robust_control.closed_loop_bvp import ClosedLoopConfig, ClosedLoopResidualEnv
from v8_robust_control.task_env import TaskConfig

ROOT = Path(__file__).resolve().parents[1]
REF_PATH = ROOT / "v8_robust_control" / "results" / "ablation_baselines" / "task_filtered_reference_final.json"
DEV_SEEDS = (440000, 440001, 440002)

UNSAFE_RAW = {
    "FORBIDDEN_CONTACT", "HIT_ARM", "BALL_COLLISION", "ARM_COLLISION",
    "POSE_LIMIT", "NONFINITE_STATE", "EXTERNAL_BALL_FORCE",
}


def parse_episode_outcome(info: dict) -> tuple[int, int, float, str, bool, bool]:
    fe = info.get("failure_event")
    catches = int(info.get("catches", 0))
    releases = int(info.get("releases", 0))
    elapsed_s = float(info.get("elapsed_s", 0.0))
    is_success = bool(catches == 30 and fe is None)
    reason = "SUCCESS" if is_success else (fe.get("reason") if isinstance(fe, dict) else "TIMEOUT")
    is_unsafe = reason in UNSAFE_RAW
    return catches, releases, elapsed_s, reason, is_success, is_unsafe


def evaluate_trial(seed: int, rule_action_on: bool, underlying_on: bool):
    ref_str = str(REF_PATH)
    cfg = ClosedLoopConfig()
    env = ClosedLoopResidualEnv(
        TaskConfig(
            throws=30,
            horizon_s=12.0,
            seat_jitter_m=0.0,
            noise_std_m=0.001,
            delay_s=0.0,
            dropout=0.0,
        ),
        reference=ref_str,
        closed_loop_cfg=cfg,
        residual_scale=0.0,
        record_action_log=True,
    )

    first_divergence = None
    try:
        if not underlying_on:
            env._sync_perturbation_gate = lambda obs: None

        obs, info = env.reset(seed=seed)
        ctrl = env.controller

        if not underlying_on and ctrl is not None:
            ctrl.perturbed_active = False

        if not rule_action_on:
            env.rule_policy.predict = lambda obs, e: np.zeros(6, dtype=np.float32)

        done = trunc = False
        k = 0
        while not (done or trunc):
            if not underlying_on and ctrl is not None:
                ctrl.perturbed_active = False

            # Check action that will be executed
            rule_act = env.rule_policy.predict(env._observation, env)

            # Record first divergence from zero action
            if first_divergence is None and np.any(np.abs(rule_act) > 1e-4):
                req = env._request
                first_divergence = dict(
                    k=k,
                    time_s=round(float(req.time_s), 4) if req else None,
                    kind=req.kind if req else None,
                    side=req.side if req else None,
                    released=int(req.released) if req else None,
                    cmd=[round(float(x), 4) for x in rule_act],
                )

            obs, _, done, trunc, info = env.step(np.zeros(6, dtype=np.float32))
            k += 1

            if not underlying_on and ctrl is not None:
                ctrl.perturbed_active = False

        catches, releases, elapsed_s, reason, is_success, is_unsafe = parse_episode_outcome(info)
        fe = info.get("failure_event")

        # Collect physical release events
        physical_releases = []
        if ctrl is not None and hasattr(ctrl, "events"):
            for e in ctrl.events:
                if e.get("kind") == "release":
                    physical_releases.append(dict(
                        time=round(float(e["time"]), 4),
                        side=e["side"],
                        ball=int(e["ball"]),
                        velocity=[round(float(x), 4) for x in e["velocity"]],
                    ))

    finally:
        env.close()

    return dict(
        seed=seed,
        rule_action_on=rule_action_on,
        underlying_on=underlying_on,
        catches=catches,
        releases=releases,
        elapsed_s=round(elapsed_s, 4),
        reason=reason,
        is_success=is_success,
        is_unsafe=is_unsafe,
        failure_event=fe,
        first_divergence=first_divergence,
        physical_releases=physical_releases,
    )


def main():
    t0 = time.perf_counter()
    arms = [
        ("Arm 1: Rule ON / Underlying ON (Full Base)", True, True),
        ("Arm 2: Rule OFF / Underlying ON (Underlying Only)", False, True),
        ("Arm 3: Rule ON / Underlying OFF (Rule Action Only)", True, False),
        ("Arm 4: Rule OFF / Underlying OFF (Pure Nominal)", False, False),
    ]

    print("=== Four-Arm Diagnostic Isolation on Seeds 440000..440002 (1mm Noise, 0.00mm Jitter) ===\n", flush=True)

    results = {}
    for arm_name, r_on, u_on in arms:
        print(f"--- Running {arm_name} ---", flush=True)
        results[arm_name] = []
        for s in DEV_SEEDS:
            res = evaluate_trial(s, r_on, u_on)
            results[arm_name].append(res)
            print(
                f"  Seed {s}: {res['reason']:<12} catches={res['catches']:2d}/30 releases={res['releases']:2d} "
                f"elapsed={res['elapsed_s']:6.3f}s (div={res['first_divergence'] is not None})",
                flush=True,
            )
        print("", flush=True)

    # Comparison Table
    print("=" * 105)
    print(f"{'Seed':<8} | {'Arm 1 (Rule+Under)':<22} | {'Arm 2 (Underlying Only)':<22} | {'Arm 3 (Rule Only)':<22} | {'Arm 4 (Pure Nominal)':<22}")
    print("-" * 105)
    for i, s in enumerate(DEV_SEEDS):
        a1 = results[arms[0][0]][i]
        a2 = results[arms[1][0]][i]
        a3 = results[arms[2][0]][i]
        a4 = results[arms[3][0]][i]
        print(
            f"{s:<8} | {a1['reason']} (c={a1['catches']:2d}, {a1['elapsed_s']:5.2f}s) "
            f"| {a2['reason']} (c={a2['catches']:2d}, {a2['elapsed_s']:5.2f}s) "
            f"| {a3['reason']} (c={a3['catches']:2d}, {a3['elapsed_s']:5.2f}s) "
            f"| {a4['reason']} (c={a4['catches']:2d}, {a4['elapsed_s']:5.2f}s)"
        )
    print("=" * 105)

    # Check Ball 0 release velocities across arms
    print("\n=== Physical Ball 0 Release Velocity Comparison (Released at t~1.1s) ===")
    for i, s in enumerate(DEV_SEEDS):
        print(f"\nSeed {s}:")
        for arm_name, _, _ in arms:
            res = results[arm_name][i]
            b0_rel = next((r for r in res["physical_releases"] if r["ball"] == 0), None)
            if b0_rel:
                vz = b0_rel["velocity"][2]
                print(f"  {arm_name:<48} : t={b0_rel['time']}s, vel={b0_rel['velocity']} (vz={vz:.3f})")
            else:
                print(f"  {arm_name:<48} : [No Ball 0 Release]")

    out_file = ROOT / "v8_robust_control" / "results" / "closed_loop_bvp" / "four_arm_noise_isolation.json"
    out_file.write_text(json.dumps(dict(
        wall_s=round(time.perf_counter() - t0, 1),
        seeds=list(DEV_SEEDS),
        arms=arms,
        results=results,
    ), indent=2), encoding="utf-8")
    print(f"\nSaved four-arm isolation report to {out_file}", flush=True)


if __name__ == "__main__":
    main()

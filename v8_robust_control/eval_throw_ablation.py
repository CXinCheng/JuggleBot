"""Evaluation & diagnostic script for throw velocity correction ablation under 1mm noise.

Ablation Arms (all with Underlying Closed-Loop Intervention ON):
1. Arm 2 (Zero Throw Correction):
   Baseline with zero rule actions; underlying guards active.
2. Arm 2 + Vertical-Only Limited (vertical correction clamped only):
   Throw horizontal corrections zeroed (act[0]=act[1]=0);
   Vertical correction clamped to [-0.20, 0.20] (max |dv_z| <= 0.05 m/s, ~30% of full trim).
3. Arm 2 + 3-Axis Limited (all 3 velocity axes clamped):
   Throw 3D velocity corrections clamped to [-0.20, 0.20] (max |dv| <= 0.05 m/s across x, y, z).
4. Arm 1 (Full Rule Throw):
   Original frozen trim_m15_wguard throw corrections.

Seeds: 440000..440002 (DEV SEEDS ONLY).
Unseen seeds 450000..450019 remain strictly untouched.
"""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path
import numpy as np

if sys.stdout is not None:
    sys.stdout.reconfigure(line_buffering=True)

from v8_robust_control.task_env import TaskConfig, action_authority
from v8_robust_control.closed_loop_bvp import ClosedLoopResidualEnv, ClosedLoopConfig, ClosedLoopBvpPolicy
from v8_robust_control.task_control import decode_action

DEV_SEEDS = [440000, 440001, 440002]
REF_PATH = Path("v8_robust_control/results/ablation_baselines/task_filtered_reference_final.json")
OUT_JSON = Path("v8_robust_control/results/closed_loop_bvp/throw_ablation_results.json")

UNSAFE_RAW = {"BALL_COLLISION", "HIT_ARM", "ARM_COLLISION", "POSE_LIMIT", "PALM_LIMIT", "TORQUE_LIMIT"}


class AblatedThrowPolicy:
    def __init__(self, arm_mode: str, cfg: ClosedLoopConfig):
        self.arm_mode = arm_mode
        self.base_policy = ClosedLoopBvpPolicy(cfg)

    def predict(self, obs: dict, env: ClosedLoopResidualEnv) -> np.ndarray:
        req = env._request
        if req is None or req.kind != "throw":
            return np.zeros(6, dtype=np.float32)

        if self.arm_mode == "zero":
            return np.zeros(6, dtype=np.float32)

        act = self.base_policy.predict(obs, env).copy()

        if self.arm_mode == "vertical_only_clamped":
            # Zero horizontal channels, clamp vertical to [-0.20, 0.20]
            act[0] = 0.0
            act[1] = 0.0
            act[2] = float(np.clip(act[2], -0.20, 0.20))
            act[3:] = 0.0
            return act

        elif self.arm_mode == "three_axis_clamped":
            # Clamp 3D velocity channels to [-0.20, 0.20], zero bank/lead/defer
            act[0] = float(np.clip(act[0], -0.20, 0.20))
            act[1] = float(np.clip(act[1], -0.20, 0.20))
            act[2] = float(np.clip(act[2], -0.20, 0.20))
            act[3:] = 0.0
            return act

        elif self.arm_mode == "full_rule":
            return act

        else:
            raise ValueError(f"Unknown arm_mode: {self.arm_mode}")


def run_trial(seed: int, arm_mode: str):
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

    env.rule_policy = AblatedThrowPolicy(arm_mode, cfg)

    # Tracking records
    throw_decisions = []
    reception_plans = []

    try:
        obs, info = env.reset(seed=seed)
        ctrl = env.controller

        # Hook into plan_reception to log predicted targets
        orig_plan_reception = ctrl.plan_reception
        def intercepted_plan_reception(side):
            res = orig_plan_reception(side)
            hand = ctrl.hands[side]
            rec = hand.get("reception")
            if rec is not None:
                motion, stop = rec
                reception_plans.append(dict(
                    time_s=round(float(ctrl.data.time), 4),
                    side=side,
                    ball_id=int(hand.get("incoming", -1)),
                    predicted_impact_time_s=round(float(stop.t0), 4),
                    predicted_target_pos_m=[round(float(x), 4) for x in stop.start],
                    target_velocity_m_s=[round(float(x), 4) for x in stop.velocity],
                ))
            return res
        ctrl.plan_reception = intercepted_plan_reception

        done = trunc = False
        k = 0
        while not (done or trunc):
            req = env._request
            rule_act = env.rule_policy.predict(env._observation, env)
            auth = float(action_authority(env._observation))

            if req is not None and req.kind == "throw":
                decoded_cmd = decode_action(req, rule_act * auth)
                throw_decisions.append(dict(
                    k=k,
                    time_s=round(float(req.time_s), 4),
                    side=req.side,
                    ball_id=int(req.ball_id),
                    released=int(req.released),
                    authority=round(auth, 4),
                    raw_action=[round(float(x), 4) for x in rule_act],
                    executed_action=[round(float(x), 4) for x in rule_act * auth],
                    nominal_parameters=[round(float(x), 4) for x in req.nominal_parameters],
                    decoded_parameters=[round(float(x), 4) for x in decoded_cmd.parameters],
                    defer_s=round(float(decoded_cmd.defer_s), 4),
                ))

            obs, _, done, trunc, info = env.step(np.zeros(6, dtype=np.float32))
            k += 1

        fe = info.get("failure_event")
        catches = int(info.get("catches", 0))
        releases = int(info.get("releases", 0))
        elapsed_s = float(info.get("elapsed_s", 0.0))
        is_success = bool(catches == 30 and fe is None)
        reason = "SUCCESS" if is_success else (fe.get("reason") if isinstance(fe, dict) else "TIMEOUT")
        is_unsafe = reason in UNSAFE_RAW

        # Physical releases
        physical_releases = []
        touch_events = []
        if ctrl is not None and hasattr(ctrl, "events"):
            for e in ctrl.events:
                if e.get("kind") == "release":
                    physical_releases.append(dict(
                        time=round(float(e["time"]), 4),
                        side=e["side"],
                        ball=int(e["ball"]),
                        velocity=[round(float(x), 4) for x in e["velocity"]],
                    ))
                elif e.get("kind") == "touch":
                    touch_events.append(dict(
                        time=round(float(e["time"]), 4),
                        side=e["side"],
                        ball=int(e["ball"]),
                        contact_distance=round(float(e.get("contact_distance", 0.0)), 6),
                        palm_side=round(float(e.get("palm_side", 0.0)), 6),
                        tangent_offset=round(float(e.get("tangent_offset", 0.0)), 6),
                    ))

        # Compare predicted reception vs touch event
        reception_tracking = []
        for plan in reception_plans:
            matching_touch = None
            for t_ev in touch_events:
                if t_ev["side"] == plan["side"] and t_ev["ball"] == plan["ball_id"]:
                    if abs(t_ev["time"] - plan["predicted_impact_time_s"]) < 0.15:
                        matching_touch = t_ev
                        break
            reception_tracking.append(dict(
                side=plan["side"],
                ball_id=plan["ball_id"],
                plan_time_s=plan["time_s"],
                predicted_time_s=plan["predicted_impact_time_s"],
                predicted_pos=plan["predicted_target_pos_m"],
                touch_time_s=matching_touch["time"] if matching_touch else None,
                timing_error_s=round(float(matching_touch["time"] - plan["predicted_impact_time_s"]), 4) if matching_touch else None,
                contact_distance=matching_touch["contact_distance"] if matching_touch else None,
                touched=bool(matching_touch is not None),
            ))

    finally:
        env.close()

    return dict(
        seed=seed,
        arm_mode=arm_mode,
        catches=catches,
        releases=releases,
        elapsed_s=round(elapsed_s, 4),
        reason=reason,
        is_success=is_success,
        is_unsafe=is_unsafe,
        failure_event=fe,
        throw_decisions=throw_decisions,
        physical_releases=physical_releases,
        reception_tracking=reception_tracking,
    )


def main():
    t0 = time.perf_counter()
    arms = [
        ("Arm 2 (Zero Throw Correction)", "zero"),
        ("Arm 2 + Vertical-Only Clamped", "vertical_only_clamped"),
        ("Arm 2 + 3-Axis Clamped", "three_axis_clamped"),
        ("Arm 1 (Full Rule Throw)", "full_rule"),
    ]

    all_results = {}
    print("=== Throw Velocity Correction Ablation on Seeds 440000..440002 (1mm Noise, 0.00mm Jitter) ===\n")

    for arm_label, arm_mode in arms:
        print(f"--- Running {arm_label} ---")
        records = []
        for seed in DEV_SEEDS:
            rec = run_trial(seed, arm_mode)
            records.append(rec)
            status_str = f"{rec['reason']:<15} catches={rec['catches']:>2}/30 releases={rec['releases']:>2} elapsed={rec['elapsed_s']:>6.3f}s"
            print(f"  Seed {seed}: {status_str}")
        all_results[arm_label] = records
        print()

    # Summary table
    print("=" * 115)
    header = f"{'Seed':<8} | {'Arm 2 (Zero Throw)':<25} | {'Arm 2 + Vert Clamped':<25} | {'Arm 2 + 3-Axis Clamped':<25} | {'Arm 1 (Full Rule)':<25}"
    print(header)
    print("-" * 115)

    for i, s in enumerate(DEV_SEEDS):
        r_zero = all_results["Arm 2 (Zero Throw Correction)"][i]
        r_vert = all_results["Arm 2 + Vertical-Only Clamped"][i]
        r_3ax = all_results["Arm 2 + 3-Axis Clamped"][i]
        r_full = all_results["Arm 1 (Full Rule Throw)"][i]

        def fmt_cell(r):
            return f"{r['reason']} (c={r['catches']}, {r['elapsed_s']:.2f}s)"

        print(f"{s:<8} | {fmt_cell(r_zero):<25} | {fmt_cell(r_vert):<25} | {fmt_cell(r_3ax):<25} | {fmt_cell(r_full):<25}")

    print("=" * 115)

    # Save to JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        wall_s=round(time.perf_counter() - t0, 1),
        seeds=DEV_SEEDS,
        arms=[a[0] for a in arms],
        results=all_results,
    )
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved throw ablation report to {OUT_JSON.resolve()}")


if __name__ == "__main__":
    main()

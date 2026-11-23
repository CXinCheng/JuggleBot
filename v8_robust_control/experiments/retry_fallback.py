"""Independent online 30-throw + 0.8 s three-ball hold A/B pairing:
  - Group A (a0_native): native retry_throw_command (selector disabled).
  - Group B (b1_fixed_analytic): fixed zero-probe analytic vx_cmd retry fallback
    ONLY on the first retry after a rejected native first search when trial-1
    is feasible with finite |err_x0| > 0.120 m:
      dvx = clip(-err_x0 / flight, -0.75, +0.75) m/s applied to archived_warm.
    Otherwise falls back to native retry_throw_command.

Features:
  - Full online t=0 planning & control (no cached decision injection).
  - 2-way concurrency with per-branch live logs, per-branch live event JSONL,
    immediate per-branch JSON persistence on completion or 3600 s wall timeout
    (PENDING), and resumable skipping of already-completed matching-config branches.
  - Does NOT modify src/, v8_robust_control/, prior batches, report/main.tex,
    Git staging, or sealed seeds 450000..450019.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "report" / "evidence"
CFG_PATH = EVIDENCE_DIR / "retry_only_independent_config.json"
ARCHIVED_RESULTS = EVIDENCE_DIR / "retry_only_independent_results.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "v8_retry_fallback_reproduction"


def compute_config_hash(cfg: dict) -> str:
    payload = json.dumps(
        {
            "shared_frozen_params": cfg["shared_frozen_params"],
            "groups": cfg["groups"],
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def make_fixed_analytic_selector(err_x0_thresh_m: float):
    def _selector(obs: dict) -> int:
        err_x0 = obs.get("trial1_err_x0_m")
        if (
            obs.get("trial1_feasible")
            and err_x0 is not None
            and np.isfinite(err_x0)
            and abs(float(err_x0)) > float(err_x0_thresh_m)
        ):
            return 1
        return 0

    return _selector


def verify_interface_smoke(cfg: dict):
    """Verify fixed threshold rule logic and confirm prior C5/S1 smoke records."""
    sel = make_fixed_analytic_selector(cfg["groups"]["b1_fixed_analytic"]["err_x0_threshold_m"])
    assert sel({"trial1_feasible": True, "trial1_err_x0_m": 0.121}) == 1
    assert sel({"trial1_feasible": True, "trial1_err_x0_m": -0.150}) == 1
    assert sel({"trial1_feasible": True, "trial1_err_x0_m": 0.120}) == 0
    assert sel({"trial1_feasible": True, "trial1_err_x0_m": -0.050}) == 0
    assert sel({"trial1_feasible": False, "trial1_err_x0_m": 0.300}) == 0
    assert sel({"trial1_feasible": True, "trial1_err_x0_m": None}) == 0
    assert sel({"trial1_feasible": True, "trial1_err_x0_m": float("nan")}) == 0

    smoke_log = EVIDENCE_DIR / "retry_only_interface_smoke.txt"
    if not smoke_log.exists():
        raise FileNotFoundError(f"Expected prior smoke log missing: {smoke_log}")
    lines = smoke_log.read_text(encoding="utf-8", errors="replace").splitlines()
    c5_ok = any(l.startswith("[SMOKE C5]") and "'selector_calls': 0" in l and "'pass': True" in l for l in lines)
    s1_ok = any(l.startswith("[SMOKE S1]") and "'rel12_dp_mm': 0.0" in l and "'pass': True" in l for l in lines)
    assert c5_ok, "Prior SMOKE C5 PASS not found in the archived smoke extract"
    assert s1_ok, "Prior SMOKE S1 PASS not found in the archived smoke extract"
    print("[INTERFACE SMOKE OK] fixed threshold unit checks PASS; prior SMOKE C5 & SMOKE S1 verified.", flush=True)


def run_worker(seed: int, group: str, output_dir: Path):
    from jugglebot.robustness.perception import PerceptionConfig
    from v8_robust_control.complete_task import CompleteCascade
    from .postsep_brake import _PostSepBrakeWrapper, eval_incoming_state
    from .retry_selector_interface import RetrySelectorWrapper

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    cfg_hash = compute_config_hash(cfg)
    sp = cfg["shared_frozen_params"]
    gcfg = cfg["groups"][group]
    psb = sp["post_sep_braking"]

    branch_dir = output_dir / "branches"
    event_dir = output_dir / "events"
    branch_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    events_path = event_dir / f"seed_{seed}_{group}_events.jsonl"
    if events_path.exists():
        events_path.unlink()
    out_file = branch_dir / f"seed_{seed}_{group}.json"

    t0 = time.perf_counter()
    controller = CompleteCascade(
        throws=int(sp["throws_target"]),
        tau=float(sp["tau_s"]),
        flight=float(sp["flight_s"]),
        obs_noise=0.0,
        seed=int(seed),
        observation_mode=str(sp["observation_mode"]),
        perception=PerceptionConfig(noise_std_m=float(sp["noise_std_m"]), seed=int(seed)),
        ball_estimator=str(sp["ball_estimator"]),
    )
    controller.T_START_S = float(sp["ending_t_start_s"])

    if gcfg["enable_analytic_retry_fallback"]:
        selector_fn = make_fixed_analytic_selector(float(gcfg["err_x0_threshold_m"]))
    else:
        selector_fn = None

    wrapper = RetrySelectorWrapper(controller, selector_fn=selector_fn)

    orig_step = controller.step
    flushed = 0
    seen_sep_keys = set()
    active_triggers = {}
    triggers = []
    min_interball = {"dist_m": 10.0, "time_s": None, "pair": None}
    rh_elbow_id = controller.model.body("right_elbow_link").id
    rh_sh_id = controller.model.body("right_shoulder_pitch_link").id
    lh_elbow_id = controller.model.body("left_elbow_link").id
    lh_sh_id = controller.model.body("left_shoulder_pitch_link").id
    max_rh_elbow_ending = {"elbow_out_m": 0.0, "time_s": None}
    max_rh_elbow_all = {"elbow_out_m": 0.0, "time_s": None}
    max_lh_elbow_all = {"elbow_out_m": 0.0, "time_s": None}

    dt_imp_max = float(psb["dt_to_impact_max_s"])
    v_away_min = float(psb["v_away_min_m_s"])
    t_post_brake = float(psb["post_sep_brake_duration_s"])
    coast_ratio = float(psb["brake_coast_ratio"])

    def hooked_step():
        nonlocal flushed
        res = orig_step()
        if not controller.silent:
            d = controller.data
            rh_eo = float(abs(d.xpos[rh_elbow_id, 1]) - abs(d.xpos[rh_sh_id, 1]))
            lh_eo = float(abs(d.xpos[lh_elbow_id, 1]) - abs(d.xpos[lh_sh_id, 1]))
            if rh_eo > max_rh_elbow_all["elbow_out_m"]:
                max_rh_elbow_all["elbow_out_m"] = rh_eo
                max_rh_elbow_all["time_s"] = float(d.time)
            if lh_eo > max_lh_elbow_all["elbow_out_m"]:
                max_lh_elbow_all["elbow_out_m"] = lh_eo
                max_lh_elbow_all["time_s"] = float(d.time)
            if controller.end is not None and rh_eo > max_rh_elbow_ending["elbow_out_m"]:
                max_rh_elbow_ending["elbow_out_m"] = rh_eo
                max_rh_elbow_ending["time_s"] = float(d.time)

            if controller.throw_count >= 1:
                ps = [d.qpos[controller.balls[b][0]:controller.balls[b][0] + 3] for b in range(3)]
                for i in range(3):
                    for j in range(i + 1, 3):
                        dist = float(np.linalg.norm(ps[i] - ps[j]))
                        if dist < min_interball["dist_m"]:
                            min_interball["dist_m"] = dist
                            min_interball["time_s"] = float(d.time)
                            min_interball["pair"] = [i, j]

            for side in ("right", "left"):
                h = controller.hands[side]
                if h.get("mode") == "throw" and h.get("release_pending") is not None:
                    key = (side, round(float(h["stroke"]) * 1000))
                    if key not in seen_sep_keys:
                        seen_sep_keys.add(key)
                        t_sep = float(h["release_pending"])
                        age_trig = t_sep - float(h["stroke"])
                        inc = eval_incoming_state(controller, side)
                        cond_ok = (
                            inc is not None
                            and inc["dt_to_impact_s"] < dt_imp_max
                            and inc["v_away_intercept_m_s"] > v_away_min
                        )
                        if cond_ok:
                            pulse = h["path"].cartesian
                            pulse.brake = _PostSepBrakeWrapper(
                                pulse.brake, age_trig, t_post_brake, coast_ratio
                            )
                            trig = {
                                "throw_idx": int(controller.throw_count + 1),
                                "side": side,
                                "held_ball": h.get("held"),
                                "t_sep_s": t_sep,
                                "dt_to_impact_s": float(inc["dt_to_impact_s"]),
                                "v_away_intercept_m_s": float(inc["v_away_intercept_m_s"]),
                                "recontact_after_sep_times": [],
                            }
                            triggers.append(trig)
                            active_triggers[side] = trig
                            print(
                                f"[{seed}/{group}] BRAKE T#{trig['throw_idx']} ({side}) t_sep={t_sep:.4f}s "
                                f"dt_imp={inc['dt_to_impact_s']:.3f}s v_away={inc['v_away_intercept_m_s']:.2f}m/s",
                                flush=True,
                            )
                if side in active_triggers:
                    trig = active_triggers[side]
                    held = h.get("held")
                    if held is not None and held == trig["held_ball"]:
                        if controller.ball_contact(held, side):
                            trig["recontact_after_sep_times"].append(float(d.time))
                    else:
                        del active_triggers[side]

            if flushed < len(controller.events):
                with events_path.open("a", encoding="utf-8") as f:
                    while flushed < len(controller.events):
                        ev = controller.events[flushed]
                        f.write(json.dumps({"record_type": "event", "event": ev}, allow_nan=False) + "\n")
                        kind = ev.get("kind")
                        if kind in ("release", "catch", "initial_hold", "collision", "plan_rejected"):
                            elapsed = time.perf_counter() - t0
                            extra = ""
                            if kind == "release":
                                extra = f"rel_cnt={ev.get('count')}"
                            elif kind == "catch":
                                extra = f"rel_id={ev.get('release_id')}"
                            elif kind == "plan_rejected":
                                extra = f"err={ev.get('error'):.4f}" if ev.get("error") is not None else ""
                            print(
                                f"[{seed}/{group} t={ev.get('time', d.time):.3f}s wall={elapsed:.1f}s] "
                                f"{kind:<13} side={ev.get('side')} ball={ev.get('ball')} "
                                f"throws={controller.throw_count} catches={controller.catch_count} {extra}",
                                flush=True,
                            )
                        flushed += 1
        return res

    controller.step = hooked_step
    print(f"[{seed}/{group}] START full30 online replay (cfg_hash={cfg_hash})", flush=True)
    result = controller.run_complete(timeout=float(sp["sim_timeout_s"]))
    wall_s = time.perf_counter() - t0

    if flushed < len(controller.events):
        with events_path.open("a", encoding="utf-8") as f:
            while flushed < len(controller.events):
                ev = controller.events[flushed]
                f.write(json.dumps({"record_type": "event", "event": ev}, allow_nan=False) + "\n")
                flushed += 1

    releases = [e for e in controller.events if e.get("kind") == "release"]
    catches = [e for e in controller.events if e.get("kind") == "catch"]
    collisions = [e for e in controller.events if e.get("kind") == "collision"]
    catches_by_rel = {int(e["release_id"]): e for e in catches if e.get("release_id") is not None}

    failure_event = getattr(controller, "failure_event", None)
    live_failure_reason = failure_event.get("reason") if isinstance(failure_event, dict) else None
    planned_rollout_violation = None
    if isinstance(failure_event, dict) and live_failure_reason == "NO_FEASIBLE_RECEPTION":
        rp = failure_event.get("reception_plan") or {}
        diag = rp.get("diagnostics") or {}
        prv = diag.get("first_forbidden_contact")
        planned_rollout_violation = prv.get("reason") if isinstance(prv, dict) else prv

    # Summarize search_log and retry decisions
    slog = wrapper.search_log
    first_searches = [s for s in slog if not s["is_retry"]]
    retry_searches = [s for s in slog if s["is_retry"]]
    by_key: dict[str, list[dict]] = {}
    for s in slog:
        by_key.setdefault(s["key"], []).append(s)

    retry_details = []
    triggered_count = 0
    skipped_below_thresh = 0
    skipped_invalid_err = 0
    for dec in wrapper.decision_log:
        key = dec["key"]
        ks = by_key.get(key, [])
        r1 = ks[1] if len(ks) >= 2 else None
        obs = dec["obs"]
        if dec["applied_analytic"]:
            triggered_count += 1
        elif gcfg["enable_analytic_retry_fallback"]:
            if not obs.get("trial1_feasible") or obs.get("trial1_err_x0_m") is None:
                skipped_invalid_err += 1
            else:
                skipped_below_thresh += 1
        retry_details.append({
            "time_s": dec["time_s"],
            "side": dec["side"],
            "key": key,
            "selector_invoked": dec["selector_invoked"],
            "action": dec["action"],
            "applied_analytic": dec["applied_analytic"],
            "fallback_reason": dec["fallback_reason"],
            "trial1_feasible": obs.get("trial1_feasible"),
            "trial1_err_x0_m": obs.get("trial1_err_x0_m"),
            "trial1_weighted_time_err": obs.get("trial1_weighted_time_err"),
            "trial1_score": obs.get("trial1_score"),
            "trial1_forward_error_share": obs.get("trial1_forward_error_share"),
            "first_search_score": obs.get("first_search_score"),
            "retry1_accepted": r1["accepted"] if r1 else None,
            "retry1_score": r1["score"] if r1 else None,
            "retry1_trials": r1["trials"] if r1 else None,
            "retry1_wall_s": round(float(r1["wall_s"]), 2) if r1 else None,
            "throw_total_searches": len(ks),
            "throw_total_trials": sum(int(x["trials"]) for x in ks),
            "throw_total_search_wall_s": round(sum(float(x["wall_s"]) for x in ks), 2),
        })

    rec = {
        "seed": int(seed),
        "group": group,
        "config_hash": cfg_hash,
        "wall_timeout": False,
        "physical_result": "COMPLETE",
        "wall_s": round(wall_s, 2),
        "sim_time_s": round(float(result.get("time", controller.data.time)), 4),
        "success": bool(result.get("success", False)),
        "error": result.get("error"),
        "live_failure_reason": live_failure_reason,
        "planned_rollout_violation": planned_rollout_violation,
        "throw_count": int(controller.throw_count),
        "catch_count": int(controller.catch_count),
        "validated_flights": result.get("validated_flights"),
        "released_count": len(releases),
        "confirmed_catches_count": len(catches_by_rel),
        "initial_hold": result.get("initial_hold"),
        "final_hold": result.get("final_hold"),
        "longest_final_support_s": result.get("longest_final_support_s"),
        "airborne_at_end": result.get("airborne_at_end"),
        "actual_ball_contacts": int(controller.ball_contacts),
        "actual_live_collisions": collisions,
        "actual_live_collision_count": len(collisions),
        "actual_minimum_distance_m": float(controller.minimum_distance),
        "actual_min_interball_live": min_interball,
        "actual_max_rh_elbow_ending": max_rh_elbow_ending if max_rh_elbow_ending["time_s"] is not None else None,
        "actual_max_rh_elbow_all": max_rh_elbow_all,
        "actual_max_lh_elbow_all": max_lh_elbow_all,
        "actual_minimum_height_m": float(controller.minimum_height),
        "post_sep_brake_trigger_count": len(triggers),
        "post_sep_brake_recontact_count": sum(len(t["recontact_after_sep_times"]) for t in triggers),
        "search_stats": {
            "total_plan_throw_calls": len(slog),
            "first_search_calls": len(first_searches),
            "first_search_accepted": sum(1 for s in first_searches if s["accepted"]),
            "first_search_rejected": sum(1 for s in first_searches if not s["accepted"]),
            "retry_search_calls": len(retry_searches),
            "first_retry_entries": len(wrapper.decision_log),
            "selector_calls": int(wrapper.selector_calls),
            "analytic_fallback_triggered_count": triggered_count,
            "analytic_fallback_skipped_below_thresh_count": skipped_below_thresh,
            "analytic_fallback_skipped_invalid_err_count": skipped_invalid_err,
            "total_trials": sum(int(s["trials"]) for s in slog),
            "first_search_trials": sum(int(s["trials"]) for s in first_searches),
            "retry_trials": sum(int(s["trials"]) for s in retry_searches),
            "extra_probe_trials": int(wrapper.extra_probe_trials),
            "total_search_wall_s": round(sum(float(s["wall_s"]) for s in slog), 2),
        },
        "retry_decisions": retry_details,
    }
    out_file.write_text(json.dumps(rec, indent=2, allow_nan=False), encoding="utf-8")
    print(
        f"[{seed}/{group} DONE] success={rec['success']} throws={rec['throw_count']} "
        f"catches={rec['catch_count']} hold={rec['longest_final_support_s']}s "
        f"analytic_trig={triggered_count}/{len(wrapper.decision_log)} "
        f"total_trials={rec['search_stats']['total_trials']} probes={rec['search_stats']['extra_probe_trials']} "
        f"ball_contacts={rec['actual_ball_contacts']} live_coll={rec['actual_live_collision_count']} "
        f"err={rec['error']} wall={wall_s:.1f}s",
        flush=True,
    )


def assemble_summary(cfg: dict, cfg_hash: str, output_dir: Path) -> Path:
    seeds = [int(s) for s in cfg["seeds"]]
    pairs = []
    for seed in seeds:
        row = {"seed": seed}
        for grp in ("a0_native", "b1_fixed_analytic"):
            p = output_dir / "branches" / f"seed_{seed}_{grp}.json"
            row[grp] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        pairs.append(row)
    summary = {
        "experiment": cfg["experiment"],
        "config_hash": cfg_hash,
        "seeds": seeds,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pairs": pairs,
    }
    out_path = output_dir / "independent_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return out_path


def check_archive(cfg: dict, cfg_hash: str) -> None:
    """Validate the frozen configuration against all six archived branch records."""
    summary = json.loads(ARCHIVED_RESULTS.read_text(encoding="utf-8"))
    seeds = cfg["seeds"]
    if (
        summary["experiment"] != cfg["experiment"]
        or summary["config_hash"] != cfg_hash
        or summary["seeds"] != seeds
        or [pair["seed"] for pair in summary["pairs"]] != seeds
    ):
        raise ValueError("Archived summary does not match the frozen configuration")
    for pair in summary["pairs"]:
        for group in cfg["groups"]:
            branch = pair[group]
            if (
                branch["seed"] != pair["seed"]
                or branch["group"] != group
                or branch["config_hash"] != cfg_hash
                or branch["physical_result"] != "COMPLETE"
                or branch["wall_timeout"]
            ):
                raise ValueError(f"Invalid archived branch for seed {pair['seed']}, group {group}")
    print(f"[ARCHIVE OK] {len(seeds)} complete pairs, configuration {cfg_hash}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--group", type=str, choices=("a0_native", "b1_fixed_analytic"))
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check-archive", action="store_true", help="Check archived results without simulation")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    if args.worker:
        run_worker(args.seed, args.group, output_dir)
        return

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    cfg_hash = compute_config_hash(cfg)
    verify_interface_smoke(cfg)

    if args.check_archive:
        check_archive(cfg, cfg_hash)
        return

    branch_dir = output_dir / "branches"
    log_dir = output_dir / "logs"
    branch_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    wall_timeout_s = float(cfg["shared_frozen_params"]["wall_timeout_s"])
    tasks = []
    for seed in cfg["seeds"]:
        for grp in ("a0_native", "b1_fixed_analytic"):
            bfile = branch_dir / f"seed_{seed}_{grp}.json"
            if bfile.exists() and cfg["execution"].get("skip_completed_matching_config", True):
                existing = json.loads(bfile.read_text(encoding="utf-8"))
                if (
                    existing.get("config_hash") == cfg_hash
                    and existing.get("physical_result") == "COMPLETE"
                    and not existing.get("wall_timeout")
                ):
                    print(f"[SKIP COMPLETE] seed={seed} group={grp}", flush=True)
                    continue
            tasks.append((int(seed), grp))

    summary_lock = threading.Lock()
    assemble_summary(cfg, cfg_hash, output_dir)

    def _run_one(seed: int, grp: str):
        bfile = branch_dir / f"seed_{seed}_{grp}.json"
        lfile = log_dir / f"seed_{seed}_{grp}.log"
        print(f"[LAUNCH] seed={seed} group={grp} timeout={wall_timeout_s:.0f}s log={lfile.name}", flush=True)
        t0 = time.perf_counter()
        with lfile.open("w", encoding="utf-8") as lf:
            try:
                proc = subprocess.run(
                    [
                        sys.executable, "-X", "faulthandler", "-u",
                        "-m", "v8_robust_control.experiments.retry_fallback",
                        "--output-dir", str(output_dir), "--worker", "--seed", str(seed), "--group", grp,
                    ],
                    cwd=REPO_ROOT,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=wall_timeout_s,
                )
                elapsed = time.perf_counter() - t0
                if proc.returncode != 0:
                    err_rec = {
                        "seed": seed,
                        "group": grp,
                        "config_hash": cfg_hash,
                        "wall_timeout": False,
                        "physical_result": "ERROR",
                        "returncode": proc.returncode,
                        "wall_s": round(elapsed, 2),
                    }
                    bfile.write_text(json.dumps(err_rec, indent=2), encoding="utf-8")
                print(f"[FINISH] seed={seed} group={grp} rc={proc.returncode} wall={elapsed:.1f}s", flush=True)
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - t0
                pending_rec = {
                    "seed": seed,
                    "group": grp,
                    "config_hash": cfg_hash,
                    "wall_timeout": True,
                    "physical_result": "PENDING",
                    "timeout_limit_s": wall_timeout_s,
                    "wall_s": round(elapsed, 2),
                }
                bfile.write_text(json.dumps(pending_rec, indent=2), encoding="utf-8")
                print(f"[TIMEOUT PENDING] seed={seed} group={grp} wall={elapsed:.1f}s -> {bfile.name}", flush=True)
        with summary_lock:
            assemble_summary(cfg, cfg_hash, output_dir)
        return (seed, grp)

    max_workers = max(1, int(args.concurrency))
    print(f"[ORCHESTRATOR] Launching {len(tasks)} branches with concurrency={max_workers}", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_run_one, s, g) for (s, g) in tasks]
        for fut in as_completed(futs):
            fut.result()

    final_path = assemble_summary(cfg, cfg_hash, output_dir)
    print(f"[ALL 6 BRANCHES DONE] summary={final_path}", flush=True)


if __name__ == "__main__":
    main()

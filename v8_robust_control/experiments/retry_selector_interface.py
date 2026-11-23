"""Minimal switchable retry-only throw warm-start selector interface (experimental wrapper).

Design:
- First search (retry_throw_key != key): 100% native plan_throw, 0 extra probes, 0 selector callbacks.
  Captures trial-1 evaluate(archived_warm_vec) error vector and first-search final residual.
- First retry (retry_throw_key == key, first retry for this throw key):
  Extracts strictly decision-time observable features (NO seed, NO throw_idx, NO future outcomes)
  and invokes selector_fn(obs) -> action in {0, 1}:
    - a = 0 (or selector disabled): keeps native retry_throw_command untouched (bit-exact native).
    - a = 1: uses archived execution_warm[key] + [clip(-trial1_err_x0 / flight, -0.75, +0.75), 0, 0, 0],
             reusing first-search trial-1 err_x0 (0 extra probes); falls back to native if trial-1 had
             no valid finite err_x0.
- Preserves native search budget, acceptance threshold (0.02), and all safety criteria.
Does NOT modify src/ or v8_robust_control/.
"""
from __future__ import annotations

import sys
import time
from typing import Callable

import numpy as np

DVX_CLIP_M_S = 0.75
COMP_NAMES = [
    "land_x_m",
    "land_y_m",
    "weighted_time_err",
    "apex_ceiling_pen",
    "upstroke_rise_pen",
    "vertical_ratio_pen",
    "fingers_fwd_pen",
    "ball_spacing_pen",
]


class RetrySelectorWrapper:
    """Attachable experimental wrapper for controller.plan_throw."""

    def __init__(self, controller, selector_fn: Callable[[dict], int] | None = None):
        self.controller = controller
        self.selector_fn = selector_fn
        self._orig_plan_throw = controller.plan_throw
        self._first_search_cache: dict[str, dict] = {}
        self._retry_decided_keys: set[str] = set()
        self.selector_calls: int = 0
        self.extra_probe_trials: int = 0
        self.decision_log: list[dict] = []
        self.search_log: list[dict] = []
        controller.plan_throw = self._wrapped_plan_throw

    def detach(self):
        if self.controller.__dict__.get("plan_throw") == self._wrapped_plan_throw:
            self.controller.__dict__.pop("plan_throw", None)

    def _extract_retry_obs(self, side: str, key: str, fs: dict) -> dict:
        c = self.controller
        h = c.hands[side]
        d = c.data
        p_seat = d.site_xpos[h["seat"]].copy()
        v_seat = c.spatial(h)[3:].copy()
        R_seat = d.site_xmat[h["seat"]].reshape(3, 3)
        p_nom = h["p"].copy()
        held = int(h["held"])
        qb, vb = c.balls[held]
        b_local = R_seat.T @ (d.qpos[qb:qb + 3] - p_seat)
        b_rel_v = d.qvel[vb:vb + 3] - v_seat

        t1 = fs.get("trial1_eval") or {}
        t1_feas = bool(t1.get("feasible", False))
        t1_c = t1.get("components") or {}
        t1_score = float(t1["score"]) if (t1_feas and t1.get("score") is not None) else None
        t1_err_x0 = float(t1_c["land_x_m"]) if (t1_feas and "land_x_m" in t1_c) else None
        fwd_share = (
            float(abs(t1_err_x0) / max(t1_score, 1e-9))
            if (t1_feas and t1_err_x0 is not None and t1_score is not None)
            else None
        )
        fc = fs.get("final_components") or {}

        return {
            "trial1_feasible": t1_feas,
            "trial1_err_x0_m": t1_err_x0,
            "trial1_land_y_m": float(t1_c["land_y_m"]) if (t1_feas and "land_y_m" in t1_c) else None,
            "trial1_weighted_time_err": float(t1_c["weighted_time_err"]) if (t1_feas and "weighted_time_err" in t1_c) else None,
            "trial1_apex_ceiling_pen": float(t1_c["apex_ceiling_pen"]) if (t1_feas and "apex_ceiling_pen" in t1_c) else None,
            "trial1_score": t1_score,
            "trial1_forward_error_share": fwd_share,
            "first_search_score": fs.get("score"),
            "first_search_land_x_m": float(fc["land_x_m"]) if "land_x_m" in fc else None,
            "first_search_land_y_m": float(fc["land_y_m"]) if "land_y_m" in fc else None,
            "first_search_weighted_time_err": float(fc["weighted_time_err"]) if "weighted_time_err" in fc else None,
            "first_search_apex_ceiling_pen": float(fc["apex_ceiling_pen"]) if "apex_ceiling_pen" in fc else None,
            "palm_offset_from_nom_m": (p_seat - p_nom).tolist(),
            "palm_vel_m_s": v_seat.tolist(),
            "ball_local_offset_m": b_local.tolist(),
            "ball_rel_vel_m_s": b_rel_v.tolist(),
        }

    def _wrapped_plan_throw(self, side: str):
        c = self.controller
        if c.silent:
            return self._orig_plan_throw(side)

        h = c.hands[side]
        key = f"throw:{side}:{c.throw_count}"
        is_retry = h.get("retry_throw_key") == key
        t_call = float(c.data.time)
        archived_warm = np.asarray(
            c.execution_warm.get(key, c.warm.get(key, c.commands[side])),
            dtype=float,
        )[:4].copy()

        # If this is the FIRST retry after a rejected first search, invoke selector (if enabled)
        retry_decision = None
        if is_retry and (key not in self._retry_decided_keys) and (key in self._first_search_cache):
            self._retry_decided_keys.add(key)
            fs = self._first_search_cache[key]
            obs = self._extract_retry_obs(side, key, fs)
            native_retry_cmd = np.asarray(h["retry_throw_command"], dtype=float)[:4].copy()
            if self.selector_fn is None:
                action = 0
                selector_invoked = False
            else:
                self.selector_calls += 1
                selector_invoked = True
                action = int(self.selector_fn(obs))
                if action not in (0, 1):
                    raise ValueError(f"Invalid retry selector action: {action}")

            applied_analytic = False
            fallback_reason = None
            chosen_cmd = native_retry_cmd.copy()
            if action == 1:
                err_x0 = obs["trial1_err_x0_m"]
                if obs["trial1_feasible"] and err_x0 is not None and np.isfinite(err_x0):
                    dvx = float(np.clip(-float(err_x0) / float(c.flight), -DVX_CLIP_M_S, DVX_CLIP_M_S))
                    chosen_cmd = fs["archived_warm_vec"].copy()
                    chosen_cmd[0] = float(chosen_cmd[0] + dvx)
                    h["retry_throw_command"] = chosen_cmd.copy()
                    applied_analytic = True
                else:
                    fallback_reason = "first_search_trial1_invalid_err_x0"

            retry_decision = {
                "time_s": t_call,
                "side": side,
                "key": key,
                "selector_invoked": selector_invoked,
                "action": action,
                "applied_analytic": applied_analytic,
                "fallback_reason": fallback_reason,
                "native_retry_cmd": native_retry_cmd.tolist(),
                "chosen_cmd": chosen_cmd.tolist(),
                "obs": obs,
            }
            self.decision_log.append(retry_decision)

        # Instrument evaluate() calls to capture trial-1 error and trial/wall counts (0 extra probes)
        ev_before = len(c.events)
        trials = 0
        infeasible = 0
        trial1_eval = None

        def prof(frame, event, arg):
            nonlocal trials, infeasible, trial1_eval
            if frame.f_code.co_qualname != "ContactCascade.plan_throw.<locals>.evaluate":
                return
            if event == "call":
                trials += 1
            elif event == "return":
                if isinstance(arg, tuple) and len(arg) == 2:
                    if trials == 1 and trial1_eval is None:
                        ev_vec = np.asarray(arg[0], dtype=float)
                        trial1_eval = {
                            "feasible": True,
                            "score": float(np.linalg.norm(ev_vec)),
                            "components": {n: float(x) for n, x in zip(COMP_NAMES, ev_vec)},
                        }
                else:
                    infeasible += 1
                    if trials == 1 and trial1_eval is None:
                        trial1_eval = {"feasible": False}

        t0 = time.perf_counter()
        prev = sys.getprofile()
        try:
            sys.setprofile(prof)
            path = self._orig_plan_throw(side)
        finally:
            sys.setprofile(prev)
        wall_s = time.perf_counter() - t0

        score = float(c.last_plan_error)
        accepted = bool(path is not None and score <= 0.02)
        new_evs = c.events[ev_before:]
        plan_ev = next((e for e in reversed(new_evs) if e.get("kind") in ("plan", "plan_rejected") and e.get("side") == side), None)
        comps = plan_ev.get("objective_components") if plan_ev else None
        comp_map = {n: float(x) for n, x in zip(COMP_NAMES, comps)} if comps else None
        safety_pen_zero = bool(
            comp_map is not None
            and all(abs(comp_map[k]) == 0.0 for k in ("apex_ceiling_pen", "upstroke_rise_pen", "vertical_ratio_pen", "fingers_fwd_pen", "ball_spacing_pen"))
        )

        rec = {
            "time_s": t_call,
            "side": side,
            "key": key,
            "is_retry": is_retry,
            "trials": trials,
            "infeasible": infeasible,
            "wall_s": wall_s,
            "score": score if np.isfinite(score) else None,
            "accepted": accepted,
            "safety_penalties_zero": safety_pen_zero,
            "trial1_eval": trial1_eval,
            "final_components": comp_map,
            "best_command": plan_ev.get("command") if plan_ev else None,
            "retry_decision_action": retry_decision["action"] if retry_decision else None,
        }
        self.search_log.append(rec)

        if (not is_retry) and (not accepted):
            self._first_search_cache[key] = {
                **rec,
                "archived_warm_vec": archived_warm.copy(),
            }
        return path

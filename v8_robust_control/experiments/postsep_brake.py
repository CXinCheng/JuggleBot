"""Braking and intercept helpers used by the archived retry comparison."""

import numpy as np

SUPPORTED_OFFSET_M = 0.0223


class _PostSepBrakeWrapper:
    """C2-continuous piecewise brake wrapper triggered at detected physical separation age."""

    def __init__(self, orig_brake, age_trig: float, duration_s: float, coast_ratio: float):
        from jugglebot.humanoid.planning import Quintic

        self.orig_brake = orig_brake
        self.age_trig = float(age_trig)
        p_s, v_s, a_s = orig_brake.sample(self.age_trig)
        self.fast_brake = Quintic(
            self.age_trig,
            float(duration_s),
            p_s,
            v_s,
            a_s,
            p_s + float(coast_ratio) * float(duration_s) * v_s,
            np.zeros(3),
            np.zeros(3),
        )

    def sample(self, t: float):
        if t <= self.age_trig:
            return self.orig_brake.sample(t)
        return self.fast_brake.sample(t)


def eval_incoming_state(ctrl, side: str):
    """Return runtime-observable incoming-ball intercept metrics for `side`, or None."""
    h = ctrl.hands[side]
    catch_p = h["catch_p"]
    p_seat = ctrl.data.site_xpos[h["seat"]]
    v_seat = ctrl.spatial(h)[3:]
    for b_id, tgt in ctrl.targets.items():
        if tgt != side:
            continue
        st = ctrl.reception_ball_state(b_id)
        if st is None:
            continue
        p_b, v_b = st
        disc = v_b[2] ** 2 + 2.0 * ctrl.gravity_m_s2 * (p_b[2] - catch_p[2] - SUPPORTED_OFFSET_M)
        dt = float((v_b[2] + np.sqrt(max(0.0, disc))) / ctrl.gravity_m_s2)
        p_int = catch_p.copy()
        p_int[:2] = p_b[:2] + v_b[:2] * dt
        to_int = p_int - p_seat
        dist_int = float(np.linalg.norm(to_int))
        dir_int = to_int / max(dist_int, 1e-9)
        v_away = float(-(v_seat @ dir_int))
        return {
            "ball": int(b_id),
            "dt_to_impact_s": dt,
            "dist_to_intercept_m": dist_int,
            "v_away_intercept_m_s": v_away,
            "p_intercept_m": p_int.tolist(),
        }
    return None

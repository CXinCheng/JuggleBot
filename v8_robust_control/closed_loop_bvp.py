"""Native closed-loop BVP throw avoidance, tight-turnaround clearance, and torque-CBF guards.

This module promotes the Scheme-B closed-loop robust control architecture into
``v8_robust_control`` as clean, zero-patch OOP subclasses of ``TaskSpaceCascade``
and ``TaskResidualEnv`` while keeping ``task_control.py`` and ``task_env.py``
(and thus all ``measured_sources()`` SHA-256 reference fingerprints) untouched.

Strict causality & zero-perturbation invariance:
- All closed-loop overrides are gated by ``action_authority(obs) > 0.5``
  (``controller.perturbed_active``). At ``seat_jitter_m == 0.0`` (unperturbed
  nominal trajectory), every callback executes the exact nominal baseline path.
- All throw/reception predictions rely exclusively on detached causal
  ``TaskRequest.balls`` / ``observed_ball()`` estimates (never future states).
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import numpy as np
import mujoco

from jugglebot.humanoid.planning import Quintic
from .task_control import THROW_RESIDUAL_LIMITS, RECEPTION_RESIDUAL_LIMITS
from .task_env import TaskConfig, TaskResidualEnv, _EpisodeCascade, action_authority


@dataclass(frozen=True)
class ClosedLoopConfig:
    """Hyperparameters for the causal closed-loop BVP + torque-CBF controller."""
    min_released: int = 2
    apex_trim_dvz: float = -0.15
    d_safe: float = 0.120
    max_defer: float = 0.022
    max_apex: float = 1.275
    dvz_grid: tuple[float, ...] = (0.0, -0.18, -0.34, -0.44, 0.08)
    k_aim: float = 0.20
    k_xz: float = 0.35
    defer_pen: float = 1.0
    time_pen: float = 0.0
    skip_bvp_on_tight: bool = True
    wrist_cbf: bool = True
    stiff_elbow_cbf: bool = True
    enable_uncaught_braking: bool = True
    uncaught_wrist_clear_m: float = 0.085
    uncaught_stand_hard_m: float = 0.020
    uncaught_obs_reset_stand_m: float = 0.045
    cbf_start_time_s: float = 0.5
    tight_window: float = 0.295
    tight_vz_max: float = 9.0
    tight_scale: float = 0.90
    ultra_tight_scale: float = 0.86
    tight_lx: float = -0.010
    tight_ly: float = 0.022
    tight_vy_comp: float = 0.26
    tight_dip: float = 0.025
    thumb_clamp: bool = True
    thumb_min_flight: int = 2
    pg_out: float = 0.084
    pg_clr: float = 0.120
    ergo_recv: bool = True
    verbose: bool = False


def predict_ball_pos(state, t_query: float, g: float) -> np.ndarray:
    """Ballistic position prediction from a causal ``BallEstimate``."""
    dt = t_query - state.sample_time_s
    p = np.asarray(state.position_m, dtype=float)
    v = np.asarray(state.velocity_m_s, dtype=float)
    return p + v * dt + np.array([0., 0., -0.5 * g * dt * dt])


def predict_crossing_time(state, z_plane: float, g: float) -> dict | None:
    """Causal downward crossing time and XY intercept of an observed ball."""
    if state is None or not state.valid or state.velocity_m_s is None:
        return None
    p = np.asarray(state.position_m, dtype=float)
    v = np.asarray(state.velocity_m_s, dtype=float)
    disc = v[2] * v[2] + 2. * g * (p[2] - z_plane)
    if disc < 0.:
        return None
    dt = (v[2] + float(np.sqrt(disc))) / g
    if dt <= 0.:
        return None
    return dict(
        time_s=float(state.sample_time_s + dt),
        xy_m=p[:2] + v[:2] * dt,
        vz=float(v[2] - g * dt),
    )


def eval_midair_clearance(
    p_rel: np.ndarray,
    v_rel: np.ndarray,
    t_rel: float,
    other_states: list,
    g: float,
) -> tuple[float, float, np.ndarray]:
    """Minimum 3D distance between an outgoing ball and all other airborne balls."""
    if not other_states:
        return 1.0, 0.0, np.zeros(3)
    ts = np.linspace(0.0, 0.36, 73)  # 5 ms grid
    d_min = 10.0
    t_best = 0.0
    diff_best = np.zeros(3)
    for dt in ts:
        t_abs = t_rel + dt
        p_out = p_rel + v_rel * dt + np.array([0., 0., -0.5 * g * dt * dt])
        for st in other_states:
            p_oth = predict_ball_pos(st, t_abs, g)
            if p_oth[2] < 0.72:
                continue
            diff = p_out - p_oth
            dist = float(np.linalg.norm(diff))
            if dist < d_min:
                d_min = dist
                t_best = float(t_abs)
                diff_best = diff
    return d_min, t_best, diff_best


def solve_closed_loop_throw(request, cfg: ClosedLoopConfig, controller) -> np.ndarray:
    """Strictly causal closed-loop BVP throw residual (normalized 6D action)."""
    if controller is None or request.released < cfg.min_released:
        return np.zeros(6)
    g = controller.gravity_m_s2
    side = request.side
    other_side = 'left' if side == 'right' else 'right'
    sign = 1.0 if side == 'right' else -1.0
    hand_idx = 0 if side == 'right' else 1
    p_start = np.asarray(request.palm_position_m[hand_idx], dtype=float)
    nom_seat = controller.hands[side]['p']
    target_catch = controller.hands[other_side]['catch_p'] + np.array([0., 0., 0.042])

    other_states = []
    for bid, st in enumerate(request.balls):
        if bid == request.ball_id or st is None or not st.valid or st.velocity_m_s is None:
            continue
        if st.position_m[2] > 0.88 or abs(st.velocity_m_s[2]) > 0.35:
            other_states.append(st)

    # Calibrated empirical release map (dt_rel ~ 0.168 s, J_cmd ~ [0.55, 0.70, 0.55])
    dt_rel_nom = 0.168
    t_rel_nom = request.time_s + dt_rel_nom
    p_rel_nom = p_start + np.array([0.012, sign * 0.055, 0.185])
    T_nom = 0.48
    vz_nom = max(2.26, (target_catch[2] - p_rel_nom[2] + 0.5 * g * T_nom * T_nom) / T_nom)
    v_rel_nom = np.array([
        (target_catch[0] - p_rel_nom[0]) / T_nom,
        (target_catch[1] - p_rel_nom[1]) / T_nom,
        vz_nom,
    ])
    J_cmd = np.array([0.55, 0.70, 0.55])

    dp_start = p_start - nom_seat
    dv_rel_aim = np.array([
        -cfg.k_aim * dp_start[0] / T_nom,
        -cfg.k_aim * dp_start[1] / T_nom,
        0.0,
    ])

    d_nom, _, _ = eval_midair_clearance(p_rel_nom, v_rel_nom, t_rel_nom, other_states, g)
    best_dv_cmd = np.clip(dv_rel_aim / J_cmd, [-0.25, -0.25, -0.35], [0.25, 0.25, 0.35])
    best_defer = 0.0

    is_tight = False
    z_self_catch = controller.hands[side]['catch_p'][2] + 0.042
    for st in other_states:
        cr = predict_crossing_time(st, z_self_catch, g)
        if cr is not None and ((side == 'right' and cr['xy_m'][1] < -0.02)
                               or (side == 'left' and cr['xy_m'][1] > 0.02)):
            if 0.12 < (cr['time_s'] - request.time_s) < cfg.tight_window:
                is_tight = True
                break

    if d_nom < cfg.d_safe and other_states and not (is_tight and cfg.skip_bvp_on_tight):
        max_defer = 0.0 if is_tight else min(request.remaining_defer_s, cfg.max_defer)
        defer_grid = [0.0]
        if max_defer >= 0.010:
            defer_grid.extend(d for d in (0.010, 0.016, 0.022) if d <= max_defer + 1e-9)

        best_score = -1e9
        best_d_try = d_nom
        cand_dv = best_dv_cmd.copy()
        cand_df = 0.0
        for df in defer_grid:
            for dvz_rel in cfg.dvz_grid:
                vz_new = v_rel_nom[2] + dvz_rel
                apex = p_rel_nom[2] + max(0., vz_new) ** 2 / (2. * g)
                if apex > cfg.max_apex:
                    continue
                disc = vz_new ** 2 + 2. * g * (p_rel_nom[2] - target_catch[2])
                if disc <= 0.:
                    continue
                T_new = (vz_new + float(np.sqrt(disc))) / g
                if T_new < 0.34 or T_new > 0.62:
                    continue
                for dlx in (0.0, -0.025, 0.025, -0.045, 0.045, -0.065, 0.065):
                    for dly in (0.0, -0.025, 0.025, -0.045, 0.045):
                        v_try = np.array([
                            (target_catch[0] + dlx - p_rel_nom[0]) / T_new,
                            (target_catch[1] + dly - p_rel_nom[1]) / T_new,
                            vz_new,
                        ])
                        dv_rel = (v_try - v_rel_nom) + dv_rel_aim
                        dv_cmd = dv_rel / J_cmd
                        if np.any(np.abs(dv_cmd) > THROW_RESIDUAL_LIMITS[:3] * 0.90):
                            continue
                        d_try, _, _ = eval_midair_clearance(p_rel_nom, v_try, t_rel_nom + df, other_states, g)
                        col_pen = 250.0 * max(0.0, 0.105 - d_try)
                        clear_term = min(d_try, cfg.d_safe + 0.025) * 24.0
                        land_pen = 8.0 * float(np.hypot(dlx, dly))
                        defer_pen = cfg.defer_pen * df
                        time_pen = cfg.time_pen * max(0., T_nom - (T_new + df))
                        apex_pen = 20.0 * max(0., apex - (cfg.max_apex - 0.025))
                        cmd_pen = 0.08 * float(np.linalg.norm(dv_cmd))
                        score = clear_term - col_pen - land_pen - defer_pen - time_pen - apex_pen - cmd_pen
                        if score > best_score:
                            best_score = score
                            best_d_try = d_try
                            cand_dv = dv_cmd
                            cand_df = df
        if best_d_try - d_nom >= 0.006:
            best_dv_cmd = cand_dv.copy()
            # Compensate ScoopPulse forward-pitch coupling when lowering vz
            if best_dv_cmd[2] < 0. and cfg.k_xz != 0.:
                best_dv_cmd[0] = float(np.clip(
                    best_dv_cmd[0] + cfg.k_xz * best_dv_cmd[2],
                    -THROW_RESIDUAL_LIMITS[0] * 0.90,
                    THROW_RESIDUAL_LIMITS[0] * 0.90,
                ))
            best_defer = cand_df

    act = np.zeros(6)
    act[:3] = best_dv_cmd / THROW_RESIDUAL_LIMITS[:3]
    if best_defer >= 0.005:
        act[5] = best_defer / 0.025
    return np.clip(act, -1., 1.)


def solve_ergonomic_reception(request, cfg: ClosedLoopConfig, controller) -> np.ndarray:
    """Clamp extreme reception coordinates into the ergonomic arm workspace."""
    act = np.zeros(6)
    if not cfg.ergo_recv or controller is None or request.released < cfg.min_released:
        return act
    rec = controller.hands[request.side].get('reception')
    if rec is None:
        return act
    nom_p = rec[1].start.copy()
    dp = np.zeros(3)
    if nom_p[0] > 0.37:
        dp[0] = float(np.clip(0.37 - nom_p[0], -0.045, 0.0))
    if abs(nom_p[1]) > 0.285:
        dp[1] = float(np.clip(np.sign(nom_p[1]) * 0.285 - nom_p[1], -0.045, 0.045))
    if nom_p[2] > 0.89:
        dp[2] = float(np.clip(0.89 - nom_p[2], -0.045, 0.0))
    if np.any(dp != 0.):
        act[:3] = np.clip(dp / RECEPTION_RESIDUAL_LIMITS[:3], -1., 1.)
    return act


class _ShiftedMotion:
    """Lightweight motion proxy adding a smooth Quintic offset without monkey-patching."""
    def __init__(self, motion, segment):
        self._motion = motion
        self._segment = segment

    def sample(self, now):
        return tuple(x + y for x, y in zip(self._motion.sample(now), self._segment.sample(now)))

    def __getattr__(self, name):
        return getattr(self._motion, name)


class _TightFingerCtrl:
    """Deepcopy-safe finger posture wrapper delaying catch closure during tight receptions."""
    def __init__(self, controller, side: str):
        self.controller = controller
        self.side = side

    def __call__(self, posture_name: str):
        c = self.controller
        fingers = c.hands[self.side]['fingers']
        base_fn = type(fingers).get_ctrl_array
        if not c.perturbed_active or posture_name != 'catch':
            return base_fn(fingers, posture_name)
        tr = c._tight_receptions.get(self.side)
        h = c.hands[self.side]
        if (tr is not None and h.get('mode') == 'receive' and h.get('held') is None
                and h.get('incoming') == tr['ball_id'] and c.data.time <= tr['t_expire']):
            u = float(np.clip((c.data.time - tr['t_flat_until']) / 0.025, 0., 1.))
            w = u * u * (3. - 2. * u)
            return (1. - w) * base_fn(fingers, 'flat') + w * base_fn(fingers, 'catch')
        return base_fn(fingers, posture_name)


class ClosedLoopBvpCascade(_EpisodeCascade):
    """Native OOP ``TaskSpaceCascade`` subclass with causal closed-loop safety modules.

    When ``self.perturbed_active`` is ``False`` (e.g. ``seat_jitter_m == 0.0``),
    every overridden method delegates directly to ``super()``, preserving exact
    bit-identical unperturbed baseline trajectories.
    """
    def __init__(self, decision=None, *, closed_loop_cfg: ClosedLoopConfig | None = None, **kwargs):
        self.cl_cfg = closed_loop_cfg or ClosedLoopConfig()
        self.perturbed_active = False
        self._tight_receptions: dict[str, dict] = {}
        self._tight_dips: dict[tuple, tuple] = {}
        self.stiff_elbow_cbf_steps = 0
        self.wrist_cbf_steps = 0
        super().__init__(decision, **kwargs)
        for side in ('right', 'left'):
            self.hands[side]['fingers'].get_ctrl_array = _TightFingerCtrl(self, side)

    def holding_fingers(self, hand, t):
        res = super().holding_fingers(hand, t)
        cfg = self.cl_cfg
        if (not self.perturbed_active or not cfg.thumb_clamp
                or hand.get('candidate_flight', 0) < cfg.thumb_min_flight
                or hand.get('capture_confirmed', True) or hand.get('held') is None):
            return res
        opp = hand.get('opposition_offset')
        if opp is not None and opp[3] < 0.:
            thumb_rows = np.array([
                idx for idx, digit in enumerate(hand['fingers'].actuator_digits) if digit == 'th'
            ])
            fix = -opp[3]
            opp[3] = 0.
            res = res.copy()
            act_idx = hand['fingers'].act_indices[thumb_rows[3]]
            res[thumb_rows[3]] = np.clip(
                res[thumb_rows[3]] + fix,
                self.model.actuator_ctrlrange[act_idx, 0],
                self.model.actuator_ctrlrange[act_idx, 1],
            )
        return res

    def plan_throw(self, side):
        cfg = self.cl_cfg
        if self.perturbed_active and self.throw_count >= cfg.min_released:
            self.posture_guard = True
            self.POSTURE_OUTWARD_TARGET_M = cfg.pg_out
            self.POSTURE_OUTWARD_ACTIVATION_M = cfg.pg_out - 0.008
            self.POSTURE_CLEARANCE_TARGET_M = cfg.pg_clr
            self.POSTURE_CLEARANCE_ACTIVATION_M = cfg.pg_clr + 0.025
        path = super().plan_throw(side)
        if path is None or not self.perturbed_active or self.throw_count < cfg.min_released or cfg.tight_window <= 0.:
            return path

        held = self.hands[side]['held']
        sign = 1.0 if side == 'right' else -1.0
        z_catch = self.hands[side]['catch_p'][2] + 0.042
        tight_ball, t_arr_min = None, 1e9
        for bid in range(len(self.balls)):
            if bid == held:
                continue
            st = self.observed_ball(bid)
            if st is None or not st.valid or st.velocity_m_s is None:
                continue
            if cfg.tight_vz_max < 9.0 and st.velocity_m_s[2] > cfg.tight_vz_max:
                continue
            cr = predict_crossing_time(st, z_catch, self.gravity_m_s2)
            if cr is None:
                continue
            if (side == 'right' and cr['xy_m'][1] < -0.02) or (side == 'left' and cr['xy_m'][1] > 0.02):
                dt_arr = cr['time_s'] - self.data.time
                if 0.12 < dt_arr < cfg.tight_window and cr['time_s'] < t_arr_min:
                    t_arr_min = cr['time_s']
                    tight_ball = bid

        if tight_ball is not None:
            dt_tight = t_arr_min - self.data.time
            s_tight = cfg.ultra_tight_scale if dt_tight < 0.24 else cfg.tight_scale
            orig_sample = path.sample
            origin = orig_sample(0.)[0]
            z = np.zeros(3)
            dl = np.array([cfg.tight_lx, sign * cfg.tight_ly, 0.])
            tau_d, dt2 = 0.070, 0.055
            v_end = np.array([0., -sign * cfg.tight_vy_comp, 0.])
            dl2 = dl + 0.5 * v_end * dt2
            seg1 = Quintic(0., tau_d, z, z, z, dl, z, z)
            seg2 = Quintic(tau_d, dt2, dl, z, z, dl2, v_end, z)

            def sh_tight(t):
                p, v, a = orig_sample(t / s_tight)
                if t < tau_d:
                    dp, dv, da = seg1.sample(t)
                elif t < tau_d + dt2:
                    dp, dv, da = seg2.sample(t)
                else:
                    dp, dv, da = dl2 + v_end * (t - tau_d - dt2), v_end, z
                return origin + s_tight * (p - origin) + dp, v + dv, a / s_tight + da

            path.sample = sh_tight
            path.duration *= s_tight
            path.left_scale = s_tight
            self._tight_receptions[side] = dict(
                ball_id=tight_ball,
                t_flat_until=t_arr_min - 0.043,
                dip_m=cfg.tight_dip,
                t_expire=t_arr_min + 0.25,
            )
        return path

    def release_fingers(self, hand, t):
        if self.perturbed_active and hand['mode'] == 'throw' and hasattr(hand.get('path'), 'left_scale'):
            t = hand['stroke'] + (t - hand['stroke']) / hand['path'].left_scale
        return super().release_fingers(hand, t)

    def follow_palm(self, hand, data, motion, time, target):
        if self.perturbed_active:
            side = hand['kin'].side
            inc = hand.get('incoming')
            receiving = hand['mode'] == 'receive' and hand.get('held') is None
            tr = self._tight_receptions.get(side)
            if tr is not None and receiving and inc == tr['ball_id'] and time <= tr['t_expire'] and tr['dip_m'] > 0.:
                key = (side, tr['ball_id'], round(tr['t_flat_until'], 2))
                if key not in self._tight_dips:
                    end = hand['reception'][1].t0
                    mid = time + 0.45 * max(0.04, end - time)
                    z = np.zeros(3)
                    dl = np.array([0., 0., -tr['dip_m']])
                    self._tight_dips[key] = (
                        mid,
                        Quintic(time, mid - time, z, z, z, dl, z, z),
                        Quintic(mid, max(0.02, end - mid), dl, z, z, z, z, z),
                    )
                mid, seg_a, seg_b = self._tight_dips[key]
                seg = seg_b if time > mid else seg_a
                return super().follow_palm(hand, data, _ShiftedMotion(motion, seg), time, target)
        return super().follow_palm(hand, data, motion, time, target)

    def _ball_hits_stationary_wrist(self, hand, data) -> bool:
        """Check whether braking in place would stall the wrist in the incoming ball's path,
        while NEVER deferring braking once stand clearance drops to ``uncaught_stand_hard_m``."""
        cfg = self.cl_cfg
        if cfg.uncaught_wrist_clear_m <= 0.:
            return False
        inc = hand.get('incoming')
        if inc is None:
            return False
        side = hand['kin'].side
        min_stand = 1e9
        for geom in self._receive_clearance_geoms[side]:
            sep = self._stand_separation(data, geom, max_distance=0.15)
            if sep is not None and sep[0] < min_stand:
                min_stand = float(sep[0])
        if min_stand <= cfg.uncaught_stand_hard_m:
            return False
        st = self.observed_ball(inc)
        if st is None or not st.valid or st.velocity_m_s is None:
            return min_stand > cfg.uncaught_obs_reset_stand_m
        wrist_body = self.model.body(side + '_wrist_yaw_link').id
        wrist_pos = data.xpos[wrist_body]
        t_now = float(data.time)
        g = self.gravity_m_s2
        for dt in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
            bp = predict_ball_pos(st, t_now + dt, g)
            if float(np.linalg.norm(bp - wrist_pos)) <= cfg.uncaught_wrist_clear_m:
                return True
        return False

    def protect_arm(self, hand, data, torque, response=None, free_acceleration=None):
        want_brake = bool(self.perturbed_active and self.cl_cfg.enable_uncaught_braking)
        if want_brake and hand.get('mode') == 'receive' and hand.get('held') is None:
            if self._ball_hits_stationary_wrist(hand, data):
                want_brake = False
        self.uncaught_braking = want_brake
        torque = super().protect_arm(hand, data, torque, response, free_acceleration)
        cfg = self.cl_cfg
        if not self.perturbed_active or response is None or data.time < cfg.cbf_start_time_s:
            return torque
        m = self.model
        side = hand['kin'].side

        # Stage-2 stiff elbow nullspace CBF near hard elbow limits (0.100 m / 0.080 m)
        if cfg.stiff_elbow_cbf:
            elbow = m.body(side + '_elbow_link').id
            shoulder = m.body(side + '_shoulder_pitch_link').id
            outward = abs(data.xpos[elbow, 1]) - abs(data.xpos[shoulder, 1])
            clearance = data.xpos[shoulder, 2] - data.xpos[elbow, 2]
            if outward > 0.092 or clearance < 0.098:
                je, js, jde, jds = (np.zeros((3, m.nv)) for _ in range(4))
                mujoco.mj_jacBody(m, data, je, None, elbow)
                mujoco.mj_jacBody(m, data, js, None, shoulder)
                mujoco.mj_jacDot(m, data, jde, None, data.xpos[elbow], elbow)
                mujoco.mj_jacDot(m, data, jds, None, data.xpos[shoulder], shoulder)
                sgn_e = np.sign(data.xpos[elbow, 1])
                cons = []
                if outward > 0.092:
                    cons.append((sgn_e * (js[1] - je[1]), sgn_e * (jds[1] - jde[1]), 0.093 - outward, outward > 0.095))
                if clearance < 0.098:
                    cons.append((js[2] - je[2], jds[2] - jde[2], clearance - 0.095, clearance < 0.090))
                nullspace = self._palm_torque_nullspace(hand, data, response)
                for grad, deriv, margin, emergency in cons:
                    row = grad @ response
                    bound = -1500. * margin - 80. * (grad @ data.qvel) - grad @ free_acceleration - deriv @ data.qvel
                    if float(row @ torque) < bound:
                        # Stage-2 uses palm-task nullspace; Stage-3 (near hard limit) uses direct per-joint clipped barrier
                        direction = row if emergency else (nullspace @ row)
                        denom = float(row @ direction)
                        if denom <= 1e-12 and not emergency:
                            direction = row
                            denom = float(row @ row)
                        if denom > 1e-12:
                            delta = direction * ((bound - float(row @ torque)) / denom)
                            if emergency:
                                torque = np.clip(torque + delta, -hand['force_limits'], hand['force_limits'])
                            else:
                                torque = self._bounded_torque_change(torque, delta, hand['force_limits'])
                            if not self.silent:
                                self.stiff_elbow_cbf_steps += 1

        # Direct minimum-norm wrist-pitch torque barrier (|q[5]| < 0.6632 rad = 38 deg)
        if cfg.wrist_cbf:
            q_wp = float(data.qpos[hand['kin'].qpos_adr][5])
            qd_wp = float(data.qvel[hand['kin'].dof_adr][5])
            if abs(q_wp) > 0.56 or abs(q_wp + 0.035 * qd_wp) > 0.630:
                sgn = 1.0 if (q_wp + 0.035 * qd_wp) >= 0. else -1.0
                margin = 0.630 - sgn * q_wp
                row = -sgn * response[hand['kin'].dof_adr[5]]
                bound = -1200. * margin + 70. * (sgn * qd_wp) + sgn * free_acceleration[hand['kin'].dof_adr[5]]
                if float(row @ torque) < bound:
                    denom = float(row @ row)
                    if denom > 1e-12:
                        torque = self._bounded_torque_change(
                            torque, row * ((bound - float(row @ torque)) / denom), hand['force_limits'])
                        if not self.silent:
                            self.wrist_cbf_steps += 1
        return torque


class ClosedLoopTaskEnv(TaskResidualEnv):
    """Native ``TaskResidualEnv`` subclass backed by ``ClosedLoopBvpCascade``."""
    def __init__(
        self,
        config: TaskConfig | None = None,
        reference=None,
        *,
        estimator: str | None = None,
        closed_loop_cfg: ClosedLoopConfig | None = None,
    ):
        self.closed_loop_cfg = closed_loop_cfg or ClosedLoopConfig()
        super().__init__(config=config, reference=reference, estimator=estimator)
        self._template = ClosedLoopBvpCascade(
            throws=self.config.throws,
            estimator=self.estimator,
            closed_loop_cfg=self.closed_loop_cfg,
        )

    def _sync_perturbation_gate(self, obs):
        if self.controller is not None and action_authority(obs) > 0.5:
            self.controller.perturbed_active = True

    def reset(self, *, seed=None, options=None):
        if getattr(self, 'controller', None) is not None:
            super().close()
            self.controller = None
            gc.collect()
        obs, info = super().reset(seed=seed, options=options)
        self._sync_perturbation_gate(obs)
        return obs, info

    def step(self, action):
        self._sync_perturbation_gate(self._observation)
        obs, reward, terminated, truncated, info = super().step(action)
        if not (terminated or truncated):
            self._sync_perturbation_gate(obs)
        return obs, reward, terminated, truncated, info


class ClosedLoopBvpPolicy:
    """Strictly causal closed-loop policy producing bounded 6D ``TaskResidualEnv`` actions."""
    def __init__(self, cfg: ClosedLoopConfig | None = None):
        self.cfg = cfg or ClosedLoopConfig()

    def predict(self, obs: dict, env: TaskResidualEnv) -> np.ndarray:
        """Compute normalized 6D action for the current callback decision."""
        if action_authority(obs) <= 0.5:
            return np.zeros(6)
        req = env._request
        if req is None or req.released < self.cfg.min_released:
            return np.zeros(6)
        controller = env.controller
        if req.kind == 'throw':
            act = solve_closed_loop_throw(req, self.cfg, controller)
            if self.cfg.apex_trim_dvz < 0.:
                trim_z = self.cfg.apex_trim_dvz / THROW_RESIDUAL_LIMITS[2]
                if act[2] > trim_z:
                    dz_cmd = (trim_z - act[2]) * THROW_RESIDUAL_LIMITS[2]
                    act[2] = trim_z
                    act[0] = float(np.clip(
                        act[0] + self.cfg.k_xz * dz_cmd / THROW_RESIDUAL_LIMITS[0],
                        -1., 1.,
                    ))
            return np.clip(act, -1., 1.)
        if req.kind == 'reception':
            return solve_ergonomic_reception(req, self.cfg, controller)
        return np.zeros(6)


class ClosedLoopResidualEnv(ClosedLoopTaskEnv):
    """Gym environment where a learned policy outputs an additive increment on top of
    the frozen rule-based ``ClosedLoopBvpPolicy`` and ``ClosedLoopTaskEnv`` safety base.

    Window gating explicitly distinguishes **decision count** (``active_max_decisions``,
    0-indexed callback count ``k < active_max_decisions``) from **release count**
    (``active_max_released``, ``req.released < active_max_released``) so deferral
    retries cannot silently widen the learned opening window.

    Execution pipeline (unified clipping + authority gating):
      1. ``rule_action = self.rule_policy.predict(self._observation, self)``
      2. ``requested_delta = np.asarray(action, float)``
      3. ``learned_delta = np.clip(requested_delta, -1., 1.) * self.residual_scale * self.active_mask``
         (zeroed whenever ``decision_k >= active_max_decisions`` or ``req.released >= active_max_released``)
      4. ``combined = np.clip(rule_action + learned_delta, -1., 1.)``
      5. ``super().step(combined)`` (applies unified ``action_authority(obs)`` & ``decode_action``)

    When ``action == 0`` (zero learned residual), ``combined == rule_action`` identically,
    guaranteeing exact bit-identical reproduction of the pure rule-based closed loop.
    """
    def __init__(
        self,
        config: TaskConfig | None = None,
        reference=None,
        *,
        estimator: str | None = None,
        closed_loop_cfg: ClosedLoopConfig | None = None,
        residual_scale: float | np.ndarray = 1.0,
        active_max_decisions: int | None = None,
        active_max_released: int | None = None,
        active_mask: tuple[float, ...] | None = None,
        record_action_log: bool = False,
    ):
        super().__init__(
            config=config,
            reference=reference,
            estimator=estimator,
            closed_loop_cfg=closed_loop_cfg,
        )
        self.rule_policy = ClosedLoopBvpPolicy(self.closed_loop_cfg)
        self.residual_scale = np.asarray(residual_scale, dtype=float)
        self.active_max_decisions = active_max_decisions
        self.active_max_released = active_max_released
        self.active_mask = np.ones(6, dtype=float) if active_mask is None else np.asarray(active_mask, dtype=float)
        self.record_action_log = bool(record_action_log)
        self.decision_index: int = 0
        self.last_rule_action: np.ndarray = np.zeros(6, dtype=float)
        self.last_requested_delta: np.ndarray = np.zeros(6, dtype=float)
        self.last_learned_delta: np.ndarray = np.zeros(6, dtype=float)
        self.last_combined_action: np.ndarray = np.zeros(6, dtype=float)
        self.action_log: list[dict] = []

    def reset(self, *, seed=None, options=None):
        self.decision_index = 0
        self.last_rule_action = np.zeros(6, dtype=float)
        self.last_requested_delta = np.zeros(6, dtype=float)
        self.last_learned_delta = np.zeros(6, dtype=float)
        self.last_combined_action = np.zeros(6, dtype=float)
        self.action_log = []
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs_before = self._observation
        req = self._request
        auth = float(action_authority(obs_before))
        k_now = self.decision_index
        rule_action = np.asarray(self.rule_policy.predict(obs_before, self), dtype=float)
        requested_delta = np.asarray(action, dtype=float)
        learned_delta = np.clip(requested_delta, -1.0, 1.0) * self.residual_scale * self.active_mask
        if self.active_max_decisions is not None and k_now >= self.active_max_decisions:
            learned_delta = np.zeros(6, dtype=float)
        if self.active_max_released is not None and (req is None or req.released >= self.active_max_released):
            learned_delta = np.zeros(6, dtype=float)
        combined = np.clip(rule_action + learned_delta, -1.0, 1.0)
        executed = combined * auth
        self.decision_index = k_now + 1
        self.last_rule_action = rule_action.copy()
        self.last_requested_delta = requested_delta.copy()
        self.last_learned_delta = learned_delta.copy()
        self.last_combined_action = combined.copy()
        if self.record_action_log and req is not None:
            self.action_log.append(dict(
                k=k_now,
                time_s=round(float(req.time_s), 4),
                kind=req.kind,
                side=req.side,
                released=int(req.released),
                authority=round(auth, 3),
                rule_action=[round(float(x), 4) for x in rule_action],
                requested_delta=[round(float(x), 4) for x in requested_delta],
                learned_delta=[round(float(x), 4) for x in learned_delta],
                combined_action=[round(float(x), 4) for x in combined],
                executed_action=[round(float(x), 4) for x in executed],
            ))
        return super().step(combined)

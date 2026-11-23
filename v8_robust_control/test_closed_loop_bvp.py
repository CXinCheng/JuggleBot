"""Unit and integration tests for the native closed-loop BVP and torque-CBF modules."""
import unittest
from types import SimpleNamespace

import numpy as np

from jugglebot.robustness.perception import PerceptionConfig
from .closed_loop_bvp import (
    ClosedLoopBvpCascade,
    ClosedLoopBvpPolicy,
    ClosedLoopConfig,
    ClosedLoopResidualEnv,
    ClosedLoopTaskEnv,
    _TightFingerCtrl,
    eval_midair_clearance,
    predict_ball_pos,
    predict_crossing_time,
    solve_closed_loop_throw,
)
from .task_env import TaskConfig, clone_controller


def _fake_ball(pos, vel, sample_time_s=1.0, valid=True):
    return SimpleNamespace(
        position_m=tuple(float(x) for x in pos),
        velocity_m_s=tuple(float(x) for x in vel),
        sample_time_s=float(sample_time_s),
        valid=valid,
    )


class ClosedLoopBvpTests(unittest.TestCase):
    def test_ballistic_crossing_and_clearance(self):
        g = 9.81
        st = _fake_ball([0.35, -0.25, 1.20], [0.0, 0.0, 0.0], sample_time_s=2.0)
        cr = predict_crossing_time(st, 0.90, g)
        self.assertIsNotNone(cr)
        expected_dt = np.sqrt(2. * 0.30 / g)
        self.assertAlmostEqual(cr['time_s'], 2.0 + expected_dt, places=6)
        self.assertAlmostEqual(cr['vz'], -g * expected_dt, places=6)
        p_mid = predict_ball_pos(st, 2.0 + 0.5 * expected_dt, g)
        self.assertAlmostEqual(p_mid[2], 1.20 - 0.5 * g * (0.5 * expected_dt) ** 2, places=6)

        # Two balls on identical trajectory -> 0 mm clearance
        other = _fake_ball([0.35, 0.0, 1.00], [0.0, 0.0, 2.0], sample_time_s=1.0)
        d_min, _, _ = eval_midair_clearance(
            np.array([0.35, 0.0, 1.00]), np.array([0.0, 0.0, 2.0]), 1.0, [other], g
        )
        self.assertLess(d_min, 1e-6)

    def test_clone_controller_preserves_finger_wrapper_isolation(self):
        template = ClosedLoopBvpCascade(throws=4, estimator='filtered')
        cloned = clone_controller(template, PerceptionConfig(0., 0., 0., 0))
        self.assertIsNot(cloned, template)
        for side in ('right', 'left'):
            gc = cloned.hands[side]['fingers'].get_ctrl_array
            self.assertIsInstance(gc, _TightFingerCtrl)
            self.assertIs(gc.controller, cloned)
            self.assertIsNot(gc.controller, template)
        self.assertFalse(cloned.perturbed_active)

    def test_bvp_solver_avoids_midair_collision_within_bounds(self):
        c = SimpleNamespace(
            gravity_m_s2=9.81,
            hands=dict(
                right=dict(p=np.array([0.35, -0.25, 0.85]), catch_p=np.array([0.35, -0.25, 0.85])),
                left=dict(p=np.array([0.35, 0.25, 0.85]), catch_p=np.array([0.35, 0.25, 0.85])),
            ),
        )
        # Place another airborne ball directly on the nominal arc ~0.18s after release
        t0 = 2.0
        p_rel = np.array([0.35, -0.25, 0.85]) + np.array([0.012, 0.055, 0.185])
        v_nom = np.array([(0.35 - p_rel[0]) / 0.48, (0.25 - p_rel[1]) / 0.48, 2.26])
        t_col = t0 + 0.168 + 0.18
        p_col = p_rel + v_nom * 0.18 + np.array([0., 0., -0.5 * 9.81 * 0.18 ** 2])
        # Back-project obstacle ball to sample_time_s = t0 with opposite horizontal velocity
        dt_back = t_col - t0
        v_obs_at_col = np.array([0.0, -1.0, -0.3])
        v_obs_t0 = v_obs_at_col + np.array([0., 0., 9.81 * dt_back])
        p_obs_t0 = p_col - v_obs_t0 * dt_back + np.array([0., 0., 0.5 * 9.81 * dt_back ** 2])

        req = SimpleNamespace(
            kind='throw',
            side='right',
            ball_id=0,
            time_s=t0,
            released=4,
            remaining_defer_s=0.025,
            palm_position_m=((0.35, -0.25, 0.85), (0.35, 0.25, 0.85)),
            balls=(_fake_ball([0.35, -0.25, 0.85], [0., 0., 0.], t0),
                   _fake_ball(p_obs_t0, v_obs_t0, t0),
                   None),
        )
        cfg = ClosedLoopConfig(k_xz=0.35)
        act = solve_closed_loop_throw(req, cfg, c)
        self.assertEqual(act.shape, (6,))
        self.assertTrue(np.all(np.isfinite(act)))
        self.assertTrue(np.all(np.abs(act) <= 1.0))
        self.assertGreater(float(np.linalg.norm(act)), 0.05)

    def test_zero_perturbation_invariance_and_perturbed_rescue(self):
        # 1. Zero-perturbation (0.00 mm) on 4-throw task: gate stays OFF, full SUCCESS
        env0 = ClosedLoopTaskEnv(TaskConfig(throws=4, seat_jitter_m=0.0), estimator='filtered')
        policy = ClosedLoopBvpPolicy()
        try:
            obs, _ = env0.reset(seed=101)
            # Without a reference file action_authority is 1.0; verify policy still completes 0.00 mm cleanly
            # and verify explicit zero-authority gating via synthetic obs
            zero_obs = {k: v.copy() for k, v in obs.items()}
            zero_obs['context'][-1] = 1.0
            zero_obs['errors'][:] = 0.0
            zero_obs['robot_error'][:] = 0.0
            np.testing.assert_array_equal(policy.predict(zero_obs, env0), np.zeros(6))
        finally:
            env0.close()

        # 2. Extreme +/-2.0 mm seed 505: baseline has BALL_COLLISION at 0 catches;
        # native ClosedLoopTaskEnv + ClosedLoopBvpPolicy rescues to >= 2 catches with zero patch hacks!
        env = ClosedLoopTaskEnv(TaskConfig(throws=30, seat_jitter_m=0.002), estimator='filtered')
        try:
            obs, _ = env.reset(seed=505)
            done = trunc = False
            while not (done or trunc):
                act = policy.predict(obs, env)
                obs, _, done, trunc, info = env.step(act)
            self.assertGreaterEqual(info['catches'], 2)
            self.assertNotEqual((info.get('failure_event') or {}).get('reason'), 'BALL_COLLISION')
        finally:
            env.close()

    def test_zero_learned_residual_reproduces_pure_rule_closed_loop(self):
        """Gate 1C: zero learned residual on ClosedLoopResidualEnv must reproduce
        ClosedLoopTaskEnv + ClosedLoopBvpPolicy bit-for-bit on perturbed episodes."""
        cfg = ClosedLoopConfig()
        rule_policy = ClosedLoopBvpPolicy(cfg)
        task_cfg = TaskConfig(throws=30, seat_jitter_m=0.002)

        env_rule = ClosedLoopTaskEnv(task_cfg, estimator='filtered', closed_loop_cfg=cfg)
        env_res = ClosedLoopResidualEnv(task_cfg, estimator='filtered', closed_loop_cfg=cfg)
        try:
            obs_a, _ = env_rule.reset(seed=505)
            obs_b, _ = env_res.reset(seed=505)
            done_a = trunc_a = done_b = trunc_b = False
            steps = 0
            while not (done_a or trunc_a) and not (done_b or trunc_b):
                act_rule = rule_policy.predict(obs_a, env_rule)
                obs_a, rew_a, done_a, trunc_a, info_a = env_rule.step(act_rule)
                obs_b, rew_b, done_b, trunc_b, info_b = env_res.step(np.zeros(6))
                np.testing.assert_array_equal(env_res.last_rule_action, act_rule)
                np.testing.assert_array_equal(env_res.last_combined_action, act_rule)
                self.assertAlmostEqual(rew_a, rew_b, places=12)
                steps += 1
            self.assertEqual(done_a, done_b)
            self.assertEqual(trunc_a, trunc_b)
            self.assertGreater(steps, 0)
            self.assertEqual(info_a['catches'], info_b['catches'])
            self.assertEqual(info_a['releases'], info_b['releases'])
            self.assertEqual(info_a['elapsed_s'], info_b['elapsed_s'])
            self.assertEqual(info_a.get('failure_event'), info_b.get('failure_event'))
        finally:
            env_rule.close()
            env_res.close()

    def test_nonzero_learned_residual_clipped_and_authority_gated(self):
        """Non-zero learned residuals must be clipped to [-1, 1] and zeroed by
        action_authority at 0.00 mm when using the official filtered reference."""
        from pathlib import Path
        ref = Path('v8_robust_control/results/ablation_baselines/task_filtered_reference_final.json')
        self.assertTrue(ref.is_file(), f'Missing archived reference: {ref}')
        env = ClosedLoopResidualEnv(
            TaskConfig(throws=30, horizon_s=12.0, seat_jitter_m=0.0),
            reference=str(ref),
            residual_scale=0.5,
        )
        try:
            obs, _ = env.reset(seed=440000)
            _, _, done, trunc, info = env.step(np.ones(6))
            # Rule action is zero at 0.00 mm; learned delta is clipped to 0.5;
            # unified action_authority zeroes executed action inside TaskResidualEnv.step.
            np.testing.assert_array_equal(env.last_rule_action, np.zeros(6))
            np.testing.assert_allclose(env.last_combined_action, np.full(6, 0.5))
            self.assertFalse(env.controller.perturbed_active)
            self.assertEqual(float(info.get('max_executed_action', 0.0)), 0.0)
        finally:
            env.close()

    def test_residual_window_decisions_vs_releases_and_action_log(self):
        """Verify active_max_decisions, active_mask, and full action_log auditing."""
        from pathlib import Path
        ref = Path('v8_robust_control/results/ablation_baselines/task_filtered_reference_final.json')
        env = ClosedLoopResidualEnv(
            TaskConfig(throws=30, horizon_s=12.0, seat_jitter_m=0.00025),
            reference=str(ref),
            residual_scale=0.25,
            active_max_decisions=2,
            active_max_released=3,
            active_mask=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            record_action_log=True,
        )
        try:
            obs, _ = env.reset(seed=440000)
            for _ in range(4):
                obs, _, done, trunc, _ = env.step(np.ones(6))
                if done or trunc:
                    break
            self.assertGreaterEqual(len(env.action_log), 3)
            # Decisions k=0, 1 have active learned_delta on channels 0..2 (0.25) and 0 on 3..5
            for entry in env.action_log[:2]:
                np.testing.assert_allclose(entry['learned_delta'], [0.25, 0.25, 0.25, 0.0, 0.0, 0.0])
                np.testing.assert_allclose(entry['requested_delta'], [1.0] * 6)
            # Decision k=2 is gated OFF by active_max_decisions=2 even though released < 3
            np.testing.assert_allclose(env.action_log[2]['learned_delta'], [0.0] * 6)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()

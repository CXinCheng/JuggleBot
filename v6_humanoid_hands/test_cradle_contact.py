"""Regression and Contact Physics Unit Tests for Stage v6 LEAP Hand Cradle.

Tests:
1. Palm contact within 1 mm, bounding both gaps and penetration.
2. Centered seating, prediction isolation and release-state reset.
3. Four-digit enclosure and opposing thumb/finger contact normals.
4. Active thumb travel between open and wrapped postures.
5. Thirty complete released-and-caught flights, with pose/contact checks.
6. Fifty complete flights, checking every ball on every integration step.

Usage:
    uv run python -m unittest v6_humanoid_hands.test_cradle_contact -v
    uv run python -u -m v6_humanoid_hands.test_cradle_contact -v --log v6_humanoid_hands/results/validation.txt
"""

import unittest
from types import SimpleNamespace
import numpy as np
import mujoco

from v6_humanoid_hands.juggle import HumanoidJuggler
from jugglebot.humanoid.contact_control import ScoopCurve, ScoopPulse, CartesianPalmMotion, log_rotation
from jugglebot.humanoid.contact_control import SCOOP_ACCELERATION_DURATION_S


class TestLeapHandCradleContact(unittest.TestCase):
    """Verifies that juggling balls rest physically inside the cupped palm with zero floating gap."""

    def setUp(self):
        self.juggler = HumanoidJuggler(pattern="cascade3", n_balls=3)
        self.model = self.juggler.model
        self.data = self.juggler.data

    def test_ball_palm_contact_zero_gap(self):
        """Assert that balls in the holding state physically contact the palm surface (no floating)."""
        from v6_humanoid_hands.verify_g1_hands import verify_v5_conditions, verify_wrist_clearance
        verify_v5_conditions(self.model)
        verify_wrist_clearance(self.model, self.data)
        # Touching a ball must still produce closing preload. At first touch
        # the surface-to-surface segment has zero length; at slight overlap
        # it reverses, so it cannot serve directly as the closing normal.
        fingers = self.juggler.r_arm.fingers
        preload = mujoco.MjData(self.model)
        mujoco.mj_copyData(preload, self.model, self.data)
        ball = self.model.geom('ball0').id
        thumb = self.model.body('rh_th_ds').id
        tips = [g for g in range(self.model.ngeom)
                if self.model.geom_contype[g] and self.model.geom_bodyid[g] == thumb]
        tip = min(tips, key=lambda g: mujoco.mj_geomDistance(self.model, preload, g, ball, .2, None))
        segment = np.zeros(6)
        gap = mujoco.mj_geomDistance(self.model, preload, tip, ball, .2, segment)
        normal = preload.geom_xpos[ball] - segment[:3]
        normal /= np.linalg.norm(normal)
        q = self.juggler.balls[0].qadr
        preload.qpos[q:q+3] -= (gap+.0001)*normal
        mujoco.mj_forward(self.model, preload)
        before = min(mujoco.mj_geomDistance(self.model, preload, g, ball, .2, None) for g in tips)
        self.assertLess(before, 0.)
        correction = fingers.opposition_correction(preload, ball, fingers.get_ctrl_array('cradle'), np.zeros(4))
        self.assertGreater(np.linalg.norm(correction), 1e-6)
        self.assertLessEqual(np.max(np.abs(correction)), .45)
        preload.qpos[self.model.jnt_qposadr[fingers.joint_ids[-4:]]] += .05*correction
        mujoco.mj_forward(self.model, preload)
        after = min(mujoco.mj_geomDistance(self.model, preload, g, ball, .2, None) for g in tips)
        self.assertLess(after, before, 'The thumb preload must close toward the ball at existing contact')
        # The positive wrist brake must stay active while throwing, just like
        # the negative brake. A unit inertia isolates its saturated response.
        controller = self.juggler._cascade
        hand = dict(controller.hands['right'], mode='throw')
        dof = hand['kin'].dof_adr
        preload.qpos[hand['kin'].qpos_adr[5]] = np.deg2rad(35.)
        preload.qvel[dof[5]] = 1.
        response = np.zeros((self.model.nv, 7))
        response[dof] = np.eye(7)
        protected = controller.protect_arm(hand, preload, np.zeros(7),
                                           response, np.zeros(self.model.nv))
        self.assertLess(protected[5], 0.)
        self.assertTrue(np.all(abs(protected) <= hand['force_limits']))
        # A rapid swing well inside the band is already scheduled to reverse;
        # an 80 ms constant-velocity forecast would falsely brake this stroke.
        preload.qpos[hand['kin'].qpos_adr[5]] = 0.
        preload.qvel[dof[5]] = 8.
        protected = controller.protect_arm(hand, preload, np.zeros(7),
                                           response, np.zeros(self.model.nv))
        self.assertEqual(protected[5], 0.)
        # Legacy motion vectors keep zero timing lead. Optional timing shifts
        # the motor reference smoothly, and invalid inputs must be atomic.
        from jugglebot.humanoid.cascade import ContactCascade, MAX_FINGER_RELEASE_LEAD_S
        saved_inputs = dict(forecast=dict(controller.warm), execution=dict(controller.execution_warm))
        try:
            command = [.1, .7, 3., 0., .020]
            controller.set_optimizer_inputs(dict(forecast={}, execution={'throw:right:0': command}))
            with self.assertRaisesRegex(ValueError, 'finger release lead'):
                controller.set_optimizer_inputs(dict(execution={'throw:right:0': command[:4] + [.026]}))
            self.assertEqual(controller.execution_warm['throw:right:0'], command)
            release_hand = dict(controller.hands['right'], stroke=0.,
                                release_start=fingers.get_ctrl_array('cradle'))
            baseline = dict(release_hand, path=SimpleNamespace(finger_release_lead_s=0.))
            unchanged = [array.copy() for array in (self.data.qpos, self.data.qvel, self.data.ctrl)]
            for lead in (-MAX_FINGER_RELEASE_LEAD_S, .020, MAX_FINGER_RELEASE_LEAD_S):
                shifted = dict(release_hand, path=SimpleNamespace(finger_release_lead_s=lead))
                np.testing.assert_array_equal(controller.release_fingers(shifted, 0.), release_hand['release_start'])
                np.testing.assert_allclose(controller.release_fingers(shifted, .065),
                                           controller.release_fingers(baseline, .065+lead), atol=1e-12)
                np.testing.assert_array_equal(controller.release_fingers(shifted, .25), fingers.get_ctrl_array('flat'))
            for before, after in zip(unchanged, (self.data.qpos, self.data.qvel, self.data.ctrl)):
                np.testing.assert_array_equal(before, after)
        finally:
            controller.set_optimizer_inputs(saved_inputs)
        # Keep a real seated ball supported while the initial palm eases into
        # its loaded posture, rather than holding the wrist bent until release.
        startup = ContactCascade(n_balls=1)
        startup.planning = False
        held = startup.hands['right']
        held['throw_time'] = 1.
        while startup.data.time < .12-1e-8:
            startup.step()
        self.assertEqual(held['held'], 0)
        self.assertTrue(startup.ball_contact(0, 'right', palm_only=True))
        wrist = startup.data.qpos[held['kin'].qpos_adr[5]]
        self.assertLess(abs(np.rad2deg(wrist)), 30.)
        # Audit the initialized controller too: a loader-only check misses
        # late motor retuning. A real floor contact must reject the run.
        controller = self.juggler._cascade
        saved = controller.__dict__.copy()
        trial = mujoco.MjData(self.model)
        mujoco.mj_copyData(trial, self.model, self.data)
        try:
            controller.data = trial
            q, _ = controller.balls[0]
            radius = self.model.geom_size[controller.ball_geom_ids[0], 0]
            trial.qpos[q:q+3] = [-.6, -.6, radius-.001]
            mujoco.mj_forward(self.model, trial)
            with self.assertRaisesRegex(RuntimeError, 'DROPPED'):
                controller.check_forbidden_contacts()
            self.assertEqual(controller.failure_event['reason'], 'DROPPED')
            result = controller.run_continuous(throws=1, timeout=0., guard=True)
            self.assertFalse(result['success'])
            self.assertEqual(result['error'], 'DROPPED')
            # A missed catch must also stop before a recovery drives an arm
            # beyond the same physical joint tolerance checked by regression.
            mujoco.mj_copyData(trial, self.model, self.data)
            controller.failure_event = None
            joint = self.model.joint('right_wrist_yaw_joint')
            trial.qpos[joint.qposadr[0]] = joint.range[1]+.01
            mujoco.mj_forward(self.model, trial)
            with self.assertRaisesRegex(RuntimeError, 'POSE_LIMIT'):
                controller.check_pose_limits()
            self.assertEqual(controller.failure_event['part'], 'right_wrist_yaw_joint')
            self.assertEqual(controller.failure_event['quantity'], 'joint_limit_violation_rad')
            # Throwing must not bypass the wrist comfort bound merely because
            # the joint's wider mechanical range still permits the pose.
            for side, angle in (('right', 39.), ('left', -39.)):
                mujoco.mj_copyData(trial, self.model, self.data)
                controller.failure_event = None
                joint = self.model.joint(side+'_wrist_pitch_joint')
                trial.qpos[joint.qposadr[0]] = np.deg2rad(angle)
                mujoco.mj_forward(self.model, trial)
                with self.assertRaisesRegex(RuntimeError, 'POSE_LIMIT'):
                    controller.check_pose_limits()
                self.assertEqual(controller.failure_event['part'], side+'_wrist_pitch_joint')
                self.assertEqual(controller.failure_event['quantity'], 'wrist_pitch_rad')
            mujoco.mj_copyData(trial, self.model, self.data)
            controller.failure_event = None
            ball_q, _ = controller.balls[0]
            trial.qpos[ball_q+2] = 1.31
            mujoco.mj_forward(self.model, trial)
            with self.assertRaisesRegex(RuntimeError, 'POSE_LIMIT'):
                controller.check_pose_limits()
            self.assertEqual(controller.failure_event['quantity'], 'ball_height_m')
            mujoco.mj_copyData(trial, self.model, self.data)
            controller.failure_event = None
            trial.qpos[self.model.joint('right_shoulder_roll_joint').qposadr[0]] = -.65
            mujoco.mj_forward(self.model, trial)
            with self.assertRaisesRegex(RuntimeError, 'POSE_LIMIT'):
                controller.check_pose_limits()
            self.assertEqual(controller.failure_event['quantity'], 'elbow_out_m')
        finally:
            controller.__dict__.clear()
            controller.__dict__.update(saved)
        d = self.data
        m = self.model

        # Step one physics step so contacts are populated
        self.juggler.step()

        # Ball 0 is held in Right hand, Ball 1 is held in Left hand
        for b_id, hand_prefix in [(0, "rh_"), (1, "lh_")]:
            b_name = f"ball{b_id}"
            b_contacts = []
            for i in range(d.ncon):
                c = d.contact[i]
                b1_name = m.body(m.geom_bodyid[c.geom1]).name
                b2_name = m.body(m.geom_bodyid[c.geom2]).name
                if b_name in b1_name or b_name in b2_name:
                    other_body = b2_name if b_name in b1_name else b1_name
                    if other_body == hand_prefix + "palm" and c.efc_address >= 0:
                        b_contacts.append((other_body, c.dist))

            # Must have at least 1 active physical contact with the hand
            self.assertGreater(
                len(b_contacts),
                0,
                f"Ball {b_id} in {hand_prefix} hand has 0 contacts with palm! It is floating in mid-air.",
            )

            # Bound both floating gaps and initial palm penetration to 1 mm.
            min_dist = min(c[1] for c in b_contacts)
            self.assertLessEqual(
                min_dist,
                0.0010,
                f"Ball {b_id} is floating above hand! Clearance: {min_dist*1000:.2f}mm",
            )
            self.assertGreaterEqual(
                min_dist,
                -0.0010,
                f"Ball {b_id} penetrates too deeply into hand: {min_dist*1000:.2f}mm",
            )

    def test_both_hands_ball_seated_in_palm_center(self):
        """Assert that balls in holding state are dead-centered in palm cradle (no wrist or fingertip offsets)."""
        from types import SimpleNamespace
        from v6_humanoid_hands.calibrate import CalibrationCascade
        # A successful retry belongs to the later contact state. Exporting it
        # as the first guess can change a previously verified physical run.
        key = 'throw:right:0'
        original = [0., .8, 3., 0.]
        accepted = dict(kind='plan', side='right', time=.105, score=.01,
                        command=[.1, .7, 3.1, -.02])
        inputs = SimpleNamespace(warm={}, execution_warm={key: original},
                                 events=[dict(kind='plan_rejected', side='right', time=.1), accepted])
        self.assertEqual(CalibrationCascade.candidate_inputs(inputs)['execution'][key], original)
        inputs.events = [accepted]
        self.assertEqual(CalibrationCascade.candidate_inputs(inputs)['execution'][key], accepted['command'])
        inputs.events = [dict(kind='plan_rejected', side='right', time=.1)]
        self.assertEqual(CalibrationCascade.candidate_inputs(inputs)['execution'][key], original)
        d = self.data
        m = self.model
        self.juggler.step()

        # Ball 0 in Right hand, Ball 1 in Left hand
        r_palm = d.site_xpos[self.juggler.r_seat_id]
        l_palm = d.site_xpos[self.juggler.l_seat_id]
        p_b0 = d.qpos[self.juggler.balls[0].qadr : self.juggler.balls[0].qadr + 3]
        p_b1 = d.qpos[self.juggler.balls[1].qadr : self.juggler.balls[1].qadr + 3]

        # Compare with actual palm geometry too. Comparing only with a seat
        # site would pass even if both the ball and site were near the roots.
        for position, prefix in ((p_b0, 'rh_'), (p_b1, 'lh_')):
            palm = m.body(prefix + 'palm').id
            local = d.xmat[palm].reshape(3, 3).T @ (position-d.xpos[palm])
            pad = m.geom(prefix + 'palm_collision_7').id
            self.assertLess(np.linalg.norm(local[:2]-m.geom_pos[pad, :2]), .006,
                            'Ball must lie over the central palm pad, not just its seat site')

        # Right hand ball offset from palm center
        dx_r = abs(p_b0[0] - r_palm[0])
        dy_r = abs(p_b0[1] - r_palm[1])
        self.assertLessEqual(
            dx_r,
            0.002,
            f"Ball 0 is offset from right palm along x by {dx_r*1000:.1f}mm (must be <= 2mm).",
        )
        self.assertLessEqual(
            dy_r,
            0.002,
            f"Ball 0 is offset from right palm along y by {dy_r*1000:.1f}mm (must be <= 2mm).",
        )

        # Left hand ball offset from palm center (prevents the wrist displacement bug)
        dx_l = abs(p_b1[0] - l_palm[0])
        dy_l = abs(p_b1[1] - l_palm[1])
        self.assertLessEqual(
            dx_l,
            0.002,
            f"Ball 1 is offset from left palm along x by {dx_l*1000:.1f}mm (must be <= 2mm; was displaced onto wrist!).",
        )
        self.assertLessEqual(
            dy_l,
            0.002,
            f"Ball 1 is offset from left palm along y by {dy_l*1000:.1f}mm (must be <= 2mm).",
        )

        self._assert_future_throw_uses_actual_origin()
        self._assert_future_reception_uses_actual_origin()
        self._assert_airborne_collision_prediction()
        # Impact loads must not relax the fingers before a real catch has
        # accumulated sustained thumb-and-finger support.
        controller = self.juggler._cascade
        hand = dict(controller.hands['right'])
        hand.update(contact_time=-1., capture_confirmed=False,
                    grip_force=np.full(4, 2.), grip_offsets=np.full(4, -.08))
        forces, offsets = hand['grip_force'].copy(), hand['grip_offsets'].copy()
        self.assertTrue(controller.ball_contact(hand['held'], 'right'))
        controller.update_grip(hand, controller.data.time)
        np.testing.assert_array_equal(hand['grip_force'], forces)
        self.assertTrue(np.all(hand['grip_offsets'] >= offsets),
                        'Unconfirmed support must never relax the grip under impact load')
        self.assertTrue(np.all(hand['grip_offsets'] <= .03))

        # A previous catch must not certify the next incoming ball after release.
        from jugglebot.humanoid.cascade import ContactCascade
        released = ContactCascade(n_balls=1)
        released.planning = False
        releasing_hand = released.hands['right']
        releasing_hand.update(capture_confirmed=True, candidate_ball=0,
                              digit_contact_times={'th': 0.})
        while released.data.time < .5 and released.throw_count == 0:
            released.step()
        self.assertEqual(released.throw_count, 1)
        self.assertIsNone(releasing_hand['held'])
        self.assertFalse(releasing_hand['capture_confirmed'])
        self.assertIsNone(releasing_hand['candidate_ball'])
        self.assertEqual(releasing_hand['digit_contact_times'], {})

        # An infeasible launch seed must leave the live contact state intact
        # and let the controller wait/replan, rather than aborting the run.
        from unittest.mock import patch
        with patch('jugglebot.humanoid.cascade.ThrowPath', side_effect=ValueError('infeasible contact launch')) as trial:
            # Use the still-held left ball in the original controller.
            held = controller.hands['left']['held']
            original = [controller.data.qpos.copy(), controller.data.qvel.copy(),
                        controller.data.ctrl.copy()]
            self.assertIsNone(controller.plan_throw('left'))
            self.assertGreater(trial.call_count, 1)
            self.assertFalse(np.isfinite(controller.last_plan_error))
            self.assertEqual(controller.hands['left']['held'], held)
            for actual, expected in zip((controller.data.qpos, controller.data.qvel,
                                         controller.data.ctrl), original):
                np.testing.assert_array_equal(actual, expected)

    def _assert_future_throw_uses_actual_origin(self):
        # A predictive rollout crossing the other arm's throw must rebuild
        # that stroke at its future pose, without changing the live state.
        c = self.juggler._cascade
        saved = c.__dict__.copy()
        live_data = c.data
        unchanged = [live_data.qpos.copy(), live_data.qvel.copy(), live_data.ctrl.copy()]
        grip_state = {side: (hand['grip_force'].copy(), hand['grip_offsets'].copy())
                      for side, hand in c.hands.items()}
        trial = mujoco.MjData(c.model)
        mujoco.mj_copyData(trial, c.model, live_data)
        try:
            c.data = trial
            c.hands = {side: dict(hand) for side, hand in c.hands.items()}
            c.paths = dict(c.paths)
            c.targets = dict(c.targets)
            c.airborne = dict(c.airborne)
            c.caught_events = set(c.caught_events)
            c.events = []
            c.planning, c.silent = False, True
            hand = c.hands['left']
            forecast_key = f'throw:left:{c.throw_count}'
            motion = np.asarray(c.warm.get(forecast_key, c.commands['left']))[:4]
            c.warm = dict(c.warm, **{forecast_key: np.r_[motion, .020]})
            # Live retry hints must never become the other arm's forecast.
            hand.update(retry_throw_key=f'throw:left:{c.throw_count}',
                        retry_throw_command=np.full(4, 10.))
            hand['throw_time'] = trial.time
            # Waist motion contributes to the palm's velocity, even with
            # unchanged arm joint coordinates.
            for name, speed in (('waist_yaw_joint', .12), ('waist_roll_joint', -.03),
                                ('waist_pitch_joint', .08)):
                trial.qvel[c.model.joint(name).dofadr[0]] = speed
            mujoco.mj_forward(c.model, trial)
            jp = np.zeros((3, c.model.nv))
            jpd = np.zeros_like(jp)
            mujoco.mj_jacSite(c.model, trial, jp, None, hand['seat'])
            mujoco.mj_jacDot(c.model, trial, jpd, None,
                             trial.site_xpos[hand['seat']], int(c.model.site_bodyid[hand['seat']]))
            expected_velocity = jp @ trial.qvel
            expected_acceleration = np.clip(jp @ trial.qacc + jpd @ trial.qvel, -50., 50.)
            old = c.paths['left']
            q0 = trial.qpos[hand['kin'].qpos_adr].copy()
            start = trial.site_xpos[hand['seat']].copy()
            t0 = trial.time
            c.step()
            self.assertEqual(hand['mode'], 'throw')
            self.assertIsNot(hand['path'], old)
            self.assertAlmostEqual(hand['path'].planned_time, t0)
            self.assertEqual(hand['path'].finger_release_lead_s, .020)
            np.testing.assert_array_equal(hand['path'].initial_q, q0)
            np.testing.assert_allclose(hand['path'].sample(0.)[0], start, atol=1e-12)
            np.testing.assert_allclose(hand['path'].sample(0.)[1], expected_velocity, atol=1e-12)
            np.testing.assert_allclose(hand['path'].sample(0.)[2], expected_acceleration, atol=1e-10)
            expected = c.warm.get(f'throw:left:{c.throw_count}', c.commands['left'])
            np.testing.assert_allclose(hand['path'].sample(SCOOP_ACCELERATION_DURATION_S)[1], expected[:3], atol=1e-10)
        finally:
            c.__dict__.clear()
            c.__dict__.update(saved)
        for actual, expected in zip((live_data.qpos, live_data.qvel, live_data.ctrl), unchanged):
            np.testing.assert_array_equal(actual, expected)
        for side, (forces, offsets) in grip_state.items():
            np.testing.assert_array_equal(c.hands[side]['grip_force'], forces)
            np.testing.assert_array_equal(c.hands[side]['grip_offsets'], offsets)

    def _assert_future_reception_uses_actual_origin(self):
        from jugglebot.humanoid.cascade import (
            SUPPORTED_BALL_CENTER_OFFSET_M, RECEIVE_PALM_BACK_TILT, RECEIVE_INWARD_BANK)
        c = self.juggler._cascade
        saved = c.__dict__.copy()
        live_data = c.data
        unchanged = live_data.qpos.copy(), live_data.qvel.copy(), live_data.ctrl.copy()
        trial = mujoco.MjData(c.model)
        mujoco.mj_copyData(trial, c.model, live_data)
        try:
            c.data = trial
            c.hands = {side: dict(hand) for side, hand in c.hands.items()}
            c.paths, c.targets, c.airborne = dict(c.paths), {2: 'right'}, {2: 0}
            c.caught_events, c.events = set(), []
            c.planning, c.silent = False, True
            parameters = np.array([.01, -.012, .008, .025, -.03])
            key = f'return:right:{c.throw_count}'
            c.warm = dict(c.warm, **{key: parameters.tolist()})
            c.execution_warm = dict(c.execution_warm, **{key: [10.]*5})
            hand = c.hands['right']
            hand.update(mode='wait', held=None, reception=None)
            q, v = c.balls[2]
            position, velocity = np.array([.40, -.21, 1.10]), np.array([.03, -.05, -1.2])
            trial.qpos[q:q+3], trial.qvel[v:v+3] = position, velocity
            mujoco.mj_forward(c.model, trial)
            origin = trial.site_xpos[hand['seat']].copy()
            height = hand['catch_p'][2] + SUPPORTED_BALL_CENTER_OFFSET_M
            remaining = (velocity[2] + np.sqrt(velocity[2]**2 +
                         2*c.gravity_m_s2*(position[2]-height))) / c.gravity_m_s2
            nominal = hand['catch_p'].copy()
            nominal[:2] = position[:2] + velocity[:2]*remaining
            c.step()
            path, stop = hand['approach_joint'], hand['reception'][1]
            np.testing.assert_allclose(path.sample(path.t0)[0], origin, atol=1e-12)
            np.testing.assert_allclose(stop.start, nominal+parameters[:3], atol=1e-12)
            expected = np.r_[parameters[3:]+[-RECEIVE_PALM_BACK_TILT, RECEIVE_INWARD_BANK], 1.]
            expected /= np.linalg.norm(expected)
            np.testing.assert_allclose(hand['approach_axis'], expected, atol=1e-12)
        finally:
            c.__dict__.clear()
            c.__dict__.update(saved)
        for actual, expected in zip((live_data.qpos, live_data.qvel, live_data.ctrl), unchanged):
            np.testing.assert_array_equal(actual, expected)

    def _assert_airborne_collision_prediction(self):
        # Two freely flying spheres with equal gravity cross after 0.2 s.
        # Check an actual collision course and a parallel path with 10 mm
        # of sphere-surface clearance,
        # using a private state copy without changing the live simulation.
        c = self.juggler._cascade
        saved = c.__dict__.copy()
        trial = mujoco.MjData(c.model)
        mujoco.mj_copyData(trial, c.model, c.data)
        try:
            c.data = trial
            c.hands = {side: dict(hand, held=None) for side, hand in c.hands.items()}
            c.airborne = {0: 1, 2: 2}
            c.targets = {0: 'left', 2: 'right'}
            for ball, position, velocity in ((0, [.30, -.10, 1.10], [0., .5, 1.]),
                                              (2, [.30, .10, 1.10], [0., -.5, 1.])):
                q, v = c.balls[ball]
                trial.qpos[q:q+3], trial.qvel[v:v+3] = position, velocity
            q, v = c.balls[2]
            self.assertAlmostEqual(c.airborne_clearance(0), 0., places=10)
            separation = 2*c.model.geom_size[c.ball_geom_ids[0], 0] + .01
            trial.qpos[q] += separation
            unchanged = trial.qpos.copy(), trial.qvel.copy(), trial.ctrl.copy()
            self.assertAlmostEqual(c.airborne_clearance(0), separation, places=10)
            for actual, expected in zip((trial.qpos, trial.qvel, trial.ctrl), unchanged):
                np.testing.assert_array_equal(actual, expected)
            for ball in (0, 2):
                _, v = c.balls[ball]
                trial.qvel[v:v+3] += [.1, .2, .3]
            self.assertAlmostEqual(c.airborne_clearance(0), separation, places=10)
            c.hands['right']['held'] = 2
            self.assertGreater(c.airborne_clearance(0), 1.)
        finally:
            c.__dict__.clear()
            c.__dict__.update(saved)

    def test_lateral_caging_funnel(self):
        """Assert that the 4 digits form a concave 3D cradle around the ball."""
        d = self.data
        m = self.model
        mujoco.mj_forward(m, d)

        # Right hand ball 0
        p_b0 = d.qpos[self.juggler.balls[0].qadr : self.juggler.balls[0].qadr + 3]
        # Check right fingertips (distal links) and thumb
        rh_th_pos = d.xpos[m.body("rh_th_ds").id]
        rh_if_pos = d.xpos[m.body("rh_if_ds").id]
        rh_rf_pos = d.xpos[m.body("rh_rf_ds").id]

        # Ball bottom height
        ball_bottom_z = p_b0[2] - m.geom_size[m.geom('ball0').id, 0]

        # Fingertips must rise above ball bottom to provide lateral caging
        self.assertGreater(
            rh_if_pos[2],
            ball_bottom_z,
            "Index fingertip is lower than ball bottom; cannot cradle laterally.",
        )
        self.assertGreater(
            rh_th_pos[2],
            ball_bottom_z,
            "Thumb tip is lower than ball bottom; cannot provide lateral boundary.",
        )

        # Lateral containment: thumb on one side, ring finger on the other side
        y_min = min(rh_th_pos[1], rh_rf_pos[1])
        y_max = max(rh_th_pos[1], rh_rf_pos[1])
        self.assertTrue(
            y_min - 0.05 <= p_b0[1] <= y_max + 0.05,
            f"Ball lateral position {p_b0[1]:.3f} is outside the thumb-to-ring span [{y_min:.3f}, {y_max:.3f}]",
        )

        # A cupped hand must actually surround the sphere. Straight fingers
        # several centimetres away previously passed the fingertip-height check.
        for ball_id, prefix in [(0, "rh_"), (1, "lh_")]:
            ball = m.geom(f"ball{ball_id}").id
            for digit in ("if", "mf", "rf", "th"):
                geoms = [g for g in range(m.ngeom) if m.geom_contype[g]
                         and m.body(m.geom_bodyid[g]).name == prefix + digit + "_ds"]
                gap = min(mujoco.mj_geomDistance(m, d, g, ball, 0.2, None) for g in geoms)
                self.assertLess(gap, 0.005, f"{prefix}{digit} does not wrap the ball")
                self.assertGreater(gap, -0.005, f"{prefix}{digit} penetrates the ball")

        for ball_id, prefix in [(0, 'rh_'), (1, 'lh_')]:
            ball = m.geom(f'ball{ball_id}').id
            seat = m.site('r_ball_seat' if ball_id == 0 else 'l_ball_seat').id
            rotation = d.site_xmat[seat].reshape(3, 3)
            body = m.body(prefix + 'th_ds').id
            geom = next(index for index in range(m.ngeom)
                        if m.geom_contype[index] and m.geom_bodyid[index] == body)
            segment = np.zeros(6)
            mujoco.mj_geomDistance(m, d, geom, ball, .2, segment)
            normal = rotation.T @ (segment[3:] - d.geom_xpos[ball]) / m.geom_size[ball, 0]
            self.assertLess(normal[1], 0., 'Thumb must oppose the forward finger contacts')
            # LEAP's thumb opposes the forward fingers primarily along palm
            # length. Check the actual opposing contact normals, rather than
            # the Shadow thumb's model-specific lateral angle.
            forward_body = m.body(prefix + 'mf_ds').id
            forward_geoms = [g for g in range(m.ngeom)
                             if m.geom_contype[g] and m.geom_bodyid[g] == forward_body]
            forward_geom = min(forward_geoms,
                               key=lambda g: mujoco.mj_geomDistance(m, d, g, ball, .2, None))
            forward_segment = np.zeros(6)
            mujoco.mj_geomDistance(m, d, forward_geom, ball, .2, forward_segment)
            forward_normal = rotation.T @ (forward_segment[3:] - d.geom_xpos[ball]) / m.geom_size[ball, 0]
            self.assertLess(float(normal @ forward_normal), np.cos(np.deg2rad(70.)))

    def test_thumb_active_opposition_and_wrap(self):
        """Assert that thumb actively articulates across palm from flat release to grasp stabilization."""
        d = self.data
        m = self.model

        for side, prefix in [("right", "rh_"), ("left", "lh_")]:
            fingers = self.juggler.r_arm.fingers if side == "right" else self.juggler.l_arm.fingers

            # 1. Flat posture (wide open at release apex)
            fingers.set_posture_qpos(d, "flat")
            mujoco.mj_forward(m, d)
            p_th_flat = d.xpos[m.body(f"{prefix}th_ds").id].copy()

            # 2. Grasp posture (wrapped around ball)
            fingers.set_posture_qpos(d, "grasp")
            mujoco.mj_forward(m, d)
            p_th_grasp = d.xpos[m.body(f"{prefix}th_ds").id].copy()

            # Thumb tip travel distance between flat and grasp must be at least 25mm (visible, active caging)
            travel = float(np.linalg.norm(p_th_grasp - p_th_flat))
            self.assertGreater(
                travel,
                0.025,
                f"{side} thumb is motionless! Tip travel between flat and grasp is only {travel*1000:.1f}mm (must be > 25mm).",
            )

    def _assert_normal_motion(self, j):
        # A loaded palm follows the rising forearm (~28 degrees nominal),
        # rather than forcing the wrist to keep the palm flat throughout.
        self.assertGreater(j.min_throw_palm_up, 0.85)
        self.assertGreater(j.min_throw_fingers_forward, 0.85)
        self.assertLess(j.min_throw_palm_up, np.cos(np.deg2rad(10.)))
        self.assertGreater(min(stroke["rise"] for stroke in j.stroke_metrics), 0.12)
        self.assertGreater(min(stroke["rise"] / max(0.001, stroke["sideways"])
                               for stroke in j.stroke_metrics), 1.4)
        # A cascade needs an outer catch and inner throw. The old global
        # vertical/horizontal ratio rewarded a straight lift and missed scooping.
        for limits in j.palm_ranges.values():
            self.assertGreater(limits['zmax'] - limits['zmin'], .18)
        self.assertLessEqual(j.max_joint_limit_violation, .004)
        self.assertFalse(any(j.model.equality(index).name.startswith('grasp_')
                             for index in range(j.model.neq)))
        self.assertEqual(j.contact_only_checks,
                         3 * round(j.data.time / j.model.opt.timestep))
        catches = [event for event in j.events if event['kind'] == 'catch']
        self.assertTrue(catches)
        for event in catches:
            self.assertGreaterEqual(event['contact_settle_time'], .070 - 1e-9)
            self.assertGreater(event['cushion_depth'], .03)
            self.assertLess(event['touch_palm_vertical_velocity'], -.2)
            self.assertLess(event['settled_relative_speed'], .4 * event['initial_relative_speed'])
            self.assertLess(event['settled_ball_speed'], event['incoming_ball_speed'])
            self.assertIn('th', event['supporting_digits'])
            self.assertGreaterEqual(len(event['supporting_digits']), 2)

        style = j.style_summary()
        self.assertLess(style['wrist_angle_p95_deg'], 43.)
        self.assertLess(style['wrist_angle_rms_deg'], 28.)
        self.assertLess(style['elbow_out_p95_m'], .09)
        self.assertLess(style['max_elbow_out_m'], .10)
        self.assertLess(max(style['torso_range_deg']), 2.)
        # Check the actual wrist joint too: a line to the wrist's distal end
        # understates the pitch angle because it already includes that bend.
        self.assertLess(style['wrist_pitch_p95_deg'], 30.)
        self.assertLess(style['wrist_pitch_max_deg'], 38.)
        self._assert_empty_return_reference(j)


        # Staggered easing must preserve the physical control limits and finish
        # in a relaxed cup, rather than snapping all digits into a flat plate.
        hand = dict(j._cascade.hands['right'])
        hand['stroke'] = 0.
        hand['contact_time'] = -1.
        fingers = hand['fingers']
        hand['closure_start'] = fingers.get_ctrl_array('cradle')
        release = j._cascade.release_fingers
        # A grip already relaxed by real contact forces must not squeeze back
        # to the nominal cradle before opening. Use a nonzero yielding case.
        hand['grip_offsets'] = np.array([-.12, -.15, -.10, -.08])
        yielded = j._cascade.holding_fingers(hand, 0.)
        self.assertGreater(np.max(abs(yielded-fingers.get_ctrl_array('cradle'))), .02)
        hand['release_start'] = yielded.copy()
        np.testing.assert_array_equal(release(hand, 0.), yielded)
        np.testing.assert_array_equal(release(hand, .04), yielded)
        np.testing.assert_allclose(release(hand, .12), fingers.get_ctrl_array('flat'))
        for t in np.linspace(.04, .12, 41):
            controls = release(hand, t)
            bounds = j.model.actuator_ctrlrange[fingers.act_indices]
            self.assertTrue(np.all(controls >= bounds[:, 0] - 1e-8))
            self.assertTrue(np.all(controls <= bounds[:, 1] + 1e-8))
            self.assertLess(np.max(abs(release(hand, t + 1e-6) - release(hand, t - 1e-6))), .001)

        # An actual elliptical arc changes direction, and its analytic
        # derivatives must match finite differences through the braking handoff.
        start = np.array([.3, -.18, .8])
        velocity = np.array([0., .6, 2.4])
        curve = ScoopCurve(.15, start, np.zeros(3), velocity)
        # Angular phase acceleration must cancel the ellipse curvature here.
        self.assertAlmostEqual(curve.sample(.15)[2][2], 0., places=8)
        for t in np.linspace(0., .15, 21):
            position = curve.sample(t)[0]
            ellipse = ((position[1] - start[1]) / curve.a) ** 2 + (1 - (position[2] - start[2]) / curve.b) ** 2
            self.assertAlmostEqual(ellipse, 1., places=8)
        self.assertGreater(curve.sample(.15)[0][1] - start[1], .03)
        pulse = ScoopPulse(start, np.zeros(3), velocity)
        for t in (.05, .12, .17, .22):
            epsilon = 1e-6
            before = pulse.sample(t - epsilon)
            after = pulse.sample(t + epsilon)
            _, speed, acceleration = pulse.sample(t)
            np.testing.assert_allclose((after[0] - before[0]) / (2 * epsilon), speed, atol=1e-5)
            np.testing.assert_allclose((after[1] - before[1]) / (2 * epsilon), acceleration, atol=1e-4)
        for before, after in zip(pulse.sample(SCOOP_ACCELERATION_DURATION_S - 1e-7), pulse.sample(SCOOP_ACCELERATION_DURATION_S + 1e-7)):
            np.testing.assert_allclose(before, after, atol=.003)

    def _assert_empty_return_reference(self, j):
        c = j._cascade
        hand = c.hands['right']
        mujoco.mj_copyData(hand['kin'].data, j.model, j.data)
        for duration in (.17, .30):
            kwargs = dict(kin=hand['kin'], t0=0., duration=duration,
                          start=j.data.site_xpos[hand['seat']].copy(),
                          velocity=c.spatial(hand)[3:].copy(),
                          q0=j.data.qpos[hand['kin'].qpos_adr].copy(),
                          goal=hand['catch_p'], end_velocity=np.array([0., 0., -1.5]),
                          axis=np.array([-.25, 0., 1.]))
            base = CartesianPalmMotion(**kwargs)
            for options in (dict(empty_arc=True), dict(empty_arc=True, arc_peak=.075),
                            dict(tilt_deg=-8., arc_peak=.075)):
                peak = options.get('arc_peak')
                arc = CartesianPalmMotion(**kwargs, **options)
                for t in (0., duration):
                    for actual, expected in zip(arc.orientation(t), base.orientation(t)):
                        np.testing.assert_allclose(actual, expected, atol=1e-7)
                for t in np.linspace(.005, duration-.005, 9):
                    eps = 1e-6
                    rotation, omega, alpha = arc.orientation(t)
                    rm, wm, _ = arc.orientation(t-eps)
                    rp, wp, _ = arc.orientation(t+eps)
                    np.testing.assert_allclose(log_rotation(rp @ rm.T)/(2*eps), omega, atol=1e-5)
                    np.testing.assert_allclose((wp-wm)/(2*eps), alpha, atol=2e-4)
                if peak is not None:
                    for before, after in zip(arc.orientation(peak-1e-7), arc.orientation(peak+1e-7)):
                        np.testing.assert_allclose(before, after, atol=.01)

        # Capture braking must stop the downward hand motion before recovering
        # to the throw seat, with continuous position/velocity/acceleration.
        kwargs['duration'] = .30
        kwargs['velocity'] = np.array([0., 0., -1.5])
        kwargs['goal'] = kwargs['start'].copy()
        kwargs['end_velocity'] = np.zeros(3)
        cushioned = CartesianPalmMotion(**kwargs, cushion_duration=.08)
        for t in (.02, .05, .079, .081, .12, .20):
            eps = 1e-6
            pm, vm, _ = cushioned.sample(t-eps)
            pp, vp, _ = cushioned.sample(t+eps)
            _, velocity, acceleration = cushioned.sample(t)
            np.testing.assert_allclose((pp-pm)/(2*eps), velocity, atol=1e-5)
            np.testing.assert_allclose((vp-vm)/(2*eps), acceleration, atol=1e-4)
        for before, after in zip(cushioned.sample(.08-1e-8), cushioned.sample(.08+1e-8)):
            np.testing.assert_allclose(before, after, atol=.0001)

    def test_cascade3_zero_drop_benchmark(self):
        """Assert that 30 consecutive cascade throws complete with zero ball drops."""
        success = self.juggler.run_benchmark(max_throws=30, timeout=25.0)
        self.assertTrue(
            success,
            f"Benchmark failed! Only {self.juggler.throw_count} throws completed before timeout/drop.",
        )
        self.assertGreaterEqual(self.juggler.catch_count, 28)
        self._assert_normal_motion(self.juggler)
        # Complete the last requested flight too, rather than stopping at its release.
        self.assertTrue(set(range(1, 31)).issubset(self.juggler.caught_events))
        result = self.juggler.benchmark_result
        self.assertEqual(sum(result['releases_by_hand'].values()), result['throws'])
        self.assertEqual(sum(result['catches_by_hand'].values()), result['catches'])
        self.assertEqual(self.juggler.catch_count, result['catches'])
        self.assertEqual(self.juggler.total_catches_with_initial, result['total_catches_with_initial'])
        self.assertEqual(result['catches'] + result['initial_airborne_catches'], result['total_catches_with_initial'])
        self.assertEqual(result['completed_requested_by_throwing_hand'], {'right': 15, 'left': 15})
        self.assertIsNone(result['failure_event'])
        self.assertEqual(self.juggler.ball_contacts, 0)
        # Natural-motion regression: low arcs, no hand raised toward the face,
        # elbows below the shoulders, and no arm/torso interpenetration.
        self.assertLess(self.juggler.max_height, 1.30)
        self.assertLess(self.juggler.max_palm_height, 1.15)
        self.assertGreater(self.juggler.min_shoulder_above_elbow, 0.08)
        self.assertLess(self.juggler.max_self_penetration, 0.004)
        self.assertGreater(self.juggler.minimum_distance, 2*self.model.geom_size[self.model.geom('ball0').id, 0])
        for event in self.juggler.events:
            if event["kind"] == "catch":
                self.assertLessEqual(event["contact_distance"], 1e-5)
                self.assertGreater(event["palm_side"], 0.010)
                self.assertLess(event["tangent_offset"], 0.05)
                self.assertEqual(event['contact_body'], 'rh_palm' if event['side'] == 'right' else 'lh_palm')

    def test_three_balls_continuous_cascade_no_drop(self):
        """Regression test for the 3rd ball drop bug: assert all 3 balls maintain active play.

        Historically, outgoing ball 0 and incoming ball 2 collided in mid-air near t=0.24s,
        causing ball 2 or ball 0 to deflect and hit the floor, reducing the juggle to only 2 balls.
        This test monitors all 3 balls at every physics step until all 50 requested
        flights have been caught, strictly asserting z >= 0.40m at all times.
        """
        j = HumanoidJuggler(pattern="cascade3", n_balls=3)
        max_throws = 50
        timeout = max_throws * 0.6 + 5.0

        step_count = 0
        ball_geoms = {j.model.geom(b.name).id for b in j.balls}
        # A release can precede the previous ball's stable catch confirmation.
        # Monitor through the final requested catch, including its cushioning,
        # instead of checking terminal catch counts at the 50th release.
        while j.data.time < timeout and not j.benchmark_complete(max_throws):
            j.step()
            step_count += 1
            # Check elevation of every ball at every single simulation step
            for b in j.balls:
                z = j.data.qpos[b.qadr + 2]
                self.assertGreater(
                    z,
                    0.40,
                    f"Ball drop detected! {b.name} fell to z={z:.3f}m at t={j.data.time:.3f}s (throw #{j.throw_count})",
                )
            self.assertFalse(any(c.geom1 in ball_geoms and c.geom2 in ball_geoms for c in j.data.contact),
                             f"Ball-ball collision at t={j.data.time:.3f}s")
            self.assertEqual(j.contact_only_checks, 3 * step_count,
                             'All balls must be checked before every physics integration')

        self.assertGreaterEqual(
            j.throw_count,
            max_throws,
            f"Expected at least {max_throws} throws, only completed {j.throw_count}",
        )
        self.assertGreaterEqual(
            j.catch_count,
            max_throws,
            f"Catches too low: {j.catch_count} catches for {j.throw_count} throws",
        )
        self.assertTrue(j.run_benchmark(max_throws=50, timeout=timeout))
        self.assertTrue(set(range(1, 51)).issubset(j.caught_events))
        self._assert_normal_motion(j)
        self.assertLess(j.max_height, 1.30)
        self.assertLess(j.max_palm_height, 1.15)
        self.assertGreater(j.min_shoulder_above_elbow, 0.08)
        self.assertLess(j.max_self_penetration, 0.004)
        releases = [event["ball"] for event in j.events if event["kind"] == "release"]
        for ball_id in range(3):
            self.assertGreaterEqual(releases.count(ball_id), 16, "Every ball must participate in the cascade")


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    from jugglebot.humanoid.logging import tee_output

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log", type=Path, help="Save UTF-8 test output while displaying it in the terminal.")
    options, test_args = parser.parse_known_args()
    argv = [sys.argv[0], *test_args]
    with tee_output(options.log):
        unittest.main(argv=argv)

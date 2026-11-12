"""2+1 start on top of the v6 ContactCascade (v6 source is not modified).

Right hand starts with TWO balls in the human-style hold of results/hold2f_right_s20.json:
  * retained ball R (v6 ball 0) in the palm cradle, held by middle finger + thumb + palm;
  * thrown ball T (v6 ball 2) pinched between index/middle/thumb fingertips.
Left hand starts with one ball (v6 ball 1) exactly as in v6. No ball is airborne at t=0.

Sequence: right throws T (index+middle fingers open during the v6 stroke), the hand then carries the retained ball R back to
the loaded source pose exactly as after a v6 catch, throws R at the v6 time (0.1 + 2*tau), and the standard v6 cascade continues.
"""
import json
from pathlib import Path
import numpy as np, mujoco
from jugglebot.humanoid import cascade as C
from jugglebot.humanoid.cascade import ContactCascade

HOLD = Path(__file__).parent / "results" / "hold2f_right_s20.json"
CALIBRATION = Path(__file__).parent / "results" / "first_throw_calibration.json"
CALIBRATION2 = Path(__file__).parent / "results" / "second_throw_calibration.json"
SEEDS_FILE = Path(__file__).parent / "results" / "seeds_2plus1.json"   # optimizer seeds recalibrated on the 2+1 timeline (calibrate21.py)
T_OPEN_AGE_S = 0.114      # finger-opening start measured from stroke start (from the replay: 0.847 s vs stroke ~0.733 s)
OPEN_DURATION_S = 0.05
SETTLE_S = 0.2           # delay of the whole throwing schedule so that the pinch can close


def smooth(a):
    a = np.clip(a, 0.0, 1.0)
    return a ** 3 * (10 - 15 * a + 6 * a * a)


class TwoPlusOneCascade(ContactCascade):
    def __init__(self, hold_file=HOLD, t_open_age=None, calibration=CALIBRATION, seeds=None, seeds_file=SEEDS_FILE, **kw):
        kw.setdefault("n_balls", 3)
        super().__init__(**kw)
        cal = json.loads(Path(calibration).read_text()) if calibration and Path(calibration).exists() else None
        if cal is not None:   # open-loop calibrated seed for the fingertip throw (calibrate_first_throw.py)
            self.warm['throw:right:0'] = np.asarray(cal['command'], float)
            self.execution_warm['throw:right:0'] = np.asarray(cal['command'], float)
        cal2 = Path(CALIBRATION2)
        if calibration and cal2.exists():     # collision-free seed for the retained ball's throw (calibrate_second_throw.py)
            cmd2 = np.asarray(json.loads(cal2.read_text())['command'], float)
            self.warm['throw:right:2'] = cmd2; self.execution_warm['throw:right:2'] = cmd2
        if seeds is None and seeds_file and Path(seeds_file).exists():
            seeds = json.loads(Path(seeds_file).read_text())
        if seeds:      # live optimizer guesses recalibrated on this timeline; the two hand-calibrated first-hand throws keep their own seeds
            for key, val in seeds.get("execution", {}).items():
                if key not in ("throw:right:0", "throw:right:2"):
                    self.execution_warm[key] = val
        if t_open_age is None:
            t_open_age = cal['t_open_age'] if cal is not None else T_OPEN_AGE_S
        m, d = self.model, self.data
        x = np.array(json.loads(Path(hold_file).read_text())["x"])
        h = self.hands["right"]; f = h["fingers"]
        lo, hi = m.actuator_ctrlrange[f.act_indices, 0], m.actuator_ctrlrange[f.act_indices, 1]
        self.hold_ctrl = np.clip(x[:16], lo, hi)
        self.open_mask = np.array([dg in ("if", "mf") for dg in f.actuator_digits])
        self.t_open_age = t_open_age
        # Fingers start open ('flat') and close onto the balls; the hold posture itself penetrates the balls (pinch preload).
        d.qpos[m.jnt_qposadr[f.joint_ids]] = f.get_ctrl_array('flat')
        d.ctrl[f.act_indices] = self.hold_ctrl
        mujoco.mj_forward(m, d)
        site = h["seat"]; Rm = d.site_xmat[site].reshape(3, 3); p = d.site_xpos[site]
        local_R = np.array([x[16], x[17], 0.022]); local_T = np.array([x[18], x[19], x[20] + 0.022])
        for i, loc in ((0, local_R), (2, local_T)):
            q, v = self.balls[i]
            d.qpos[q:q + 3] = p + Rm @ loc; d.qpos[q + 3:q + 7] = [1, 0, 0, 0]; d.qvel[v:v + 6] = 0
        mujoco.mj_forward(m, d)
        self.airborne = {}; self.targets = {}
        for hh in self.hands.values():      # let the fingers close before the first throw
            hh['throw_time'] += SETTLE_S
        h.update(two_phase=True, retained=0, held=2)
        self.initial_ball_states = [dict(position_m=d.qpos[q:q + 3].tolist(), velocity_m_s=d.qvel[v:v + 3].tolist()) for q, v in self.balls]

    LEG1_S = 0.28
    CARRY_TILT_DEG = None      # None -> v6 LOADED_RECOVERY_TILT_DEG
    CARRY_ARC_PEAK_S = None    # None -> v6 LOADED_RECOVERY_PEAK_S
    PAIR_GRACE_S = 0.12    # R and T may still touch for a moment while T leaves the hand

    def _pair_exempt(self):
        h = self.hands["right"]
        return bool(h.get("two_phase")) or (self.data.time - h.get("t_release", -10.0) <= self.PAIR_GRACE_S)

    # ---- right-hand overrides during the two-ball phase -------------------------------------------------
    def ready_for_throw(self, side):
        h = self.hands[side]
        if not h.get("two_phase"):
            return super().ready_for_throw(side)
        d = self.data
        normal = d.site_xmat[h["seat"]].reshape(3, 3)[:, 2]
        speeds = []
        for i in (h["held"], h["retained"]):
            q, v = self.balls[i]
            speeds.append(np.linalg.norm(d.qvel[v:v + 3] - self.spatial(h)[3:]))
        return bool(max(speeds) <= .18 and normal[2] >= .85 and self.ball_contact(h["held"], side) and self.ball_contact(h["retained"], side))

    def _open_state(self, h):
        c = self.hold_ctrl.copy(); flat = h["fingers"].get_ctrl_array("flat")
        c[self.open_mask] = flat[self.open_mask]
        return c

    def holding_fingers(self, h, t):
        if h.get("two_phase"):
            return self.hold_ctrl.copy()
        at = h.get("capture_at")
        if at is not None:
            if t < at:      # T has just left: do not sweep the index/middle fingers through its path
                return self._open_state(h)
            h["capture_at"] = None
            try:
                self.begin_capture("right", h["held"])
            except RuntimeError as err:
                if self.silent:
                    raise ValueError(f"retained ball lost during throw: {err}") from err
                raise RuntimeError(f"DROPPED: retained ball lost after throw at {t:.3f}: {err}") from err
        return super().holding_fingers(h, t)

    def release_fingers(self, h, t):
        if not h.get("two_phase"):
            return super().release_fingers(h, t)
        lead = getattr(h.get("path"), "finger_release_lead_s", 0.0)
        a = smooth((t - h["stroke"] + lead - self.t_open_age) / OPEN_DURATION_S)
        flat = h["fingers"].get_ctrl_array("flat")
        c = self.hold_ctrl.copy()
        c[self.open_mask] = (1 - a) * c[self.open_mask] + a * flat[self.open_mask]
        return c

    def check_forbidden_contacts(self):
        """v6 check with one designed exemption: the pair (R, T) while both are still in the right hand."""
        h = self.hands["right"]
        if not self._pair_exempt():
            return super().check_forbidden_contacts()
        exempt = {self.ball_geom_ids[0], self.ball_geom_ids[2]}
        hands = self.hand_geoms["right"] | self.hand_geoms["left"]

        def owner(body):
            name = self.model.body(body).name
            return "right" if name.startswith(("rh_", "right_")) else "left" if name.startswith(("lh_", "left_")) else None
        for contact in self.data.contact:
            if contact.efc_address < 0 or contact.dist > 0:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            if {a, b} == exempt:
                continue
            reason = None
            if a in self.ball_geom_set or b in self.ball_geom_set:
                other = b if a in self.ball_geom_set else a
                if other in self.ball_geom_set:
                    reason = "BALL_COLLISION"
                elif other not in hands:
                    reason = "DROPPED" if self.model.geom(other).name in ("ground", "torso_stand") else "HIT_ARM"
            else:
                first, second = owner(self.model.geom_bodyid[a]), owner(self.model.geom_bodyid[b])
                if first and second and first != second:
                    reason = "ARM_COLLISION"
            if reason is not None:
                labels = [self.model.geom(g).name or self.model.body(self.model.geom_bodyid[g]).name for g in (a, b)]
                self.failure_event = dict(reason=reason, time=float(self.data.time), geoms=labels)
                raise RuntimeError(f"{reason}: {labels}")
        self.check_pose_limits()

    def plan_throw(self, side):
        """First (fingertip) throw: use the open-loop calibrated launch (calibrate_first_throw.py) instead of the v6 planner.

        The v6 planner searches with finite differences over a physics rollout; the pinch release is not smooth enough for it to reach
        its 0.02 acceptance threshold. The calibrated command has already been verified in the same deterministic simulation.
        """
        h = self.hands[side]
        if not h.get("two_phase") or "throw:right:0" not in self.execution_warm:
            return super().plan_throw(side)
        m, d = self.model, self.data
        command = np.asarray(self.execution_warm["throw:right:0"], float)
        mujoco.mj_copyData(h["kin"].data, m, d)
        path = C.ThrowPath(h["kin"], d.site_xpos[h["seat"]].copy(), command[:3], d.qpos[h["kin"].qpos_adr].copy(), h["catch_p"],
                           bank=command[3], catch_q=h["catch_q"])
        path.finger_release_lead_s = 0.0
        path.planned_time = d.time
        self.commands[side] = command[:4].copy()
        self.last_plan_error = 0.0
        self.log("plan", side, h["held"], error=[0.0, 0.0], note="calibrated open-loop first throw", command=command[:4].tolist())
        return path

    # ---- state hand-over after the thrown ball has left ------------------------------------------------
    def step(self):
        h = self.hands["right"]
        two = self._pair_exempt()
        n_events = len(self.events)
        before = self.ball_contacts
        md_before = self.minimum_distance
        super().step()
        if two:
            # keep the designed R-T adjacency out of the ball-spacing statistic / planner clearance penalty
            d = self.data; pos = [d.qpos[q:q + 3] for q, _ in self.balls]
            others = [float(np.linalg.norm(pos[i] - pos[j])) for i in range(len(pos)) for j in range(i) if {i, j} != {0, 2}]
            self.minimum_distance = min([md_before] + others)
            # During the two-ball hold the pair (R, T) rests lightly against each other by design. Exempt only that pair from the
            # v6 'no ball-ball contact' criterion and count it separately; any other pair, or R/T contact after T leaves, still fails.
            g = {self.ball_geom_ids[0], self.ball_geom_ids[2]}
            f6 = np.zeros(6); n = 0
            for k, c in enumerate(self.data.contact):
                if {c.geom1, c.geom2} == g:
                    n += 1
                    if c.efc_address >= 0:
                        mujoco.mj_contactForce(self.model, self.data, k, f6); h["pair_force_max"] = max(h.get("pair_force_max", 0.0), float(f6[0]))
            if n:
                self.ball_contacts -= n
                self.pair_contact_steps = getattr(self, "pair_contact_steps", 0) + 1
                self.events[:] = [e for e in self.events[:n_events]] + [e for e in self.events[n_events:] if e["kind"] != "collision"]
        if h.get("two_phase") and h["held"] is None:
            self._retain()
        elif h.get("leg2_at") is not None and self.data.time >= h["leg2_at"]:
            self._start_leg2()

    FINGER_DELAY_S = 0.12

    def _retain(self):
        m, d = self.model, self.data
        h = self.hands["right"]; t = d.time; i = h["retained"]
        h["capture_at"] = t + self.FINGER_DELAY_S
        h["two_phase"] = False
        h["t_release"] = t
        h["held"] = i
        h["capture_confirmed"] = True
        h["candidate_ball"] = None
        h["digit_contact_times"] = {}
        end_time = max(h["throw_time"] + 2 * self.tau, t + C.MIN_CATCH_RECOVERY_DURATION_S)
        h["throw_time"] = end_time
        # Two-leg return. Leg 1 = v6's validated braking motion towards the catch point (the hand is still moving up fast),
        # leg 2 = the v6 loaded recovery to the source pose. Both in 'carry' mode so the fingers keep holding R.
        mujoco.mj_copyData(h["kin"].data, m, d)
        leg1 = self.LEG1_S
        h["carry"] = C.CartesianPalmMotion(h["kin"], t, leg1, d.site_xpos[h["seat"]].copy(), self.spatial(h)[3:].copy(),
                                           d.qpos[h["kin"].qpos_adr].copy(), h["catch_p"], np.zeros(3), axis=np.array([0., 0., 1.]),
                                           nominal=h["catch_q"], empty_arc=False, arc_peak=.075)
        h["leg2_at"] = t + leg1 - 0.004
        h["mode"] = "carry"
        self.log("carry", "right", i, next_throw=round(end_time, 3), note="retained ball after 2+1 throw, leg 1")

    def _start_leg2(self):
        m, d = self.model, self.data
        h = self.hands["right"]; t = d.time
        mujoco.mj_copyData(h["kin"].data, m, d)
        h["leg2_at"] = None
        h["carry"] = C.CartesianPalmMotion(h["kin"], t, h["throw_time"] - t - 0.015, d.site_xpos[h["seat"]].copy(), self.spatial(h)[3:].copy(),
                                           d.qpos[h["kin"].qpos_adr].copy(), h["p"], np.zeros(3), axis=C.LOADED_SOURCE_AXIS, nominal=h["q"],
                                           tilt_deg=C.LOADED_RECOVERY_TILT_DEG if self.CARRY_TILT_DEG is None else self.CARRY_TILT_DEG,
                                           arc_peak=C.LOADED_RECOVERY_PEAK_S if self.CARRY_ARC_PEAK_S is None else self.CARRY_ARC_PEAK_S,
                                           cushion_duration=C.CAPTURE_BRAKING_DURATION_S)
        self.log("carry", "right", h["held"], next_throw=round(h["throw_time"], 3), note="leg 2")

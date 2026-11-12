"""Whole-body 2+1 ENDING on top of TwoPlusOneCascade (v6 source unmodified).

After `stop_after` throws (even: the last one by the left hand) neither hand throws again. The right hand keeps the ball it holds (c) and,
while the last incoming ball (a) is in flight, goes to a fingertip-catch pose: palm orientation = v6 loaded-source axis (10 deg tilt, i.e. the
seat frame of the single-hand optimisation), fingers transition from the v6 cradle to the posture Ka of results/catchtipM_*.json
(ring finger + palm hold c, index/middle/thumb open as a bowl), the hand is positioned so that the fingertip slot meets a (horizontal velocity
matched, vertical hand velocity 0 so the relative speed is the full landing speed), then the fingers close (Ka -> Kb) around the impact time.
Success = both right-hand balls held with small relative speed for HOLD_AFTER_S after the catch, no drop / collision / forbidden contact.
"""
import json
from pathlib import Path
import numpy as np, mujoco
from jugglebot.humanoid import cascade as C
from jugglebot.humanoid.contact_control import CartesianPalmMotion, SmoothStop
from cascade21 import TwoPlusOneCascade, smooth
import search_catch_tip as S

CATCH = Path(__file__).parent / "results" / "catchtipM_az0_el90.json"
T_IMPACT_OPT = 0.086        # impact time in search_catch_tip.run for 2.5 m/s relative speed, 0.25 m start distance
HOLD_AFTER_S = 0.8


class EndingCascade(TwoPlusOneCascade):
    ELBOW_OUT_END_M = 0.12
    _elbow_v6 = C.MAX_ELBOW_OUT_M
    T_START_S = 0.50        # start the maneuver when the last ball is predicted to land within this time
    PREP_S = 0.15           # cradle -> Ka transition time
    ARRIVE_EARLY_S = 0.10   # the palm path ends this long before impact (then holds / follows the stop profile)
    LATERAL_RATIO = 0.0     # hand horizontal velocity / ball horizontal velocity at impact
    VERT_RATIO = 0.0        # hand vertical velocity / ball vertical velocity at impact

    def __init__(self, stop_after=4, catch_file=CATCH, **kw):
        super().__init__(**kw)
        d = json.loads(Path(catch_file).read_text()); x = np.array(d["x"])
        self.Ka, self.Kb = x[:16].copy(), x[16:32].copy()
        self.t_close_rel = float(x[32]) - T_IMPACT_OPT
        self.dur_close = float(np.clip(x[33], 0.04, 0.25))
        self.stop_after = stop_after
        self.stopped = False
        self.end = None
        self.end_result = None

    def _goal_rotation(self):
        normal = C.LOADED_SOURCE_AXIS / np.linalg.norm(C.LOADED_SOURCE_AXIS)
        forward = np.array([1., 0., 0.]) - normal * normal[0]
        forward /= np.linalg.norm(forward)
        return np.column_stack([np.cross(forward, normal), forward, normal])

    def step(self):
        h = self.hands["right"]
        if self.silent:      # private planner rollouts forecast ordinary continued juggling
            super().step()
            return
        if not self.stopped and self.throw_count >= self.stop_after:
            self.stopped = True
        if self.stopped and self.end is None and not h.get("two_phase") and h["held"] is not None:
            self._maybe_start_end()
        if self.end is not None:
            self._end_control()
        # Deviation from v6, ending phase only: the fingertip slot sits ~9 cm behind the palm origin, so the palm is closer to the torso
        # than in a v6 catch and the elbow flares out slightly further. Relax the v6 comfort limit (elbow_out < 0.10 m) to ELBOW_OUT_END_M.
        C.MAX_ELBOW_OUT_M = self.ELBOW_OUT_END_M if self.end is not None else self._elbow_v6
        super().step()
        if self.end is not None:
            self._end_monitor()

    def ready_for_throw(self, side):
        if self.stopped and not self.silent:      # live run: no further throws; v6 keeps re-checking every 20 ms
            return False
        return super().ready_for_throw(side)

    def _incoming(self):
        for i, side in self.targets.items():
            if side == "right":
                return i
        return None

    def _maybe_start_end(self):
        m, d = self.model, self.data
        h = self.hands["right"]
        t = d.time
        a = self._incoming()
        if a is None or h["mode"] not in ("hold", "carry"):
            return
        Rm = self._goal_rotation()
        off = Rm @ S.B_TGT
        q, v = self.balls[a]
        p = d.qpos[q:q + 3].copy()
        vel = d.qvel[v:v + 3].copy()
        g = self.gravity_m_s2
        zb = h["catch_p"][2] + off[2]
        disc = vel[2] ** 2 + 2 * g * (p[2] - zb)
        dt = (vel[2] + np.sqrt(max(0.0, disc))) / g
        if not (0.05 < dt < self.T_START_S):
            return
        land = p + np.r_[vel[:2] * dt, 0.0]
        goal = np.r_[land[:2] - off[:2], h["catch_p"][2]]
        speed = np.r_[vel[:2] * self.LATERAL_RATIO, (vel[2] - g * dt) * self.VERT_RATIO]
        mujoco.mj_copyData(h["kin"].data, m, d)
        path = CartesianPalmMotion(h["kin"], t, dt - self.ARRIVE_EARLY_S, d.site_xpos[h["seat"]].copy(), self.spatial(h)[3:].copy(), d.qpos[h["kin"].qpos_adr].copy(),
                                   goal, speed, axis=C.LOADED_SOURCE_AXIS, nominal=h["catch_q"])
        stop = SmoothStop(t + dt, C.CATCH_CUSHION_DURATION_S, goal, speed, g)
        f = h["fingers"]
        self.end = dict(a=a, c=h["held"], path=path, stop=stop, impact=t + dt, t_start=t, ctrl0=d.ctrl[f.act_indices].copy(), goal=goal, off=off,
                        rel_first=None)
        h["mode"] = "ending"
        h["reception"] = None
        self.log("end_start", "right", a, catch_p=h["catch_p"].round(3).tolist(), p=h["p"].round(3).tolist(), impact=round(t + dt, 3), goal=goal.round(3).tolist(), relspeed=float(np.linalg.norm(vel - speed)))

    def _end_control(self):
        d = self.data
        h = self.hands["right"]
        t = d.time
        e = self.end
        if t <= e["path"].t1:
            self.follow_palm(h, d, e["path"], t, h["q"])
        else:
            pp, vv, aa = e["stop"].sample(t)
            if "q_hold" not in e:        # keep the arm's redundant (elbow) posture reached at the end of the approach
                e["q_hold"] = d.qpos[h["kin"].qpos_adr].copy()
            self.return_servo(h, d, e["q_hold"], position=pp, velocity=vv, acceleration=aa, axis_goal=C.LOADED_SOURCE_AXIS)

    def holding_fingers(self, h, t):
        e = self.end
        if e is None or h["mode"] != "ending":
            return super().holding_fingers(h, t)
        w = smooth((t - e["t_start"]) / self.PREP_S)
        base = (1 - w) * e["ctrl0"] + w * self.Ka
        tc = e["impact"] + self.t_close_rel
        a = smooth((t - tc) / self.dur_close)
        return (1 - a) * base + a * self.Kb

    def _end_monitor(self):
        d = self.data
        h = self.hands["right"]
        e = self.end
        t = d.time
        qa, va = self.balls[e["a"]]
        if int(round(t * 1000)) % 20 == 0:
            mm = self.model
            eo = abs(d.xpos[mm.body("right_elbow_link").id, 1]) - abs(d.xpos[mm.body("right_shoulder_pitch_link").id, 1])
            e.setdefault("trace", []).append((round(float(t), 3), round(float(eo), 4), d.site_xpos[h["seat"]].round(3).tolist(),
                                              d.qpos[qa:qa + 3].round(3).tolist(), round(float(d.qpos[self.balls[e["c"]][0] + 2]), 3)))
        if e["rel_first"] is None and self.ball_contact(e["a"], "right"):
            Rm = d.site_xmat[h["seat"]].reshape(3, 3)
            rel = d.qvel[va:va + 3] - self.spatial(h)[3:]
            e["rel_first"] = dict(t=float(t), rel_world=rel.round(3).tolist(), rel_seat=(Rm.T @ rel).round(3).tolist(), speed=float(np.linalg.norm(rel)),
                                  pos_seat=(Rm.T @ (d.qpos[qa:qa + 3] - d.site_xpos[h["seat"]])).round(3).tolist())
        if t >= e["impact"] + HOLD_AFTER_S and self.end_result is None:
            self.end_result = self._evaluate()

    def _evaluate(self):
        d = self.data
        h = self.hands["right"]
        e = self.end
        Rm = d.site_xmat[h["seat"]].reshape(3, 3)
        out = {}
        for nm, i in (("c", e["c"]), ("a", e["a"])):
            q, v = self.balls[i]
            loc = Rm.T @ (d.qpos[q:q + 3] - d.site_xpos[h["seat"]])
            rel = np.linalg.norm(d.qvel[v:v + 3] - self.spatial(h)[3:])
            out[nm] = dict(local=loc.round(3).tolist(), rel_speed=float(rel), contact=bool(self.ball_contact(i, "right")), height=float(d.qpos[q + 2]))
        left = self.hands["left"]
        b = [i for i in range(len(self.balls)) if i not in (e["c"], e["a"])][0]
        qb, vb = self.balls[b]
        out["b_left"] = dict(contact=bool(self.ball_contact(b, "left")), height=float(d.qpos[qb + 2]), speed=float(np.linalg.norm(d.qvel[vb:vb + 3])),
                             local=[0.0, 0.0, 0.0], rel_speed=0.0)
        ok = all(o["contact"] and o["rel_speed"] < 0.15 and np.linalg.norm(o["local"]) < 0.16 and o["height"] > C.BALL_DROP_HEIGHT_M for o in out.values())
        return dict(ok=bool(ok), balls=out, first_contact=e["rel_first"])

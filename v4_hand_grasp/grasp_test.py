"""E1 (v4): can the LEAP hand on a UR5e hold a juggling-sized ball through throw-level
accelerations, and let it go when the fingers do not hold it?

Hand postures (position targets of the 16 finger joints):
    flat   : all zero, the ball rests on the open palm
    cradle : fingers curled to 0.8 rad (mcp = pip, dip = 0.6 x), thumb th_cmc = 1.2 -- found by
             a small search so the ball, resting on the palm, has walls on the finger, thumb and
             heel sides (LEAP has no little finger: the ring-finger side stays open)
    grasp  : fingers commanded past contact (curl 1.2, th_cmc 1.2, th_mcp 0.5): all three fingers,
             the thumb and the palm touch the ball and the torque-limited servos press on it
             (tighter postures, e.g. th_cmc 1.6-1.8, squeeze the ball out through the open side)

Test: the arm holds the "ready" pose (palm up), the ball settles in the posture, then the hand
moves straight up with a fixed orientation (the joint path is integrated from the 6-D task
[z; 0]) along a smoothstep speed profile -- accelerate to V with peak acceleration A_UP, then
stop with peak deceleration A_DOWN. A deceleration above 1 g makes an unheld ball leave the
palm (that is a throw; here it falls back into the hand); a held ball must stay in contact.
Recorded: THROWN if the ball was out of contact with the hand for more than AIRTIME during the
stroke, else HELD; DROPPED if it reached the ground. Also the largest excursion in the palm
frame while in contact, peak finger servo torque / limit, peak arm joint speed / 180 deg/s and
peak arm actuator force / limit.

Usage (from the repo root):
    uv run python v4_hand_grasp/grasp_test.py
    uv run python v4_hand_grasp/grasp_test.py --images v4_hand_grasp/results   # posture snapshots
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from handsim import HandSim, Q_READY

SETTLE = 1.0  # s in the posture before the stroke
AFTER = 0.6  # s after the stroke
AIRTIME = 0.02  # s without hand contact that counts as having left the hand
BALL_ON_PALM = np.array([-0.046, -0.036, -0.035])  # palm-frame point on the grasping surface (palm centre)


def posture(sim: HandSim, curl: float, th_cmc: float, th_mcp: float = 0.0, th_ipl: float = 0.0,
            pip: float | None = None) -> np.ndarray:
    """Fingers: mcp = curl, pip = pip (default curl), dip = 0.6 pip; thumb joints as given."""
    J = {n: i for i, n in enumerate(sim.hand_joints)}
    q = np.zeros(len(J))
    pip = curl if pip is None else pip
    for f in ("if", "mf", "rf"):
        q[J[f"leap_{f}_mcp"]], q[J[f"leap_{f}_pip"]], q[J[f"leap_{f}_dip"]] = curl, pip, 0.6 * pip
    q[J["leap_th_cmc"]], q[J["leap_th_mcp"]], q[J["leap_th_ipl"]] = th_cmc, th_mcp, th_ipl
    return q


def postures(sim: HandSim) -> dict:
    return {"flat": posture(sim, 0.0, 0.0), "cradle": posture(sim, 0.8, 1.2), "grasp": posture(sim, 1.2, 1.2, 0.5)}


class VerticalStroke:
    """Straight line up with fixed hand orientation: q(s) integrated from dq/ds = J+ [z; 0] at the
    flange (s = height gained, m), timed by a smoothstep speed profile (peak accel 1.5 V / T)."""

    DS = 0.0005  # m, integration step along the line

    def __init__(self, sim: HandSim, t0: float, V: float, a_up: float, a_down: float):
        m = sim.model
        self.t0, self.V = t0, V
        self.T1, self.T2 = 1.5 * V / a_up, 1.5 * V / a_down
        length = V * (self.T1 + self.T2) / 2
        scratch = mujoco.MjData(m)
        scratch.qpos[:] = sim.data.qpos
        site = m.site("flange").id
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        task = np.array([0, 0, 1.0, 0, 0, 0])

        def f(q):
            scratch.qpos[sim.arm_qadr] = q
            mujoco.mj_kinematics(m, scratch)
            mujoco.mj_comPos(m, scratch)
            mujoco.mj_jacSite(m, scratch, jp, jr, site)
            return np.linalg.pinv(np.vstack([jp[:, sim.arm_dof], jr[:, sim.arm_dof]])) @ task

        n = int(np.ceil(length / self.DS)) + 2
        self.q = np.zeros((n, 6))
        self.dq = np.zeros((n, 6))
        self.q[0] = sim.data.qpos[sim.arm_qadr]
        for i in range(n - 1):  # midpoint (RK2)
            k1 = f(self.q[i])
            self.dq[i] = k1
            self.q[i + 1] = self.q[i] + self.DS * f(self.q[i] + 0.5 * self.DS * k1)
        self.dq[-1] = f(self.q[-1])
        self.ddq = np.gradient(self.dq, self.DS, axis=0)

    def _at(self, s):
        x = np.clip(s / self.DS, 0, len(self.q) - 1.000001)
        i = int(x)
        w = x - i
        return [(1 - w) * a[i] + w * a[i + 1] for a in (self.q, self.dq, self.ddq)]

    def sample(self, t):
        tau = t - self.t0
        V, T1, T2 = self.V, self.T1, self.T2
        if tau <= 0:
            s, sd, sdd = 0.0, 0.0, 0.0
        elif tau < T1:
            x = tau / T1
            s, sd, sdd = V * T1 * (x ** 3 - x ** 4 / 2), V * (3 * x ** 2 - 2 * x ** 3), V * (6 * x - 6 * x ** 2) / T1
        elif tau < T1 + T2:
            x = (tau - T1) / T2
            s = V * T1 / 2 + V * T2 * (x - x ** 3 + x ** 4 / 2)
            sd, sdd = V * (1 - 3 * x ** 2 + 2 * x ** 3), V * (-6 * x + 6 * x ** 2) / T2
        else:
            s, sd, sdd = V * (T1 + T2) / 2, 0.0, 0.0
        q, dq, ddq = self._at(s)
        return q, dq * sd, ddq * sd ** 2 + dq * sdd

    @property
    def t_end(self):
        return self.t0 + self.T1 + self.T2


@dataclass
class Result:
    posture: str
    a_up: float
    a_down: float
    outcome: str  # HELD / THROWN (out of contact with the hand for > AIRTIME) / DROPPED (on the ground)
    excursion: float  # m, largest ball displacement in the palm frame while in contact
    finger_torque: float  # peak finger servo torque / limit
    arm_speed: float  # peak arm joint speed / 180 deg/s
    arm_force: float  # peak arm actuator force / limit
    released_at: float = np.nan  # s after the stroke start
    airtime: float = 0.0  # s, longest time without hand contact during the stroke


def settle(sim: HandSim, q_hand: np.ndarray) -> np.ndarray:
    """Ball onto the palm with the hand in the cradle, then (first half of SETTLE) the fingers move
    to q_hand: a ball dropped into an already closed hand would be knocked out."""
    sim.reset(Q_READY, postures(sim)["cradle"])
    sim.place_ball(BALL_ON_PALM + np.array([0, 0, -sim.ball_radius - 0.002]))
    while sim.data.time < SETTLE:
        if sim.data.time >= SETTLE / 2:
            sim.set_hand(q_hand)
        sim.control_arm(Q_READY)
        sim.step()
    return sim.ball_in_palm()


def run(sim: HandSim, name: str, q_hand: np.ndarray, V: float, a_up: float, a_down: float) -> Result:
    m, d = sim.model, sim.data
    rest = settle(sim, q_hand)
    stroke = VerticalStroke(sim, d.time, V, a_up, a_down)
    exc, ft, af, released, airtime, t_lost = 0.0, 0.0, 0.0, np.nan, 0.0, None
    force_limit = m.actuator_forcerange[sim.arm_act, 1]
    while d.time < stroke.t_end + AFTER:
        sim.control_arm(*stroke.sample(d.time))
        sim.step()
        c = sim.ball_contacts()
        if c.ground:
            return Result(name, a_up, a_down, "DROPPED", exc, ft, sim.speed_peak, af, released, airtime)
        if c.hand:
            exc = max(exc, float(np.linalg.norm(sim.ball_in_palm() - rest)))
            t_lost = None
        else:
            t_lost = d.time if t_lost is None else t_lost
            airtime = max(airtime, d.time - t_lost)
            if airtime > AIRTIME and np.isnan(released):
                released = t_lost - stroke.t0
        ft = max(ft, sim.finger_torque_ratio())
        af = max(af, float(np.max(np.abs(d.actuator_force[sim.arm_act]) / force_limit)))
    outcome = "THROWN" if airtime > AIRTIME else "HELD"
    return Result(name, a_up, a_down, outcome, exc, ft, sim.speed_peak, af, released, airtime)


def snapshots(sim: HandSim, out: Path) -> None:
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    ren = mujoco.Renderer(sim.model, 480, 640)
    tiles = []
    for name, q in postures(sim).items():
        settle(sim, q)
        ren.update_scene(sim.data, camera="close")
        tiles.append(ren.render())
    Image.fromarray(np.concatenate(tiles, axis=1)).save(out / "postures.png")
    print(f"saved {out / 'postures.png'} (flat | cradle | grasp)")


def main() -> int:
    p = argparse.ArgumentParser(description="v4 E1: hold / release test of the LEAP hand.")
    p.add_argument("--V", type=float, default=0.9, help="peak upward hand speed (m/s)")
    p.add_argument("--a-up", nargs="+", type=float, default=[3.0], help="peak acceleration (g)")
    p.add_argument("--a-down", nargs="+", type=float, default=[0.5, 1.5, 3.0], help="peak deceleration (g)")
    p.add_argument("--images", type=Path, default=None, help="save posture snapshots to this directory")
    args = p.parse_args()
    sim = HandSim()
    if args.images is not None:
        snapshots(sim, args.images)
    g = 9.81
    print(f"ball: radius {1e3 * sim.ball_radius:.1f} mm, mass {sim.model.body_mass[sim.model.body('ball').id] * 1e3:.0f} g; "
          f"hand mass {sim.model.body_subtreemass[sim.palm]:.3f} kg; V = {args.V} m/s")
    for name, q in postures(sim).items():
        for a_up in args.a_up:
            for a_dn in args.a_down:
                r = run(sim, name, q, args.V, a_up * g, a_dn * g)
                rel = (f", left the hand {1e3 * r.released_at:.0f} ms into the stroke, {1e3 * r.airtime:.0f} ms in the air"
                       if r.outcome == "THROWN" else "")
                print(f"{name:6s} up {a_up:.1f} g / down {a_dn:.1f} g: {r.outcome:7s}{rel} | excursion {1e3 * r.excursion:5.1f} mm"
                      f" | finger torque {r.finger_torque:.2f} x limit | arm speed {r.arm_speed:.2f} | arm force {r.arm_force:.2f}",
                      flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

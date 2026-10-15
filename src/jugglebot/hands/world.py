"""Multi-arm simulation for the UR5e + LEAP stages (v5 on).

World holds a model.load_world() scene and one Arm per placement. An Arm is planned on its own
single-arm model (model.load_arm(side), base at (0, 0, pedestal)); in the world its base stands at
(x, y, pedestal) turned by yaw about z, so a point p of the planning frame is at arm.to_world(p) =
Rz(yaw) p + offset (to_arm / vec_to_arm / vec_to_world for the inverse and for vectors). The arm's controller is v4's (position servos + inverse-dynamics feed-forward,
computed on the single-arm model with the arm's current finger positions).

Contacts are classified per ball: palm / finger / arm link of which arm, ground (or a pedestal),
another ball.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .handsim import ARM_JOINTS, ARM_SPEED_LIMIT, ArmDynamics, HandSim
from .model import ArmPlacement, load_arm, load_world, parked_position


class Arm:
    """One arm of the world: its joints / actuators / hand in the world model, and its own
    single-arm planning model (`plan`, a HandSim used for kinematics, dynamics and rollouts)."""

    def __init__(self, world: "World", placement: ArmPlacement, plan_model: mujoco.MjModel):
        m, p = world.model, placement.prefix
        self.world, self.prefix, self.side = world, p, placement.side
        c, s = np.cos(placement.yaw), np.sin(placement.yaw)
        self.rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # arm frame -> world
        self.offset = np.array([placement.base[0], placement.base[1], 0.0])
        self.plan = HandSim(plan_model, placement.side)
        self.qadr = np.array([m.jnt_qposadr[m.joint(p + j).id] for j in ARM_JOINTS])
        self.dof = np.array([m.jnt_dofadr[m.joint(p + j).id] for j in ARM_JOINTS])
        self.act = np.array([m.actuator(p + j.replace("_joint", "")).id for j in ARM_JOINTS])
        self.hand_joints = [p + n for n in self.plan.hand_joints]  # same order as the planning model
        self.hand_qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.hand_joints])
        self.hand_act = np.array([m.actuator(n + "_act").id for n in self.hand_joints])
        self.hand_lo = m.actuator_ctrlrange[self.hand_act, 0].copy()
        self.hand_hi = m.actuator_ctrlrange[self.hand_act, 1].copy()
        self.kp = m.actuator_gainprm[self.act, 0].copy()
        self.kd = -m.actuator_biasprm[self.act, 2].copy()
        self.force_limit = m.actuator_forcerange[self.act, 1].copy()
        self.dynamics = self.plan.dynamics
        self.palm = m.body(p + "leap_palm").id
        bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(p)}
        self.palm_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.palm}
        self.hand_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and m.body(m.geom_bodyid[g]).name.startswith(p + "leap_")}
        self.link_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and g not in self.hand_geoms}
        # hand-fixed sites placed on the planning model (e.g. set_bowl_sites) -> copy to the world
        for s in range(plan_model.nsite):
            name = plan_model.site(s).name
            if name:
                w = m.site(p + name).id
                m.site_pos[w], m.site_quat[w] = plan_model.site_pos[s], plan_model.site_quat[s]
        self.speed_peak = 0.0
        self.force_peak = 0.0

    def site(self, name: str) -> int:
        return self.world.model.site(self.prefix + name).id

    def to_world(self, p_arm) -> np.ndarray:
        return self.rot @ np.asarray(p_arm, dtype=float) + self.offset

    def to_arm(self, p_world) -> np.ndarray:
        return self.rot.T @ (np.asarray(p_world, dtype=float) - self.offset)

    def vec_to_world(self, v_arm) -> np.ndarray:
        return self.rot @ np.asarray(v_arm, dtype=float)

    def vec_to_arm(self, v_world) -> np.ndarray:
        return self.rot.T @ np.asarray(v_world, dtype=float)

    def q(self) -> np.ndarray:
        return self.world.data.qpos[self.qadr].copy()

    def set_hand(self, targets, teleport: bool = False) -> None:
        d = self.world.data
        targets = np.clip(targets, self.hand_lo, self.hand_hi)
        d.ctrl[self.hand_act] = targets
        if teleport:
            d.qpos[self.hand_qadr] = targets

    def control(self, q, qd=None, qdd=None) -> None:
        d = self.world.data
        qd = np.zeros(6) if qd is None else qd
        qdd = np.zeros(6) if qdd is None else qdd
        tau = self.dynamics(q, qd, qdd, d.qpos[self.hand_qadr])
        d.ctrl[self.act] = q + (self.kd / self.kp) * qd + tau / self.kp

    def reset(self, q, hand) -> None:
        d = self.world.data
        d.qpos[self.qadr] = q
        d.qvel[self.dof] = 0
        d.ctrl[self.act] = q
        self.set_hand(hand, teleport=True)
        self.speed_peak = self.force_peak = 0.0

    def palm_pose(self):
        d = self.world.data
        return d.xpos[self.palm].copy(), d.xmat[self.palm].reshape(3, 3).copy()


@dataclass
class Contacts:
    palm: set = field(default_factory=set)  # prefixes of the arms whose palm touches the ball
    finger: set = field(default_factory=set)
    link: set = field(default_factory=set)
    ground: bool = False
    ball: bool = False

    def hand(self, prefix: str) -> bool:
        return prefix in self.palm or prefix in self.finger


class World:
    def __init__(self, placements: list[ArmPlacement], pedestal: float = 0.4, n_balls: int = 2,
                 plan_models: dict | None = None):
        """plan_models: {prefix: single-arm model}, e.g. with bowl sites set; default load_arm(side)."""
        self.model = m = load_world(placements, pedestal, n_balls)
        self.data = mujoco.MjData(m)
        self.pedestal = pedestal
        plan_models = plan_models or {}
        self.arms = {a.prefix: Arm(self, a, plan_models.get(a.prefix) or load_arm(a.side, pedestal)) for a in placements}
        self.balls = []
        for i in range(n_balls):
            j = m.joint(f"ball{i}_free").id
            self.balls.append((m.jnt_qposadr[j], m.jnt_dofadr[j], m.geom(f"ball{i}").id))
        self.ball_geoms = {b[2] for b in self.balls}
        self.ground_geoms = {g for g in range(m.ngeom) if m.geom(g).name == "ground" or m.geom(g).name.endswith("pedestal")}
        self.ball_radius = float(m.geom_size[self.balls[0][2]][0])
        self.owner = {g: p for p, a in self.arms.items() for g in a.hand_geoms | a.link_geoms}

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for a in self.arms.values():
            a.speed_peak = a.force_peak = 0.0

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        d = self.data
        for a in self.arms.values():
            a.speed_peak = max(a.speed_peak, float(np.max(np.abs(d.qvel[a.dof]))) / ARM_SPEED_LIMIT)
            a.force_peak = max(a.force_peak, float(np.max(np.abs(d.actuator_force[a.act]) / a.force_limit)))

    # ------------------------------------------------------------ balls
    def ball_state(self, i: int):
        qa, da, _ = self.balls[i]
        return self.data.qpos[qa:qa + 3].copy(), self.data.qvel[da:da + 3].copy()

    def set_ball_state(self, i: int, p, v) -> None:
        qa, da, _ = self.balls[i]
        self.data.qpos[qa:qa + 3] = p
        self.data.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
        self.data.qvel[da:da + 3] = v
        self.data.qvel[da + 3:da + 6] = 0

    def park_ball(self, i: int) -> None:
        self.set_ball_state(i, parked_position(i), np.zeros(3))

    def place_in_palm(self, i: int, arm: Arm, p_palm_right) -> None:
        from .model import palm_point
        p, R = arm.palm_pose()
        self.set_ball_state(i, p + R @ palm_point(arm.side, p_palm_right), np.zeros(3))
        self.forward()

    def contacts(self, i: int) -> Contacts:
        d, out, geom = self.data, Contacts(), self.balls[i][2]
        for k in range(d.ncon):
            c = d.contact[k]
            if geom not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == geom else c.geom1
            if other in self.ball_geoms:
                out.ball = True
            elif other in self.ground_geoms:
                out.ground = True
            else:
                for p, a in self.arms.items():
                    if other in a.palm_geoms:
                        out.palm.add(p)
                    elif other in a.hand_geoms:
                        out.finger.add(p)
                    elif other in a.link_geoms:
                        out.link.add(p)
        return out

    def rel_to_site(self, i: int, arm: Arm, site: str = "ball_seat"):
        """Ball i's position and velocity relative to an arm site."""
        m, d = self.model, self.data
        sid = arm.site(site)
        v = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, v, 0)
        p, vb = self.ball_state(i)
        return p - d.site_xpos[sid], vb - v[3:]

    def arm_arm_contact(self) -> bool:
        """Any contact between geoms of two different arms."""
        d = self.data
        for k in range(d.ncon):
            c = d.contact[k]
            o1, o2 = self.owner.get(c.geom1), self.owner.get(c.geom2)
            if o1 and o2 and o1 != o2:
                return True
        return False

"""Model assembly for the UR5e + LEAP stages (from v5 on).

Two kinds of model, both built through MjSpec from assets/ur5e_leap_hands/:

* load_arm(side): ONE arm (ur5e_leap_right.xml, or its mirror image for the left) with its base at
  (0, 0, pedestal), names without prefix -- v4's single-arm model. Planning (kinematics, catch
  planning, throw calibration, inverse dynamics) runs on it, in the arm's own base frame.
* load_world(arms): several arms in one scene, each with a prefix (r_, l_, ...) at its base
  position, on its own pedestal, plus n_balls balls (ball0, ball1, ...). The joint trajectories
  planned on the single-arm models are executed here (the arms are fixed-base chains, so their
  dynamics do not depend on where the base stands or which way it faces about z).

Both apply v4's changes (handsim.load_model) to every LEAP body / geom / actuator ("leap_" in
the name): gravity compensation, finger torque limits, compliant palm / fingertip contacts.

The two arms are exact mirror images of each other across the base's xz plane (mirror_spec):
* right: a MIRRORED UR5e with the LEAP right hand (ur5e_leap_right.xml with only the arm reflected).
  The UR5e's lateral link offsets then point to its right, so the two wrists stay outboard -- with
  the original UR5e on the right (and its mirror on the left) both wrists point inwards and collide
  when the hands are close (cascade3: hands 0.2 m apart). The hand is turned about the flange axis
  so that at Q_READY the palm has v4's pose exactly (_remount_hand).
* left: the mirror image of the right arm (= the original UR5e with the mirrored hand, a left hand).
The same joint angles (arm and fingers) give mirrored poses. A mirrored UR5e is not a product --
an idealisation, like the arms of a humanoid -- but its kinematics, dynamics and limits are the
UR5e's, and both hands have the same reach and timing windows (v5 first used two identical UR5e with
Menagerie's left hand). In the palm frame a point of the right hand maps to MIRROR @ p.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parents[3] / "assets" / "ur5e_leap_hands"
ARM_XML = ASSETS / "ur5e_leap_right.xml"  # v4's arm; both arms are built from it (_arm_spec)
WORLD_XML = ASSETS / "world.xml"

FINGER_TORQUE = 0.5  # N m, estimated XC330-M288-T torque at the LEAP current limit (unverified)
SKIN_SOLREF = (0.006, 1.2)  # time constant (s), damping ratio (> 1: no rebound)
SKIN_SOLIMP = (0.9, 0.95, 0.004, 0.5, 2.0)
SKIN_SOLMIX = 100.0
SKIN_SUFFIXES = ("palm_collision_", "if_tip", "mf_tip", "rf_tip", "th_tip")  # after "leap_"
BALL_RADIUS, BALL_MASS = 0.0425, 0.12  # 85 mm / 120 g (v4)
PEDESTAL_HALF = 0.1  # m, half side of the pedestal box

# right <-> left arm (base frame, and every body frame of the arm and hand)
MIRROR = np.diag([1.0, -1.0, 1.0])
Q_READY = np.array([0.302, -1.649, -2.206, -0.857, -1.571, -1.873])  # v1/v2 "ready": flange up


def palm_point(side: str, p_right) -> np.ndarray:
    """A point given in the RIGHT hand's palm frame, in `side`'s palm frame."""
    p = np.asarray(p_right, dtype=float)
    return p.copy() if side == "right" else MIRROR @ p


def _is_hand(e) -> bool:
    return e.name == "hand_offset" or e.name.startswith("leap_") or (e.parent is not None and e.parent.name.startswith("leap_"))


def mirror_spec(spec: mujoco.MjSpec, hand: bool = True, arm: bool = True) -> mujoco.MjSpec:
    """Reflect a model across its world xz plane, in place (hand / arm = False: leave the LEAP
    hand subtree / the rest as it is). Every frame (p, R) becomes (M p, M R M), a proper rotation,
    so in every body's local coordinates the reflection is M too; meshes get a negative y scale
    (MuJoCo then flips their faces). A hinge's motion maps as M R(a, q) M = R(-M a, q): with the
    axis -M a the joint angle stays the same."""
    def pos(p):
        return [p[0], -p[1], p[2]]

    def quat(q):
        return [q[0], -q[1], q[2], -q[3]]

    def chosen(e):
        return hand if _is_hand(e) else arm

    for b in spec.bodies:
        if b.name != "world" and chosen(b):
            b.pos, b.quat, b.ipos, b.iquat = pos(b.pos), quat(b.quat), pos(b.ipos), quat(b.iquat)
    for e in (*spec.frames, *spec.geoms, *spec.sites, *spec.cameras):
        if chosen(e):
            e.pos, e.quat = pos(e.pos), quat(e.quat)
    for j in spec.joints:
        if chosen(j):
            j.pos, j.axis = pos(j.pos), list(-MIRROR @ np.asarray(j.axis))
    for mesh in spec.meshes:
        if hand if mesh.name.startswith("leap_") else arm:
            mesh.scale = pos(mesh.scale)
    return spec


def _rot(quat) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    return R.reshape(3, 3)


def _remount_hand(spec: mujoco.MjSpec, H: np.ndarray, h: np.ndarray) -> None:
    """After mirror_spec(hand=False): move the hand (frame hand_offset, in hand_mount coordinates)
    so that the palm is where the original model had it for the same joint angles, given the
    original hand_mount pose (H, h) at those angles. The mirrored hand_mount is (M H M, M h); for a
    flange in the base's xz plane (Q_READY) this turns the hand about the flange axis only."""
    f = spec.frame("hand_offset")
    Hm, hm = MIRROR @ H @ MIRROR, MIRROR @ h
    R = Hm.T @ H @ _rot(f.quat)
    f.pos = list(Hm.T @ (h + H @ np.asarray(f.pos) - hm))
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    f.quat = list(q)


def _arm_spec(side: str) -> mujoco.MjSpec:
    """The right arm (mirrored UR5e, LEAP right hand) or its mirror image (see the module doc)."""
    m = mujoco.MjSpec.from_file(str(ARM_XML)).compile()
    d = mujoco.MjData(m)
    d.qpos[:6] = Q_READY
    mujoco.mj_kinematics(m, d)
    flange = m.site("flange").id  # at the hand_mount origin, same axes
    spec = mujoco.MjSpec.from_file(str(ARM_XML))
    mirror_spec(spec, hand=False)
    _remount_hand(spec, d.site_xmat[flange].reshape(3, 3).copy(), d.site_xpos[flange].copy())
    return spec if side == "right" else mirror_spec(spec)


def _tweak(spec: mujoco.MjSpec, finger_torque: float, skin: bool) -> None:
    for body in spec.bodies:
        if "leap_" in body.name:
            body.gravcomp = 1.0
    for act in spec.actuators:
        if "leap_" in act.name:
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-finger_torque, finger_torque]
    if skin:
        for geom in spec.geoms:
            if "leap_" in geom.name and geom.name.split("leap_", 1)[1].startswith(SKIN_SUFFIXES):
                geom.solref = list(SKIN_SOLREF)
                geom.solimp = list(SKIN_SOLIMP)
                geom.solmix = SKIN_SOLMIX


def _add_ball(spec: mujoco.MjSpec, name: str, pos, material: str | None = None) -> None:
    body = spec.worldbody.add_body(name=name, pos=list(pos))
    body.add_freejoint(name=f"{name}_free")
    kw = dict(name=name, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS, 0, 0], condim=6,
              friction=[0.8, 0.005, 0.0005], mass=BALL_MASS, solref=[0.003, 0.5], solimp=[0.95, 0.99, 0.001, 0.5, 2])
    if material:
        kw["material"] = material
    body.add_geom(**kw)


def _add_pedestal(spec: mujoco.MjSpec, name: str, xy, height: float) -> None:
    spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[PEDESTAL_HALF, PEDESTAL_HALF, height / 2],
                            pos=[xy[0], xy[1], height / 2], material="pedestal")


def parked_position(i: int) -> list:
    """Where ball i waits on the floor, behind the robots and out of the cameras' view."""
    return [-2.5, -2.5 + 0.2 * i, BALL_RADIUS]


def load_arm(side: str = "right", pedestal: float = 0.4, n_balls: int = 1, finger_torque: float = FINGER_TORQUE,
             skin: bool = True) -> mujoco.MjModel:
    """Single-arm model (names as in v4: shoulder_pan_joint, leap_palm, ball_seat, ..., balls ball,
    ball_1, ...) with the base on a pedestal of the given height."""
    spec = mujoco.MjSpec.from_file(str(WORLD_XML))
    arm = _arm_spec(side)
    frame = spec.worldbody.add_frame(pos=[0.0, 0.0, pedestal])
    frame.attach_body(arm.body("base"), "", "")
    if pedestal > 0:
        _add_pedestal(spec, "pedestal", (0.0, 0.0), pedestal)
    for i in range(n_balls):
        _add_ball(spec, "ball" if i == 0 else f"ball_{i}", [1.5, 0.0, BALL_RADIUS] if i == 0 else parked_position(i),
                  f"ball{i % 4}")
    _tweak(spec, finger_torque, skin)
    return spec.compile()


@dataclass
class ArmPlacement:
    prefix: str  # e.g. "r_"
    side: str  # "right" / "left": which hand
    base: tuple  # (x, y) of the base; it stands on a pedestal
    yaw: float = 0.0  # rad about z: the base faces (cos yaw, sin yaw)


def load_world(arms: list[ArmPlacement], pedestal: float = 0.4, n_balls: int = 2,
               finger_torque: float = FINGER_TORQUE, skin: bool = True) -> mujoco.MjModel:
    """Several arms on pedestals and n_balls balls (ball0 ... , parked) in one scene."""
    spec = mujoco.MjSpec.from_file(str(WORLD_XML))
    for a in arms:
        arm = _arm_spec(a.side)
        frame = spec.worldbody.add_frame(pos=[a.base[0], a.base[1], pedestal],
                                         quat=[np.cos(a.yaw / 2), 0.0, 0.0, np.sin(a.yaw / 2)])
        frame.attach_body(arm.body("base"), a.prefix, "")
        if pedestal > 0:
            _add_pedestal(spec, f"{a.prefix}pedestal", a.base, pedestal)
    for i in range(n_balls):
        _add_ball(spec, f"ball{i}", parked_position(i), f"ball{i % 4}")
    _tweak(spec, finger_torque, skin)
    return spec.compile()

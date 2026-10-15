"""Model assembly for the UR5e + LEAP stages (from v5 on).

Two kinds of model, both built through MjSpec from assets/ur5e_leap_hands/:

* load_arm(side): ONE arm (ur5e_leap_right.xml or ur5e_leap_left.xml) with its base at
  (0, 0, pedestal), names without prefix -- v4's single-arm model. Planning (kinematics, catch
  planning, throw calibration, inverse dynamics) runs on it, in the arm's own base frame.
* load_world(arms): several arms in one scene, each with a prefix (r_, l_, ...) at its base
  position, on its own pedestal, plus n_balls balls (ball0, ball1, ...). The joint trajectories
  planned on the single-arm models are executed here (the arms are fixed-base chains, so their
  dynamics do not depend on where the base stands or which way it faces about z).

Both apply v4's changes (handsim.load_model) to every LEAP body / geom / actuator ("leap_" in
the name): gravity compensation, finger torque limits, compliant palm / fingertip contacts.

The left hand is Menagerie's left_hand.xml mounted as the mirror image of the right hand across
the hand_mount plane y = 0 (ur5e_leap_left.xml); in its palm frame a point of the right hand
maps to MIRROR @ p + MIRROR_T (fingertips agree to 2 mm for equal flexion angles; the abduction
joints rot / th_axl change sign).
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parents[3] / "assets" / "ur5e_leap_hands"
ARM_XML = {"right": ASSETS / "ur5e_leap_right.xml", "left": ASSETS / "ur5e_leap_left.xml"}
WORLD_XML = ASSETS / "world.xml"

FINGER_TORQUE = 0.5  # N m, estimated XC330-M288-T torque at the LEAP current limit (unverified)
SKIN_SOLREF = (0.006, 1.2)  # time constant (s), damping ratio (> 1: no rebound)
SKIN_SOLIMP = (0.9, 0.95, 0.004, 0.5, 2.0)
SKIN_SOLMIX = 100.0
SKIN_SUFFIXES = ("palm_collision_", "if_tip", "mf_tip", "rf_tip", "th_tip")  # after "leap_"
BALL_RADIUS, BALL_MASS = 0.0425, 0.12  # 85 mm / 120 g (v4)
PEDESTAL_HALF = 0.1  # m, half side of the pedestal box

# right palm frame -> left palm frame
MIRROR = np.diag([1.0, -1.0, 1.0])
MIRROR_T = np.array([-0.0759, -0.0908, 0.026])
# abduction joints change sign between the hands (order of the 16 LEAP joints)
LEAP_JOINTS = ("if_mcp", "if_rot", "if_pip", "if_dip", "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
               "rf_mcp", "rf_rot", "rf_pip", "rf_dip", "th_cmc", "th_axl", "th_mcp", "th_ipl")


def palm_point(side: str, p_right) -> np.ndarray:
    """A point given in the RIGHT hand's palm frame, in `side`'s palm frame."""
    p = np.asarray(p_right, dtype=float)
    return p.copy() if side == "right" else MIRROR @ p + MIRROR_T


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
    arm = mujoco.MjSpec.from_file(str(ARM_XML[side]))
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
        arm = mujoco.MjSpec.from_file(str(ARM_XML[a.side]))
        frame = spec.worldbody.add_frame(pos=[a.base[0], a.base[1], pedestal],
                                         quat=[np.cos(a.yaw / 2), 0.0, 0.0, np.sin(a.yaw / 2)])
        frame.attach_body(arm.body("base"), a.prefix, "")
        if pedestal > 0:
            _add_pedestal(spec, f"{a.prefix}pedestal", a.base, pedestal)
    for i in range(n_balls):
        _add_ball(spec, f"ball{i}", parked_position(i), f"ball{i % 4}")
    _tweak(spec, finger_torque, skin)
    return spec.compile()

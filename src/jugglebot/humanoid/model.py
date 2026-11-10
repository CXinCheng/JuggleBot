"""Model assembly for Unitree G1 Upper Body + full-size LEAP Hand (Stage v6).

Builds an upper-body bimanual manipulation humanoid:
- Unitree G1 (Menagerie official model): 7-DOF arms (pitch, roll, yaw, elbow, roll, pitch, yaw).
- Native sagittal symmetry of the G1 arms; official left/right hands calibrated independently.
- Upper-body only on a pedestal stand (legs removed for clean bimanual juggling focus).
- Official left/right LEAP hands: four digits, 16 joints each, original dimensions and inertia.
- Bright studio scene with original LEAP materials, shared with v5.
- Hand-fixed sites: flange, ball_seat, catch_point.
"""

from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSETS = PROJECT_ROOT / "assets"
G1_XML = ASSETS / "unitree_g1" / "g1.xml"
LEAP_RIGHT_XML = ASSETS / "leap_hand" / "right_hand.xml"
LEAP_LEFT_XML = ASSETS / "leap_hand" / "left_hand.xml"
WORLD_XML = ASSETS / "ur5e_leap_hands" / "world.xml"

# 7-DOF arm joint names for Unitree G1
ARM_JOINTS_RIGHT = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
ARM_JOINTS_LEFT = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

# Physical and contact constants
DEFAULT_STAND_HEIGHT_M = 0.75  # Calibrated pedestal height; alternate heights are constructor arguments.
BALL_SEAT_CENTER_OFFSET_M = 0.022  # Nominal seat-site to ball-center offset, metres.
PALM_SURFACE_CLEARANCE_M = .0003  # Rounded palm mesh projects beyond its nominal surface.
FINGER_TORQUE = 0.5  # N m, estimated simulation limit; not hardware-identified.
# Simulation motor/contact calibration, not hardware-identified parameters.
LEAP_SEAT_SURFACE = {
    "right": (-.046, -.036, -.0340),
    "left": (-.1218, -.0545, -.0078),
}
# Keep the wrist shell below the palm plane; the full-size LEAP body needs
# a raised bracket rather than the former downward offset from the wrist.
LEAP_SEAT_IN_WRIST_M = (.135, 0., .070)
# A small wrist wedge keeps the full-size palm level with less wrist flexion.
# Its added height also keeps the G1 shell below the tilted palm surface.
LEAP_MOUNT_PITCH_DEG = -5.0
ADAPTER_MASS_KG = .045
ADAPTER_RADIUS_M = .008
SKIN_SOLREF = (0.006, 1.2)
SKIN_SOLIMP = (0.9, 0.95, 0.004, 0.5, 2.0)
SKIN_SOLMIX = 100.0
# Same physical pads as v5; rigid finger links retain upstream contacts.
SKIN_SUFFIXES = ("palm_collision_", "if_tip", "mf_tip", "rf_tip", "th_tip")
BALL_RADIUS, BALL_MASS = 0.0425, 0.12  # 85 mm / 120 g, same as v5.

# Nominal ready configurations for G1 arms (exact mirror symmetry across sagittal plane):
# [pitch, roll, yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
# Symmetric (same sign): pitch, elbow, wrist_pitch
# Antisymmetric (opposite sign): roll, yaw, wrist_roll, wrist_yaw
# Full-size hands: level seat sites at x=.32 m, y=+/- .18 m, z=.85 m.
Q_READY_ARM_RIGHT = np.array([.05761839, -.04238861, -.09887297, .06056416, .04838773, -.11139992, .11754670])
Q_READY_ARM_LEFT = Q_READY_ARM_RIGHT * np.array([1., -1., -1., 1., -1., 1., -1.])
Q_READY_ARM = Q_READY_ARM_RIGHT  # default alias


def _tweak_hands(spec: mujoco.MjSpec, finger_torque: float = FINGER_TORQUE, skin: bool = True) -> None:
    """Apply gravity compensation, torque limits, and compliant skin to hand bodies."""
    for body in spec.bodies:
        if any(prefix in body.name for prefix in ["rh_", "lh_"]):
            body.gravcomp = 1.0
            if skin:
                for geom in body.geoms:
                    suffix = geom.name.removeprefix("rh_").removeprefix("lh_")
                    if not suffix.startswith(SKIN_SUFFIXES):
                        continue
                    geom.solref = list(SKIN_SOLREF)
                    geom.solimp = list(SKIN_SOLIMP)
                    geom.solmix = SKIN_SOLMIX
    for act in spec.actuators:
        if any(prefix in act.name for prefix in ["rh_", "lh_"]):
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-finger_torque, finger_torque]


def _add_ball(spec: mujoco.MjSpec, name: str, pos, material: str | None = None) -> None:
    body = spec.worldbody.add_body(name=name, pos=list(pos))
    body.add_freejoint(name=f"{name}_free")
    kw = dict(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[BALL_RADIUS, 0, 0],
        condim=6,
        friction=[0.8, 0.005, 0.0005],
        mass=BALL_MASS,
        solref=[0.003, 0.5],
        solimp=[0.95, 0.99, 0.001, 0.5, 2],
    )
    if material:
        kw["material"] = material
    body.add_geom(**kw)


def load_g1_dual_arm(
    n_balls: int = 3,
    upper_body_only: bool = True,
    torso_stand_height: float = DEFAULT_STAND_HEIGHT_M,
    use_world_scene: bool = True,
    finger_torque: float = FINGER_TORQUE,
    skin: bool = True,
    hand_color: tuple[float, float, float, float] | None = None,
    tip_color: tuple[float, float, float, float] | None = None,
) -> mujoco.MjModel:
    """Build G1 with unscaled official LEAP hands and rigid adapter brackets.

    Args:
        n_balls: Number of juggling balls to add.
        upper_body_only: If True, removes legs and mounts pelvis to a pedestal stand.
        torso_stand_height: Height of torso pedestal base (meters).
        use_world_scene: If True, embeds robot into bright studio environment with skybox and ground grid.
        finger_torque: Motor torque limit for hand joints (N m).
        skin: Enable soft contact skin parameters on fingertips and palm.
        hand_color: Optional chassis RGBA override; None preserves upstream LEAP materials.
        tip_color: Optional fingertip RGBA override; None preserves upstream LEAP materials.
    """
    g1_spec = mujoco.MjSpec.from_file(str(G1_XML))
    g1_spec.modelname = "g1_leap"
    g1_spec.option.impratio = 100.0
    g1_spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

    # Fix floating base joint and keyframes
    for j in list(g1_spec.joints):
        if j.name == "floating_base_joint":
            g1_spec.delete(j)
    for k in list(g1_spec.keys):
        g1_spec.delete(k)

    # Upper-body only: cleanly delete hip links (and entire leg subtrees)
    if upper_body_only:
        g1_spec.delete(g1_spec.body("left_hip_pitch_link"))
        g1_spec.delete(g1_spec.body("right_hip_pitch_link"))

    # Remove rubber hands
    for g in list(g1_spec.geoms):
        if g.meshname and "rubber_hand" in g.meshname:
            g1_spec.delete(g)

    # Recolour robot torso / materials
    for mat in g1_spec.materials:
        if mat.name == "black":
            mat.rgba = [.85, .85, .88, 1.]
            mat.specular = 0.6
            mat.shininess = 0.8

    # Original hand geometry, mass and inertia are retained. The demo root pose
    # in each upstream XML is replaced only by a rigid wrist adapter transform.
    # These brackets are simulation mounting designs, not manufactured hardware.
    for side, prefix, source in (("right", "rh_", LEAP_RIGHT_XML), ("left", "lh_", LEAP_LEFT_XML)):
        hand = mujoco.MjSpec.from_file(str(source))
        palm = hand.body("palm")
        # Left upstream palm coordinates have a different CAD origin.
        center = np.array(LEAP_SEAT_SURFACE[side])
        # Site +y points along the fingers (+palm x), +z out of the palm (-palm z).
        site_rotation = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
        site_quat = np.empty(4)
        mujoco.mju_mat2Quat(site_quat, site_rotation.ravel())
        # Preserve native hand geometry; adapt its rigid wrist flange instead.
        angle = np.deg2rad(LEAP_MOUNT_PITCH_DEG)
        pitch = np.array([[np.cos(angle), 0., np.sin(angle)],
                          [0., 1., 0.],
                          [-np.sin(angle), 0., np.cos(angle)]])
        mount_rotation = pitch @ np.diag([1., -1., -1.])
        # Seat site is 22 mm below the initialized ball centre.
        seat = center + [0., 0., -BALL_RADIUS - PALM_SURFACE_CLEARANCE_M + BALL_SEAT_CENTER_OFFSET_M]
        # Forward extension clears the G1 wrist housing without scaling.
        seat_in_wrist = np.array(LEAP_SEAT_IN_WRIST_M)
        palm.pos = seat_in_wrist - mount_rotation @ seat
        mount_quat = np.empty(4)
        mujoco.mju_mat2Quat(mount_quat, mount_rotation.ravel())
        palm.quat = mount_quat
        palm.add_site(name="ball_seat", pos=seat, quat=site_quat, size=[.005, 0, 0], rgba=[1,1,0,.5])
        palm.add_site(name="catch_point", pos=seat + [0,0,-BALL_SEAT_CENTER_OFFSET_M], quat=site_quat, size=[.005,0,0], rgba=[0,1,0,.5])
        for material in hand.materials:
            if material.name == "black" and hand_color is not None:
                material.rgba = list(hand_color)
            elif material.name == "white" and tip_color is not None:
                material.rgba = list(tip_color)
        wrist = g1_spec.body(f"{side}_wrist_yaw_link")
        adapter = wrist.add_body(name=f"{side}_leap_adapter")
        back = seat_in_wrist + mount_rotation @ np.array([0., 0., .070])
        adapter.add_geom(name=f"{side}_leap_adapter_collision", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                         fromto=[.025,0,0,*back], size=[ADAPTER_RADIUS_M,0,0], mass=ADAPTER_MASS_KG, rgba=[.25,.28,.32,1])
        adapter.add_site(name=("r_" if side == "right" else "l_") + "flange", size=[.005,0,0])
        adapter.add_frame().attach_body(palm, prefix, "")
        # Public TCP names are independent of the hand body's namespace.
        for suffix in ("ball_seat", "catch_point"):
            g1_spec.site(prefix + suffix).name = ("r_" if side == "right" else "l_") + suffix

    if use_world_scene:
        root_spec = mujoco.MjSpec.from_file(str(WORLD_XML))
        root_spec.modelname = "g1_leap_world"
        root_spec.option.impratio = 100.0
        root_spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

        # Add cylindrical pedestal stand for torso
        stand_radius = 0.11
        root_spec.worldbody.add_geom(
            name="torso_stand",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[stand_radius, torso_stand_height / 2, 0],
            pos=[0.0, 0.0, torso_stand_height / 2],
            material="pedestal",
        )

        pelvis = g1_spec.body("pelvis")
        pelvis.pos = [0.0, 0.0, 0.0]  # attach flush with stand top
        frame = root_spec.worldbody.add_frame(pos=[0.0, 0.0, torso_stand_height])
        frame.attach_body(pelvis, "", "")

        # Configure humanoid studio cameras
        for c in list(root_spec.cameras):
            root_spec.delete(c)

        # Configure humanoid studio cameras with cinematic close-up framing
        c_persp = root_spec.worldbody.add_camera(name="three_quarter", pos=[1.00, -0.68, torso_stand_height + 0.28])
        c_persp.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
        c_persp.targetbody = "torso_link"

        c_front = root_spec.worldbody.add_camera(name="front", pos=[1.15, 0.0, torso_stand_height + 0.20])
        c_front.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
        c_front.targetbody = "torso_link"

        spec = root_spec
    else:
        g1_spec.body("pelvis").pos = [0.0, 0.0, torso_stand_height]
        spec = g1_spec

    # Park uninitialized balls outside the work area. The controller seats them
    # using the actual ready-pose sites, rather than old world-space coordinates.
    ball_colors = ["ball0", "ball1", "ball2", "ball3"]
    for i in range(n_balls):
        pos = [-2.5, -2.5 + 0.2*i, BALL_RADIUS]
        mat = ball_colors[i % len(ball_colors)] if use_world_scene else None
        _add_ball(spec, f"ball{i}", pos, material=mat)

    _tweak_hands(spec, finger_torque, skin)
    return spec.compile()


def load_g1_scene(n_balls: int = 3, upper_body_only: bool = True) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Convenience helper returning compiled MjModel and initialized MjData."""
    model = load_g1_dual_arm(n_balls=n_balls, upper_body_only=upper_body_only)
    data = mujoco.MjData(model)
    return model, data

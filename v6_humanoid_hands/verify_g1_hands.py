"""Verification and interactive inspection for Stage v6: Unitree G1 Upper-Body + LEAP Hand.

Usage:
    uv run python v6_humanoid_hands/verify_g1_hands.py           # Headless numeric checks
    uv run python v6_humanoid_hands/verify_g1_hands.py --viewer  # Interactive 3D viewer
    uv run python -u v6_humanoid_hands/verify_g1_hands.py --log v6_humanoid_hands/results/assembly.txt
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import mujoco

from jugglebot.humanoid.model import (
    ARM_JOINTS_RIGHT,
    ARM_JOINTS_LEFT,
    Q_READY_ARM_RIGHT,
    Q_READY_ARM_LEFT,
    load_g1_dual_arm,
)
from jugglebot.humanoid.kinematics import HumanoidArmKinematics
from jugglebot.humanoid.logging import tee_output


def verify_assembly():
    print("[1/4] Compiling Upper-Body Unitree G1 with full-size LEAP Hands in world scene...")
    m = load_g1_dual_arm(
        n_balls=3,
        upper_body_only=True,
        torso_stand_height=0.75,
        use_world_scene=True,
    )
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    assert m.nbody == 58, f"Expected 58 bodies (G1, adapters, LEAP hands, balls), got {m.nbody}"
    assert m.njnt == 52, f"Expected 52 joints (17 G1 + 32 fingers + 3 balls), got {m.njnt}"
    assert m.nu == 49, f"Expected 49 actuators (17 G1 + 32 fingers), got {m.nu}"
    print(f"      OK: Compiled successfully! (Bodies: {m.nbody}, Joints: {m.njnt}, Actuators: {m.nu})")

    # The real left/right CAD models have different origins and inertias.
    # Verify that assembly preserves each upstream hand's physical properties.
    from jugglebot.humanoid.model import LEAP_RIGHT_XML, LEAP_LEFT_XML
    for prefix, path in (("rh_", LEAP_RIGHT_XML), ("lh_", LEAP_LEFT_XML)):
        upstream = mujoco.MjModel.from_xml_path(str(path))
        for i in range(1, upstream.nbody):
            attached = m.body(prefix + upstream.body(i).name).id
            np.testing.assert_allclose(m.body_mass[attached], upstream.body_mass[i], rtol=0, atol=1e-12)
            np.testing.assert_allclose(m.body_inertia[attached], upstream.body_inertia[i], rtol=0, atol=1e-12)
            np.testing.assert_allclose(m.body_ipos[attached], upstream.body_ipos[i], rtol=0, atol=1e-12)
        for i in range(upstream.njnt):
            attached = m.joint(prefix + upstream.joint(i).name).id
            np.testing.assert_array_equal(m.jnt_range[attached], upstream.jnt_range[i])
            np.testing.assert_array_equal(m.jnt_axis[attached], upstream.jnt_axis[i])
        for i in range(upstream.ngeom):
            attached = m.geom(prefix + upstream.geom(i).name).id
            np.testing.assert_allclose(m.geom_size[attached], upstream.geom_size[i], rtol=0, atol=1e-12)
            np.testing.assert_allclose(m.geom_pos[attached], upstream.geom_pos[i], rtol=0, atol=1e-12)
        for i in range(upstream.nmat):
            attached = m.material(prefix + upstream.material(i).name).id
            np.testing.assert_array_equal(m.mat_rgba[attached], upstream.mat_rgba[i])
    assert m.neq == 0, "No hand-ball or other equality constraints are expected"
    print("      OK: Original geometry, masses, inertias and joint axes/limits preserved; no equalities.")
    return m, d


def verify_7dof_kinematics(m):
    print("[2/4] Verifying 7-DOF Redundant Kinematics (FK / DLS IK with nullspace projection)...")
    for side in ["right", "left"]:
        kin = HumanoidArmKinematics(m, side=side)
        q_nom = Q_READY_ARM_RIGHT if side == "right" else Q_READY_ARM_LEFT
        p_nom, a_nom = kin.fk(q_nom)

        # Test IK at nominal pose
        ik_res = kin.solve(p_nom, a_nom, q_nom)
        assert ik_res.success, f"{side} 7-DOF IK failed at nominal pose!"
        assert ik_res.pos_err < 1e-5, f"{side} 7-DOF IK position error too large: {ik_res.pos_err}"

        # Test IK reachable target perturbation (2cm displacement)
        p_target = p_nom + np.array([0.02, 0.01 if side == "left" else -0.01, 0.01])
        ik_target = kin.solve(p_target, a_nom, q_nom)
        assert ik_target.success, f"{side} 7-DOF IK failed at perturbed target!"
        print(f"      OK: {side.capitalize()} 7-DOF Arm IK passed (pos_err: {ik_target.pos_err:.2e} m, {ik_target.iters} iters).")


def verify_hand_postures(m, d):
    print("[3/4] Setting Ready Postures for G1 Dual Arms & LEAP Hands...")
    for i, name in enumerate(ARM_JOINTS_RIGHT):
        j_id = m.joint(name).id
        d.qpos[m.jnt_qposadr[j_id]] = Q_READY_ARM_RIGHT[i]

    for i, name in enumerate(ARM_JOINTS_LEFT):
        j_id = m.joint(name).id
        d.qpos[m.jnt_qposadr[j_id]] = Q_READY_ARM_LEFT[i]

    from jugglebot.humanoid.throwing import LeapFingers
    for side in ("right", "left"):
        LeapFingers(m, side).set_posture_qpos(d, "cradle")
    for i in range(m.nu):
        joint = m.actuator_trnid[i, 0]
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            d.ctrl[i] = d.qpos[m.jnt_qposadr[joint]]
    mujoco.mj_forward(m, d)
    # Validate the actual controller ready pose, not arbitrary CAD zero angles.
    from jugglebot.humanoid.cascade import ContactCascade
    controller = ContactCascade(n_balls=2)
    for side, hand in controller.hands.items():
        normal = controller.data.site_xmat[hand['seat']].reshape(3,3)[:,2]
        assert normal[2] > .999, f"{side}: ready palm is not level"
        assert controller.ball_contact(hand['held'], side, palm_only=True)
    for contact in controller.data.contact:
        assert contact.dist > -.004, f"Ready pose penetration: {contact.dist}"
    print("      OK: Level ready palms, real initial ball contact, no deep penetration.")
    return controller.model, controller.data


def verify_v5_conditions(model):
    """Compare effective runtime settings with v5, without changing its files.

    Model geometry, hand posture and G1 arm actuators intentionally differ.
    Checking an initialized controller catches motor overrides after assembly.
    """
    from jugglebot.hands.model import load_arm
    reference = load_arm('right')
    for name in dir(reference.opt):
        if name.startswith('_'):
            continue
        value = getattr(reference.opt, name)
        if isinstance(value, (int, float, np.ndarray)):
            np.testing.assert_array_equal(getattr(model.opt, name), value,
                                          err_msg=f'v5/v6 physics option: {name}')
    for prefix in ('rh_', 'lh_'):
        for i in range(reference.nu):
            name = reference.actuator(i).name
            if not name.startswith('leap_'):
                continue
            attached = model.actuator(prefix + name.removeprefix('leap_')).id
            for field in ('actuator_gainprm', 'actuator_biasprm', 'actuator_gear',
                          'actuator_ctrlrange', 'actuator_forcerange',
                          'actuator_ctrllimited', 'actuator_forcelimited',
                          'actuator_dyntype', 'actuator_dynprm', 'actuator_gaintype', 'actuator_biastype'):
                np.testing.assert_array_equal(getattr(model, field)[attached], getattr(reference, field)[i],
                                              err_msg=f'{prefix}{name}: {field}')
            ja, jb = reference.actuator_trnid[i, 0], model.actuator_trnid[attached, 0]
            for field in ('dof_damping', 'dof_frictionloss', 'dof_armature'):
                np.testing.assert_array_equal(getattr(model, field)[model.jnt_dofadr[jb]],
                                              getattr(reference, field)[reference.jnt_dofadr[ja]],
                                              err_msg=f'{prefix}{name}: {field}')
        for i in range(reference.nbody):
            name = reference.body(i).name
            if name.startswith('leap_'):
                j = model.body(prefix + name.removeprefix('leap_')).id
                np.testing.assert_array_equal(model.body_gravcomp[j], reference.body_gravcomp[i])
        for i in range(reference.ngeom):
            name = reference.geom(i).name
            if not name.startswith('leap_'):
                continue
            j = model.geom(prefix + name.removeprefix('leap_')).id
            for field in ('geom_friction', 'geom_solref', 'geom_solimp', 'geom_solmix',
                          'geom_condim', 'geom_margin', 'geom_gap', 'geom_contype', 'geom_conaffinity'):
                np.testing.assert_array_equal(getattr(model, field)[j], getattr(reference, field)[i],
                                              err_msg=f'{prefix}{name}: {field}')
    for name in (model.geom(i).name for i in range(model.ngeom)
                 if model.geom(i).name.startswith('ball') and model.geom(i).name[4:].isdigit()):
        j, i = model.geom(name).id, reference.geom('ball').id
        for field in ('geom_size', 'geom_friction', 'geom_solref', 'geom_solimp', 'geom_condim', 'geom_solmix', 'geom_margin', 'geom_gap'):
            np.testing.assert_array_equal(getattr(model, field)[j], getattr(reference, field)[i],
                                          err_msg=f'{name}: {field}')
        np.testing.assert_array_equal(model.body_mass[model.geom_bodyid[j]],
                                      reference.body_mass[reference.geom_bodyid[i]])
        np.testing.assert_array_equal(model.body_inertia[model.geom_bodyid[j]],
                                      reference.body_inertia[reference.geom_bodyid[i]])
    print("      OK: Runtime physics, LEAP gains/damping/limits, gravity compensation and all hand contacts match v5.")


def verify_wrist_clearance(model, data):
    """The wrist housing must sit below the actual palm contact surface."""
    for side, prefix in (("right", "rh_"), ("left", "lh_")):
        seat = model.site(("r_" if side == "right" else "l_") + "ball_seat").id
        normal = data.site_xmat[seat].reshape(3, 3)[:, 2]
        pad = model.geom(prefix + "palm_collision_7").id
        pad_top = (data.geom_xpos[pad] @ normal +
                   np.abs(data.geom_xmat[pad].reshape(3, 3).T @ normal) @ model.geom_size[pad])
        wrist = model.body(side + "_wrist_yaw_link").id
        shell_top = -np.inf
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != wrist or not model.geom_contype[geom]:
                continue
            rotation = data.geom_xmat[geom].reshape(3, 3)
            bounds = model.geom_aabb[geom]
            top = ((data.geom_xpos[geom] + rotation @ bounds[:3]) @ normal +
                   np.abs(rotation.T @ normal) @ bounds[3:])
            shell_top = max(shell_top, top)
        assert np.isfinite(shell_top), f"{side}: missing wrist collision geometry"
        assert pad_top-shell_top > .005, f"{side}: wrist shell protrudes into palm catch plane"


def launch_viewer(m, d):
    print("\n[Viewer] Launching MuJoCo 3D viewer for Upper-Body Unitree G1 + LEAP Hands...")
    print("      [提示] 当前脚本为底层装配与 7-DOF 运动学静态校验工具（非杂耍运行环路）。")
    print("      [提示] 运行三球抛接请使用 juggle.py --pattern cascade3。")
    try:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(m, d) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.95]
            viewer.cam.distance = 1.8
            viewer.cam.azimuth, viewer.cam.elevation = 215.0, -12.0
            print(">>> 3D Viewer 运行中。按空格键可暂停/继续，关闭窗口即可退出。<<<")
            while viewer.is_running():
                mujoco.mj_step(m, d)
                viewer.sync()
                time.sleep(0.002)
    except Exception as e:
        print(f"Viewer notice: {e}")


def run_verification(viewer=False):
    print("==================================================")
    print("    JuggleBot Stage v6: Upper-Body G1 + LEAP    ")
    print("==================================================")
    m, d = verify_assembly()
    verify_7dof_kinematics(m)
    m, d = verify_hand_postures(m, d)
    verify_wrist_clearance(m, d)
    print("[4/4] Comparing effective runtime simulation and hand settings with v5...")
    verify_v5_conditions(m)
    print("\n>>> ALL STAGE v6 VERIFICATION CHECKS PASSED! <<<\n")

    if viewer:
        launch_viewer(m, d)


def main():
    p = argparse.ArgumentParser(description="Verify Stage v6: Upper-Body Unitree G1 + LEAP Hands.")
    p.add_argument("--viewer", action="store_true", help="Launch interactive 3D viewer.")
    p.add_argument("--log", type=Path,
                   help="Replace a UTF-8 log while also displaying output in the terminal.")
    args = p.parse_args()
    with tee_output(args.log):
        try:
            run_verification(viewer=args.viewer)
        except Exception:
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

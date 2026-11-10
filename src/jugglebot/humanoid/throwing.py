"""Full-size LEAP finger control through sixteen limited position motors."""
from typing import Literal
import mujoco
import numpy as np
from .model import BALL_SEAT_CENTER_OFFSET_M

DIGITS = ("th", "if", "mf", "rf")
DIGIT_CLOSURE_ONSET_S = (0., .004, .008, .012)
DIGIT_CLOSURE_DURATION_S = (.055, .060, .064, .068)
DIGIT_RELEASE_ONSET_S = (.045, .061, .057, .053)
WRAP_IK_DAMPING_M = .006
MAX_WRAP_DISPLACEMENT_M = .025
MAX_WRAP_CORRECTION_RAD = .55
OPPOSITION_CONTACT_INDENT_M = .0015
MAX_OPPOSITION_CORRECTION_RAD = .45
LEAP_JOINTS = tuple(f"{digit}_{joint}" for digit in ("if", "mf", "rf")
                    for joint in ("mcp", "rot", "pip", "dip")) + ("th_cmc", "th_axl", "th_mcp", "th_ipl")

def get_juggling_postures(side="right"):
    """Initial geometric calibration, refined by contact regression."""
    def posture(mcp, pip, dip, thumb):
        values = {f"{d}_{j}_act": v for d in ("if", "mf", "rf")
                  for j, v in (("mcp", mcp), ("rot", 0.), ("pip", pip), ("dip", dip))}
        values.update({f"th_{j}_act": v for j,v in zip(("cmc", "axl", "mcp", "ipl"), thumb)})
        return values
    closed = {joint+"_act": value for joint,value in
              zip(LEAP_JOINTS, np.asarray(CRADLE_JOINT_TARGETS_RAD[side]).ravel())}
    relaxed = posture(.18, .2, .1, (.1, 0., 0., 0.))
    # An open bowl admits the descending sphere before contact closure. A
    # scaled wrapped posture leaves hooked tips across the incoming path.
    catch = posture(1.1, .3, .18, (1.2, 0., 0., 0.))
    return dict(flat=relaxed, catch=catch, cradle=closed, grasp=closed.copy(), cushion=closed.copy())

# Offline geometric contact calibration for the centrally seated 85 mm ball.
# Ordinary fingers were fitted with a DIP/PIP flexion synergy to avoid hooks.
# Rows: index (mcp, rot, pip, dip), middle, ring, thumb (cmc, axl, mcp, ipl).
CRADLE_JOINT_TARGETS_RAD = {
    "right": (
        (0.8419160308777981, 0.5061727938136594, 1.007626014782847, 0.6054171100329617),
        (1.0217343678488784, 0.1772651322475864, 0.8552812866059739, 0.5080957314864654),
        (0.5338015066865409, 0.74999999, 1.4332056412651721, 0.8424475481853465),
        (1.0932707790724496, 0.4595004988260653, 0.26061124361157745, 1.2769918600871688),
    ),
    "left": (
        (0.9432451318580348, -0.39565058747849935, 0.9741182888823712, 0.5907274875428256),
        (0.47148020318266487, -0.1624684266095583, 1.2545605447772863, 0.7428761766678634),
        (0.9482591551002214, 0.24650682849738587, 1.0309479165862827, 0.6218392579042078),
        (1.1623048128254592, -0.34899999, 0.11010680278601753, 1.4438733331323825),
    ),
}

class LeapFingers:
    """Four digit force groups; missing actuators or invalid targets fail early."""
    digits = DIGITS
    def __init__(self, model: mujoco.MjModel, side: Literal["right", "left"]="right", postures=None):
        if side not in ("right", "left"):
            raise ValueError("hand side must be right or left")
        self.model, self.side = model, side
        self.prefix = "rh_" if side == "right" else "lh_"
        self.suffixes = tuple(j + "_act" for j in LEAP_JOINTS)
        self.act_indices = np.array([model.actuator(self.prefix+s).id for s in self.suffixes])
        self.joint_ids = np.array([model.joint(self.prefix+j).id for j in LEAP_JOINTS])
        self.actuator_digits = tuple(j[:2] for j in LEAP_JOINTS)
        self.closure_onset = np.array([dict(zip(DIGITS,DIGIT_CLOSURE_ONSET_S))[d] for d in self.actuator_digits])
        self.closure_duration = np.array([dict(zip(DIGITS,DIGIT_CLOSURE_DURATION_S))[d] for d in self.actuator_digits])
        self.release_onset = np.array([dict(zip(DIGITS,DIGIT_RELEASE_ONSET_S))[d] for d in self.actuator_digits])
        self.grip_map = np.zeros((len(LEAP_JOINTS),len(DIGITS)))
        for row,j in enumerate(LEAP_JOINTS):
            if j.endswith(("mcp","pip","dip")) or j == "th_ipl":
                self.grip_map[row,DIGITS.index(j[:2])] = 1.
        self.postures = {name: values.copy() for name,values in
                         (get_juggling_postures(side) if postures is None else postures).items()}
        bounds=model.actuator_ctrlrange[self.act_indices]
        for name in self.postures:
            values=self.get_ctrl_array(name)
            if np.any(values < bounds[:,0]) or np.any(values > bounds[:,1]):
                raise ValueError(f"{side} {name} posture exceeds actuator limits")
    def get_ctrl_array(self, posture_name):
        return np.array([self.postures[posture_name][s] for s in self.suffixes])
    def apply_posture(self, data, posture_name):
        data.ctrl[self.act_indices] = self.get_ctrl_array(posture_name)
    def set_posture_qpos(self, data, posture_name):
        """Initialization/tests only: runtime control writes actuator targets."""
        self.apply_posture(data, posture_name)
        data.qpos[self.model.jnt_qposadr[self.joint_ids]] = self.get_ctrl_array(posture_name)

    def calibrate_wrapping(self, data, seat, ball_geom):
        """Linearized fingertip IK around the physically seated ready posture.

        Store each digit's local Jacobian, retaining the real joint axes and
        independent left/right kinematics. Runtime corrections affect motor
        targets only; no ball state, equality, or external force is involved.
        """
        m = self.model
        calibration = mujoco.MjData(m)
        mujoco.mj_copyData(calibration,m,data)
        ball_joint=m.body_jntadr[m.geom_bodyid[ball_geom]]
        ball_q=m.jnt_qposadr[ball_joint]
        rotation=data.site_xmat[seat].reshape(3,3)
        calibration.qpos[ball_q:ball_q+3]=data.site_xpos[seat]+BALL_SEAT_CENTER_OFFSET_M*rotation[:,2]
        mujoco.mj_forward(m,calibration)
        data=calibration
        rotation = data.site_xmat[seat].reshape(3,3)
        self.wrap_map = np.zeros((len(self.act_indices),3))
        for digit in self.digits:
            body=m.body(self.prefix+digit+'_ds').id
            geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g]==body]
            geom=min(geoms,key=lambda g:mujoco.mj_geomDistance(m,data,g,ball_geom,.2,None))
            segment=np.zeros(6)
            mujoco.mj_geomDistance(m,data,geom,ball_geom,.2,segment)
            jac=np.zeros((3,m.nv))
            mujoco.mj_jac(m,data,jac,None,segment[:3],body)
            rows=np.array([i for i,d in enumerate(self.actuator_digits) if d==digit])
            dofs=m.jnt_dofadr[self.joint_ids[rows]]
            local=rotation.T@jac[:,dofs]
            self.wrap_map[rows]=local.T@np.linalg.inv(local@local.T+WRAP_IK_DAMPING_M**2*np.eye(3))

    def wrapping_correction(self, local_ball_error):
        """Bound finger IK to 0.55 rad per joint around the calibrated cup."""
        error=np.clip(local_ball_error,-MAX_WRAP_DISPLACEMENT_M,MAX_WRAP_DISPLACEMENT_M)
        return np.clip(self.wrap_map@error,-MAX_WRAP_CORRECTION_RAD,MAX_WRAP_CORRECTION_RAD)

    def opposition_correction(self, data, ball_geom, target, previous):
        """Close the thumb gap and establish bounded contact preload.

        Used only for a delayed, unconfirmed catch. Retaining the bounded
        correction prevents the thumb from reopening when contact is reached.
        """
        m = self.model
        body = m.body(self.prefix+'th_ds').id
        geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] == body]
        geom = min(geoms, key=lambda g: mujoco.mj_geomDistance(m, data, g, ball_geom, .1, None))
        segment = np.zeros(6)
        gap = mujoco.mj_geomDistance(m, data, geom, ball_geom, .1, segment)
        if not -OPPOSITION_CONTACT_INDENT_M < gap < .04:
            return previous
        # Closest surface endpoints coincide at first touch and reverse order
        # on penetration. The sphere centre gives a consistent closing normal.
        normal = data.geom_xpos[ball_geom] - segment[:3]
        normal /= max(np.linalg.norm(normal), 1e-9)
        rows = np.array([i for i, digit in enumerate(self.actuator_digits) if digit == 'th'])
        jac = np.zeros((3, m.nv))
        mujoco.mj_jac(m, data, jac, None, segment[:3], body)
        gradient = normal @ jac[:, m.jnt_dofadr[self.joint_ids[rows]]]
        delta = (gap + OPPOSITION_CONTACT_INDENT_M)*gradient/(gradient@gradient + WRAP_IK_DAMPING_M**2)
        current = data.qpos[m.jnt_qposadr[self.joint_ids[rows]]]
        return np.clip(current + np.clip(delta, -.08, .08) - target[rows],
                       -MAX_OPPOSITION_CORRECTION_RAD, MAX_OPPOSITION_CORRECTION_RAD)

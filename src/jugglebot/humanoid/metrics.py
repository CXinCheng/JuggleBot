"""Passive execution-demand measurements; no hardware certification implied."""
import mujoco
import numpy as np


class ExecutionMetrics:
    def __init__(self, model):
        self.model = model
        self.ids = np.array([
            i for i in range(model.nu)
            if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
            and model.jnt_type[model.actuator_trnid[i, 0]] == mujoco.mjtJoint.mjJNT_HINGE
        ], dtype=int)
        self.joints = model.actuator_trnid[self.ids, 0]
        self.dofs = model.jnt_dofadr[self.joints]
        self.gear = model.actuator_gear[self.ids, 0]
        self.count = 0
        self.speed_peak = np.zeros(len(self.ids))
        self.torque_peak = np.zeros(len(self.ids))
        self.torque_squared = np.zeros(len(self.ids))
        self.near_limit = np.zeros(len(self.ids), dtype=np.int64)

    def record(self, data):
        force = data.actuator_force[self.ids]
        torque = force * self.gear
        self.speed_peak = np.maximum(self.speed_peak, abs(data.qvel[self.dofs]))
        self.torque_peak = np.maximum(self.torque_peak, abs(torque))
        self.torque_squared += torque**2
        bounds = self.model.actuator_forcerange[self.ids]
        # Within 1% of either configured force bound; not proof of clipping.
        tolerance = .01 * (bounds[:, 1] - bounds[:, 0])
        limited = self.model.actuator_forcelimited[self.ids].astype(bool)
        self.near_limit += limited & ((force <= bounds[:, 0] + tolerance) |
                                      (force >= bounds[:, 1] - tolerance))
        self.count += 1

    def summary(self):
        rows = []
        for k, actuator in enumerate(self.ids):
            limited = bool(self.model.actuator_forcelimited[actuator])
            torque_bounds = sorted((self.model.actuator_forcerange[actuator] * self.gear[k]).tolist())
            rows.append(dict(
                actuator=self.model.actuator(int(actuator)).name,
                joint=self.model.joint(int(self.joints[k])).name,
                peak_speed_rad_s=float(self.speed_peak[k]),
                peak_actuator_torque_nm=float(self.torque_peak[k]),
                rms_actuator_torque_nm=float(np.sqrt(self.torque_squared[k] / max(1, self.count))),
                configured_torque_range_nm=torque_bounds if limited else None,
                near_force_limit_fraction=float(self.near_limit[k] / self.count) if limited and self.count else None,
            ))
        return dict(samples=self.count, sample_period_s=float(self.model.opt.timestep),
                    scope='live steps only; hinge joint actuators; excludes passive and constraint forces',
                    near_limit_definition='within 1% of configured actuator force range',
                    hardware_limits_verified=False, actuators=rows)

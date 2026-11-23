"""Bounded arm-joint residuals with position-only ball observations."""
from pathlib import Path
from dataclasses import replace
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy

from .cascade_env import CascadeEnv, runtime_sources as cascade_sources
from .compare_residual import digest
from .train_transfer import TransferFeatures


def runtime_sources():
    return dict(cascade_sources(), **{'v8_robust_control/motor_env.py': digest(Path(__file__))})


class MotorResidualPolicy(ActorCriticPolicy):
    """A point-mass zero action while startup motor authority is unavailable."""
    @staticmethod
    def motor_mask(observation):
        return ((observation[..., 189] > 0.) & (observation[..., 190] > .5)).to(observation.dtype)

    def forward(self, observation, deterministic=False):
        action, value, probability = super().forward(observation, deterministic)
        mask = self.motor_mask(observation)
        return action*mask[..., None], value, probability*mask

    def evaluate_actions(self, observation, actions):
        value, probability, entropy = super().evaluate_actions(observation, actions)
        mask = self.motor_mask(observation)
        return value, probability*mask, None if entropy is None else entropy*mask

    def _predict(self, observation, deterministic=False):
        return super()._predict(observation, deterministic)*self.motor_mask(observation)[..., None]


class MotorFeatures(TransferFeatures):
    def __init__(self, observation_space):
        super().__init__(observation_space)
        for offset in (0, 14, 28):
            self.scale[offset:offset+6] = 0.
            self.scale[offset+6:offset+9] = 2000.
            self.scale[offset+9:offset+12] = 50.
        self.scale[42:56] = 2000.
        self.scale[56:70] = 200.
        self.scale[70:102] = 100.
        self.scale[102:134] = 50.
        self.scale[134:140] = 100.
        self.scale[140:146] = 10.
        self.scale[146:166] = .1


class MotorResidualCascadeEnv(CascadeEnv):
    """Three motor residuals per arm: shoulder roll, elbow and wrist pitch.

    Startup robot proprioception estimates departure from the nominal motion.
    Ball measurement noise cannot itself open motor authority. This assumes
    the current environment's noiseless joint encoders; encoder robustness is
    not inferred from these tests. The nominal finger commands are retained.
    """
    ENVIRONMENT_VERSION = 8
    CONTROLLER_NAME = 'motor_residual'
    OBSERVATION_LAYOUT = 'ball innovation; arm, finger and palm reference errors; uncertainty and startup motor authority'
    ACTION_LAYOUT = 'shoulder roll, elbow, wrist pitch residuals per arm'
    CONTROLLED_JOINTS = (1, 3, 5)
    STARTUP_OBSERVATION_S = .22
    POSITION_ERROR_SCALE_RAD = .0005
    VELOCITY_ERROR_SCALE_RAD_S = .05

    def reset(self, *, seed=None, options=None):
        self.motor_authority = 0.
        self.requested_translation_peak_m = 0.
        self.motor_slew_peak_rad_s = 0.
        self.joint_goal = np.zeros((2, 7))
        return super().reset(seed=seed, options=options)

    def _robot_reference(self):
        if self.index == 0:
            positions, velocities = self.initial_data.qpos, self.initial_data.qvel
        else:
            reference = self.expected_states[min(self.index-1, len(self.expected_states)-1)]
            positions, velocities = reference[:self.model.nq], reference[self.model.nq:]
        return positions, velocities

    def _robot_error(self):
        positions, velocities = self._robot_reference()
        return (np.array([self.data.qpos[hand['kin'].qpos_adr]-positions[hand['kin'].qpos_adr]
                          for hand in self.hands]),
                np.array([self.data.qvel[hand['kin'].dof_adr]-velocities[hand['kin'].dof_adr]
                          for hand in self.hands]))

    def _notify_support(self):
        super()._notify_support()
        if self.data.time > 0.:
            change = np.clip(self.joint_goal-self.joint_offset,
                             -self.config.max_joint_rate_rad_s*self.dt,
                             self.config.max_joint_rate_rad_s*self.dt)
            self.joint_offset += change
            self.motor_slew_peak_rad_s = max(self.motor_slew_peak_rad_s, float(np.max(np.abs(change)))/self.dt)
            self.max_joint_offset = max(self.max_joint_offset, float(np.max(np.abs(self.joint_offset))))
        # Called before the parent's physics index advances.
        if 0. < self.data.time <= self.STARTUP_OBSERVATION_S+1e-12:
            reference = self.expected_states[min(self.index, len(self.expected_states)-1)]
            error_q = np.concatenate([self.data.qpos[hand['kin'].qpos_adr]-reference[hand['kin'].qpos_adr]
                                      for hand in self.hands])
            error_v = np.concatenate([self.data.qvel[hand['kin'].dof_adr]-reference[self.model.nq+hand['kin'].dof_adr]
                                      for hand in self.hands])
            authority = max(float(np.max(np.abs(error_q)))/self.POSITION_ERROR_SCALE_RAD,
                            float(np.max(np.abs(error_v)))/self.VELOCITY_ERROR_SCALE_RAD_S)
            self.motor_authority = max(self.motor_authority, min(1., authority))

    def permissions(self):
        return np.full(6, self.data.time >= self.STARTUP_OBSERVATION_S and self.motor_authority > 0., dtype=bool)

    def _targets(self, action):
        desired = np.zeros((2, 7))
        request = np.asarray(action).reshape(2, 3)
        for side, hand in enumerate(self.hands):
            desired[side, self.CONTROLLED_JOINTS] = request[side]*self.motor_authority*self.config.max_joint_offset_rad
            if np.any(desired[side]):
                jp = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jp, None, hand['seat'])
                translation = jp[:, hand['kin'].dof_adr]@desired[side]
                scale = max(1., float(np.max(np.abs(translation)))/self.config.max_cartesian_offset_m)
                desired[side] /= scale
                self.requested_translation_peak_m = max(self.requested_translation_peak_m,
                                                        float(np.max(np.abs(translation/scale))))
        self.joint_goal = desired
        self.grip_offset[:] = 0.
        self.wrapping_gain = np.zeros(2)

    def _wrapping_target(self):
        return np.zeros((2, 16))

    def _ending_control(self):
        # The executed nominal ending is already part of the motor reference.
        # The policy, rather than an additional noisy planner, controls residuals.
        pass

    def _observation(self):
        values = super()._observation()
        error_q, error_v = self._robot_error()
        values[42:56] = error_q.ravel()
        values[56:70] = error_v.ravel()/10.
        positions, velocities = self._robot_reference()
        for side, hand in enumerate(self.hands):
            fingers = hand['fingers']
            q = self.model.jnt_qposadr[fingers.joint_ids]
            v = self.model.jnt_dofadr[fingers.joint_ids]
            values[70+side*16:86+side*16] = self.data.qpos[q]-positions[q]
            values[102+side*16:118+side*16] = (self.data.qvel[v]-velocities[v])/10.
            if self.index == 0:
                reference_position = self.initial_data.site_xpos[hand['seat']]
                spatial = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, self.initial_data, mujoco.mjtObj.mjOBJ_SITE, hand['seat'], spatial, 0)
                reference_velocity = spatial[3:]
            else:
                index = self._reference_index()
                reference_position, reference_velocity = self.palm_reference[index, side, :3], self.palm_reference[index, side, 3:]
            values[134+side*3:137+side*3] = self.data.site_xpos[hand['seat']]-reference_position
            values[140+side*3:143+side*3] = self.controller.spatial(hand)[3:]-reference_velocity
        values[185:188] = 10.
        velocity_uncertainty = []
        for ball, tracker in enumerate(self.trackers):
            uncertainty = tracker.uncertainty(float(self.data.time))
            values[185+ball] = min(10., uncertainty[0]/.015)
            velocity_uncertainty.append(uncertainty[1]/.5)
        values[188:191] = [min(10., max(velocity_uncertainty)), self.motor_authority,
                           float(self.data.time >= self.STARTUP_OBSERVATION_S)]
        return values

    def _info(self, reason):
        return dict(super()._info(reason), controller=self.CONTROLLER_NAME,
                    motor_authority=self.motor_authority,
                    controlled_arm_joint_indices=list(self.CONTROLLED_JOINTS),
                    max_requested_translation_m=self.requested_translation_peak_m,
                    max_motor_slew_rad_s=self.motor_slew_peak_rad_s,
                    proprioception='noiseless robot joint positions and velocities')


def training_curriculum():
    """Training prefixes only; formal evaluation always runs the full task."""
    return [dict(min_steps=4096, seat_jitter_m=.0005, catch_goal=1, noise_std_m=[0.]),
            dict(min_steps=4096, seat_jitter_m=.001, catch_goal=1, noise_std_m=[0.]),
            dict(min_steps=4096, seat_jitter_m=.002, catch_goal=1, noise_std_m=[0.]),
            dict(min_steps=4096, seat_jitter_m=.002, catch_goal=3, noise_std_m=[0.]),
            dict(min_steps=8192, seat_jitter_m=.002, catch_goal=6, noise_std_m=[0., .001]),
            dict(min_steps=0, seat_jitter_m=.002, catch_goal=30,
                 noise_std_m=[0., .001, .003, .005, .01], delay_s=[0., .02], dropout=[0., .1])]


class MotorCurriculum(gym.Wrapper):
    """Train on confirmed catch prefixes without changing task acceptance.

    Prefix completion is a training terminal, not a full-task success. The
    underlying environment's success flag and physical failure remain intact.
    The auxiliary potential is bounded, discounted and zero at all terminals.
    """
    ADVANCE_WINDOW = 32
    ADVANCE_RATE = .75

    def __init__(self, env, schedule):
        super().__init__(env)
        self.schedule = schedule
        self.base_config = env.config
        self.steps = 0
        self.phi = 0.
        self.goal_count = 0
        self.active = False
        self.stage_index = 0
        self.stage_start_step = 0
        self.outcomes = deque(maxlen=self.ADVANCE_WINDOW)

    def reset(self, *, seed=None, options=None):
        if seed is not None or not hasattr(self, 'conditions'):
            self.conditions = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1871,)))
        self.stage = self.schedule[self.stage_index]
        if (self.stage_index+1 < len(self.schedule)
                and self.steps-self.stage_start_step >= self.stage['min_steps']
                and len(self.outcomes) == self.ADVANCE_WINDOW
                and np.mean(self.outcomes) >= self.ADVANCE_RATE):
            self.stage_index += 1
            self.stage_start_step = self.steps
            self.outcomes.clear()
            self.stage = self.schedule[self.stage_index]
            print('[Motor curriculum] advancing to catch goal', self.stage['catch_goal'],
                  'after', self.steps, 'training steps', flush=True)
        clean = self.conditions.random() < .2
        self.env.config = replace(self.base_config, seat_jitter_m=0. if clean else self.stage['seat_jitter_m'],
                                  noise_std_m=0. if clean else float(self.conditions.choice(self.stage['noise_std_m'])),
                                  delay_s=0. if clean else float(self.conditions.choice(self.stage.get('delay_s', [0.]))),
                                  dropout=0. if clean else float(self.conditions.choice(self.stage.get('dropout', [0.]))))
        self.phi = 0.
        self.active = True
        return self.env.reset(seed=seed, options=options)

    def _potential(self):
        errors = []
        index = self.env._reference_index()
        for ball, tracker in enumerate(self.env.trackers):
            if self.env.modes[ball] != 'flight':
                continue
            estimate = tracker.estimate(float(self.env.data.time))
            if estimate is None or not estimate.valid or not self.env.estimate_valid[index, ball]:
                continue
            reference = self.env.estimate_reference[index, ball]
            p = (np.asarray(estimate.position_m)-reference[:3])/.03
            v = (np.asarray(estimate.velocity_m_s)-reference[3:])/.4
            errors.append(float(np.exp(-p@p-v@v)))
        return float(np.mean(errors)) if errors else 0.

    def progress_summary(self):
        return dict(stage=self.stage_index, catch_goal=self.stage['catch_goal'],
                    stage_steps=self.steps-self.stage_start_step,
                    seat_jitter_m=self.stage['seat_jitter_m'], prefix_goals=self.goal_count,
                    physical_window_episodes=len(self.outcomes),
                    physical_window_goals=sum(self.outcomes))

    def step(self, action):
        if not self.active:
            raise RuntimeError('Reset the training episode before stepping')
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        goal = min(self.stage['catch_goal'], self.env.target_throws)
        reached = not (terminated or truncated) and goal < self.env.target_throws and info['catches'] >= goal
        if reached:
            self.goal_count += 1
            if self.goal_count % 32 == 0:
                print('[Motor curriculum]', self.steps, 'steps;', self.goal_count,
                      'confirmed prefixes; current catch goal', goal, flush=True)
        terminal = terminated or truncated or reached
        if terminal and self.env.config.seat_jitter_m:
            self.outcomes.append(bool(reached or info['is_success']))
        next_phi = 0. if terminated or reached else self._potential()
        shaping = .5*(.99*next_phi-self.phi)
        self.phi = next_phi
        bonus = 10. if reached else 0.
        self.active = not terminal
        return observation, reward+shaping+bonus, terminated or reached, truncated, dict(info,
            curriculum_catch_goal=goal, curriculum_goal_reached=reached,
            curriculum_stage=self.stage_index, curriculum_stage_steps=self.steps-self.stage_start_step,
            curriculum_shaping=shaping, curriculum_terminal_bonus=bonus)

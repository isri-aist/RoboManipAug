import time
from enum import Enum
import numpy as np
import pinocchio as pin


class MotionInterpolator(object):
    class TargetSpace(Enum):
        JOINT = 0
        EEF = 1

    def __init__(self, env, motion_manager):
        self.env = env
        self.motion_manager = motion_manager
        self.clear_target()

    def clear_target(self):
        self.target_space = None
        self.target_state = None

    def set_target(self, space: TargetSpace, target, duration=None):
        self.target_space = space
        self.target_state = target
        self.duration = duration or 3.0  # TODO

        if self.target_space == self.TargetSpace.JOINT:
            self.start_state = self.motion_manager.joint_pos.copy()
        elif self.target_space == self.TargetSpace.EEF:
            self.start_state = self.motion_manager.target_se3.copy()
        self.start_time = self.env.unwrapped.get_time()

    def update(self):
        if self.target_state is None:
            return

        current_time = self.env.unwrapped.get_time()
        current_ratio = np.clip(
            (current_time - self.start_time) / self.duration, 0.0, 1.0
        )
        if self.target_space == self.TargetSpace.JOINT:
            # joint_vel_limit = np.deg2rad(10.0)  # [rad/s]
            # joint_delta_limit = joint_vel_limit * self.env.unwrapped.dt
            # self.motion_manager.joint_pos += np.clip(
            #     self.target_state - self.motion_manager.joint_pos,
            #     -1 * joint_delta_limit,
            #     joint_delta_limit,
            # )
            delta_state = current_ratio * (self.target_state - self.start_state)
            self.motion_manager.joint_pos = self.start_state + delta_state
            self.motion_manager.forward_kinematics()
            self.motion_manager.target_se3 = self.motion_manager.current_se3.copy()
        elif self.target_space == self.TargetSpace.EEF:
            # error_vec = pin.log(self.motion_manager.target_se3.actInv(self.target_state))
            # eef_vel_limit = np.array([0.1] * 3 + [np.deg2rad(30)] * 3)  # [m/s & rad/s]
            # eef_vel_limit = eef_vel_limit * self.env.unwrapped.dt
            # delta_vec = np.clip(error_vec, -1 * eef_vel_limit, eef_vel_limit)
            # self.motion_manager.target_se3 = self.motion_manager.target_se3 * pin.exp(
            #     delta_vec
            # )
            delta_state = pin.exp(
                current_ratio * pin.log(self.start_state.actInv(self.target_state))
            )
            self.motion_manager.target_se3 = self.start_state * delta_state
            self.motion_manager.inverse_kinematics()

    # def completed(self):
    #     if self.target_space == self.TargetSpace.JOINT:
    #         joint_thre = 1e-2  # [rad]
    #         joint_error = np.sum(np.abs(self.target_state - self.motion_manager.joint_pos))
    #         return joint_error < joint_thre
    #     elif self.target_space == self.TargetSpace.EEF:
    #         eef_thre = 1e-2  # [rad & m]
    #         eef_error_se3 = self.motion_manager.current_se3.actInv(self.target_state)
    #         eef_error = np.linalg.norm(pin.log(eef_error_se3).vector)
    #         return eef_error < eef_thre

    def wait(self, clear_target=True):
        if self.target_state is None:
            return

        end_time = self.start_time + self.duration
        while self.env.unwrapped.get_time() < end_time:
            time.sleep(0.01)

        if clear_target:
            self.clear_target()

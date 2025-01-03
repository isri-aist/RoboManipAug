import time
from enum import Enum
import numpy as np
import pinocchio as pin


class MotionInterpolator(object):
    """
    Motion interpolator.

    The class interpolates target in joint space or end-effector space.
    """

    class TargetSpace(Enum):
        """Target space."""

        JOINT = 0
        EEF = 1

    def __init__(self, env, motion_manager):
        self.env = env
        self.motion_manager = motion_manager
        self.clear_target()

    def clear_target(self):
        self.target_space = None
        self.target_state = None

    def set_target(
        self, target_space: TargetSpace, target_state, duration=None, vel_limit=None
    ):
        """Set target.

        Args:
            target_space (TargetSpace): target space (JOINT or EEF)
            target_state (np.array or pin.SE3): target joint position (np.array) or target EEF pose (pin.SE3)
            duration (float): interpolation duration [s]
            vel_limit (np.array): joint velocity limit (joint-dimensional np.array) or EEF velocity limit (6-dimensional np.array)

        Note:
            If duration is not set, it is automatically set from vel_limit.
        """
        self.target_space = target_space
        self.target_state = target_state
        self.duration = duration

        if self.target_space == self.TargetSpace.JOINT:
            self.start_state = self.motion_manager.joint_pos.copy()
        elif self.target_space == self.TargetSpace.EEF:
            self.start_state = self.motion_manager.target_se3.copy()
        self.start_time = self.env.unwrapped.get_time()

        if self.duration is None:
            if self.target_space == self.TargetSpace.JOINT:
                self.duration = np.max(
                    np.divide(np.abs(self.target_state - self.start_state), vel_limit)
                )
            elif self.target_space == self.TargetSpace.EEF:
                self.duration = np.max(
                    np.divide(
                        np.abs(pin.log(self.target_state.actInv(self.start_state))),
                        vel_limit,
                    )
                )

    def update(self):
        if self.target_state is None:
            return

        current_time = self.env.unwrapped.get_time()
        current_ratio = np.clip(
            (current_time - self.start_time) / self.duration, 0.0, 1.0
        )
        if self.target_space == self.TargetSpace.JOINT:
            delta_state = current_ratio * (self.target_state - self.start_state)
            self.motion_manager.joint_pos = self.start_state + delta_state
            self.motion_manager.forward_kinematics()
            self.motion_manager.target_se3 = self.motion_manager.current_se3.copy()
        elif self.target_space == self.TargetSpace.EEF:
            delta_state = pin.exp(
                current_ratio * pin.log(self.start_state.actInv(self.target_state))
            )
            self.motion_manager.target_se3 = self.start_state * delta_state
            self.motion_manager.inverse_kinematics()

    def wait(self, clear_target=True):
        if self.target_state is None:
            return

        end_time = self.start_time + self.duration
        while self.env.unwrapped.get_time() < end_time:
            time.sleep(0.01)

        self.update()
        if clear_target:
            self.clear_target()

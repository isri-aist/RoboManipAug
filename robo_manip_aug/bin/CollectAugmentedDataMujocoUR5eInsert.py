import gymnasium as gym
import numpy as np
from robo_manip_baselines.common import DataKey

from robo_manip_aug import CollectAugmentedDataBase


class CollectAugmentedDataMujocoUR5eInsert(CollectAugmentedDataBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eInsertEnv-v0", render_mode="human"
        )
        self.demo_name = self.args.demo_name or "MujocoUR5eInsert"

    def set_gripper_command(self):
        if self.phase_manager.is_phase("GraspPhase"):
            self.motion_manager.set_command_data(
                DataKey.COMMAND_GRIPPER_JOINT_POS, np.array([170.0])
            )
        else:
            super().set_gripper_command()


if __name__ == "__main__":
    collect = CollectAugmentedDataMujocoUR5eInsert()
    collect.run()

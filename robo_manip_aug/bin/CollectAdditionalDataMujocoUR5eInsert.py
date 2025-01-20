import gymnasium as gym
import numpy as np
import pinocchio as pin
from robo_manip_baselines.common import DataKey, Phase

from robo_manip_aug import CollectAdditionalDataBase


class CollectAdditionalDataMujocoUR5eInsert(CollectAdditionalDataBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eInsertEnv-v0", render_mode="human"
        )
        self.demo_name = self.args.demo_name or "MujocoUR5eInsert"

    def set_gripper_command(self):
        if self.phase_manager.phase == Phase.GRASP:
            self.motion_manager.set_command_data(
                DataKey.COMMAND_GRIPPER_JOINT_POS, np.array([170.0])
            )
        else:
            super().set_gripper_command()


if __name__ == "__main__":
    collect = CollectAdditionalDataMujocoUR5eInsert()
    collect.run()

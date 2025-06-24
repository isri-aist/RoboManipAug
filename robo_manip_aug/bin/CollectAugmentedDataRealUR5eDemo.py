import gymnasium as gym
import numpy as np
import yaml
from robo_manip_baselines.common import DataKey, GraspPhaseBase

from robo_manip_aug import CollectAugmentedDataBase


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([170.0])
        self.duration = 0.5  # [s]


class CollectAugmentedDataRealUR5eDemo(CollectAugmentedDataBase):
    def setup_args(self, parser=None):
        super().setup_args(parser)

        parser.add_argument(
            "--config",
            type=str,
            required=True,
            default=None,
            help="Config file(.yaml) for UR5e.",
        )

    def setup_env(self):
        with open(self.args.config, "r") as f:
            config = yaml.safe_load(f)
        self.robot_ip = config["robot_ip"]
        self.camera_ids = config["camera_ids"]
        self.gelsight_ids = (
            None if "gelsight_ids" not in config.keys() else config["gelsight_ids"]
        )
        self.env = gym.make(
            "robo_manip_baselines/RealUR5eDemoEnv-v0",
            robot_ip=self.robot_ip,
            camera_ids=self.camera_ids,
            gelsight_ids=self.gelsight_ids,
        )
        self.demo_name = self.args.demo_name or "RealUR5eDemo"

    def set_gripper_command(self):
        if self.phase_manager.is_phase("GraspPhase"):
            self.motion_manager.set_command_data(
                DataKey.COMMAND_GRIPPER_JOINT_POS, np.array([170.0])
            )
        else:
            super().set_gripper_command()

    def get_pre_motion_phases(self):
        return [GraspPhase(self)]


if __name__ == "__main__":
    collect = CollectAugmentedDataRealUR5eDemo()
    collect.run()

import gymnasium as gym

from robo_manip_aug import CollectAugmentedDataBase


class CollectAugmentedDataMujocoUR5eCabinet(CollectAugmentedDataBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eCabinetEnv-v0", render_mode="human"
        )
        self.demo_name = self.args.demo_name or "MujocoUR5eCabinet"


if __name__ == "__main__":
    collect = CollectAugmentedDataMujocoUR5eCabinet()
    collect.run()

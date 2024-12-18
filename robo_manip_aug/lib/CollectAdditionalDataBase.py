import argparse
import time
import datetime
import numpy as np
from robo_manip_baselines.common import (
    MotionStatus,
    DataKey,
    DataManager,
)
from robo_manip_baselines.teleop import TeleopBase


class CollectAdditionalDataBase(TeleopBase):
    def __init__(self):
        super().__init__()

        MotionStatus.TELEOP._name_ = "AUTO_AUGMENTATION"

        # Setup data manager for base data
        self.base_data_manager = DataManager(self.env, demo_name=self.demo_name)
        self.base_data_manager.setup_camera_info()
        self.datetime_now = datetime.datetime.now()

    def setup_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument(
            "base_data",
            type=str,
            help="log file path from which data augmentation will be based",
        )

        super().setup_args(parser)

        if self.args.replay_log is not None:
            raise ValueError("replay_log must not be specified.")

    def run(self):
        self.reset_flag = True
        self.quit_flag = False
        iteration_duration_list = []

        while True:
            iteration_start_time = time.time()

            # Reset
            if self.reset_flag:
                self.reset()
                self.reset_flag = False

            # Get action
            if self.args.replay_log is not None and self.data_manager.status in (
                MotionStatus.TELEOP,
                MotionStatus.END,
            ):
                action = self.data_manager.get_single_data(
                    DataKey.COMMAND_JOINT_POS, self.teleop_time_idx
                )
            else:
                # Set commands
                self.set_arm_command()
                self.set_gripper_command()

                # Solve IK
                self.motion_manager.draw_markers()
                self.motion_manager.inverse_kinematics()

                action = self.motion_manager.get_action()

            # Record data
            if (
                self.data_manager.status == MotionStatus.TELEOP
                and self.args.replay_log is None
            ):
                self.record_data(obs, action, info)  # noqa: F821

            # Step environment
            obs, _, _, _, info = self.env.step(action)

            # Draw images
            self.draw_image(info)

            # Draw point clouds
            if self.args.enable_3d_plot:
                self.draw_point_cloud(info)

            # Manage status
            self.manage_status()
            if self.quit_flag:
                break

            iteration_duration = time.time() - iteration_start_time
            if self.data_manager.status == MotionStatus.TELEOP:
                iteration_duration_list.append(iteration_duration)
            if iteration_duration < self.env.unwrapped.dt:
                time.sleep(self.env.unwrapped.dt - iteration_duration)

        print("- Statistics on teleoperation")
        if len(iteration_duration_list) > 0:
            iteration_duration_list = np.array(iteration_duration_list)
            print(
                f"  - Real-time factor | {self.env.unwrapped.dt / iteration_duration_list.mean():.2f}"
            )
            print(
                "  - Iteration duration [s] | "
                f"mean: {iteration_duration_list.mean():.3f}, std: {iteration_duration_list.std():.3f} "
                f"min: {iteration_duration_list.min():.3f}, max: {iteration_duration_list.max():.3f}"
            )

        # self.env.close()

    def reset(self):
        # Reset managers
        self.motion_manager.reset()
        self.data_manager.reset()

        # Load base data
        self.base_data_manager.load_data(self.args.base_data)
        print("- Load teleoperation data: {}".format(self.args.base_data))
        world_idx = self.base_data_manager.get_data("world_idx").tolist()
        self.data_manager.setup_sim_world(world_idx)
        self.base_data_manager.setup_sim_world(world_idx)
        obs, info = self.env.reset()
        print(
            "[{}] data_idx: {}, world_idx: {}".format(
                self.demo_name,
                self.data_manager.data_idx,
                self.data_manager.world_idx,
            )
        )
        print("- Press the 'n' key to start automatic grasping.")

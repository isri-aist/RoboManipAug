import yaml
import argparse
import time
import datetime
import numpy as np
import cv2
import threading
from robo_manip_baselines.common import (
    MotionStatus,
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
            "--base_demo_path",
            type=str,
            help="path of teleoperation data as base for data augmentation",
            required=True,
        )
        parser.add_argument(
            "--annotation_path",
            type=str,
            help="Path of annotation data describing the region of data augmentation",
            required=True,
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
            self.set_arm_command()
            self.set_gripper_command()
            action = self.motion_manager.get_action()

            # Record data
            if self.data_manager.status == MotionStatus.TELEOP:
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

        # self.env.close()

    def reset(self):
        # Reset managers
        self.motion_manager.reset()
        self.data_manager.reset()
        self.base_data_manager.reset()

        # Load base demo data
        print(
            "[CollectAdditionalDataBase] Load teleoperation data: {}".format(
                self.args.base_demo_path
            )
        )
        self.base_data_manager.load_data(self.args.base_demo_path)

        # Load annotation data
        with open(self.args.annotation_path, "r") as f:
            self.annotation_data = yaml.load(f, Loader=yaml.SafeLoader)

        # Reset env
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
        print(
            "[CollectAdditionalDataBase] Press the 'n' key to start automatic grasping."
        )

    def collect_data(self):
        def is_motion_completed():
            if self.motion_space == "JOINT":
                joint_thre = 1e-2  # [rad]
                return (
                    np.sum(np.abs(self.motion_target - self.motion_manager.joint_pos))
                    < joint_thre
                )

        def wait_motion_completion():
            while not is_motion_completed():
                time.sleep(0.01)

        for acceptable_region_idx, acceptable_region in enumerate(
            self.annotation_data["acceptable_region_list"]
        ):
            self.motion_space = "JOINT"
            joint_pos = np.array(acceptable_region["joint_pos"])
            self.motion_target = joint_pos[self.env.unwrapped.ik_arm_joint_ids]
            wait_motion_completion()

    def set_arm_command(self):
        if self.data_manager.status == MotionStatus.TELEOP:
            if self.motion_space == "JOINT":
                joint_vel_limit = np.deg2rad(10.0)  # [rad/s]
                joint_delta_limit = joint_vel_limit * self.env.unwrapped.dt
                self.motion_manager.joint_pos += np.clip(
                    self.motion_target - self.motion_manager.joint_pos,
                    -1 * joint_delta_limit,
                    joint_delta_limit,
                )
                self.motion_manager.forward_kinematics()
                self.motion_manager.target_se3 = self.motion_manager.current_se3.copy()
            else:  # if self.motion_space == "POSE":
                self.motion_manager.inverse_kinematics()

    def set_gripper_command(self):
        if self.data_manager.status == MotionStatus.GRASP:
            self.motion_manager.gripper_pos = self.env.action_space.high[
                self.env.unwrapped.gripper_action_idx
            ]

    def manage_status(self):
        key = cv2.waitKey(1)
        if self.data_manager.status == MotionStatus.INITIAL:
            if key == ord("n"):
                self.data_manager.go_to_next_status()
        elif self.data_manager.status == MotionStatus.PRE_REACH:
            pre_reach_duration = 0.7  # [s]
            if self.data_manager.status_elapsed_duration > pre_reach_duration:
                self.data_manager.go_to_next_status()
        elif self.data_manager.status == MotionStatus.REACH:
            reach_duration = 0.3  # [s]
            if self.data_manager.status_elapsed_duration > reach_duration:
                print(
                    "[CollectAdditionalDataBase] Press the 'n' key to start data collection."
                )
                self.data_manager.go_to_next_status()
        elif self.data_manager.status == MotionStatus.GRASP:
            if key == ord("n"):
                self.teleop_time_idx = 0
                self.thread = threading.Thread(target=self.collect_data)
                self.thread.start()
                self.thread.join(0.1)
                print("[CollectAdditionalDataBase] Start a thread for data collection.")
                self.data_manager.go_to_next_status()
        elif self.data_manager.status == MotionStatus.TELEOP:
            self.teleop_time_idx += 1
            if not self.thread.is_alive():
                print(
                    "[CollectAdditionalDataBase] Press the 's' key to save the collected data, press the 'f' key to exit without saving."
                )
                self.data_manager.go_to_next_status()
        elif self.data_manager.status == MotionStatus.END:
            if key == ord("s"):
                # Save data
                self.save_data()
                self.quit_flag = True
            elif key == ord("f"):
                self.quit_flag = True
        if key == 27:  # escape key
            self.quit_flag = True

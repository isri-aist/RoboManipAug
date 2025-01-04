from os import path
import yaml
import argparse
import time
import datetime
import numpy as np
import cv2
import threading
import pinocchio as pin
from robo_manip_baselines.common import (
    MotionStatus,
    DataKey,
    DataManager,
)
from robo_manip_baselines.teleop import TeleopBase
from robo_manip_aug import MotionInterpolator


def sample_points_on_sphere(center, radius, num_points):
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    cos_theta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(cos_theta)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    points = np.stack((x, y, z), axis=-1) * radius + center

    return points


class CollectAdditionalDataBase(TeleopBase):
    def __init__(self):
        super().__init__()

        self.data_manager.general_info["format"] = "RoboManipBaselines-AugmentedData"

        MotionStatus.TELEOP._name_ = "AUGMENTATION"

        # Setup data manager for base data
        self.base_data_manager = DataManager(self.env, demo_name=self.demo_name)
        self.base_data_manager.setup_camera_info()
        self.datetime_now = datetime.datetime.now()

        # Setup motion interpolator
        self.motion_interpolator = MotionInterpolator(self.env, self.motion_manager)

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
        self.save_flag = False
        self.executing_augmented_motion = False
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
            if (
                self.data_manager.status == MotionStatus.TELEOP
                and self.executing_augmented_motion
            ):
                self.record_data(self.obs, action, info)  # noqa: F821

            # Step environment
            self.obs, _, _, _, info = self.env.step(action)

            # Save data
            if self.save_flag:
                self.save_flag = False
                self.save_data()

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
        self.obs, info = self.env.reset()
        print(
            "[CollectAdditionalDataBase] demo_name: {}, world_idx: {}".format(
                self.demo_name,
                self.data_manager.world_idx,
            )
        )
        print(
            "[CollectAdditionalDataBase] Press the 'n' key to start automatic grasping."
        )

    def collect_data(self):
        eef_offset_se3 = pin.SE3(
            np.array(self.annotation_data["eef_offset"]["rot"]),
            np.array(self.annotation_data["eef_offset"]["pos"]),
        )

        for self.acceptable_region_idx, acceptable_region in enumerate(
            self.annotation_data["acceptable_region_list"]
        ):
            # Sample EEF position
            num_points = 4
            center_pos = np.array(acceptable_region["center"]["eef_pos"])
            center_rot = np.array(acceptable_region["center"]["eef_rot"])
            radius = acceptable_region["radius"]
            sample_pos_list = sample_points_on_sphere(center_pos, radius, num_points)

            for self.sample_idx, sample_pos in enumerate(sample_pos_list):
                # Move to convergence point
                joint_pos = np.array(acceptable_region["convergence"]["joint_pos"])
                self.motion_interpolator.set_target(
                    MotionInterpolator.TargetSpace.JOINT,
                    joint_pos[self.env.unwrapped.ik_arm_joint_ids],
                    vel_limit=np.deg2rad(30.0),  # [rad/s]
                )
                self.motion_interpolator.wait()
                self.wait_until_motion_stop()

                # Move to sampled point
                eef_se3 = pin.SE3(center_rot, sample_pos) * eef_offset_se3.inverse()
                self.motion_interpolator.set_target(
                    MotionInterpolator.TargetSpace.EEF,
                    eef_se3,
                    duration=1.0,  # [s]
                )
                self.executing_augmented_motion = True
                self.motion_interpolator.wait()
                self.executing_augmented_motion = False
                self.save_flag = True
                self.wait_until_motion_stop()

                if self.quit_flag:
                    return

    def wait_until_motion_stop(self):
        joint_vel_thre = 1e-3  # [rad/s]
        while True:
            joint_vel = np.linalg.norm(self.motion_manager.get_joint_vel(self.obs))
            if joint_vel < joint_vel_thre:
                break
            time.sleep(self.env.unwrapped.dt)

    def set_arm_command(self):
        if self.data_manager.status == MotionStatus.TELEOP:
            self.motion_interpolator.update()

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
                    "[CollectAdditionalDataBase] Finish a thread for data collection."
                )
                print("[CollectAdditionalDataBase] Press the 'n' key to quit.")
                self.data_manager.go_to_next_status()
        elif self.data_manager.status == MotionStatus.END:
            if key == ord("n"):
                self.quit_flag = True
        if key == 27:  # escape key
            self.quit_flag = True

    def save_data(self):
        all_data_seq = self.data_manager.all_data_seq
        seq_len = len(all_data_seq[DataKey.TIME])

        # Reverse motion data
        for key in list(all_data_seq.keys()):
            if key == DataKey.TIME:
                continue
            elif (
                isinstance(all_data_seq[key], list)
                and len(all_data_seq[key]) == seq_len
            ):
                all_data_seq[key].reverse()
            elif (
                isinstance(all_data_seq[key], np.ndarray)
                and all_data_seq[key].ndim > 0
                and len(all_data_seq[key]) == seq_len
            ):
                all_data_seq[key] = all_data_seq[key][::-1]

        # Dump to file
        filename = "augmented_data/{}_Augmented_{:0>3}_{:0>2}.npz".format(
            path.splitext(path.basename(self.args.annotation_path))[0].removesuffix(
                "_Annotation"
            ),
            self.acceptable_region_idx,
            self.sample_idx,
        )
        super().save_data(filename)

        # Clear motion data
        for key in list(all_data_seq.keys()):
            if (
                isinstance(all_data_seq[key], list)
                or (
                    isinstance(all_data_seq[key], np.ndarray)
                    and all_data_seq[key].ndim > 0
                )
            ) and len(all_data_seq[key]) == seq_len:
                del all_data_seq[key]

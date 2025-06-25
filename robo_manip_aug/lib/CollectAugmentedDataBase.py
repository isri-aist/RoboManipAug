import argparse
import os
import pickle
import threading
import time
from os import path

import cv2
import numpy as np
import pinocchio as pin
from robo_manip_baselines.common import (
    ArmConfig,
    DataKey,
    DataManager,
    PhaseBase,
    get_se3_from_pose,
)
from robo_manip_baselines.teleop import TeleopBase

from robo_manip_aug import MotionInterpolator


class EndCollectAugmentedDataPhase(PhaseBase):
    def start(self):
        super().start()

        print(f"[{self.op.__class__.__name__}] Finished Collecting augmented data.")

    def post_update(self):
        if self.op.key == ord("n"):
            self.op.quit_flag = True


def sample_points_on_sphere(center, radius, num_points):
    """Sample points from the surface of the sphere."""
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    cos_theta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(cos_theta)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    points = np.stack((x, y, z), axis=-1) * radius + center

    return points


def sample_random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(-1.0 * max_angle, max_angle)
    return pin.AngleAxis(angle, axis).toRotationMatrix()


class CollectAugmentedDataBase(TeleopBase):
    def __init__(self):
        super().__init__()

        self.data_manager.meta_data["format"] = "RoboManipBaselines-AugmentedData"
        self.phase_manager.phase_order[-1] = EndCollectAugmentedDataPhase(op=self)

        # Setup data manager for base data
        self.base_data_manager = DataManager(self.env, demo_name=self.demo_name)
        self.base_data_manager.setup_camera_info()

        # Setup motion interpolator
        self.motion_interpolator = MotionInterpolator(self.env, self.motion_manager)

    def setup_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

        parser.add_argument(
            "base_demo_path",
            type=str,
            help="path of teleoperation data as base for data augmentation",
        )
        parser.add_argument(
            "annotation_path",
            type=str,
            help="Path of annotation data describing the region of data augmentation",
        )
        parser.add_argument(
            "--without_merge_base_demo",
            action="store_true",
            help="Do not merge the base motion after the augmented motion",
        )
        parser.add_argument(
            "--num_sphere_sample",
            type=int,
            default=16,
            help="Number of samples from each sphere of acceptable region",
        )
        parser.add_argument(
            "--rotation_random_scale",
            type=float,
            default=2.0,
            help="Scale of randomness to add to end-effector rotation",
        )
        parser.add_argument(
            "--overwrite_radius",
            type=float,
            default=None,
            help="Overwrite radius of acceptable region with a fixed value",
        )
        parser.add_argument(
            "--return_to_center",
            action="store_true",
            help="Return to the center of the acceptable region",
        )

        parser.add_argument(
            "--auto_mode",
            action="store_true",
            help="Whether to enable automatic mode that does not wait for key inputs",
        )

        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Config file(.yaml) for some envs.",
        )

        super().setup_args(parser)

        if self.args.replay_log is not None:
            raise ValueError("replay_log must not be specified.")

    def run(self):
        base_demo_format = os.path.splitext(self.args.base_demo_path.rstrip("/"))[-1]
        # Create a symbolic link to the base demo file
        symlink_filename = "augmented_data/{}_{:%Y%m%d_%H%M%S}/base_demo{}".format(
            self.demo_name, self.datetime_now, base_demo_format
        )
        os.makedirs(os.path.dirname(symlink_filename), exist_ok=True)
        os.symlink(path.abspath(self.args.base_demo_path), symlink_filename)
        print(
            f"[CollectAugmentedDataBase] Create a symbolic link to the base demo file: {symlink_filename}"
        )

        self.reset_flag = True
        self.quit_flag = False
        self.save_flag = False
        self.executing_augmented_motion = False
        self.follow_demo_info = None
        self.iteration_duration_list = []

        while True:
            iteration_start_time = time.time()

            # Reset
            if self.reset_flag:
                self.reset()
                self.reset_flag = False

            self.phase_manager.pre_update()
            self.motion_manager.draw_markers()

            self.set_gripper_command()
            self.set_arm_command()

            # Set action
            action = self.motion_manager.get_command_data(DataKey.COMMAND_JOINT_POS)

            # Record data
            if (
                self.phase_manager.is_phase("TeleopPhase")
                and self.executing_augmented_motion
            ):
                self.record_data()  # noqa: F821

            # Step environment
            self.obs, self.reward, _, _, self.info = self.env.step(action)

            # Save data
            if self.save_flag:
                self.save_flag = False
                self.save_data()

            # Draw images
            self.draw_image()

            # Draw point clouds
            if self.args.enable_3d_plot:
                self.draw_point_cloud()

            self.phase_manager.post_update()

            # Manage phase
            self.manage_phase()
            if self.quit_flag:
                break

            iteration_duration = time.time() - iteration_start_time
            if self.phase_manager.is_phases(["TeleopPhase", "ReplayPhase"]) and (
                self.teleop_time_idx > 0
            ):
                self.iteration_duration_list.append(iteration_duration)

            if (not self.auto_mode) and (iteration_duration < self.env.unwrapped.dt):
                time.sleep(self.env.unwrapped.dt - iteration_duration)

        # self.env.close()

    def reset(self):
        # Reset managers
        self.motion_manager.reset()
        self.phase_manager.reset()
        self.data_manager.reset()
        self.base_data_manager.reset()

        # Load base demo data
        print(
            f"[CollectAugmentedDataBase] Load teleoperation data: {self.args.base_demo_path}"
        )
        self.base_data_manager.load_data(self.args.base_demo_path)

        # Load annotation data
        print(
            f"[CollectAugmentedDataBase] Load annotation data: {self.args.annotation_path}"
        )
        with open(self.args.annotation_path, "rb") as f:
            self.annotation_data = pickle.load(f)

        # Reset environment
        world_idx = self.base_data_manager.get_meta_data("world_idx")
        self.data_manager.setup_env_world(world_idx)
        self.base_data_manager.world_idx = self.data_manager.world_idx
        self.obs, info = self.env.reset()

        print(
            "[CollectAugmentedDataBase] demo_name: {}, world_idx: {}".format(
                self.demo_name,
                self.data_manager.world_idx,
            )
        )
        print(
            "[CollectAugmentedDataBase] Press the 'n' key to start automatic grasping."
        )

    def collect_data(self):
        eef_offset_se3 = get_se3_from_pose(self.annotation_data["eef_offset_pose"])
        if self.args.return_to_center:
            convergence_key = "center"
        else:
            convergence_key = "convergence"

        for self.acceptable_region_idx, acceptable_region in enumerate(
            list(self.annotation_data["acceptable_region_list"]) + [None]
        ):
            # Move along base demo motion
            print("[CollectAugmentedDataBase] Move along the base demo motion.")
            self.follow_demo_info = {}
            if self.acceptable_region_idx == 0:
                self.follow_demo_info["start_time_idx"] = 0
            else:
                prev_acceptable_region = self.annotation_data["acceptable_region_list"][
                    self.acceptable_region_idx - 1
                ]
                self.follow_demo_info["start_time_idx"] = prev_acceptable_region[
                    convergence_key
                ]["time_idx"]
            if self.acceptable_region_idx == len(
                self.annotation_data["acceptable_region_list"]
            ):
                self.follow_demo_info["end_time_idx"] = (
                    len(self.base_data_manager.get_data_seq(DataKey.TIME)) - 1
                )
            else:
                self.follow_demo_info["end_time_idx"] = acceptable_region[
                    convergence_key
                ]["time_idx"]
            self.follow_demo_info["current_time_idx"] = self.follow_demo_info[
                "start_time_idx"
            ]
            while self.follow_demo_info is not None:
                time.sleep(0.01)

            if self.acceptable_region_idx == len(
                self.annotation_data["acceptable_region_list"]
            ):
                break

            # Sample end-effector position
            print(
                "[CollectAugmentedDataBase] Collect data from acceptable region: "
                f"{self.acceptable_region_idx+1} / {len(self.annotation_data['acceptable_region_list'])}"
            )
            center_se3 = get_se3_from_pose(acceptable_region["center"]["eef_pose"])
            if self.args.overwrite_radius is None:
                radius = acceptable_region["radius"]
            else:
                radius = self.args.overwrite_radius
            sample_pos_list = sample_points_on_sphere(
                center_se3.translation, radius, self.args.num_sphere_sample
            )

            for self.sample_idx, sample_pos in enumerate(
                list(sample_pos_list) + [None]
            ):
                # Move to convergence point
                joint_pos = acceptable_region[convergence_key]["joint_pos"]
                vel_limit = np.full_like(joint_pos, np.deg2rad(20.0))  # [rad/s]
                for body_config in self.env.unwrapped.body_config_list:
                    if isinstance(body_config, ArmConfig):
                        gripper_joint_idxes = body_config.gripper_joint_idxes
                        break
                vel_limit[gripper_joint_idxes] = 100.0
                self.motion_interpolator.set_target(
                    MotionInterpolator.TargetSpace.JOINT,
                    joint_pos,
                    vel_limit=vel_limit,
                )
                self.motion_interpolator.wait()
                self.wait_until_motion_stop()
                time.sleep(0.5)  # [s]

                if self.sample_idx == len(sample_pos_list):
                    break

                # Move to sampled point
                sample_rot = center_se3.rotation @ sample_random_rotation(
                    self.args.rotation_random_scale * radius
                )
                eef_se3 = pin.SE3(sample_rot, sample_pos) * eef_offset_se3.inverse()
                self.motion_interpolator.set_target(
                    MotionInterpolator.TargetSpace.EEF,
                    eef_se3,
                    duration=2.0,  # [s]
                )
                self.aug_end_time_idx = acceptable_region[convergence_key]["time_idx"]
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
            joint_vel = np.linalg.norm(
                self.motion_manager.get_measured_data(
                    DataKey.MEASURED_JOINT_VEL, self.obs
                )
            )
            if joint_vel < joint_vel_thre:
                break
            time.sleep(self.env.unwrapped.dt)

    def set_arm_command(self):
        if self.phase_manager.is_phase("TeleopPhase"):
            if self.follow_demo_info is None:
                self.motion_interpolator.update()
            else:
                self.motion_manager.set_command_data(
                    DataKey.COMMAND_JOINT_POS,
                    self.base_data_manager.get_single_data(
                        DataKey.COMMAND_JOINT_POS,
                        self.follow_demo_info["current_time_idx"],
                    ),
                )
                if (
                    self.follow_demo_info["current_time_idx"]
                    == self.follow_demo_info["end_time_idx"]
                ):
                    self.follow_demo_info = None
                else:
                    self.follow_demo_info["current_time_idx"] += 1

    def set_gripper_command(self):
        if self.phase_manager.is_phase("GraspPhase"):
            self.motion_manager.set_command_data(
                DataKey.COMMAND_GRIPPER_JOINT_POS,
                self.env.action_space.high[self.env.unwrapped.gripper_joint_idxes],
            )

    def manage_phase(self):
        self.key = cv2.waitKey(1)
        if self.phase_manager.is_phase("InitialTeleopPhase"):
            self.phase_manager.check_transition()
        elif self.phase_manager.is_phase("ReachPhase1"):
            pre_reach_duration = 0.7  # [s]
            if self.phase_manager.get_phase_elapsed_duration() > pre_reach_duration:
                self.phase_manager.check_transition()
        elif self.phase_manager.is_phase("ReachPhase2"):
            reach_duration = 0.3  # [s]
            if self.phase_manager.get_phase_elapsed_duration() > reach_duration:
                print(
                    "[CollectAugmentedDataBase] Press the 'n' key to start data collection."
                )
                self.phase_manager.check_transition()
        elif self.phase_manager.is_phase("GraspPhase"):
            self.phase_manager.check_transition()
        elif self.phase_manager.is_phase("StandbyTeleopPhase"):
            if self.key == ord("n"):
                self.teleop_time_idx = 0
                self.thread = threading.Thread(target=self.collect_data)
                self.thread.start()
                self.thread.join(0.1)
                print("[CollectAugmentedDataBase] Start a thread for data collection.")
                self.phase_manager.check_transition()
        elif self.phase_manager.is_phase("TeleopPhase"):
            self.teleop_time_idx += 1
            if not self.thread.is_alive():
                print("[CollectAugmentedDataBase] Finish a thread for data collection.")
                print("[CollectAugmentedDataBase] Press the 'n' key to quit teleop.")
                self.phase_manager.check_transition()
        elif self.phase_manager.is_phase("EndCollectAugmentedDataPhase"):
            if self.args.auto_mode:
                self.quit_flag = True
            else:
                if self.key == ord("n"):
                    self.quit_flag = True
        if self.key == 27:  # escape key
            self.quit_flag = True

    def save_data(self):
        # Reverse motion data
        self.data_manager.reverse_data()

        # Merge base demo motion
        if not self.args.without_merge_base_demo:
            for key in self.data_manager.all_data_seq.keys():
                self.data_manager.all_data_seq[key] += list(
                    self.base_data_manager.all_data_seq[key][
                        self.aug_end_time_idx + 1 :
                    ]
                )
            self.data_manager.all_data_seq[DataKey.TIME] = list(
                self.env.unwrapped.dt
                * np.arange(len(self.data_manager.all_data_seq[DataKey.TIME]))
            )

        # Dump to file
        filename = "augmented_data/{}_{:%Y%m%d_%H%M%S}/region{:0>3}/{}_augmented_region{:0>3}_{:0>2}.rmb".format(
            self.demo_name,
            self.datetime_now,
            self.acceptable_region_idx,
            self.demo_name,
            self.acceptable_region_idx,
            self.sample_idx,
        )
        super().save_data(filename)

        # Clear motion data
        self.data_manager.reset()

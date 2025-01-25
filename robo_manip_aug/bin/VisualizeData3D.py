import argparse
import glob
import pickle
import time
from os import path

import numpy as np
import open3d as o3d
import pytransform3d as pytrans3d
import pytransform3d.visualizer as pv
from pytransform3d.urdf import UrdfTransformManager
from robo_manip_baselines.common import (
    DataKey,
    DataManager,
    get_pose_from_rot_pos,
    get_rot_pos_from_pose,
)
from tqdm import tqdm


class VisualizeData3D(object):
    def __init__(self):
        self.setup_args()

        self.setup_variables()

        self.setup_data()

        self.setup_visualization()

    def setup_args(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("teleop_data_dir", type=str)
        parser.add_argument(
            "--base_demo_path",
            type=str,
            default=None,
            help="path of teleoperation data as base for data augmentation",
        )
        parser.add_argument("--point_cloud_path", type=str, default=None)
        self.args = parser.parse_args()

    def setup_variables(self):
        self.quit_flag = False

        eef_offset_pos = np.array([0.0, 0.0, 0.15])  # [m]
        eef_offset_rot = np.identity(3)
        self.eef_offset_mat = pytrans3d.transformations.transform_from(
            eef_offset_rot, eef_offset_pos
        )

        self.time_idx = 0
        self.data_idx = 0

    def setup_data(self):
        teleop_data_path_list = glob.glob(
            path.join(self.args.teleop_data_dir, "**/*.hdf5"), recursive=True
        )
        teleop_data_path_list.sort()
        print(
            f"[VisualizeData3D] Load {len(teleop_data_path_list)} files from {self.args.teleop_data_dir}."
        )

        load_keys = [DataKey.TIME, DataKey.COMMAND_JOINT_POS, DataKey.COMMAND_EEF_POSE]
        self.data_manager_list = []
        for teleop_data_path in tqdm(
            teleop_data_path_list, desc="[VisualizeData3D] Load data"
        ):
            data_manager = DataManager(env=None)
            data_manager.load_data(teleop_data_path, load_keys)
            self.data_manager_list.append(data_manager)

        self.base_data_manager = None
        if self.args.base_demo_path is not None:
            self.base_data_manager = DataManager(env=None)
            self.base_data_manager.load_data(self.args.base_demo_path, load_keys)

    def setup_visualization(self):
        # Initialize figure
        self.fig = pv.figure("RoboManipAug VisualizeData3D", with_key_callbacks=True)
        self.fig.view_init()

        # Set key callbacks
        # See https://www.glfw.org/docs/latest/group__keys.html for key numbers
        self.fig.visualizer.register_key_action_callback(256, self.escape_callback)
        self.fig.visualizer.register_key_action_callback(262, self.right_callback)
        self.fig.visualizer.register_key_action_callback(263, self.left_callback)
        self.fig.visualizer.register_key_action_callback(264, self.down_callback)
        self.fig.visualizer.register_key_action_callback(265, self.up_callback)

        print("""[VisualizeData3D] Key bindings:
  - [esc] Quit.
  - [right-arrow/left-arrow] Go to the next/previous time step. Press [shift] together to increase the increment.
  - [down-arrow/up-arrow] Switch to the next/previous demo.""")

        # Load a URDF model of robot
        self.urdf_tm = UrdfTransformManager()
        urdf_dir = path.join(path.dirname(__file__), "../assets/mujoco_ur5e/")
        urdf_path = path.join(urdf_dir, "mujoco_ur5e.urdf")
        with open(urdf_path, "r") as f:
            self.urdf_tm.load_urdf(f.read(), mesh_path=urdf_dir)
        self.urdf_graph = self.fig.plot_graph(self.urdf_tm, "world", show_visuals=True)

        # Draw EEF trajectory
        data_manager_list = self.data_manager_list
        if self.base_data_manager is not None:
            data_manager_list.append(self.base_data_manager)
        for data_idx, data_manager in enumerate(
            tqdm(data_manager_list, desc="[VisualizeData3D] Draw trajectories")
        ):
            seq_len = len(data_manager.get_data_seq(DataKey.TIME))
            eef_traj_mat_seq = np.empty((seq_len, 4, 4))
            for time_idx in range(seq_len):
                eef_pose = data_manager.get_single_data(
                    DataKey.COMMAND_EEF_POSE, time_idx
                )
                eef_mat = pytrans3d.transformations.transform_from(
                    *get_rot_pos_from_pose(eef_pose)
                )
                eef_traj_mat_seq[time_idx] = eef_mat @ self.eef_offset_mat
            traj_color = [0.0, 0.0, 0.0]
            self.eef_traj = pv.Trajectory(eef_traj_mat_seq, c=traj_color)
            self.fig.add_geometry(self.eef_traj.geometries[0])
            for time_idx in range(seq_len):
                waypoint_mat = np.identity(4)
                if (self.base_data_manager is not None) and (
                    data_idx == len(data_manager_list) - 1
                ):
                    waypoint_radius = 3e-3  # [m]
                    waypoint_color = [0.8, 0.3, 0.8]
                else:
                    waypoint_radius = 2e-3  # [m]
                    waypoint_color = [0.0, 1.0, 0.0]
                waypoint_mat = eef_traj_mat_seq[time_idx]
                waypoint_sphere = pv.Sphere(
                    radius=waypoint_radius, A2B=waypoint_mat, c=waypoint_color
                )
                self.fig.add_geometry(waypoint_sphere.geometries[0])

        # Draw point cloud
        if self.args.point_cloud_path is not None:
            point_cloud = o3d.io.read_point_cloud(self.args.point_cloud_path)
            self.fig.add_geometry(point_cloud)
            opt = self.fig.visualizer.get_render_option()
            opt.point_size = 10.0

        # Set camera pose
        view_ctrl = self.fig.visualizer.get_view_control()
        view_ctrl.set_lookat([0.0, 0.0, 1.4])
        view_ctrl.set_front([1.2, 0.0, 1.8])
        view_ctrl.set_up([0.0, 0.0, 1.0])
        view_ctrl.set_zoom(0.6)

    def run(self):
        while not self.quit_flag:
            self.update_once()

            for geom in self.urdf_graph.geometries:
                self.fig.update_geometry(geom)

            self.fig.visualizer.poll_events()
            self.fig.visualizer.update_renderer()

            time.sleep(0.01)

    def update_once(self):
        joint_pos = self.data_manager_list[self.data_idx].get_single_data(
            DataKey.COMMAND_JOINT_POS, self.time_idx
        )

        # Set arm joints
        arm_joint_name_list = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        for joint_idx, joint_name in enumerate(arm_joint_name_list):
            self.urdf_tm.set_joint(joint_name, joint_pos[joint_idx])

        # Set gripper joints
        gripper_joint_name_list = [
            "right_driver_joint",
            "right_spring_link_joint",
            "right_follower_joint",
            "left_driver_joint",
            "left_spring_link_joint",
            "left_follower_joint",
        ]
        for joint_idx, joint_name in enumerate(gripper_joint_name_list):
            scale = 1.0
            if "follower" in joint_name:
                scale = -1.0
            self.urdf_tm.set_joint(
                joint_name, np.deg2rad(scale * joint_pos[-1] / 255.0 * 45.0)
            )

        self.urdf_graph.set_data()

    def update_time_idx(self, delta_time_idx):
        self.time_idx = np.clip(
            self.time_idx + delta_time_idx,
            0,
            len(self.data_manager_list[self.data_idx].get_data_seq(DataKey.TIME)) - 1,
        )

    def update_data_idx(self, delta_data_idx):
        self.data_idx = np.clip(
            self.data_idx + delta_data_idx, 0, len(self.data_manager_list) - 1
        )
        self.time_idx = 0

    def escape_callback(self, vis, action, mods):
        if action != 1:  # NOT press
            return

        self.quit_flag = True

    def right_callback(self, vis, action, mods):
        if action != 1:  # NOT press
            return

        if mods == 1:  # shift key
            delta_time_idx = 10
        else:
            delta_time_idx = 1

        self.update_time_idx(delta_time_idx)

    def left_callback(self, vis, action, mods):
        if action != 1:  # NOT press
            return

        if mods == 1:  # shift key
            delta_time_idx = -10
        else:
            delta_time_idx = -1

        self.update_time_idx(delta_time_idx)

    def up_callback(self, vis, action, mods):
        if action != 1:  # NOT press
            return

        self.update_data_idx(1)

    def down_callback(self, vis, action, mods):
        if action != 1:  # NOT press
            return

        self.update_data_idx(-1)


if __name__ == "__main__":
    visualize = VisualizeData3D()
    visualize.run()

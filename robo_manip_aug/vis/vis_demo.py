from os import path
import time
import argparse
import numpy as np

import pytransform3d as pytrans3d
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.visualizer as pv

from robo_manip_baselines.common import (
    DataKey,
    DataManager,
)


class VisualizeDemo(object):
    def __init__(self):
        self.setup_args()

        self.setup_data()

        self.setup_visualization()

    def setup_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("teleop_data_path", type=str)
        parser.add_argument("--skip", default=1, type=int, help="skip")
        self.args = parser.parse_args()

    def setup_data(self):
        self.data_manager = DataManager(env=None)
        self.data_manager.load_data(self.args.teleop_data_path)
        time_seq = self.data_manager.get_data(DataKey.TIME)
        self.data_len = len(time_seq)
        self.dt = np.mean(time_seq[1:] - time_seq[:-1])

    def setup_visualization(self):
        # Initialize figure
        self.fig = pv.figure("RoboManipAug VisualizeDemo")
        self.fig.view_init()

        # Load a URDF model of robot
        self.urdf_tm = UrdfTransformManager()
        urdf_dir = path.join(path.dirname(__file__), "../assets/mujoco_ur5e/")
        urdf_path = path.join(urdf_dir, "mujoco_ur5e.urdf")
        with open(urdf_path, "r") as f:
            self.urdf_tm.load_urdf(f.read(), mesh_path=urdf_dir)
        self.urdf_graph = self.fig.plot_graph(self.urdf_tm, "world", show_visuals=True)

        # Draw EEF trajectory
        measured_eef_pose_seq = self.data_manager.get_data(DataKey.MEASURED_EEF_POSE)
        eef_traj_mat_list = np.empty((self.data_len, 4, 4))
        offset_pos = np.array([0.0, 0.0, 0.15])  # [m]
        offset_mat = pytrans3d.transformations.transform_from(
            np.identity(3), offset_pos
        )
        for time_idx in range(self.data_len):
            pos = measured_eef_pose_seq[time_idx, 0:3]
            quat = measured_eef_pose_seq[time_idx, 3:7]
            rot = pytrans3d.rotations.matrix_from_quaternion(quat)
            mat = pytrans3d.transformations.transform_from(rot, pos)
            eef_traj_mat_list[time_idx] = mat @ offset_mat
        traj_color = [0.0, 0.0, 0.0]
        eef_traj = pv.Trajectory(eef_traj_mat_list, c=traj_color)
        self.fig.add_geometry(eef_traj.geometries[0])
        for time_idx in range(self.data_len):
            waypoint_mat = np.identity(4)
            waypoint_radius = 4e-3  # [m]
            waypoint_mat = eef_traj_mat_list[time_idx]
            waypoint_color = [1.0, 0.0, 0.0]
            self.fig.plot_sphere(
                radius=waypoint_radius, A2B=waypoint_mat, c=waypoint_color
            )

        # Set camera pose
        view_ctrl = self.fig.visualizer.get_view_control()
        view_ctrl.set_lookat([0.0, 0.0, 1.4])
        view_ctrl.set_front([1.2, 0.0, 1.8])
        view_ctrl.set_up([0.0, 0.0, 1.0])
        view_ctrl.set_zoom(0.6)

    def main(self):
        for time_idx in range(0, self.data_len, self.args.skip):
            self.update_once(time_idx)

            for geom in self.urdf_graph.geometries:
                self.fig.update_geometry(geom)

            self.fig.visualizer.poll_events()
            self.fig.visualizer.update_renderer()

            time.sleep(self.dt)

        self.fig.show()

    def update_once(self, time_idx):
        measured_joint_pos = self.data_manager.get_single_data(
            DataKey.MEASURED_JOINT_POS, time_idx
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
            self.urdf_tm.set_joint(joint_name, measured_joint_pos[joint_idx])

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
                joint_name, np.deg2rad(scale * measured_joint_pos[-1] / 255.0 * 45.0)
            )

        self.urdf_graph.set_data()


if __name__ == "__main__":
    vis_demo = VisualizeDemo()
    vis_demo.main()

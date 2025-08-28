import argparse
from os import path

import numpy as np
import open3d as o3d
import pytransform3d as pytrans3d
import pytransform3d.visualizer as pv
from pytransform3d.urdf import UrdfTransformManager
from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    get_rot_pos_from_pose,
)


class VisualizeRollout(object):
    def __init__(self):
        self.setup_args()

        self.setup_variables()

        self.setup_data()

        self.setup_visualization()

    def setup_args(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("--proposed_path", type=str)
        parser.add_argument("--baseline_path", type=str)
        parser.add_argument("--human_path", type=str)
        parser.add_argument("--replay_path", type=str)
        parser.add_argument("--point_cloud_path", type=str, default=None)
        parser.add_argument("--plot_measured", action="store_true")
        self.args = parser.parse_args()

    def setup_variables(self):
        eef_offset_pos = np.array([0.0, 0.0, 0.15])  # [m]
        eef_offset_rot = np.identity(3)
        self.eef_offset_mat = pytrans3d.transformations.transform_from(
            eef_offset_rot, eef_offset_pos
        )

    def setup_data(self):
        method_info = {
            "SART (Ours)": self.args.proposed_path,
            "Contact Free MILES": self.args.baseline_path,
            "BC": self.args.human_path,
            "Single Demo Replay": self.args.replay_path,
        }
        self.all_traj = {}
        self.joint_pos = None
        if self.args.plot_measured:
            eef_pose_key = DataKey.MEASURED_EEF_POSE
        else:
            eef_pose_key = DataKey.COMMAND_EEF_POSE
        for method_name, rmb_path in method_info.items():
            if rmb_path is None:
                continue
            with RmbData(rmb_path) as rmb_data:
                self.all_traj[method_name] = rmb_data[eef_pose_key][:]
                if self.joint_pos is None:
                    self.joint_pos = rmb_data[DataKey.COMMAND_JOINT_POS][0]

    def setup_visualization(self):
        # Initialize figure
        self.fig = pv.figure("RoboManipAug VisualizeRollout", with_key_callbacks=True)
        self.fig.view_init()

        # Load a URDF model of robot
        self.urdf_tm = UrdfTransformManager()
        urdf_dir = path.join(path.dirname(__file__), "../assets/mujoco_ur5e/")
        urdf_path = path.join(urdf_dir, "mujoco_ur5e.urdf")
        with open(urdf_path, "r") as f:
            self.urdf_tm.load_urdf(f.read(), mesh_path=urdf_dir)
        self.urdf_graph = self.fig.plot_graph(self.urdf_tm, "world", show_visuals=True)
        self.set_joint_pos()

        # Draw EEF trajectory
        for method_name, eef_pose_seq in self.all_traj.items():
            seq_len = len(eef_pose_seq)
            eef_traj_mat_seq = np.empty((seq_len, 4, 4))
            for time_idx in range(seq_len):
                eef_pose = eef_pose_seq[time_idx]
                eef_mat = pytrans3d.transformations.transform_from(
                    *get_rot_pos_from_pose(eef_pose)
                )
                eef_traj_mat_seq[time_idx] = eef_mat @ self.eef_offset_mat
            traj_color = [0.0, 0.0, 0.0]
            eef_traj = pv.Trajectory(eef_traj_mat_seq, c=traj_color)
            self.fig.add_geometry(eef_traj.geometries[0])
            for time_idx in range(seq_len):
                waypoint_mat = np.identity(4)
                waypoint_radius = 5e-3  # [m]
                if method_name == "SART (Ours)":
                    waypoint_color = [1.0, 0.0, 0.0]
                elif method_name == "Contact Free MILES":
                    waypoint_color = [0.0, 1.0, 0.0]
                elif method_name == "BC":
                    waypoint_color = [0.8, 0.8, 0.0]
                elif method_name == "Single Demo Replay":
                    waypoint_color = [0.0, 0.0, 1.0]
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
        view_ctrl.set_lookat(
            [0.558179720447653, -1.7352158757130276, 1.473582109591307]
        )
        view_ctrl.set_front(
            [0.40564832335049433, -0.8887510421392286, 0.2134737053113755]
        )
        view_ctrl.set_up(
            [-0.06792247271831278, 0.20359614352599106, 0.9766960366670759]
        )
        view_ctrl.set_zoom(0.6)

    def run(self):
        self.fig.show()

    def set_joint_pos(self):
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
            self.urdf_tm.set_joint(joint_name, self.joint_pos[joint_idx])

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
                joint_name, np.deg2rad(scale * self.joint_pos[-1] / 255.0 * 45.0)
            )

        self.urdf_graph.set_data()

        for geom in self.urdf_graph.geometries:
            self.fig.update_geometry(geom)


if __name__ == "__main__":
    visualize = VisualizeRollout()
    visualize.run()

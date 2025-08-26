import argparse
import glob
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


class VisualizeTrainingData(object):
    def __init__(self):
        self.setup_args()

        self.setup_variables()

        self.setup_data()

        self.setup_visualization()

    def setup_args(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("rmb_dir", type=str)
        parser.add_argument(
            "--base_demo_path",
            type=str,
            default=None,
            help="path of teleoperation data as base for data augmentation",
        )
        parser.add_argument("--point_cloud_path", type=str, default=None)
        self.args = parser.parse_args()

    def setup_variables(self):
        eef_offset_pos = np.array([0.0, 0.0, 0.15])  # [m]
        eef_offset_rot = np.identity(3)
        self.eef_offset_mat = pytrans3d.transformations.transform_from(
            eef_offset_rot, eef_offset_pos
        )

    def setup_data(self):
        rmb_path_list = glob.glob(
            path.join(self.args.rmb_dir, "**/*.rmb"), recursive=True
        )
        rmb_path_list.sort()

        self.joint_pos = None

        self.all_traj = []
        for rmb_path in rmb_path_list:
            with RmbData(rmb_path) as rmb_data:
                self.all_traj.append(rmb_data[DataKey.COMMAND_EEF_POSE][:])
                if self.args.base_demo_path is None and self.joint_pos is None:
                    self.joint_pos = rmb_data[DataKey.COMMAND_JOINT_POS][0]

        self.base_traj = None
        if self.args.base_demo_path is not None:
            with RmbData(self.args.base_demo_path) as rmb_data:
                self.base_traj = rmb_data[DataKey.COMMAND_EEF_POSE][:]
                self.joint_pos = rmb_data[DataKey.COMMAND_JOINT_POS][0]

    def setup_visualization(self):
        # Initialize figure
        self.fig = pv.figure(
            "RoboManipAug VisualizeTrainingData", with_key_callbacks=True
        )
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
        all_traj = self.all_traj
        if self.base_traj is not None:
            all_traj.append(self.base_traj)
        for traj_idx, eef_pose_seq in enumerate(all_traj):
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
                if (self.base_traj is None) or (
                    (self.base_traj is not None) and (traj_idx == len(all_traj) - 1)
                ):
                    if self.base_traj is None:
                        waypoint_radius = 2e-3  # [m]
                    else:
                        waypoint_radius = 4e-3  # [m]
                    waypoint_color = [0.6, 0.2, 0.8]
                else:
                    waypoint_radius = 2e-3  # [m]
                    waypoint_color = [1.0, 0.5, 0.0]
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

        self.fig.visualizer.register_key_callback(ord("P"), self.print_current_view)

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

    def print_current_view(self, vis):
        vc = vis.get_view_control()

        # 可能なら getter を使って“そのまま再現できる”値を出力
        try:
            lookat = vc.get_lookat()
            front = vc.get_front()
            up = vc.get_up()
            zoom = vc.get_zoom()
            print("view_ctrl.set_lookat([{}, {}, {}])".format(*lookat))
            print("view_ctrl.set_front([{}, {}, {}])".format(*front))
            print("view_ctrl.set_up([{}, {}, {}])".format(*up))
            print("view_ctrl.set_zoom({})".format(zoom))
            return False
        except Exception:
            pass  # 古いバージョン等で getter が無い場合は行列から復元

        # フォールバック：行列から front / up / lookat を計算
        cam_params = vc.convert_to_pinhole_camera_parameters()
        extrinsic = cam_params.extrinsic
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]

        # カメラ位置（world）
        cam_pos = -R.T @ t

        # Open3D は「カメラ -Z が視線方向」
        # world での front = R^T @ [0,0,-1]，up = R^T @ [0,-1,0]
        front = -(R.T[:, 2])
        up = -(R.T[:, 1])

        # 正規化（Open3D は unit を期待）
        front = front / np.linalg.norm(front)
        up = up / np.linalg.norm(up)

        # 現在の lookat は extrinsic だけでは一意に決まらないので、
        # 既存の lookat が取得できない環境では“見た目の中心”として
        # 適当に front 方向へ少し先を指す（必要なら後で zoom で調整）
        lookat = cam_pos + front  # 1m 先相当（単位ベクトル）

        # ズームは getter が無い場合は固定値（現在の set_zoom と整合させる）
        print("view_ctrl.set_lookat([{}, {}, {}])".format(*lookat))
        print("view_ctrl.set_front([{}, {}, {}])".format(*front))
        print("view_ctrl.set_up([{}, {}, {}])".format(*up))
        print("view_ctrl.set_zoom(0.6)")
        return False


if __name__ == "__main__":
    visualize = VisualizeTrainingData()
    visualize.run()

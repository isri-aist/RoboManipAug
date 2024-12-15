from os import path
import time
import argparse
import numpy as np

import open3d as o3d
import pytransform3d as pytrans3d
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.visualizer as pv

from robo_manip_baselines.common import DataKey, DataManager


class AcceptableRegion(object):
    def __init__(self, time_idx, width, center, convergent_time_idx=None):
        self.time_idx = time_idx
        self.width = width
        self.center = center
        self.convergent_time_idx = convergent_time_idx

    def make_sphere(self, color):
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.width, resolution=8
        )
        sphere.transform(self.center)
        sphere_lineset = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
        sphere_lineset.colors = o3d.utility.Vector3dVector(
            [color] * len(sphere_lineset.lines)
        )
        return sphere_lineset


class VisualizeDemo(object):
    def __init__(self):
        self.setup_args()

        self.setup_variables()

        self.setup_data()

        self.setup_visualization()

    def setup_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("teleop_data_path", type=str)
        self.args = parser.parse_args()

    def setup_variables(self):
        self.current_time_idx = 0
        self.next_time_idx = None
        self.acceptable_region_list = []
        self.sphere_list = []

    def setup_data(self):
        self.data_manager = DataManager(env=None)
        self.data_manager.load_data(self.args.teleop_data_path)
        time_seq = self.data_manager.get_data(DataKey.TIME)
        self.data_len = len(time_seq)
        self.dt = np.mean(time_seq[1:] - time_seq[:-1])

    def setup_visualization(self):
        # Initialize figure
        self.fig = pv.figure("RoboManipAug VisualizeDemo", with_key_callbacks=True)
        self.fig.view_init()

        # Set key callbacks
        # See https://www.glfw.org/docs/latest/group__keys.html for key numbers
        self.fig.visualizer.register_key_action_callback(256, self.escape_callback)
        self.fig.visualizer.register_key_action_callback(262, self.right_callback)
        self.fig.visualizer.register_key_action_callback(263, self.left_callback)
        self.fig.visualizer.register_key_action_callback(264, self.down_callback)
        self.fig.visualizer.register_key_action_callback(265, self.up_callback)

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
        self.eef_traj = pv.Trajectory(eef_traj_mat_list, c=traj_color)
        self.fig.add_geometry(self.eef_traj.geometries[0])
        for time_idx in range(self.data_len):
            waypoint_mat = np.identity(4)
            waypoint_radius = 2e-3  # [m]
            waypoint_mat = eef_traj_mat_list[time_idx]
            waypoint_color = [0.0, 1.0, 0.0]
            waypoint_sphere = pv.Sphere(
                radius=waypoint_radius, A2B=waypoint_mat, c=waypoint_color
            )
            self.fig.add_geometry(waypoint_sphere.geometries[0])

        # Set camera pose
        view_ctrl = self.fig.visualizer.get_view_control()
        view_ctrl.set_lookat([0.0, 0.0, 1.4])
        view_ctrl.set_front([1.2, 0.0, 1.8])
        view_ctrl.set_up([0.0, 0.0, 1.0])
        view_ctrl.set_zoom(0.6)

        # Draw acceptable sphere
        self.acceptable_width = 0.04  # [m]
        self.acceptable_sphere_lineset = None
        self.update_acceptable_sphere()

    def update_acceptable_scale(self, delta, action, mods):
        if action == 0:  # release
            return
        if mods == 1:  # shift key
            scale = 5.0
        elif mods == 2:  # ctrl key
            scale = 0.2
        else:
            scale = 1.0
        self.acceptable_width = np.clip(
            self.acceptable_width + scale * delta, 1e-3, 1.0
        )

        self.update_acceptable_sphere()

    def update_acceptable_sphere(self):
        # Calculate next time index
        current_center_mat = self.eef_traj.H[self.current_time_idx]
        current_center_pos = current_center_mat[0:3, 3]
        current_acceptable_region = AcceptableRegion(
            self.current_time_idx, self.acceptable_width, current_center_mat
        )
        if self.current_time_idx < self.data_len - 1:
            subseq_start_time_idx = self.current_time_idx + 1
            rel_pos_subseq = (
                self.eef_traj.H[subseq_start_time_idx:, 0:3, 3] - current_center_pos
            )
            satisfied_time_idxes = np.argwhere(
                np.linalg.norm(rel_pos_subseq, axis=1) > self.acceptable_width
            )
            if len(satisfied_time_idxes) == 0:
                self.next_time_idx = self.data_len - 1
            else:
                self.next_time_idx = subseq_start_time_idx + satisfied_time_idxes[0, 0]
        else:
            self.next_time_idx = None

        # Draw spheres
        for sphere in self.sphere_list:
            self.fig.visualizer.remove_geometry(sphere, reset_bounding_box=False)
        self.sphere_list = []
        acceptable_region_list = self.acceptable_region_list + [
            current_acceptable_region
        ]
        for acceptable_region_idx, acceptable_region in enumerate(
            acceptable_region_list
        ):
            if acceptable_region_idx == len(self.acceptable_region_list):
                if self.next_time_idx is None:
                    continue
                else:
                    sphere_color = [1.0, 0.6, 0.0]
            else:
                sphere_color = [1.0, 0.0, 0.0]
            sphere = acceptable_region.make_sphere(sphere_color)
            self.fig.visualizer.add_geometry(sphere, reset_bounding_box=False)
            self.sphere_list.append(sphere)

    def escape_callback(self, vis, action, mods):
        if action != 1:  # NOT press
            return

        self.quit_flag = True

    def right_callback(self, vis, action, mods):
        if action != 1:  # NOT press
            return

        if self.next_time_idx is None:
            return

        # Store acceptable region
        current_center_mat = self.eef_traj.H[self.current_time_idx]
        self.acceptable_region_list.append(
            AcceptableRegion(
                self.current_time_idx,
                self.acceptable_width,
                current_center_mat,
                self.next_time_idx,
            )
        )

        # Increment time index
        self.current_time_idx = self.next_time_idx

        self.update_acceptable_sphere()

    def left_callback(self, vis, action, mods):
        if action != 1:  # NOT press
            return

        if len(self.acceptable_region_list) == 0:
            return

        # Decrement time index
        if len(self.acceptable_region_list) == 1:
            self.current_time_idx = 0
        else:
            self.current_time_idx = self.acceptable_region_list[-1].time_idx
        self.acceptable_width = self.acceptable_region_list[-1].width

        # Remove acceptable region
        self.acceptable_region_list.pop(-1)

        self.update_acceptable_sphere()

    def up_callback(self, vis, action, mods):
        delta = 0.02
        self.update_acceptable_scale(delta, action, mods)

    def down_callback(self, vis, action, mods):
        delta = -0.02
        self.update_acceptable_scale(delta, action, mods)

    def main(self):
        self.quit_flag = False
        while not self.quit_flag:
            self.update_once()

            for geom in self.urdf_graph.geometries:
                self.fig.update_geometry(geom)

            self.fig.visualizer.poll_events()
            self.fig.visualizer.update_renderer()

            time.sleep(0.01)

    def update_once(self):
        measured_joint_pos = self.data_manager.get_single_data(
            DataKey.MEASURED_JOINT_POS, self.current_time_idx
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

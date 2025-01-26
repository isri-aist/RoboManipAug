import argparse
import os
from os import path

import numpy as np
import open3d as o3d
from robo_manip_baselines.common import (
    DataKey,
    DataManager,
    convert_depth_image_to_point_cloud,
    get_pose_from_rot_pos,
    get_rot_pos_from_pose,
)
from tqdm import tqdm

from robo_manip_aug import get_trans_from_rot_pos


class GenerateMergedPointCloud(object):
    def __init__(self):
        self.setup_args()

        self.setup_data()

    def setup_args(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("teleop_data_path", type=str)
        parser.add_argument("point_cloud_path", type=str)
        parser.add_argument("--skip", type=int, default=5)
        parser.add_argument("--image_skip", type=int, default=10)
        parser.add_argument("--near_clip", type=float, default=0.2)
        parser.add_argument("--far_clip", type=float, default=2.0)
        parser.add_argument("--voxel_size", type=float, default=0.005)
        self.args = parser.parse_args()

    def setup_data(self):
        self.data_manager = DataManager(env=None)

        load_keys = [
            DataKey.MEASURED_EEF_POSE,
            DataKey.get_rgb_image_key("hand"),
            DataKey.get_depth_image_key("hand"),
        ]
        self.data_manager.load_data(self.args.teleop_data_path, load_keys)

        # Check the calculation of the transformation from end-effector to camera from the MuJoCo model
        self.trans_from_eef_to_camera = get_trans_from_rot_pos(
            o3d.geometry.get_rotation_matrix_from_xyz([0.0, 0.0, np.pi]),
            np.array([0.0, 0.05, 0.0]),
        )

    def run(self):
        eef_pose_seq = self.data_manager.get_data_seq(DataKey.MEASURED_EEF_POSE)[
            :: self.args.skip
        ]
        rgb_image_seq = self.data_manager.get_data_seq(
            DataKey.get_rgb_image_key("hand")
        )[:: self.args.skip, :: self.args.image_skip, :: self.args.image_skip]
        depth_image_seq = self.data_manager.get_data_seq(
            DataKey.get_depth_image_key("hand")
        )[:: self.args.skip, :: self.args.image_skip, :: self.args.image_skip]
        fovy = self.data_manager.get_meta_data(
            DataKey.get_depth_image_key("hand") + "_fovy"
        )

        merged_point_cloud = o3d.geometry.PointCloud()
        for eef_pose, rgb_image, depth_image in tqdm(
            zip(eef_pose_seq, rgb_image_seq, depth_image_seq),
            total=len(eef_pose_seq),
            desc="[GenerateMergedPointCloud] Merge point clouds",
        ):
            xyz_array, rgb_array = convert_depth_image_to_point_cloud(
                depth_image,
                fovy=fovy,
                rgb_image=rgb_image,
                near_clip=self.args.near_clip,
                far_clip=self.args.far_clip,
            )

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(xyz_array)
            point_cloud.colors = o3d.utility.Vector3dVector(rgb_array)

            trans = (
                get_trans_from_rot_pos(*get_rot_pos_from_pose(eef_pose))
                @ self.trans_from_eef_to_camera
            )
            point_cloud.transform(trans)

            merged_point_cloud += point_cloud

        downsampled_point_cloud = merged_point_cloud.voxel_down_sample(
            voxel_size=self.args.voxel_size
        )

        print(
            f"[GenerateMergedPointCloud] Save merged point cloud to {self.args.point_cloud_path}"
        )
        os.makedirs(os.path.dirname(self.args.point_cloud_path), exist_ok=True)
        o3d.io.write_point_cloud(self.args.point_cloud_path, downsampled_point_cloud)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(downsampled_point_cloud)
        opt = vis.get_render_option()
        opt.point_size = 2000.0 * self.args.voxel_size
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    gen_pc = GenerateMergedPointCloud()
    gen_pc.run()

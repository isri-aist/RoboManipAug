import argparse

import open3d as o3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("point_cloud_path", type=str)
    args = parser.parse_args()

    point_cloud = o3d.io.read_point_cloud(args.point_cloud_path)
    o3d.visualization.draw_geometries([point_cloud])

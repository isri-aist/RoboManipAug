import argparse
import open3d as o3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_path", type=str)
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply_path)
    o3d.visualization.draw_geometries([pcd])

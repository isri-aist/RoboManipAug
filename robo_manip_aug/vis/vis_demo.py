from os import path
import time
import numpy as np

from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.visualizer as pv


class VisualizeDemo(object):
    def __init__(self):
        self.urdf_tm = UrdfTransformManager()
        urdf_dir = path.join(path.dirname(__file__), "../assets/mujoco_ur5e/")
        urdf_path = path.join(urdf_dir, "mujoco_ur5e.urdf")
        with open(urdf_path, "r") as f:
            self.urdf_tm.load_urdf(f.read(), mesh_path=urdf_dir)

        self.fig = pv.figure("RoboManipAug VisualizeDemo")
        self.fig.view_init()
        self.fig.set_zoom(1.2)
        self.urdf_graph = self.fig.plot_graph(self.urdf_tm, "world", show_visuals=True)

    def main(self):
        idx = 0
        while True:
            idx += 1
            angle = np.sin(idx / 100.0)
            self.urdf_tm.set_joint("elbow_joint", angle)
            self.urdf_graph.set_data()

            for geom in self.urdf_graph.geometries:
                self.fig.update_geometry(geom)

            self.fig.visualizer.poll_events()
            self.fig.visualizer.update_renderer()

            time.sleep(0.1)


if __name__ == "__main__":
    vis_demo = VisualizeDemo()
    vis_demo.main()

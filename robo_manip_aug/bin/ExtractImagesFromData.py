import argparse
import os

import cv2
from robo_manip_baselines.common import DataKey, DataManager


class ExtractImagesFromData(object):
    def __init__(self):
        self.setup_args()

        self.setup_data()

    def setup_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("teleop_data_path", type=str)
        parser.add_argument("--out_dir", type=str, default="env_data")
        parser.add_argument("--camera_name", type=str, default="hand")
        parser.add_argument("--skip", type=int, default=6)
        self.args = parser.parse_args()

    def setup_data(self):
        self.data_manager = DataManager(env=None)
        self.data_manager.load_data(self.args.teleop_data_path)

    def run(self):
        image_seq = self.data_manager.get_data_seq(
            DataKey.get_rgb_image_key(self.args.camera_name)
        )[:: self.args.skip]
        # Crop images to exclude gripper
        image_seq = image_seq[:, :335, :, :]

        os.makedirs(self.args.out_dir, exist_ok=True)
        for image_idx, image in enumerate(image_seq):
            cv2.imwrite(
                f"{self.args.out_dir}/{image_idx+1:06}.jpg",
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )


if __name__ == "__main__":
    extract = ExtractImagesFromData()
    extract.run()

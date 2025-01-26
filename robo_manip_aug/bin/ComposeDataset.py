import argparse
import glob
import os
from os import path


class ComposeDataset(object):
    def __init__(self):
        self.setup_args()

    def setup_args(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("augmented_data_dir", type=str)
        parser.add_argument("learning_data_dir", type=str)
        parser.add_argument("--num_data_per_region", type=int, default=None)
        self.args = parser.parse_args()

    def run(self):
        os.makedirs(self.args.learning_data_dir, exist_ok=True)

        base_demo_filename = "base_demo.hdf5"
        src_base_demo_path = os.path.join(
            self.args.augmented_data_dir, base_demo_filename
        )
        dest_base_demo_path = os.path.join(
            self.args.learning_data_dir, base_demo_filename
        )
        if not os.path.exists(src_base_demo_path):
            raise RuntimeError(
                f"[ComposeDataset] Base demo file not found: {src_base_demo_path}"
            )
        os.symlink(path.abspath(src_base_demo_path), dest_base_demo_path)

        for subdir in os.listdir(self.args.augmented_data_dir):
            src_subdir_path = os.path.join(self.args.augmented_data_dir, subdir)

            if not os.path.isdir(src_subdir_path):
                continue

            dest_subdir_path = os.path.join(self.args.learning_data_dir, subdir)
            os.makedirs(dest_subdir_path, exist_ok=True)

            filename_list = sorted(os.listdir(src_subdir_path))
            if self.args.num_data_per_region is not None:
                filename_list = filename_list[: self.args.num_data_per_region]

            print(
                f"[ComposeDataset] Create {len(filename_list)} symbolic links in {dest_subdir_path}"
            )
            for filename in filename_list:
                src_file_path = os.path.join(src_subdir_path, filename)
                dest_file_path = os.path.join(dest_subdir_path, filename)

                if os.path.exists(dest_file_path):
                    raise RuntimeError(
                        f"[ComposeDataset] File already exists: {dest_file_path}"
                    )
                else:
                    os.symlink(path.abspath(src_file_path), dest_file_path)
                    # print(
                    #     f"[ComposeDataset] Make a symbolic link: {src_file_path} -> {dest_file_path}"
                    # )


if __name__ == "__main__":
    compose = ComposeDataset()
    compose.run()

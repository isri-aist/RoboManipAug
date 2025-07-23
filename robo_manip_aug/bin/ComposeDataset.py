import argparse
import os
import random
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
        parser.add_argument("--num_total_data", type=int, default=None)
        parser.add_argument("--seed", type=int, default=None)
        self.args = parser.parse_args()

    def run(self):
        os.makedirs(self.args.learning_data_dir, exist_ok=True)

        base_demo_filename = "base_demo.rmb"
        src_base_demo_path = os.path.join(
            self.args.augmented_data_dir, base_demo_filename
        )
        dest_base_demo_path = os.path.join(
            self.args.learning_data_dir, base_demo_filename
        )
        if not os.path.exists(src_base_demo_path):
            raise RuntimeError(
                f"[{self.__class__.__name__}] Base demo file not found: {src_base_demo_path}"
            )
        os.symlink(path.abspath(src_base_demo_path), dest_base_demo_path)

        all_file_entries = []  # list of (src_file_path, dest_file_path) pairs

        for subdir in os.listdir(self.args.augmented_data_dir):
            if base_demo_filename in subdir:
                continue

            src_subdir_path = os.path.join(self.args.augmented_data_dir, subdir)
            if not os.path.isdir(src_subdir_path):
                continue

            dest_subdir_path = os.path.join(self.args.learning_data_dir, subdir)

            filename_list = sorted(os.listdir(src_subdir_path))
            if self.args.num_data_per_region is not None:
                filename_list = filename_list[: self.args.num_data_per_region]

            for filename in filename_list:
                src_file_path = os.path.join(src_subdir_path, filename)
                dest_file_path = os.path.join(dest_subdir_path, filename)
                all_file_entries.append((src_file_path, dest_file_path))

        # If num_total_data is specified, randomly sample that number of files
        if self.args.num_total_data is not None:
            if self.args.seed is not None:
                random.seed(self.args.seed)
            if self.args.num_total_data > len(all_file_entries):
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Requested num_total_data ({self.args.num_total_data}) exceeds available files ({len(all_file_entries)})"
                )
            all_file_entries = random.sample(all_file_entries, self.args.num_total_data)

        # Create symbolic links
        for src_file_path, dest_file_path in all_file_entries:
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
            if os.path.exists(dest_file_path):
                raise RuntimeError(
                    f"[{self.__class__.__name__}] File already exists: {dest_file_path}"
                )
            os.symlink(path.abspath(src_file_path), dest_file_path)

        print(
            f"[{self.__class__.__name__}] Created symbolic links for 1 base trajectory and {len(all_file_entries)} augmented trajectories."
        )


if __name__ == "__main__":
    compose = ComposeDataset()
    compose.run()

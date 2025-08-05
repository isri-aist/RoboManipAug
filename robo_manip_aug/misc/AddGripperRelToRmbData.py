import argparse

import numpy as np
from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "path",
        type=str,
        help="path to data (*.hdf5 or *.rmb) or directory containing them",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether to overwrite existing value if it exists",
    )

    return parser.parse_args()


class AddGripperRelToRmbData:
    def __init__(self, path, overwrite=False):
        self.path = path
        self.overwrite = overwrite

    def run(self):
        rmb_path_list = find_rmb_files(self.path)
        for rmb_path in rmb_path_list:
            print(f"[{self.__class__.__name__}] Open {rmb_path}")
            with RmbData(rmb_path, mode="r+") as rmb_data:
                for rel_key in (
                    DataKey.MEASURED_GRIPPER_JOINT_POS_REL,
                    DataKey.COMMAND_GRIPPER_JOINT_POS_REL,
                ):
                    if rel_key in rmb_data.keys():
                        if self.overwrite:
                            del rmb_data.h5file[rel_key]
                        else:
                            raise ValueError(
                                f"[{self.__class__.__name__}] {rel_key} already exists in RMB data (use --overwrite to replace)"
                            )

                    abs_key = DataKey.get_abs_key(rel_key)
                    abs_data = rmb_data[abs_key][:]
                    rel_data = np.concatenate(
                        [
                            np.zeros_like(abs_data[0])[np.newaxis],
                            abs_data[1:] - abs_data[:-1],
                        ]
                    )
                    rmb_data.h5file[rel_key] = rel_data


if __name__ == "__main__":
    add = AddGripperRelToRmbData(**vars(parse_argument()))
    add.run()

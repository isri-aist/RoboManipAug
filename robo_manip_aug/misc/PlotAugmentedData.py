import argparse

import matplotlib.pylab as plt
from robo_manip_baselines.common import DataKey, RmbData


class PlotAugmentedData(object):
    def __init__(self):
        self.setup_args()

        self.setup_data()

    def setup_args(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        parser.add_argument("base_rmb_path", type=str)
        parser.add_argument("aug_rmb_path", type=str)

        parser.add_argument(
            "--data_key",
            type=str,
            default="joint_pos",
            choices=["joint_pos", "eef_pose"],
        )

        self.args = parser.parse_args()

    def setup_data(self):
        self.base_rmb_data = RmbData(self.args.base_rmb_path)
        self.base_rmb_data.open()
        self.aug_rmb_data = RmbData(self.args.aug_rmb_path)
        self.aug_rmb_data.open()

    def run(self):
        measured_data_key = "measured_" + self.args.data_key
        command_data_key = "command_" + self.args.data_key

        base_time = self.base_rmb_data[DataKey.TIME][:]
        aug_time = self.aug_rmb_data[DataKey.TIME][:]
        time_shift = base_time[-1] - aug_time[-1]
        aug_time_shifted = aug_time + time_shift

        colors = {
            (measured_data_key, "base"): "lightcoral",
            (measured_data_key, "aug"): "firebrick",
            (command_data_key, "base"): "mediumseagreen",
            (command_data_key, "aug"): "forestgreen",
        }
        linestyles = {
            measured_data_key: "-",
            command_data_key: "--",
        }

        for data_key in [command_data_key, measured_data_key]:
            plt.plot(
                base_time,
                self.base_rmb_data[data_key][:, 1],
                label=f"base_{data_key}",
                color=colors[(data_key, "base")],
                linestyle=linestyles[data_key],
                linewidth=5.0,
                marker="o",
                markersize=8,
                alpha=0.4,
                zorder=1,
            )
            plt.plot(
                aug_time_shifted,
                self.aug_rmb_data[data_key][:, 1],
                label=f"aug_{data_key}",
                color=colors[(data_key, "aug")],
                linestyle=linestyles[data_key],
                linewidth=2.5,
                marker="x",
                markersize=6,
                alpha=1.0,
                zorder=2,
            )

        plt.legend(fontsize=12)
        plt.show()


if __name__ == "__main__":
    plot = PlotAugmentedData()
    plot.run()

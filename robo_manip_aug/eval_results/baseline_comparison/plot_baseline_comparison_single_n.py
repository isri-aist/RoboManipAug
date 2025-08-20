import argparse

import matplotlib
import numpy as np
import yaml
from matplotlib import pyplot as plt

success_score_map = {"1": 1, "0": 0, ">": 0.5}


def compute_success_stats(data_by_N, N):
    """
    Process rollouts for single N and return mean success rate and standard deviation.
    """
    for entry in data_by_N:
        if N == entry["N"]:
            success_rates = [
                np.mean([success_score_map[c] for c in rollout])
                for rollout in entry["Rollouts"]
            ]
            return np.mean(success_rates), np.std(success_rates)


def main():
    parser = argparse.ArgumentParser(description="Plot success rates from YAML file.")
    parser.add_argument("yaml_file", type=str, help="Path to input YAML file")
    parser.add_argument(
        "target_N",
        type=int,
        default=30,
    )
    parser.add_argument("--output_pdf", type=str, help="Output PDF filename")
    parser.add_argument(
        "--ignore_partial_success",
        action="store_true",
        help="Whether calculating > as false instead of partial_success or not",
    )
    args = parser.parse_args()

    # Load YAML
    with open(args.yaml_file, "r") as f:
        data = yaml.safe_load(f)

    # Get statistics for each method
    plot_mean = []
    plot_std = []
    plot_label = []
    N = args.target_N
    if args.ignore_partial_success:
        success_score_map[">"] = 0
    proposed_mean, proposed_std = compute_success_stats(data["Proposed"], N)
    plot_mean.append(proposed_mean)
    plot_std.append(proposed_std)
    plot_label.append("Proposed")
    if "Baseline" in data.keys():
        baseline_mean, baseline_std = compute_success_stats(data["Baseline"], N)
        plot_mean.append(baseline_mean)
        plot_std.append(baseline_std)
        plot_label.append("Baseline")
    if "Human" in data.keys():
        human_mean, human_std = compute_success_stats(data["Human"], N)
        plot_mean.append(human_mean)
        plot_std.append(human_std)
        plot_label.append("Human")
    # Replay (calculate only mean success rate)
    replay_success_rates = [
        np.mean([success_score_map[c] for c in rollout])
        for rollout in data["Replay"][0]["Rollouts"]
    ]
    replay_mean = np.mean(replay_success_rates)
    plot_mean.append(replay_mean)
    plot_label.append("Replay")

    # Plot settings
    plt.figure(figsize=(10, 6))

    policy_colors = {
        "Proposed": "r",
        "Baseline": "g",
        "Human": "y",
        "Replay": "b",
    }

    # Plot Data
    if "Replay" in plot_label:
        plot_std.append(0)
    plot_color = [policy_colors[label] for label in plot_label]
    plt.bar(plot_label, np.array(plot_mean).reshape(4), color=plot_color, alpha=0.7)
    plt.errorbar(
        plot_label,
        plot_mean,
        plot_std,
        capsize=5,
        linestyle=None,
        fmt="x",
        color="k",
    )

    # Axes and labels
    task_name = f": {data['Task']}" if "Task" in data else ""
    plt.ylim(0, 1)
    plt.xlabel("Policy")
    plt.ylabel("Success Rate")
    plt.title("Evaluation of Data Augmentation Methods" + task_name)
    plt.grid(True)
    plt.tight_layout()

    # Use TrueType fonts (avoid Type 3 fonts)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    if args.output_pdf is not None:
        matplotlib.use("pdf")  # use PDF backend to avoid Type 3 fonts
        plt.savefig(args.output_pdf)
        print(f"Saved plot to {args.output_pdf}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

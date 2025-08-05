import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml


def compute_success_stats(data):
    """
    Process rollouts and return mean success rate and standard deviation.
    """
    labels = []
    means = []
    stds = []
    for entry_k in data.keys():
        if "Task" in entry_k:
            continue
        success_rates = [
            np.mean([int(c) for c in rollout]) for rollout in data[entry_k]
        ]
        labels.append(entry_k)
        means.append(np.mean(success_rates))
        stds.append(np.std(success_rates))

    return labels, np.array(means), np.array(stds)


def main():
    parser = argparse.ArgumentParser(description="Plot success rates from YAML file.")
    parser.add_argument("yaml_file", type=str, help="Path to input YAML file")
    parser.add_argument("--output_pdf", type=str, help="Output PDF filename")
    args = parser.parse_args()

    # Load YAML
    with open(args.yaml_file, "r") as f:
        data = yaml.safe_load(f)

    # Get statistics for each condition
    labels, means, stds = compute_success_stats(data)

    # Plot settings
    plt.figure(figsize=(10, 6))

    # Plot bar and errorbar
    bar_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    plt.bar(list(range(len(labels))), means, label=labels, color=bar_colors)
    plt.errorbar(list(range(len(labels))), means, stds, fmt="x", color="C9")
    # Axes and labels
    task_name = f": {data['Task']}" if "Task" in data else ""
    plt.xticks([])
    plt.ylim(0, 1)
    plt.ylabel("Success Rate")
    plt.title("Evaluation of Proposed Data Augmentation Methods" + task_name)
    plt.legend()
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

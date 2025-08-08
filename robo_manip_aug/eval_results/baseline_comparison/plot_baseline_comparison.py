import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

success_score_map = {"1": 1, "0": 0, ">": 0.5}


def compute_success_stats(data_by_N):
    """
    Process rollouts for each N and return mean success rate and standard deviation.
    """
    Ns = []
    means = []
    stds = []
    for entry in data_by_N:
        N = entry["N"]
        success_rates = [
            np.mean([success_score_map[c] for c in rollout])
            for rollout in entry["Rollouts"]
        ]
        Ns.append(N)
        means.append(np.mean(success_rates))
        stds.append(np.std(success_rates))

    return np.array(Ns), np.array(means), np.array(stds)


def main():
    parser = argparse.ArgumentParser(description="Plot success rates from YAML file.")
    parser.add_argument("yaml_file", type=str, help="Path to input YAML file")
    parser.add_argument("--output_pdf", type=str, help="Output PDF filename")
    parser.add_argument(
        "--gt_is_false",
        action="store_true",
        help="Whether calculating > as false instead of partial_success or not",
    )
    args = parser.parse_args()

    # Load YAML
    with open(args.yaml_file, "r") as f:
        data = yaml.safe_load(f)

    # Get statistics for each method
    if args.gt_is_false:
        success_score_map[">"] = 0
    proposed_N, proposed_mean, proposed_std = compute_success_stats(data["Proposed"])
    if "Baseline" in data.keys():
        baseline_N, baseline_mean, baseline_std = compute_success_stats(
            data["Baseline"]
        )
    if "Human" in data.keys():
        human_N, human_mean, human_std = compute_success_stats(data["Human"])

    # Replay (calculate only mean success rate)
    replay_success_rates = [
        np.mean([success_score_map[c] for c in rollout])
        for rollout in data["Replay"][0]["Rollouts"]
    ]
    replay_mean = np.mean(replay_success_rates)

    # Plot settings
    plt.figure(figsize=(10, 6))

    # Proposed (red)
    plt.plot(proposed_N, proposed_mean, "r-o", label="Proposed")
    plt.fill_between(
        proposed_N,
        proposed_mean - proposed_std,
        proposed_mean + proposed_std,
        color="red",
        alpha=0.2,
    )

    # Baseline (green)
    if "Baseline" in data.keys():
        plt.plot(baseline_N, baseline_mean, "g-o", label="Baseline")
        plt.fill_between(
            baseline_N,
            baseline_mean - baseline_std,
            baseline_mean + baseline_std,
            color="green",
            alpha=0.2,
        )

    if "Human" in data.keys():
        plt.plot(human_N, human_mean, "y-o", label="Human")
        plt.fill_between(
            human_N,
            human_mean - human_std,
            human_mean + human_std,
            color="yellow",
            alpha=0.2,
        )

    # Replay (horizontal blue line)
    plt.axhline(y=replay_mean, color="blue", linestyle="--", label="Replay")

    # Axes and labels
    task_name = f": {data['Task']}" if "Task" in data else ""
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.xlabel("N")
    plt.ylabel("Success Rate")
    plt.title("Evaluation of Data Augmentation Methods" + task_name)
    plt.grid(True)
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

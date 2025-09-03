import argparse
import glob
import os
from scipy import stats
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
    parser = argparse.ArgumentParser(description="Plot success rates from multiple YAML files in a directory.")
    parser.add_argument("src_dir", type=str, help="Directory containing YAML files.")
    parser.add_argument("--output_pdf", type=str, help="Output PDF filename")
    args = parser.parse_args()

    yaml_files = sorted([f for f in glob.glob(os.path.join(args.src_dir, "*.yaml")) 
                        if "Real" not in f])
    if not yaml_files:
        print(f"No YAML files found in directory: {args.src_dir}")
        return

    # Aggregate stats for each method and N
    method_stats = {}
    replay_means = []
    for file_path in yaml_files:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        for method in ["Proposed", "Baseline", "Human"]:
            if method in data:
                Ns, means, stds = compute_success_stats(data[method])[:3]
                means = means * 100  # Convert to percentage
                stds = stds * 100    # Convert standard deviation to percentage
                for i, N in enumerate(Ns):
                    if method not in method_stats:
                        method_stats[method] = {}
                    if N not in method_stats[method]:
                        method_stats[method][N] = {"means": [], "stds": [], "raw_rates": []}
                    method_stats[method][N]["means"].append(means[i])
                    method_stats[method][N]["stds"].append(stds[i])
                    # Store raw success rates for statistical testing
                    success_rates = [np.mean([success_score_map[c] for c in rollout]) * 100
                                   for rollout in data[method][i]["Rollouts"]]
                    method_stats[method][N]["raw_rates"].extend(success_rates)

        # Replay
        if "Replay" in data:
            replay_success_rates = [
                np.mean([success_score_map[c] for c in rollout]) * 100
                for rollout in data["Replay"][0]["Rollouts"]
            ]
            replay_means.append(np.mean(replay_success_rates))

    # Perform statistical testing
    significant_results = {}
    for N in sorted(method_stats["Proposed"].keys()):
        proposed_data = method_stats["Proposed"][N]["raw_rates"]
        significant_results[N] = {}
        for method in ["Baseline", "Human"]:
            if method in method_stats and N in method_stats[method]:
                other_data = method_stats[method][N]["raw_rates"]
                t_stat, p_value = stats.ttest_ind(proposed_data, other_data)
                significant_results[N][method] = p_value < 0.05 and np.mean(proposed_data) > np.mean(other_data)

    # Prepare plot
    plt.figure(figsize=(10, 6))
    method_colors = {"Proposed": "red", 
                    "Baseline": "green", 
                    "Human": "yellow"}
    method_labels = {"Proposed": "SART (Ours)", 
                    "Baseline": "Contact Free MILES", 
                    "Human": "BC"}

    for method in method_stats:
        Ns_sorted = sorted(method_stats[method].keys())
        mean_across_files = [np.mean(method_stats[method][N]["means"]) for N in Ns_sorted]
        std_across_files = [np.mean(method_stats[method][N]["stds"]) for N in Ns_sorted]
        plt.plot(Ns_sorted, mean_across_files, "-o", color=method_colors[method], label=method_labels[method])
        plt.fill_between(Ns_sorted,
                         np.array(mean_across_files) - np.array(std_across_files),
                         np.array(mean_across_files) + np.array(std_across_files),
                         color=method_colors[method], alpha=0.2)

    # Print significant differences
    print("\nSignificant improvements (p < 0.05):")
    for N in sorted(significant_results.keys()):
        improvements = [method for method, is_better in significant_results[N].items() if is_better]
        if improvements:
            print(f"N={N}: Proposed significantly better than {', '.join(improvements)}")
        # Add star annotation if significantly better than both methods
        if len(improvements) == 2:
            plt.text(N, 100., '*', fontsize=20, ha='center')

    # Replay (horizontal blue line)
    if replay_means:
        plt.axhline(y=np.mean(replay_means), color="blue", linestyle="--", label="Single Demo Replay")

    # Axes and labels
    plt.xlim(0, 105)
    plt.ylim(0, 105)
    plt.xlabel("N", fontsize=20)
    plt.ylabel("Success Rate [%]", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.title("Evaluation of Data Augmentation Methods (Aggregated)")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', fontsize=15, ncol=4)
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

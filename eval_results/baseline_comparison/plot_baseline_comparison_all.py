import argparse
import glob
import os

import numpy as np
import yaml
from scipy import stats

success_score_map = {"1": 1, "0": 0, ">": 0.0}


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
    parser = argparse.ArgumentParser(
        description="Plot success rates from multiple YAML files in a directory."
    )
    parser.add_argument("src_dir", type=str, help="Directory containing YAML files.")
    parser.add_argument("--output_pdf", type=str, help="Output PDF filename")
    args = parser.parse_args()

    yaml_files = sorted(
        [
            f
            for f in glob.glob(os.path.join(args.src_dir, "*.yaml"))
            if any(
                x in f for x in ["Mujoco", "RealCloseLid", "RealPutBottle_ObjPosWide"]
            )
        ]
    )
    yaml_files = [
        f
        for f in yaml_files
        if "MujocoDoor_ObjPos005" not in f and "MujocoCabinetHinge_ObjPos005" not in f
    ]

    if not yaml_files:
        print(f"No YAML files found in directory: {args.src_dir}")
        return

    # Aggregate stats for each method and N
    method_stats = {}
    replay_means = {}  # Changed to dict to store by task
    replay_stds = {}  # To store standard deviations for each task
    for file_path in yaml_files:
        task_name = os.path.basename(file_path).split("_")[2]  # Extract task name
        task_name_map = {
            "MujocoCabinetHinge": "LidOpening",
            "MujocoDoor": "DoorOpening",
            "MujocoInsert": "Peg-in-hole",
            "MujocoToolboxPick": "ToolboxPicking",
            "RealPickTape": "TapePicking",
            "RealCloseLid": "Lid closing",
            "RealPutBottle": "PutBottle",
        }
        task_name = task_name_map.get(task_name, task_name)
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        for method in ["Proposed", "Baseline", "Human"]:
            if method in data:
                Ns, means, stds = compute_success_stats(data[method])[:3]
                means = means * 100  # Convert to percentage
                stds = stds * 100  # Convert standard deviation to percentage
                for i, N in enumerate(Ns):
                    if method not in method_stats:
                        method_stats[method] = {}
                    if N not in method_stats[method]:
                        method_stats[method][N] = {
                            "means": [],
                            "stds": [],
                            "raw_rates": [],
                            "tasks": [],
                        }
                    method_stats[method][N]["means"].append(means[i])
                    method_stats[method][N]["stds"].append(stds[i])
                    method_stats[method][N]["tasks"].append(task_name)
                    # Store raw success rates for statistical testing
                    success_rates = [
                        np.mean([success_score_map[c] for c in rollout]) * 100
                        for rollout in data[method][i]["Rollouts"]
                    ]
                    method_stats[method][N]["raw_rates"].extend(success_rates)

        # Replay
        if "Replay" in data:
            replay_success_rates_mean = [
                np.mean([success_score_map[c] for c in rollout]) * 100
                for rollout in data["Replay"][0]["Rollouts"]
            ]
            replay_means[task_name] = []
            replay_means[task_name].append(replay_success_rates_mean[0])
            replay_success_rates_std = [
                np.std([success_score_map[c] for c in rollout]) * 100
                for rollout in data["Replay"][0]["Rollouts"]
            ]
            replay_stds[task_name] = []
            replay_stds[task_name].append(replay_success_rates_std[0])

    # Filter methods by specific N values
    for method in list(method_stats.keys()):
        new_stats = {}
        for N in method_stats[method]:
            if N == 40 or N == 30:  # Keep only N=40 (Mujoco) and N=30 (Real)
                new_stats[N] = method_stats[method][N]
        method_stats[method] = new_stats

    # Merge N=30 and N=40 data
    for method in method_stats:
        merged_stats = {"means": [], "stds": [], "raw_rates": [], "tasks": []}
        for N in method_stats[method]:
            merged_stats["means"].extend(method_stats[method][N]["means"])
            merged_stats["stds"].extend(method_stats[method][N]["stds"])
            merged_stats["raw_rates"].extend(method_stats[method][N]["raw_rates"])
            merged_stats["tasks"].extend(method_stats[method][N]["tasks"])
        method_stats[method] = merged_stats

    # print("\nMethod Stats:")
    # for method in method_stats:
    #     print(f"\n{method}:")
    #     for key, value in method_stats[method].items():
    #         if key in ['means', 'stds', 'tasks']:
    #             print(f"  {key}: {value}")
    #         else:
    #             print(f"  {key}: [{len(value)} values]")

    # Update method names in method_stats
    method_name_map = {
        "Replay": "SingleDemoReplay",
        "Human": "BC",
        "Baseline": "ContactFreeMILES",
        "Proposed": "SART(Ours)",
    }

    old_method_stats = method_stats.copy()
    method_stats = {}
    for old_name, new_name in method_name_map.items():
        if old_name in old_method_stats:
            method_stats[new_name] = old_method_stats[old_name]
    print("=" * 50)
    # Print results by task and method
    print("\nResults by Task:")
    print("-" * 50)
    method_order = ["SingleDemoReplay", "BC", "ContactFreeMILES", "SART(Ours)"]

    # Get unique tasks
    tasks = sorted(set(method_stats["SART(Ours)"]["tasks"]))

    for task in tasks:
        print(f"\n{task}:")

        # Replay
        if task in replay_means:
            print(f"Replay: {replay_means[task][0]:.2f}% ± {replay_stds[task][0]:.2f}%")

        # Other methods
        for method in method_order[1:]:  # Skip Replay as it's handled above
            if method in method_stats:
                task_indices = [
                    i for i, t in enumerate(method_stats[method]["tasks"]) if t == task
                ]
                if task_indices:
                    mean_value = np.mean(
                        [method_stats[method]["means"][i] for i in task_indices]
                    )
                    std_value = np.mean(
                        [method_stats[method]["stds"][i] for i in task_indices]
                    )
                    print(f"{method}: {mean_value:.2f}% ± {std_value:.2f}%")
        print("-" * 50)

    # Perform statistical testing
    print("=" * 50)
    print("\nStatistical Testing Results:")
    print("-" * 50)
    significant_results = {}
    proposed_data = method_stats["SART(Ours)"]["raw_rates"]
    significant_results = {}
    for method in ["ContactFreeMILES", "BC"]:
        if method in method_stats:
            other_data = method_stats[method]["raw_rates"]
            t_stat, p_value = stats.ttest_ind(proposed_data, other_data)
            significant_results[method] = p_value < 0.05 and np.mean(
                proposed_data
            ) > np.mean(other_data)
            print(f"\nComparing Proposed vs {method}:")
            print(f"p-value: {p_value}")
            print(
                f"Means - Proposed: {np.mean(proposed_data):.2f}, {method}: {np.mean(other_data):.2f}"
            )
            print(
                f"Standard Deviations - Proposed: {np.std(proposed_data):.2f}, {method}: {np.std(other_data):.2f}"
            )
            print(f"Significant improvement: {significant_results[method]}")
    # import ipdb; ipdb.set_trace()
    # Write results to a text file
    output_file = "sim_real_quanti_comparison_v1.txt"
    with open(output_file, "w") as f:
        f.write("Results by Task:\n")
        f.write("-" * 50 + "\n")

        tasks = sorted(set(method_stats["SART(Ours)"]["tasks"]))
        for task in tasks:
            f.write(f"\n{task}:\n")

            if task in replay_means:
                f.write(f"Replay: {replay_means[task][0]:.2f}%\n")

            for method in method_order[1:]:
                if method in method_stats:
                    task_indices = [
                        i
                        for i, t in enumerate(method_stats[method]["tasks"])
                        if t == task
                    ]
                    if task_indices:
                        mean_value = np.mean(
                            [method_stats[method]["means"][i] for i in task_indices]
                        )
                        std_value = np.mean(
                            [method_stats[method]["stds"][i] for i in task_indices]
                        )
                        f.write(f"{method}: {mean_value:.2f}% \n")
            f.write("-" * 50 + "\n")

        f.write("\nStatistical Testing Results:\n")
        f.write("-" * 50 + "\n")
        for method in ["ContactFreeMILES", "BC"]:
            if method in method_stats:
                other_data = method_stats[method]["raw_rates"]
                t_stat, p_value = stats.ttest_ind(proposed_data, other_data)
                f.write(f"\nComparing Proposed vs {method}:\n")
                f.write(f"p-value: {p_value}\n")
                f.write(
                    f"Proposed: {np.mean(proposed_data):.2f} ± {np.std(proposed_data):.2f}, {method}: {np.mean(other_data):.2f} ± {np.std(other_data):.2f}\n"
                )
                f.write(f"Significant improvement: {significant_results[method]}\n")


if __name__ == "__main__":
    main()

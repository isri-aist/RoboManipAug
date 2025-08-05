import argparse
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import yaml


def load_success_stats(file_path):
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    task = data.get("Task", os.path.basename(file_path))
    accuracies = {}
    for method, results in data.items():
        if method == "Task":
            continue
        method_accuracies = [s.count("1") / len(s) for s in results]
        accuracies[method] = method_accuracies
    return task, accuracies


def plot_accuracies_from_yaml_files(file_paths, output_pdf=None):
    desired_task_order = [
        "Peg-in-Hole",
        "Door Opening",
        "Lid Opening",
        "Toolbox Picking",
    ]
    desired_method_order = [
        "Default",
        "FixedRotation",
        "WithoutMergeBaseDemo",
        "ReturnCenter",
    ]

    method_colors = {
        "Default": "red",
        "FixedRotation": "darkviolet",
        "WithoutMergeBaseDemo": "darkorange",
        "ReturnCenter": "sienna",
    }
    method_labels = {
        "Default": "Ours",
        "FixedRotation": "w/o Rotation Augmentation",
        "WithoutMergeBaseDemo": "w/o Merge Base Demo",
        "ReturnCenter": "Return Center",
    }

    task_method_accs = []
    task_seen = set()

    for file_path in file_paths:
        task, acc_dict = load_success_stats(file_path)
        if task not in desired_task_order:
            continue
        task_seen.add(task)
        for method, accs in acc_dict.items():
            if method not in desired_method_order:
                continue
            task_method_accs.append({"task": task, "method": method, "accs": accs})

    all_tasks = [task for task in desired_task_order if task in task_seen]
    all_methods = [
        m
        for m in desired_method_order
        if any(d["method"] == m for d in task_method_accs)
    ]

    plt.figure(figsize=(14, 6))
    box_width = 0.2
    gap_between_files = 1.0

    xticks = []
    xtick_labels = []
    current_x = 0

    for task in all_tasks:
        method_idx = 0
        for method in all_methods:
            acc_list = next(
                (
                    item["accs"]
                    for item in task_method_accs
                    if item["task"] == task and item["method"] == method
                ),
                None,
            )
            if acc_list:
                pos = current_x + method_idx * box_width
                plt.boxplot(
                    acc_list,
                    positions=[pos],
                    widths=box_width * 0.8,
                    patch_artist=True,
                    medianprops=dict(color="black"),
                    boxprops=dict(facecolor=method_colors.get(method, "gray")),
                )
                method_idx += 1

        if method_idx > 0:
            mid = current_x + (method_idx - 1) * box_width / 2
            xticks.append(mid)
            xtick_labels.append(task)
            current_x += method_idx * box_width + gap_between_files

    plt.xticks(xticks, xtick_labels, rotation=0)
    plt.ylabel("Success Rate")
    plt.title("Ablation Study of Data Augmentation Methods")

    handles = [
        plt.Line2D(
            [],
            [],
            color=method_colors[m],
            marker="s",
            linestyle="None",
            markersize=10,
            label=method_labels[m],
        )
        for m in all_methods
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.0, 1.0), loc="upper left")

    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(
        description="Plot accuracy boxplots from YAML files in a directory."
    )
    parser.add_argument("src_dir", help="Directory containing YAML files.")
    parser.add_argument("--output_pdf", type=str, help="Output PDF filename")
    args = parser.parse_args()

    yaml_files = sorted(glob.glob(os.path.join(args.src_dir, "*.yaml")))
    if not yaml_files:
        print(f"No YAML files found in directory: {args.src_dir}")
        return

    plot_accuracies_from_yaml_files(yaml_files, args.output_pdf)

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

import argparse
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import yaml
from scipy import stats


def load_success_stats(file_path):
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    task = data.get("Task", os.path.basename(file_path))
    accuracies = {}
    for method, results in data.items():
        if method == "Task":
            continue
        method_accuracies = [s.count("1") / len(s) * 100 for s in results]
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
        "FixedPosition",
        "WithoutMergeBaseDemo",
        "ReturnCenter",
    ]

    method_colors = {
        "Default": "red",
        "FixedRotation": "darkviolet",
        "FixedPosition": "deeppink",
        "WithoutMergeBaseDemo": "darkorange",
        "ReturnCenter": "sienna",
    }
    method_labels = {
        "Default": "SART",
        "FixedRotation": "w/o Ori. Aug.",
        "FixedPosition": "w/o Pos. Aug.",
        "WithoutMergeBaseDemo": "w/o Merge Demo",
        "ReturnCenter": "w/ Return Center",
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

    all_methods = [
        m
        for m in desired_method_order
        if any(d["method"] == m for d in task_method_accs)
    ]

    # Define font size
    font_size = 20

    plt.figure(figsize=(14, 6))
    box_width = 0.2
        
    # Initialize dictionaries to store merged accuracies for each method
    merged_method_accs = {method: [] for method in all_methods}

    # Merge accuracies from all tasks for each method
    for item in task_method_accs:
        method = item['method']
        merged_method_accs[method].extend(item['accs'])

    print("\nMerged Method Statistics:")
    for method in all_methods:
        accs = merged_method_accs[method]
        if accs:
            print(f"{method_labels[method]}: mean={sum(accs)/len(accs):.3f}, samples={len(accs)}")

    # Create box plot for merged accuracies
    plt.figure(figsize=(10, 6))
    box_width = 0.6
    positions = range(len(all_methods))
    
    boxes = []
    for i, method in enumerate(all_methods):
        box = plt.boxplot(
            merged_method_accs[method],
            positions=[i],
            widths=box_width,
            patch_artist=True,
            medianprops=dict(color="black"),
            boxprops=dict(facecolor=method_colors.get(method, "gray")),
        )
        boxes.append(box)

    # Remove x-axis ticks and labels
    plt.xticks([])
    plt.xlabel('')
    plt.ylabel("Success Rate [%]", fontsize=20)
    plt.ylim(-1, 105.)
    plt.yticks(fontsize=15)
    
    # Perform t-test between Default (Ours) and other methods
    default_accs = merged_method_accs['Default']

    for i, method in enumerate(all_methods[1:], 1):  # Skip Default (first method)
        other_accs = merged_method_accs[method]
        t_stat, p_value = stats.ttest_ind(default_accs, other_accs)
        # Add asterisk if significantly better (p < 0.05)
        marker = '*' if p_value < 0.05 else ''
        plt.text(i, 98., f'{marker}', ha='center', va='bottom', fontsize=font_size)

    # Add legend
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
    # Increase right margin to accommodate legend
    plt.subplots_adjust(right=0.85)
    plt.legend(handles=handles, bbox_to_anchor=(0.5, 1.05), loc="lower center", ncol=3, fontsize=15)
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

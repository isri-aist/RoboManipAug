import numpy as np
import pandas as pd
import yaml

# ===== 並び順指定 =====
VARIANT_ORDER = [
    "Proposed",
    "MilesRadius001",
    "MilesRadius010",
    "NextSphereConvergence",
    "ExtendedConvergence",
    "InsideSphere",
]

TASK_ORDER = [
    "MujocoUR5eInsert",
    "MujocoUR5eDoor",
    "MujocoUR5eCabinetHinge",
    "MujocoUR5eToolbox",
]


def compute_success_rate(success_str_list):
    """Compute mean success rate (1 の割合)"""
    rates = []
    for s in success_str_list:
        arr = np.array([int(c) for c in s if c in "01"])
        if len(arr) > 0:
            rates.append(arr.mean())
    return np.mean(rates) if rates else np.nan


def main(input_path="AdditionalEval_20251112.yaml"):
    with open(input_path, "r") as f:
        merged = yaml.safe_load(f)

    table = []
    for variant in VARIANT_ORDER:
        if variant not in merged:
            continue
        for task in TASK_ORDER:
            if task not in merged[variant]:
                continue
            rate = compute_success_rate(merged[variant][task])
            table.append({"Variant": variant, "Task": task, "SuccessRate": rate})

    df = pd.DataFrame(table)
    df_pivot = df.pivot(index="Task", columns="Variant", values="SuccessRate")

    # 列・行の順序を固定
    df_pivot = df_pivot.reindex(index=TASK_ORDER, columns=VARIANT_ORDER)

    # 各variant列の平均（全タスク平均）を最下行に追加
    df_pivot.loc["Average"] = df_pivot.mean(axis=0)

    # Markdown形式で出力
    print(df_pivot.to_markdown(floatfmt=".2f"))


if __name__ == "__main__":
    main()

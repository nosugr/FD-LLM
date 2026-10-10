"""Training curves reusable across alignment and diagnosis."""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(history, path):
    if not history:
        raise ValueError("history is empty")
    keys = sorted(set(history[0]["train"]) | set(history[0]["validation"]))
    figure, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for axis, key in zip(axes[0], keys):
        # 训练和验证共用指标子图，只绘制实际存在该指标的 epoch。
        for split in ("train", "validation"):
            entries = [item for item in history if key in item[split]]
            if entries:
                axis.plot([item["epoch"] for item in entries], [item[split][key] for item in entries],
                          marker="o", markersize=3, label=split)
        axis.set(xlabel="Epoch", ylabel=key, title=key)
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)

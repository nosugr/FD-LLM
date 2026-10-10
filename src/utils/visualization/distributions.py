"""Dataset-independent class count plots."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_class_distribution(
    class_names: Sequence[str], train_counts: Sequence[int], test_counts: Sequence[int],
    *, title: str = "Class distribution",
) -> Figure:
    """Return a stacked train/test count chart from caller-provided counts."""
    train = np.asarray(train_counts)
    test = np.asarray(test_counts)
    if train.shape != (len(class_names),) or test.shape != train.shape or not len(train):
        raise ValueError("Class names and counts must have the same nonzero length")
    if not np.isfinite(train).all() or not np.isfinite(test).all() or (train < 0).any() or (test < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    positions = np.arange(len(train))
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.bar(positions, train, color="#17647a", label="Train")
    # 堆叠显示训练与测试数量，柱子总高表示该类全部样本数。
    ax.bar(positions, test, bottom=train, color="#e5a44f", label="Test")
    ax.set(xticks=positions, xticklabels=class_names, ylabel="Samples", title=title)
    ax.tick_params(axis="x", labelrotation=55)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_ylim(0, max(1, float((train + test).max())) * 1.12)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    return fig

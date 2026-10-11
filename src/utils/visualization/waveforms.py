"""Dataset-independent time-domain waveform plots."""

from collections.abc import Sequence
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_waveform_grid(
    signals: Sequence[np.ndarray],
    captions: Sequence[str],
    *,
    rows: int = 1,
    columns: int = 1,
    start: int = 0,
    points: int = 1600,
    title: Optional[str] = None,
) -> Figure:
    """Plot raw vectors in row-major order; return a figure for saving/customizing.

    X coordinates are original sample indices. Each panel uses its own amplitude
    limits so faults with different amplitudes remain readable. No normalization
    or resampling is performed. All requested points must exist in every vector.
    """
    for name, value, minimum in (("rows", rows, 1), ("columns", columns, 1),
                                 ("start", start, 0), ("points", points, 2)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if len(signals) != rows * columns or len(captions) != len(signals):
        raise ValueError("signals and captions must match the grid size")
    segments = []
    for signal in signals:
        vector = np.asarray(signal)
        if vector.ndim != 1 or not np.issubdtype(vector.dtype, np.number):
            raise ValueError("Each signal must be a numeric one-dimensional vector")
        if len(vector) < start + points:
            raise ValueError(f"Signal has {len(vector)} points; {start + points} required")
        segment = vector[start:start + points]
        if not np.isfinite(segment).all():
            raise ValueError("Signal segment contains nonfinite values")
        segments.append(segment)

    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(5 * columns, 3.5 * rows), layout="constrained")
    # 使用原始采样点编号与幅值，不在绘图时重新归一化或重采样。
    x = np.arange(start, start + points)
    for ax, segment, caption in zip(axes.flat, segments, captions):
        ax.plot(x, segment, color="#007fa3", linewidth=0.65)
        ax.set(xlim=(x[0], x[-1]), xlabel="Sample index",
               ylabel="Vibration amplitude", title=caption)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.15, linewidth=0.5)
    if title:
        fig.suptitle(title, fontsize=16)
    return fig

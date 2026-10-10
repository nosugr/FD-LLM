"""Check waveform coordinates, raw amplitudes and CWRU panel ordering."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.visualization import plot_waveform_grid
from src.data_processing import visualization


def test_waveform_preserves_raw_segment():
    signal = np.arange(20, dtype=float) * 3 - 12
    figure = plot_waveform_grid([signal], ["Example"], start=4, points=8)
    try:
        line = figure.axes[0].lines[0]
        np.testing.assert_array_equal(line.get_xdata(), np.arange(4, 12))
        np.testing.assert_array_equal(line.get_ydata(), signal[4:12])
    finally:
        plt.close(figure)


def test_short_signal_is_rejected():
    with pytest.raises(ValueError, match="required"):
        plot_waveform_grid([np.arange(3)], ["Short"], points=4)


def test_cwru_panel_order_and_sources(monkeypatch):
    monkeypatch.setattr(visualization, "load_signal",
                        lambda raw_dir, source: (np.full(10, source.label), None))
    figures = visualization.create_waveform_figures(Path("unused"), points=10)
    try:
        expected = {"normal": [0], "fan_end": [4, 5, 6, 1, 2, 3, 7, 8, 9],
                    "drive_end": [13, 14, 15, 10, 11, 12, 16, 17, 18]}
        for name, labels in expected.items():
            assert len(figures[name].axes) == len(labels)
            for ax, label in zip(figures[name].axes, labels):
                np.testing.assert_array_equal(ax.lines[0].get_ydata(), np.full(10, label))
    finally:
        for figure in figures.values():
            plt.close(figure)

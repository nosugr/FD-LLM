"""Organize CWRU data and call reusable plotting utilities.

Run: python -m data.preprocessing.CWRU.visualization
Add --distribution to also regenerate the existing class count chart.
This entry point reads data only; it does not rebuild the processed dataset.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.visualization import plot_class_distribution, plot_waveform_grid

from .config import DEFAULT_CONFIG, load_section
from .manifest import CLASSES
from .pipeline import PROJECT_ROOT, load_config, load_signal


FAULT_ROWS = (("ball", "Ball"), ("inner_race", "Inner race"), ("outer_race", "Outer race"))


def create_waveform_figures(raw_dir: Path, *, start: int = 0, points: int = 1600) -> dict:
    """Return normal, fan-end and drive-end figures using explicit source mappings."""
    groups = [("normal", "Normal bearing (DE channel)", 1, 1),
              ("fan_end", "Fan-end bearing faults (FE channel)", 3, 3),
              ("drive_end", "Drive-end bearing faults (DE channel)", 3, 3)]
    figures = {}
    try:
        for end, title, rows, columns in groups:
            sources = [source for source in CLASSES if source.bearing_position == end]
            if end != "normal":
                # 面板按故障类型分行、缺陷尺寸分列，便于对照比较。
                row_order = {fault: index for index, (fault, _) in enumerate(FAULT_ROWS)}
                sources.sort(key=lambda source: (row_order[source.fault_type], source.fault_size_inches))
            signals = [load_signal(raw_dir, source)[0] for source in sources]
            names = dict(FAULT_ROWS)
            captions = ["Normal" if end == "normal" else
                        f'{names[source.fault_type]} {source.fault_size_inches:.3f}" Fault'
                        for source in sources]
            figures[end] = plot_waveform_grid(signals, captions, rows=rows, columns=columns,
                                              start=start, points=points, title=title)
    except Exception:
        for figure in figures.values():
            plt.close(figure)
        raise
    return figures


def create_distribution_figure(processed_dir: Path):
    """Read actual saved labels, leaving generic plotting independent of CWRU."""
    metadata = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
    sources = sorted(metadata["sources"], key=lambda source: source["label"])
    names = []
    for source in sources:
        if source["label"] == 0:
            names.append("Normal")
        else:
            end = "FE" if source["bearing_position"] == "fan_end" else "DE"
            fault = {"inner_race": "IR", "ball": "Ball", "outer_race": "OR"}[source["fault_type"]]
            names.append(f'{end}-{fault}-{round(source["fault_size_inches"] * 1000):02d}')
    # 从实际保存的标签统计分布，避免图中样本数与处理后数据不一致。
    with np.load(processed_dir / "train.npz") as train, np.load(processed_dir / "test.npz") as test:
        train_counts = [(train["y"] == source["label"]).sum() for source in sources]
        test_counts = [(test["y"] == source["label"]).sum() for source in sources]
    return plot_class_distribution(names, train_counts, test_counts,
                                   title=f"CWRU class distribution · {len(sources)} classes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CWRU raw waveform comparisons")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--distribution", action="store_true")
    args = parser.parse_args()
    config = load_section(args.config, "visualization")
    preprocessing_path = (PROJECT_ROOT / config["preprocessing_config"]
                          if "preprocessing_config" in config else args.config)
    preprocessing = load_config(preprocessing_path)
    figures = create_waveform_figures(PROJECT_ROOT / preprocessing["raw_dir"],
                                      start=config["start_index"], points=config["points"])
    filenames = {"normal": "cwru_normal_waveform.png", "fan_end": "cwru_fan_end_waveforms.png",
                 "drive_end": "cwru_drive_end_waveforms.png"}
    try:
        if args.distribution:
            figures["distribution"] = create_distribution_figure(PROJECT_ROOT / preprocessing["output_dir"])
            filenames["distribution"] = "cwru_class_distribution.png"
        output_dir = PROJECT_ROOT / config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            path = output_dir / filenames[name]
            figure.savefig(path, dpi=config["dpi"], facecolor="white")
            print(path)
    finally:
        for figure in figures.values():
            plt.close(figure)


if __name__ == "__main__":
    main()

"""Build reusable CWRU train/test arrays without overlapping windows."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .config import DEFAULT_CONFIG, load_section
from .manifest import CLASSES, ClassSource


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = load_section(path, "preprocessing")
    for key in ("window_length", "stride", "samples_per_class", "random_seed"):
        if type(config[key]) is not int or config[key] < (0 if key == "random_seed" else 1):
            raise ValueError(f"{key} must be a positive integer (seed may be zero)")
    if config["stride"] < config["window_length"]:
        raise ValueError("stride must be at least window_length to avoid overlap")
    if not 0 < config["train_ratio"] < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    train_count = round(config["samples_per_class"] * config["train_ratio"])
    if not 0 < train_count < config["samples_per_class"]:
        raise ValueError("train_ratio must leave at least one sample in each split")
    return config


def load_signal(raw_dir: Path, source: ClassSource) -> tuple[np.ndarray, int | None]:
    path = raw_dir / source.source_file
    if not path.is_file():
        raise FileNotFoundError(f"Missing CWRU source file: {path}")
    # 使用清单指定的振动通道，避免同一 MATLAB 文件中的其他通道被误读。
    rpm_field = f"X{path.stem.zfill(3)}RPM"
    mat = loadmat(path, variable_names=(source.signal_field, rpm_field))
    if source.signal_field not in mat:
        raise ValueError(f"{path}: expected signal field {source.signal_field} not found")
    signal = np.asarray(mat[source.signal_field])
    if signal.ndim != 2 or 1 not in signal.shape:
        raise ValueError(f"{path}: {source.signal_field} is not a one-channel time series")
    signal = signal.reshape(-1)
    if not np.issubdtype(signal.dtype, np.number) or not np.isfinite(signal).all():
        raise ValueError(f"{path}: {source.signal_field} contains invalid values")
    rpm = int(mat[rpm_field].item()) if rpm_field in mat else None
    return signal, rpm


def select_window_indices(available: int, samples: int, train_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Use the earliest train block and latest test block; spare windows form a gap."""
    if available < samples:
        raise ValueError(f"Only {available} nonoverlapping windows available; {samples} required")
    test_count = samples - train_count
    # 每类取前段训练、末段测试，中间多余窗口留作时间间隔。
    return np.arange(train_count), np.arange(available - test_count, available)


def take_windows(signal: np.ndarray, indices: np.ndarray, length: int, stride: int) -> np.ndarray:
    positions = indices[:, None] * stride + np.arange(length)
    return signal[positions, None].astype(np.float64, copy=False)


def standardize(train_raw: np.ndarray, test_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    # 均值和标准差只由训练数据拟合，测试数据沿用，避免统计信息泄漏。
    mean = float(train_raw.mean(dtype=np.float64))
    std = float(train_raw.std(dtype=np.float64))
    if not np.isfinite(mean) or not np.isfinite(std) or std == 0:
        raise ValueError("Training data has invalid or zero standard deviation")
    train = ((train_raw - mean) / std).astype(np.float32)
    test = ((test_raw - mean) / std).astype(np.float32)
    return train, test, mean, std


def build_dataset(config_path: Path = DEFAULT_CONFIG) -> dict:
    config = load_config(config_path)
    raw_dir = PROJECT_ROOT / config["raw_dir"]
    output_dir = PROJECT_ROOT / config["output_dir"]
    length = config["window_length"]
    stride = config["stride"]
    samples = config["samples_per_class"]
    train_count = round(samples * config["train_ratio"])

    parts = {split: {key: [] for key in ("X_raw", "y", "window_index", "source_file")}
             for split in ("train", "test")}
    sources = []
    for source in CLASSES:
        signal, rpm = load_signal(raw_dir, source)
        available = 1 + (len(signal) - length) // stride if len(signal) >= length else 0
        try:
            train_indices, test_indices = select_window_indices(available, samples, train_count)
        except ValueError as exc:
            raise ValueError(f"{source.source_file} ({source.signal_field}, {len(signal)} points): {exc}") from exc

        for split, indices in (("train", train_indices), ("test", test_indices)):
            parts[split]["X_raw"].append(take_windows(signal, indices, length, stride))
            parts[split]["y"].append(np.full(len(indices), source.label, dtype=np.int64))
            parts[split]["window_index"].append(indices.astype(np.int32))
            parts[split]["source_file"].append(np.full(len(indices), source.source_file))

        sources.append({**asdict(source), "signal_length": len(signal), "available_windows": available,
                        "rpm": rpm, "train_windows": len(train_indices), "test_windows": len(test_indices),
                        "unused_gap_windows": available - samples})

    arrays = {}
    rng = np.random.default_rng(config["random_seed"])
    for split in ("train", "test"):
        arrays[split] = {key: np.concatenate(values) for key, values in parts[split].items()}
        # 划分已按时间确定；随机打乱只改变保存顺序，且所有字段同步排列。
        order = rng.permutation(len(arrays[split]["y"]))
        arrays[split] = {key: value[order] for key, value in arrays[split].items()}
        arrays[split]["start_index"] = arrays[split]["window_index"] * stride

    train, test, mean, std = standardize(arrays["train"]["X_raw"], arrays["test"]["X_raw"])
    arrays["train"]["X"] = train
    arrays["test"]["X"] = test
    metadata = {
        # 保留通道、源文件及归一化参数，便于文本生成和后续实验追溯。
        "config": config,
        "class_count": len(CLASSES),
        "train_count": len(train),
        "test_count": len(test),
        "normalization": {"method": "global_z_score", "fit_split": "train", "mean": mean, "std": std},
        "sampling_strategy": "first train_count windows and last test_count windows per class; all intermediate windows unused",
        "split_strategy": "chronological blocks per source, with an unused temporal gap; seeded shuffle only changes saved array order",
        "source_pages": [
            "https://engineering.case.edu/bearingdatacenter/normal-baseline-data",
            "https://engineering.case.edu/bearingdatacenter/12k-fan-end-bearing-fault-data",
            "https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data",
        ],
        "sources": sources,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        np.savez_compressed(output_dir / f"{split}.npz", **arrays[split])
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata

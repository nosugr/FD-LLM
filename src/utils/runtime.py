"""Configuration, device selection and reproducibility shared by both stages."""

import random
import os
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/fd_llm.yaml"


def load_config(path=DEFAULT_CONFIG):
    config = yaml.safe_load(project_path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def project_path(path):
    value = os.path.expandvars(os.path.expanduser(os.fspath(path)))
    if os.name != "nt" and (PureWindowsPath(value).drive or "\\" in value):
        raise ValueError(f"Windows path is not usable on Linux: {value}; use a server path with '/' separators")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def select_dtype(name, device):
    if name == "auto":
        # CPU 默认 float32；CUDA 优先使用硬件支持的 bfloat16。
        return (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                ) if device.type == "cuda" else torch.float32
    if name not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be auto, float32, float16 or bfloat16")
    dtype = getattr(torch, name)
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("Use float32 or bfloat16 on CPU")
    return dtype


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    # 同时记录各随机数源，续训时恢复采样、打乱和 dropout 的随机状态。
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])

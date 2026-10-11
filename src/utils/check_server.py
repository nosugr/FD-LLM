"""Run on AutoDL: python -m src.utils.check_server [--config ...]."""

import argparse
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys


def check_server(config_path):
    print(f"Python: {sys.version.split()[0]} ({sys.executable})", flush=True)
    if sys.platform != "linux" or sys.version_info[:2] != (3, 9):
        raise RuntimeError("Run this check on the Linux server with fdllm Python 3.9")
    if Path(sys.prefix).name != "fdllm":
        raise RuntimeError(f"Expected conda environment fdllm; sys.prefix={sys.prefix}")
    # A PyCharm SSH interpreter may not set CONDA_DEFAULT_ENV; sys.prefix is authoritative.
    print(f"Conda prefix: {sys.prefix}; working directory: {Path.cwd()}")
    for package, module in (("torch", "torch"), ("numpy", "numpy"), ("scipy", "scipy"),
                            ("PyYAML", "yaml"), ("transformers", "transformers"),
                            ("peft", "peft"), ("sentencepiece", "sentencepiece"),
                            ("scikit-learn", "sklearn"), ("matplotlib", "matplotlib")):
        importlib.import_module(module)
        print(f"{package}: {version(package)}", flush=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    import torch
    from .runtime import load_config, project_path, select_device, select_dtype
    from .huggingface import model_source
    from .data import CWRUDataset, load_descriptions

    config = load_config(config_path)
    device = select_device("cuda")
    dtype = select_dtype(config["model"].get("dtype", "auto"), device)
    properties = torch.cuda.get_device_properties(device)
    print(f"GPU: {properties.name}; memory: {properties.total_memory / 2**30:.1f} GiB")
    print(f"Torch CUDA runtime: {torch.version.cuda}; LLM dtype: {dtype}")
    if config["device"] not in ("cuda", "cuda:0", "auto"):
        raise ValueError("The training config must select the server CUDA device")
    # Exercise the CUDA kernel and float32 cuDNN LSTM backward without loading 7B weights.
    matrix = torch.randn(16, 16, device=device, dtype=dtype)
    if not torch.isfinite(matrix @ matrix).all():
        raise RuntimeError("CUDA matrix multiplication produced nonfinite values")
    lstm = torch.nn.LSTM(1, 8, batch_first=True).to(device)
    lstm(torch.randn(2, 16, 1, device=device))[0].square().mean().backward()
    torch.cuda.synchronize()
    print("CUDA matrix multiplication and LSTM backward: OK")

    for key in ("train_path", "test_path"):
        path = project_path(config["data"][key])
        dataset = CWRUDataset(path)
        expected = (config["data_encoder"]["sequence_length"], config["data_encoder"]["input_size"])
        if tuple(dataset[0][0].shape) != expected:
            raise ValueError(f"{path}: input shape does not match {expected}")
        if int(dataset.labels.max()) >= config["data"]["num_classes"]:
            raise ValueError(f"{path}: labels exceed configured num_classes")
        print(f"{key}: {path} ({len(dataset)} samples)")
    load_descriptions(config["data"]["descriptions_path"], config["data"]["num_classes"])
    source = Path(model_source(config["model"]))
    if not source.is_dir():
        raise ValueError("Server preflight requires model.name_or_path to be a local weight directory")
    model_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if model_config.get("hidden_size") != config["data_encoder"]["embedding_dim"]:
        raise ValueError("Model hidden_size differs from data_encoder.embedding_dim")
    indices = list(source.glob("*.index.json"))
    if indices:
        for index in indices:
            mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            for shard in set(mapping.values()):
                if not (source / shard).is_file() or (source / shard).stat().st_size == 0:
                    raise FileNotFoundError(f"Missing/empty model shard: {source / shard}")
    elif not any((source / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError(f"No model weights in {source}")
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
    print(f"Model files and tokenizer: {source} OK")
    # Import the modules whose annotations previously failed under Python 3.9.
    for module in ("src.data_processing.manifest", "src.data_processing.pipeline",
                   "src.utils.visualization.waveforms", "src.evaluation.evaluate",
                   "baselines.train"):
        importlib.import_module(module)
    print("Server preflight passed. Full 7B loading/training and peak VRAM are not tested.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fd_llm.yaml")
    args = parser.parse_args()
    try:
        check_server(args.config)
    except Exception as error:
        print(f"Server preflight FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

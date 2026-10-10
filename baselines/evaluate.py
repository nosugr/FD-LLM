"""Re-evaluate a saved baseline checkpoint, independently of its training run."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baselines.data import prepare_data
from baselines.factory import build_classifier, build_simclr
from baselines.simclr import LinearProbe
from src.evaluation.evaluate import evaluate_model, save_evaluation
from src.utils.runtime import select_device
from src.utils.training import read_checkpoint


def load_model(checkpoint_path, device):
    checkpoint = read_checkpoint(checkpoint_path)
    if checkpoint["stage"] != "training":
        raise ValueError("Use a supervised/downstream checkpoint, not a SimCLR pretraining checkpoint")
    config = checkpoint["config"]
    # 按 checkpoint 中的结构重建模型，避免当前配置变化导致权重不匹配。
    name = config["model"]["name"]
    architecture = config["model"]["architecture"]
    if name == "simclr":
        representation = build_simclr(architecture, config["data"])
        model = LinearProbe(representation.encoder, finetune=architecture["downstream"] == "finetune")
    else:
        model = build_classifier(name, architecture, config["data"])
    model.load_state_dict(checkpoint["model"]["network"])
    return model.to(device).eval(), checkpoint


def run(checkpoint_path, *, device="auto", output_dir=None):
    selected = select_device(device)
    model, checkpoint = load_model(checkpoint_path, selected)
    config = checkpoint["config"]
    prepared = prepare_data(config)
    # 独立评估前验证数据身份及测试范围，与训练时记录的协议保持一致。
    if prepared["protocol"] != checkpoint["signature"]["protocol"]:
        raise ValueError("Checkpoint dataset/test protocol differs from current artifacts")
    loader = DataLoader(prepared["test"], batch_size=config["evaluation"]["batch_size"], shuffle=False)
    metrics, arrays = evaluate_model(model, loader, selected, prepared["class_names"])
    target = Path(output_dir) if output_dir else Path(checkpoint_path).resolve().parents[1] / "test"
    save_evaluation(metrics, arrays, target, config, checkpoint["history"])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    run(args.checkpoint, device=args.device, output_dir=args.output)


if __name__ == "__main__":
    main()

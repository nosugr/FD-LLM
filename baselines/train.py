"""Single/batched baseline experiments, repeat statistics, or explicit CPU smoke."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from baselines.comparison import collect_results, write_comparison
from baselines.data import make_loaders, prepare_data
from baselines.factory import MODEL_NAMES, build_classifier, build_simclr
from baselines.simclr import LinearProbe
from baselines.tasks import BaselineTrainer, PretrainingTask, SupervisedTask
from src.evaluation.evaluate import evaluate_model, save_evaluation
from src.utils.runtime import load_config, project_path, seed_everything, select_device
from src.utils.training import atomic_save, read_checkpoint

DEFAULT_CONFIG = project_path("configs/baselines.yaml")


def smoke_config(config):
    # 在配置副本上缩小模型及样本量，只验证 CPU 流程，不作为正式实验。
    config = copy.deepcopy(config)
    config.update(device="cpu", execution="smoke", smoke={"train_per_class": 3, "test_per_class": 1})
    config["training"].update(epochs=1, batch_size=19)
    config["pretraining"].update(epochs=1, batch_size=19)
    config["evaluation"]["batch_size"] = 19
    config["models"]["cnn"].update(channels=[4, 8], kernels=[9, 3], strides=[2, 1])
    config["models"]["resnet18"]["base_channels"] = 4
    config["models"]["transformer"].update(d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    for name in ("attention_lstm", "attention_gru"):
        config["models"][name].update(hidden_size=8, num_layers=1, attention_size=4)
    config["models"]["simclr"].update(projection_hidden=16, projection_dim=8,
                                       encoder_kwargs={"base_channels": 4, "stem_kernel": 7})
    return config


def train_task(task, prepared, config, stage, directory, device, resume=None, contrastive=False):
    loaders = make_loaders(prepared, config, stage, task.shuffle_generator, device, contrastive=contrastive)
    optimizer = torch.optim.AdamW([p for p in task.parameters() if p.requires_grad],
                                 lr=config[stage]["learning_rate"], weight_decay=config[stage]["weight_decay"])
    trainer = BaselineTrainer(task, optimizer, device, directory, config, stage)
    history = trainer.fit(*loaders, resume=resume)
    best = Path(directory) / "best.pt"
    if not best.exists() and resume:
        # 续训可更换输出目录，此时沿用上一目录中已验证的最佳模型。
        previous_best = Path(resume).parent / "best.pt"
        if previous_best.exists():
            atomic_save(read_checkpoint(previous_best), best)
    if not best.exists():
        raise RuntimeError("No best checkpoint exists; epochs must be >= 1")
    # 训练结束后加载验证集选出的最佳状态，再进入测试评估。
    checkpoint = read_checkpoint(best)
    if checkpoint["signature"] != trainer.signature():
        raise ValueError("Best checkpoint belongs to a different experiment configuration")
    task.load_checkpoint_state(checkpoint["model"])
    return history


def run_once(name, config, prepared, seed, run_dir, device, *, resume=None, resume_pretraining=None):
    config = copy.deepcopy(config)
    config["seed"] = seed
    config["model"] = {"name": name, "architecture": config["models"][name]}
    config["protocol"], config["split"] = prepared["protocol"], prepared["split"]
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    if name != "simclr" and resume_pretraining:
        raise ValueError("Only SimCLR has representation pretraining")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    np.savez_compressed(run_dir / "split_indices.npz", **prepared["indices"])
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    architecture = config["models"][name]
    pretraining_history = None
    if name == "simclr":
        # SimCLR 先对比预训练，再用独立分类头做线性探测或整体微调。
        representation = build_simclr(architecture, config["data"]).to(device)
        if not resume:
            task = PretrainingTask(representation, generator, architecture,
                                    config["pretraining"].get("validation_seed", 12345))
            pretraining_history = train_task(task, prepared, config, "pretraining", run_dir / "pretraining",
                                             device, resume_pretraining, contrastive=True)
        mode = architecture["downstream"]
        if mode not in {"linear_probe", "finetune"}:
            raise ValueError("SimCLR downstream must be linear_probe or finetune")
        # 重置下游阶段随机状态，与监督基线采用相同的训练种子和初始打乱条件。
        seed_everything(seed)
        generator = torch.Generator().manual_seed(seed)
        network = LinearProbe(representation.encoder, finetune=mode == "finetune").to(device)
    else:
        network = build_classifier(name, architecture, config["data"]).to(device)
    task = SupervisedTask(network, generator)
    history = train_task(task, prepared, config, "training", run_dir / "training", device, resume)
    test_loader = DataLoader(prepared["test"], batch_size=config["evaluation"]["batch_size"], shuffle=False)
    metrics, arrays = evaluate_model(network, test_loader, device, prepared["class_names"])
    save_evaluation(metrics, arrays, run_dir / "test", config, history)
    result = {"model_name": name, "seed": seed, "metrics": metrics, "protocol": prepared["protocol"],
              "split": prepared["split"], "parameter_count": sum(p.numel() for p in network.parameters()),
              "pretraining_epochs": len(pretraining_history) if pretraining_history is not None else None,
              "training_epochs": len(history), "downstream": architecture.get("downstream"),
              "best_checkpoint": str((run_dir / "training/best.pt").resolve())}
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def run(config, models=MODEL_NAMES, repeats=None, *, smoke=False, output_dir=None,
        resume=None, resume_pretraining=None):
    config = smoke_config(config) if smoke else copy.deepcopy(config)
    seeds = config["repeat_seeds"]
    repeats = (1 if smoke else len(seeds)) if repeats is None else repeats
    if repeats < 1 or repeats > len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("repeats must select a positive number of distinct configured seeds")
    models = list(models)
    if not models or len(set(models)) != len(models) or any(name not in MODEL_NAMES for name in models):
        raise ValueError("Choose unique supported model names")
    if (resume or resume_pretraining) and (len(models) != 1 or repeats != 1):
        raise ValueError("Resume requires exactly one model and one repetition")
    if resume and resume_pretraining:
        raise ValueError("Resume either pretraining or downstream training, not both at once")
    if resume:
        seeds = [read_checkpoint(resume)["config"]["seed"]]
    elif resume_pretraining:
        seeds = [read_checkpoint(resume_pretraining)["config"]["seed"]]
    else:
        seeds = seeds[:repeats]
    root = project_path(output_dir or config["output_dir"])
    if smoke and output_dir is None:
        root = root / "smoke"
    device = select_device(config["device"])
    previous_threads = torch.get_num_threads()
    if smoke:
        torch.set_num_threads(1)
    try:
        # 数据和划分只准备一次，各模型及不同训练种子复用相同测试集。
        prepared = prepare_data(config)
        results = []
        for name in models:
            for seed in seeds:
                directory = root / name / f"seed_{seed}"
                print(f"Baseline={name} seed={seed} execution={prepared['protocol']['execution']}", flush=True)
                results.append(run_once(name, config, prepared, seed, directory, device,
                                         resume=resume, resume_pretraining=resume_pretraining))
        write_comparison(collect_results(root), root)
        return results
    finally:
        if smoke:
            torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--smoke", action="store_true", help="CPU subsets, tiny networks, one epoch (not formal experiments)")
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument("--resume-pretraining")
    args = parser.parse_args()
    names = MODEL_NAMES if args.model == "all" else [args.model]
    run(load_config(args.config), names, args.repeats, smoke=args.smoke, output_dir=args.output,
        resume=args.resume, resume_pretraining=args.resume_pretraining)


if __name__ == "__main__":
    main()

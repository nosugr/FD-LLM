"""Shared epoch trainer: accumulation, validation, logs and resumable state."""

import json
import logging
import math
import os
from pathlib import Path

import torch

from .runtime import rng_state, restore_rng
from .data import description_fingerprint


def read_checkpoint(path):
    # checkpoint 含 Python/NumPy 随机状态，需完整反序列化；只读取可信文件。
    return torch.load(path, map_location="cpu", weights_only=False)


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 先写临时文件再原子替换，避免写入中断损坏已有 checkpoint。
    torch.save(payload, temporary)
    os.replace(temporary, path)


class Trainer:
    """model.batch_metrics(signals, labels) returns loss and scalar metrics.

    Checkpoints resume at epoch boundaries and include optimizer, RNG, history,
    best score and model state. Model checkpoint_state/load_checkpoint_state can
    exclude frozen LLM weights. This is single-device training, not DDP.
    """

    def __init__(self, model, optimizer, device, output_dir, config, stage):
        self.model, self.optimizer, self.device = model, optimizer, device
        self.output_dir = Path(output_dir)
        self.config, self.stage = config, stage
        self.settings = config[stage]
        self.accumulation = self.settings.get("gradient_accumulation_steps", 1)
        if self.accumulation < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.history, self.start_epoch, self.best = [], 0, math.inf

    def signature(self):
        keys = ["seed", "data", "model", "data_encoder", "text_encoder", "contrastive"]
        if self.stage == "diagnosis":
            keys.extend(["prompt", "lora"])
        signature = {key: self.config[key] for key in keys if key in self.config}
        signature["descriptions_sha256"] = description_fingerprint(self.config["data"]["descriptions_path"])
        return signature

    def resume(self, path):
        checkpoint = read_checkpoint(path)
        if checkpoint["stage"] != self.stage or checkpoint["signature"] != self.signature():
            raise ValueError("Checkpoint stage/model/data configuration does not match")
        if hasattr(self.model, "load_checkpoint_state"):
            self.model.load_checkpoint_state(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        # 优化器状态跟随参数迁移，兼容 CPU 保存后在 CUDA 上续训。
        for parameter, state in self.optimizer.state.items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(parameter.device)
        self.history = checkpoint["history"]
        self.start_epoch, self.best = checkpoint["epoch"] + 1, checkpoint["best"]
        restore_rng(checkpoint["rng"])

    def run_epoch(self, loader, training):
        if len(loader) == 0:
            raise ValueError("Loader must contain at least one batch")
        self.model.train(training)
        totals, counts = {}, {}
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            for index, (signals, labels) in enumerate(loader):
                signals, labels = signals.to(self.device), labels.to(self.device)
                metrics = self.model.batch_metrics(signals, labels)
                loss = metrics["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss encountered")
                if training:
                    start = index // self.accumulation * self.accumulation
                    # 最后一个累积组可能不足设定批次数，按实际组大小缩放损失。
                    group_size = min(self.accumulation, len(loader) - start)
                    (loss / group_size).backward()
                    if (index + 1) % self.accumulation == 0 or index + 1 == len(loader):
                        parameters = [p for p in self.model.parameters() if p.requires_grad]
                        torch.nn.utils.clip_grad_norm_(parameters, self.settings.get("max_grad_norm", 1.0),
                                                       error_if_nonfinite=True)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                for key, value in metrics.items():
                    if key == "loss_weight":
                        continue
                    # 语言模型指标按有效答案 token 加权，分类指标按样本数加权。
                    count = metrics.get("loss_weight", len(labels)) if key in {"loss", "token_accuracy"} else len(labels)
                    totals[key] = totals.get(key, 0.0) + float(value.detach() if isinstance(value, torch.Tensor) else value) * count
                    counts[key] = counts.get(key, 0) + count
        return {key: value / counts[key] for key, value in totals.items()}

    def fit(self, train_loader, validation_loader, *, resume=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"fd_llm.{self.stage}.{id(self)}")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.output_dir / "train.log", encoding="utf-8")
        logger.addHandler(handler)
        try:
            if resume:
                self.resume(resume)
            for epoch in range(self.start_epoch, self.settings["epochs"]):
                train = self.run_epoch(train_loader, True)
                validation = self.run_epoch(validation_loader, False)
                record = {"epoch": epoch + 1, "train": train, "validation": validation}
                self.history.append(record)
                # 只依据验证损失选择最佳模型，last.pt 另存最近状态用于续训。
                improved = validation["loss"] < self.best
                self.best = min(self.best, validation["loss"])
                state = (self.model.checkpoint_state() if hasattr(self.model, "checkpoint_state")
                         else self.model.state_dict())
                payload = {"format_version": 1, "stage": self.stage, "epoch": epoch,
                           "best": self.best, "history": self.history, "signature": self.signature(),
                           "config": self.config, "model": state, "optimizer": self.optimizer.state_dict(),
                           "rng": rng_state()}
                atomic_save(payload, self.output_dir / "last.pt")
                if improved:
                    atomic_save(payload, self.output_dir / "best.pt")
                logger.info(json.dumps(record))
                print(json.dumps(record), flush=True)
                (self.output_dir / "history.json").write_text(json.dumps(self.history, indent=2) + "\n", encoding="utf-8")
                # 根据已恢复的历史重写指标文件，避免续训产生重复 epoch 记录。
                (self.output_dir / "metrics.jsonl").write_text(
                    "".join(json.dumps(item) + "\n" for item in self.history), encoding="utf-8")
            if self.history:
                from .visualization.training import plot_training_curves
                plot_training_curves(self.history, self.output_dir / "curves.png")
        finally:
            logger.removeHandler(handler)
            handler.close()
        return self.history

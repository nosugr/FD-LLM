"""Adapt baseline tasks to the existing shared Trainer without editing src/."""

import torch
from torch import nn

from baselines.simclr import NTXentLoss, SignalAugmenter
from src.utils.training import Trainer


class SupervisedTask(nn.Module):
    def __init__(self, network, shuffle_generator):
        super().__init__()
        self.network, self.shuffle_generator = network, shuffle_generator

    def batch_metrics(self, signals, labels):
        return self.network.batch_metrics(signals, labels)

    def checkpoint_state(self):
        # 显式保存 DataLoader 的独立随机状态，续训可恢复样本打乱顺序。
        return {"network": self.network.state_dict(), "shuffle_rng": self.shuffle_generator.get_state()}

    def load_checkpoint_state(self, state):
        self.network.load_state_dict(state["network"])
        self.shuffle_generator.set_state(state["shuffle_rng"].cpu())


class PretrainingTask(SupervisedTask):
    def __init__(self, network, shuffle_generator, architecture, validation_seed=12345):
        super().__init__(network, shuffle_generator)
        self.augmenter = SignalAugmenter(**architecture["augmentations"])
        self.criterion = NTXentLoss(architecture["temperature"])
        self.validation_seed, self.validation_generator = validation_seed, None

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            # 每次验证重置增强随机种子，使不同 epoch 的损失具有可比性。
            device = next(self.network.parameters()).device
            self.validation_generator = torch.Generator(device=device).manual_seed(self.validation_seed)
        return self

    def batch_metrics(self, signals, labels):
        # 表征预训练不使用类别标签，只对同一信号的两种增强视图计算对比损失。
        generator = None if self.training else self.validation_generator
        first, second = self.augmenter(signals, generator), self.augmenter(signals, generator)
        return {"loss": self.criterion(self.network(first), self.network(second)),
                "loss_weight": 2 * len(signals)}


class BaselineTrainer(Trainer):
    def signature(self):
        signature = super().signature()
        signature["protocol"] = self.config["protocol"]
        signature["split"] = self.config["split"]
        signature["settings"] = {key: value for key, value in self.config[self.stage].items()
                                 if key != "epochs"}
        return signature

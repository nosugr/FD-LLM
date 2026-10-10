"""SimCLR [Chen et al., FD-LLM ref. 61] and separate downstream classification."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from baselines.common import SignalClassifier


class NTXentLoss(nn.Module):
    """2N anchors, paired-view positives, all other instances as negatives."""

    def __init__(self, temperature=0.5):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = temperature

    def forward(self, first, second):
        if first.ndim != 2 or first.shape != second.shape or len(first) < 2:
            raise ValueError("NT-Xent needs matching (N,D) views with N >= 2")
        batch = len(first)
        features = F.normalize(torch.cat((first, second)).float(), dim=-1)
        logits = features @ features.T / self.temperature
        # 排除自身相似度；另一组视图中相隔 N 的向量是同一样本的正样本。
        logits = logits.masked_fill(torch.eye(2 * batch, dtype=torch.bool, device=features.device), -torch.inf)
        targets = (torch.arange(2 * batch, device=features.device) + batch) % (2 * batch)
        return F.cross_entropy(logits, targets)


class SimCLR(nn.Module):
    """Feature backbone and nonlinear projection head, with no supervised head."""

    def __init__(self, encoder, projection_hidden=256, projection_dim=128):
        super().__init__()
        if min(projection_hidden, projection_dim) < 1:
            raise ValueError("Projection dimensions must be positive")
        # 预训练去掉监督分类头，非线性投影仅用于对比损失。
        encoder.classifier = nn.Identity()
        self.encoder = encoder
        self.projector = nn.Sequential(nn.Linear(encoder.feature_dim, projection_hidden), nn.ReLU(),
                                       nn.Linear(projection_hidden, projection_dim))

    def forward(self, signals):
        return self.projector(self.encoder.forward_features(signals))


class LinearProbe(SignalClassifier):
    """Discard projection head; freeze the pretrained encoder by default."""

    def __init__(self, encoder, *, finetune=False):
        super().__init__(input_size=encoder.input_size, sequence_length=encoder.sequence_length,
                         num_classes=encoder.num_classes)
        encoder.classifier = nn.Identity()
        self.encoder = encoder.requires_grad_(finetune)
        self.finetune, self.feature_dim = finetune, encoder.feature_dim
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        if not finetune:
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if not self.finetune:
            # 冻结线性探测的编码器时，同时固定 BatchNorm 的运行统计量。
            self.encoder.eval()
        return self

    def forward_features(self, signals):
        self.validate(signals)
        if self.finetune:
            return self.encoder.forward_features(signals)
        with torch.no_grad():
            return self.encoder.forward_features(signals)

"""Common classification contract and input validation for 1D baselines."""

import torch
from torch import nn
from torch.nn import functional as F


class SignalClassifier(nn.Module):
    def __init__(self, *, input_size=1, sequence_length=1024, num_classes=19):
        super().__init__()
        if min(input_size, sequence_length, num_classes) < 1:
            raise ValueError("Input dimensions and class count must be positive")
        self.input_size, self.sequence_length, self.num_classes = input_size, sequence_length, num_classes

    def validate(self, signals):
        if signals.ndim != 3 or signals.shape[1:] != (self.sequence_length, self.input_size) or len(signals) == 0:
            raise ValueError(f"Expected nonempty (B,{self.sequence_length},{self.input_size}) signals")
        if not signals.is_floating_point():
            raise TypeError("Signals must be floating-point tensors")

    def forward_features(self, signals):
        raise NotImplementedError

    def forward(self, signals):
        # 各基线只实现特征提取和分类头，训练及评估统一复用此接口。
        return self.classifier(self.forward_features(signals))

    def batch_metrics(self, signals, labels):
        logits = self(signals)
        return {"loss": F.cross_entropy(logits, labels),
                "accuracy": (logits.argmax(dim=-1) == labels).float().mean()}

    @torch.no_grad()
    def predict(self, signals, candidate_chunk_size=None):
        features = self.forward_features(signals)
        logits = self.classifier(features)
        return {"class_ids": logits.argmax(dim=-1), "scores": logits, "features": features}

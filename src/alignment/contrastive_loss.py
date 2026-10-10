"""Cosine similarity and symmetric multi-positive InfoNCE, equations (3)-(6)."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def cosine_logits(data, text):
    if data.ndim != 2 or text.ndim != 2 or data.shape[1] != text.shape[1]:
        raise ValueError("Features must be matrices with the same embedding dimension")
    return F.normalize(data.float(), dim=-1) @ F.normalize(text.float(), dim=-1).T


class BidirectionalContrastiveLoss(nn.Module):
    """Single-positive case equals the paper; same-class positives share mass.

    Extension: logsumexp over ALL same-class positive pairs in the numerator.
    Text rows without positives in a minibatch are omitted from text-to-data.
    Every data row must have at least one matching text. No duplicate-class
    observation is used as a negative. Fixed tau is the default implementation
    choice; optionally learn its log with a bounded positive temperature.
    """

    def __init__(self, temperature=0.07, epsilon=1e-6, learn_temperature=False):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("temperature and epsilon must be positive and finite")
        value = torch.tensor(math.log(temperature))
        if learn_temperature:
            self.log_temperature = nn.Parameter(value)
        else:
            self.register_buffer("log_temperature", value)
        self.epsilon = epsilon
        self.learn_temperature = learn_temperature

    @property
    def temperature(self):
        value = self.log_temperature.exp()
        return value.clamp(1e-4, 1.0) if self.learn_temperature else value

    @staticmethod
    def _direction(logits, positive):
        # 忽略当前批次中没有正样本的文本行；用 logsumexp 稳定汇总同类正样本。
        valid = positive.any(dim=1)
        logits, positive = logits[valid], positive[valid]
        return (torch.logsumexp(logits, dim=1)
                - torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)).mean()

    def forward(self, data, text, data_labels=None, text_labels=None):
        logits = cosine_logits(data, text) / (self.temperature + self.epsilon)
        if len(data) == 0 or len(text) == 0:
            raise ValueError("Empty feature banks are not supported")
        if data_labels is None and text_labels is None:
            if len(data) != len(text):
                raise ValueError("Unlabeled pairs require equal batch sizes")
            positive = torch.eye(len(data), device=data.device, dtype=torch.bool)
        else:
            if data_labels is None or text_labels is None:
                raise ValueError("Both data and text labels are required")
            if data_labels.shape != (len(data),) or text_labels.shape != (len(text),):
                raise ValueError("Labels must match feature bank lengths")
            # 同类数据和多条描述均为正样本，避免共享文本被误当成负样本。
            positive = data_labels[:, None] == text_labels[None, :]
        if not positive.any(dim=1).all():
            raise ValueError("Each data sample needs a matching class text")
        # 分别计算数据到文本、文本到数据的损失，再取平均。
        return (self._direction(logits, positive) + self._direction(logits.T, positive.T)) / 2

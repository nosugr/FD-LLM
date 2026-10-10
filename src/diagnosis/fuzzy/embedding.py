"""Fuzzy semantic embedding, exactly cosine -> softmax -> weighted sum (7)-(9)."""

import torch
from torch import nn

from src.alignment.contrastive_loss import cosine_logits


class FuzzySemanticEmbedding(nn.Module):
    def __init__(self, class_text_features):
        super().__init__()
        if class_text_features.ndim != 2 or min(class_text_features.shape) == 0:
            raise ValueError("class_text_features must be a nonempty (classes, dim) matrix")
        if not torch.isfinite(class_text_features).all():
            raise ValueError("Text features must be finite")
        self.register_buffer("text_features", class_text_features.detach().float())

    def forward(self, data_features):
        # 公式 (8) 直接对余弦相似度做 softmax，不使用第一阶段的温度参数。
        membership = cosine_logits(data_features, self.text_features).softmax(dim=-1)
        # 公式 (9) 用隶属度加权原始文本向量，保留向量的幅值信息。
        return membership @ self.text_features, membership

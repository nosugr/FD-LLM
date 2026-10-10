"""Encoder-only sequence classifier using Vaswani et al. (FD-LLM ref. 60).

Nonoverlapping signal patching and mean pooling adapt the original translation
model to bearing classification; these choices were not specified by FD-LLM.
"""

import math

import torch
from torch import nn

from baselines.common import SignalClassifier


class Transformer(SignalClassifier):
    def __init__(self, *, patch_size=16, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        if patch_size < 1 or self.sequence_length % patch_size:
            raise ValueError("patch_size must divide sequence_length")
        if min(d_model, nhead, num_layers, dim_feedforward) < 1 or d_model % nhead:
            raise ValueError("Positive dimensions are required and nhead must divide d_model")
        self.patch_size, self.feature_dim = patch_size, d_model
        tokens = self.sequence_length // patch_size
        # 正弦位置编码补充时间顺序，不额外引入可训练的位置参数。
        position = torch.arange(tokens, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(tokens, d_model)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies[:d_model // 2])
        self.register_buffer("position_encoding", encoding.unsqueeze(0))
        self.patch_projection = nn.Linear(patch_size * self.input_size, d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                           activation="relu", batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(d_model, self.num_classes)
        # PyTorch 通过克隆构造编码器层，各层矩阵需独立初始化。
        for parameter in self.encoder.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward_features(self, signals):
        self.validate(signals)
        # 将不重叠片段映射为 token，减少自注意力的序列长度和计算开销。
        patches = signals.reshape(len(signals), self.sequence_length // self.patch_size, -1)
        tokens = self.patch_projection(patches) * math.sqrt(self.feature_dim)
        # 对全部时间 token 取均值，得到用于故障分类的窗口特征。
        return self.encoder(self.dropout(tokens + self.position_encoding)).mean(dim=1)

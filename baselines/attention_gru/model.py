"""Encoder-decoder attention GRU classification adaptation, FD-LLM ref. [63].

The original ABGRU addresses RUL regression. One decoder query and a 19-class
head are engineering adaptations; no RUL targets are used in this classifier.
"""

import torch
from torch import nn

from baselines.common import SignalClassifier


class AttentionGRU(SignalClassifier):
    def __init__(self, *, hidden_size=128, num_layers=2, attention_size=64, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        if min(hidden_size, num_layers, attention_size) < 1:
            raise ValueError("Recurrent and attention dimensions must be positive")
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(self.input_size, hidden_size, num_layers, batch_first=True,
                               dropout=recurrent_dropout)
        self.decoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True,
                               dropout=recurrent_dropout)
        self.key = nn.Linear(hidden_size, attention_size, bias=False)
        self.query = nn.Linear(hidden_size, attention_size, bias=False)
        self.energy = nn.Linear(attention_size, 1, bias=False)
        self.feature_dim = hidden_size * 2
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, self.num_classes))

    def context_and_attention(self, signals):
        self.validate(signals)
        hidden, state = self.encoder(signals)
        # 以编码器末步输出和最终状态驱动一次解码，生成注意力查询向量。
        decoded, _ = self.decoder(hidden[:, -1:], state)
        query = decoded[:, 0]
        # 查询与每个时间步的键做加性注意力，汇总为上下文向量。
        weights = self.energy(torch.tanh(self.key(hidden) + self.query(query).unsqueeze(1))).squeeze(-1).softmax(dim=1)
        context = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        return context, query, weights

    def forward_features(self, signals):
        context, query, _ = self.context_and_attention(signals)
        # 拼接上下文与解码查询，供下游分类头使用。
        return torch.cat((context, query), dim=-1)

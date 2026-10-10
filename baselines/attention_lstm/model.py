"""Temporal attention + LSTM classification adaptation of FD-LLM ref. [62].

The cited task is connector RUL prediction. This uses learned additive pooling
of hidden states and a classification head; its precise layers are project choices.
"""

from torch import nn

from baselines.common import SignalClassifier


class AttentionLSTM(SignalClassifier):
    def __init__(self, *, hidden_size=128, num_layers=2, attention_size=64,
                 dropout=0.1, bidirectional=False, **kwargs):
        super().__init__(**kwargs)
        if min(hidden_size, num_layers, attention_size) < 1:
            raise ValueError("Recurrent and attention dimensions must be positive")
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.feature_dim = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.Sequential(nn.Linear(self.feature_dim, attention_size), nn.Tanh(),
                                       nn.Linear(attention_size, 1, bias=False))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, self.num_classes))

    def hidden_and_attention(self, signals):
        self.validate(signals)
        hidden, _ = self.lstm(signals)
        # 给每个时间步打分，再沿时间轴归一化为注意力权重。
        weights = self.attention(hidden).squeeze(-1).softmax(dim=1)
        return hidden, weights

    def forward_features(self, signals):
        hidden, weights = self.hidden_and_attention(signals)
        # 对所有隐藏状态加权汇总，让模型学习哪些时刻更有诊断价值。
        return (hidden * weights.unsqueeze(-1)).sum(dim=1)

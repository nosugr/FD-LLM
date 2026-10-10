"""1D CNN adaptation inspired by FD-LLM ref. [58], Ruan et al. (2023).

The wide first kernel captures local vibration decay; its size is an engineering
default, not a fitted physics-guided kernel or an exact PGCNN reproduction.
"""

from torch import nn

from baselines.common import SignalClassifier


class CNN(SignalClassifier):
    def __init__(self, *, channels=(32, 64, 128), kernels=(65, 5, 5),
                 strides=(4, 1, 1), dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        if not channels or len(channels) != len(kernels) or len(channels) != len(strides):
            raise ValueError("channels, kernels and strides must have matching nonempty lengths")
        if min(*channels, *kernels, *strides) < 1 or any(kernel % 2 == 0 for kernel in kernels):
            raise ValueError("Channels/strides must be positive and kernels must be positive odd numbers")
        layers, previous = [], self.input_size
        for width, kernel, stride in zip(channels, kernels, strides):
            layers.extend([nn.Conv1d(previous, width, kernel, stride=stride, padding=kernel // 2, bias=False),
                           nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(2)])
            previous = width
        # 逐级提取局部波形特征，全局平均池化后输出固定长度向量。
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten(1))
        self.feature_dim = channels[-1]
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, self.num_classes))

    def forward_features(self, signals):
        self.validate(signals)
        # 公共输入为 (B,L,C)，Conv1d 需要 (B,C,L)。
        return self.encoder(signals.transpose(1, 2))

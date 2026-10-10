"""ResNet-18 [He et al., FD-LLM ref. 59] with spatial convolutions changed to 1D."""

from torch import nn

from baselines.common import SignalClassifier


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        # 通道数或时间长度变化时，用 1×1 卷积对齐残差分支的形状。
        self.shortcut = (nn.Identity() if stride == 1 and in_channels == out_channels else
                         nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                                       nn.BatchNorm1d(out_channels)))

    def forward(self, signals):
        residual = self.shortcut(signals)
        features = self.relu(self.bn1(self.conv1(signals)))
        return self.relu(self.bn2(self.conv2(features)) + residual)


class ResNet18(SignalClassifier):
    def __init__(self, *, base_channels=64, stem_kernel=7, **kwargs):
        super().__init__(**kwargs)
        if base_channels < 1 or stem_kernel < 1 or stem_kernel % 2 == 0:
            raise ValueError("base_channels must be positive; stem_kernel must be positive and odd")
        self.stem = nn.Sequential(nn.Conv1d(self.input_size, base_channels, stem_kernel, stride=2,
                                          padding=stem_kernel // 2, bias=False),
                                  nn.BatchNorm1d(base_channels), nn.ReLU(),
                                  nn.MaxPool1d(3, stride=2, padding=1))
        stages, previous = [], base_channels
        # 沿用 ResNet18 的四个阶段，每阶段两个残差块，卷积改为一维并从头训练。
        for index, width in enumerate([base_channels * factor for factor in (1, 2, 4, 8)]):
            stages.append(nn.Sequential(BasicBlock1D(previous, width, 1 if index == 0 else 2),
                                        BasicBlock1D(width, width)))
            previous = width
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = previous
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, signals):
        self.validate(signals)
        return self.pool(self.stages(self.stem(signals.transpose(1, 2)))).squeeze(-1)

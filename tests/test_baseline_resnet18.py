"""Verify residual topology, downsampling and end-to-end classifier gradients."""

import torch
from torch import nn

from baselines.resnet18 import BasicBlock1D, ResNet18


def test_resnet18_default_shape_and_backward():
    model = ResNet18()
    assert sum(isinstance(module, BasicBlock1D) for module in model.modules()) == 8
    assert [len(stage) for stage in model.stages] == [2, 2, 2, 2]
    logits = model(torch.randn(2, 1024, 1))
    assert logits.shape == (2, 19)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 18])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.eval().predict(torch.randn(1, 1024, 1))["features"].shape == (1, 512)


def test_basic_block_shortcut_and_downsampling():
    block = BasicBlock1D(4, 4).eval()
    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv2.weight.zero_()
    signals = torch.randn(2, 4, 17)
    torch.testing.assert_close(block(signals), signals.relu())
    downsample = BasicBlock1D(4, 8, stride=2)
    assert isinstance(downsample.shortcut, nn.Sequential)
    assert downsample(signals).shape == (2, 8, 9)
    assert ResNet18(base_channels=4)(torch.randn(2, 1024, 1)).shape == (2, 19)

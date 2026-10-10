"""CNN feature extraction, class logits and trainable convolution checks."""

import pytest
import torch

from baselines.cnn import CNN


def test_cnn_forward_backward_and_features():
    model = CNN()
    signals = torch.randn(2, 1024, 1, requires_grad=True)
    logits = model(signals)
    assert logits.shape == (2, 19)
    loss = model.batch_metrics(signals, torch.tensor([0, 18]))["loss"]
    loss.backward()
    assert torch.isfinite(signals.grad).all() and signals.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    result = model.eval().predict(signals[:1])
    assert result["features"].shape == (1, 128)
    assert result["class_ids"].shape == (1,)


def test_cnn_configuration_and_input_guards():
    model = CNN(channels=(4, 8), kernels=(9, 3), strides=(2, 1), num_classes=3)
    assert model(torch.randn(2, 1024, 1)).shape == (2, 3)
    with pytest.raises(ValueError, match="Expected"):
        model(torch.randn(2, 1, 1024))
    with pytest.raises(ValueError):
        CNN(channels=(4,), kernels=(4,), strides=(1,))

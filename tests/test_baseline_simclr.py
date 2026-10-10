"""Correct NT-Xent pair indexing, independent views, and frozen linear probing."""

import math

import pytest
import torch

from baselines.cnn import CNN
from baselines.simclr import LinearProbe, NTXentLoss, SignalAugmenter, SimCLR


def tiny_encoder():
    return CNN(channels=(4, 8), kernels=(9, 3), strides=(2, 1))


def test_nt_xent_hand_computed_and_gradients():
    first = torch.eye(2, requires_grad=True)
    second = torch.eye(2, requires_grad=True)
    loss = NTXentLoss(temperature=1)(first, second)
    # 每个锚点有一个相似度为 1 的正样本，以及两个相似度为 0 的负样本。
    torch.testing.assert_close(loss, torch.tensor(math.log(math.e + 2) - 1))
    loss.backward()
    assert first.grad.abs().sum() > 0 and torch.isfinite(second.grad).all()
    with pytest.raises(ValueError, match="N >= 2"):
        NTXentLoss()(torch.ones(1, 4), torch.ones(1, 4))


def test_simclr_augmented_views_pretraining_backward():
    signals = torch.randn(3, 1024, 1)
    original = signals.clone()
    augmenter = SignalAugmenter()
    first, second = augmenter(signals), augmenter(signals)
    assert first.shape == second.shape == signals.shape
    assert not torch.equal(first, second)
    torch.testing.assert_close(signals, original, rtol=0, atol=0)
    model = SimCLR(tiny_encoder(), projection_hidden=16, projection_dim=8)
    projection = model(first)
    assert projection.shape == (3, 8)
    NTXentLoss()(projection, model(second)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_frozen_linear_probe_and_optional_finetuning():
    pretrained = SimCLR(tiny_encoder(), 16, 8)
    probe = LinearProbe(pretrained.encoder)
    before = {key: value.clone() for key, value in probe.encoder.state_dict().items()}
    probe.train()
    assert not probe.encoder.training
    probe.batch_metrics(torch.randn(2, 1024, 1), torch.tensor([0, 18]))["loss"].backward()
    assert all(p.grad is None for p in probe.encoder.parameters())
    assert probe.classifier.weight.grad.abs().sum() > 0
    for key, value in before.items():
        torch.testing.assert_close(probe.encoder.state_dict()[key], value, rtol=0, atol=0)
    finetuned = LinearProbe(tiny_encoder(), finetune=True)
    finetuned.batch_metrics(torch.randn(2, 1024, 1), torch.tensor([0, 18]))["loss"].backward()
    assert any(p.grad is not None for p in finetuned.encoder.parameters())


def test_disabled_augmentation_is_identity_and_seeded_views_repeat():
    signals = torch.randn(2, 1024, 1)
    torch.testing.assert_close(SignalAugmenter(0, 0, 1, 0)(signals), signals, rtol=0, atol=0)
    augmenter = SignalAugmenter()
    first = augmenter(signals, torch.Generator().manual_seed(42))
    second = augmenter(signals, torch.Generator().manual_seed(42))
    torch.testing.assert_close(first, second, rtol=0, atol=0)

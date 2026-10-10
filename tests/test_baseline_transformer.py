"""Position-aware patch tokens, batch isolation and attention gradients."""

import pytest
import torch

from baselines.transformer import Transformer


def test_transformer_forward_backward_and_batch_isolation():
    model = Transformer()
    signals = torch.randn(2, 1024, 1, requires_grad=True)
    logits = model(signals)
    assert logits.shape == (2, 19)
    model.batch_metrics(signals, torch.tensor([1, 18]))["loss"].backward()
    assert signals.grad.abs().sum() > 0 and torch.isfinite(signals.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    torch.testing.assert_close(model(signals)[:1], model(signals[:1]), atol=1e-5, rtol=1e-5)
    assert model.position_encoding.shape == (1, 64, 128)
    assert not torch.equal(model.position_encoding[:, 0], model.position_encoding[:, 1])


def test_transformer_configuration_and_guards():
    assert Transformer(d_model=12, nhead=3, num_layers=1, patch_size=32,
                       dim_feedforward=24)(torch.randn(2, 1024, 1)).shape == (2, 19)
    with pytest.raises(ValueError, match="divide"):
        Transformer(patch_size=15)
    with pytest.raises(ValueError, match="nhead"):
        Transformer(d_model=11, nhead=4)

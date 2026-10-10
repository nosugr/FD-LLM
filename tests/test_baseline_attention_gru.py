"""Encoder-decoder gradients and normalized temporal attention."""

import torch

from baselines.attention_gru import AttentionGRU


def test_attention_gru_forward_backward_and_attention():
    model = AttentionGRU()
    signals = torch.randn(2, 1024, 1, requires_grad=True)
    assert model(signals).shape == (2, 19)
    model.batch_metrics(signals, torch.tensor([3, 18]))["loss"].backward()
    assert signals.grad[:, :32].abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.encoder.weight_ih_l0.grad.abs().sum() > 0
    assert model.decoder.weight_ih_l0.grad.abs().sum() > 0
    context, query, weights = model.eval().context_and_attention(signals)
    assert context.shape == query.shape == (2, 128)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    assert weights.shape == (2, 1024)


def test_attention_gru_single_layer_configuration():
    model = AttentionGRU(hidden_size=8, num_layers=1, attention_size=4)
    assert model.encoder.dropout == model.decoder.dropout == 0
    assert model.eval().predict(torch.randn(1, 1024, 1))["features"].shape == (1, 16)

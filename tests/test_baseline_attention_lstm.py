"""Temporal probability weights, early-time gradients and configurable directions."""

import torch

from baselines.attention_lstm import AttentionLSTM


def test_attention_lstm_forward_backward_and_pooling():
    model = AttentionLSTM()
    signals = torch.randn(2, 1024, 1, requires_grad=True)
    assert model(signals).shape == (2, 19)
    model.batch_metrics(signals, torch.tensor([2, 18]))["loss"].backward()
    assert signals.grad[:, :32].abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    hidden, weights = model.eval().hidden_and_attention(signals)
    assert weights.shape == (2, 1024) and (weights > 0).all()
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    torch.testing.assert_close(model.forward_features(signals), (hidden * weights.unsqueeze(-1)).sum(dim=1))


def test_attention_lstm_single_layer_and_bidirectional():
    model = AttentionLSTM(hidden_size=8, num_layers=1, attention_size=4, bidirectional=True)
    assert model.lstm.dropout == 0
    assert model.forward_features(torch.randn(2, 1024, 1)).shape == (2, 16)
    assert model(torch.randn(2, 1024, 1)).shape == (2, 19)

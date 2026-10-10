"""Check the data encoder contract and differentiable feature extraction."""

import json
from pathlib import Path

import pytest
import torch

from src.alignment.encoders import LSTMDataEncoder


@pytest.fixture(scope="module", autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_default_forward_and_backward(batch_size):
    encoder = LSTMDataEncoder()
    signals = torch.randn(batch_size, 1024, 1, requires_grad=True)
    features = encoder(signals)
    assert features.shape == (batch_size, 4096)
    assert torch.isfinite(features).all()

    features.square().mean().backward()
    for tensor in [signals, *encoder.parameters()]:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


@pytest.mark.parametrize("bidirectional", [False, True])
def test_configurable_structure_and_final_states(bidirectional):
    encoder = LSTMDataEncoder(input_size=2, sequence_length=16, hidden_size=8,
                              num_layers=3, embedding_dim=32, dropout=0.2,
                              bidirectional=bidirectional).eval()
    signals = torch.randn(2, 16, 2)
    with torch.no_grad():
        _, (hidden, _) = encoder.lstm(signals)
        pooled = torch.cat((hidden[-2], hidden[-1]), dim=-1) if bidirectional else hidden[-1]
        expected = encoder.projection(pooled)
        actual = encoder(signals)
    assert actual.shape == (2, 32)
    torch.testing.assert_close(actual, expected)
    # 独立调用和不同批次大小之间不能残留上一次的循环状态。
    torch.testing.assert_close(encoder(signals[:1]), actual[:1])
    encoder(signals).square().mean().backward()
    for parameter in encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_single_layer_disables_inter_layer_dropout():
    encoder = LSTMDataEncoder(sequence_length=8, hidden_size=4, num_layers=1,
                              embedding_dim=16, dropout=0.5)
    assert encoder.lstm.dropout == 0.0
    signals = torch.randn(2, 8, 1)
    torch.testing.assert_close(encoder(signals), encoder(signals))


def test_json_config_matches_defaults():
    path = Path(__file__).resolve().parents[1] / "configs/lstm_data_encoder.json"
    encoder = LSTMDataEncoder(**json.loads(path.read_text(encoding="utf-8")))
    assert encoder.lstm.hidden_size == 256
    assert encoder.lstm.num_layers == 2
    assert encoder.lstm.dropout == 0.1
    assert encoder.projection.in_features == 256
    assert encoder.projection.out_features == 4096
    assert encoder.sequence_length == 1024
    assert encoder.input_size == 1
    assert not encoder.lstm.bidirectional


@pytest.mark.parametrize("shape", [(1024, 1), (2, 1, 1024), (2, 512, 1),
                                   (2, 1024, 2), (0, 1024, 1)])
def test_reject_invalid_shapes(shape):
    with pytest.raises(ValueError, match="shape"):
        LSTMDataEncoder()(torch.zeros(shape))


def test_reject_integer_input():
    with pytest.raises(TypeError, match="floating-point"):
        LSTMDataEncoder()(torch.zeros(1, 1024, 1, dtype=torch.long))


@pytest.mark.parametrize("kwargs", [
    {"input_size": 0}, {"sequence_length": 0}, {"hidden_size": -1},
    {"num_layers": 0}, {"embedding_dim": 0}, {"hidden_size": 3.5},
    {"num_layers": True}, {"dropout": -0.1}, {"dropout": 1.0},
    {"dropout": float("nan")}, {"bidirectional": "false"},
])
def test_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        LSTMDataEncoder(**kwargs)

"""Stage 1 equations, frozen text features, split hygiene and exact resume."""

import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.alignment.contrastive_loss import BidirectionalContrastiveLoss, cosine_logits
from src.alignment.encoders import LSTMDataEncoder
from src.alignment.encoders.text_encoder import TextEncoder, encode_text_bank
from src.alignment.train_alignment import AlignmentModel
from src.utils.data import CWRUDataset, load_descriptions, make_loaders, stratified_split
from src.utils.runtime import load_config, project_path, seed_everything, select_device
from src.utils.training import Trainer, read_checkpoint
from tests.helpers import small_config, tiny_llama


def test_single_positive_matches_equations_3_to_6():
    data, text = torch.randn(4, 8, requires_grad=True), torch.randn(4, 8)
    criterion = BidirectionalContrastiveLoss()
    logits = cosine_logits(data, text) / (0.07 + 1e-6)
    expected = (F.cross_entropy(logits, torch.arange(4))
                + F.cross_entropy(logits.T, torch.arange(4))) / 2
    loss = criterion(data, text)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(data.grad).all()


def test_same_class_samples_are_all_positive():
    criterion = BidirectionalContrastiveLoss()
    features = torch.randn(3, 8, requires_grad=True)
    labels = torch.zeros(3, dtype=torch.long)
    loss = criterion(features, features.detach(), labels, labels)
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    torch.testing.assert_close(features.grad, torch.zeros_like(features))


def test_partial_class_bank_and_trainable_temperature():
    criterion = BidirectionalContrastiveLoss(learn_temperature=True)
    data, text = torch.randn(3, 8, requires_grad=True), torch.randn(5, 8)
    loss = criterion(data, text, torch.tensor([0, 0, 1]), torch.tensor([0, 0, 1, 1, 2]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(criterion.log_temperature.grad)
    assert data.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="matching"):
        criterion(data, text, torch.tensor([0, 1, 3]), torch.tensor([0, 0, 1, 1, 2]))


@pytest.mark.parametrize("pooling", ["mean", "last"])
def test_text_encoder_masked_pooling_and_freeze(pooling):
    lm, tokenizer = tiny_llama()
    encoder = TextEncoder(lm.model, tokenizer, pooling=pooling)
    encoder.train()
    assert not encoder.model.training
    assert all(not p.requires_grad for p in encoder.parameters())
    result = encoder(["normal", "fault bearing"])
    assert result.shape == (2, 16) and not result.requires_grad
    # 批次中的额外填充不应改变较短句子的编码结果。
    torch.testing.assert_close(result[:1], encoder(["normal"]), atol=1e-5, rtol=1e-5)
    assert encode_text_bank(encoder, ["normal", "fault bearing"], 1).shape == (2, 16)
    assert all(p.grad is None for p in encoder.parameters())


def test_real_loader_from_tiny_local_model(tmp_path):
    lm, tokenizer = tiny_llama()
    lm.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    encoder = TextEncoder.from_config({"name_or_path": str(tmp_path), "local_files_only": True,
                                       "dtype": "float32"}, {"pooling": "mean", "max_length": 32},
                                      torch.device("cpu"))
    assert encoder(["normal"]).shape == (1, 16)


def test_existing_cwru_inputs_and_validation_split():
    config = load_config()
    train, validation = make_loaders(config, "alignment")
    assert not set(train.dataset.indices) & set(validation.dataset.indices)
    assert len(train.dataset) + len(validation.dataset) == 1748
    for subset in (train.dataset, validation.dataset):
        assert set(subset.dataset.labels[subset.indices].tolist()) == set(range(19))
    dataset = CWRUDataset(project_path(config["data"]["test_path"]))
    assert len(dataset) == 437 and dataset[0][0].shape == (1024, 1)
    names, texts, labels = load_descriptions(config["data"]["descriptions_path"], 19)
    assert len(names) == 19 and len(texts) == len(labels) == 38
    first = stratified_split(np.array([0, 0, 1, 1]), 0.2, 42)
    assert first == stratified_split(np.array([0, 0, 1, 1]), 0.2, 42)


def test_alignment_trainer_checkpoint_and_exact_resume(tmp_path):
    config = small_config()
    config["alignment"].update(epochs=2, gradient_accumulation_steps=2)
    seed_everything(123)
    signals, labels = torch.randn(9, 16, 1), torch.arange(9) % 3
    text = torch.randn(3, 16)
    initial = LSTMDataEncoder(**config["data_encoder"]).state_dict()

    def setup(config, path):
        encoder = LSTMDataEncoder(**config["data_encoder"])
        encoder.load_state_dict(initial)
        model = AlignmentModel(encoder, text, torch.arange(3), BidirectionalContrastiveLoss())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        return Trainer(model, optimizer, torch.device("cpu"), path, config, "alignment")

    def loader():
        return DataLoader(TensorDataset(signals, labels), batch_size=4, shuffle=True)

    uninterrupted = setup(config, tmp_path / "full")
    seed_everything(42)
    full = uninterrupted.fit(loader(), loader())
    one_epoch = copy.deepcopy(config)
    one_epoch["alignment"]["epochs"] = 1
    interrupted = setup(one_epoch, tmp_path / "resumed")
    seed_everything(42)
    interrupted.fit(loader(), loader())
    resumed = setup(config, tmp_path / "resumed")
    history = resumed.fit(loader(), loader(), resume=tmp_path / "resumed/last.pt")
    assert history == full
    for key, value in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    for file in ("best.pt", "last.pt", "history.json", "metrics.jsonl", "train.log", "curves.png"):
        assert (tmp_path / "resumed" / file).is_file()
    checkpoint = read_checkpoint(tmp_path / "resumed/last.pt")
    assert checkpoint["epoch"] == 1 and checkpoint["stage"] == "alignment"


def test_cpu_and_missing_weights_fail_without_fallback(tmp_path):
    assert select_device("cpu").type == "cpu"
    with pytest.raises((OSError, ValueError)):
        TextEncoder.from_config({"name_or_path": str(tmp_path), "local_files_only": True},
                                {"pooling": "mean"}, torch.device("cpu"))

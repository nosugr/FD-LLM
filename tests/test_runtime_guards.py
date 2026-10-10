"""Prevent stale text banks and reject mismatched windows before model loading."""

import json

import pytest
import torch

from src.alignment.contrastive_loss import BidirectionalContrastiveLoss
from src.alignment.encoders import LSTMDataEncoder
from src.alignment.train_alignment import AlignmentModel
from src.utils.data import make_loaders
from src.utils.runtime import load_config
from src.utils.training import Trainer, atomic_save
from tests.helpers import small_config


def test_same_path_description_edit_invalidates_resume(tmp_path):
    descriptions = tmp_path / "texts.json"
    descriptions.write_text(json.dumps({"text": "normal"}), encoding="utf-8")
    config = small_config()
    config["data"]["descriptions_path"] = str(descriptions)
    model = AlignmentModel(LSTMDataEncoder(**config["data_encoder"]), torch.randn(2, 16),
                           torch.arange(2), BidirectionalContrastiveLoss())
    trainer = Trainer(model, torch.optim.AdamW(model.parameters()), torch.device("cpu"),
                       tmp_path, config, "alignment")
    checkpoint = tmp_path / "checkpoint.pt"
    atomic_save({"stage": "alignment", "signature": trainer.signature()}, checkpoint)
    descriptions.write_text(json.dumps({"text": "changed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        trainer.resume(checkpoint)


def test_window_config_mismatch_is_rejected_before_backbone_load():
    config = load_config()
    config["data_encoder"]["sequence_length"] = 16
    with pytest.raises(ValueError, match="signal shape"):
        make_loaders(config, "alignment")

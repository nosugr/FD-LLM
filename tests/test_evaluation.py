"""Numerical metric expectations, stable class axes and exported figures."""

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluate import evaluate_model, save_evaluation
from src.evaluation.metrics import classification_metrics
from src.evaluation.visualization import plot_tsne
from tests.helpers import small_config
from tests.test_diagnosis import make_model


def test_hand_computed_classification_metrics():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 3)
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]
    assert metrics["accuracy"] == 0.75
    assert metrics["precision_macro"] == pytest.approx((1 + 2 / 3) / 3)
    assert metrics["recall_macro"] == pytest.approx(1.5 / 3)
    assert metrics["f1_macro"] == pytest.approx((2 / 3 + 0.8) / 3)
    assert metrics["per_class"][2]["support"] == 0


def test_metrics_keep_all_19_classes():
    metrics = classification_metrics([0, 18], [0, 0])
    assert np.asarray(metrics["confusion_matrix"]).shape == (19, 19)
    assert len(metrics["per_class"]) == 19
    with pytest.raises(ValueError):
        classification_metrics([19], [0])
    with pytest.raises(ValueError):
        classification_metrics([], [])


def test_evaluate_and_export_artifacts(tmp_path):
    config = small_config()
    model = make_model(config)
    loader = DataLoader(TensorDataset(torch.randn(6, 16, 1), torch.tensor([0, 1, 2, 0, 1, 2])), batch_size=2)
    metrics, arrays = evaluate_model(model, loader, torch.device("cpu"), model.class_names)
    assert arrays["features"].shape == (6, 16) and arrays["scores"].shape == (6, 3)
    assert metrics["sample_count"] == 6
    save_evaluation(metrics, arrays, tmp_path, config)
    for file in ("metrics.json", "predictions.npz", "tsne.npz", "confusion_matrix.png",
                 "confusion_matrix_normalized.png", "tsne.png"):
        assert (tmp_path / file).stat().st_size > 0
    for file in ("confusion_matrix.png", "confusion_matrix_normalized.png", "tsne.png"):
        with Image.open(tmp_path / file) as image:
            assert min(image.size) > 400


def test_tsne_subsampling_and_reproducibility(tmp_path):
    features = np.random.default_rng(42).normal(size=(12, 64)).astype(np.float32)
    labels = np.arange(12) % 3
    coordinates, indices = plot_tsne(features, labels, ["a", "b", "c"], tmp_path / "first.png",
                                     max_samples=6, perplexity=30)
    other, other_indices = plot_tsne(features, labels, ["a", "b", "c"], tmp_path / "second.png",
                                    max_samples=6, perplexity=30)
    assert coordinates.shape == (6, 2)
    np.testing.assert_array_equal(indices, other_indices)
    np.testing.assert_allclose(coordinates, other)

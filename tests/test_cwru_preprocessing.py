import numpy as np
import pytest
import yaml
from scipy.io import savemat

from data.preprocessing.CWRU.manifest import CLASSES, ClassSource
from data.preprocessing.CWRU import pipeline


def test_table6_file_mapping_and_fields():
    assert [item.label for item in CLASSES] == list(range(19))
    assert [item.source_file.split("/")[-1] for item in CLASSES] == [
        "97.mat", "278.mat", "274.mat", "270.mat", "282.mat", "286.mat", "290.mat",
        "294.mat", "313.mat", "315.mat", "105.mat", "169.mat", "209.mat",
        "118.mat", "185.mat", "222.mat", "130.mat", "197.mat", "234.mat",
    ]
    assert CLASSES[0].signal_field == "X097_DE_time"
    assert all(item.signal_field.endswith("_FE_time") for item in CLASSES[1:10])
    assert all(item.signal_field.endswith("_DE_time") for item in CLASSES[10:])
    assert all(item.outer_race_position == "@6" for item in (*CLASSES[7:10], *CLASSES[16:19]))


def test_load_signal_uses_explicit_field(tmp_path):
    source = ClassSource(1, "fan_end", "inner_race", 0.007, "278.mat", "X278_FE_time")
    savemat(tmp_path / "278.mat", {"i": np.arange(10000),
                                   "X278_DE_time": np.full((4, 1), 99),
                                   "X278_FE_time": np.arange(4).reshape(-1, 1),
                                   "X278RPM": [[1796]]})
    signal, rpm = pipeline.load_signal(tmp_path, source)
    np.testing.assert_array_equal(signal, np.arange(4))
    assert rpm == 1796
    with pytest.raises(ValueError, match="expected signal field"):
        pipeline.load_signal(tmp_path, ClassSource(1, "fan_end", "inner_race", 0.007,
                                                       "278.mat", "X278_WRONG_time"))


def test_temporal_blocks_and_insufficient_windows():
    train, test = pipeline.select_window_indices(117, 115, 92)
    np.testing.assert_array_equal(train, np.arange(92))
    np.testing.assert_array_equal(test, np.arange(94, 117))
    assert test[0] - train[-1] == 3
    with pytest.raises(ValueError, match="114 nonoverlapping windows available; 115 required"):
        pipeline.select_window_indices(114, 115, 92)


def test_standardization_uses_training_data_only():
    train = np.array([0., 2., 4., 6.]).reshape(2, 2, 1)
    test = np.array([100., 102.]).reshape(1, 2, 1)
    normalized_train, normalized_test, mean, std = pipeline.standardize(train, test)
    assert mean == 3.0
    assert std == pytest.approx(np.sqrt(5))
    assert normalized_train.dtype == normalized_test.dtype == np.float32
    np.testing.assert_allclose(normalized_test, ((test - mean) / std).astype(np.float32))
    assert abs(normalized_train.mean()) < 1e-7


def test_end_to_end_outputs_and_reproducibility(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "CLASSES", (
        ClassSource(0, "normal", "none", None, "raw/97.mat", "X097_DE_time"),
        ClassSource(1, "fan_end", "inner_race", 0.007, "raw/278.mat", "X278_FE_time"),
    ))
    raw = tmp_path / "raw" / "raw"
    raw.mkdir(parents=True)
    savemat(raw / "97.mat", {"X097_DE_time": np.arange(117 * 1024).reshape(-1, 1)})
    savemat(raw / "278.mat", {"X278_FE_time": np.arange(117 * 1024).reshape(-1, 1) + 1000})
    config = {"raw_dir": "raw", "output_dir": "processed", "window_length": 1024,
              "stride": 1024, "samples_per_class": 115, "train_ratio": 0.8, "random_seed": 42}
    config_path = tmp_path / "cwru.yaml"
    config_path.write_text(yaml.safe_dump({"preprocessing": config}), encoding="utf-8")
    metadata = pipeline.build_dataset(config_path)
    assert (metadata["train_count"], metadata["test_count"], metadata["class_count"]) == (184, 46, 2)
    with np.load(tmp_path / "processed" / "train.npz") as train, np.load(tmp_path / "processed" / "test.npz") as test:
        assert train["X"].shape == (184, 1024, 1)
        assert test["X"].shape == (46, 1024, 1)
        assert train["X"].dtype == test["X"].dtype == np.float32
        assert train["X_raw"].dtype == np.float64
        np.testing.assert_array_equal(np.bincount(train["y"]), [92, 92])
        np.testing.assert_array_equal(np.bincount(test["y"]), [23, 23])
        for label in (0, 1):
            assert test["window_index"][test["y"] == label].min() - train["window_index"][train["y"] == label].max() == 3
        assert abs(float(train["X"].mean())) < 1e-6
        assert abs(float(train["X"].std()) - 1) < 1e-6
        np.testing.assert_allclose(test["X"], ((test["X_raw"] - metadata["normalization"]["mean"])
                                              / metadata["normalization"]["std"]).astype(np.float32))
        first_order = train["window_index"].copy()
    pipeline.build_dataset(config_path)
    with np.load(tmp_path / "processed" / "train.npz") as repeated:
        np.testing.assert_array_equal(repeated["window_index"], first_order)

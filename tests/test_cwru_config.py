"""Check the merged configuration and its three consumer entry points."""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from data.preprocessing.CWRU import descriptions, pipeline, visualization
from data.preprocessing.CWRU.config import DEFAULT_CONFIG, load_section


def test_shared_defaults_and_legacy_flat_config(tmp_path):
    assert pipeline.DEFAULT_CONFIG == descriptions.DEFAULT_CONFIG == visualization.DEFAULT_CONFIG == DEFAULT_CONFIG
    preprocessing = pipeline.load_config()
    assert (preprocessing["window_length"], preprocessing["stride"],
            preprocessing["samples_per_class"], preprocessing["train_ratio"],
            preprocessing["random_seed"]) == (1024, 1024, 115, 0.8, 42)
    assert load_section(DEFAULT_CONFIG, "visualization")["points"] == 1600
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(preprocessing), encoding="utf-8")
    assert pipeline.load_config(legacy) == preprocessing


@pytest.mark.parametrize("content", ["[]", "preprocessing: {}", "visualization: {dpi: 180}"])
def test_invalid_or_missing_section(tmp_path, content):
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_section(path, "preprocessing")


def test_description_entry_reads_shared_processed_dir(tmp_path):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["preprocessing"]["output_dir"] = str(tmp_path)
    config_path = tmp_path / "cwru.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metadata_text = (pipeline.PROJECT_ROOT / "data/datasets/CWRU/processed/metadata.json").read_text(encoding="utf-8")
    (tmp_path / "metadata.json").write_text(metadata_text, encoding="utf-8")
    for split in ("train", "test"):
        np.savez(tmp_path / f"{split}.npz", y=np.arange(19))
    output = descriptions.generate_descriptions(config_path)
    result = json.loads(output.read_text(encoding="utf-8"))
    expected = descriptions.build_class_descriptions(json.loads(metadata_text), config["descriptions"])
    assert output == tmp_path / "class_descriptions.json"
    assert result == expected


def test_visualization_entry_reads_shared_paths(tmp_path, monkeypatch):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["preprocessing"]["raw_dir"] = str(tmp_path / "raw")
    config["preprocessing"]["output_dir"] = str(tmp_path / "processed")
    config["visualization"].update(output_dir=str(tmp_path / "figures"), start_index=3, points=8)
    config_path = tmp_path / "cwru.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def waveforms(raw_dir, *, start, points):
        assert raw_dir == tmp_path / "raw"
        assert (start, points) == (3, 8)
        return {"normal": plt.figure(figsize=(1, 1))}

    def distribution(processed_dir):
        assert processed_dir == tmp_path / "processed"
        return plt.figure(figsize=(1, 1))

    monkeypatch.setattr(visualization, "create_waveform_figures", waveforms)
    monkeypatch.setattr(visualization, "create_distribution_figure", distribution)
    monkeypatch.setattr(sys, "argv", ["visualization", "--config", str(config_path), "--distribution"])
    visualization.main()
    assert (tmp_path / "figures/cwru_normal_waveform.png").is_file()
    assert (tmp_path / "figures/cwru_class_distribution.png").is_file()

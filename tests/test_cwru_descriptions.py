"""Validate class text correspondence, prompt combinations and invalid inputs."""

import copy
import json

import pytest

from data.preprocessing.CWRU.config import load_section
from data.preprocessing.CWRU.descriptions import DEFAULT_CONFIG, build_class_descriptions
from data.preprocessing.CWRU.pipeline import PROJECT_ROOT


@pytest.fixture
def inputs():
    metadata = json.loads((PROJECT_ROOT / "data/datasets/CWRU/processed/metadata.json").read_text(encoding="utf-8"))
    config = load_section(DEFAULT_CONFIG, "descriptions")
    return metadata, config


def test_all_class_texts_match_metadata(inputs):
    metadata, config = inputs
    result = build_class_descriptions(metadata, config)
    assert [record["class_id"] for record in result["classes"]] == list(range(19))
    assert len({record["class_name"] for record in result["classes"]}) == 19
    for record, source in zip(result["classes"], sorted(metadata["sources"], key=lambda item: item["label"])):
        assert record["source_file"] == source["source_file"]
        assert record["attributes"]["rpm"] == source["rpm"]
        assert record["attributes"]["load_hp"] == 0
        assert len(record["descriptions"]) == 2
        assert record["description"] == record["descriptions"][0]["text"]
        for variant in record["descriptions"]:
            assert f'{source["rpm"]} rpm' in variant["text"]
            assert "0 HP" in variant["text"]
            if source["label"] == 0:
                assert "healthy" in variant["text"]
                assert "defect diameter" not in variant["text"]
                assert "installed at" not in variant["text"]
            else:
                assert f'{source["fault_size_inches"]:.3f} inches' in variant["text"]
                assert source["bearing_position"].replace("_", " ") in variant["text"]
                assert record["state"]["fault_type"] == source["fault_type"]


def test_prompt_sets_extend_by_cartesian_product(inputs):
    metadata, config = inputs
    config["state_prompts"]["fault"]["alternative"] = (
        "A {fault_name} defect is present; its diameter is {fault_size_inches:.3f} inches.")
    records = build_class_descriptions(metadata, config)["classes"]
    assert len(records[0]["descriptions"]) == 2
    assert all(len(record["descriptions"]) == 4 for record in records[1:])


@pytest.mark.parametrize("problem", ["duplicate_label", "mapping", "rpm", "load", "placeholder"])
def test_invalid_inputs_fail_before_generation(inputs, problem):
    metadata, config = copy.deepcopy(inputs)
    if problem == "duplicate_label":
        metadata["sources"][1]["label"] = 0
    elif problem == "mapping":
        metadata["sources"][1]["fault_type"] = "ball"
    elif problem == "rpm":
        metadata["sources"][0]["rpm"] = None
    elif problem == "load":
        config["load_hp_by_source_folder"].pop("normal")
    else:
        config["attribute_prompts"]["operating_conditions"] = "Unknown {missing}"
    with pytest.raises(ValueError):
        build_class_descriptions(metadata, config)

"""Generate class texts using attribute/state prompt sets (FD-LLM section 3.1).

Run from the project root: python -m src.data_processing.descriptions
Add prompt entries in configs/cwru.yaml (descriptions) to extend text variants.
No language model or signal transformation is needed for template generation.
"""

import argparse
import json
import math
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np

from .config import DEFAULT_CONFIG, load_section
from .manifest import CLASSES
from .pipeline import PROJECT_ROOT, load_config


FAULT_NAMES = {"inner_race": "inner race", "outer_race": "outer race", "ball": "rolling element (ball)"}
POSITIONS = {"fan_end": "fan end", "drive_end": "drive end"}


def _prompt_set(prompts: dict) -> None:
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("Each prompt set must be a nonempty mapping")
    if any(not isinstance(key, str) or not key or not isinstance(text, str) or not text.strip()
           for key, text in prompts.items()):
        raise ValueError("Prompt IDs and templates must be nonempty strings")


def build_class_descriptions(metadata: dict, config: dict) -> dict:
    """Read the saved mapping and compose every attribute/state prompt pair.

    Manifest comparison prevents silently pairing stale or changed labels with
    the current CWRU dataset. Missing operating conditions raise an error.
    The first variant is the default description, following config entry order.
    """
    sources = metadata["sources"]
    labels = [source["label"] for source in sources]
    expected = {source.label: asdict(source) for source in CLASSES}
    # 核对保存的类别映射与当前清单，避免旧标签和新描述被错误配对。
    if (metadata["class_count"] != len(expected) or len(sources) != len(expected)
            or any(type(label) is not int for label in labels)
            or set(labels) != set(expected)):
        raise ValueError("Expected exactly the 19 unique CWRU labels 0 through 18")
    attributes = config["attribute_prompts"]
    _prompt_set(attributes)
    for state in ("normal", "fault"):
        _prompt_set(config["state_prompts"][state])
    _prompt_set({"combination": config["combination_template"]})

    records = []
    for source in sorted(sources, key=lambda item: item["label"]):
        label = source["label"]
        if any(source.get(key) != value for key, value in expected[label].items()):
            raise ValueError(f"Class {label}: saved mapping disagrees with the CWRU manifest")
        rpm = source.get("rpm")
        if isinstance(rpm, bool) or not isinstance(rpm, (int, float)) or not math.isfinite(rpm) or rpm <= 0:
            raise ValueError(f"Class {label}: missing or invalid measured rpm")
        folder = Path(source["source_file"]).parts[0]
        load = source.get("load_hp", config["load_hp_by_source_folder"].get(folder))
        if isinstance(load, bool) or not isinstance(load, (int, float)) or not math.isfinite(load) or load < 0:
            raise ValueError(f"Class {label}: missing or invalid motor load")
        normal = source["fault_type"] == "none"
        position = source["bearing_position"]
        location = ("No fault location is specified for this normal baseline."
                    if normal else f"The bearing is installed at the {POSITIONS[position]} of the motor.")
        context = {**source, "load_hp": load, "location_sentence": location,
                   "fault_name": "none" if normal else FAULT_NAMES[source["fault_type"]]}
        name = ("Normal" if normal else
                f'{POSITIONS[position].capitalize()} / {context["fault_name"]} / {source["fault_size_inches"]:.3f} in')
        variants = []
        states = config["state_prompts"]["normal" if normal else "fault"]
        # 组合属性与状态模板；配置中的首个组合为该类默认描述。
        for (attribute_id, attribute), (state_id, state) in product(attributes.items(), states.items()):
            try:
                attribute_text = attribute.format(**context)
                state_text = state.format(**context)
                text = config["combination_template"].format(
                    attribute_description=attribute_text, state_description=state_text)
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"Class {label}, prompts {attribute_id}/{state_id}: {exc}") from exc
            if not text.strip() or "{" in text or "}" in text:
                raise ValueError(f"Class {label}: empty text or unresolved template placeholders")
            variants.append({"attribute_prompt_id": attribute_id, "state_prompt_id": state_id,
                             "text": text})
        records.append({"class_id": label, "class_name": name, "description": variants[0]["text"],
                        "descriptions": variants, "attributes": {
                            "bearing_position": position, "load_hp": load, "rpm": rpm},
                        "state": {"status": "normal" if normal else "fault",
                                  "fault_type": source["fault_type"],
                                  "defect_diameter_inches": source["fault_size_inches"]},
                        "source_file": source["source_file"], "signal_field": source["signal_field"]})
    return {"schema_version": 1, "dataset": "CWRU", "language": "en", "class_count": len(records),
            "method": "Cartesian combination of attribute and state prompt sets; FD-LLM section 3.1",
            "attribute_provenance": {"rpm": "processed metadata, measured source-file RPM",
                                     "load_hp": config["load_reference"],
                                     "bearing_position": "processed class mapping; normal has no fault location"},
            "prompt_sets": {key: config[key] for key in
                            ("attribute_prompts", "state_prompts", "combination_template")},
            "classes": records}


def generate_descriptions(config_path: Path = DEFAULT_CONFIG) -> Path:
    config = load_section(config_path, "descriptions")
    # 默认共用 cwru.yaml 的预处理分组，也兼容旧配置的外部文件引用。
    preprocessing_path = (PROJECT_ROOT / config["preprocessing_config"]
                          if "preprocessing_config" in config else config_path)
    preprocessing = load_config(preprocessing_path)
    processed_dir = PROJECT_ROOT / preprocessing["output_dir"]
    metadata = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
    result = build_class_descriptions(metadata, config)
    expected_labels = np.array([item["class_id"] for item in result["classes"]])
    # 写入前确认描述涵盖训练及测试的全部类别，不改动已有样本。
    for split in ("train", "test"):
        with np.load(processed_dir / f"{split}.npz", allow_pickle=False) as arrays:
            labels = arrays["y"]
            if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError(f"{split}: labels must be a one-dimensional integer array")
            if not np.array_equal(np.unique(labels), expected_labels):
                raise ValueError(f"{split}: saved labels disagree with description class IDs")
    output = processed_dir / "class_descriptions.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CWRU class descriptions from prompt sets")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(generate_descriptions(args.config))


if __name__ == "__main__":
    main()

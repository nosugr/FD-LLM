"""Offline CWRU -> alignment -> LoRA diagnosis -> evaluation connectivity test.

This test-only entry point creates random tiny Llama weights locally. Production
training imports no test fixtures and never substitutes unavailable Vicuna weights.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.alignment.train_alignment import run as run_alignment
from src.diagnosis.train_diagnosis import load_diagnosis, run as run_diagnosis
from src.evaluation.evaluate import run as run_evaluation
from src.utils.runtime import project_path, seed_everything
from src.utils.training import read_checkpoint
from tests.helpers import small_config, tiny_llama


def run_smoke(output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    seed_everything(42)
    config = small_config()
    original_data = dict(config["data"])
    # 保留真实的 1024 点输入窗口，仅缩小语言模型及隐藏层用于离线验证。
    config["data_encoder"]["sequence_length"] = 1024
    config["data_encoder"]["num_layers"] = 1
    for split, per_class in (("train", 2), ("test", 1)):
        source = project_path(original_data[f"{split}_path"])
        with np.load(source, allow_pickle=False) as archive:
            indices = np.concatenate([np.flatnonzero(archive["y"] == label)[:per_class]
                                      for label in range(19)])
            if len(indices) != 19 * per_class:
                raise ValueError("Smoke test needs sufficient real CWRU samples for all 19 classes")
            target = output_dir / f"{split}.npz"
            np.savez_compressed(target, X=archive["X"][indices], y=archive["y"][indices])
        config["data"][f"{split}_path"] = str(target)
    config["data"]["descriptions_path"] = str(project_path(original_data["descriptions_path"]))
    config["data"]["validation_fraction"] = 0.5
    base, tokenizer = tiny_llama()
    # 测试替身保存到独立目录，随后验证正式加载接口，不替换生产默认配置。
    base_dir = output_dir / "tiny_random_llama"
    base.save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)
    config["model"].update(name_or_path=str(base_dir), local_files_only=True, dtype="float32")
    config["alignment"].update(batch_size=19, text_batch_size=4, output_dir=str(output_dir / "alignment"))
    config["diagnosis"].update(batch_size=19, output_dir=str(output_dir / "diagnosis"),
                               alignment_checkpoint=str(output_dir / "alignment/best.pt"))
    config["evaluation"].update(batch_size=4, candidate_chunk_size=4, output_dir=str(output_dir / "evaluation"))
    config_path = output_dir / "smoke_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    alignment = run_alignment(config)
    diagnosis = run_diagnosis(config)
    checkpoint = output_dir / "diagnosis/best.pt"
    metrics = run_evaluation(config, checkpoint)
    model = load_diagnosis(config, checkpoint, torch.device("cpu"))
    with np.load(config["data"]["test_path"]) as archive:
        signals = torch.from_numpy(archive["X"][:2])
    predictions = model.predict(signals)
    generated = model.generate(signals[:1], max_new_tokens=4)
    saved = read_checkpoint(checkpoint)
    assert len(saved["model"]["class_names"]) == 19
    assert predictions["scores"].shape == (2, 19)
    assert np.asarray(metrics["confusion_matrix"]).shape == (19, 19)
    assert metrics["sample_count"] == 19
    summary = {"purpose": "Offline connectivity only; random tiny Llama, not Vicuna accuracy",
               "source_train_path": str(project_path(original_data["train_path"])),
               "source_test_path": str(project_path(original_data["test_path"])),
               "input_shape": list(signals.shape), "data_feature_shape": list(predictions["features"].shape),
               "class_score_shape": list(predictions["scores"].shape), "class_ids": predictions["class_ids"].tolist(),
               "class_names": [model.class_names[index] for index in predictions["class_ids"].tolist()],
               "generated_test_text": generated, "alignment_history": alignment,
               "diagnosis_history": diagnosis, "test_sample_count": metrics["sample_count"],
               "test_accuracy_not_a_performance_result": metrics["accuracy"],
               "config": str(config_path), "checkpoint": str(checkpoint)}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"smoke": "passed", "class_score_shape": summary["class_score_shape"],
                      "output_dir": str(output_dir)}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(project_path("outputs/smoke")))
    args = parser.parse_args()
    run_smoke(args.output)


if __name__ == "__main__":
    main()

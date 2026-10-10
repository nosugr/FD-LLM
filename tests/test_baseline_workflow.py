"""Uniform splits, complete offline pipeline, shuffle resume and repeat statistics."""

import copy
import json

import numpy as np
import pytest
import torch

from baselines.comparison import import_fd_result, summarize_results, write_comparison
from baselines.data import PairBatchSampler, make_loaders, prepare_data
from baselines.evaluate import load_model, run as evaluate_checkpoint
from baselines.factory import MODEL_NAMES, build_classifier
from baselines.tasks import BaselineTrainer, PretrainingTask, SupervisedTask
from baselines.train import DEFAULT_CONFIG, run, smoke_config
from baselines.factory import build_simclr
from src.utils.runtime import load_config, seed_everything
from src.utils.training import atomic_save, read_checkpoint


def test_fixed_split_and_model_independent_batch_order():
    config = smoke_config(load_config(DEFAULT_CONFIG))
    prepared = prepare_data(config)
    assert not set(prepared["indices"]["train"]) & set(prepared["indices"]["validation"])
    assert len(prepared["train"]) == 38 and len(prepared["validation"]) == 19
    assert len(prepared["test"]) == 19
    orders = []
    for name in ("cnn", "resnet18"):
        build_classifier(name, config["models"][name], config["data"])
        loader, _ = make_loaders(prepared, config, "training", torch.Generator().manual_seed(42), torch.device("cpu"))
        orders.append(torch.cat([labels for _, labels in loader]))
    torch.testing.assert_close(*orders, rtol=0, atol=0)
    full_config = load_config(DEFAULT_CONFIG)
    full = prepare_data(full_config)
    assert len(full["train"]) + len(full["validation"]) == 1748 and len(full["test"]) == 437


@pytest.mark.parametrize("length,batch_size", [(2, 4), (5, 2), (7, 3), (8, 4), (10, 3)])
def test_pair_batches_merge_singleton_without_losing_samples(length, batch_size):
    sampler = PairBatchSampler(list(range(length)), batch_size)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert all(len(batch) >= 2 for batch in batches)
    assert [index for batch in batches for index in batch] == list(range(length))


def test_all_six_models_uniform_smoke_pipeline(tmp_path):
    results = run(load_config(DEFAULT_CONFIG), smoke=True, output_dir=tmp_path)
    assert [result["model_name"] for result in results] == list(MODEL_NAMES)
    assert all(result["metrics"]["sample_count"] == 19 for result in results)
    assert all(result["protocol"] == results[0]["protocol"] for result in results)
    assert all(result["split"] == results[0]["split"] for result in results)
    for name in MODEL_NAMES:
        root = tmp_path / name / "seed_42"
        for file in ("config.yaml", "split_indices.npz", "training/best.pt", "training/last.pt",
                     "training/train.log", "training/history.json", "test/metrics.json",
                     "test/confusion_matrix.png", "test/tsne.png", "result.json"):
            assert (root / file).stat().st_size > 0
        restored, _ = load_model(root / "training/best.pt", torch.device("cpu"))
        assert restored(torch.randn(1, 1024, 1)).shape == (1, 19)
    assert (tmp_path / "simclr/seed_42/pretraining/best.pt").is_file()
    assert results[-1]["pretraining_epochs"] == 1
    metrics = evaluate_checkpoint(tmp_path / "cnn/seed_42/training/best.pt", device="cpu", output_dir=tmp_path / "reevaluation")
    assert metrics == results[0]["metrics"]
    for file in ("comparison.json", "comparison.csv", "comparison.md"):
        assert (tmp_path / file).is_file()


def test_two_repeated_cpu_smoke_runs_and_statistics(tmp_path):
    results = run(load_config(DEFAULT_CONFIG), models=["cnn"], repeats=2, smoke=True, output_dir=tmp_path)
    assert [item["seed"] for item in results] == [42, 43]
    summary = json.loads((tmp_path / "comparison.json").read_text())
    assert summary["rows"][0]["runs"] == 2
    assert summary["rows"][0]["accuracy"]["std"] is not None
    assert results[0]["split"] == results[1]["split"]


def test_ten_run_sample_standard_deviation_and_protocol_guard(tmp_path):
    values = np.linspace(0.1, 1.0, 10)
    results = [{"model_name": "cnn", "seed": i, "protocol": {"execution": "experiment"},
                "metrics": {key: value for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")}}
               for i, value in enumerate(values)]
    summary = write_comparison(results, tmp_path)
    assert summary["rows"][0]["accuracy"]["mean"] == pytest.approx(values.mean())
    assert summary["rows"][0]["accuracy"]["std"] == pytest.approx(values.std(ddof=1))
    invalid = copy.deepcopy(results)
    invalid[-1]["protocol"]["execution"] = "smoke"
    with pytest.raises(ValueError, match="different data"):
        summarize_results(invalid)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_results(results + results[:1])


def test_baseline_exact_epoch_boundary_resume_including_shuffle_rng(tmp_path):
    config = smoke_config(load_config(DEFAULT_CONFIG))
    prepared = prepare_data(config)
    config["protocol"], config["split"] = prepared["protocol"], prepared["split"]
    config["model"] = {"name": "cnn", "architecture": config["models"]["cnn"]}
    config["training"]["epochs"] = 2

    def setup(settings, directory):
        seed_everything(123)
        model = build_classifier("cnn", settings["models"]["cnn"], settings["data"])
        generator = torch.Generator().manual_seed(42)
        task = SupervisedTask(model, generator)
        trainer = BaselineTrainer(task, torch.optim.AdamW(task.parameters(), lr=0.001),
                                  torch.device("cpu"), directory, settings, "training")
        loaders = make_loaders(prepared, settings, "training", generator, torch.device("cpu"))
        return trainer, loaders

    full, full_loaders = setup(config, tmp_path / "full")
    seed_everything(42)
    history = full.fit(*full_loaders)
    partial_config = copy.deepcopy(config)
    partial_config["training"]["epochs"] = 1
    partial, partial_loaders = setup(partial_config, tmp_path / "partial")
    seed_everything(42)
    partial.fit(*partial_loaders)
    restored, restored_loaders = setup(config, tmp_path / "partial")
    actual = restored.fit(*restored_loaders, resume=tmp_path / "partial/last.pt")
    assert actual == history
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)


def test_simclr_pretraining_does_not_use_labels_and_validation_views_are_fixed():
    config = smoke_config(load_config(DEFAULT_CONFIG))
    architecture = config["models"]["simclr"]
    task = PretrainingTask(build_simclr(architecture, config["data"]), torch.Generator().manual_seed(42), architecture)
    signals = torch.randn(3, 1024, 1)
    task.eval()
    first = task.batch_metrics(signals, torch.tensor([0, 1, 2]))["loss"]
    task.eval()  # Reset fixed validation view generator.
    second = task.batch_metrics(signals, torch.tensor([18, 18, 18]))["loss"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_import_fd_llm_from_matching_full_test_artifacts(tmp_path):
    from src.utils.runtime import load_config as fd_config
    baseline = load_config(DEFAULT_CONFIG)
    prepared = prepare_data(baseline)
    config = fd_config()
    checkpoint = tmp_path / "fd.pt"
    atomic_save({"stage": "diagnosis", "config": config,
                 "model": {"class_names": prepared["class_names"]}}, checkpoint)
    evaluation = tmp_path / "fd_evaluation"
    evaluation.mkdir()
    truth = prepared["test"].dataset.labels.numpy()
    # 人工构造的全正确预测仅用于临时目录中的单元测试，不作为实验结果。
    np.savez(evaluation / "predictions.npz", y_true=truth, y_pred=truth)
    summary = import_fd_result(checkpoint, evaluation, tmp_path / "comparison")
    assert summary["rows"][0]["model"] == "fd_llm"
    assert summary["rows"][0]["accuracy"]["mean"] == 1.0
    np.savez(evaluation / "predictions.npz", y_true=truth[:19], y_pred=truth[:19])
    with pytest.raises(ValueError, match="full saved test set"):
        import_fd_result(checkpoint, evaluation, tmp_path / "comparison")

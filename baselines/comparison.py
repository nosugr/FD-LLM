"""Repeated-run summaries and a Table-7-shaped comparison, with data identity checks."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from baselines.data import indices_digest, sha256_file
from src.evaluation.metrics import classification_metrics
from src.utils.data import CWRUDataset, load_descriptions
from src.utils.runtime import project_path
from src.utils.training import read_checkpoint

METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro")
TABLE_ORDER = ("cnn", "resnet18", "transformer", "simclr", "attention_lstm", "attention_gru", "fd_llm")
DISPLAY_NAMES = {"cnn": "CNN", "resnet18": "ResNet18", "transformer": "Transformer", "simclr": "SimCLR",
                 "attention_lstm": "Attention-LSTM", "attention_gru": "Attention-GRU", "fd_llm": "FD-LLM"}


def summarize_results(results):
    if not results:
        raise ValueError("No completed experiment results found")
    protocol = results[0]["protocol"]
    grouped = {}
    for result in results:
        # 不混合不同测试集、类别顺序或 smoke/正式实验结果。
        if result["protocol"] != protocol:
            raise ValueError("Results use different data/test subsets/class orders or execution modes")
        group = grouped.setdefault(result["model_name"], [])
        if result["seed"] in {item["seed"] for item in group}:
            raise ValueError("Duplicate model/seed result would bias the repeated-run statistics")
        group.append(result)
    rows = []
    for name in sorted(grouped, key=lambda item: TABLE_ORDER.index(item) if item in TABLE_ORDER else len(TABLE_ORDER)):
        items = grouped[name]
        row = {"model": name, "runs": len(items), "seeds": sorted(item["seed"] for item in items)}
        for metric in METRICS:
            values = np.array([item["metrics"][metric] for item in items], dtype=float)
            # 使用样本标准差；单次结果不足以估计标准差，输出 null。
            row[metric] = {"mean": float(values.mean()),
                           "std": float(values.std(ddof=1)) if len(values) > 1 else None}
        rows.append(row)
    return {"protocol": protocol, "std_definition": "sample standard deviation, ddof=1; null for n=1",
            "rows": rows}


def write_comparison(results, output_dir):
    summary = summarize_results(results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["model", "runs", "seeds"] + [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary["rows"]:
            flat = {"model": row["model"], "runs": row["runs"], "seeds": ",".join(map(str, row["seeds"]))}
            flat.update({f"{metric}_{stat}": row[metric][stat] for metric in METRICS for stat in ("mean", "std")})
            writer.writerow(flat)
    mode = summary["protocol"]["execution"]
    lines = [f"Execution: {mode}. Values in percent; sample SD (n=1: unavailable).",
             "", "| Model | Runs | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["rows"]:
        cells = []
        for metric in METRICS:
            stats = row[metric]
            cells.append(f'{100 * stats["mean"]:.2f} ± {100 * stats["std"]:.2f}' if stats["std"] is not None
                         else f'{100 * stats["mean"]:.2f} (SD unavailable)')
        lines.append(f'| {DISPLAY_NAMES.get(row["model"], row["model"])} | {row["runs"]} | ' + " | ".join(cells) + " |")
    if "fd_llm" not in {row["model"] for row in summary["rows"]}:
        lines.append("| FD-LLM | 0 | Pending real experiment | — | — | — |")
    if mode == "smoke":
        lines.extend(["", "Smoke results verify connectivity only and are not model performance comparisons."])
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def collect_results(root):
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(root).glob("*/seed_*/result.json"))]


def import_fd_result(checkpoint_path, evaluation_dir, output_dir):
    """Import existing FD-LLM predictions; verify full outer split and class order.

    No LLM is loaded. This records supplied artifact provenance and recomputes
    metrics; it does not prove the predictions came from the supplied checkpoint.
    """
    checkpoint = read_checkpoint(checkpoint_path)
    if checkpoint["stage"] != "diagnosis":
        raise ValueError("Only an FD-LLM diagnosis checkpoint can enter this table")
    config = checkpoint["config"]
    test_path, train_path = project_path(config["data"]["test_path"]), project_path(config["data"]["train_path"])
    dataset = CWRUDataset(test_path)
    names, _, _ = load_descriptions(config["data"]["descriptions_path"], config["data"]["num_classes"])
    if checkpoint["model"]["class_names"] != names:
        raise ValueError("FD-LLM checkpoint class order differs from data")
    with np.load(Path(evaluation_dir) / "predictions.npz", allow_pickle=False) as archive:
        truth, predicted = archive["y_true"], archive["y_pred"]
    # 导入 FD-LLM 结果时要求覆盖完整测试集，并保持原始样本顺序。
    if not np.array_equal(truth, dataset.labels.numpy()):
        raise ValueError("FD-LLM predictions must cover the full saved test set in its original order")
    result = {"model_name": "fd_llm", "seed": config["seed"], "metrics": classification_metrics(truth, predicted, len(names), names),
              "protocol": {"train_sha256": sha256_file(train_path), "test_sha256": sha256_file(test_path),
                           "test_indices_sha256": indices_digest(range(len(dataset))), "test_count": len(dataset),
                           "class_names": names, "num_classes": len(names), "averaging": "macro", "execution": "experiment"},
              "provenance": {"checkpoint": str(Path(checkpoint_path).resolve()),
                             "checkpoint_sha256": sha256_file(checkpoint_path), "evaluation_dir": str(Path(evaluation_dir).resolve()),
                             "inner_validation_seed": config["seed"],
                             "validation_fraction": config["data"]["validation_fraction"]}}
    root = Path(output_dir)
    existing = collect_results(root)
    if existing and existing[0]["protocol"] != result["protocol"]:
        raise ValueError("FD-LLM artifacts do not match the baseline data/test protocol")
    target = root / "fd_llm" / f'seed_{result["seed"]}'
    target.mkdir(parents=True, exist_ok=True)
    (target / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return write_comparison(collect_results(root), root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(project_path("outputs/baselines")))
    parser.add_argument("--fd-checkpoint")
    parser.add_argument("--fd-evaluation")
    args = parser.parse_args()
    if bool(args.fd_checkpoint) != bool(args.fd_evaluation):
        parser.error("Supply both --fd-checkpoint and --fd-evaluation")
    if args.fd_checkpoint:
        import_fd_result(args.fd_checkpoint, args.fd_evaluation, args.root)
    else:
        write_comparison(collect_results(args.root), args.root)


if __name__ == "__main__":
    main()

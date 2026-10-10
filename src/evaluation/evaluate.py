"""Evaluate stage-1 matching or stage-2 diagnosis on untouched CWRU test data."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.alignment.contrastive_loss import BidirectionalContrastiveLoss
from src.alignment.encoders import LSTMDataEncoder
from src.alignment.train_alignment import AlignmentModel
from src.diagnosis.train_diagnosis import load_diagnosis
from src.evaluation.metrics import classification_metrics
from src.evaluation.visualization import plot_confusion_matrix, plot_training_curves, plot_tsne
from src.utils.data import CWRUDataset, description_fingerprint, load_descriptions
from src.utils.runtime import DEFAULT_CONFIG, load_config, project_path, seed_everything, select_device
from src.utils.training import read_checkpoint


@torch.no_grad()
def evaluate_model(model, loader, device, class_names, *, candidate_chunk_size=4):
    model.eval()
    truth, predicted, features, scores = [], [], [], []
    for signals, labels in loader:
        output = model.predict(signals.to(device), candidate_chunk_size=candidate_chunk_size)
        truth.append(labels.cpu().numpy())
        predicted.append(output["class_ids"].cpu().numpy())
        features.append(output["features"].float().cpu().numpy())
        scores.append(output["scores"].float().cpu().numpy())
    if not truth:
        raise ValueError("Evaluation loader is empty")
    arrays = {"y_true": np.concatenate(truth), "y_pred": np.concatenate(predicted),
              "features": np.concatenate(features), "scores": np.concatenate(scores)}
    return classification_metrics(arrays["y_true"], arrays["y_pred"], len(class_names), class_names), arrays


def save_evaluation(metrics, arrays, output_dir, config, history=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 保存逐样本预测和特征，方便复核指标、重画图及汇总模型对比。
    np.savez_compressed(output_dir / "predictions.npz", **arrays)
    names = metrics["class_names"]
    plot_confusion_matrix(metrics["confusion_matrix"], names, output_dir / "confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], names, output_dir / "confusion_matrix_normalized.png", True)
    if len(arrays["features"]) >= 3:
        plot_tsne(arrays["features"], arrays["y_true"], names, output_dir / "tsne.png",
                  perplexity=config["evaluation"]["tsne_perplexity"],
                  max_samples=config["evaluation"]["tsne_max_samples"], seed=config["seed"])
    if history:
        plot_training_curves(history, output_dir / "curves.png")


def run(config, checkpoint_path, stage="diagnosis"):
    seed_everything(config["seed"])
    device = select_device(config["device"])
    checkpoint = read_checkpoint(checkpoint_path)
    if checkpoint["stage"] != stage:
        raise ValueError("Checkpoint stage differs from requested evaluation stage")
    names, _, labels = load_descriptions(config["data"]["descriptions_path"], config["data"]["num_classes"])
    if stage == "diagnosis":
        model = load_diagnosis(config, checkpoint_path, device)
        if model.class_names != names:
            raise ValueError("Checkpoint classes differ from the current descriptions")
    elif stage == "alignment":
        if checkpoint["signature"]["descriptions_sha256"] != description_fingerprint(config["data"]["descriptions_path"]):
            raise ValueError("Descriptions changed after the alignment checkpoint was created")
        for key in ("data_encoder", "text_encoder", "model"):
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"Checkpoint {key} configuration differs")
        state = checkpoint["model"]
        if not torch.equal(state["text_labels"], labels):
            raise ValueError("Checkpoint class text labels differ")
        model = AlignmentModel(LSTMDataEncoder(**config["data_encoder"]), state["text_features"],
                               state["text_labels"], BidirectionalContrastiveLoss(**config["contrastive"]))
        model.load_state_dict(state)
        model.to(device)
    else:
        raise ValueError("stage must be alignment or diagnosis")
    # 保持测试集原始顺序，保证导出的预测可与其他模型逐样本对应。
    loader = DataLoader(CWRUDataset(project_path(config["data"]["test_path"])),
                        batch_size=config["evaluation"]["batch_size"], shuffle=False)
    metrics, arrays = evaluate_model(model, loader, device, names,
                                     candidate_chunk_size=config["evaluation"]["candidate_chunk_size"])
    save_evaluation(metrics, arrays, project_path(config["evaluation"]["output_dir"]), config, checkpoint["history"])
    print(json.dumps({key: metrics[key] for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")}), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", choices=("alignment", "diagnosis"), default="diagnosis")
    args = parser.parse_args()
    run(load_config(args.config), args.checkpoint, args.stage)


if __name__ == "__main__":
    main()

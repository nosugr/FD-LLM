"""Publication-exportable confusion matrices, training curves and t-SNE."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.utils.visualization.training import plot_training_curves


def plot_confusion_matrix(matrix, class_names, path, normalize=False):
    matrix = np.asarray(matrix)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError("Confusion matrix shape differs from class names")
    values = matrix.astype(float)
    if normalize:
        # 按真实类别所在行归一化；没有样本的行保留为零，避免除零。
        values = np.divide(values, values.sum(axis=1, keepdims=True),
                           out=np.zeros_like(values), where=values.sum(axis=1, keepdims=True) != 0)
    figure, axis = plt.subplots(figsize=(max(7, len(class_names) * 0.6), max(6, len(class_names) * 0.55)))
    image = axis.imshow(values, cmap="Blues", vmin=0)
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            axis.text(column, row, f"{values[row, column]:.2f}" if normalize else str(int(values[row, column])),
                      ha="center", va="center", fontsize=7,
                      color="white" if values[row, column] > values.max() / 2 else "black")
    axis.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
             xticklabels=class_names, yticklabels=class_names, xlabel="Predicted class", ylabel="True class")
    plt.setp(axis.get_xticklabels(), rotation=65, ha="right")
    axis.tick_params(labelsize=8)
    figure.colorbar(image, ax=axis, fraction=0.04)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_tsne(features, labels, class_names, path, *, perplexity=30, max_samples=1000, seed=42):
    features, labels = np.asarray(features), np.asarray(labels)
    if features.ndim != 2 or labels.shape != (len(features),) or len(features) < 3:
        raise ValueError("t-SNE requires at least 3 labeled feature vectors")
    if not np.isfinite(features).all() or max_samples < 3 or perplexity <= 0:
        raise ValueError("Invalid t-SNE input or settings")
    if np.any(labels < 0) or np.any(labels >= len(class_names)):
        raise ValueError("t-SNE class labels are out of range")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(features))
    if len(features) > max_samples:
        indices = np.sort(rng.choice(len(features), max_samples, replace=False))
    sampled = features[indices].astype(np.float32)
    # 高维特征先经 PCA 压缩，降低 CPU 上执行 t-SNE 的开销。
    dimensions = min(50, sampled.shape[1], len(sampled) - 1)
    if sampled.shape[1] > dimensions:
        sampled = PCA(n_components=dimensions, random_state=seed).fit_transform(sampled)
    embedding = TSNE(n_components=2, perplexity=min(float(perplexity), len(sampled) - 1),
                     init="random", learning_rate="auto", random_state=seed,
                     max_iter=500).fit_transform(sampled)
    figure, axis = plt.subplots(figsize=(9, 7))
    colors = plt.get_cmap("tab20")
    for class_id in np.unique(labels[indices]):
        selected = labels[indices] == class_id
        axis.scatter(embedding[selected, 0], embedding[selected, 1], s=12,
                     color=colors(int(class_id) % 20), label=class_names[int(class_id)], alpha=0.8)
    axis.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="Data encoder features (t-SNE)")
    axis.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    # 导出坐标及原始样本索引，使随机子采样和可视化结果可追溯。
    np.savez_compressed(Path(path).with_suffix(".npz"), coordinates=embedding, sample_indices=indices,
                        labels=labels[indices])
    return embedding, indices


__all__ = ["plot_confusion_matrix", "plot_training_curves", "plot_tsne"]

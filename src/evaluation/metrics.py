"""Closed-set metrics with an explicit, stable CWRU class order."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true, y_pred, num_classes=19, class_names=None):
    truth, predicted = np.asarray(y_true), np.asarray(y_pred)
    if truth.ndim != 1 or truth.shape != predicted.shape or len(truth) == 0:
        raise ValueError("Nonempty, matching one-dimensional labels are required")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    for values in (truth, predicted):
        if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values >= num_classes):
            raise ValueError("Labels must be integer class IDs in range")
    names = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]
    if len(names) != num_classes:
        raise ValueError("class_names must match num_classes")
    # 固定类别顺序；即使某类未出现在当前批次，仍保留对应指标及矩阵行列。
    labels = np.arange(num_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0)
    result = {"accuracy": float(accuracy_score(truth, predicted)), "sample_count": len(truth),
              "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
              "class_names": names,
              "per_class": [{"class_id": i, "class_name": names[i], "precision": float(precision[i]),
                             "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
                            for i in range(num_classes)]}
    # macro 对每类等权，weighted 按真实样本数加权，两种口径同时输出。
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(truth, predicted, labels=labels,
                                                   average=average, zero_division=0)
        result.update({f"precision_{average}": float(p), f"recall_{average}": float(r), f"f1_{average}": float(f)})
    return result

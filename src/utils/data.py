"""Read existing CWRU artifacts; reserve validation exclusively from training."""

import hashlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .runtime import project_path


def description_fingerprint(path):
    """Detect edited texts even when their file path and class IDs are unchanged."""
    return hashlib.sha256(project_path(path).read_bytes()).hexdigest()


class CWRUDataset(Dataset):
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            x, y = archive["X"], archive["y"]
        if x.ndim != 3 or x.shape[-1] != 1 or y.shape != (len(x),):
            raise ValueError("Expected X=(N,L,1), y=(N,) in processed NPZ")
        if not np.isfinite(x).all() or not np.issubdtype(y.dtype, np.integer):
            raise ValueError("Dataset contains invalid signals or labels")
        if len(x) == 0 or np.any(y < 0):
            raise ValueError("Dataset must be nonempty with nonnegative labels")
        self.signals = torch.from_numpy(x.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.signals[index], self.labels[index]


def load_descriptions(path, num_classes):
    document = json.loads(project_path(path).read_text(encoding="utf-8"))
    records = sorted(document["classes"], key=lambda item: item["class_id"])
    if [item["class_id"] for item in records] != list(range(num_classes)):
        raise ValueError("Descriptions must match contiguous class IDs 0..num_classes-1")
    names, texts, labels = [], [], []
    for item in records:
        names.append(item["class_name"])
        # 展开同类的全部描述变体，并为每条文本保留对应类别标签。
        variants = item.get("descriptions") or [{"text": item["description"]}]
        for variant in variants:
            if not variant["text"].strip():
                raise ValueError("Class descriptions must not be empty")
            texts.append(variant["text"])
            labels.append(item["class_id"])
    return names, texts, torch.tensor(labels, dtype=torch.long)


def stratified_split(labels, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    # 在训练集内部按类别划分，保证两侧均有样本；测试集不参与调参。
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < 2:
            raise ValueError("Each class needs at least two training samples for validation")
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        validation.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    return train, validation


def seed_worker(worker_id):
    import random
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loaders(config, stage):
    data = config["data"]
    dataset = CWRUDataset(project_path(data["train_path"]))
    encoder_config = config["data_encoder"]
    if dataset.signals.shape[1:] != (encoder_config["sequence_length"], encoder_config["input_size"]):
        raise ValueError("Processed signal shape differs from the data encoder configuration")
    if set(dataset.labels.tolist()) != set(range(data["num_classes"])):
        raise ValueError("Training labels disagree with configured class count")
    train, validation = stratified_split(dataset.labels.numpy(), data["validation_fraction"], config["seed"])
    options = {"batch_size": config[stage]["batch_size"],
               "num_workers": data.get("num_workers", 0), "worker_init_fn": seed_worker,
               "pin_memory": torch.cuda.is_available()}
    # 使用全局 Torch 随机状态，使按 epoch 保存的 checkpoint 能恢复打乱顺序。
    return (DataLoader(Subset(dataset, train), shuffle=True, **options),
            DataLoader(Subset(dataset, validation), shuffle=False, **options))


def class_prototypes(features, labels, num_classes):
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("Text bank and labels have incompatible shapes")
    if set(labels.tolist()) != set(range(num_classes)):
        raise ValueError("Text bank must cover every configured class")
    # 同类多条描述取均值，按类别 ID 顺序构成第二阶段的语义原型库。
    return torch.stack([features[labels == i].mean(dim=0) for i in range(num_classes)])

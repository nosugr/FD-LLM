"""Reuse CWRU dataset/split primitives without changing FD-LLM configuration."""

import hashlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset

from src.utils.data import CWRUDataset, load_descriptions, seed_worker, stratified_split
from src.utils.runtime import project_path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def indices_digest(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


class PairBatchSampler:
    """Keep every sample; merge a singleton tail into the preceding batch."""

    def __init__(self, sampler, batch_size):
        if batch_size < 2 or len(sampler) < 2:
            raise ValueError("Contrastive loaders need batch_size >= 2 and at least two samples")
        self.sampler, self.batch_size = sampler, batch_size

    def __iter__(self):
        pending, current = None, []
        for index in self.sampler:
            current.append(index)
            if len(current) == self.batch_size:
                if pending is not None:
                    yield pending
                pending, current = current, []
        if pending is not None:
            # 对比损失至少需要两个样本，将末尾单个样本并入上一批且不丢数据。
            yield pending + current if len(current) == 1 else pending
        if len(current) > 1:
            yield current

    def __len__(self):
        count, remainder = divmod(len(self.sampler), self.batch_size)
        return count + (remainder > 1)


def prepare_data(config):
    data = config["data"]
    train_path, test_path = project_path(data["train_path"]), project_path(data["test_path"])
    training, test = CWRUDataset(train_path), CWRUDataset(test_path)
    expected = (data["sequence_length"], data["input_size"])
    for dataset in (training, test):
        if dataset.signals.shape[1:] != expected or set(dataset.labels.tolist()) != set(range(data["num_classes"])):
            raise ValueError("Processed input dimensions/classes differ from baseline configuration")
    names, _, _ = load_descriptions(data["descriptions_path"], data["num_classes"])
    chosen_train, chosen_test = list(range(len(training))), list(range(len(test)))
    if config.get("execution") == "smoke":
        def choose(dataset, limit):
            result = []
            for class_id in range(data["num_classes"]):
                indices = np.flatnonzero(dataset.labels.numpy() == class_id)
                if len(indices) < limit:
                    raise ValueError("Insufficient real samples for the requested smoke subset")
                result.extend(indices[:limit].tolist())
            return result
        chosen_train = choose(training, config["smoke"]["train_per_class"])
        chosen_test = choose(test, config["smoke"]["test_per_class"])
    # 划分种子独立于训练种子，所有模型及重复实验共用同一验证集。
    inner_train, inner_validation = stratified_split(training.labels[chosen_train].numpy(),
                                                    data["validation_fraction"], config["split_seed"])
    train_indices = [chosen_train[i] for i in inner_train]
    validation_indices = [chosen_train[i] for i in inner_validation]
    # 记录数据、测试子集和类别顺序，汇总时只接受实验条件一致的结果。
    protocol = {"train_sha256": sha256_file(train_path), "test_sha256": sha256_file(test_path),
                "test_indices_sha256": indices_digest(chosen_test), "test_count": len(chosen_test),
                "class_names": names, "num_classes": data["num_classes"], "averaging": "macro",
                "execution": config.get("execution", "experiment")}
    split = {"split_seed": config["split_seed"], "validation_fraction": data["validation_fraction"],
             "train_indices_sha256": indices_digest(train_indices),
             "validation_indices_sha256": indices_digest(validation_indices)}
    return {"train": Subset(training, train_indices), "validation": Subset(training, validation_indices),
            "test": Subset(test, chosen_test), "class_names": names, "protocol": protocol, "split": split,
            "indices": {"train": train_indices, "validation": validation_indices, "test": chosen_test}}


def make_loaders(prepared, config, stage, generator, device, *, contrastive=False):
    options = {"num_workers": config["data"]["num_workers"], "worker_init_fn": seed_worker,
               "pin_memory": device.type == "cuda"}
    batch_size = config[stage]["batch_size"]
    if contrastive:
        train = DataLoader(prepared["train"], batch_sampler=PairBatchSampler(
            RandomSampler(prepared["train"], generator=generator), batch_size), generator=generator, **options)
        validation = DataLoader(prepared["validation"], batch_sampler=PairBatchSampler(
            SequentialSampler(prepared["validation"]), batch_size), **options)
    else:
        train = DataLoader(prepared["train"], batch_size=batch_size, shuffle=True, generator=generator, **options)
        validation = DataLoader(prepared["validation"], batch_size=batch_size, shuffle=False, **options)
    return train, validation

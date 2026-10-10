"""Baseline construction from YAML architecture dictionaries."""

from baselines.cnn import CNN
from baselines.resnet18 import ResNet18
from baselines.transformer import Transformer
from baselines.attention_lstm import AttentionLSTM
from baselines.attention_gru import AttentionGRU
from baselines.simclr import SimCLR

CLASSIFIERS = {"cnn": CNN, "resnet18": ResNet18, "transformer": Transformer,
               "attention_lstm": AttentionLSTM, "attention_gru": AttentionGRU}
MODEL_NAMES = (*CLASSIFIERS, "simclr")


def build_classifier(name, architecture, data):
    if name not in CLASSIFIERS:
        raise ValueError(f"Unknown supervised architecture: {name}")
    return CLASSIFIERS[name](**architecture, **{key: data[key] for key in
                                               ("input_size", "sequence_length", "num_classes")})


def build_simclr(architecture, data):
    # 编码器结构复用已有基线，SimCLR 自行添加独立的对比投影头。
    encoder = build_classifier(architecture["encoder"], architecture["encoder_kwargs"], data)
    return SimCLR(encoder, architecture["projection_hidden"], architecture["projection_dim"])

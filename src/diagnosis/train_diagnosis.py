"""Stage 2: load aligned embeddings, freeze encoders, optimize prompt and LoRA."""

import argparse

import torch

from src.alignment.encoders.data_encoder import LSTMDataEncoder
from src.diagnosis.llm import load_lora_backbone
from src.diagnosis.model import FDLLM
from src.utils.data import class_prototypes, description_fingerprint, load_descriptions, make_loaders
from src.utils.runtime import DEFAULT_CONFIG, load_config, project_path, seed_everything, select_device
from src.utils.training import Trainer, read_checkpoint


def build_from_alignment(config, device, *, training=True):
    checkpoint = read_checkpoint(project_path(config["diagnosis"]["alignment_checkpoint"]))
    if checkpoint["stage"] != "alignment":
        raise ValueError("An alignment checkpoint is required")
    for key in ("model", "data_encoder", "text_encoder", "data"):
        if checkpoint["config"][key] != config[key]:
            raise ValueError(f"Alignment checkpoint has a different {key} configuration")
    # 校验描述内容，防止修改文本后仍误用旧特征库。
    if checkpoint["signature"]["descriptions_sha256"] != description_fingerprint(config["data"]["descriptions_path"]):
        raise ValueError("Descriptions changed after the alignment text bank was created")
    state = checkpoint["model"]
    names, _, labels = load_descriptions(config["data"]["descriptions_path"], config["data"]["num_classes"])
    if not torch.equal(state["text_labels"], labels):
        raise ValueError("Alignment text bank differs from the descriptions")
    data_encoder = LSTMDataEncoder(**config["data_encoder"])
    encoder_state = {key.removeprefix("data_encoder."): value for key, value in state.items()
                     if key.startswith("data_encoder.")}
    data_encoder.load_state_dict(encoder_state)
    prototypes = class_prototypes(state["text_features"], labels, len(names))
    # 同类描述聚合为原型并复用第一阶段缓存，只加载诊断所需的语言模型。
    llm, tokenizer = load_lora_backbone(config["model"], config["lora"], device,
                                       gradient_checkpointing=training and config["diagnosis"]["gradient_checkpointing"])
    return FDLLM(data_encoder, llm, tokenizer, prototypes, names, config["prompt"]).to(device)


def load_diagnosis(config, checkpoint_path, device):
    """Inference is self-contained apart from the original base-model weights."""
    checkpoint = read_checkpoint(checkpoint_path)
    if checkpoint["stage"] != "diagnosis":
        raise ValueError("A diagnosis checkpoint is required")
    for key in ("model", "data_encoder", "text_encoder", "prompt", "lora"):
        if checkpoint["config"][key] != config[key]:
            raise ValueError(f"Diagnosis checkpoint has a different {key} configuration")
    state = checkpoint["model"]
    # 诊断 checkpoint 已含数据编码器和语义原型，推理无需再读取对齐 checkpoint。
    llm, tokenizer = load_lora_backbone(config["model"], config["lora"], device)
    model = FDLLM(LSTMDataEncoder(**config["data_encoder"]), llm, tokenizer,
                  state["fuzzy"]["text_features"], state["class_names"], config["prompt"]).to(device)
    model.load_checkpoint_state(state)
    return model.eval()


def run(config, resume=None):
    seed_everything(config["seed"])
    device = select_device(config["device"])
    train_loader, validation_loader = make_loaders(config, "diagnosis")
    if resume:
        model = load_diagnosis(config, resume, device)
        if config["diagnosis"]["gradient_checkpointing"]:
            model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model = build_from_alignment(config, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["diagnosis"]["learning_rate"],
                                 weight_decay=config["diagnosis"]["weight_decay"])
    trainer = Trainer(model, optimizer, device, project_path(config["diagnosis"]["output_dir"]), config, "diagnosis")
    return trainer.fit(train_loader, validation_loader, resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume")
    args = parser.parse_args()
    run(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()

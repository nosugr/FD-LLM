"""Stage 1: frozen text bank + trainable LSTM, symmetric contrastive alignment."""

import argparse

import torch
from torch import nn

from src.alignment.contrastive_loss import BidirectionalContrastiveLoss, cosine_logits
from src.alignment.encoders.data_encoder import LSTMDataEncoder
from src.alignment.encoders.text_encoder import TextEncoder, encode_text_bank
from src.utils.data import load_descriptions, make_loaders
from src.utils.runtime import DEFAULT_CONFIG, load_config, project_path, seed_everything, select_device
from src.utils.training import Trainer, read_checkpoint


class AlignmentModel(nn.Module):
    def __init__(self, data_encoder, text_features, text_labels, criterion):
        super().__init__()
        if text_features.shape[1] != data_encoder.embedding_dim:
            raise ValueError("LSTM and text encoder dimensions must agree")
        self.data_encoder, self.criterion = data_encoder, criterion
        # 文本特征作为 buffer 随模型迁移和保存，不参与梯度更新。
        self.register_buffer("text_features", text_features.detach().float())
        self.register_buffer("text_labels", text_labels.detach().long())

    def batch_metrics(self, signals, labels):
        features = self.data_encoder(signals)
        loss = self.criterion(features, self.text_features, labels, self.text_labels)
        # Top-1 按匹配文本的类别判断，同类任一描述匹配均算正确。
        predictions = self.text_labels[cosine_logits(features, self.text_features).argmax(dim=1)]
        return {"loss": loss, "top1": (predictions == labels).float().mean()}

    @torch.no_grad()
    def predict(self, signals, candidate_chunk_size=None):
        features = self.data_encoder(signals)
        logits = cosine_logits(features, self.text_features)
        return {"class_ids": self.text_labels[logits.argmax(dim=-1)],
                "scores": logits, "features": features}


def run(config, resume=None):
    seed_everything(config["seed"])
    device = select_device(config["device"])
    train_loader, validation_loader = make_loaders(config, "alignment")
    _, texts, labels = load_descriptions(config["data"]["descriptions_path"], config["data"]["num_classes"])
    if resume:
        # 续训直接复用 checkpoint 的文本库，避免重新加载并编码 Vicuna。
        checkpoint = read_checkpoint(resume)
        if checkpoint["stage"] != "alignment":
            raise ValueError("An alignment checkpoint is required")
        if not torch.equal(checkpoint["model"]["text_labels"], labels):
            raise ValueError("Checkpoint text labels differ from the descriptions")
        text_features = checkpoint["model"]["text_features"]
    else:
        encoder = TextEncoder.from_config(config["model"], config["text_encoder"], device)
        if encoder.embedding_dim != config["data_encoder"]["embedding_dim"]:
            raise ValueError("Configured embedding_dim differs from the real backbone hidden_size")
        text_features = encode_text_bank(encoder, texts, config["alignment"]["text_batch_size"])
        # 文本库已缓存并会保存到 checkpoint，释放 7B 模型后再训练 LSTM。
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    model = AlignmentModel(LSTMDataEncoder(**config["data_encoder"]), text_features, labels,
                           BidirectionalContrastiveLoss(**config["contrastive"])).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["alignment"]["learning_rate"],
                                 weight_decay=config["alignment"]["weight_decay"])
    trainer = Trainer(model, optimizer, device, project_path(config["alignment"]["output_dir"]), config, "alignment")
    return trainer.fit(train_loader, validation_loader, resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume")
    args = parser.parse_args()
    run(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()

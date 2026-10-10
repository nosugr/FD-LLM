"""Frozen Vicuna backbone features for FD-LLM Section 3.1."""

import torch
from torch import nn

from src.utils.huggingface import load_backbone


class TextEncoder(nn.Module):
    def __init__(self, model, tokenizer, *, pooling="mean", max_length=256):
        super().__init__()
        if pooling not in {"mean", "last"} or max_length < 1:
            raise ValueError("pooling must be mean/last and max_length must be positive")
        self.model, self.tokenizer = model, tokenizer
        self.pooling, self.max_length = pooling, max_length
        self.embedding_dim = model.config.hidden_size
        # 第一阶段仅训练数据编码器，文本模型参数始终冻结。
        self.model.requires_grad_(False)
        self.model.eval()

    @classmethod
    def from_config(cls, model_config, text_config, device):
        model, tokenizer = load_backbone(model_config, device)
        return cls(model, tokenizer, **text_config)

    def train(self, mode=True):
        super().train(mode)
        # 外层切换训练模式时，仍保持冻结模型的 dropout 关闭。
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, texts):
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain nonempty strings")
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(list(texts), padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()
                  if key in {"input_ids", "attention_mask"}}
        self.model.eval()
        output = self.model(**tokens, use_cache=False, return_dict=True)
        hidden = output.last_hidden_state.float()
        mask = tokens["attention_mask"]
        if (mask.sum(dim=1) == 0).any():
            raise ValueError("Tokenizer produced an empty sequence")
        if self.pooling == "mean":
            # 按掩码池化，避免填充 token 改变描述文本的特征。
            return (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        # 根据掩码定位最后一个有效 token，兼容左右两种填充方式。
        positions = torch.arange(mask.shape[1], device=device).expand_as(mask)
        last = positions.masked_fill(mask == 0, -1).max(dim=1).values
        return hidden[torch.arange(len(texts), device=device), last]


@torch.no_grad()
def encode_text_bank(encoder, texts, batch_size=8):
    if batch_size < 1 or not texts:
        raise ValueError("Nonempty texts and a positive batch_size are required")
    # 分批编码后缓存到 CPU，减少一次性加载全部描述的显存占用。
    return torch.cat([encoder(texts[start:start + batch_size]).cpu()
                      for start in range(0, len(texts), batch_size)])

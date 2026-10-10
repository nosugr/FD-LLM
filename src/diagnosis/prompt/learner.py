"""Mixed continuous/discrete prompt following the Section 3.2 template."""

import torch
from torch import nn


class PromptLearner(nn.Module):
    def __init__(self, embedding_dim, tokenizer, *, num_tokens=8, init_std=0.02,
                 human_prefix="### Human: <Data>", data_suffix="</Data>",
                 fuzzy_prefix="<Fuzzy semantic>",
                 assistant_suffix="</Fuzzy semantic> What is the state of this bearing? ### Assistant:",
                 answer_template=None, max_answer_tokens=96):
        super().__init__()
        if num_tokens < 1 or embedding_dim < 1 or init_std <= 0:
            raise ValueError("Prompt dimensions and init_std must be positive")
        self.continuous = nn.Parameter(torch.empty(num_tokens, embedding_dim))
        # 连续 Prompt 直接学习 embedding；离散文本仅保存固定 token ID。
        nn.init.normal_(self.continuous, std=init_std)
        for name, text in (("human", human_prefix), ("data_end", data_suffix),
                           ("fuzzy_start", fuzzy_prefix), ("assistant", assistant_suffix)):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if name == "human" and tokenizer.bos_token_id is not None:
                ids = [tokenizer.bos_token_id] + ids
            if not ids:
                raise ValueError("Discrete prompt segments cannot be empty")
            self.register_buffer(name + "_ids", torch.tensor(ids, dtype=torch.long))

    def forward(self, data_features, fuzzy_features, embedding_layer):
        if data_features.ndim != 2 or fuzzy_features.shape != data_features.shape:
            raise ValueError("Data and fuzzy features must have the same (batch, dim) shape")
        if data_features.shape[-1] != self.continuous.shape[-1]:
            raise ValueError("Prompt and feature dimensions differ")
        batch = len(data_features)
        dtype = embedding_layer.weight.dtype

        def discrete(name):
            return embedding_layer(getattr(self, name + "_ids")).unsqueeze(0).expand(batch, -1, -1)

        # 数据和模糊语义向量各占一个 token 位置，按论文模板顺序拼接。
        embeddings = torch.cat((discrete("human"), data_features[:, None].to(dtype),
                                discrete("data_end"), self.continuous[None].expand(batch, -1, -1).to(dtype),
                                discrete("fuzzy_start"), fuzzy_features[:, None].to(dtype),
                                discrete("assistant")), dim=1)
        return embeddings, torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long)

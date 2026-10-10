"""FD-LLM stage 2: frozen data/fuzzy encoders + prompt + LoRA causal LM."""

import torch
from torch import nn
from torch.nn import functional as F

from src.diagnosis.fuzzy import FuzzySemanticEmbedding
from src.diagnosis.prompt import PromptLearner


class FDLLM(nn.Module):
    def __init__(self, data_encoder, llm, tokenizer, class_text_features, class_names, prompt_config):
        super().__init__()
        dimension = llm.config.hidden_size
        if data_encoder.embedding_dim != dimension or class_text_features.shape != (len(class_names), dimension):
            raise ValueError("Data, text and LLM embedding dimensions must match")
        if len(set(class_names)) != len(class_names) or not class_names:
            raise ValueError("class_names must be unique and nonempty")
        if not any("lora_" in name and p.requires_grad for name, p in llm.named_parameters()):
            raise ValueError("FDLLM requires a causal LM with trainable LoRA adapters")
        # 第二阶段固定已对齐的数据特征，仅优化连续 Prompt 和 LoRA。
        self.data_encoder = data_encoder.requires_grad_(False)
        self.data_encoder.eval()
        self.llm, self.tokenizer = llm, tokenizer
        self.class_names = list(class_names)
        self.fuzzy = FuzzySemanticEmbedding(class_text_features)
        self.prompt = PromptLearner(dimension, tokenizer, **prompt_config)
        template = prompt_config.get("answer_template", " The bearing state is {class_name}.")
        self.answers = [template.format(class_name=name, class_id=i) for i, name in enumerate(class_names)]
        if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer requires EOS and padding IDs")
        ids = [tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
               for answer in self.answers]
        if max(map(len, ids)) > prompt_config.get("max_answer_tokens", 96):
            raise ValueError("A class answer exceeds max_answer_tokens; increase the configuration limit")
        if len({tuple(item) for item in ids}) != len(ids):
            raise ValueError("Different class answers tokenize identically; check tokenizer/template")
        answer_ids = torch.full((len(ids), max(map(len, ids))), tokenizer.pad_token_id, dtype=torch.long)
        # 单独保存有效位置掩码，即使 PAD 与 EOS 共用 ID，也不会屏蔽真实 EOS。
        answer_mask = torch.zeros_like(answer_ids)
        for index, tokens in enumerate(ids):
            answer_ids[index, :len(tokens)] = torch.tensor(tokens)
            answer_mask[index, :len(tokens)] = 1
        self.register_buffer("answer_ids", answer_ids)
        self.register_buffer("answer_mask", answer_mask)

    def train(self, mode=True):
        super().train(mode)
        self.data_encoder.eval()
        return self

    @torch.no_grad()
    def encode(self, signals):
        data = self.data_encoder(signals)
        fuzzy, membership = self.fuzzy(data)
        return data, fuzzy, membership

    def prefix(self, signals):
        data, fuzzy, membership = self.encode(signals)
        embeddings, mask = self.prompt(data, fuzzy, self.llm.get_input_embeddings())
        return embeddings, mask, data, membership

    def append_answers(self, prefix, prefix_mask, class_labels):
        if class_labels.shape != (len(prefix),) or (class_labels < 0).any() or (class_labels >= len(self.class_names)).any():
            raise ValueError("class_labels must contain valid class IDs for every sample")
        ids, mask = self.answer_ids[class_labels], self.answer_mask[class_labels]
        answers = self.llm.get_input_embeddings()(ids)
        embeddings = torch.cat((prefix, answers), dim=1)
        attention_mask = torch.cat((prefix_mask, mask), dim=1)
        # -100 忽略提示和填充位置，交叉熵只监督诊断答案（含 EOS）。
        labels = torch.cat((torch.full_like(prefix_mask, -100), ids.masked_fill(mask == 0, -100)), dim=1)
        limit = getattr(self.llm.config, "max_position_embeddings", None)
        if limit and embeddings.shape[1] > limit:
            raise ValueError("Combined prompt and answers exceed the backbone context limit")
        return {"inputs_embeds": embeddings, "attention_mask": attention_mask, "labels": labels}

    def forward(self, signals, class_labels):
        prefix, mask, _, _ = self.prefix(signals)
        inputs = self.append_answers(prefix, mask, class_labels)
        outputs = self.llm(**inputs, use_cache=False, return_dict=True)
        # 因果模型用当前位置预测下一个 token，统计准确率时需错位对齐。
        targets = inputs["labels"][:, 1:]
        valid = targets != -100
        correct = (outputs.logits[:, :-1].detach().argmax(dim=-1) == targets) & valid
        return {"loss": outputs.loss, "token_accuracy": correct.sum().float() / valid.sum(),
                "loss_weight": int(valid.sum())}

    def batch_metrics(self, signals, labels):
        result = self(signals, labels)
        if not self.training:
            prediction = self.predict(signals)
            result["accuracy"] = (prediction["class_ids"] == labels).float().mean()
        return result

    @torch.no_grad()
    def predict(self, signals, candidate_chunk_size=4):
        """Closed-set diagnosis by mean answer-token log likelihood, without labels.

        Enumerate canonical class answers using the SAME causal token objective
        as training. This is an implementation choice for deterministic 19-class
        evaluation, not a separate learned classification head.
        """
        if self.training:
            raise RuntimeError("Call eval() before predict() for deterministic inference")
        if candidate_chunk_size < 1:
            raise ValueError("candidate_chunk_size must be positive")
        prefix, mask, data, membership = self.prefix(signals)
        batch = len(signals)
        scores = []
        # 分块枚举标准类别答案，控制显存占用；推理时无需输入真实标签。
        for start in range(0, len(self.class_names), candidate_chunk_size):
            candidates = torch.arange(start, min(start + candidate_chunk_size, len(self.class_names)), device=signals.device)
            count = len(candidates)
            expanded = prefix[:, None].expand(-1, count, -1, -1).reshape(batch * count, *prefix.shape[1:])
            expanded_mask = mask[:, None].expand(-1, count, -1).reshape(batch * count, -1)
            inputs = self.append_answers(expanded, expanded_mask, candidates.repeat(batch))
            output = self.llm(**inputs, use_cache=False, return_dict=True)
            targets = inputs["labels"][:, 1:]
            token_loss = F.cross_entropy(output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
                                         targets.reshape(-1), reduction="none", ignore_index=-100).reshape_as(targets)
            # 按有效答案长度取平均对数似然，避免较短答案天然占优。
            score = -token_loss.sum(dim=1) / (targets != -100).sum(dim=1)
            scores.append(score.reshape(batch, count))
        scores = torch.cat(scores, dim=1)
        return {"class_ids": scores.argmax(dim=-1), "scores": scores,
                "memberships": membership, "features": data}

    @torch.no_grad()
    def generate(self, signals, max_new_tokens=96):
        """Generate free-text diagnostics; classification metrics use predict()."""
        if self.training:
            raise RuntimeError("Call eval() before generate()")
        prefix, mask, _, _ = self.prefix(signals)
        if prefix.shape[1] + max_new_tokens > self.llm.config.max_position_embeddings:
            raise ValueError("Generation request exceeds the context limit")
        ids = self.llm.generate(inputs_embeds=prefix, attention_mask=mask,
                                max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def checkpoint_state(self):
        from peft import get_peft_model_state_dict
        # 保存适配器、Prompt 和特征模块；冻结的 LLM 基座权重由配置路径提供。
        return {"adapter": get_peft_model_state_dict(self.llm, save_embedding_layers=False),
                "prompt": self.prompt.state_dict(), "data_encoder": self.data_encoder.state_dict(),
                "fuzzy": self.fuzzy.state_dict(), "class_names": self.class_names,
                "answers": self.answers, "answer_ids": self.answer_ids, "answer_mask": self.answer_mask}

    def load_checkpoint_state(self, state):
        from peft import set_peft_model_state_dict
        if state["class_names"] != self.class_names or state["answers"] != self.answers:
            raise ValueError("Checkpoint class order or answer templates differ")
        if not torch.equal(state["answer_ids"].cpu(), self.answer_ids.cpu()):
            raise ValueError("Checkpoint tokenizer differs from the current tokenizer")
        current = self.checkpoint_state()["adapter"]
        if current.keys() != state["adapter"].keys():
            raise ValueError("Checkpoint LoRA modules differ")
        set_peft_model_state_dict(self.llm, state["adapter"])
        self.prompt.load_state_dict(state["prompt"])
        self.data_encoder.load_state_dict(state["data_encoder"])
        self.fuzzy.load_state_dict(state["fuzzy"])

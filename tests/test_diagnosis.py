"""Stage 2 equations, token masks, real PEFT gradients and offline inference."""

import copy

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.alignment.encoders import LSTMDataEncoder
from src.diagnosis.fuzzy import FuzzySemanticEmbedding
from src.diagnosis.llm import attach_lora
from src.diagnosis.model import FDLLM
from src.diagnosis.train_diagnosis import load_diagnosis
from src.utils.training import Trainer, read_checkpoint
from tests.helpers import small_config, tiny_llama


def make_model(config=None, *, checkpointing=False):
    config = config or small_config()
    base, tokenizer = tiny_llama()
    llm = attach_lora(base, config["lora"], gradient_checkpointing=checkpointing)
    return FDLLM(LSTMDataEncoder(**config["data_encoder"]), llm, tokenizer,
                  torch.randn(3, 16), ["normal", "fault", "ball"], config["prompt"])


def test_fuzzy_matches_equations_and_uses_raw_text_features():
    text = torch.tensor([[2.0, 0.0], [0.0, 4.0]], requires_grad=True)
    data = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    fuzzy = FuzzySemanticEmbedding(text)
    result, membership = fuzzy(data)
    expected_weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).softmax(dim=-1)
    torch.testing.assert_close(membership, expected_weights)
    torch.testing.assert_close(membership.sum(-1), torch.ones(2))
    torch.testing.assert_close(result, expected_weights @ text.detach())
    result.sum().backward()
    assert data.grad is not None and text.grad is None
    assert list(fuzzy.parameters()) == []


@pytest.mark.parametrize("checkpointing", [False, True])
def test_diagnosis_forward_backward_and_freeze(checkpointing):
    model = make_model(checkpointing=checkpointing)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if not parameter.requires_grad}
    model.train()
    assert not model.data_encoder.training
    signals, labels = torch.randn(2, 16, 1), torch.tensor([0, 2])
    output = model(signals, labels)
    assert torch.isfinite(output["loss"]) and output["loss_weight"] > 0
    output["loss"].backward()
    assert model.prompt.continuous.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for name, p in model.named_parameters() if "lora_" in name)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert all("lora_" in name or name == "prompt.continuous" for name in trainable)
    for name, parameter in model.named_parameters():
        if name in before:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_prompt_and_answer_mask_including_eos_as_pad():
    model = make_model()
    # 模拟 Vicuna 复用 EOS 作填充，显式掩码必须保留真实答案末尾的 EOS。
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.answer_ids[model.answer_mask == 0] = model.tokenizer.eos_token_id
    prefix, mask, _, _ = model.prefix(torch.randn(2, 16, 1))
    labels = torch.tensor([0, 1])
    inputs = model.append_answers(prefix, mask, labels)
    length = prefix.shape[1]
    assert prefix.shape[2] == 16
    assert (inputs["labels"][:, :length] == -100).all()
    for index, label in enumerate(labels):
        count = int(model.answer_mask[label].sum())
        assert inputs["labels"][index, length + count - 1] == model.tokenizer.eos_token_id
        assert (inputs["labels"][index, length + count:] == -100).all()
        assert inputs["attention_mask"][index, length + count - 1] == 1


def test_prediction_chunk_equivalence_and_generation():
    model = make_model().eval()
    signals = torch.randn(2, 16, 1)
    first = model.predict(signals, candidate_chunk_size=1)
    second = model.predict(signals, candidate_chunk_size=3)
    assert first["scores"].shape == (2, 3)
    assert first["class_ids"].shape == (2,)
    torch.testing.assert_close(first["scores"], second["scores"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(first["memberships"].sum(-1), torch.ones(2))
    answer = model.generate(signals[:1], max_new_tokens=3)
    assert len(answer) == 1 and isinstance(answer[0], str)
    # 验证类别分数等于答案 token 交叉熵均值的相反数，与训练目标一致。
    prefix, mask, _, _ = model.prefix(signals[:1])
    inputs = model.append_answers(prefix, mask, torch.tensor([0]))
    reference = model.llm(**inputs, use_cache=False).loss
    torch.testing.assert_close(first["scores"][0, 0], -reference, atol=1e-5, rtol=1e-5)


def test_prompt_checkpoint_round_trip_excludes_frozen_llm(tmp_path):
    model = make_model().eval()
    signals = torch.randn(2, 16, 1)
    output = model.predict(signals)
    state = copy.deepcopy(model.checkpoint_state())
    assert all("lora_" in name for name in state["adapter"])
    assert "llm" not in state and "embed_tokens.weight" not in state["adapter"]
    with torch.no_grad():
        model.prompt.continuous.add_(1)
        for name, parameter in model.llm.named_parameters():
            if "lora_" in name:
                parameter.add_(0.1)
    model.load_checkpoint_state(state)
    torch.testing.assert_close(model.predict(signals)["scores"], output["scores"])


def test_diagnosis_trainer_and_standalone_real_loader(tmp_path):
    config = small_config()
    base, tokenizer = tiny_llama()
    source = tmp_path / "base"
    base.save_pretrained(source)
    tokenizer.save_pretrained(source)
    config["model"].update(name_or_path=str(source), dtype="float32")
    names = ["normal", "fault", "ball"]
    model = FDLLM(LSTMDataEncoder(**config["data_encoder"]), attach_lora(base, config["lora"]),
                  tokenizer, torch.randn(3, 16), names, config["prompt"])
    loader = DataLoader(TensorDataset(torch.randn(4, 16, 1), torch.tensor([0, 1, 2, 0])), batch_size=2)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    trainer = Trainer(model, optimizer, torch.device("cpu"), tmp_path / "run", config, "diagnosis")
    trainer.fit(loader, loader)
    checkpoint = read_checkpoint(tmp_path / "run/last.pt")
    assert "adapter" in checkpoint["model"] and len(checkpoint["history"]) == 1
    restored = load_diagnosis(config, tmp_path / "run/last.pt", torch.device("cpu"))
    signals = torch.randn(2, 16, 1)
    torch.testing.assert_close(model.eval().predict(signals)["scores"], restored.predict(signals)["scores"])
    config["diagnosis"]["epochs"] = 2
    resumed = Trainer(restored, torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=0.001),
                       torch.device("cpu"), tmp_path / "resumed", config, "diagnosis")
    resumed.fit(loader, loader, resume=tmp_path / "run/last.pt")
    assert len(resumed.history) == 2


def test_invalid_feature_dimension_and_class_labels():
    model = make_model()
    with pytest.raises(ValueError, match="valid class"):
        model(torch.randn(2, 16, 1), torch.tensor([0, 3]))
    with pytest.raises(RuntimeError, match="eval"):
        model.predict(torch.randn(1, 16, 1))
    with pytest.raises(ValueError):
        FuzzySemanticEmbedding(torch.randn(16))

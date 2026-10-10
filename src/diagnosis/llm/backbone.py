"""Vicuna causal LM with PEFT LoRA; production always loads configured weights."""

from src.utils.huggingface import load_backbone


def attach_lora(model, config, *, gradient_checkpointing=False):
    from peft import LoraConfig, TaskType, get_peft_model

    # 冻结基座后注入 LoRA，仅新增的低秩适配参数参与微调。
    model.requires_grad_(False)
    model.config.use_cache = False
    result = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, bias="none", **config))
    if gradient_checkpointing:
        # 非重入梯度检查点保留 inputs_embeds 的梯度，使连续 Prompt 能正常训练。
        result.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return result


def load_lora_backbone(model_config, lora_config, device, *, gradient_checkpointing=False):
    model, tokenizer = load_backbone(model_config, device, causal=True)
    return attach_lora(model, lora_config, gradient_checkpointing=gradient_checkpointing), tokenizer

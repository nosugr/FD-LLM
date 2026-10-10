"""Real Hugging Face loaders. No test models or fallback weights are used here."""

from .runtime import project_path, select_dtype


def model_source(config):
    source = config["name_or_path"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError("model.name_or_path must be a model ID or local directory")
    local = project_path(source)
    return str(local) if local.exists() else source


def load_backbone(config, device, *, causal=False):
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    source = model_source(config)
    options = {"local_files_only": config.get("local_files_only", True),
               "revision": config.get("revision", "main"), "trust_remote_code": False}
    # 正式入口必须读取真实权重，缺失时直接报错，不回退到随机模型。
    tokenizer = AutoTokenizer.from_pretrained(source, **options)
    if tokenizer.eos_token_id is None:
        raise ValueError("A causal tokenizer with an EOS token is required")
    if tokenizer.pad_token_id is None:
        # 无独立 PAD 时复用 EOS，后续靠 attention_mask 区分有效 token。
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    loader = AutoModelForCausalLM if causal else AutoModel
    model = loader.from_pretrained(source, torch_dtype=select_dtype(config.get("dtype", "auto"), device),
                                   attn_implementation="eager", **options).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

"""Random tiny Llama fixtures using real Transformers APIs, with no downloads."""

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def tiny_llama(hidden_size=16):
    # 仅供测试：本地随机初始化微型 Llama 和词表，不访问模型下载服务。
    words = ["<pad>", "<bos>", "<eos>", "<unk>", "normal", "fault", "bearing",
             "Normal", "Fan", "Drive", "end", "inner", "outer", "race", "rolling",
             "element", "ball", "in", "0", "007", "014", "021", "028", "state",
             "The", "is", ".", "healthy"] + [f"class{i}" for i in range(19)]
    tokenizer_backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend,
                                        pad_token="<pad>", bos_token="<bos>",
                                        eos_token="<eos>", unk_token="<unk>")
    config = LlamaConfig(vocab_size=len(words), hidden_size=hidden_size,
                         intermediate_size=hidden_size * 2, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=512, pad_token_id=0,
                         bos_token_id=1, eos_token_id=2, attention_dropout=0.0)
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config), tokenizer


def small_config():
    from src.utils.runtime import load_config
    config = load_config()
    config["device"] = "cpu"
    config["data_encoder"].update(hidden_size=8, embedding_dim=16, sequence_length=16)
    config["alignment"].update(epochs=1, batch_size=4)
    config["diagnosis"].update(epochs=1, batch_size=2, gradient_accumulation_steps=1,
                               gradient_checkpointing=False)
    config["prompt"].update(num_tokens=3)
    config["lora"].update(r=2, lora_alpha=4, lora_dropout=0.0)
    return config

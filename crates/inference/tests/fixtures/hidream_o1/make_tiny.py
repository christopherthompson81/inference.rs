#!/usr/bin/env python3
"""Writes config.json and tokenizer.json: a tiny HiDream-O1 checkpoint skeleton (no weights) for engine-behavior tests.

The tokenizer is the tiny Qwen-VL one (../qwen_vl/qwen3_vl) with HiDream's image and timestep tokens added; the config
is a Qwen3-VL text stack shrunk to two layers, with that fixture's two-layer vision tower for reference images. The
tests record random weights from what the constructor asks for.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).parent
tokenizer = json.loads((ROOT.parent / "qwen_vl" / "qwen3_vl" / "tokenizer.json").read_text())
vocab_size = max(t["id"] for t in tokenizer["added_tokens"]) + 1
for content in ["<|boi_token|>", "<|tms_token|>"]:
    tokenizer["added_tokens"].append(
        {"id": vocab_size, "content": content, "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": False, "special": True}
    )
    vocab_size += 1


def tokenizer_id(content):
    return next(t["id"] for t in tokenizer["added_tokens"] if t["content"] == content)


# head_dim 64, as the flash kernels take; the MRoPE sections sum to 32
config = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "model_type": "qwen3_vl",
    "tie_word_embeddings": False,
    "image_token_id": tokenizer_id("<|image_pad|>"),
    "video_token_id": tokenizer_id("<|video_pad|>"),
    "vision_start_token_id": tokenizer_id("<|vision_start|>"),
    "vision_end_token_id": tokenizer_id("<|vision_end|>"),
    "vision_config": json.loads((ROOT.parent / "qwen_vl" / "qwen3_vl" / "config.json").read_text())["vision_config"],
    "text_config": {
        "vocab_size": vocab_size, "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64, "hidden_act": "silu",
        "max_position_embeddings": 8192, "rms_norm_eps": 1e-06, "rope_theta": 5000000.0,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [8, 12, 12], "rope_type": "default"},
    },
}
(ROOT / "tokenizer.json").write_text(json.dumps(tokenizer, indent=2) + "\n")
(ROOT / "config.json").write_text(json.dumps(config, indent=2) + "\n")

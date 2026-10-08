#!/usr/bin/env python3
"""Writes qwen2_vl/, qwen2_5_vl/, qwen3_vl/, qwen3_vl_moe/ and qwen3_5_moe/: tiny Qwen-VL checkpoint skeletons (no weights) for engine-behavior tests.

The tokenizer is a byte-fallback BPE with only the special tokens the processors and chat template use; the configs
shrink every dimension and the preprocessor configs keep images to a handful of patches. The tests generate random
weights by recording what the model constructor asks for. The chat template is written here, in the Qwen format.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).parent
SPECIALS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|placeholder|>",
]

vocab = {"<unk>": 0}
for b in range(256):
    vocab[f"<0x{b:02X}>"] = len(vocab)
ids = {tok: len(vocab) + i for i, tok in enumerate(SPECIALS)}
vocab_size = len(vocab) + len(SPECIALS)

tokenizer = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
        for t, i in [("<unk>", 0)] + list(ids.items())
    ],
    "normalizer": None,
    "pre_tokenizer": None,
    "post_processor": None,
    "decoder": {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]},
    "model": {
        "type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None, "end_of_word_suffix": None,
        "fuse_unk": True, "byte_fallback": True, "ignore_merges": False, "vocab": vocab, "merges": [],
    },
}
tokenizer_config = {
    "add_bos_token": False, "add_eos_token": False, "bos_token": None, "eos_token": "<|im_end|>",
    "pad_token": "<|endoftext|>", "unk_token": "<unk>", "additional_special_tokens": SPECIALS[1:],
    "clean_up_tokenization_spaces": False, "model_max_length": 4096, "tokenizer_class": "Qwen2Tokenizer",
}
chat_template = """{%- for message in messages -%}
    {{- '<|im_start|>' + message['role'] + '\\n' -}}
    {%- if message['content'] is string -%}
        {{- message['content'] -}}
    {%- else -%}
        {%- for content in message['content'] -%}
            {%- if content['type'] == 'image' -%}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' -}}
            {%- elif content['type'] == 'video' -%}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' -}}
            {%- elif content['type'] == 'text' -%}
                {{- content['text'] -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {{- '<|im_end|>\\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\\n' -}}
{%- endif -%}
"""
# head_dim 64, the smallest the CUDA and Metal paged attention kernels take; each MRoPE section list sums to 32.
text = {
    "vocab_size": vocab_size, "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2,
    "num_attention_heads": 2, "num_key_value_heads": 1, "hidden_act": "silu", "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-06, "rope_theta": 10000.0, "sliding_window": None, "tie_word_embeddings": False,
}
token_ids = {
    "image_token_id": ids["<|image_pad|>"], "video_token_id": ids["<|video_pad|>"],
    "vision_start_token_id": ids["<|vision_start|>"], "vision_end_token_id": ids["<|vision_end|>"],
    "eos_token_id": ids["<|im_end|>"],
}


def image_processor(patch_size, kind):
    # min/max pixels hold an image to a few merged patches
    side = patch_size * 2
    return {
        "do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "rescale_factor": 1 / 255, "resample": 3,
        "patch_size": patch_size, "merge_size": 2, "temporal_patch_size": 2,
        "min_pixels": side * side, "max_pixels": side * side * 4, "image_processor_type": kind,
    }


models = {
    "qwen2_vl": {
        "config": {
            "architectures": ["Qwen2VLForConditionalGeneration"], "model_type": "qwen2_vl", **text, **token_ids,
            "rope_scaling": {"type": "mrope", "mrope_section": [8, 12, 12]}, "quantization_config": None,
            "vision_config": {
                "depth": 2, "embed_dim": 32, "hidden_size": 128, "hidden_act": "quick_gelu", "mlp_ratio": 2.0,
                "num_heads": 2, "in_channels": 3, "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2,
            },
        },
        "preprocessor": image_processor(14, "Qwen2VLImageProcessor"),
        "video_preprocessor": None,
    },
    "qwen3_vl": {
        "config": {
            "architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl", **token_ids,
            "tie_word_embeddings": False, "quantization_config": None,
            "text_config": {**text, "head_dim": 64, "rope_scaling": {"mrope_section": [8, 12, 12]}},
            "vision_config": {
                "depth": 2, "hidden_size": 32, "out_hidden_size": 128, "hidden_act": "gelu_pytorch_tanh",
                "intermediate_size": 64, "num_heads": 2, "in_channels": 3, "patch_size": 16, "spatial_merge_size": 2,
                "temporal_patch_size": 2, "num_position_embeddings": 64, "deepstack_visual_indexes": [0],
            },
        },
        "preprocessor": image_processor(16, "Qwen2VLImageProcessorFast"),
        "video_preprocessor": {**image_processor(16, "Qwen3VLVideoProcessor"), "fps": 2.0},
    },
}

# Qwen2.5-VL: window attention in the vision blocks (a 56-pixel window is 2x2 merged patches) and one full block.
models["qwen2_5_vl"] = {
    "config": {
        **models["qwen2_vl"]["config"], "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "model_type": "qwen2_5_vl",
        "vision_config": {
            "depth": 2, "hidden_size": 32, "out_hidden_size": 128, "hidden_act": "silu", "intermediate_size": 64,
            "num_heads": 2, "in_chans": 3, "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2,
            "window_size": 56, "fullatt_block_indexes": [1], "tokens_per_second": 2,
        },
    },
    "preprocessor": image_processor(14, "Qwen2VLImageProcessor"),
    "video_preprocessor": None,
}
# Qwen3-VL-MoE: the Qwen3-VL text stack with four experts in every layer.
models["qwen3_vl_moe"] = {
    "config": {
        **models["qwen3_vl"]["config"], "architectures": ["Qwen3VLMoeForConditionalGeneration"],
        "model_type": "qwen3_vl_moe",
        "text_config": {
            **models["qwen3_vl"]["config"]["text_config"], "num_experts": 4, "num_experts_per_tok": 2,
            "moe_intermediate_size": 64, "decoder_sparse_step": 1, "mlp_only_layers": [], "norm_topk_prob": True,
        },
    },
    "preprocessor": models["qwen3_vl"]["preprocessor"],
    "video_preprocessor": models["qwen3_vl"]["video_preprocessor"],
}

# Qwen3.5-MoE: three linear-attention (GDN) layers then one full-attention layer, four experts per layer.
QWEN3_5_MOE_EXPERTS = 4
qwen3_5_text = {
    "head_dim": 64, "vocab_size": vocab_size, "hidden_size": 128, "num_hidden_layers": 4,
    "num_attention_heads": 2, "num_key_value_heads": 1, "hidden_act": "silu", "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-06, "tie_word_embeddings": False,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.25,
                        "mrope_section": [2, 3, 3]},
    "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_num_key_heads": 2, "linear_num_value_heads": 2,
    "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64, "num_experts": QWEN3_5_MOE_EXPERTS,
    "num_experts_per_tok": 2, "mtp_num_hidden_layers": 1,
}
models["qwen3_5_moe"] = {
    "config": {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe", **token_ids,
        "tie_word_embeddings": False, "quantization_config": None, "text_config": qwen3_5_text,
        "vision_config": models["qwen3_vl"]["config"]["vision_config"],
    },
    "preprocessor": models["qwen3_vl"]["preprocessor"],
    "video_preprocessor": models["qwen3_vl"]["video_preprocessor"],
}

for name, files in models.items():
    out = ROOT / name
    out.mkdir(exist_ok=True)
    written = [
        ("tokenizer.json", tokenizer),
        ("tokenizer_config.json", tokenizer_config),
        ("config.json", files["config"]),
        ("preprocessor_config.json", files["preprocessor"]),
        ("generation_config.json", {"eos_token_id": ids["<|im_end|>"], "pad_token_id": ids["<|endoftext|>"]}),
    ]
    if files["video_preprocessor"]:
        written.append(("video_preprocessor_config.json", files["video_preprocessor"]))
    for file, obj in written:
        (out / file).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    (out / "chat_template.jinja").write_text(chat_template)

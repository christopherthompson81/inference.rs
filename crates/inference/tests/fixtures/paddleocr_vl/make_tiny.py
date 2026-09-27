#!/usr/bin/env python3
"""Writes tiny/: a PaddleOCR-VL checkpoint skeleton (no weights) for engine-behavior tests.

The tokenizer is a byte-fallback BPE with only the special tokens the processor and chat template use; the config
shrinks every dimension. The test generates random weights by recording what the model constructor asks for.
chat_template.jinja is copied from PaddlePaddle/PaddleOCR-VL (Apache-2.0).
"""
import json
import pathlib

OUT = pathlib.Path(__file__).parent / "tiny"
SPECIALS = [
    "<|begin_of_sentence|>",
    "<|end_of_sentence|>",
    "<|IMAGE_PLACEHOLDER|>",
    "<|image_pad|>",
    "<|IMAGE_START|>",
    "<|IMAGE_END|>",
    "<|video_pad|>",
]

vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
for b in range(256):
    vocab[f"<0x{b:02X}>"] = len(vocab)
ids = {tok: len(vocab) + i for i, tok in enumerate(SPECIALS)}
vocab_size = len(vocab) + len(SPECIALS)

added = [
    {"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
    for t, i in [("<unk>", 0), ("<s>", 1), ("</s>", 2)] + list(ids.items())
]
tokenizer = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": added,
    "normalizer": {"type": "Sequence", "normalizers": [{"type": "Replace", "pattern": {"String": " "}, "content": "\u2581"}]},
    "pre_tokenizer": None,
    "post_processor": None,
    "decoder": {
        "type": "Sequence",
        "decoders": [{"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}],
    },
    "model": {
        "type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None, "end_of_word_suffix": None,
        "fuse_unk": True, "byte_fallback": True, "ignore_merges": False, "vocab": vocab, "merges": [],
    },
}
tokenizer_config = {
    "add_bos_token": False, "add_eos_token": False, "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>",
    "pad_token": "<unk>", "cls_token": "<|begin_of_sentence|>", "sep_token": "<|end_of_sentence|>",
    "image_token": "<|IMAGE_PLACEHOLDER|>", "additional_special_tokens": SPECIALS[2:], "clean_up_tokenization_spaces": False,
    "model_max_length": 4096, "tokenizer_class": "LlamaTokenizer", "processor_class": "PaddleOCRVLProcessor",
}
config = {
    "architectures": ["PaddleOCRVLForConditionalGeneration"], "model_type": "paddleocr_vl", "compression_ratio": 1.0,
    "head_dim": 64, "hidden_act": "silu", "hidden_size": 32, "intermediate_size": 64, "max_position_embeddings": 4096,
    "num_attention_heads": 2, "num_hidden_layers": 2, "num_key_value_heads": 1, "pad_token_id": 0,
    "rms_norm_eps": 1e-05, "rope_scaling": {"mrope_section": [8, 12, 12], "rope_type": "default", "type": "default"},
    "rope_theta": 500000, "tie_word_embeddings": False, "torch_dtype": "float32", "use_bias": False,
    "vocab_size": vocab_size, "image_token_id": ids["<|IMAGE_PLACEHOLDER|>"], "video_token_id": ids["<|video_pad|>"],
    "vision_start_token_id": ids["<|IMAGE_START|>"], "vision_end_token_id": ids["<|IMAGE_END|>"],
    "weight_share_add_bias": True, "use_3d_rope": True, "rope_is_neox_style": True,
    "vision_config": {
        "architectures": ["PaddleOCRVisionModel"], "model_type": "paddleocr_vl", "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 32, "image_size": 384, "intermediate_size": 64, "layer_norm_eps": 1e-06, "num_attention_heads": 2,
        "num_channels": 3, "num_hidden_layers": 2, "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2,
        "tokens_per_second": 2,
    },
}
preprocessor_config = {
    "do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "do_resize": True, "image_mean": [0.5, 0.5, 0.5],
    "image_processor_type": "PaddleOCRVLImageProcessor", "image_std": [0.5, 0.5, 0.5], "max_pixels": 1003520,
    "merge_size": 2, "min_pixels": 112896, "patch_size": 14, "processor_class": "PaddleOCRVLProcessor", "resample": 3,
    "rescale_factor": 1 / 255, "temporal_patch_size": 1,
}
chat_template = """{%- if not add_generation_prompt is defined -%}
    {%- set add_generation_prompt = true -%}
{%- endif -%}
{%- if not cls_token is defined -%}
    {%- set cls_token = "<|begin_of_sentence|>" -%}
{%- endif -%}
{%- if not eos_token is defined -%}
    {%- set eos_token = "</s>" -%}
{%- endif -%}
{{- cls_token -}}
{%- for message in messages -%}
    {%- if message["role"] == "user" -%}
        {{- "User: " -}}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "image" -%}
                {{ "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>" }}
            {%- endif -%}
        {%- endfor -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] }}
            {%- endif -%}
        {%- endfor -%}
        {{ "\\n" -}}
    {%- elif message["role"] == "assistant" -%}
        {{- "Assistant:\\n" -}}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] }}
            {%- endif -%}
        {%- endfor -%}
        {{ eos_token -}}
    {%- elif message["role"] == "system" -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] + "\\n" }}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- "Assistant:\\n" -}}
{%- endif -%}
"""

OUT.mkdir(exist_ok=True)
for name, obj in [
    ("tokenizer.json", tokenizer),
    ("tokenizer_config.json", tokenizer_config),
    ("config.json", config),
    ("preprocessor_config.json", preprocessor_config),
    ("generation_config.json", {"eos_token_id": 2, "pad_token_id": 0}),
]:
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
(OUT / "chat_template.jinja").write_text(chat_template)

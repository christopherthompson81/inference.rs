#!/usr/bin/env python3
"""Writes gemma3/: a tiny Gemma 3 checkpoint skeleton (no weights) for engine-behavior tests.

The tokenizer is a byte-fallback BPE with only the special tokens the processor and chat template use; the config
shrinks every dimension, keeps a sliding and a full layer, and turns a 32x32 image into 4 soft tokens. The tests
generate random weights by recording what the model constructor asks for.
"""
import json
import pathlib

OUT = pathlib.Path(__file__).parent / "gemma3"
SPECIALS = ["<bos>", "<eos>", "<pad>", "<start_of_turn>", "<end_of_turn>", "<start_of_image>", "<end_of_image>",
            "<image_soft_token>"]
IMAGE_SIDE = 32
PATCH = 8
MM_TOKENS = 4
# Bounds the random weights' logits, so greedy steps keep a spread a pin can see
FINAL_SOFTCAP = 5.0

vocab = {"<unk>": 0}
for b in range(256):
    vocab[f"<0x{b:02X}>"] = len(vocab)
ids = {tok: len(vocab) + i for i, tok in enumerate(SPECIALS)}
vocab_size = len(vocab) + len(SPECIALS)

tokenizer = {
    "version": "1.0", "truncation": None, "padding": None,
    "added_tokens": [
        {"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False,
         "special": True}
        for t, i in [("<unk>", 0)] + list(ids.items())
    ],
    "normalizer": None, "pre_tokenizer": None, "post_processor": None,
    "decoder": {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]},
    "model": {
        "type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None,
        "end_of_word_suffix": None, "fuse_unk": True, "byte_fallback": True, "ignore_merges": False, "vocab": vocab,
        "merges": [],
    },
}
tokenizer_config = {
    "add_bos_token": False, "add_eos_token": False, "bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
    "unk_token": "<unk>", "additional_special_tokens": SPECIALS[3:], "clean_up_tokenization_spaces": False,
    "model_max_length": 4096, "tokenizer_class": "GemmaTokenizer",
}
chat_template = """{{- bos_token -}}
{%- for message in messages -%}
    {{- '<start_of_turn>' + (message['role'] if message['role'] != 'assistant' else 'model') + '\\n' -}}
    {%- if message['content'] is string -%}
        {{- message['content'] -}}
    {%- else -%}
        {%- for content in message['content'] -%}
            {%- if content['type'] == 'image' -%}
                {{- '<start_of_image>' -}}
            {%- elif content['type'] == 'text' -%}
                {{- content['text'] -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {{- '<end_of_turn>\\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<start_of_turn>model\\n' -}}
{%- endif -%}
"""
# head_dim 64, the smallest the CUDA and Metal paged attention kernels take; layer 0 slides, layer 1 is global.
config = {
    "architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3",
    "image_token_index": ids["<image_soft_token>"], "mm_tokens_per_image": MM_TOKENS,
    "text_config": {
        "vocab_size": vocab_size, "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64, "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 4096, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0,
        "rope_local_base_freq": 10000.0, "sliding_window": 4, "sliding_window_pattern": 2,
        "query_pre_attn_scalar": 64, "attn_logit_softcapping": None, "final_logit_softcapping": FINAL_SOFTCAP,
        "rope_scaling": None, "quantization_config": None, "tie_word_embeddings": True,
    },
    "vision_config": {
        "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_channels": 3, "image_size": IMAGE_SIDE, "patch_size": PATCH, "hidden_act": "gelu_pytorch_tanh",
        "layer_norm_eps": 1e-06,
    },
}
preprocessor = {
    "do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "do_resize": True, "do_pan_and_scan": False,
    "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "rescale_factor": 1 / 255, "resample": 2,
    "size": {"height": IMAGE_SIDE, "width": IMAGE_SIDE}, "image_processor_type": "Gemma3ImageProcessor",
}
processor = {"image_seq_length": MM_TOKENS, "processor_class": "Gemma3Processor"}
generation = {"bos_token_id": ids["<bos>"], "eos_token_id": [ids["<eos>"], ids["<end_of_turn>"]],
              "pad_token_id": ids["<pad>"]}

OUT.mkdir(exist_ok=True)
for name, value in [("config.json", config), ("tokenizer.json", tokenizer), ("tokenizer_config.json", tokenizer_config),
                    ("preprocessor_config.json", preprocessor), ("processor_config.json", processor),
                    ("generation_config.json", generation)]:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")
(OUT / "chat_template.jinja").write_text(chat_template)

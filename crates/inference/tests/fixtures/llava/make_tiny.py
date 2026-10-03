#!/usr/bin/env python3
"""Writes llava15/ and llava_next/: tiny LLaVA checkpoint skeletons (no weights) for engine-behavior tests.

LLaVA 1.5 runs a Llama text model and LLaVA-NeXT a Mistral one, so both text paths the LLaVA wrappers dispatch to are
covered. The tokenizer is a byte-fallback BPE with only BOS/EOS; prompts carry the literal `<image>` tag, which the
LLaVA input processor splits on. A 28-pixel CLIP image at patch 14 is 4 vision tokens. The tests generate random
weights by recording what the model constructor asks for.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).parent
SPECIALS = ["<s>", "</s>"]

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
chat_template = (
    "{{- bos_token -}}{%- for message in messages -%}"
    "{%- if message['role'] == 'user' -%}{{- 'USER: ' + message['content'] + '\\n' -}}"
    "{%- else -%}{{- 'ASSISTANT: ' + message['content'] + eos_token -}}{%- endif -%}"
    "{%- endfor -%}{%- if add_generation_prompt -%}{{- 'ASSISTANT:' -}}{%- endif -%}"
)
tokenizer_config = {
    "add_bos_token": False, "add_eos_token": False, "bos_token": "<s>", "eos_token": "</s>", "pad_token": "<unk>",
    "unk_token": "<unk>", "clean_up_tokenization_spaces": False, "model_max_length": 4096,
    "tokenizer_class": "LlamaTokenizer", "chat_template": chat_template,
}
generation_config = {"bos_token_id": ids["<s>"], "eos_token_id": ids["</s>"]}

IMAGE = 28
PATCH = 14
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def text_config(model_type):
    return {
        "model_type": model_type, "vocab_size": vocab_size, "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "sliding_window": None, "rope_scaling": None,
    }


vision_config = {
    "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
    "image_size": IMAGE, "patch_size": PATCH,
}
processor = {
    "crop_size": {"height": IMAGE, "width": IMAGE}, "do_center_crop": True, "do_normalize": True, "do_resize": True,
    "do_rescale": True, "rescale_factor": 1 / 255, "image_mean": CLIP_MEAN, "image_std": CLIP_STD, "resample": 3,
    "size": {"shortest_edge": IMAGE},
}
MODELS = {
    "llava15": (
        {"architectures": ["LlavaForConditionalGeneration"], "model_type": "llava", "text_config": text_config("llama")},
        {**processor, "image_processor_type": "CLIPImageProcessor"},
    ),
    "llava_next": (
        {
            "architectures": ["LlavaNextForConditionalGeneration"], "model_type": "llava_next",
            "text_config": text_config("mistral"), "image_grid_pinpoints": [[IMAGE, 2 * IMAGE], [2 * IMAGE, IMAGE]],
        },
        {
            **processor, "image_processor_type": "LlavaNextImageProcessor", "do_pad": True,
            "image_grid_pinpoints": [[IMAGE, 2 * IMAGE], [2 * IMAGE, IMAGE]],
        },
    ),
}

for name, (config, preprocessor) in MODELS.items():
    out = ROOT / name
    out.mkdir(exist_ok=True)
    config = {
        **config, "vision_config": vision_config, "projector_hidden_act": "gelu", "vision_feature_layer": -2,
        "vision_feature_select_strategy": "default", "image_token_index": vocab_size,
        "torch_dtype": "float32", "tie_word_embeddings": False,
    }
    for file, body in [
        ("config.json", config), ("tokenizer.json", tokenizer), ("tokenizer_config.json", tokenizer_config),
        ("preprocessor_config.json", preprocessor), ("generation_config.json", generation_config),
    ]:
        (out / file).write_text(json.dumps(body, indent=1) + "\n")

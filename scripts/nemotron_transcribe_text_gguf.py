#!/usr/bin/env python3
"""Nemotron 3.5 Transcribe's decoder as llama.cpp's `nemotron_h_moe` GGUF (BF16), the source `inference quantize gguf`
quantizes. Runs llama.cpp's convert_hf_to_gguf.py on the decoder read in place from NVIDIA's NeMo checkpoint: its
tensors offered under the Hub's Nemotron-H names (stacked experts sliced and transposed, `_fp32_params` flattened), the
MTP head, audio tower and CTC head left out; the config is the checkpoint's `text_config` and the tokenizer drops
`<|audio|>`, which lies past the text vocabulary (`inference quantize nemotron-transcribe-audio` writes the rest).

Usage: scripts/nemotron_transcribe_text_gguf.py <llama.cpp dir> <checkpoint dir> <out.gguf>
"""

import json
import os
import re
import shutil
import sys
import tempfile

AUDIO_TOKEN = "<|audio|>"
LAYER = re.compile(r"^llm\.model\.layers\.(\d+)\.(.*)$")
ROOT = {
    "llm.model.embed_tokens.weight": "backbone.embeddings.weight",
    "llm.model.norm.weight": "backbone.norm_f.weight",
    "llm.lm_head.weight": "lm_head.weight",
}
STACKS = {
    "mixer.experts.gate_and_up_projs": "up_proj",
    "mixer.experts.down_projs": "down_proj",
}
COPIED = ["chat_template.jinja", "generation_config.json"]
# JSON has no infinity; the converter only needs a large bound
FLOAT_INFINITY = 1e30


def finite(value):
    if isinstance(value, dict) and set(value) == {"__float__"}:
        return (
            FLOAT_INFINITY
            if value["__float__"] == "Infinity"
            else float(value["__float__"])
        )
    if isinstance(value, list):
        return [finite(v) for v in value]
    return value


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(value, path, **kwargs):
    with open(path, "w") as f:
        json.dump(value, f, **kwargs)


def text_source(checkpoint, out):
    config = read_json(os.path.join(checkpoint, "config.json"))["text_config"]
    config = {k: finite(v) for k, v in config.items()}
    config["architectures"] = ["NemotronHForCausalLM"]
    write_json(config, os.path.join(out, "config.json"), indent=2)
    tokenizer = read_json(os.path.join(checkpoint, "tokenizer.json"))
    tokenizer["added_tokens"] = [
        t for t in tokenizer["added_tokens"] if t["content"] != AUDIO_TOKEN
    ]
    write_json(tokenizer, os.path.join(out, "tokenizer.json"), ensure_ascii=False)
    tokenizer_config = read_json(os.path.join(checkpoint, "tokenizer_config.json"))
    tokenizer_config.pop("audio_token", None)
    tokenizer_config.pop("added_tokens_decoder", None)
    for key in ("model_specific_special_tokens", "extra_special_tokens"):
        if isinstance(tokenizer_config.get(key), dict):
            tokenizer_config[key] = {
                k: v for k, v in tokenizer_config[key].items() if v != AUDIO_TOKEN
            }
    write_json(tokenizer_config, os.path.join(out, "tokenizer_config.json"), indent=2)
    for name in COPIED:
        shutil.copy(os.path.join(checkpoint, name), out)


def main():
    llama_cpp, checkpoint, out = sys.argv[1:4]
    sys.path.insert(0, llama_cpp)
    sys.path.insert(0, os.path.join(llama_cpp, "gguf-py"))
    import gguf
    from conversion import base

    weights = os.path.join(checkpoint, "model.safetensors")

    def nemo_tensors(self, remote_hf_model_id=None):
        tensors = {}

        def offer(name, gen):
            if item := self.filter_tensors((name, gen)):
                tensors[item[0]] = item[1]

        with gguf.utility.SafetensorsLocal(weights) as part:
            for name in part:
                data = part[name]
                lazy = lambda data=data: base.LazyTorchTensor.from_local_tensor(data)
                if name in ROOT:
                    offer(ROOT[name], lazy)
                    continue
                m = LAYER.match(name)
                if not m or name.endswith("._extra_state"):
                    continue
                layer, rest = m.groups()
                rest = rest.replace("mixer._fp32_params.", "mixer.")
                if rest in STACKS:
                    # `[experts, in, out]` to one `[out, in]` matrix per expert
                    for e in range(data.shape[0]):
                        offer(
                            f"backbone.layers.{layer}.mixer.experts.{e}.{STACKS[rest]}.weight",
                            lambda lazy=lazy, e=e: lazy()[e].transpose(0, 1),
                        )
                    continue
                offer(f"backbone.layers.{layer}.{rest}", lazy)
        return tensors

    base.ModelBase.index_tensors = nemo_tensors
    with tempfile.TemporaryDirectory() as src:
        text_source(checkpoint, src)
        sys.argv = ["convert_hf_to_gguf.py", src, "--outtype", "bf16", "--outfile", out]
        import convert_hf_to_gguf

        convert_hf_to_gguf.main()


if __name__ == "__main__":
    main()

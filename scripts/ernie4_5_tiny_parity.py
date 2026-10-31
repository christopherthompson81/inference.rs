#!/usr/bin/env python3
"""Parity check of our ERNIE 4.5 dense and MoE text models against transformers, on tiny random checkpoints.

transformers builds a small Ernie4_5ForCausalLM and an Ernie4_5_MoeForCausalLM (layer 0 dense, 8 experts top 3, two
shared experts, a random selection bias) with seeded random weights over ERNIE's real 103,424-token vocabulary
(tokenizer files from baidu/ERNIE-4.5-0.3B-PT at REVISION) and saves them, the MoE also with its experts stacked. For
each, both sides score the same token ids (mean KL(transformers || ours), top-1 agreement, largest logit difference)
and decode greedily, and our tokenizer and chat template are compared with transformers' on the README and a short
chat. With --llama-cpp, that checkout's converter also writes f32 GGUFs (llama.cpp rebuilds the tokenizer from
tokenizer.model), scored the same way. Not part of local_ci.sh: it needs transformers 5 with torch and sentencepiece,
and our library built with `cargo build -p inference-ffi` (add `--features cuda` for --device cuda:0), found through
the checkout's bindings.

Usage: PYTHONPATH=bindings/python python3 scripts/ernie4_5_tiny_parity.py [--max-kl X] [--device cpu|cuda:0]
           [--dtype f32|bf16] [--keep DIR] [--seed N] [--llama-cpp DIR]
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOKENIZER_REPO = "baidu/ERNIE-4.5-0.3B-PT"
REVISION = "b565cf6caebdb7a1eadf00100857b1ed5e044f12"
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
]
DEFAULT_MAX_KL = 1e-6
# bf16 weights and activations against an f32 reference
BF16_MAX_KL = 2e-3
# how far below transformers' best logit a bf16 step's token may score: a near tie, not a wrong selection
BF16_TIE_MARGIN = 0.25
GENERATED = 12
IDS = [11, 57, 203, 4, 98, 140, 33, 77, 19, 64, 222, 5, 180, 91, 42, 7, 130, 1000, 2000]
CHAT = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "4."},
    {"role": "user", "content": "And 3 + 3?"},
]
ROOT = Path(__file__).resolve().parent.parent
GGUF_FILE = "model.gguf"
BASE = {
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "use_bias": False,
    "tie_word_embeddings": True,
    "vocab_size": 103424,
    # wide enough that the logits are peaked: at the default 0.02 every distribution is near uniform
    "initializer_range": 0.1,
}
MOE = {
    **BASE,
    "moe_intermediate_size": 32,
    "moe_num_experts": 8,
    "moe_k": 3,
    "moe_num_shared_experts": 2,
    "moe_layer_start_index": 1,
    "moe_layer_end_index": -1,
    "moe_layer_interval": 1,
    "moe_norm_min": 1e-12,
}


def fetch_tokenizer(model_dir: Path):
    from huggingface_hub import hf_hub_download

    for name in TOKENIZER_FILES:
        shutil.copy(
            hf_hub_download(TOKENIZER_REPO, name, revision=REVISION), model_dir / name
        )


def build(model_dir: Path, seed: int, moe: bool):
    import torch
    from transformers import (
        Ernie4_5_MoeConfig,
        Ernie4_5_MoeForCausalLM,
        Ernie4_5Config,
        Ernie4_5ForCausalLM,
    )

    torch.manual_seed(seed)
    config = Ernie4_5_MoeConfig(**MOE) if moe else Ernie4_5Config(**BASE)
    model = (
        (Ernie4_5_MoeForCausalLM if moe else Ernie4_5ForCausalLM)(config)
        .to(torch.float32)
        .eval()
    )
    # random init leaves norms at one and the selection bias at zero; perturb them so a dropped one shows
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.add_(torch.randn_like(param) * 0.1)
            if name.endswith("e_score_correction_bias"):
                param.copy_(torch.randn_like(param) * 0.1)
    model.save_pretrained(model_dir, safe_serialization=True)
    fetch_tokenizer(model_dir)
    return model


def reference(model):
    import torch

    with torch.no_grad():
        logits = model(input_ids=torch.tensor([IDS])).logits[0].float()
        generated = model.generate(
            torch.tensor([IDS]),
            max_new_tokens=GENERATED,
            do_sample=False,
            eos_token_id=None,
        )
    return logits.tolist(), generated[0, len(IDS) :].tolist()


def tie_gaps(model, continuation: list[int]) -> list[float]:
    """Teacher-forced: at each step, transformers' best logit minus the logit of the token we chose."""
    import torch

    with torch.no_grad():
        logits = model(input_ids=torch.tensor([IDS + continuation])).logits[
            0, len(IDS) - 1 : -1
        ]
    return [(row.max() - row[token]).item() for row, token in zip(logits, continuation)]


def stack_experts(model_dir: Path, stacked_dir: Path):
    import torch
    from safetensors.torch import load_file, save_file

    shutil.copytree(model_dir, stacked_dir)
    tensors = load_file(model_dir / "model.safetensors")
    for layer in range(MOE["num_hidden_layers"]):
        prefix = f"model.layers.{layer}.mlp.experts"
        if f"{prefix}.0.gate_proj.weight" not in tensors:
            continue
        gate, up, down = (
            [
                tensors.pop(f"{prefix}.{e}.{proj}.weight")
                for e in range(MOE["moe_num_experts"])
            ]
            for proj in ("gate_proj", "up_proj", "down_proj")
        )
        tensors[f"{prefix}.gate_up_proj"] = torch.stack(
            [torch.cat([g, u]) for g, u in zip(gate, up)]
        )
        tensors[f"{prefix}.down_proj"] = torch.stack(down)
    save_file(tensors, stacked_dir / "model.safetensors", metadata={"format": "pt"})


def convert(model_dir: Path, llama_cpp: Path) -> Path:
    gguf = model_dir / GGUF_FILE
    command = [
        sys.executable,
        "-E",
        "convert_hf_to_gguf.py",
        str(model_dir),
        "--outtype",
        "f32",
        "--outfile",
        str(gguf),
    ]
    subprocess.run(command, cwd=llama_cpp, check=True, capture_output=True)
    return gguf


def spec(model: Path, device: str, dtype: str) -> str:
    if model.suffix == ".gguf":
        selected = {
            "GGUF": {
                "quantized_model_id": str(model.parent),
                "quantized_filename": model.name,
                "dtype": dtype,
            }
        }
    else:
        selected = {"Plain": {"model_id": str(model), "dtype": dtype}}
    return json.dumps(
        {
            "model": selected,
            "runtime": {"device": device, "paged_attn": device != "cpu"},
        }
    )


def ours(model: Path, device: str, dtype: str, text: str):
    import inference_rs as ir

    request = {
        "model": "default",
        "prompt": IDS,
        "max_tokens": GENERATED,
        "temperature": 0.0,
        "ignore_eos": True,
        "logprobs": 1,
    }
    with ir.JsonEngine(spec(model, device, dtype)) as engine:
        scores, logits = engine.prompt_logits(
            json.dumps({"prompt": IDS, "output": "logits"})
        )
        choice = json.loads(engine.completion(json.dumps(request)))["choices"][0]
        encoded = json.loads(
            engine.tokenize(json.dumps({"text": text, "add_special_tokens": False}))
        )["tokens"]
        chat = json.loads(
            engine.tokenize_chat(json.dumps({"model": "default", "messages": CHAT}))
        )["tokens"]
    vocab = json.loads(scores)["vocab_size"]
    rows = [list(logits[i * vocab : (i + 1) * vocab]) for i in range(len(IDS))]
    # greedy, so each step's top alternative is the token it chose, and that entry carries the id
    continuation = [
        step["top_logprobs"][0]["token"] for step in choice["logprobs"]["content"]
    ]
    return rows, continuation, encoded, chat


def compare(expected, actual):
    import torch

    e, a = torch.tensor(expected), torch.tensor(actual)
    if e.shape != a.shape:
        sys.exit(f"transformers' logits are {tuple(e.shape)}, ours {tuple(a.shape)}")
    le, la = e.log_softmax(-1), a.log_softmax(-1)
    kl = (le.exp() * (le - la)).sum(-1)
    agree = (e.argmax(-1) == a.argmax(-1)).float().mean().item()
    return kl.mean().item(), agree, (e - a).abs().max().item()


def check(label, model_path, model, expected, args, dtype, max_kl, text) -> list[str]:
    logits, generated, tokens, chat = expected
    actual, continuation, encoded, our_chat = ours(model_path, args.device, dtype, text)
    kl, agree, max_diff = compare(logits, actual)
    print(
        f"{label}: {len(IDS)} tokens, mean KL {kl:.2e}, top-1 agreement {agree:.3f}, max logit diff {max_diff:.4f}"
    )
    print(
        f"{label}: greedy {GENERATED} tokens, transformers {generated}, ours {continuation}"
    )
    print(
        f"{label}: README {len(tokens)} tokens, ours {'the same' if encoded == tokens else len(encoded)}"
    )
    print(
        f"{label}: chat {len(chat)} tokens, ours {'the same' if our_chat == chat else our_chat}"
    )
    failures = []
    if kl > max_kl:
        failures.append(f"{label}: mean KL {kl:.2e} exceeds {max_kl:.0e}")
    # bf16 can flip a near tie, after which the continuations part ways; each of our steps must still be one
    if dtype == "f32" and continuation != generated:
        failures.append(f"{label}: the greedy continuations differ")
    gap = max(tie_gaps(model, continuation))
    if gap > BF16_TIE_MARGIN:
        failures.append(
            f"{label}: a step's token scores {gap:.3f} below transformers' best"
        )
    if encoded != tokens:
        failures.append(f"{label}: the README tokenizes differently")
    if our_chat != chat:
        failures.append(f"{label}: the chat renders differently")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-kl", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["f32", "bf16"])
    parser.add_argument(
        "--keep", type=Path, help="write the tiny checkpoints here and keep them"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--llama-cpp",
        type=Path,
        help="a llama.cpp checkout whose converter writes the GGUFs",
    )
    args = parser.parse_args()
    dtype = args.dtype or ("f32" if args.device == "cpu" else "bf16")
    max_kl = args.max_kl or (DEFAULT_MAX_KL if dtype == "f32" else BF16_MAX_KL)
    from transformers import AutoTokenizer

    text = (ROOT / "README.md").read_text()
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        root = args.keep or Path(tmp)
        for name, moe in [("dense", False), ("moe", True)]:
            model_dir = root / name
            shutil.rmtree(model_dir, ignore_errors=True)
            model_dir.mkdir(parents=True)
            model = build(model_dir, args.seed, moe)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            # the template writes its own `<|begin_of_sentence|>`, so the rendered text is encoded without specials
            rendered = tokenizer.apply_chat_template(
                CHAT, add_generation_prompt=True, tokenize=False
            )
            chat = tokenizer.encode(rendered, add_special_tokens=False)
            expected = (
                *reference(model),
                tokenizer.encode(text, add_special_tokens=False),
                chat,
            )
            loads = [(name, model_dir)]
            if moe:
                stacked_dir = root / "moe-stacked"
                shutil.rmtree(stacked_dir, ignore_errors=True)
                stack_experts(model_dir, stacked_dir)
                loads.append(("moe stacked", stacked_dir))
            if args.llama_cpp:
                loads.append(
                    (f"{name} gguf", convert(model_dir, args.llama_cpp.resolve()))
                )
            for label, path in loads:
                failures += check(
                    label, path, model, expected, args, dtype, max_kl, text
                )
    if failures:
        sys.exit("; ".join(failures))
    print("OK")


if __name__ == "__main__":
    main()

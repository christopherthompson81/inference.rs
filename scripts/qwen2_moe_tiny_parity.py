#!/usr/bin/env python3
"""Parity check of our Qwen2-MoE text model against transformers, on a tiny random checkpoint.

transformers builds a small Qwen2MoeForCausalLM (q/k/v bias, routed experts without top-k renormalization, a gated
shared expert, one dense layer) with seeded random weights and saves it with its experts one by one, then a copy with
them stacked (`gate_up_proj`, `down_proj`) as newer checkpoints store them; for each, both sides score the same token ids, and the script reports the mean KL(transformers || ours) over positions,
top-1 agreement and the largest logit difference, then compares a greedy continuation. With --llama-cpp, an all-MoE
checkpoint (llama.cpp's converter maps no dense layer) is also converted to an f32 GGUF by that checkout and loaded
from it. Fails past --max-kl or on a different continuation. Not part of local_ci.sh: it needs transformers 5 with
torch, and our library built with `cargo build -p inference-ffi` (add `--features cuda` for --device cuda:0), found
through the checkout's bindings.

Usage: PYTHONPATH=bindings/python python3 scripts/qwen2_moe_tiny_parity.py [--max-kl X] [--device cpu|cuda:0]
           [--dtype f32|bf16] [--keep DIR] [--seed N] [--llama-cpp DIR]
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_MAX_KL = 1e-6
# bf16 weights and activations against an f32 reference
BF16_MAX_KL = 2e-3
# how far below transformers' best logit a bf16 step's token may score: a near tie, not a wrong selection
BF16_TIE_MARGIN = 0.25
FIXTURES = Path(__file__).resolve().parent.parent / "crates/inference/tests/fixtures"
TOKENIZER = FIXTURES / "paddleocr_vl/tiny/tokenizer.json"
VOCAB = 266
GENERATED = 12
IDS = [11, 57, 203, 4, 98, 140, 33, 77, 19, 64, 222, 5, 180, 91, 42, 7, 130]
GGUF_FILE = "model.gguf"
# our fixture tokenizer's pre-tokenizer hash is unknown to llama.cpp; the converter only records the name
CONVERT = """
import runpy, sys
sys.path.insert(0, ".")
import conversion.base as base
base.TextModel.get_vocab_base_pre = lambda self, tokenizer: "qwen2"
sys.argv = ["convert_hf_to_gguf.py", *sys.argv[1:]]
runpy.run_path("convert_hf_to_gguf.py", run_name="__main__")
"""
CONFIG = {
    "architectures": ["Qwen2MoeForCausalLM"],
    "model_type": "qwen2_moe",
    "vocab_size": VOCAB,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 48,
    "norm_topk_prob": False,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [1],
    "qkv_bias": True,
    "use_sliding_window": False,
    "tie_word_embeddings": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
    # wide enough that the logits are peaked: at the default 0.02 every distribution is near uniform
    "initializer_range": 0.1,
}


def build(model_dir: Path, seed: int, config: dict):
    import torch
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    torch.manual_seed(seed)
    config = Qwen2MoeConfig(**{k: v for k, v in config.items() if k != "architectures"})
    model = Qwen2MoeForCausalLM(config).to(torch.float32).eval()
    # random init leaves norms at one and biases at zero; perturb them so a dropped one shows
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("norm.weight", ".bias")):
                param.add_(torch.randn_like(param) * 0.1)
    model.save_pretrained(model_dir, safe_serialization=True)
    shutil.copy(TOKENIZER, model_dir / "tokenizer.json")
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([IDS])).logits[0].float().tolist()
        generated = model.generate(
            torch.tensor([IDS]),
            max_new_tokens=GENERATED,
            do_sample=False,
            eos_token_id=None,
        )[0, len(IDS) :]
    return logits, generated.tolist()


def tie_gaps(model_dir: Path, continuation: list[int]) -> list[float]:
    """Teacher-forced: at each step, transformers' best logit minus the logit of the token we chose."""
    import torch
    from transformers import Qwen2MoeForCausalLM

    model = Qwen2MoeForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
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
    for layer in range(CONFIG["num_hidden_layers"]):
        prefix = f"model.layers.{layer}.mlp.experts"
        if f"{prefix}.0.gate_proj.weight" not in tensors:
            continue
        gate, up, down = (
            [
                tensors.pop(f"{prefix}.{e}.{proj}.weight")
                for e in range(CONFIG["num_experts"])
            ]
            for proj in ("gate_proj", "up_proj", "down_proj")
        )
        tensors[f"{prefix}.gate_up_proj"] = torch.stack(
            [torch.cat([g, u]) for g, u in zip(gate, up)]
        )
        tensors[f"{prefix}.down_proj"] = torch.stack(down)
    save_file(tensors, stacked_dir / "model.safetensors", metadata={"format": "pt"})


def convert(model_dir: Path, llama_cpp: Path) -> Path:
    # AutoTokenizer would add Qwen2's own specials past the tiny vocab
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
    )
    gguf = model_dir / GGUF_FILE
    command = [
        sys.executable,
        "-E",
        "-c",
        CONVERT,
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


def ours(model_dir: Path, device: str, dtype: str):
    import inference_rs as ir

    request = {
        "model": "default",
        "prompt": IDS,
        "max_tokens": GENERATED,
        "temperature": 0.0,
        "ignore_eos": True,
        "logprobs": 1,
    }
    with ir.JsonEngine(spec(model_dir, device, dtype)) as engine:
        scores, logits = engine.prompt_logits(
            json.dumps({"prompt": IDS, "output": "logits"})
        )
        choice = json.loads(engine.completion(json.dumps(request)))["choices"][0]
    vocab = json.loads(scores)["vocab_size"]
    rows = [list(logits[i * vocab : (i + 1) * vocab]) for i in range(len(IDS))]
    # greedy, so each step's top alternative is the token it chose, and that entry carries the id
    return rows, [
        step["top_logprobs"][0]["token"] for step in choice["logprobs"]["content"]
    ]


def compare(expected, actual):
    import torch

    e, a = torch.tensor(expected), torch.tensor(actual)
    if e.shape != a.shape:
        sys.exit(f"transformers' logits are {tuple(e.shape)}, ours {tuple(a.shape)}")
    le, la = e.log_softmax(-1), a.log_softmax(-1)
    kl = (le.exp() * (le - la)).sum(-1)
    agree = (e.argmax(-1) == a.argmax(-1)).float().mean().item()
    return kl.mean().item(), agree, (e - a).abs().max().item()


def check(label, model, reference_dir, reference, args, dtype, max_kl) -> list[str]:
    expected, generated = reference
    actual, continuation = ours(model, args.device, dtype)
    kl, agree, max_diff = compare(expected, actual)
    print(
        f"{label}: {len(IDS)} tokens, mean KL {kl:.2e}, top-1 agreement {agree:.3f}, max logit diff {max_diff:.4f}"
    )
    print(
        f"{label}: greedy {GENERATED} tokens, transformers {generated}, ours {continuation}"
    )
    failures = []
    if kl > max_kl:
        failures.append(f"{label}: mean KL {kl:.2e} exceeds {max_kl:.0e}")
    # bf16 can flip a near tie, after which the continuations part ways; each of our steps must still be one
    if dtype == "f32" and continuation != generated:
        failures.append(f"{label}: the greedy continuations differ")
    gap = max(tie_gaps(reference_dir, continuation))
    if gap > BF16_TIE_MARGIN:
        failures.append(
            f"{label}: a step's token scores {gap:.3f} below transformers' best"
        )
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
        help="a llama.cpp checkout whose converter writes the GGUF",
    )
    args = parser.parse_args()
    dtype = args.dtype or ("f32" if args.device == "cpu" else "bf16")
    max_kl = args.max_kl or (DEFAULT_MAX_KL if dtype == "f32" else BF16_MAX_KL)
    with tempfile.TemporaryDirectory() as tmp:
        root = args.keep or Path(tmp)
        model_dir, stacked_dir, all_moe_dir = (
            root / "per-expert",
            root / "stacked",
            root / "all-moe",
        )
        for directory in (model_dir, stacked_dir, all_moe_dir):
            shutil.rmtree(directory, ignore_errors=True)
        model_dir.mkdir(parents=True)
        reference = build(model_dir, args.seed, CONFIG)
        stack_experts(model_dir, stacked_dir)
        failures = check(
            "per-expert", model_dir, model_dir, reference, args, dtype, max_kl
        )
        failures += check(
            "stacked", stacked_dir, model_dir, reference, args, dtype, max_kl
        )
        if args.llama_cpp:
            all_moe_dir.mkdir()
            all_moe = build(all_moe_dir, args.seed, {**CONFIG, "mlp_only_layers": []})
            gguf = convert(all_moe_dir, args.llama_cpp.resolve())
            failures += check("gguf", gguf, all_moe_dir, all_moe, args, dtype, max_kl)
    if failures:
        sys.exit("; ".join(failures))
    print("OK")


if __name__ == "__main__":
    main()

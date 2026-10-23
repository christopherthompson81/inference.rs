#!/usr/bin/env python3
"""Parity check of our Qwen3.8-Flash-Next (qwen4exp) text model against transformers, on a tiny random checkpoint.

transformers builds a small Qwen4ExpForCausalLM (hyper-connections, PLE at layer 2, a sigmoid GDN gate, a few experts)
with seeded random weights and saves it; both sides score the same token ids, with an EOS inside so PLE's n-gram reset
is exercised, and the script reports the mean KL(transformers || ours) over positions, top-1 agreement and the largest
logit difference. Both then decode GENERATED tokens greedily from it, step by step, which carries PLE's token history
and conv state across steps, and the token ids must match. By default QSA's budget sits above the prompt, so both run dense attention.
Fails past --max-kl or on a different continuation. Not part of
local_ci.sh: it needs transformers 5.19 or newer with torch, and our library built with
`cargo build -p inference-ffi` (add `--features cuda` for --device cuda:0), found through the checkout's bindings.

Usage: PYTHONPATH=bindings/python python3 scripts/qwen4exp_tiny_parity.py [--max-kl X] [--device cpu|cuda:0]
           [--dtype f32|bf16] [--keep DIR] [--seed N] [--budget N]
       Our GDN kernels on CUDA take half precision, so a CUDA run loads ours in bf16 against transformers' f32 and
       holds it to the looser bf16 bar. --budget shrinks QSA's budget below the sequence, so the indexer's selection
       runs (our sparse path needs paged attention on CUDA).
"""

import argparse
import json
import shutil
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
EOS = 2
GENERATED = 12
# a fixed id sequence with an EOS in the middle, so the second segment's n-grams start fresh
IDS = [11, 57, 203, 4, 98, 140, 33, 77, EOS, 19, 64, 222, 5, 180, 91, 42, 7, 130]
# small enough to build in a second, wide enough to keep every new piece's shapes honest
CONFIG = {
    "architectures": ["Qwen4ExpForCausalLM"],
    "model_type": "qwen4_exp_text",
    "vocab_size": VOCAB,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    # fattn, which QSA's selection runs on, has no head-dim-32 kernel
    "head_dim": 64,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [4, 2, 2],
        "mrope_interleaved": True,
    },
    "hc_count": 4,
    "hc_lowrank": 8,
    "output_gate_type": "sigmoid",
    "ple_layer_ids": [2],
    "ple_embed_dim": 64,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ngram_vocab_size_base": 1000,
    "make_ngram_vocab_size_divisible_by": 128,
    "split_ngram_parts": 1,
    "indexer_n_heads": 2,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 16,
    "indexer_budget": 1024,
    "indexer_compress_ratio": 4,
    "eos_token_id": EOS,
    "bos_token_id": 1,
    "tie_word_embeddings": False,
    "mtp_num_hidden_layers": 0,
    # wide enough that the logits are peaked: at the default 0.02 every distribution is near uniform
    "initializer_range": 0.1,
}


def tie_gaps(model_dir: Path, continuation: list[int]) -> list[float]:
    """Teacher-forced: at each step, transformers' best logit minus the logit of the token we chose."""
    import torch
    from transformers import Qwen4ExpForCausalLM

    model = Qwen4ExpForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([IDS + continuation])).logits[0, len(IDS) - 1 : -1]
    return [(row.max() - row[token]).item() for row, token in zip(logits, continuation)]


def build(model_dir: Path, seed: int, budget: int):
    import torch
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    torch.manual_seed(seed)
    fields = {k: v for k, v in CONFIG.items() if k != "architectures"}
    config = Qwen4ExpTextConfig(**{**fields, "indexer_budget": budget})
    model = Qwen4ExpForCausalLM(config).to(torch.float32).eval()
    # random init leaves the zero-centred norms at zero; perturb them so a missed `1 + w` shows
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.add_(torch.randn_like(param) * 0.1)
    model.save_pretrained(model_dir, safe_serialization=True)
    shutil.copy(TOKENIZER, model_dir / "tokenizer.json")
    config = json.loads((model_dir / "config.json").read_text())
    config["architectures"] = CONFIG["architectures"]
    (model_dir / "config.json").write_text(json.dumps(config))
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([IDS])).logits[0].float().tolist()
        generated = model.generate(
            torch.tensor([IDS]), max_new_tokens=GENERATED, do_sample=False, eos_token_id=None
        )[0, len(IDS) :]
    return logits, generated.tolist()


def our_continuation(model_dir: Path, device: str, dtype: str) -> list[int]:
    import inference_rs as ir

    spec = {
        "model": {"Plain": {"model_id": str(model_dir), "dtype": dtype}},
        "runtime": {"device": device, "paged_attn": device != "cpu"},
    }
    request = {
        "model": "default",
        "prompt": IDS,
        "max_tokens": GENERATED,
        "temperature": 0.0,
        "ignore_eos": True,
        "logprobs": 1,
    }
    with ir.JsonEngine(json.dumps(spec)) as engine:
        choice = json.loads(engine.completion(json.dumps(request)))["choices"][0]
    # greedy, so each step's top alternative is the token it chose, and that entry carries the id
    return [step["top_logprobs"][0]["token"] for step in choice["logprobs"]["content"]]


def our_logits(model_dir: Path, device: str, dtype: str):
    import inference_rs as ir

    spec = {
        "model": {"Plain": {"model_id": str(model_dir), "dtype": dtype}},
        "runtime": {"device": device, "paged_attn": device != "cpu"},
    }
    with ir.JsonEngine(json.dumps(spec)) as engine:
        scores, logits = engine.prompt_logits(json.dumps({"prompt": IDS, "output": "logits"}))
    vocab = json.loads(scores)["vocab_size"]
    return [list(logits[i * vocab : (i + 1) * vocab]) for i in range(len(IDS))]


def compare(expected, actual):
    import torch

    e, a = torch.tensor(expected), torch.tensor(actual)
    if e.shape != a.shape:
        sys.exit(f"transformers' logits are {tuple(e.shape)}, ours {tuple(a.shape)}")
    le, la = e.log_softmax(-1), a.log_softmax(-1)
    kl = (le.exp() * (le - la)).sum(-1)
    agree = (e.argmax(-1) == a.argmax(-1)).float().mean().item()
    return kl.mean().item(), kl.tolist(), agree, (e - a).abs().max().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-kl", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["f32", "bf16"])
    parser.add_argument("--keep", type=Path, help="write the tiny checkpoint here and keep it")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=CONFIG["indexer_budget"])
    args = parser.parse_args()
    dtype = args.dtype or ("f32" if args.device == "cpu" else "bf16")
    max_kl = args.max_kl or (DEFAULT_MAX_KL if dtype == "f32" else BF16_MAX_KL)
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = args.keep or Path(tmp)
        model_dir.mkdir(parents=True, exist_ok=True)
        expected, generated = build(model_dir, args.seed, args.budget)
        actual = our_logits(model_dir, args.device, dtype)
        continuation = our_continuation(model_dir, args.device, dtype)
        gaps = tie_gaps(model_dir, continuation)
    kl, per_position, agree, max_diff = compare(expected, actual)
    print(f"{len(IDS)} tokens: mean KL {kl:.2e}, top-1 agreement {agree:.3f}, max logit diff {max_diff:.4f}")
    print("per-position KL:", " ".join(f"{value:.1e}" for value in per_position))
    print(f"greedy {GENERATED} tokens: transformers {generated}, ours {continuation}")
    print("our tokens below transformers' best logit:", " ".join(f"{gap:.3f}" for gap in gaps))
    if kl > max_kl:
        sys.exit(f"mean KL {kl:.2e} is past {max_kl}")
    # bf16 can flip a near tie, after which the continuations part ways; each of our steps must still be one
    if dtype == "f32" and continuation != generated:
        sys.exit("the greedy continuations differ")
    if max(gaps) > BF16_TIE_MARGIN:
        sys.exit(f"a step's token scores {max(gaps):.3f} below transformers' best")


if __name__ == "__main__":
    main()

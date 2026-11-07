#!/usr/bin/env python3
"""Parity check of our Falcon-H1 against transformers on a real checkpoint (e.g. tiiuae/Falcon-H1-0.5B-Instruct).

Both sides score the same prompt, long enough to cross a Mamba chunk, in f32 on the CPU; the script reports the mean
KL(transformers || ours) over positions, top-1 agreement and the largest logit difference, then decodes GENERATED tokens
greedily on both sides and compares them. The checkpoint's muP multipliers run on the activations here; a GGUF has
them folded in. Fails past --max-kl or on a different continuation. Not part of local_ci.sh: it needs transformers with
torch and our library (`scripts/local_ci.sh --bindings` builds it), found through the checkout's bindings.

Usage: PYTHONPATH=bindings/python INFERENCE_NATIVE_DIR=target/debug python3 scripts/falcon_h1_parity.py <checkpoint>
           [--max-kl X] [--device cpu|cuda:0] [--dtype f32|bf16]
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MAX_KL = 1e-4
# bf16 weights and activations against an f32 reference
BF16_MAX_KL = 2e-2
GENERATED = 24
# a few hundred tokens: past the 0.5B's 128-token Mamba chunk, so the chunked scan carries state between chunks
PROMPT = (
    "The history of the printing press begins in the fifteenth century. Johannes Gutenberg, a goldsmith from "
    "Mainz, combined a screw press with movable metal type, an oil-based ink and a hand mould for casting letters. "
    "Before his work, books were copied by hand or printed from carved wooden blocks, which made them expensive and "
    "slow to produce. Within decades, presses spread across Europe, from Venice and Paris to London, and the price of "
    "books fell sharply. Printers standardised spelling, published scientific results, and spread pamphlets during the "
    "Reformation. Historians often argue that the press changed how knowledge moved: a scholar in one city could read "
    "the same text, page for page, as a scholar in another. In the following centuries, steam power, rotary presses and "
    "later offset lithography made printing faster still. Summarise the main effects of the printing press in three "
    "short points:"
)


def reference(checkpoint: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    ids = tokenizer(PROMPT)["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).eval()
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0].float().tolist()
        generated = model.generate(
            torch.tensor([ids]), max_new_tokens=GENERATED, do_sample=False, min_new_tokens=GENERATED
        )[0, len(ids) :]
    return ids, logits, generated.tolist()


def ours(checkpoint: Path, ids: list[int], device: str, dtype: str):
    import inference_rs as ir

    spec = {
        "model": {"Plain": {"model_id": str(checkpoint), "dtype": dtype}},
        "runtime": {"device": device, "paged_attn": device != "cpu"},
    }
    request = {
        "model": "default",
        "prompt": ids,
        "max_tokens": GENERATED,
        "temperature": 0.0,
        "ignore_eos": True,
        "logprobs": 1,
    }
    with ir.JsonEngine(json.dumps(spec)) as engine:
        scores, logits = engine.prompt_logits(json.dumps({"prompt": ids, "output": "logits"}))
        choice = json.loads(engine.completion(json.dumps(request)))["choices"][0]
    vocab = json.loads(scores)["vocab_size"]
    rows = [list(logits[i * vocab : (i + 1) * vocab]) for i in range(len(ids))]
    # greedy, so each step's top alternative is the token it chose, and that entry carries the id
    continuation = [step["top_logprobs"][0]["token"] for step in choice["logprobs"]["content"]]
    return rows, continuation


def compare(expected, actual):
    import torch

    e, a = torch.tensor(expected), torch.tensor(actual)
    if e.shape != a.shape:
        sys.exit(f"transformers' logits are {tuple(e.shape)}, ours {tuple(a.shape)}")
    le, la = e.log_softmax(-1), a.log_softmax(-1)
    kl = (le.exp() * (le - la)).sum(-1)
    agree = (e.argmax(-1) == a.argmax(-1)).float().mean().item()
    return kl.mean().item(), kl.max().item(), agree, (e - a).abs().max().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--max-kl", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["f32", "bf16"])
    args = parser.parse_args()
    dtype = args.dtype or ("f32" if args.device == "cpu" else "bf16")
    max_kl = args.max_kl or (DEFAULT_MAX_KL if dtype == "f32" else BF16_MAX_KL)
    ids, expected, generated = reference(args.checkpoint)
    actual, continuation = ours(args.checkpoint, ids, args.device, dtype)
    kl, worst, agree, max_diff = compare(expected, actual)
    print(f"{len(ids)} tokens: mean KL {kl:.2e} (worst {worst:.2e}), top-1 agreement {agree:.3f}, "
          f"max logit diff {max_diff:.4f}")
    print(f"greedy {GENERATED} tokens: transformers {generated}")
    print(f"                    ours         {continuation}")
    if kl > max_kl:
        sys.exit(f"mean KL {kl:.2e} is past {max_kl}")
    if dtype == "f32" and continuation != generated:
        sys.exit("the greedy continuations differ")


if __name__ == "__main__":
    main()

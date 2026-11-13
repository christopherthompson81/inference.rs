#!/usr/bin/env python3
"""Our collected input Hessians against transformers': the same calibration chunks (1024 tokens, the BOS token
first where the tokenizer has one) through the Hugging Face model with forward hooks on the chosen linears, each
H = sum x^T x / rows, compared to the `<key>.safetensors` (`hin`) files the `exl3_hessians` example wrote.

Run with a Python that has torch, transformers and safetensors:
  python3 scripts/exl3_hessian_parity.py <model dir> <calibration.txt> <hessian dir> [layer index ...]
"""

import os
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

CHUNK = 1024
# both sides run the model in bf16, so the activations, and the Hessians built from them, agree to bf16 rounding
TOLERANCE = 2e-2
PROJECTIONS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def main():
    model_dir, calibration, hessians = sys.argv[1:4]
    layers = [int(i) for i in sys.argv[4:]] or [0]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = (
        AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    keys = [f"model.layers.{i}.{p}" for i in layers for p in PROJECTIONS]
    sums = {}

    def hook(key):
        def run(_module, args):
            x = args[0].reshape(-1, args[0].shape[-1]).float()
            h, rows = sums.get(key, (0, 0))
            sums[key] = (h + x.T @ x, rows + x.shape[0])

        return run

    modules = dict(model.named_modules())
    for key in keys:
        modules[key].register_forward_pre_hook(hook(key))
    ids = tokenizer(open(calibration).read(), add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    with torch.no_grad():
        for start in range(0, len(ids), CHUNK):
            chunk = ([bos] if bos is not None else []) + ids[start : start + CHUNK]
            model(torch.tensor([chunk], device="cuda"))
    worst = 0.0
    for key in keys:
        h, rows = sums[key]
        ref = (h / rows).cpu()
        ours = load_file(os.path.join(hessians, f"{key}.safetensors"))["hin"].float()
        err = ((ours - ref).norm() / ref.norm()).item()
        worst = max(worst, err)
        print(f"{key}: relative error {err:.2e} over {rows} rows")
    print(f"worst {worst:.2e}")
    sys.exit(0 if worst < TOLERANCE else 1)


if __name__ == "__main__":
    main()

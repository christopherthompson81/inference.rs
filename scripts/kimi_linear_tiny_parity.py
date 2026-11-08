#!/usr/bin/env python3
"""Parity check of our Kimi-Linear against transformers, on a tiny random checkpoint.

transformers builds a small KimiLinearForCausalLM (KDA layers around two MLA layers without positions, a dense first
layer, routed experts with a shared one) with seeded random weights; the script writes it in Moonshot's own layout
(`q_conv1d`/`k_conv1d`/`v_conv1d`, the forget gate's tensors under `self_attn`, `block_sparse_moe.experts.N.w1/w2/w3`),
reloads that into transformers to check the layout, then both sides score the same token ids: the mean KL(transformers
|| ours) over positions, top-1 agreement and the largest logit difference, then a greedy continuation. Fails past
--max-kl or on a different continuation. Not part of local_ci.sh: it needs transformers 5 with torch, and our library
built with `cargo build -p inference-ffi` (add `--features cuda` for --device cuda:0), found through the checkout's
bindings.

Usage: PYTHONPATH=bindings/python python3 scripts/kimi_linear_tiny_parity.py [--max-kl X] [--device cpu|cuda:0]
           [--dtype f32|bf16] [--keep DIR] [--seed N]
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
FIXTURES = Path(__file__).resolve().parent.parent / "crates/inference/tests/fixtures"
TOKENIZER = FIXTURES / "paddleocr_vl/tiny/tokenizer.json"
VOCAB = 266
GENERATED = 12
# long enough that the KDA state carries over many steps
IDS = [11, 57, 203, 4, 98, 140, 33, 77, 19, 64, 222, 5, 180, 91, 42, 7, 130] * 3
LAYERS = 5
# one-based, as the checkpoint numbers them
FULL_ATTN_LAYERS = [3, 5]
EXPERTS = 4
CONFIG = {
    "architectures": ["KimiLinearForCausalLM"],
    "model_type": "kimi_linear",
    "vocab_size": VOCAB,
    "hidden_size": 128,
    "intermediate_size": 96,
    "moe_intermediate_size": 32,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_experts": EXPERTS,
    "num_local_experts": EXPERTS,
    "num_experts_per_token": 2,
    "num_shared_experts": 1,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "moe_renormalize": True,
    "routed_scaling_factor": 2.5,
    "num_expert_group": 1,
    "topk_group": 1,
    "kv_lora_rank": 512,
    "q_lora_rank": None,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 64,
    "v_head_dim": 16,
    "mla_use_nope": True,
    "rms_norm_eps": 1e-5,
    "model_max_length": 4096,
    "tie_word_embeddings": False,
    "linear_attn_config": {
        "full_attn_layers": FULL_ATTN_LAYERS,
        "kda_layers": [i for i in range(1, LAYERS + 1) if i not in FULL_ATTN_LAYERS],
        # the CUDA kernel runs heads of 64 or 128
        "head_dim": 64,
        "num_heads": 2,
        "short_conv_kernel_size": 4,
    },
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    # wide enough that the logits are peaked
    "initializer_range": 0.1,
}


def moonshot_layout(state: dict) -> dict:
    """transformers' parameters under the names and shapes Moonshot's checkpoint has."""
    out = {}
    for name, tensor in state.items():
        if ".self_attn.conv1d.weight" in name:
            prefix = name[: -len("conv1d.weight")]
            for which, part in zip("qkv", tensor.chunk(3, dim=0)):
                out[f"{prefix}{which}_conv1d.weight"] = part.contiguous()
        elif ".self_attn.forget_gate." in name:
            out[name.replace(".forget_gate.", ".")] = tensor
        elif ".mlp.experts.gate_up_proj" in name:
            prefix = name[: -len("mlp.experts.gate_up_proj")]
            for e, expert in enumerate(tensor):
                gate, up = expert.chunk(2, dim=0)
                out[f"{prefix}block_sparse_moe.experts.{e}.w1.weight"] = gate.contiguous()
                out[f"{prefix}block_sparse_moe.experts.{e}.w3.weight"] = up.contiguous()
        elif ".mlp.experts.down_proj" in name:
            prefix = name[: -len("mlp.experts.down_proj")]
            for e, expert in enumerate(tensor):
                out[f"{prefix}block_sparse_moe.experts.{e}.w2.weight"] = expert.contiguous()
        elif ".mlp.gate." in name or ".mlp.shared_experts." in name:
            out[name.replace(".mlp.", ".block_sparse_moe.")] = tensor
        else:
            out[name] = tensor
    return out


def build(model_dir: Path, seed: int):
    import torch
    from safetensors.torch import save_file
    from transformers import KimiLinearConfig, KimiLinearForCausalLM

    torch.manual_seed(seed)
    config = KimiLinearConfig(**{k: v for k, v in CONFIG.items() if k != "architectures"})
    model = KimiLinearForCausalLM(config).to(torch.float32).eval()
    # random init leaves norms at one, the decay terms and the router bias at zero; perturb them so a dropped one shows
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("norm.weight", "dt_bias", "A_log")):
                param.add_(torch.randn_like(param) * 0.3)
        for name, buffer in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buffer.add_(torch.randn_like(buffer) * 0.1)
    model_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    save_file(moonshot_layout(state), model_dir / "model.safetensors", metadata={"format": "pt"})
    (model_dir / "config.json").write_text(json.dumps(CONFIG))
    shutil.copy(TOKENIZER, model_dir / "tokenizer.json")
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
    )
    # the written layout loads back as the same model
    reloaded = KimiLinearForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        ids = torch.tensor([IDS])
        logits = model(input_ids=ids).logits[0].float()
        again = reloaded(input_ids=ids).logits[0].float()
        if (logits - again).abs().max().item() > 1e-5:
            sys.exit("Moonshot's layout did not reload into transformers as the same model")
        generated = model.generate(
            ids, max_new_tokens=GENERATED, do_sample=False, eos_token_id=None
        )[0, len(IDS) :]
    return logits.tolist(), generated.tolist()


def ours(model_dir: Path, device: str, dtype: str):
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
        scores, logits = engine.prompt_logits(json.dumps({"prompt": IDS, "output": "logits"}))
        choice = json.loads(engine.completion(json.dumps(request)))["choices"][0]
    vocab = json.loads(scores)["vocab_size"]
    rows = [list(logits[i * vocab : (i + 1) * vocab]) for i in range(len(IDS))]
    # greedy, so each step's top alternative is the token it chose, and that entry carries the id
    return rows, [step["top_logprobs"][0]["token"] for step in choice["logprobs"]["content"]]


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
    parser.add_argument("--max-kl", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["f32", "bf16"])
    parser.add_argument("--keep", type=Path, help="write the checkpoint here and keep it")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    dtype = args.dtype or ("f32" if args.device == "cpu" else "bf16")
    max_kl = args.max_kl or (DEFAULT_MAX_KL if dtype == "f32" else BF16_MAX_KL)
    with tempfile.TemporaryDirectory() as scratch:
        model_dir = args.keep or Path(scratch) / "kimi_linear_tiny"
        expected, generated = build(model_dir, args.seed)
        actual, continuation = ours(model_dir, args.device, dtype)
    kl, worst, agree, max_diff = compare(expected, actual)
    print(
        f"{len(IDS)} tokens: mean KL {kl:.2e} (worst {worst:.2e}), top-1 agreement {agree:.3f}, "
        f"max logit diff {max_diff:.4f}"
    )
    print(f"greedy {GENERATED} tokens: transformers {generated}")
    print(f"                    ours         {continuation}")
    if kl > max_kl:
        sys.exit(f"mean KL {kl:.2e} is past {max_kl}")
    if dtype == "f32" and continuation != generated:
        sys.exit("the greedy continuations differ")


if __name__ == "__main__":
    main()

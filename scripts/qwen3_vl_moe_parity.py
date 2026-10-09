#!/usr/bin/env python3
"""One-off parity check of Qwen3-VL-MoE's text stack against transformers, in BF16.

Both sides load the first LAYERS decoder layers of a real checkpoint (HF layout; its files linked into a temporary
directory under a config and index with fewer layers, so the model fits one GPU and only the shards those layers use are
needed), score the same token ids, and the script reports the mean KL(transformers || ours) over positions, top-1
agreement and the largest logit difference. Fails past --max-kl, a coarse gate: telling two norms apart needed the
numbers themselves. Not part of local_ci.sh: it needs a real checkpoint, transformers with torch, and our library built
with `cargo build -p inference-ffi --features cuda`, found through the checkout's bindings.

Usage: PYTHONPATH=bindings/python python3 scripts/qwen3_vl_moe_parity.py <checkpoint-dir>
           [--layers N] [--max-kl X] [--device cuda:0|cpu] [--reference-cache FILE] [--reference-only]
"""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

DEFAULT_LAYERS = 12
DEFAULT_MAX_KL = 0.02
# a fixed English passage; any text works, it only has to be the same on both sides
TEXT = (
    "The lighthouse keeper climbed the spiral stairs every evening at dusk, carrying a can of oil and a cloth. "
    "From the gallery he could see the fishing boats returning, their lamps swaying with the swell, and beyond them "
    "the dark line where the sea met the sky. In winter the storms came from the north, and the waves broke so high "
    "against the rocks that the spray reached the lantern room. He kept a log of every ship that passed, the weather, "
    "the state of the light, and the small repairs he made, so that whoever came after him would know the place."
)


INDEX = "model.safetensors.index.json"
LAYER = re.compile(r"language_model\.layers\.(\d+)\.")


def truncated_checkpoint(checkpoint: Path, layers: int, into: Path) -> Path:
    """The checkpoint's files linked into `into`, under a config and index with only the first `layers` text layers."""
    for entry in checkpoint.iterdir():
        if entry.name not in ("config.json", INDEX) and not entry.name.startswith("."):
            (into / entry.name).symlink_to(entry.resolve())
    config = json.loads((checkpoint / "config.json").read_text())
    config["text_config"]["num_hidden_layers"] = layers
    (into / "config.json").write_text(json.dumps(config))
    index = json.loads((checkpoint / INDEX).read_text())
    kept = {}
    for name, shard in index["weight_map"].items():
        layer = LAYER.search(name)
        if layer and int(layer.group(1)) >= layers:
            continue
        if not (checkpoint / shard).is_file():
            sys.exit(f"{shard} holds {name}, which the first {layers} layers need; download it")
        kept[name] = shard
    (into / INDEX).write_text(json.dumps({"metadata": index.get("metadata", {}), "weight_map": kept}))
    return into


def reference_logits(model_dir: Path, ids):
    import torch
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0].float()
    return logits.tolist()


def our_logits(model_dir: Path, ids, device: str):
    import inference_rs as ir

    spec = {
        "model": {"MultimodalPlain": {"model_id": str(model_dir), "dtype": "bf16"}},
        "runtime": {"device": device},
    }
    with ir.JsonEngine(json.dumps(spec)) as engine:
        scores, logits = engine.prompt_logits(json.dumps({"prompt": ids, "output": "logits"}))
    vocab = json.loads(scores)["vocab_size"]
    return [list(logits[i * vocab : (i + 1) * vocab]) for i in range(len(ids))]


def compare(expected, actual):
    import torch

    e, a = torch.tensor(expected), torch.tensor(actual)
    if e.shape != a.shape:
        sys.exit(f"transformers' logits are {tuple(e.shape)}, ours {tuple(a.shape)}")
    le, la = e.log_softmax(-1), a.log_softmax(-1)
    kl = (le.exp() * (le - la)).sum(-1).mean().item()
    agree = (e.argmax(-1) == a.argmax(-1)).float().mean().item()
    return kl, agree, (e - a).abs().max().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--max-kl", type=float, default=DEFAULT_MAX_KL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-cache", type=Path, help="reuse transformers' logits saved by an earlier run")
    parser.add_argument("--reference-only", action="store_true", help="only compute (and cache) transformers' logits")
    args = parser.parse_args()
    if args.reference_only and not args.reference_cache:
        parser.error("--reference-only needs --reference-cache to keep what it computes")

    from transformers import AutoTokenizer

    ids = AutoTokenizer.from_pretrained(args.checkpoint)(TEXT)["input_ids"]
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = truncated_checkpoint(args.checkpoint, args.layers, Path(tmp))
        cache = args.reference_cache
        key = {"layers": args.layers, "ids": ids}
        if cache and cache.is_file():
            saved = json.loads(cache.read_text())
            if {k: saved.get(k) for k in key} != key:
                sys.exit(f"{cache} holds logits for other layers or tokens; remove it or pick another file")
            expected = saved["logits"]
        else:
            expected = reference_logits(model_dir, ids)
            if cache:
                cache.write_text(json.dumps({**key, "logits": expected}))
        if args.reference_only:
            return
        actual = our_logits(model_dir, ids, args.device)
    kl, agree, max_diff = compare(expected, actual)
    print(
        f"{len(ids)} tokens, {args.layers} layers: mean KL {kl:.5f}, top-1 agreement {agree:.3f}, "
        f"max logit diff {max_diff:.3f}"
    )
    if kl > args.max_kl:
        sys.exit(f"mean KL {kl:.5f} is past {args.max_kl}")


if __name__ == "__main__":
    main()

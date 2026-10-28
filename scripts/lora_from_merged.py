#!/usr/bin/env python3
"""Recover a LoRA adapter from a fine-tune that was published with its adapter merged into the base weights.

`measure BASE TUNED` reads a sample of matching weight matrices (from a local directory, or from a Hugging Face repo by
range requests, without downloading the model), and reports how much of each fine-tuned weight is left when the
difference from the base is cut to a few ranks. A merged LoRA of rank r shows as a drop at r to the rounding floor of
the weights' dtype; a full fine-tune shows no such drop. It also prints what an adapter of the suggested rank costs.

`extract BASE TUNED OUT --rank R` writes a PEFT adapter (`adapter_config.json`, `adapter_model.safetensors`) holding the
rank-R truncation of every changed weight matrix, and reports what the adapter cannot hold: changed norms, biases,
embedding rows and output heads, and each matrix's residual.

`rename ADAPTER BASE OUT --prefix P` rewrites a published adapter whose module names are relative to an inner model
(`layers.0.mlp.down_proj`, from training on the text model alone) to the checkpoint's own names
(`model.language_model.layers.0.mlp.down_proj`), checking that each renamed module is a weight of BASE.

BASE and TUNED are local directories of `*.safetensors` files or Hub repo ids; extract reads local directories only.
"""

import argparse
import itertools
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

DTYPES = {"BF16": 2, "F16": 2, "F32": 4}
SAMPLE_TENSORS = 6
RANKS = (16, 32, 64, 128, 256, 512)
# a merged LoRA's rank is where what is left falls by at least this factor from the measured rank below it
KNEE_DROP = 5.0
# a delta smaller than this, relative to the weight, counts as unchanged
UNCHANGED = 1e-6
# an embedding row counts as changed past this relative change
ROW_THRESHOLD = 1e-3
# matrices that are lookup tables or output heads, which a LoRA on linear layers does not cover
NOT_LINEAR = ("embed_tokens", "lm_head", "embeddings", "wte", "word_embeddings")
ADAPTER_PREFIX = "base_model.model."
HEADER_BYTES = 8


@dataclass
class Entry:
    file: str
    dtype: str
    shape: list
    start: int
    end: int


class Checkpoint:
    """The tensors of a local directory or Hub repo of safetensors files, read one at a time."""

    def __init__(self, source: str):
        self.local = Path(source).is_dir()
        self.source = source
        if self.local:
            self.files = sorted(p.name for p in Path(source).glob("*.safetensors"))
        else:
            from huggingface_hub import HfFileSystem

            self.fs = HfFileSystem()
            self.files = sorted(
                Path(p).name for p in self.fs.glob(f"{source}/*.safetensors")
            )
        if not self.files:
            raise SystemExit(f"{source}: no .safetensors files")
        self.entries = {}
        for name in self.files:
            with self.open(name) as f:
                (n,) = struct.unpack("<Q", f.read(HEADER_BYTES))
                header = json.loads(f.read(n))
            for key, info in header.items():
                if key == "__metadata__":
                    continue
                start, end = info["data_offsets"]
                self.entries[key] = Entry(
                    name,
                    info["dtype"],
                    info["shape"],
                    HEADER_BYTES + n + start,
                    HEADER_BYTES + n + end,
                )

    def open(self, name: str):
        if self.local:
            return open(Path(self.source) / name, "rb")
        return self.fs.open(f"{self.source}/{name}", "rb")

    def nbytes(self) -> int:
        return sum(e.end - e.start for e in self.entries.values())

    def get(self, key: str):
        import torch

        e = self.entries[key]
        with self.open(e.file) as f:
            f.seek(e.start)
            raw = bytearray(f.read(e.end - e.start))
        dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[
            e.dtype
        ]
        return torch.frombuffer(raw, dtype=dtype).reshape(e.shape).float()


def is_linear(key: str, entry: Entry) -> bool:
    return (
        key.endswith(".weight")
        and len(entry.shape) == 2
        and not any(part in key for part in NOT_LINEAR)
    )


def matching(base: Checkpoint, tuned: Checkpoint) -> list:
    """The tuned model's tensors that the base has with the same shape, in name order."""
    keys = [
        k
        for k, e in tuned.entries.items()
        if k in base.entries and base.entries[k].shape == e.shape
    ]
    return sorted(keys)


def residuals(delta, weight_norm: float, ranks) -> dict:
    """What a rank-r truncation of `delta` leaves, relative to the weight, for each r below the matrix's rank (a rank at
    or past it leaves nothing, which says nothing about how the delta was made)."""
    import torch

    s = torch.linalg.svdvals(delta)
    energy = (s.double() ** 2).flip(0).cumsum(0).flip(0)
    # rank 0 leaves the whole delta, the start a fall at the smallest measured rank is seen from
    return {
        r: math.sqrt(energy[r].item()) / weight_norm for r in (0, *ranks) if r < len(s)
    }


def knee(left: dict):
    """The rank after the largest fall between measured ranks, if it falls by `KNEE_DROP` or more; else None."""
    ranks = sorted(left)
    falls = [(left[a] / max(left[b], 1e-30), b) for a, b in itertools.pairwise(ranks)]
    drop, rank = max(falls, default=(0.0, None))
    return rank if drop >= KNEE_DROP else None


def adapter_bytes(base: Checkpoint, tuned: Checkpoint, rank: int) -> int:
    keys = [k for k in matching(base, tuned) if is_linear(k, tuned.entries[k])]
    return sum(rank * sum(tuned.entries[k].shape) * DTYPES["BF16"] for k in keys)


def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def measure(args) -> int:
    base, tuned = Checkpoint(args.base), Checkpoint(args.tuned)
    keys = [k for k in matching(base, tuned) if is_linear(k, tuned.entries[k])]
    if not keys:
        raise SystemExit("no weight matrices match between the two checkpoints")
    step = max(1, len(keys) // args.tensors)
    sample = keys[step // 2 :: step][: args.tensors]
    knees = []
    print(
        f"{len(keys)} matching weight matrices, {len(sample)} sampled; left after rank r, relative to the weight:"
    )
    for key in sample:
        weight = tuned.get(key).to(device())
        delta = weight - base.get(key).to(device())
        ratio = (delta.norm() / weight.norm()).item()
        if ratio < UNCHANGED:
            print(f"  {key}: unchanged")
            continue
        left = residuals(delta, weight.norm().item(), args.ranks)
        if len(left) < 2:
            print(f"  {key} {tuple(weight.shape)}: too narrow for the measured ranks")
            continue
        knees.append(knee(left))
        cells = "  ".join(f"r{r} {v:.4f}" for r, v in left.items() if r > 0)
        print(
            f"  {key} {tuple(weight.shape)}: delta {ratio:.3f} | {cells} | knee {knees[-1]}"
        )
    if not knees:
        print("the sampled matrices are unchanged")
        return 0
    if None in knees:
        print(
            "no sharp drop in every sampled matrix: a full fine-tune, which a low-rank adapter only approximates"
        )
        return 0
    rank = max(knees)
    print(
        f"suggested rank {rank}: what is left falls to the weights' rounding floor there, a merged LoRA"
    )
    print(
        f"an adapter at rank {rank} on every matching matrix: at most {adapter_bytes(base, tuned, rank) / 1e9:.2f} GB in "
        f"bf16 (unchanged ones drop out), against {tuned.nbytes() / 1e9:.2f} GB of fine-tuned weights"
    )
    return 0


def extract(args) -> int:
    import torch
    from safetensors.torch import save_file

    for source in (args.base, args.tuned):
        if not Path(source).is_dir():
            raise SystemExit(
                f"{source}: extract reads local directories (download it first, e.g. `hf download`)"
            )
    base, tuned = Checkpoint(args.base), Checkpoint(args.tuned)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tensors, targets, report = (
        {},
        [],
        {"rank": args.rank, "residual": {}, "left_out": {}, "unmatched": []},
    )
    report["unmatched"] = sorted(set(tuned.entries) - set(matching(base, tuned)))
    for key in matching(base, tuned):
        weight = tuned.get(key).to(device())
        delta = weight - base.get(key).to(device())
        ratio = (delta.norm() / weight.norm().clamp_min(1e-30)).item()
        if ratio < UNCHANGED:
            continue
        entry = tuned.entries[key]
        if not is_linear(key, entry):
            change = {"delta": ratio}
            if len(entry.shape) == 2:
                rows = (
                    delta.norm(dim=1) / weight.norm(dim=1).clamp_min(1e-30)
                ) > ROW_THRESHOLD
                change["rows_changed"] = int(rows.sum().item())
                change["rows"] = entry.shape[0]
            report["left_out"][key] = change
            continue
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        r = min(args.rank, len(s))
        module = key.removesuffix(".weight")
        tensors[f"{ADAPTER_PREFIX}{module}.lora_A.weight"] = (
            vh[:r].contiguous().to(torch.bfloat16).cpu()
        )
        tensors[f"{ADAPTER_PREFIX}{module}.lora_B.weight"] = (
            (u[:, :r] * s[:r]).contiguous().to(torch.bfloat16).cpu()
        )
        targets.append(module)
        left = math.sqrt((s[r:].double() ** 2).sum().item()) / weight.norm().item()
        report["residual"][key] = round(left, 6)
        print(f"  {key}: delta {ratio:.3f}, left {left:.4f}", flush=True)
    if not targets:
        raise SystemExit("no changed weight matrices: nothing to extract")
    save_file(tensors, out / "adapter_model.safetensors")
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": args.base_id or args.base,
        "r": args.rank,
        "lora_alpha": args.rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "use_rslora": False,
        "target_modules": targets,
    }
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "extraction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    worst = max(report["residual"].items(), key=lambda kv: kv[1])
    print(
        f"wrote {len(targets)} matrices at rank {args.rank} to {out}; worst residual {worst[1]:.4f} ({worst[0]})"
    )
    if report["left_out"]:
        print(
            f"the adapter leaves out {len(report['left_out'])} changed tensors (norms, biases, embeddings, heads):"
        )
        for key, change in sorted(
            report["left_out"].items(), key=lambda kv: -kv[1]["delta"]
        )[:10]:
            print(f"  {key}: {change}")
    if report["unmatched"]:
        print(
            f"{len(report['unmatched'])} tuned tensors have no base counterpart (new heads?), e.g. {report['unmatched'][:3]}"
        )
    return 0


def rename(args) -> int:
    from safetensors.torch import load_file, save_file

    adapter, out = Path(args.adapter), Path(args.out)
    base = Checkpoint(args.base)
    tensors = load_file(adapter / "adapter_model.safetensors")
    renamed, missing = {}, set()
    for key, value in tensors.items():
        module = key.removeprefix(ADAPTER_PREFIX).split(".lora_")[0]
        target = args.prefix + module
        if f"{target}.weight" not in base.entries:
            missing.add(target)
        renamed[ADAPTER_PREFIX + args.prefix + key.removeprefix(ADAPTER_PREFIX)] = value
    if missing:
        raise SystemExit(
            f"{len(missing)} renamed modules are not weights of the base, e.g. {sorted(missing)[:3]}"
        )
    out.mkdir(parents=True, exist_ok=True)
    save_file(renamed, out / "adapter_model.safetensors")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if isinstance(config.get("target_modules"), list):
        config["target_modules"] = sorted(
            {k.removeprefix(ADAPTER_PREFIX).split(".lora_")[0] for k in renamed}
        )
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {len(renamed)} tensors under `{args.prefix}` to {out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser(
        "measure", help="estimate the rank of the difference from a sample of matrices"
    )
    m.add_argument("base")
    m.add_argument("tuned")
    m.add_argument(
        "--tensors",
        type=int,
        default=SAMPLE_TENSORS,
        help="how many matrices to sample",
    )
    m.add_argument(
        "--ranks", type=lambda s: [int(r) for r in s.split(",")], default=list(RANKS)
    )
    e = sub.add_parser("extract", help="write a PEFT adapter of the given rank")
    e.add_argument("base")
    e.add_argument("tuned")
    e.add_argument("out")
    e.add_argument("--rank", type=int, required=True)
    e.add_argument(
        "--base-id", help="the base's Hub id, recorded in adapter_config.json"
    )
    n = sub.add_parser(
        "rename",
        help="prefix a published adapter's module names to match the checkpoint",
    )
    n.add_argument("adapter")
    n.add_argument("base")
    n.add_argument("out")
    n.add_argument("--prefix", required=True, help="e.g. model.language_model.")
    args = parser.parse_args(argv)
    return {"measure": measure, "extract": extract, "rename": rename}[args.command](
        args
    )


if __name__ == "__main__":
    sys.exit(main())

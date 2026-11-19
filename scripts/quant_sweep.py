#!/usr/bin/env python3
"""Per-tensor-group quantization sensitivity, and recipes allocated from it (#1013).

`sweep` lowers one group of tensors at a time from a base recipe to each candidate type, quantizing with
`inference quantize gguf` and scoring the KL divergence from a full-precision base with the perplexity example, and
writes a table of bytes saved and KLD added. `allocate` reads the table and picks a type per group: the least summed
KLD within a size, or the smallest size within a KLD (the groups' costs add near-linearly; recipe investigation
Run 17).

Usage:
  scripts/quant_sweep.py sweep --source BF16.gguf --kld-base base.kld --groups groups.json --out sweep.json
      [--imatrix imatrix.dat] [--work DIR] [--jobs N]
  scripts/quant_sweep.py allocate --sweep sweep.json (--max-bytes N | --max-kld X)
  scripts/quant_sweep.py measure --source BF16.gguf --kld-base base.kld --recipes recipes.json --out scores.json

groups.json: {"base": ["regex=type", ...], "groups": {"name": {"pattern": "regex", "types": ["q6_k", ...]}}}; the base
rules come last, so each group's rule overrides them for its tensors.
"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "target", "debug", "inference")
EXAMPLE = os.path.join(ROOT, "target", "debug", "examples", "perplexity")
# the example's summary lines
SCORE = {
    "kld": re.compile(r"^KLD mean\s+(-?[0-9.]+)", re.M),
    "median": re.compile(r"^KLD median\s+(-?[0-9.]+)", re.M),
    "p99": re.compile(r"^KLD p99\s+(-?[0-9.]+)", re.M),
    "same_top": re.compile(r"^same top-1\s+([0-9.]+)%", re.M),
}


def quantize(source, rules, out, imatrix):
    args = [CLI, "quantize", "gguf", "-m", source, "-o", out]
    if imatrix:
        args += ["--imatrix", imatrix]
    for rule in rules:
        args += ["--tensor-type", rule]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.getsize(out)


def score(gguf, kld_base):
    text = subprocess.run(
        [EXAMPLE, "--gguf", gguf, "--file", os.devnull, "--kld-base", kld_base],
        check=True, capture_output=True, text=True,
    ).stdout
    return {k: float(r.search(text).group(1)) for k, r in SCORE.items()}


def run_points(args, points):
    """Every `(name, rules)` point quantized on one worker and scored on `args.jobs` others, so the CPU's quantizing
    overlaps the GPU's scoring and small models score side by side; at most `jobs + 1` quantized files exist at once."""
    slots = threading.BoundedSemaphore(args.jobs + 1)
    results = {}

    def scored(name, out, size):
        try:
            results[name] = {"bytes": size, **score(out, args.kld_base)}
        finally:
            os.remove(out)
            slots.release()
        r = results[name]
        print(f"{name:28s} {size:>14,} B  KLD {r['kld']:.5f}  p99 {r['p99']:.4f}  top {r['same_top']:.2f}%",
              flush=True)

    with ThreadPoolExecutor(1) as quantizing, ThreadPoolExecutor(args.jobs) as scoring:
        pending = []
        for name, rules in points:
            slots.acquire()
            out = os.path.join(args.work, f"{name}.gguf")
            size = quantizing.submit(quantize, args.source, rules, out, args.imatrix).result()
            pending.append(scoring.submit(scored, name, out, size))
        for future in pending:
            future.result()
    return results


def sweep(args):
    spec = json.load(open(args.groups))
    os.makedirs(args.work, exist_ok=True)
    points = [("base", spec["base"])]
    for name, group in spec["groups"].items():
        for ty in group["types"]:
            points.append((f"{name}-{ty}", [f"{group['pattern']}={ty}"] + spec["base"]))
    results = run_points(args, points)
    table = {"base_rules": spec["base"], "base": results["base"], "groups": {}}
    for name, group in spec["groups"].items():
        table["groups"][name] = {
            "pattern": group["pattern"],
            "types": {ty: results[f"{name}-{ty}"] for ty in group["types"]},
        }
    json.dump(table, open(args.out, "w"), indent=1)


def measure(args):
    """Each named recipe in `--recipes` ({"name": ["regex=type", ...]}) quantized and scored, as `sweep` does."""
    recipes = json.load(open(args.recipes))
    os.makedirs(args.work, exist_ok=True)
    json.dump(run_points(args, list(recipes.items())), open(args.out, "w"), indent=1)


def recipe(table, pick):
    """The rules for one type per group (`None` keeps the base), and the size and KLD the sweep predicts for them."""
    base, groups = table["base"], table["groups"]
    size, kld, rules = base["bytes"], base["kld"], []
    for name, ty in pick.items():
        if ty is None:
            continue
        r = groups[name]["types"][ty]
        size -= base["bytes"] - r["bytes"]
        kld += r["kld"] - base["kld"]
        rules.append(f"{groups[name]['pattern']}={ty}")
    return rules + table["base_rules"], size, kld


def allocate(args):
    table = json.load(open(args.sweep))
    base = table["base"]
    groups = table["groups"]
    # each group's options: keep the base (no saving, no cost) or one swept type
    options = []
    for name, group in groups.items():
        opts = [(None, 0, 0.0)]
        for ty, r in group["types"].items():
            opts.append((ty, base["bytes"] - r["bytes"], r["kld"] - base["kld"]))
        options.append((name, opts))
    best = None
    for pick in itertools.product(*(opts for _, opts in options)):
        saved = sum(p[1] for p in pick)
        added = sum(p[2] for p in pick)
        size, kld = base["bytes"] - saved, base["kld"] + added
        if args.max_bytes is not None:
            fits, key = size <= args.max_bytes, kld
        else:
            fits, key = kld <= args.max_kld, size
        if fits and (best is None or key < best[0]):
            best = (key, size, kld, pick)
    if best is None:
        sys.exit("no allocation meets the target")
    _, size, kld, pick = best
    print(f"predicted: {size:,} B, KLD {kld:.5f}")
    rules = []
    for (name, _), (ty, saved, added) in zip(options, pick):
        if ty is not None:
            rules.append(f"{groups[name]['pattern']}={ty}")
            print(f"  {name:20s} {ty:8s} -{saved / 1e6:7.1f} MB  +{added:.5f} KLD")
    rules += table["base_rules"]
    print("rules:")
    for rule in rules:
        print(f"  --tensor-type '{rule}'")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--source", required=True)
    s.add_argument("--kld-base", required=True)
    s.add_argument("--groups", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--imatrix")
    s.add_argument("--work", default=".")
    s.add_argument("--jobs", type=int, default=1, help="models scored at once (as many as fit the GPU)")
    m = sub.add_parser("measure")
    m.add_argument("--source", required=True)
    m.add_argument("--kld-base", required=True)
    m.add_argument("--recipes", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--imatrix")
    m.add_argument("--work", default=".")
    m.add_argument("--jobs", type=int, default=1)
    a = sub.add_parser("allocate")
    a.add_argument("--sweep", required=True)
    target = a.add_mutually_exclusive_group(required=True)
    target.add_argument("--max-bytes", type=int)
    target.add_argument("--max-kld", type=float)
    args = parser.parse_args()
    {"sweep": sweep, "measure": measure, "allocate": allocate}[args.command](args)


if __name__ == "__main__":
    main()

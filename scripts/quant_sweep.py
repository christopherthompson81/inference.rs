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
      [--reuse earlier.gguf ...]

groups.json: {"base": ["regex=type", ...], "groups": {"name": {"pattern": "regex", "types": ["q6_k", ...]}}}; the base
rules come last, so each group's rule overrides them for its tensors.
"""
import argparse
import itertools
import json
import os
import re
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "target", "debug", "inference")
EXAMPLE = os.path.join(ROOT, "target", "debug", "examples", "perplexity")
# GGUF metadata value types with a fixed size (bytes), and the string and array type ids
GGUF_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
GGUF_STRING, GGUF_ARRAY = 8, 9
GGUF_ALIGNMENT = 32
# the example's summary lines
SCORE = {
    "kld": re.compile(r"^KLD mean\s+(-?[0-9.]+)", re.M),
    "median": re.compile(r"^KLD median\s+(-?[0-9.]+)", re.M),
    "p99": re.compile(r"^KLD p99\s+(-?[0-9.]+)", re.M),
    "same_top": re.compile(r"^same top-1\s+([0-9.]+)%", re.M),
}


def tensor_bytes(path):
    """Each tensor's stored bytes in a GGUF (padding included), by name, from its header."""
    with open(path, "rb") as f:
        def read(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

        def string():
            return f.read(read("Q")).decode()

        def skip(ty):
            if ty == GGUF_STRING:
                f.seek(read("Q"), 1)
            elif ty == GGUF_ARRAY:
                inner, n = read("I"), read("Q")
                if inner in GGUF_FIXED:
                    f.seek(GGUF_FIXED[inner] * n, 1)
                else:
                    for _ in range(n):
                        skip(inner)
            else:
                f.seek(GGUF_FIXED[ty], 1)

        f.seek(8)
        n_tensors, n_kv = read("Q"), read("Q")
        for _ in range(n_kv):
            string()
            skip(read("I"))
        offsets = []
        for _ in range(n_tensors):
            name = string()
            f.seek(8 * read("I") + 4, 1)
            offsets.append((read("Q"), name))
        data = -(-f.tell() // GGUF_ALIGNMENT) * GGUF_ALIGNMENT
    ends = [o for o, _ in offsets[1:]] + [os.path.getsize(path) - data]
    return {name: end - start for (start, name), end in zip(offsets, ends)}


def quantize(source, rules, out, imatrix, reuse=(), only=None):
    args = [CLI, "quantize", "gguf", "-m", source, "-o", out]
    if only:
        args += ["--only", only]
    if imatrix:
        args += ["--imatrix", imatrix]
    for path in reuse:
        args += ["--reuse", path]
    for rule in rules:
        args += ["--tensor-type", rule]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.getsize(out)


def score(ggufs, kld_base):
    text = subprocess.run(
        [EXAMPLE, "--gguf", *ggufs, "--file", os.devnull, "--kld-base", kld_base],
        check=True, capture_output=True, text=True,
    ).stdout
    return {k: float(r.search(text).group(1)) for k, r in SCORE.items()}


def run_points(args, points, reuse=(), base=None):
    """Every `(name, rules, only)` point quantized and scored on one of `args.jobs` workers, its file deleted after;
    with `base`, a point with `only` set is written as an overlay of those tensors and scored over the base. The
    quantizer uses only part of the cores, so points overlap."""
    results = {}
    base_sizes = tensor_bytes(base) if base else None

    def point(name, rules, only):
        out = os.path.join(args.work, f"{name}.gguf")
        try:
            size = quantize(args.source, rules, out, args.imatrix, reuse, only if base else None)
            if base and only:
                sizes = tensor_bytes(out)
                size = os.path.getsize(base) + sum(sizes.values()) - sum(base_sizes[n] for n in sizes)
                ggufs = [base, out]
            else:
                ggufs = [out]
            results[name] = r = {"bytes": size, **score(ggufs, args.kld_base)}
        except subprocess.CalledProcessError as e:
            print(f"{name:28s} failed: {e}", flush=True)
            return
        finally:
            if os.path.exists(out):
                os.remove(out)
        print(f"{name:28s} {size:>14,} B  KLD {r['kld']:.5f}  p99 {r['p99']:.4f}  top {r['same_top']:.2f}%",
              flush=True)

    with ThreadPoolExecutor(args.jobs) as pool:
        for future in [pool.submit(point, *p) for p in points]:
            future.result()
    return results


def sweep(args):
    spec = json.load(open(args.groups))
    os.makedirs(args.work, exist_ok=True)
    # the base kept while the points run: each point quantizes only its group, as an overlay scored over the base
    base = os.path.join(args.work, "base.gguf")
    try:
        if args.keep_base and os.path.exists(base):
            size = os.path.getsize(base)
        else:
            size = quantize(args.source, spec["base"], base, args.imatrix)
        results = {"base": {"bytes": size, **score([base], args.kld_base)}}
        print(f"{'base':28s} {size:>14,} B  KLD {results['base']['kld']:.5f}", flush=True)
        points = []
        for name, group in spec["groups"].items():
            for ty in group["types"]:
                rules = [f"{group['pattern']}={ty}"] + spec["base"]
                points.append((f"{name}-{ty}", rules, group["pattern"]))
        results.update(run_points(args, points, base=base))
    finally:
        if os.path.exists(base) and not args.keep_base:
            os.remove(base)
    table = {"base_rules": spec["base"], "base": results["base"], "groups": {}}
    for name, group in spec["groups"].items():
        table["groups"][name] = {
            "pattern": group["pattern"],
            "types": {ty: results[f"{name}-{ty}"] for ty in group["types"] if f"{name}-{ty}" in results},
        }
    json.dump(table, open(args.out, "w"), indent=1)


def measure(args):
    """Each named recipe in `--recipes` ({"name": ["regex=type", ...]}) quantized and scored, as `sweep` does."""
    recipes = json.load(open(args.recipes))
    os.makedirs(args.work, exist_ok=True)
    points = [(name, rules, None) for name, rules in recipes.items()]
    json.dump(run_points(args, points, args.reuse), open(args.out, "w"), indent=1)


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
    s.add_argument("--jobs", type=int, default=1, help="points quantized and scored at once (as many models as fit the GPU)")
    s.add_argument("--keep-base", action="store_true", help="use the work dir's base.gguf if there, and keep it")
    m = sub.add_parser("measure")
    m.add_argument("--source", required=True)
    m.add_argument("--kld-base", required=True)
    m.add_argument("--recipes", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--imatrix")
    m.add_argument("--work", default=".")
    m.add_argument("--jobs", type=int, default=1)
    m.add_argument("--reuse", action="append", default=[], help="earlier outputs to copy matching tensors from")
    a = sub.add_parser("allocate")
    a.add_argument("--sweep", required=True)
    target = a.add_mutually_exclusive_group(required=True)
    target.add_argument("--max-bytes", type=int)
    target.add_argument("--max-kld", type=float)
    args = parser.parse_args()
    {"sweep": sweep, "measure": measure, "allocate": allocate}[args.command](args)


if __name__ == "__main__":
    main()

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
  scripts/quant_sweep.py refine --source BF16.gguf --kld-base a.kld [--kld-base b.kld ...] --spec refine.json
      --max-bytes N --out rules.txt [--hessians DIR] [--work DIR] [--rounds N]

`refine` improves a recipe in place, priced in context (allocation investigation Run 29): each round every group one
step up and down its ladder is quantized as an overlay of the current recipe and all are scored in one streamed pass
(`quantize kld` with INFERENCE_STREAM_CANDIDATES) per base; a move is taken when its paired per-chunk KLD change
beats twice its standard error and the recipe still fits, else the best up/down pairs are measured together.
refine.json: {"rules": ["regex=type", ...], "groups": {"name": {"pattern": "regex", "ladder": ["q5_k", ...],
"type": "q6_k"}}}; the groups' rules come first, then `rules`.

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


def quantize(source, rules, out, imatrix, reuse=(), only=None, hessians=None):
    args = [CLI, "quantize", "gguf", "-m", source, "-o", out]
    if hessians:
        args += ["--hessians", hessians]
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


# the paired change of a candidate from the recipe, chunk by chunk over every base: its mean and standard error
def paired(base_chunks, chunks):
    d = [c - b for b, c in zip(base_chunks, chunks)]
    mean = sum(d) / len(d)
    var = sum((x - mean) ** 2 for x in d) / (len(d) - 1)
    return mean, (var / len(d)) ** 0.5


def score_candidates(base, overlays, kld_bases, work):
    """Each overlay's chunk KLDs over `base` and the recipe's own, every base's chunks in turn, one pass a base."""
    runs = [[] for _ in range(len(overlays) + 1)]
    for i, kld_base in enumerate(kld_bases):
        out = os.path.join(work, f"scores-{i}.json")
        env = dict(os.environ, INFERENCE_STREAM_CANDIDATES=",".join(overlays), INFERENCE_STREAM_SCORES_OUT=out)
        subprocess.run([CLI, "quantize", "kld", "-m", base, "--kld-base", kld_base], check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(out) as f:
            for run, scored in zip(runs, json.load(f)):
                run.extend(scored["chunk_klds"])
    return runs


def refine(args):
    with open(args.spec) as f:
        spec = json.load(f)
    groups = spec["groups"]
    types = {name: g["type"] for name, g in groups.items()}
    os.makedirs(args.work, exist_ok=True)

    def rules(assign):
        return [f"{groups[g]['pattern']}={t}" for g, t in assign.items()] + spec["rules"]

    base = os.path.join(args.work, "recipe.gguf")
    log = []
    # a group's overlay depends on its own type alone (each tensor quantizes alone), so it serves every round
    overlays = {}
    for round_ in range(args.rounds):
        built = os.path.join(args.work, "recipe-next.gguf")
        reuse = [base] if os.path.exists(base) else []
        quantize(args.source, rules(types), built, None, reuse, hessians=args.hessians)
        os.replace(built, base)
        size, held = os.path.getsize(base), tensor_bytes(base)

        def overlay(name, assign, changed, held=held):
            out = os.path.join(args.work, f"{name}.gguf")
            if out not in overlays:
                only = "|".join(f"({groups[g]['pattern']})" for g in changed)
                quantize(args.source, rules(assign), out, None, only=only, hessians=args.hessians)
                overlays[out] = tensor_bytes(out)
            ov = overlays[out]
            return out, sum(ov.values()) - sum(held[n] for n in ov)

        moves = []
        for g, t in types.items():
            ladder = groups[g]["ladder"]
            i = ladder.index(t)
            for j in (i - 1, i + 1):
                if 0 <= j < len(ladder):
                    moves.append({"name": f"{g}={ladder[j]}", "assign": {g: ladder[j]}})
        for m in moves:
            m["path"], m["delta_bytes"] = overlay(m["name"], {**types, **m["assign"]}, m["assign"])
        runs = score_candidates(base, [m["path"] for m in moves], args.kld_base, args.work)
        base_kld = sum(runs[0]) / len(runs[0])
        print(f"round {round_ + 1}: {size:,} B, KLD {base_kld:.5f} over {len(runs[0])} chunks", flush=True)
        for m, run in zip(moves, runs[1:]):
            m["delta"], m["se"] = paired(runs[0], run)
            print(f"  {m['name']:28s} {m['delta_bytes']:>+14,} B  dKLD {m['delta']:+.5f} +/- {m['se']:.5f}", flush=True)
        def fits(m, size=size):
            return size + m["delta_bytes"] <= args.max_bytes

        def better(m):
            return m["delta"] + 2 * m["se"] < 0

        take = None
        if size > args.max_bytes:
            down = [m for m in moves if m["delta_bytes"] < 0]
            take = min(down, key=lambda m: max(m["delta"], 0) / -m["delta_bytes"]) if down else None
        else:
            singles = [m for m in moves if fits(m) and better(m)]
            if singles:
                take = min(singles, key=lambda m: m["delta"])
            else:
                ups = [m for m in moves if m["delta_bytes"] > 0]
                downs = [m for m in moves if m["delta_bytes"] < 0]
                pairs = sorted(
                    (u["delta"] + d["delta"], u, d)
                    for u, d in itertools.product(ups, downs)
                    if list(u["assign"]) != list(d["assign"]) and size + u["delta_bytes"] + d["delta_bytes"] <= args.max_bytes
                    and u["delta"] + d["delta"] < 0
                )[: args.pairs]
                joint = []
                for _, u, d in pairs:
                    assign = {**u["assign"], **d["assign"]}
                    name = f"{u['name']}+{d['name']}"
                    path, delta_bytes = overlay(name.replace("+", "_"), {**types, **assign}, assign)
                    joint.append({"name": name, "assign": assign, "path": path, "delta_bytes": delta_bytes})
                if joint:
                    runs = score_candidates(base, [m["path"] for m in joint], args.kld_base, args.work)
                    for m, run in zip(joint, runs[1:]):
                        m["delta"], m["se"] = paired(runs[0], run)
                        print(f"  {m['name']:28s} {m['delta_bytes']:>+14,} B  dKLD {m['delta']:+.5f} +/- {m['se']:.5f}",
                              flush=True)
                    good = [m for m in joint if better(m)]
                    take = min(good, key=lambda m: m["delta"]) if good else None
        log.append({"round": round_ + 1, "bytes": size, "kld": base_kld,
                    "moves": [{k: m[k] for k in ("name", "delta_bytes", "delta", "se")} for m in moves],
                    "taken": take and take["name"]})
        with open(args.out + ".json", "w") as f:
            json.dump({"types": types, "log": log}, f, indent=1)
        if not take:
            print("  nothing beats the noise: done", flush=True)
            break
        print(f"  take {take['name']}", flush=True)
        types.update(take["assign"])
    for path in overlays:
        os.remove(path)
    with open(args.out, "w") as f:
        f.write("\n".join(rules(types)) + "\n")
    print(f"wrote {args.out}", flush=True)


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
    r = sub.add_parser("refine")
    r.add_argument("--source", required=True)
    r.add_argument("--kld-base", required=True, action="append")
    r.add_argument("--spec", required=True)
    r.add_argument("--max-bytes", required=True, type=int)
    r.add_argument("--out", required=True)
    r.add_argument("--hessians")
    r.add_argument("--work", default=".")
    r.add_argument("--rounds", type=int, default=8)
    r.add_argument("--pairs", type=int, default=3, help="up/down pairs measured jointly when no single move helps")
    a = sub.add_parser("allocate")
    a.add_argument("--sweep", required=True)
    target = a.add_mutually_exclusive_group(required=True)
    target.add_argument("--max-bytes", type=int)
    target.add_argument("--max-kld", type=float)
    args = parser.parse_args()
    {"sweep": sweep, "measure": measure, "allocate": allocate, "refine": refine}[args.command](args)


if __name__ == "__main__":
    main()

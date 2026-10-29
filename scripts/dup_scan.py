"""Duplicated code across Rust sources: windows of normalized lines that occur in more than one place.

`dup_scan.py` prints the total, the per-crate split and the file pairs sharing the most windows;
`dup_scan.py --pair A B` lists the spans two files share, as line ranges in each.
"""

import argparse
import collections
import hashlib
import itertools
import os
import re
import sys
from pathlib import Path

WINDOW = 12
TOP = 40
SKIP_DIRS = {"target", "third_party", ".git", "node_modules", "static"}
TRIVIAL = re.compile(r"^[\s{}()\[\];,]*$")
# A window shared by more files than this is boilerplate (imports, derives); its pairs would swamp the listing.
MAX_PAIR_FANOUT = 8


def normalize(text):
    lines = []
    for i, raw in enumerate(text.splitlines()):
        s = raw.strip()
        if not s or s.startswith("//") or TRIVIAL.match(s):
            continue
        lines.append((i + 1, re.sub(r"\s+", " ", s)))
    return lines


def rust_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in sorted(filenames):
            if f.endswith(".rs"):
                yield os.path.join(dirpath, f)


def window_index(norm, window):
    index = collections.defaultdict(list)
    for path, lines in norm.items():
        for k in range(len(lines) - window + 1):
            key = "\n".join(s for _, s in lines[k : k + window]).encode()
            index[hashlib.blake2b(key, digest_size=10).digest()].append((path, k))
    return index


def scan(norm, window):
    """Duplicated normalized lines per file, and windows shared per cross-file pair."""
    dup = collections.defaultdict(set)
    pairs = collections.Counter()
    for occ in window_index(norm, window).values():
        if len(occ) < 2:
            continue
        for path, k in occ:
            dup[path].update(range(k, k + window))
        paths = sorted({p for p, _ in occ})
        if 1 < len(paths) <= MAX_PAIR_FANOUT:
            for pair in itertools.combinations(paths, 2):
                pairs[pair] += 1
    return dup, pairs


def shared_spans(a_lines, b_lines, window):
    """Line ranges ((a_start, a_end), (b_start, b_end)) of the runs of windows `a` and `b` share."""
    index = window_index({"a": a_lines, "b": b_lines}, window)
    starts = set()
    for occ in index.values():
        ka = [k for p, k in occ if p == "a"]
        kb = [k for p, k in occ if p == "b"]
        starts.update(itertools.product(ka, kb))
    spans = []
    for ka, kb in sorted(starts):
        if (ka - 1, kb - 1) in starts:
            continue
        n = 1
        while (ka + n, kb + n) in starts:
            n += 1
        last = n - 1 + window - 1
        spans.append(
            (
                (a_lines[ka][0], a_lines[ka + last][0]),
                (b_lines[kb][0], b_lines[kb + last][0]),
            )
        )
    return spans


def crate_of(rel):
    parts = rel.split(os.sep)
    return parts[1] if parts[0] == "crates" and len(parts) > 1 else parts[0]


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default="crates")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--top", type=int, default=TOP)
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"))
    args = ap.parse_args(argv)

    if args.pair:
        a, b = (normalize(Path(p).read_text(errors="replace")) for p in args.pair)
        spans = shared_spans(a, b, args.window)
        for (a0, a1), (b0, b1) in spans:
            print(
                f"{args.pair[0]}:{a0}-{a1}  {args.pair[1]}:{b0}-{b1}  ({a1 - a0 + 1} lines)"
            )
        print(f"{len(spans)} shared spans")
        return

    norm = {
        p: normalize(Path(p).read_text(errors="replace")) for p in rust_files(args.root)
    }
    dup, pairs = scan(norm, args.window)
    rel = lambda p: os.path.relpath(p, args.root)
    print(
        f"files {len(norm)}, lines {sum(map(len, norm.values()))} normalized, {sum(map(len, dup.values()))} duplicated"
    )
    by_crate = collections.Counter()
    for p, v in dup.items():
        by_crate[
            crate_of(os.path.join(os.path.basename(os.path.abspath(args.root)), rel(p)))
        ] += len(v)
    print("\nduplicated lines by crate:")
    for c, n in by_crate.most_common(20):
        print(f"  {n:6d}  {c}")
    print(f"\ncross-file pairs sharing the most {args.window}-line windows:")
    for (a, b), n in pairs.most_common(args.top):
        print(f"  {n:5d}  {rel(a)}  <->  {rel(b)}")


if __name__ == "__main__":
    main(sys.argv[1:])

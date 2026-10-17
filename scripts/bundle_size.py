#!/usr/bin/env python3
"""Compare a built `libinference_ffi` and its CUDA kernel libraries with the committed section sizes (Linux).

Usage: bundle_size.py LIBRARY COMPUTE_CAP CUDA_VERSION [--update]. `file` is the total over the library and its kernel
libraries, `file:<name>` each one, measured stripped as the wheels ship them. `copies` is the library's code in
functions compiled more than once under one name, most often a generic instantiated again in each crate that calls it.
Exits 1 when the total, a tracked section or `copies` grew past the limit, 0 otherwise; a build for another compute
capability or toolkit is only reported. `--update` rewrites the baseline.
"""

import collections
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BASELINE = Path(__file__).with_name("bundle_size_baseline.json")
FILE = "file"
COPIES = "copies"
# How many of the most-copied functions a growing `copies` lists.
COPIES_SHOWN = 10
TEXT_SYMBOL = re.compile(r"^[0-9a-f]+ ([0-9a-f]+) [tTwW] (.+)$")
# Sections smaller than this on both sides move with unrelated noise, so they are kept but not compared.
TRACKED_MIN = 1 << 20
# A change counts only past both limits, so a small section's jitter and a large one's few KiB both pass.
CHANGE_FRACTION = 0.01
CHANGE_BYTES = 256 << 10
MIB = 1 << 20

sys.path.insert(0, str(Path(__file__).parent / "release"))
from kernel_libraries import MissingKernels, built_kernels


def sections(size_output: str) -> dict[str, int]:
    """Section name to bytes, from `size -A` output."""
    out = {}
    for line in size_output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith(".") and parts[1].isdigit():
            out[parts[0]] = int(parts[1])
    return out


def copies(library: Path) -> collections.Counter[str] | None:
    """Bytes past the first copy, per function name compiled more than once; None when the library has no symbols."""
    out = subprocess.run(
        ["nm", "--print-size", "--size-sort", "--demangle", str(library)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    sizes: dict[str, list[int]] = collections.defaultdict(list)
    for line in out.splitlines():
        if m := TEXT_SYMBOL.match(line):
            sizes[m.group(2)].append(int(m.group(1), 16))
    if not sizes:
        return None
    return collections.Counter({name: sum(s) - max(s) for name, s in sizes.items() if len(s) > 1})


def stripped(library: Path, into: Path) -> Path:
    out = into / library.name
    subprocess.run(["strip", "--strip-all", "-o", str(out), str(library)], check=True)
    return out


def compared(baseline: dict[str, int], current: dict[str, int]) -> list[str]:
    """The file and every section of at least `TRACKED_MIN` bytes on either side, largest first."""
    names = set(baseline) | set(current)
    big = [n for n in names if n == FILE or max(baseline.get(n, 0), current.get(n, 0)) >= TRACKED_MIN]
    return sorted(big, key=lambda n: -max(baseline.get(n, 0), current.get(n, 0)))


def past_limits(before: int, after: int) -> bool:
    change = abs(after - before)
    return change > CHANGE_BYTES and change > max(before, after) * CHANGE_FRACTION


def regressions(baseline: dict[str, int], current: dict[str, int]) -> list[str]:
    """Compared names that grew past both limits (a new section from 0); one that shrank or went away passes."""
    return [
        n
        for n in compared(baseline, current)
        if current.get(n, 0) > baseline.get(n, 0) and past_limits(baseline.get(n, 0), current.get(n, 0))
    ]


def shrinks(baseline: dict[str, int], current: dict[str, int]) -> list[str]:
    """Compared names that shrank past both limits, so lowering the baseline would keep the gain."""
    return [
        n
        for n in compared(baseline, current)
        if current.get(n, 0) < baseline.get(n, 0) and past_limits(baseline.get(n, 0), current.get(n, 0))
    ]


def report(baseline: dict[str, int], current: dict[str, int]) -> str:
    rows = [f"{'':18} {'baseline':>12} {'now':>12} {'change':>10}"]
    for name in compared(baseline, current):
        before, after = baseline.get(name, 0), current.get(name, 0)
        rows.append(f"{name:18} {before / MIB:10.2f}Mi {after / MIB:10.2f}Mi {(after - before) / MIB:+8.2f}Mi")
    return "\n".join(rows)


def build_key(compute_cap: str, cuda_version: str) -> dict[str, str]:
    # inference-kernel-build's list: 86, 8.6, sm_90 or 90a, split on commas, semicolons or spaces, in any order
    archs = {int(re.sub(r"\D", "", arch)) for arch in re.split(r"[,;\s]+", compute_cap) if arch}
    return {"compute_cap": ",".join(map(str, sorted(archs))), "cuda": cuda_version}


def main(argv: list[str]) -> int:
    library, key = Path(argv[1]), build_key(argv[2], argv[3])
    if not BASELINE.exists() and "--update" not in argv:
        print(f"no {BASELINE.name}: run local_ci.sh --size-update first", file=sys.stderr)
        return 2
    try:
        files = [library] + [kernel.path for kernel in built_kernels(library)]
    except MissingKernels as error:
        print(error, file=sys.stderr)
        return 2
    scratch = tempfile.TemporaryDirectory()
    files[0] = stripped(library, Path(scratch.name))
    # what ships: the sections summed over every file, the total, and each library summed over its archs
    sizes: dict[str, int] = {}
    for path in files:
        out = subprocess.run(["size", "-A", str(path)], check=True, capture_output=True, text=True).stdout
        for name, size in sections(out).items():
            sizes[name] = sizes.get(name, 0) + size
        file_key = f"{FILE}:{path.name}"
        sizes[file_key] = sizes.get(file_key, 0) + path.stat().st_size
    sizes[FILE] = sum(path.stat().st_size for path in files)
    copied = copies(library)
    if copied is None:
        print(f"{library.name} has no symbols, so {COPIES} is not measured (the bundle profile keeps them)")
    else:
        sizes[COPIES] = sum(copied.values())
    if "--update" in argv:
        BASELINE.write_text(json.dumps({**key, "sizes": sizes}, indent=2, sort_keys=True) + "\n")
        print(f"wrote {BASELINE}")
        return 0
    saved = json.loads(BASELINE.read_text())
    if copied is None:
        saved["sizes"].pop(COPIES, None)
    print(report(saved["sizes"], sizes))
    saved_key = {name: saved.get(name) for name in key}
    if saved_key != key:
        print(f"baseline is for {saved_key}, this build is {key}: not compared")
        return 0
    grown = regressions(saved["sizes"], sizes)
    if COPIES in grown and copied is not None:
        print("most-copied functions (bytes past the first copy):", file=sys.stderr)
        for name, extra in copied.most_common(COPIES_SHOWN):
            print(f"  {extra >> 10:6} KiB  {name[:150]}", file=sys.stderr)
    if grown:
        print(f"grew past {CHANGE_FRACTION:.0%} and {CHANGE_BYTES >> 10} KiB: {', '.join(grown)}", file=sys.stderr)
        print("if intended, rerun with --size-update and commit the baseline", file=sys.stderr)
        return 1
    if shrunk := shrinks(saved["sizes"], sizes):
        print(f"shrank past the limits: {', '.join(shrunk)}; --size-update keeps the gain")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

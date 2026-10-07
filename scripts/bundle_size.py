#!/usr/bin/env python3
"""Compare a built `libinference_ffi` with the committed section sizes; `local_ci.sh --size` runs it (Linux).

Usage: bundle_size.py LIBRARY COMPUTE_CAP CUDA_VERSION [--update]. Exits 1 when the file or a tracked section grew past
the limit, 0 otherwise; a build for another compute capability or toolkit is only reported. `--update` rewrites the
baseline.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

BASELINE = Path(__file__).with_name("bundle_size_baseline.json")
FILE = "file"
# Sections smaller than this on both sides move with unrelated noise, so they are kept but not compared.
TRACKED_MIN = 1 << 20
# A change counts only past both limits, so a small section's jitter and a large one's few KiB both pass.
CHANGE_FRACTION = 0.01
CHANGE_BYTES = 256 << 10
MIB = 1 << 20


def sections(size_output: str) -> dict[str, int]:
    """Section name to bytes, from `size -A` output."""
    out = {}
    for line in size_output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith(".") and parts[1].isdigit():
            out[parts[0]] = int(parts[1])
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
    sizes = sections(subprocess.run(["size", "-A", str(library)], check=True, capture_output=True, text=True).stdout)
    sizes[FILE] = library.stat().st_size
    if "--update" in argv:
        BASELINE.write_text(json.dumps({**key, "sizes": sizes}, indent=2, sort_keys=True) + "\n")
        print(f"wrote {BASELINE}")
        return 0
    saved = json.loads(BASELINE.read_text())
    print(report(saved["sizes"], sizes))
    saved_key = {name: saved.get(name) for name in key}
    if saved_key != key:
        print(f"baseline is for {saved_key}, this build is {key}: not compared")
        return 0
    grown = regressions(saved["sizes"], sizes)
    if grown:
        print(f"grew past {CHANGE_FRACTION:.0%} and {CHANGE_BYTES >> 10} KiB: {', '.join(grown)}", file=sys.stderr)
        print("if intended, rerun with --size-update and commit the baseline", file=sys.stderr)
        return 1
    if shrunk := shrinks(saved["sizes"], sizes):
        print(f"shrank past the limits: {', '.join(shrunk)}; --size-update keeps the gain")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
"""Delete target-dir artifacts no replayed build uses (cargo never removes stale variants itself).

stdin is `cargo ... --message-format=json` from a replay of every mode to keep. Incremental dir names carry a hash
cargo does not report, so each crate keeps its newest N, N being its live unit count.
"""

import argparse
import collections
import json
import re
import shutil
import sys
from pathlib import Path

HASH = re.compile(r"-([0-9a-f]{16})(?:[.-]|$)")


def artifact_hash(name):
    m = HASH.search(name)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = args.profile_dir.resolve()

    live_hashes, live_inodes, live_dirs, live_units = set(), set(), set(), set()
    for line in sys.stdin:
        if not line.startswith("{"):
            continue
        msg = json.loads(line)
        if msg.get("reason") == "compiler-artifact":
            for f in msg["filenames"] + [msg.get("executable") or ""]:
                if h := artifact_hash(Path(f).name):
                    live_hashes.add(h)
                elif f and Path(f).exists():
                    # uplifted bins are hardlinks to the hashed original in deps/ or examples/
                    live_inodes.add(Path(f).stat().st_ino)
                if "/build/" in f:
                    live_dirs.add(Path(f).parent.resolve())
            if msg["package_id"].startswith("path+"):
                # a unit shared by several replays is reported by each of them
                live_units.add((msg["target"]["name"].replace("-", "_"), tuple(sorted(msg["filenames"]))))
        elif msg.get("reason") == "build-script-executed":
            live_dirs.add(Path(msg["out_dir"]).parent.resolve())
            for p in msg["linked_paths"]:
                live_dirs.add(Path(p.split("=", 1)[-1]).resolve())
    if not live_hashes:
        sys.exit("sweep: no compiler-artifact messages on stdin, refusing to delete everything")

    for sub in ("deps", "examples"):
        for f in (root / sub).glob("*"):
            if f.is_file() and f.stat().st_ino in live_inodes and (h := artifact_hash(f.name)):
                live_hashes.add(h)

    stale = []
    for sub in ("deps", "examples"):
        for f in (root / sub).glob("*"):
            h = artifact_hash(f.name)
            if f.is_file() and h and h not in live_hashes:
                stale.append(f)
    for sub in ("build", "cuda-kernels"):
        stale += [d for d in (root / sub).glob("*") if d.is_dir() and d.resolve() not in live_dirs]
    units_per_crate = collections.Counter(crate for crate, _ in live_units)
    by_crate = collections.defaultdict(list)
    for d in (root / "incremental").glob("*"):
        by_crate[d.name.rsplit("-", 1)[0]].append(d)
    for crate, dirs in by_crate.items():
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        stale += dirs[units_per_crate[crate]:]
        # rustc loads only the newest session and deletes older ones at the start of the next compile
        for d in dirs[: units_per_crate[crate]]:
            sessions = sorted((s for s in d.glob("s-*") if s.is_dir()), key=lambda s: s.stat().st_mtime)
            for old in sessions[:-1]:
                stale.append(old)
                lock = d / (old.name.rsplit("-", 1)[0] + ".lock")
                if lock.exists():
                    stale.append(lock)

    freed = 0
    for p in stale:
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
        freed += size
        if args.dry_run:
            print(f"{size / 2**20:9.1f} MiB  {p.relative_to(root)}")
        else:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    verb = "would free" if args.dry_run else "freed"
    print(f"sweep: {verb} {freed / 2**30:.1f} GiB in {len(stale)} entries under {root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Delete target-dir artifacts no replayed build uses (cargo never removes stale variants itself).

stdin is `cargo ... --message-format=json` from a replay of every mode to keep. Incremental dir names carry a hash
cargo does not report, so each crate keeps its newest N, N being its live unit count.

`--superseded` instead needs no replay: it deletes what live units no longer read. The split debug info (`.dwo`) of
codegen units no live rlib or linked binary still names (an incremental rebuild renames a unit's codegen units under the
same hash, and rustc never removes the old ones' `.dwo`), incremental sessions older than a dir's newest (rustc removes
them only when that unit next compiles), and temp dirs an interrupted rustc left.
"""

import argparse
import collections
import fcntl
import json
import re
import shutil
import struct
import sys
from pathlib import Path

HASH = re.compile(r"-([0-9a-f]{16})(?:[.-]|$)")
DWO_SUFFIX = b".rcgu.dwo"
ELF_MAGIC = b"\x7fELF"
AR_MAGIC = b"!<arch>\n"
AR_HEADER = 60


def artifact_hash(name):
    m = HASH.search(name)
    return m.group(1) if m else None


def rlib_members(path):
    """The member names of an ar archive, GNU long names resolved."""
    data = path.read_bytes()
    if not data.startswith(AR_MAGIC):
        return []
    names, long_names, at = [], b"", len(AR_MAGIC)
    while at + AR_HEADER <= len(data):
        header = data[at : at + AR_HEADER]
        name, size = header[:16].rstrip(), int(header[48:58])
        body = data[at + AR_HEADER : at + AR_HEADER + size]
        if name == b"//":
            long_names = body
        elif name.startswith(b"/") and name[1:].isdigit():
            start = int(name[1:])
            names.append(long_names[start : long_names.index(b"/\n", start)].decode())
        elif name not in (b"/", b"/SYM64/"):
            names.append(name.rstrip(b"/").decode())
        at += AR_HEADER + size + (size & 1)
    return names


def elf_dwo_names(path):
    """The `.dwo` names in an ELF file's string sections, where its skeleton units keep them."""
    with open(path, "rb") as f:
        head = f.read(64)
        if not head.startswith(ELF_MAGIC) or head[4] != 2:
            return set()
        shoff, (shentsize, shnum, shstrndx) = (
            int.from_bytes(head[40:48], "little"),
            struct.unpack("<HHH", head[58:64]),
        )
        f.seek(shoff)
        table = f.read(shentsize * shnum)
        sections = [
            struct.unpack("<IIQQQQ", table[i * shentsize : i * shentsize + 40])
            for i in range(shnum)
        ]
        f.seek(sections[shstrndx][4])
        section_names = f.read(sections[shstrndx][5])
        found = set()
        for name_at, _, _, _, offset, size in sections:
            name = section_names[name_at : section_names.index(b"\0", name_at)]
            if name.startswith(b".debug_str") or name.startswith(b".debug_line_str"):
                f.seek(offset)
                strings = f.read(size)
                # a fixed suffix finds the names far faster than a regex over every symbol string
                at = strings.find(DWO_SUFFIX)
                while at >= 0:
                    start = strings.rfind(b"\0", 0, at) + 1
                    found.add(
                        strings[start : at + len(DWO_SUFFIX)]
                        .rsplit(b"/", 1)[-1]
                        .decode()
                    )
                    at = strings.find(DWO_SUFFIX, at + len(DWO_SUFFIX))
        return found


def superseded_sessions(dirs):
    """Each incremental dir's sessions but its newest finalized one, and their locks: rustc loads only that one. Under
    the cargo lock no session is being written, so one still named `-working` was interrupted."""
    stale = []
    for d in dirs:
        sessions = sorted(
            (s for s in d.glob("s-*") if s.is_dir()), key=lambda s: s.stat().st_mtime
        )
        finalized = [s for s in sessions if not s.name.endswith("-working")]
        for old in (s for s in sessions if not finalized or s != finalized[-1]):
            stale.append(old)
            lock = d / (old.name.rsplit("-", 1)[0] + ".lock")
            if lock.exists():
                stale.append(lock)
    return stale


def superseded(root):
    stale = []
    for sub in ("deps", "examples"):
        files = [f for f in (root / sub).glob("*") if f.is_file()]
        live = set()
        for f in files:
            if f.suffix == ".rlib":
                live.update(
                    m[: -len(".o")] + ".dwo"
                    for m in rlib_members(f)
                    if m.endswith(".rcgu.o")
                )
            elif f.suffix == ".so" or (not f.suffix and f.stat().st_mode & 0o100):
                live |= elf_dwo_names(f)
        stale += [f for f in files if f.suffix == ".dwo" and f.name not in live]
    # temp dirs an interrupted rustc left behind
    stale += [d for d in (root / "deps").glob("rmeta*") if d.is_dir()]
    return stale + superseded_sessions(
        d for d in (root / "incremental").glob("*") if d.is_dir()
    )


def delete(stale, root, dry_run):
    freed = 0
    for p in stale:
        size = (
            sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            if p.is_dir()
            else p.stat().st_size
        )
        freed += size
        if dry_run:
            print(f"{size / 2**20:9.1f} MiB  {p.relative_to(root)}")
        else:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    verb = "would free" if dry_run else "freed"
    print(f"sweep: {verb} {freed / 2**30:.1f} GiB in {len(stale)} entries under {root}")


def lock_target(root):
    # cargo holds this for every build, so nothing can write the dir while it is swept
    lock = open(root / ".cargo-lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(
            f"sweep: {root} is in use by another cargo process (rust-analyzer?), not sweeping"
        )
    return lock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--superseded",
        action="store_true",
        help="only what live units no longer read, no replay",
    )
    args = ap.parse_args()
    root = args.profile_dir.resolve()
    if args.superseded:
        lock = lock_target(root)
        delete(superseded(root), root, args.dry_run)
        return

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
                live_units.add(
                    (
                        msg["target"]["name"].replace("-", "_"),
                        tuple(sorted(msg["filenames"])),
                    )
                )
        elif msg.get("reason") == "build-script-executed":
            live_dirs.add(Path(msg["out_dir"]).parent.resolve())
            for p in msg["linked_paths"]:
                live_dirs.add(Path(p.split("=", 1)[-1]).resolve())
    if not live_hashes:
        sys.exit(
            "sweep: no compiler-artifact messages on stdin, refusing to delete everything"
        )

    for sub in ("deps", "examples"):
        for f in (root / sub).glob("*"):
            if (
                f.is_file()
                and f.stat().st_ino in live_inodes
                and (h := artifact_hash(f.name))
            ):
                live_hashes.add(h)

    lock = lock_target(root)

    stale = []
    for sub in ("deps", "examples"):
        for f in (root / sub).glob("*"):
            h = artifact_hash(f.name)
            if f.is_file() and h and h not in live_hashes:
                stale.append(f)
    for sub in ("build", "cuda-kernels"):
        stale += [
            d
            for d in (root / sub).glob("*")
            if d.is_dir() and d.resolve() not in live_dirs
        ]
    units_per_crate = collections.Counter(crate for crate, _ in live_units)
    by_crate = collections.defaultdict(list)
    for d in (root / "incremental").glob("*"):
        by_crate[d.name.rsplit("-", 1)[0]].append(d)
    for crate, dirs in by_crate.items():
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        stale += dirs[units_per_crate[crate] :]

    # the kept incremental dirs' old sessions are part of the superseded pass; a dropped dir goes whole
    dropped = set(stale)
    stale += [
        p
        for p in superseded(root)
        if not any(parent in dropped for parent in p.parents)
    ]
    delete(stale, root, args.dry_run)


if __name__ == "__main__":
    main()

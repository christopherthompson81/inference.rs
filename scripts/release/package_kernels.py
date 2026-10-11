#!/usr/bin/env python3
"""Package the CUDA kernel libraries of a release build per GPU arch, with a checksummed manifest.

Usage:
    python scripts/release/package_kernels.py target/bundle/libinference_ffi.so
    python scripts/release/package_kernels.py LIBRARY --out target/kernels

Writes `<out>/<version>/sm<arch>/lib<name>.so` for every arch the build linked (a dir `INFERENCE_KERNEL_DIR` can point
at), `<out>/<version>/kernels-cu<major>.json` describing each file, and `<out>/<version>/assets/`: the same files
under flat names (`cu12-sm86-libinferencequant_q4_k.so`) plus the manifest, ready to attach to the GitHub release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import tomllib

sys.path.insert(0, str(Path(__file__).parent))
from kernel_libraries import (
    ARCH_DIR_PREFIX,
    CHECKOUT_KERNELS,
    KernelFile,
    MissingKernels,
    built_kernels,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "target" / "kernels"
ASSETS_DIR = "assets"
MANIFEST_SCHEMA = 1
CUDART_NEEDED = re.compile(rb"libcudart\.so\.(\d+)")
HASH_CHUNK = 1 << 20


def workspace_version() -> str:
    with (REPO_ROOT / "Cargo.toml").open("rb") as manifest:
        return tomllib.load(manifest)["workspace"]["package"]["version"]


def cuda_major(kernels: list[KernelFile]) -> str:
    for kernel in kernels:
        found = CUDART_NEEDED.search(kernel.path.read_bytes())
        if found:
            return found.group(1).decode()
    sys.exit("no kernel library names the CUDA runtime it links")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def asset_name(cuda: str, kernel: KernelFile) -> str:
    return f"cu{cuda}-{ARCH_DIR_PREFIX}{kernel.arch}-{kernel.path.name}"


def manifest_name(cuda: str) -> str:
    return f"kernels-cu{cuda}.json"


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # the tree and the assets hold the same bytes; a link saves a second copy where the filesystem allows one
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def package(library: Path, out: Path) -> Path:
    if CHECKOUT_KERNELS in library.read_bytes():
        sys.exit("the library loads its CUDA kernels from this checkout (a dev build); package a release build")
    try:
        kernels = built_kernels(library)
    except MissingKernels as error:
        sys.exit(str(error))
    if not kernels:
        sys.exit(f"{library.name} was built without CUDA kernel libraries")
    version, cuda = workspace_version(), cuda_major(kernels)
    root = out / version
    if root.exists():
        shutil.rmtree(root)
    files = []
    for kernel in sorted(kernels, key=lambda k: (k.arch, k.library)):
        tree = root / kernel.relative
        tree.parent.mkdir(parents=True, exist_ok=True)
        # a copy, never a link into the build cache, which a later edit of the package would change
        shutil.copy2(kernel.path, tree)
        link_or_copy(tree, root / ASSETS_DIR / asset_name(cuda, kernel))
        files.append(
            {
                "library": kernel.library,
                "arch": kernel.arch,
                "path": kernel.relative,
                "asset": asset_name(cuda, kernel),
                "sha256": sha256(tree),
                "size": tree.stat().st_size,
            }
        )
    libraries = {
        kernel.library: {"build_id": kernel.build_id, "per_type": kernel.per_type}
        for kernel in sorted(kernels, key=lambda k: k.library)
    }
    manifest = {"schema": MANIFEST_SCHEMA, "version": version, "cuda": cuda, "libraries": libraries, "files": files}
    text = json.dumps(manifest, indent=2) + "\n"
    (root / manifest_name(cuda)).write_text(text)
    (root / ASSETS_DIR / manifest_name(cuda)).write_text(text)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", type=Path, help="the built libinference_ffi whose kernel libraries to package")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(package(args.library.resolve(), args.out.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Print `cargo doc` arguments for the crates whose files are on stdin, with the features the workspace build gives them.

A changed workspace manifest or lockfile means `--workspace`; paths outside every crate document nothing.
"""

import json
import subprocess
import sys
from pathlib import Path

WORKSPACE_WIDE = {"Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml"}


def doc_args(paths: list[str], crates: list[tuple[Path, str]], features: dict[str, list[str]]) -> str:
    """Arguments for `paths`, given each crate as (directory, name) and the workspace-resolved features per crate."""
    by_depth = sorted(crates, key=lambda crate: len(crate[0].parts), reverse=True)
    packages = set()
    for line in paths:
        if not line.strip():
            continue
        path = Path(line.strip())
        if str(path) in WORKSPACE_WIDE:
            return "--workspace"
        for crate_dir, name in by_depth:
            if crate_dir != Path(".") and path.is_relative_to(crate_dir):
                packages.add(name)
                break
    args = [f"-p {name}" for name in sorted(packages)]
    enabled = [f"{name}/{feature}" for name in sorted(packages) for feature in features.get(name, [])]
    if enabled:
        args.append("--features " + ",".join(enabled))
    return " ".join(args)


def main() -> None:
    root = Path.cwd().resolve()
    metadata = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1"], check=True, capture_output=True, text=True
        ).stdout
    )
    members = set(metadata["workspace_members"])
    names = {package["id"]: package["name"] for package in metadata["packages"]}
    crates = [
        (Path(package["manifest_path"]).parent.resolve().relative_to(root), package["name"])
        for package in metadata["packages"]
        if package["id"] in members
    ]
    # `-p X` alone enables only X's own defaults; a `--workspace` build unifies what the other members enable on it.
    features = {
        names[node["id"]]: [feature for feature in node["features"] if feature != "default"]
        for node in metadata["resolve"]["nodes"]
        if node["id"] in members
    }
    print(doc_args(sys.stdin.read().splitlines(), crates, features))


if __name__ == "__main__":
    main()

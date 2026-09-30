#!/usr/bin/env python3
"""Exit 0 when a path on stdin can change what `local_ci.sh --slim` lints, 3 when none can.

The slim lint checks inference-core under each model family's features, so it covers core and every workspace crate
core depends on (dev-dependencies too, since it lints the tests). A workspace-wide file or this lint's own script
always counts.
"""

import json
import subprocess
import sys
from pathlib import Path

from doc_targets import WORKSPACE_WIDE

SLIM_ROOT = "inference-core"
ALWAYS = WORKSPACE_WIDE | {"scripts/local_ci.sh", "scripts/slim_needed.py", "scripts/doc_targets.py"}
# Distinct from the 1 an uncaught exception exits with, so a crash runs the lint instead of skipping it.
NOT_NEEDED = 3


def slim_needed(paths: list[str], crates: list[tuple[Path, str]], slim_crates: set[str]) -> bool:
    """Whether any of `paths` lies in a crate of `slim_crates`, given each crate as (directory, name)."""
    by_depth = sorted(crates, key=lambda crate: len(crate[0].parts), reverse=True)
    for line in paths:
        if not line.strip():
            continue
        path = Path(line.strip())
        if str(path) in ALWAYS:
            return True
        for crate_dir, name in by_depth:
            if crate_dir != Path(".") and path.is_relative_to(crate_dir):
                if name in slim_crates:
                    return True
                break
    return False


def slim_crates(root_id: str, nodes: dict[str, list[str]], members: set[str], names: dict[str, str]) -> set[str]:
    """`root_id` and the workspace members it reaches through any kind of dependency."""
    seen, stack = {root_id}, [root_id]
    while stack:
        for dep in nodes.get(stack.pop(), []):
            if dep in members and dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return {names[package_id] for package_id in seen}


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
    nodes = {node["id"]: node["dependencies"] for node in metadata["resolve"]["nodes"]}
    root_id = next(package_id for package_id in members if names[package_id] == SLIM_ROOT)
    needed = slim_needed(sys.stdin.read().splitlines(), crates, slim_crates(root_id, nodes, members, names))
    sys.exit(0 if needed else NOT_NEEDED)


if __name__ == "__main__":
    main()

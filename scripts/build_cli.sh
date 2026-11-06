#!/usr/bin/env bash
# The `inference` binary (target/debug/inference) for local runs, built one way: scripts/build_cli.sh [features], e.g.
# `cuda` or `cuda,vulkan`. `--workspace` keeps the features unified as the CI modes do, and the build is recorded in the
# ledger local_ci.sh sweeps by, so its variant lives while it is used; `cargo build -p inference-cli` forks another.
set -euo pipefail
cd "$(dirname "$0")/.."

LEDGER=target/debug/.ledger
features=${1:-}
args=(build --workspace --bins)
if [[ -n $features ]]; then args+=(--features "$features"); fi
cargo "${args[@]}"
mkdir -p "$LEDGER"
# a no-op replay names the build's live artifacts
cargo "${args[@]}" --message-format=json > "$LEDGER/cli-${features:-default}.json.new"
mv "$LEDGER/cli-${features:-default}.json.new" "$LEDGER/cli-${features:-default}.json"

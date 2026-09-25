#!/usr/bin/env bash
# Slow CI checks run locally instead of on hosted runners; usage: scripts/local_ci.sh [--docs] [--tests] (default: --docs)
set -euo pipefail
cd "$(dirname "$0")/.."

docs=0
tests=0
[[ $# -eq 0 ]] && docs=1
for arg in "$@"; do
    case $arg in
        --docs) docs=1 ;;
        --tests) tests=1 ;;
        *) echo "unknown option $arg" >&2; exit 2 ;;
    esac
done

if [[ $docs -eq 1 ]]; then
    RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -D warnings" cargo doc --workspace --no-deps
fi
if [[ $tests -eq 1 ]]; then
    cargo test -p inference-core -p inference-quant -p inference-vision
fi

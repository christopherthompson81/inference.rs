#!/usr/bin/env bash
# The canonical local checks; usage: scripts/local_ci.sh [--lint] [--tests] [--cuda] [--docs] (default: --lint --tests)
# Each mode always builds the same package and feature set, so cargo reuses its artifacts between runs instead of
# rebuilding for a new combination. Keep CC/CXX/NVCC and the INFERENCE_TEST_* model paths in ~/.cargo/config.toml
# [env] rather than on the command line: build scripts track them, and changing one rebuilds everything above ring.
set -euo pipefail
cd "$(dirname "$0")/.."

lint=0
tests=0
cuda=0
docs=0
[[ $# -eq 0 ]] && lint=1 && tests=1
for arg in "$@"; do
    case $arg in
        --lint) lint=1 ;;
        --tests) tests=1 ;;
        --cuda) cuda=1 ;;
        --docs) docs=1 ;;
        *) echo "unknown option $arg" >&2; exit 2 ;;
    esac
done

if [[ $lint -eq 1 ]]; then
    cargo fmt --all -- --check
    cargo clippy --workspace --tests --examples -- -D warnings
fi
# Examples are compile-checked by clippy --examples; linking them in the test modes costs minutes and many GB.
TEST_TARGETS=(--no-fail-fast --lib --bins --tests)
if [[ $tests -eq 1 || $cuda -eq 1 ]] && ! cargo nextest --version > /dev/null 2>&1; then
    # nextest runs each test in its own process (CUDA tests stop sharing a context) and schedules nextest.toml groups
    echo "cargo-nextest is required: curl -LsSf https://get.nexte.st/latest/linux | tar zxf - -C ~/.cargo/bin" >&2
    exit 2
fi
if [[ $tests -eq 1 ]]; then
    cargo nextest run --workspace "${TEST_TARGETS[@]}"
    # nextest does not run doctests
    cargo test --workspace --no-fail-fast --doc
fi
if [[ $cuda -eq 1 ]]; then
    # GPU tests skip themselves without a device; model-backed tests run when their INFERENCE_TEST_* path is set
    cargo clippy --workspace --features cuda --tests --examples -- -D warnings
    cargo nextest run --workspace --features cuda "${TEST_TARGETS[@]}"
fi
if [[ $docs -eq 1 ]]; then
    RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -D warnings" cargo doc --workspace --no-deps
fi

#!/usr/bin/env bash
# Canonical local checks with fixed package/feature sets: scripts/local_ci.sh [--lint] [--tests] [--cuda] [--docs]
# --sweep then deletes target/debug artifacts the selected modes no longer use (stale variants pile up otherwise).
# Build env (CC/CXX/NVCC) and INFERENCE_TEST_* paths belong in ~/.cargo/config.toml [env]; changing one rebuilds deps.
set -euo pipefail
cd "$(dirname "$0")/.."

lint=0
tests=0
cuda=0
docs=0
sweep=0
for arg in "$@"; do
    case $arg in
        --lint) lint=1 ;;
        --tests) tests=1 ;;
        --cuda) cuda=1 ;;
        --docs) docs=1 ;;
        --sweep) sweep=1 ;;
        *) echo "unknown option $arg" >&2; exit 2 ;;
    esac
done
[[ $((lint + tests + cuda + docs)) -eq 0 ]] && lint=1 && tests=1

# Examples are compile-checked by clippy --examples; the test modes only link the smoke set below.
CLIPPY=(clippy --workspace --tests --examples)
TEST_TARGETS=(--workspace --lib --bins --tests)
# The rest of examples/rust is built on request (`-p inference-examples --example <name>`).
# --workspace keeps the same feature unification as the tests, so no second copy of the crates gets built.
SMOKE=(build --workspace --example text_generation --example streaming --example multimodal_basic)

if [[ $lint -eq 1 ]]; then
    cargo fmt --all -- --check
    cargo "${CLIPPY[@]}" -- -D warnings
fi
if [[ $tests -eq 1 || $cuda -eq 1 ]] && ! cargo nextest --version > /dev/null 2>&1; then
    # nextest runs each test in its own process (CUDA tests stop sharing a context) and schedules nextest.toml groups
    echo "cargo-nextest is required: curl -LsSf https://get.nexte.st/latest/linux | tar zxf - -C ~/.cargo/bin" >&2
    exit 2
fi
if [[ $tests -eq 1 ]]; then
    cargo nextest run --no-fail-fast "${TEST_TARGETS[@]}"
    # nextest does not run doctests
    cargo test --workspace --no-fail-fast --doc
    cargo "${SMOKE[@]}"
fi
if [[ $cuda -eq 1 ]]; then
    # GPU tests skip themselves without a device; model-backed tests run when their INFERENCE_TEST_* path is set
    cargo "${CLIPPY[@]}" --features cuda -- -D warnings
    cargo nextest run --no-fail-fast --features cuda "${TEST_TARGETS[@]}"
fi
if [[ $docs -eq 1 ]]; then
    RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -D warnings" cargo doc --workspace --no-deps
fi
if [[ $sweep -eq 1 ]]; then
    # No-op rebuilds of exactly what the modes above built; their JSON names every live artifact. A failed replay
    # would under-report, so nothing is deleted unless all of them succeed.
    live=$(mktemp)
    trap 'rm -f "$live"' EXIT
    # clippy's `-- -D warnings` is part of its fingerprint, so the replay passes it too or it rebuilds every lint unit
    replay() { cargo "$@" --message-format=json >> "$live"; }
    lint_replay() { cargo "${CLIPPY[@]}" "$@" --message-format=json -- -D warnings >> "$live"; }
    if [[ $lint -eq 1 ]]; then lint_replay; fi
    if [[ $tests -eq 1 ]]; then
        replay test --no-run "${TEST_TARGETS[@]}"
        replay "${SMOKE[@]}"
    fi
    if [[ $cuda -eq 1 ]]; then
        lint_replay --features cuda
        replay test --no-run --features cuda "${TEST_TARGETS[@]}"
    fi
    scripts/sweep_target.py target/debug < "$live"
fi

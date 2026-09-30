#!/usr/bin/env bash
# Canonical local checks with fixed package/feature sets: scripts/local_ci.sh [--lint] [--tests] [--cuda] [--metal]
# [--models] [--slim] [--docs|--docs-all] [--bindings]; --models runs the real-checkpoint parity tests on CPU (--cuda
# keeps one GPU parity check).
# --metal is the macOS counterpart of --cuda: the metal-only code paths are invisible to a CPU or CUDA lint.
# --slim lints inference-core with no model families and with each family alone, so feature gates stay intact.
# With --cuda, the GPU-bound CUDA suite runs in the background while the CPU lint and tests run.
# --bindings builds libinference_ffi and runs the C# (needs the .NET SDK) and Python binding tests on the tiny checkpoint.
# --docs checks the docs of the crates whose files differ from origin/master (rustdoc is never incremental), not their
# dependents; --docs-all checks every crate. Neither renders HTML; `cargo doc` does.
# --sweep then deletes target/debug artifacts the selected modes no longer use (stale variants pile up otherwise).
# Build env (CC/CXX/NVCC) and INFERENCE_TEST_* paths belong in ~/.cargo/config.toml [env]; changing one rebuilds deps.
set -euo pipefail
cd "$(dirname "$0")/.."

lint=0
tests=0
cuda=0
metal=0
models=0
slim=0
docs=0
docs_all=0
bindings=0
sweep=0
for arg in "$@"; do
    case $arg in
        --lint) lint=1 ;;
        --tests) tests=1 ;;
        --cuda) cuda=1 ;;
        --metal) metal=1 ;;
        --models) models=1 ;;
        --slim) slim=1 ;;
        --docs) docs=1 ;;
        --docs-all) docs=1 docs_all=1 ;;
        --bindings) bindings=1 ;;
        --sweep) sweep=1 ;;
        *) echo "unknown option $arg" >&2; exit 2 ;;
    esac
done
[[ $((lint + tests + cuda + metal + models + slim + docs + bindings)) -eq 0 ]] && lint=1 && tests=1

# --examples compile-checks the examples (tests only link the smoke set below); --bins checks bins without dev-deps, like CI.
CLIPPY=(clippy --workspace --bins --tests --examples)
TEST_TARGETS=(--workspace --lib --bins --tests)
# The rest of examples/rust is built on request (`-p inference-examples --example <name>`).
# --workspace keeps the same feature unification as the tests, so no second copy of the crates gets built.
SMOKE=(build --workspace --example text_generation --example streaming --example multimodal_basic)
SLIM_FAMILIES=("" models-gemma models-llama models-other models-phi models-qwen)
# --workspace keeps the tests' feature unification, so the cdylib reuses their artifacts instead of rebuilding deps.
BINDINGS=(build --workspace --lib --example tiny_checkpoint)
CSHARP=bindings/csharp
PYTHON=bindings/python
slim_clippy() { cargo clippy -p inference-core --lib --tests --no-default-features ${1:+--features $1} "${@:2}"; }

if [[ $lint -eq 1 ]]; then
    cargo fmt --all -- --check
    python3 -m unittest discover -s scripts -p "test_*.py" -q
    # CI's typos job; skipped with a note where the binary is missing so lint still runs everywhere.
    if command -v typos > /dev/null; then typos --config .typos.toml; else echo "typos not installed: cargo install typos-cli" >&2; fi
fi
if [[ $tests -eq 1 || $cuda -eq 1 || $metal -eq 1 || $models -eq 1 ]] && ! cargo nextest --version > /dev/null 2>&1; then
    # nextest runs each test in its own process (CUDA tests stop sharing a context) and schedules nextest.toml groups
    platform=linux; [[ $OSTYPE == darwin* ]] && platform=mac
    echo "cargo-nextest is required: curl -LsSf https://get.nexte.st/latest/$platform | tar zxf - -C ~/.cargo/bin" >&2
    exit 2
fi
# RLIMIT_NPROC counts every process the user runs, so this test never shares the machine with another suite.
ALONE='package(inference-sandbox) & test(rlimit_nproc_caps_processes)'
cuda_pid=
if [[ $cuda -eq 1 ]]; then
    cargo "${CLIPPY[@]}" --features cuda -- -D warnings
    # GPU tests skip themselves without a device; model-backed tests run when their INFERENCE_TEST_* path is set
    if [[ $lint -eq 1 || $tests -eq 1 ]]; then
        # The CUDA suite is GPU-bound, so it runs in the background while the CPU lint and tests use the cores.
        cargo nextest run --no-run --features cuda "${TEST_TARGETS[@]}"
        cuda_log=$(mktemp)
        cargo nextest run --no-fail-fast --profile cuda --features cuda "${TEST_TARGETS[@]}" -E "not ($ALONE)" \
            > "$cuda_log" 2>&1 &
        cuda_pid=$!
        trap 'kill "$cuda_pid" 2> /dev/null; rm -f "$cuda_log"' EXIT
    else
        cargo nextest run --no-fail-fast --profile cuda --features cuda "${TEST_TARGETS[@]}"
    fi
fi
failed=0
if [[ $metal -eq 1 ]]; then
    # Metal and the CPU suite share one chip, so this runs in series rather than beside them like the CUDA suite
    cargo "${CLIPPY[@]}" --features metal -- -D warnings || failed=1
    cargo nextest run --no-fail-fast --profile metal --features metal "${TEST_TARGETS[@]}" || failed=1
fi
if [[ $lint -eq 1 ]]; then
    cargo "${CLIPPY[@]}" -- -D warnings || failed=1
fi
if [[ $tests -eq 1 ]]; then
    if [[ -n $cuda_pid ]]; then
        cargo nextest run --no-fail-fast "${TEST_TARGETS[@]}" -E "not ($ALONE)" || failed=1
    else
        cargo nextest run --no-fail-fast "${TEST_TARGETS[@]}" || failed=1
    fi
    # nextest does not run doctests
    cargo test --workspace --no-fail-fast --doc || failed=1
    cargo "${SMOKE[@]}" || failed=1
fi
if [[ -n $cuda_pid ]]; then
    wait "$cuda_pid" || failed=1
    trap - EXIT
    echo "---- CUDA suite ----"
    cat "$cuda_log"
    rm -f "$cuda_log"
    cargo nextest run --no-fail-fast --profile cuda --features cuda "${TEST_TARGETS[@]}" -E "$ALONE" || failed=1
fi
[[ $failed -eq 0 ]] || exit 1
if [[ $models -eq 1 ]]; then
    cargo nextest run --no-fail-fast --profile models "${TEST_TARGETS[@]}"
fi
if [[ $slim -eq 1 ]]; then
    for family in "${SLIM_FAMILIES[@]}"; do slim_clippy "$family" -- -D warnings; done
fi
if [[ $bindings -eq 1 ]]; then
    cargo "${BINDINGS[@]}"
    tiny=$(mktemp -d)
    target/debug/examples/tiny_checkpoint "$tiny" > /dev/null
    # The library just built, not a release build a resolver would prefer
    native=$PWD/target/debug
    bindings_failed=0
    INFERENCE_NATIVE_DIR=$native INFERENCE_TEST_TINY_CHECKPOINT=$tiny \
        python3 -m unittest discover -s "$PYTHON/tests" -t "$PYTHON" || bindings_failed=1
    # A missing or broken .NET SDK fails the C# tests without skipping the Python ones above
    if dotnet build "$CSHARP/InferenceRs.slnx" -v quiet; then
        INFERENCE_NATIVE_DIR=$native dotnet run --project "$CSHARP/tests/InferenceRs.BindingCoverage" --no-build \
            || bindings_failed=1
        INFERENCE_NATIVE_DIR=$native INFERENCE_TEST_TINY_CHECKPOINT=$tiny \
            dotnet run --project "$CSHARP/tests/InferenceRs.EngineTest" --no-build || bindings_failed=1
    else
        bindings_failed=1
    fi
    rm -rf "$tiny"
    [[ $bindings_failed -eq 0 ]] || exit 1
fi
DOC_TARGETS=()
if [[ $docs -eq 1 ]]; then
    if [[ $docs_all -eq 1 ]]; then
        DOC_TARGETS=(--workspace)
    else
        if ! base=$(git merge-base HEAD origin/master 2> /dev/null); then
            echo "--docs needs origin/master to diff against; fetch it or pass --docs-all" >&2
            exit 2
        fi
        # An assignment, not `read <<< "$(...)"`, so a failed lookup stops the run instead of documenting nothing
        targets=$({ git diff --name-only --no-renames "$base"; git ls-files --others --exclude-standard; } |
            scripts/doc_targets.py)
        read -ra DOC_TARGETS <<< "$targets"
    fi
fi
# Checks without rendering: each crate's HTML merge takes target/doc's lock, which serialized the rustdoc runs.
doc_build() { RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -D warnings --emit dep-info" cargo doc --no-deps "${DOC_TARGETS[@]}" "$@"; }
if [[ $docs -eq 1 ]]; then
    if [[ ${#DOC_TARGETS[@]} -eq 0 ]]; then
        echo "--docs: no crate differs from master, nothing to document"
    else
        doc_build
    fi
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
    if [[ $tests -eq 1 || $models -eq 1 ]]; then replay test --no-run "${TEST_TARGETS[@]}"; fi
    if [[ $tests -eq 1 ]]; then replay "${SMOKE[@]}"; fi
    if [[ $cuda -eq 1 ]]; then
        lint_replay --features cuda
        replay test --no-run --features cuda "${TEST_TARGETS[@]}"
    fi
    if [[ $metal -eq 1 ]]; then
        lint_replay --features metal
        replay test --no-run --features metal "${TEST_TARGETS[@]}"
    fi
    if [[ $slim -eq 1 ]]; then
        for family in "${SLIM_FAMILIES[@]}"; do
            slim_clippy "$family" --message-format=json -- -D warnings >> "$live"
        done
    fi
    if [[ ${#DOC_TARGETS[@]} -gt 0 ]]; then doc_build --message-format=json >> "$live"; fi
    if [[ $bindings -eq 1 ]]; then replay "${BINDINGS[@]}"; fi
    scripts/sweep_target.py target/debug < "$live"
fi

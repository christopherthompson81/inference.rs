#!/usr/bin/env bash
# Canonical local checks with fixed package/feature sets: scripts/local_ci.sh [--lint] [--tests] [--cuda] [--metal]
# [--vulkan] [--slim] [--docs|--docs-all] [--bindings] [--size|--size-update] [--filter=EXPR]. Real-checkpoint tests are
# not run here:
# scripts/deep_checks.sh runs them on request.
# --metal is the macOS counterpart of --cuda: the metal-only code paths are invisible to a CPU or CUDA lint.
# --vulkan lints inference-vulkan, inference-tensor and the SDK with the `vulkan` feature and runs the Vulkan tests (needs
# glslc; the GPU tests skip without a device).
# --slim lints inference-core with no model families and with each family alone, so feature gates stay intact; it is
# skipped when nothing inference-core builds on differs from origin/master.
# With --cuda, the GPU-bound CUDA suite runs in the background while the CPU lint and tests run; the binding tests run
# in the background too, and the slim lint and docs check overlap both, since only cargo needs the build lock. With
# --tests, the --size bundle builds beside the suites too (under its own target/bundle lock).
# --bindings builds libinference_ffi and runs the C# (needs the .NET SDK) and Python binding tests on the tiny checkpoint.
# --docs checks the docs of the crates whose files differ from origin/master (rustdoc is never incremental), not their
# dependents; --docs-all checks every crate. Neither renders HTML; `cargo doc` does.
# --size (Linux) builds the CUDA C ABI library as the wheels do and fails when its file or a large section grew past
# scripts/bundle_size_baseline.json; --size-update rewrites the baseline (commit it with the change that moved it).
# Each passing mode records what it built (target/debug/.ledger, from a no-op replay), and every run deletes what no mode
# recorded in the last LEDGER_DAYS: cargo never removes a variant (another feature set or `-p` selection) itself. A
# build run by hand outside the modes is spared for GRACE_HOURS. --sweep deletes all but the selected modes' artifacts.
# --filter=EXPR narrows the test suites to a nextest filter expression (doctests skipped) while building exactly what
# the full suites build: iterating with `cargo nextest -p ...` builds another variant of every crate it touches.
# Build env (CC/CXX/NVCC) and INFERENCE_TEST_* paths belong in ~/.cargo/config.toml [env]; changing one rebuilds deps.
set -euo pipefail
cd "$(dirname "$0")/.."

lint=0
tests=0
cuda=0
metal=0
vulkan=0
slim=0
docs=0
docs_all=0
bindings=0
size=0
size_update=0
sweep=0
filter=
for arg in "$@"; do
    case $arg in
        --lint) lint=1 ;;
        --tests) tests=1 ;;
        --cuda) cuda=1 ;;
        --metal) metal=1 ;;
        --vulkan) vulkan=1 ;;
        --slim) slim=1 ;;
        --docs) docs=1 ;;
        --docs-all) docs=1 docs_all=1 ;;
        --bindings) bindings=1 ;;
        --size) size=1 ;;
        --size-update) size=1 size_update=1 ;;
        --sweep) sweep=1 ;;
        --filter=*) filter=${arg#--filter=} ;;
        *) echo "unknown option $arg" >&2; exit 2 ;;
    esac
done
[[ $((lint + tests + cuda + metal + vulkan + slim + docs + bindings + size)) -eq 0 ]] && lint=1 && tests=1

# --examples compile-checks the examples (tests only link the smoke set below); --bins checks bins without dev-deps, like CI.
CLIPPY=(clippy --workspace --bins --tests --examples)
TEST_TARGETS=(--workspace --lib --bins --tests)
# examples/rust links only on request (`-p inference-examples --example <name>`); clippy type-checks every example.
SLIM_FAMILIES=("" models-gemma models-llama models-other models-phi models-qwen)
# --workspace keeps the tests' feature unification, so the cdylib reuses their artifacts instead of rebuilding deps.
BINDINGS=(build --workspace --lib --example tiny_checkpoint)
CSHARP=bindings/csharp
PYTHON=bindings/python
LEDGER=target/debug/.ledger
LEDGER_DAYS=3
GRACE_HOURS=24
# incremental dirs per crate kept past its recorded units: their names carry no hash a replay reports
INCREMENTAL_SLACK=4
slim_clippy() { cargo clippy -p inference-core -p inference-gguf --lib --tests --no-default-features ${1:+--features inference-core/$1} "${@:2}"; }

if [[ $lint -eq 1 ]]; then
    cargo fmt --all -- --check
    python3 -m unittest discover -s scripts -p "test_*.py" -q
    # CI's typos job; skipped with a note where the binary is missing so lint still runs everywhere.
    if command -v typos > /dev/null; then typos --config .typos.toml; else echo "typos not installed: cargo install typos-cli" >&2; fi
fi
if [[ $tests -eq 1 || $cuda -eq 1 || $metal -eq 1 ]] && ! cargo nextest --version > /dev/null 2>&1; then
    # nextest runs each test in its own process (CUDA tests stop sharing a context) and schedules nextest.toml groups
    platform=linux; [[ $OSTYPE == darwin* ]] && platform=mac
    echo "cargo-nextest is required: curl -LsSf https://get.nexte.st/latest/$platform | tar zxf - -C ~/.cargo/bin" >&2
    exit 2
fi
# RLIMIT_NPROC counts every process the user runs, so this test never shares the machine with another suite.
ALONE='package(inference-sandbox) & test(rlimit_nproc_caps_processes)'
# a suite's own expression narrowed by --filter; a filter matching none of a suite's tests passes it
narrowed() { if [[ -n $filter ]]; then echo "($1) & ($filter)"; else echo "$1"; fi; }
NO_TESTS=(--no-tests=pass)
cuda_pid= bindings_pid= size_pid= cuda_log= bindings_log= size_log= tiny=
cleanup() {
    if [[ -n $cuda_pid ]]; then kill "$cuda_pid" 2> /dev/null || true; fi
    if [[ -n $bindings_pid ]]; then kill "$bindings_pid" 2> /dev/null || true; fi
    if [[ -n $size_pid ]]; then kill "$size_pid" 2> /dev/null || true; fi
    rm -rf "$cuda_log" "$bindings_log" "$size_log" "$tiny"
}
# What the wheels bundle: the default model set, fat LTO, stripped, no host-specific codegen (build_wheels.py). Its
# fat LTO runs mostly on one core under target/bundle's own lock, so it builds while the suites run.
size_build() { RUSTFLAGS= cargo build --profile bundle -p inference-ffi --features cuda; }
start_size_build() {
    if [[ $size -eq 1 && -z $size_pid && $OSTYPE == linux* ]]; then
        size_log=$(mktemp)
        size_build > "$size_log" 2>&1 &
        size_pid=$!
    fi
}
trap cleanup EXIT
if [[ $cuda -eq 1 ]]; then
    cargo "${CLIPPY[@]}" --features cuda -- -D warnings
    # GPU tests skip themselves without a device
    if [[ $lint -eq 1 || $tests -eq 1 ]]; then
        # The CUDA suite is GPU-bound, so it runs in the background while the CPU lint and tests use the cores.
        cargo nextest run --no-run --features cuda "${TEST_TARGETS[@]}"
        cuda_log=$(mktemp)
        cargo nextest run --no-fail-fast "${NO_TESTS[@]}" --profile cuda --features cuda "${TEST_TARGETS[@]}" \
            -E "$(narrowed "not ($ALONE)")" > "$cuda_log" 2>&1 &
        cuda_pid=$!
    else
        cargo nextest run --no-fail-fast "${NO_TESTS[@]}" --profile cuda --features cuda "${TEST_TARGETS[@]}" \
            -E "$(narrowed "all()")"
    fi
fi
failed=0
if [[ $metal -eq 1 ]]; then
    # Metal and the CPU suite share one chip, so this runs in series rather than beside them like the CUDA suite
    cargo "${CLIPPY[@]}" --features metal -- -D warnings || failed=1
    cargo nextest run --no-fail-fast --profile metal --features metal "${TEST_TARGETS[@]}" || failed=1
fi
if [[ $vulkan -eq 1 ]]; then
    VULKAN_PACKAGES=(-p inference-vulkan -p inference-tensor -p inference-paged-attn -p inference-nn -p inference)
    cargo clippy "${VULKAN_PACKAGES[@]}" --features vulkan --tests -- -D warnings || failed=1
    cargo nextest run --no-fail-fast "${NO_TESTS[@]}" "${VULKAN_PACKAGES[@]}" --features vulkan \
        -E "$(narrowed 'package(inference-vulkan) | test(vulkan)')" || failed=1
fi
if [[ $lint -eq 1 ]]; then
    cargo "${CLIPPY[@]}" -- -D warnings || failed=1
fi
if [[ $tests -eq 1 ]]; then
    cargo nextest run --no-run "${TEST_TARGETS[@]}"
    start_size_build
    if [[ -n $cuda_pid ]]; then
        cargo nextest run --no-fail-fast "${NO_TESTS[@]}" "${TEST_TARGETS[@]}" -E "$(narrowed "not ($ALONE)")" \
            || failed=1
    else
        cargo nextest run --no-fail-fast "${NO_TESTS[@]}" "${TEST_TARGETS[@]}" -E "$(narrowed "all()")" || failed=1
    fi
    # nextest does not run doctests
    if [[ -z $filter ]]; then cargo test --workspace --no-fail-fast --doc || failed=1; fi
fi
# The binding tests and the CUDA suite need no cargo lock, so the slim lint and the docs check run while they do.
if [[ $bindings -eq 1 ]]; then
    cargo "${BINDINGS[@]}"
    tiny=$(mktemp -d)
    target/debug/examples/tiny_checkpoint "$tiny" > /dev/null
    bindings_log=$(mktemp)
    run_binding_tests() {
        # The library just built, not a release build a resolver would prefer
        local native=$PWD/target/debug status=0
        INFERENCE_NATIVE_DIR=$native INFERENCE_TEST_TINY_CHECKPOINT=$tiny \
            python3 -m unittest discover -s "$PYTHON/tests" -t "$PYTHON" || status=1
        # A missing or broken .NET SDK fails the C# tests without skipping the Python ones above
        if dotnet build "$CSHARP/InferenceRs.slnx" -v quiet --disable-build-servers; then
            INFERENCE_NATIVE_DIR=$native dotnet run --project "$CSHARP/tests/InferenceRs.BindingCoverage" --no-build \
                || status=1
            INFERENCE_NATIVE_DIR=$native INFERENCE_TEST_TINY_CHECKPOINT=$tiny \
                dotnet run --project "$CSHARP/tests/InferenceRs.EngineTest" --no-build || status=1
        else
            status=1
        fi
        return $status
    }
    run_binding_tests > "$bindings_log" 2>&1 &
    bindings_pid=$!
fi
# Files that differ from origin/master, committed or not; fails when origin/master is unknown.
changed_since_master() {
    local base
    base=$(git merge-base HEAD origin/master 2> /dev/null) || return 1
    { git diff --name-only --no-renames "$base"; git ls-files --others --exclude-standard; }
}
# Exit 3 means nothing slim lints changed; any other failure, or no origin/master, runs it.
slim_skip=0
if [[ $slim -eq 1 ]] && changed=$(changed_since_master); then
    slim_status=0
    scripts/slim_needed.py <<< "$changed" || slim_status=$?
    if [[ $slim_status -eq 3 ]]; then
        slim_skip=1
        echo "--slim: nothing inference-core builds on differs from master, skipping"
    fi
fi
if [[ $slim -eq 1 && $slim_skip -eq 0 ]]; then
    for family in "${SLIM_FAMILIES[@]}"; do slim_clippy "$family" -- -D warnings || failed=1; done
fi
DOC_TARGETS=()
if [[ $docs -eq 1 ]]; then
    if [[ $docs_all -eq 1 ]]; then
        DOC_TARGETS=(--workspace)
    else
        if ! changed=$(changed_since_master); then
            echo "--docs needs origin/master to diff against; fetch it or pass --docs-all" >&2
            exit 2
        fi
        # An assignment, not `read <<< "$(...)"`, so a failed lookup stops the run instead of documenting nothing
        targets=$(scripts/doc_targets.py <<< "$changed")
        read -ra DOC_TARGETS <<< "$targets"
    fi
fi
# Checks without rendering: each crate's HTML merge takes target/doc's lock, which serialized the rustdoc runs.
doc_build() { RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -D warnings --emit dep-info" cargo doc --no-deps "${DOC_TARGETS[@]}" "$@"; }
if [[ $docs -eq 1 ]]; then
    if [[ ${#DOC_TARGETS[@]} -eq 0 ]]; then
        echo "--docs: no crate differs from master, nothing to document"
    else
        doc_build || failed=1
    fi
fi
if [[ -n $bindings_pid ]]; then
    wait "$bindings_pid" || failed=1
    echo "---- binding tests ----"
    cat "$bindings_log"
fi
if [[ -n $cuda_pid ]]; then
    wait "$cuda_pid" || failed=1
    echo "---- CUDA suite ----"
    cat "$cuda_log"
    # Last and alone: it counts every process the user runs, the lint and binding tests above included.
    cargo nextest run --no-fail-fast "${NO_TESTS[@]}" --profile cuda --features cuda "${TEST_TARGETS[@]}" \
        -E "$(narrowed "$ALONE")" || failed=1
fi
cuda_pid= bindings_pid=
builds=$((lint + tests + cuda + metal + vulkan + slim + docs + bindings))
# A failed run records nothing (a replay would rebuild what failed) and deletes only what live units no longer read:
# renamed codegen units' split debug info and older incremental sessions, which need no replay to find.
if [[ $failed -ne 0 ]]; then
    if [[ $builds -gt 0 ]]; then scripts/sweep_target.py target/debug --superseded || true; fi
    exit 1
fi
if [[ $size -eq 1 ]]; then
    if [[ $OSTYPE != linux* ]]; then echo "--size measures the Linux CUDA library" >&2; exit 2; fi
    compute_cap=${CUDA_COMPUTE_CAP:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)}
    # the toolkit decides which kernels build (cuTile from 13.2), so the baseline holds for one
    cuda_version=$(nvcc --version | grep -o 'release [0-9.]*' | cut -d' ' -f2)
    if [[ -n $size_pid ]]; then
        size_status=0
        wait "$size_pid" || size_status=$?
        size_pid=
        echo "---- bundle build ----"
        cat "$size_log"
        [[ $size_status -eq 0 ]] || exit 1
    else
        size_build
    fi
    size_args=(target/bundle/libinference_ffi.so "$compute_cap" "$cuda_version")
    if [[ $size_update -eq 1 ]]; then size_args+=(--update); fi
    scripts/bundle_size.py "${size_args[@]}"
fi
# --size builds under target/bundle, which the sweep leaves alone
if [[ $builds -gt 0 ]]; then
    # No-op rebuilds of exactly what the modes above built; their JSON names every live artifact. A failed replay would
    # under-report, so it records nothing.
    mkdir -p "$LEDGER"
    recorded=()
    record() {
        local mode=$1
        shift
        if "$@" > "$LEDGER/$mode.json.new"; then
            mv "$LEDGER/$mode.json.new" "$LEDGER/$mode.json"
            recorded+=("$LEDGER/$mode.json")
        else
            rm -f "$LEDGER/$mode.json.new"
        fi
    }
    replay() { cargo "$@" --message-format=json; }
    # clippy's `-- -D warnings` is part of its fingerprint, so the replay passes it too or it rebuilds every lint unit
    lint_replay() { cargo "$@" --message-format=json -- -D warnings; }
    slim_replay() {
        for family in "${SLIM_FAMILIES[@]}"; do slim_clippy "$family" --message-format=json -- -D warnings || return 1; done
    }
    if [[ $lint -eq 1 ]]; then record lint lint_replay "${CLIPPY[@]}"; fi
    if [[ $tests -eq 1 ]]; then record tests replay test --no-run "${TEST_TARGETS[@]}"; fi
    if [[ $cuda -eq 1 ]]; then
        record cuda-lint lint_replay "${CLIPPY[@]}" --features cuda
        record cuda-tests replay test --no-run --features cuda "${TEST_TARGETS[@]}"
    fi
    if [[ $metal -eq 1 ]]; then
        record metal-lint lint_replay "${CLIPPY[@]}" --features metal
        record metal-tests replay test --no-run --features metal "${TEST_TARGETS[@]}"
    fi
    if [[ $vulkan -eq 1 ]]; then
        record vulkan-lint lint_replay clippy "${VULKAN_PACKAGES[@]}" --features vulkan --tests
        record vulkan-tests replay test --no-run "${VULKAN_PACKAGES[@]}" --features vulkan
    fi
    if [[ $slim -eq 1 && $slim_skip -eq 0 ]]; then record slim slim_replay; fi
    if [[ ${#DOC_TARGETS[@]} -gt 0 ]]; then record docs doc_build --message-format=json; fi
    if [[ $bindings -eq 1 ]]; then record bindings replay "${BINDINGS[@]}"; fi
    if [[ $sweep -eq 1 ]]; then
        ledgers=("${recorded[@]}")
        sweep_args=()
    else
        mapfile -t ledgers < <(find "$LEDGER" -name '*.json' -mtime -"$LEDGER_DAYS")
        sweep_args=(--grace-hours "$GRACE_HOURS" --slack "$INCREMENTAL_SLACK")
    fi
    if [[ ${#ledgers[@]} -gt 0 ]]; then
        cat "${ledgers[@]}" | scripts/sweep_target.py target/debug "${sweep_args[@]}" || true
    else
        scripts/sweep_target.py target/debug --superseded || true
    fi
fi

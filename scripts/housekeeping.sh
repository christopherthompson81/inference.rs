#!/usr/bin/env bash
# Housekeeping measurements (docs/src/content/docs/developer/housekeeping.md): scripts/housekeeping.sh [--dups] [--deps]
# [--ir] [--incremental] [--crate-time] [--slow-tests LOG]. With no mode: --dups and --deps, the two that build nothing.
# --dups: duplicated-code scan (scripts/dup_scan.py; `--pair A B` there lists the spans two files share).
# --deps: unused dependencies (cargo machete).
# --ir: LLVM IR lines of the crates that set build time, under the CPU test features so their deps are reused.
# --incremental: touch one crate low in the stack at a time and time the workspace check and the test build (a touch
#   costs the dependents' re-check and the links; rustc's incremental cache makes the touched crate itself nearly free).
# --crate-time: each crate that ends a build compiled whole, alone (fresh incremental dir, dependencies built).
# --slow-tests LOG: slowest tests and modules per nextest run of a `local_ci.sh ... 2>&1 | tee LOG` log.
set -euo pipefail
cd "$(dirname "$0")/.."

IR_CRATES=(inference-core inference-nn inference-quant inference-api inference-server-core inference-models-llama
    inference-models-qwen inference-models-gemma inference-models-phi inference-models-other)
TIMED_CRATES=(inference-core inference-api inference-http inference-server-core inference-webui)
TOUCH_FILES=(crates/inference-nn/src/lib.rs crates/inference-core/src/lib.rs crates/inference-models-other/src/lib.rs)

dups=0 deps=0 ir=0 incremental=0 crate_time=0 slow_log=
while [[ $# -gt 0 ]]; do
    case $1 in
        --dups) dups=1 ;;
        --deps) deps=1 ;;
        --ir) ir=1 ;;
        --incremental) incremental=1 ;;
        --crate-time) crate_time=1 ;;
        --slow-tests) slow_log=${2:?--slow-tests needs a log}; shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
[[ $((dups + deps + ir + incremental + crate_time)) -eq 0 && -z $slow_log ]] && dups=1 deps=1

elapsed() { local t0=$SECONDS; "$@" > /dev/null 2>&1; echo $((SECONDS - t0)); }

if [[ $dups -eq 1 ]]; then
    echo "== duplicated code"
    python3 scripts/dup_scan.py
fi
if [[ $deps -eq 1 ]]; then
    echo "== unused dependencies"
    command -v cargo-machete > /dev/null || { echo "cargo-machete missing: cargo install cargo-machete" >&2; exit 2; }
    cargo machete || true
fi
if [[ -n $slow_log ]]; then
    echo "== slowest tests"
    python3 scripts/slow_tests.py "$slow_log"
fi
if [[ $ir -eq 1 ]]; then
    echo "== LLVM IR lines"
    command -v cargo-llvm-lines > /dev/null || { echo "cargo-llvm-lines missing: cargo install cargo-llvm-lines" >&2; exit 2; }
    for crate in "${IR_CRATES[@]}"; do
        total=$(cargo llvm-lines -p "$crate" --lib 2> /dev/null | awk '/\(TOTAL\)/ {print $1; exit}')
        printf '  %12s  %s\n' "${total:-?}" "$crate"
    done
fi
if [[ $incremental -eq 1 ]]; then
    echo "== incremental rebuilds (s): workspace check --tests, test build"
    cargo check --workspace --tests > /dev/null 2>&1
    cargo nextest run --workspace --lib --bins --tests --no-run > /dev/null 2>&1
    for f in "${TOUCH_FILES[@]}"; do
        touch "$f"
        check=$(elapsed cargo check --workspace --tests)
        touch "$f"
        build=$(elapsed cargo nextest run --workspace --lib --bins --tests --no-run)
        printf '  %4s %4s  %s\n' "$check" "$build" "$f"
    done
fi
if [[ $crate_time -eq 1 ]]; then
    echo "== whole-crate compile (s)"
    scratch=$(mktemp -d)
    for crate in "${TIMED_CRATES[@]}"; do
        cargo build -p "$crate" --lib > /dev/null 2>&1
        # a new incremental dir changes rustc's arguments, so only this crate recompiles, from nothing
        printf '  %4s  %s\n' "$(elapsed cargo rustc -p "$crate" --lib -- -C incremental="$scratch/$crate")" "$crate"
    done
    rm -rf "$scratch"
fi

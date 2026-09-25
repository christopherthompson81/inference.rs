#!/usr/bin/env bash
# Usage: tests/run.sh [--features cuda] [--backend cpu|cuda] [model_dir image]  (C consumer vs Rust detector parity)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
crate="$(dirname "$here")"
root="$(dirname "$crate")"
features=()
backend=cpu
while [[ $# -gt 0 && $1 == --* ]]; do
    case $1 in
        --features) features=(--features "$2"); shift 2 ;;
        --backend) backend=$2; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
if [[ $backend != cpu && $backend != cuda ]]; then
    # the Rust detect example used for parity only selects CPU or CUDA
    echo "parity runs on cpu or cuda, not $backend" >&2
    exit 2
fi
model=${1:-${INFERENCE_TEST_LAYOUT_MODEL:-}}
image=${2:-${INFERENCE_TEST_LAYOUT_IMAGE:-}}

cd "$root"
cargo build --release -p inference-ffi "${features[@]}"
lib="$root/target/release/libinference_ffi.so"
python3 "$here/export_surface.py" "$lib" "$crate/include/inference.h"

INFERENCE_TEST_LAYOUT_MODEL="$model" INFERENCE_TEST_LAYOUT_IMAGE="$image" \
    cargo test --release -p inference-ffi "${features[@]}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cc -std=c99 -Wall -Wextra -Werror -pedantic -I "$crate/include" "$here/c/layout_test.c" \
    -L "$root/target/release" -linference_ffi -Wl,-rpath,"$root/target/release" -o "$work/layout_test"

if [[ -z $model || -z $image ]]; then
    "$work/layout_test" || [[ $? -eq 77 ]]
    echo "C consumer: ABI checks passed; model tests skipped (pass model_dir and image)"
    exit 0
fi
convert "$image" -depth 8 "ppm:$work/page.ppm"
"$work/layout_test" "$model" "$work/page.ppm" "$backend" | grep '^parity:' > "$work/c.txt"

cargo run --release -q -p inference-layout "${features[@]}" --example pp_doclayout_v3_detect -- \
    --model "$model" $([[ $backend == cpu ]] && echo --cpu) "$work/page.ppm" > "$work/rust.jsonl"
python3 "$here/compare_parity.py" "$work/rust.jsonl" "$work/c.txt"

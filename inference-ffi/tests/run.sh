#!/usr/bin/env bash
# Builds libinference_ffi, checks its export surface, runs the Rust ABI tests and the C99 consumer, and compares the
# C consumer's detections with the Rust detect example on the same page.
#   usage: tests/run.sh [--features cuda] [--backend cuda] [model_dir image]
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
python3 - "$work/rust.jsonl" > "$work/rust.txt" <<'EOF'
import json, sys
for line in open(sys.argv[1]):
    for i, d in enumerate(json.loads(line)["detections"]):
        x1, y1, x2, y2 = d["bbox"]
        print(f"parity: {i} {d['class_id']} {d['label']} {d['score']:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")
EOF
diff "$work/rust.txt" "$work/c.txt"
echo "C consumer matches the Rust detector: $(wc -l < "$work/c.txt") detections"

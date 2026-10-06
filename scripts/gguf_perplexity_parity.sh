#!/usr/bin/env bash
# One-off parity check of GGUF weights against a llama.cpp-family reference (mainline or ik_llama.cpp): scores every
# *.gguf in a directory with the reference's llama-perplexity and with inference.rs, and fails past a relative drift.
# Not part of local_ci.sh: it needs the reference's build and real checkpoints, and verifies an adoption once.
#
# Usage: scripts/gguf_perplexity_parity.sh <llama-perplexity> <gguf-dir> [text] [tolerance]
#   text       scored by both sides; default README.md (llama-perplexity needs two 512-token windows)
#   tolerance  relative perplexity drift allowed; default 0.02 (the IQ and IQK types drift up to ~1.6%)
set -euo pipefail
shopt -s nullglob

CTX=512
DEFAULT_TOLERANCE=0.02

if [[ $# -lt 2 ]]; then
    sed -n '6,8p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
reference=$(command -v "$1" || true)
[[ -x $reference ]] || { echo "no llama-perplexity at $1" >&2; exit 2; }
dir=$(realpath "$2")
root=$(cd "$(dirname "$0")/.." && pwd)
text=$(realpath "${3:-$root/README.md}")
tolerance=${4:-$DEFAULT_TOLERANCE}
[[ $tolerance =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "tolerance must be a number, got $tolerance" >&2; exit 2; }
ggufs=("$dir"/*.gguf)
(( ${#ggufs[@]} )) || { echo "no *.gguf in $dir" >&2; exit 2; }

# the IQ kernels run on CUDA; a build without it scores on the CPU
features=()
if command -v nvcc >/dev/null 2>&1; then
    features=(--features cuda)
fi
ours=$(cargo build --manifest-path "$root/Cargo.toml" -p inference-examples --example perplexity "${features[@]}" \
    --message-format=json-render-diagnostics \
    | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' | tail -n 1)
[[ -x $ours ]] || { echo "the perplexity example did not build" >&2; exit 2; }

# ik's build writes llama.log into its working directory
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

failed=0
printf '%-40s %12s %12s %8s\n' file reference ours drift
for gguf in "${ggufs[@]}"; do
    name=$(basename "$gguf")
    # mainline logs `Final estimate: PPL = x` to stderr, ik `... PPL over 1 chunks for n_ctx=512 = x` to stdout
    expected=$( (cd "$scratch" && "$reference" -m "$gguf" -f "$text" -c "$CTX" --chunks 1 -ngl 99) \
        > "$scratch/$name.reference.log" 2>&1 || true
        grep 'Final estimate: PPL' "$scratch/$name.reference.log" | sed 's/.* = //' | awk '{print $1}' || true)
    actual=$("$ours" --gguf "$gguf" --file "$text" --llama-cpp-ctx "$CTX" 2> "$scratch/$name.ours.log" \
        | sed -n 's/^PPL = //p' || true)
    if [[ -z $expected || -z $actual ]]; then
        printf '%-40s %12s %12s %8s\n' "$name" "${expected:-error}" "${actual:-error}" "-"
        [[ -z $expected ]] && tail -n 5 "$scratch/$name.reference.log" >&2
        [[ -z $actual ]] && tail -n 5 "$scratch/$name.ours.log" >&2
        failed=1
        continue
    fi
    read -r drift over < <(awk -v a="$actual" -v e="$expected" -v t="$tolerance" \
        'BEGIN { d = a / e - 1; if (d < 0) d = -d; over = d > t; printf "%.4f %d\n", d, over }')
    printf '%-40s %12s %12s %8s\n' "$name" "$expected" "$actual" "$drift"
    if [[ $over == 1 ]]; then
        failed=1
    fi
done
exit "$failed"

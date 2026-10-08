#!/usr/bin/env bash
# Regenerates the GGUF directories the parity checks score, from a Hugging Face Qwen3.5-0.8B checkout and the references
# scripts/build_gguf_references.sh builds: <out>/iq (mainline IQ types), <out>/kt (ik trellis types, pure) and
# <out>/iqk (ik types in ik's default mixes), plus <out>/work with the F16 GGUF and both imatrices.
#
# Usage: scripts/make_gguf_test_dirs.sh <qwen3.5-0.8b hf dir> <out> [reference dir]
#   reference dir  default $INFERENCE_REFERENCE_DIR, else ~/.local/share/inference-rs/references
set -euo pipefail

IQ_TYPES=(IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S)
KT_TYPES=(IQ1_KT IQ2_KT IQ3_KT IQ4_KT)
IQK_TYPES=(IQ2_K IQ2_KS IQ2_KL IQ2_KT IQ3_K IQ3_KS IQ4_K IQ4_KS IQ4_KSS IQ5_K IQ5_KS IQ6_K)
IMATRIX_CTX=512
MAINLINE_IMATRIX_CHUNKS=32

if [[ $# -lt 2 ]]; then
    sed -n '6,7p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
[[ -f $1/config.json ]] || { echo "no Hugging Face checkpoint at $1" >&2; exit 2; }
hf=$(realpath "$1")
out=$2
refs=${3:-${INFERENCE_REFERENCE_DIR:-$HOME/.local/share/inference-rs/references}}
mainline=$refs/llama.cpp
ik=$refs/ik_llama.cpp
for bin in "$mainline/build/bin/llama-quantize" "$ik/build/bin/llama-quantize" "$refs/venv/bin/python"; do
    [[ -x $bin ]] || { echo "no $bin; run scripts/build_gguf_references.sh first" >&2; exit 2; }
done
mkdir -p "$out"/{work,iq,kt,iqk}
out=$(realpath "$out")
work=$out/work
f16=$work/qwen35-0.8b-f16.gguf

# The F16 and imatrices are kept across runs, so drop them when the references moved
pins="$(git -C "$mainline" rev-parse HEAD) $(git -C "$ik" rev-parse HEAD)"
if [[ $(cat "$work/pins" 2>/dev/null || true) != "$pins" ]]; then
    rm -f "$work"/*
    echo "$pins" > "$work/pins"
fi

# The MTP layer has no imatrix entries, and quantizing with it present fails
# Outputs are written to .tmp and moved into place, so an interrupted run never leaves a file the next one reuses
if [[ ! -f $f16 ]]; then
    "$refs/venv/bin/python" "$mainline/convert_hf_to_gguf.py" "$hf" --no-mtp --outtype f16 --outfile "$f16.tmp"
    mv "$f16.tmp" "$f16"
fi

# Calibration text: mainline's README and build guide at the pinned commit
cat "$mainline/README.md" "$mainline/docs/build.md" > "$work/calib.txt"

# ik's quantize cannot read mainline's GGUF-format imatrix, so each side computes its own
if [[ ! -f $work/mainline.imatrix ]]; then
    "$mainline/build/bin/llama-imatrix" -m "$f16" -f "$work/calib.txt" -o "$work/mainline.imatrix.tmp" \
        -c "$IMATRIX_CTX" --chunks "$MAINLINE_IMATRIX_CHUNKS" -ngl 99 > "$work/imatrix.log" 2>&1
    mv "$work/mainline.imatrix.tmp" "$work/mainline.imatrix"
fi
if [[ ! -f $work/ik.imatrix ]]; then
    (cd "$work" && "$ik/build/bin/llama-imatrix" -m "$f16" -f "$work/calib.txt" -o "$work/ik.imatrix.tmp" \
        -c "$IMATRIX_CTX" -ngl 99 > "$work/ik-imatrix.log" 2>&1)
    mv "$work/ik.imatrix.tmp" "$work/ik.imatrix"
fi

quantize() {
    local bin=$1 imatrix=$2 dir=$3 type=$4
    shift 4
    "$bin" "$@" --imatrix "$imatrix" "$f16" "$out/$dir/qwen35-0.8b-$type.gguf" "$type" \
        > "$work/quantize-$dir-$type.log" 2>&1 || { tail -n 5 "$work/quantize-$dir-$type.log" >&2; exit 1; }
    echo "$dir/qwen35-0.8b-$type.gguf"
}

for t in "${IQ_TYPES[@]}"; do
    quantize "$mainline/build/bin/llama-quantize" "$work/mainline.imatrix" iq "$t"
done
# ik's default KT mixes pull in IQ*_K tensors; the KT directory checks the trellis types alone
for t in "${KT_TYPES[@]}"; do
    quantize "$ik/build/bin/llama-quantize" "$work/ik.imatrix" kt "$t" --pure --token-embedding-type q8_0
done
for t in "${IQK_TYPES[@]}"; do
    quantize "$ik/build/bin/llama-quantize" "$work/ik.imatrix" iqk "$t"
done

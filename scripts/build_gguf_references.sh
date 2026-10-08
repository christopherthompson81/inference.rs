#!/usr/bin/env bash
# Builds the llama.cpp and ik_llama.cpp references the GGUF parity checks run against, at pinned commits, into
# <dir>/{llama.cpp,ik_llama.cpp}/build/bin, plus <dir>/venv with mainline's pinned convert_hf_to_gguf.py requirements.
# Not part of local_ci.sh; see docs/src/content/docs/developer/deep-checks.md.
#
# Usage: scripts/build_gguf_references.sh [dir]   (default $INFERENCE_REFERENCE_DIR, else ~/.local/share/inference-rs/references)
set -euo pipefail

# The commits the recorded parity results (GGUF investigation, Run 15) were taken against
LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git
LLAMA_CPP_COMMIT=4a89937354190cef5a97baf8eeb17336105eb72d
IK_LLAMA_CPP_REPO=https://github.com/ikawrakow/ik_llama.cpp.git
IK_LLAMA_CPP_COMMIT=5f89bfc81268b4d56d2af63ccbed59de17c64c09
TARGETS=(llama-quantize llama-imatrix llama-perplexity llama-simple)

dir=${1:-${INFERENCE_REFERENCE_DIR:-$HOME/.local/share/inference-rs/references}}
mkdir -p "$dir"
dir=$(realpath "$dir")

# the IQ, KT and IQK references run on CUDA when a toolkit is present, as the recorded results did
cuda=(-DGGML_CUDA=OFF)
if command -v nvcc >/dev/null 2>&1; then
    cuda=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native)
fi

build() {
    local name=$1 repo=$2 commit=$3 src="$dir/$1"
    [[ -d $src/.git ]] || git init -q "$src"
    git -C "$src" remote get-url origin >/dev/null 2>&1 || git -C "$src" remote add origin "$repo"
    if [[ $(git -C "$src" rev-parse -q --verify HEAD || true) != "$commit" ]]; then
        git -C "$src" fetch -q --depth 1 origin "$commit"
        git -C "$src" checkout -q --detach FETCH_HEAD
    fi
    echo "$name at $commit"
    cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF "${cuda[@]}" > "$src/cmake.log" 2>&1 \
        || { tail -n 20 "$src/cmake.log" >&2; exit 1; }
    cmake --build "$src/build" -j "$(nproc)" --target "${TARGETS[@]}" > "$src/build.log" 2>&1 \
        || { tail -n 20 "$src/build.log" >&2; exit 1; }
}

build llama.cpp "$LLAMA_CPP_REPO" "$LLAMA_CPP_COMMIT"
build ik_llama.cpp "$IK_LLAMA_CPP_REPO" "$IK_LLAMA_CPP_COMMIT"

# A venv keeps the conversion off whatever torch the user site has
[[ -x $dir/venv/bin/pip ]] || python3 -m venv --clear "$dir/venv"
"$dir/venv/bin/pip" install -q -r "$dir/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt" > "$dir/venv.log" 2>&1 \
    || { tail -n 20 "$dir/venv.log" >&2; exit 1; }
echo "mainline: $dir/llama.cpp/build/bin"
echo "ik:       $dir/ik_llama.cpp/build/bin"

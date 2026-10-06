#!/usr/bin/env bash
# Deep checks on real checkpoints, run on request rather than by local_ci.sh: parity with recorded reference outputs
# (PaddleOCR-VL against transformers, IQ4_XS against llama.cpp) and end-to-end runs on real weights (Qwen3.5 MTP
# drafting, FLUX image generation, layout detection through the C ABI). Each test skips unless its INFERENCE_TEST_*
# path is set in ~/.cargo/config.toml [env]; the set is the `deep` profile in .config/nextest.toml.
#
# Usage: scripts/deep_checks.sh [nextest args...]   (e.g. `-E 'test(/^qwen3_5_mtp::/)'` to narrow)
set -euo pipefail
cd "$(dirname "$0")/.."

# the CUDA build is what most of these check; without a toolkit they run on the CPU, where the GPU-only ones skip
features=()
if command -v nvcc >/dev/null 2>&1; then
    features=(--features cuda)
fi
cargo nextest run --no-fail-fast --profile deep "${features[@]}" --workspace --lib --bins --tests "$@"

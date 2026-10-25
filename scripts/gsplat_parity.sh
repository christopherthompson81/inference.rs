#!/usr/bin/env bash
# One-off parity check of inference-nn's Gaussian splatting kernels against gsplat with the MaskGaussian gate (the
# HY-World 2.0 fork, built in place): renders one camera of a trained checkpoint on both sides and compares the render
# and every parameter gradient. Not part of local_ci.sh: it needs the fork's build and a trained scene.
#
# Usage: scripts/gsplat_parity.sh <python> <gsplat-fork-dir> <checkpoint.pt>
#   python           an interpreter with torch (CUDA)
#   gsplat-fork-dir  the fork's checkout with its extension built in place (put first on PYTHONPATH)
#   checkpoint.pt    a world_gs_trainer checkpoint (its "splats": means, quats, scales, opacities, sh0, mask_score)
set -euo pipefail

if [[ $# -lt 3 ]]; then
    sed -n '6,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
python=$1
fork=$(realpath "$2")
ckpt=$(realpath "$3")
root=$(cd "$(dirname "$0")/.." && pwd)
fixture=$(mktemp -d)
trap 'rm -rf "$fixture"' EXIT

PYTHONPATH="$fork" "$python" "$root/scripts/gsplat_parity.py" "$ckpt" "$fixture"
INFERENCE_GSPLAT_FIXTURE="$fixture" cargo test --manifest-path "$root/Cargo.toml" -p inference-nn --features cuda \
    --lib cuda::gsplat::tests::matches_gsplat_fixture -- --ignored --nocapture

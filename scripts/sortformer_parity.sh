#!/usr/bin/env bash
# One-off parity check of our Streaming Sortformer v2 against NeMo's own `forward_streaming` at the checkpoint's
# schedule: the log-mel features and the per-frame speaker probabilities. Not part of local_ci.sh: it needs
# nemo_toolkit (2.6+) with torch, so NEMO_PYTHON names the interpreter that has them (default python3).
#
# Usage: scripts/sortformer_parity.sh <.nemo> <wav>... [--cpu]
#   .nemo  nvidia/diar_streaming_sortformer_4spk-v2.1's checkpoint
#   wav    16-bit PCM wav files, mono, 16 kHz
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '6,8p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
nemo=$(realpath "$1")
shift
wavs=()
cpu=()
for arg in "$@"; do
    case $arg in
        --cpu) cpu=(--cpu) ;;
        *) wavs+=("$(realpath "$arg")") ;;
    esac
done
root=$(cd "$(dirname "$0")/.." && pwd)
dumps=$(mktemp -d)
trap 'rm -rf "$dumps"' EXIT

# NeMo on CPU: its top-k tie order is a CPU kernel's, the one the cache's near-ties were studied against
"${NEMO_PYTHON:-python3}" -P -W ignore - "$nemo" "$dumps" "${wavs[@]}" <<'PY'
import logging, math, sys, wave
import numpy as np, torch
from safetensors.torch import save_file
logging.disable(logging.WARNING)
from nemo.collections.asr.models import SortformerEncLabelModel

nemo, out, *wavs = sys.argv[1:]
model = SortformerEncLabelModel.restore_from(nemo, map_location="cpu").eval()
sub = model.encoder.subsampling_factor
for n, path in enumerate(wavs):
    w = wave.open(path)
    assert w.getframerate() == 16000 and w.getnchannels() == 1, f"{path}: 16 kHz mono only"
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    with torch.inference_mode():
        signal = torch.from_numpy(pcm).unsqueeze(0)
        mel, mel_len = model.preprocessor(input_signal=signal, length=torch.tensor([len(pcm)]))
        probs = model.forward_streaming(mel, mel_len)
    valid = int(mel_len[0])
    save_file({
        "pcm": torch.from_numpy(np.ascontiguousarray(pcm)),
        "mel": mel[0, :, :valid].T.contiguous(),
        "probs": probs[0, :math.ceil(valid / sub)].contiguous(),
    }, f"{out}/{n}.safetensors")
print(f"reference: {len(wavs)} clips")
PY

features=()
if [[ ${#cpu[@]} -eq 0 ]] && command -v nvcc >/dev/null 2>&1; then
    features=(--features cuda)
fi
cargo run --manifest-path "$root/Cargo.toml" -q -p inference-models-speech --example sortformer_parity \
    "${features[@]}" -- --nemo "$nemo" --dumps "$dumps" "${cpu[@]}"

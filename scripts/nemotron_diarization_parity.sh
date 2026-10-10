#!/usr/bin/env bash
# One-off parity check of our Nemotron-3 Diarization against transformers' on fixed audio: per-frame speaker
# probabilities within a tolerance over the valid frames, and the segments transformers' processor cuts.
# Not part of local_ci.sh: it needs torch, transformers (5.x, with nemotron3_diarization) and librosa.
#
# Usage: scripts/nemotron_diarization_parity.sh <checkpoint dir> <wav>... [--cpu] [--bf16]
#   dir   holds config.json, processor_config.json and model.safetensors (nvidia/Nemotron-3-Diarization)
#   wav   16-bit PCM wav files, mono, 16 kHz
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '6,8p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
model=$(realpath "$1")
shift
wavs=()
cpu=()
bf16=()
for arg in "$@"; do
    case $arg in
        --cpu) cpu=(--cpu) ;;
        --bf16) bf16=(--bf16) ;;
        *) wavs+=("$(realpath "$arg")") ;;
    esac
done
root=$(cd "$(dirname "$0")/.." && pwd)
dumps=$(mktemp -d)
trap 'rm -rf "$dumps"' EXIT

python3 -P -W ignore - "$model" "$dumps" "${wavs[@]}" <<'PY'
import sys, wave
import numpy as np, torch
from safetensors.torch import save_file
from transformers import AutoProcessor, Nemotron3DiarizationForAudioFrameClassification

model_dir, out, *wavs = sys.argv[1:]
proc = AutoProcessor.from_pretrained(model_dir)
model = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
for n, path in enumerate(wavs):
    w = wave.open(path)
    assert w.getframerate() == 16000 and w.getnchannels() == 1, f"{path}: 16 kHz mono only"
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    inputs = proc(pcm, sampling_rate=16000)
    valid = int(inputs["attention_mask"].sum())
    with torch.no_grad():
        logits = model(**inputs).logits
    segments = proc.extract_speaker_dict(logits, inputs["attention_mask"])[0]
    save_file({
        "pcm": torch.from_numpy(np.ascontiguousarray(pcm)),
        "probs": logits.sigmoid()[0, :valid].contiguous(),
        "segments": torch.tensor([[s["Speaker"], round(s["Start"] * 100), round(s["End"] * 100)] for s in segments],
                                 dtype=torch.int64).reshape(-1, 3),
    }, f"{out}/{n}.safetensors")
print(f"reference: {len(wavs)} clips")
PY

features=()
if [[ ${#cpu[@]} -eq 0 ]] && command -v nvcc >/dev/null 2>&1; then
    features=(--features cuda)
fi
cargo run --manifest-path "$root/Cargo.toml" -q -p inference-models-speech --example nemotron_diarization_parity \
    "${features[@]}" -- --model "$model" --dumps "$dumps" "${cpu[@]}" "${bf16[@]}"

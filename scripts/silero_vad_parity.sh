#!/usr/bin/env bash
# One-off parity check of our Silero VAD against the official package's TorchScript model on fixed audio: per-chunk
# probabilities within a tolerance, and `get_speech_timestamps`'s segments exactly.
# Not part of local_ci.sh: it needs torch and the `silero-vad` package (pip install --no-deps silero-vad).
#
# Usage: scripts/silero_vad_parity.sh <silero gguf> <wav>... [--max-speech-s N] [--cpu]
#   gguf  made by `cargo run -p inference-models-speech --example silero_vad_gguf -- --weights <safetensors> <out>`
#   wav   16-bit PCM wav files, mono, 16 kHz
#   N     max_speech_duration_s for both sides, which exercises the split at the longest silence
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '6,8p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
gguf=$(realpath "$1")
shift
wavs=()
cpu=()
max_speech=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu) cpu=(--cpu) ;;
        --max-speech-s) max_speech=(--max-speech-s "$2"); shift ;;
        *) wavs+=("$(realpath "$1")") ;;
    esac
    shift
done
root=$(cd "$(dirname "$0")/.." && pwd)
dumps=$(mktemp -d)
trap 'rm -rf "$dumps"' EXIT

python3 -P -W ignore - "$dumps" "${max_speech[1]:-inf}" "${wavs[@]}" <<'PY'
import sys, wave
import numpy as np, torch
from safetensors.torch import save_file
from silero_vad import get_speech_timestamps, load_silero_vad

out, max_speech, *wavs = sys.argv[1:]
model = load_silero_vad()
for n, path in enumerate(wavs):
    w = wave.open(path)
    assert w.getframerate() == 16000 and w.getnchannels() == 1, f"{path}: 16 kHz mono only"
    pcm = torch.from_numpy(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0)
    model.reset_states()
    probs = []
    with torch.no_grad():
        for start in range(0, len(pcm), 512):
            chunk = pcm[start:start + 512]
            chunk = torch.nn.functional.pad(chunk, (0, 512 - len(chunk)))
            probs.append(model(chunk, 16000).item())
        segments = get_speech_timestamps(pcm, model, sampling_rate=16000, max_speech_duration_s=float(max_speech))
    save_file({
        "pcm": pcm.contiguous(),
        "probs": torch.tensor(probs, dtype=torch.float32),
        "segments": torch.tensor([[s["start"], s["end"]] for s in segments], dtype=torch.int64).reshape(-1, 2),
    }, f"{out}/{n}.safetensors")
print(f"reference: {len(wavs)} clips")
PY

features=()
if [[ ${#cpu[@]} -eq 0 ]] && command -v nvcc >/dev/null 2>&1; then
    features=(--features cuda)
fi
cargo run --manifest-path "$root/Cargo.toml" -q -p inference-models-speech --example silero_vad_parity "${features[@]}" -- \
    --gguf "$gguf" --dumps "$dumps" "${max_speech[@]}" "${cpu[@]}"

#!/usr/bin/env bash
# One-off parity check of our `.nemo` loading against NeMo itself, for recognisers with no transformers port (the
# hybrid RNN-T/CTC FastConformers): features, encoder output, and the greedy hypothesis's tokens, frames and text.
# Not part of local_ci.sh: it needs nemo_toolkit (2.6+) with torch, so NEMO_PYTHON names the interpreter that has
# them (default python3).
#
# Usage: scripts/nemo_asr_parity.sh <.nemo> <wav>... [--cpu]
#   .nemo  e.g. nvidia/stt_en_fastconformer_hybrid_large_pc's checkpoint
#   wav    16-bit PCM wav files, mono, 16 kHz
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '7,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
nemo=$(realpath "$1")
shift
wavs=()
cpu=()
for arg in "$@"; do
    if [[ $arg == --cpu ]]; then cpu=(--cpu); else wavs+=("$(realpath "$arg")"); fi
done
root=$(cd "$(dirname "$0")/.." && pwd)
dumps=$(mktemp -d)
trap 'rm -rf "$dumps"' EXIT

"${NEMO_PYTHON:-python3}" -P -W ignore - "$nemo" "$dumps" "${wavs[@]}" <<'PY'
import json, logging, sys, wave
import numpy as np, torch
from safetensors.torch import save_file
logging.disable(logging.WARNING)
from nemo.collections.asr.models import ASRModel

nemo, out, *wavs = sys.argv[1:]
model = ASRModel.restore_from(nemo, map_location="cpu").eval()
for n, path in enumerate(wavs):
    w = wave.open(path)
    assert w.getframerate() == 16000 and w.getnchannels() == 1, f"{path}: 16 kHz mono only"
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    with torch.inference_mode():
        signal, length = torch.from_numpy(pcm).unsqueeze(0), torch.tensor([len(pcm)])
        mel, mel_len = model.preprocessor(input_signal=signal, length=length)
        encoded, enc_len = model.encoder(audio_signal=mel, length=mel_len)
        hyp = model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=enc_len, return_hypotheses=True
        )[0]
    tokens = hyp.y_sequence.tolist() if torch.is_tensor(hyp.y_sequence) else list(hyp.y_sequence)
    frames = hyp.timestamp.tolist() if torch.is_tensor(hyp.timestamp) else list(hyp.timestamp)
    emissions = [(t, f, 1) for t, f in zip(tokens, frames)]
    save_file({
        "pcm": torch.from_numpy(np.ascontiguousarray(pcm)),
        "features": mel[0, :, : int(mel_len[0])].T.contiguous(),
        "encoded": encoded[0, :, : int(enc_len[0])].T.contiguous(),
        "emissions": torch.tensor(emissions, dtype=torch.int64).reshape(-1, 3),
    }, f"{out}/{n}.safetensors")
    with open(f"{out}/{n}.json", "w") as f:
        json.dump({"wav": path, "text": model.tokenizer.ids_to_text(tokens).strip()}, f)
print(f"reference: {len(wavs)} clips")
PY

features=()
if [[ ${#cpu[@]} -eq 0 ]] && command -v nvcc >/dev/null 2>&1; then
    features=(--features cuda)
fi
# the example's `--model` directory goes unread when `--nemo` names the checkpoint
cargo run --manifest-path "$root/Cargo.toml" -q -p inference-models-speech --example parakeet_parity "${features[@]}" -- \
    --model "$dumps" --dumps "$dumps" --nemo "$nemo" "${cpu[@]}"

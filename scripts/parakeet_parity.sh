#!/usr/bin/env bash
# One-off parity check of our Parakeet (and the streaming Nemotron ASR, offline) against transformers' on fixed audio: the features and encoder output must
# reach a minimum cosine, and the emitted tokens, their frames and the transcript must match exactly.
# Not part of local_ci.sh: it needs torch, transformers (5.x, with Parakeet) and librosa.
#
# Usage: scripts/parakeet_parity.sh <checkpoint dir> <wav>... [--cpu]
#   dir   holds config.json, processor_config.json, tokenizer.json and model.safetensors (e.g. parakeet-tdt-0.6b-v3)
#   wav   16-bit PCM wav files (mono, any rate: both sides resample to 16 kHz)
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '6,8p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
model=$(realpath "$1")
shift
wavs=()
cpu=()
for arg in "$@"; do
    if [[ $arg == --cpu ]]; then cpu=(--cpu); else wavs+=("$(realpath "$arg")"); fi
done
root=$(cd "$(dirname "$0")/.." && pwd)
dumps=$(mktemp -d)
trap 'rm -rf "$dumps"' EXIT

python3 -P -W ignore - "$model" "$dumps" "${wavs[@]}" <<'PY'
import json, sys, wave
import librosa, numpy as np, torch
from safetensors.torch import save_file
import transformers
from transformers import AutoConfig, AutoProcessor

model_dir, out, *wavs = sys.argv[1:]
config = AutoConfig.from_pretrained(model_dir)
classes = {
    "parakeet_ctc": "ParakeetForCTC", "parakeet_rnnt": "ParakeetForRNNT", "parakeet_tdt": "ParakeetForTDT",
    "nemotron_asr_streaming": "NemotronAsrStreamingForRNNT", "nemotron3_5_asr": "Nemotron3_5AsrForRNNT",
}
cls = getattr(transformers, classes[config.model_type])
prompted = config.model_type == "nemotron3_5_asr"
proc = AutoProcessor.from_pretrained(model_dir)
model = cls.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
rate = proc.feature_extractor.sampling_rate
for n, path in enumerate(wavs):
    w = wave.open(path)
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
    if w.getframerate() != rate:
        pcm = librosa.resample(pcm, orig_sr=w.getframerate(), target_sr=rate)
    inputs = proc(pcm, sampling_rate=rate)
    valid = int(inputs["attention_mask"].sum())
    with torch.no_grad():
        enc = model.encoder(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"])
        frames = int(enc.attention_mask.sum())
        hidden = enc.last_hidden_state[:, :frames]
        if prompted:
            # the language prompt (auto) joins each frame, as our encoder output carries it
            one_hot = torch.nn.functional.one_hot(torch.tensor([config.default_prompt_id]), config.num_prompts)
            one_hot = one_hot.to(hidden.dtype)[:, None, :].expand(-1, hidden.shape[1], -1)
            hidden = model.prompt_projector(torch.cat([hidden, one_hot], dim=-1))
        hidden = hidden[0]
        gen = model.generate(**inputs)
    emissions = []
    if config.model_type == "parakeet_ctc":
        seq = gen.sequences[0] if hasattr(gen, "sequences") else gen[0]
        prev = None
        for f, t in enumerate(seq[:frames].tolist()):
            if t != config.pad_token_id and t != prev:
                emissions.append((t, f, 1))
            prev = t
        text = proc.decode(seq, skip_special_tokens=True)
    else:
        seq, dur = gen.sequences[0].tolist(), gen.durations[0].tolist()
        frame = 0
        for i, tok in enumerate(seq):
            if i > 0 and tok != config.blank_token_id and tok != config.pad_token_id:
                span = dur[i] if config.model_type == "parakeet_tdt" else 1
                emissions.append((tok, frame, span))
            frame += dur[i]
        text = proc.decode(gen.sequences, durations=gen.durations, skip_special_tokens=True)[0][0]
    save_file({
        "pcm": torch.from_numpy(np.ascontiguousarray(pcm)),
        "features": inputs["input_features"][0, :valid].contiguous(),
        "encoded": hidden.contiguous(),
        "emissions": torch.tensor(emissions, dtype=torch.int64).reshape(-1, 3),
    }, f"{out}/{n}.safetensors")
    with open(f"{out}/{n}.json", "w") as f:
        json.dump({"wav": path, "text": text.strip()}, f)
print(f"reference: {len(wavs)} clips")
PY

features=()
if [[ ${#cpu[@]} -eq 0 ]] && command -v nvcc >/dev/null 2>&1; then
    features=(--features cuda)
fi
cargo run --manifest-path "$root/Cargo.toml" -q -p inference-models-speech --example parakeet_parity "${features[@]}" -- \
    --model "$model" --dumps "$dumps" "${cpu[@]}"

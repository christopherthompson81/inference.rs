#!/usr/bin/env bash
# One-off parity check of our Cohere Transcribe against transformers' `CohereAsrForConditionalGeneration` (5.19):
# the log-mel features, the encoder output and the greedy token ids of each clip, in f32 with TF32 off. The
# reference's feature extractor adds a 1e-5 dither seeded by torch's generator; it is off here on both sides.
# Not part of local_ci.sh: it needs torch and transformers 5.19, the checkpoint directory, and a CUDA build.
#
# Usage: scripts/cohere_asr_parity.sh <checkpoint dir> <language> <clip.wav>...
#   COHERE_CPU=1 runs ours on the CPU; COHERE_DTYPE=bf16 or f16 runs ours at that precision
set -euo pipefail

if [[ $# -lt 3 ]]; then
    sed -n '7,8p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
model=$(realpath "$1")
language=$2
shift 2
root=$(cd "$(dirname "$0")/.." && pwd)
dumps=$(mktemp -d)
trap 'rm -rf "$dumps"' EXIT

cd "$root"
cargo build -q -p inference-models-speech --example cohere_asr_parity --features cuda
for clip in "$@"; do
    python3 -P -W ignore - "$model" "$language" "$(realpath "$clip")" "$dumps/clip.wav" "$dumps/ref.safetensors" <<'PY'
import sys
import soundfile as sf
import torch
import torchaudio
from safetensors.torch import save_file
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
model_dir, language, clip, out_wav, out = sys.argv[1:6]
wav, sr = sf.read(clip, dtype="float32", always_2d=True)
wav = torch.from_numpy(wav.mean(1))
wav = torchaudio.functional.resample(wav, sr, 16000) if sr != 16000 else wav
sf.write(out_wav, wav.numpy(), 16000, subtype="FLOAT")
proc = AutoProcessor.from_pretrained(model_dir)
proc.feature_extractor.dither = 0.0
m = CohereAsrForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).to("cuda").eval()
inputs = proc(wav.numpy(), sampling_rate=16000, return_tensors="pt", language=language)
chunks = inputs.pop("audio_chunk_index")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
with torch.no_grad():
    ids = m.generate(**inputs, max_new_tokens=256, do_sample=False)
    first = {k: v[:1] for k, v in inputs.items()}
    frames = int(first["attention_mask"].sum())
    encoded = m.get_encoder()(input_features=first["input_features"], attention_mask=first["attention_mask"]).last_hidden_state
prompt = inputs["decoder_input_ids"].shape[1]
dump = {"features": first["input_features"][0, :frames].cpu().contiguous(), "encoded": encoded[0].float().cpu().contiguous()}
for i in range(ids.shape[0]):
    row = ids[i, prompt:].tolist()
    row = row[: row.index(3)] if 3 in row else row
    dump[f"ids.{i}"] = torch.tensor(row, dtype=torch.int64)
save_file(dump, out)
text = proc.decode(ids, skip_special_tokens=True, audio_chunk_index=chunks, language=language)
print("reference:", text[0] if isinstance(text, list) else text)
PY
    target/debug/examples/cohere_asr_parity --model "$model" --language "$language" --wav "$dumps/clip.wav" \
        --dump "$dumps/ref.safetensors" ${COHERE_CPU:+--cpu} --dtype "${COHERE_DTYPE:-f32}"
done

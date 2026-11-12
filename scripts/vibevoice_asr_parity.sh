#!/usr/bin/env bash
# One-off parity check of our VibeVoice-ASR against transformers' `VibeVoiceAsrForConditionalGeneration` (5.19), F32 on
# the CPU: the normalized clip and its audio embeddings (`examples/vibevoice_audio_parity.rs`), then the greedy answer
# through our server against the reference's. The reference adds Gaussian noise to the acoustic latents; it is off
# here (`vae_std` 0), as ours leaves it out. Each 7B pass takes about two minutes on 16 cores.
# Not part of local_ci.sh: it needs torch and transformers 5.19, the checkpoint (about 30 GB of RAM in F32), and a CUDA
# build (the F32 runs are on the CPU).
#
# Usage: scripts/vibevoice_asr_parity.sh <checkpoint dir> <clip.wav>...
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '9p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
model=$(realpath "$1")
shift
root=$(cd "$(dirname "$0")/.." && pwd)
dumps=$(mktemp -d)
port=18499
trap 'rm -rf "$dumps"; [[ -n ${server:-} ]] && kill "$server" 2>/dev/null || true' EXIT

cd "$root"
cargo build -q -p inference-models-qwen --example vibevoice_audio_parity
scripts/build_cli.sh cuda > /dev/null
for clip in "$@"; do
    python3 -P -W ignore - "$model" "$(realpath "$clip")" "$dumps/clip.wav" "$dumps/ref.safetensors" <<'PY'
import sys, torch, soundfile as sf, torchaudio
from safetensors.torch import save_file
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
model_dir, clip, out_wav, out = sys.argv[1:5]
proc = AutoProcessor.from_pretrained(model_dir)
m = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
m.config.acoustic_tokenizer_encoder_config.vae_std = 0.0
wav, sr = sf.read(clip, dtype="float32", always_2d=True)
wav = torch.from_numpy(wav.mean(1))
wav = torchaudio.functional.resample(wav, sr, 24000) if sr != 24000 else wav
sf.write(out_wav, wav.numpy(), 24000, subtype="FLOAT")
inputs = proc.apply_transcription_request(audio=wav.numpy())
with torch.no_grad():
    emb = m.model.get_audio_features(input_values=inputs["input_values"], padding_mask=inputs["padding_mask"]).pooler_output
    gen = m.generate(**inputs, max_new_tokens=2048, do_sample=False)
new = gen[0, inputs["input_ids"].shape[1]:]
save_file({"audio": emb.float().contiguous(), "input_values": inputs["input_values"][0].float().contiguous()}, out)
print("reference prompt tokens:", inputs["input_ids"].shape[1])
print("reference:", proc.tokenizer.decode(new, skip_special_tokens=True).strip())
PY
    target/debug/examples/vibevoice_audio_parity "$model" "$dumps/clip.wav" "$dumps/ref.safetensors"
    if [[ -z ${server:-} ]]; then
        target/debug/inference serve -m "$model" --cpu --dtype f32 -p "$port" > "$dumps/serve.log" 2>&1 &
        server=$!
        until curl -s "localhost:$port/health" > /dev/null; do sleep 2; done
    fi
    python3 - "$dumps/clip.wav" "$port" <<'PY'
import base64, json, sys, urllib.request
wav, port = sys.argv[1:3]
part = {"type": "input_audio", "input_audio": {"data": base64.b64encode(open(wav, "rb").read()).decode(), "format": "wav"}}
body = {"model": "default", "temperature": 0, "max_tokens": 2048, "messages": [{"role": "user", "content": [part]}]}
request = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
response = json.load(urllib.request.urlopen(request, timeout=7200))
print("ours prompt tokens:     ", response["usage"]["prompt_tokens"])
print("ours:     ", response["choices"][0]["message"]["content"].strip())
PY
done

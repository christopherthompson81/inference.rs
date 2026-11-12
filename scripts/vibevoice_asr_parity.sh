#!/usr/bin/env bash
# One-off parity check of our VibeVoice-ASR against transformers' `VibeVoiceAsrForConditionalGeneration` (5.19), F32 on
# the CPU: the normalized clip and its audio embeddings (`examples/vibevoice_audio_parity.rs`), then the greedy answer
# through our server against the reference's. The reference adds Gaussian noise to the acoustic latents; it is off
# here (`vae_std` 0), as ours leaves it out. Each 7B pass takes about two minutes on 16 cores.
# Not part of local_ci.sh: it needs torch and transformers 5.19, the checkpoint (about 30 GB of RAM in F32), and a CUDA
# build (the F32 runs are on the CPU).
#
# --streaming checks a streaming checkpoint (VibeVoice-ASR-Streaming-1.5B/7B) instead, against Microsoft's
# `streaming_generate` (`pip install git+https://github.com/microsoft/VibeVoice`, transformers 4.57): each window's
# greedy token ids, F32 on the CPU, its latent noise (`fix_std`) zeroed; --context passes hotwords to both.
#
# Usage: scripts/vibevoice_asr_parity.sh [--streaming [--context <text>]] <checkpoint dir> <clip.wav>...
set -euo pipefail

streaming=0
context=""
while [[ ${1:-} == --* ]]; do
    case $1 in
        --streaming) streaming=1 ;;
        --context) context=$2; shift ;;
        *) break ;;
    esac
    shift
done
if [[ $# -lt 2 ]]; then
    sed -n '13p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
model=$(realpath "$1")
shift
root=$(cd "$(dirname "$0")/.." && pwd)

if [[ $streaming == 1 ]]; then
    dumps=$(mktemp -d)
    trap 'rm -rf "$dumps"' EXIT
    cd "$root"
    cargo build -q -p inference-models-qwen --example vibevoice_streaming_parity
    for clip in "$@"; do
        python3 -P -W ignore - "$model" "$(realpath "$clip")" "$dumps/ref.safetensors" "$context" <<'PY'
import json, sys, torch, soundfile as sf, torchaudio
from safetensors.torch import save_file
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
model_dir, clip, out, context = sys.argv[1:5]
cfg = json.load(open(f"{model_dir}/preprocessor_config.json"))
frame = cfg["speech_tok_compress_ratio"] / cfg["target_sample_rate"]
tok = VibeVoiceASRProcessor.from_pretrained(model_dir).tokenizer
m = VibeVoiceASRForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
m.model.acoustic_tokenizer.fix_std.zero_()
wav, sr = sf.read(clip, dtype="float32", always_2d=True)
wav = torch.from_numpy(wav.mean(1))
wav = torchaudio.functional.resample(wav, sr, 24000) if sr != 24000 else wav
# each window's ids, as the loop hands them to the tokenizer
ids, decode = [], tok.decode
tok.decode = lambda t, **k: (ids.append(list(t)), decode(t, **k))[1]
with torch.no_grad():
    for i, n, text in m.streaming_generate(audio_tensor=wav, tokenizer=tok, chunk_duration=cfg["chunk_frames"] * frame,
                                           text_audio_delay=cfg["lookahead_frames"] * frame, sample_rate=24000,
                                           max_new_tokens_per_chunk=256, temperature=0.0, context_info=context or None):
        print(f"reference [{i + 1}/{n}] {text!r}")
save_file({"ids": torch.tensor([t for w in ids for t in w + [-1]], dtype=torch.int64)}, out)
PY
        args=("$model" "$(realpath "$clip")" "$dumps/ref.safetensors")
        [[ -n $context ]] && args+=(--context "$context")
        target/debug/examples/vibevoice_streaming_parity "${args[@]}"
    done
    exit 0
fi
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

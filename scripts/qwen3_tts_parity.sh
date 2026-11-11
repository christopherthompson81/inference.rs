#!/usr/bin/env bash
# One-off parity check of our Qwen3-TTS against the reference `qwen-tts` package: greedy decoding in f32 must give the
# reference's prompt ids and codes, and our codec decoder must reach a minimum SNR on the reference's codes.
# Not part of local_ci.sh: it needs torch and qwen-tts 0.1.1 with transformers 4.57.3 (a venv over the system torch:
# `python3 -m venv --system-site-packages venv && venv/bin/pip install transformers==4.57.3 accelerate==1.12.0 einops
# && venv/bin/pip install --no-deps qwen-tts==0.1.1`), the checkpoint, and a CUDA build.
#
# Usage: scripts/qwen3_tts_parity.sh <checkpoint dir> <python> [voice] [instructions]
#   checkpoint    a Qwen3-TTS-12Hz CustomVoice or VoiceDesign directory
#   python        the venv's python
#   voice         a preset voice (CustomVoice), default ryan; ignored for VoiceDesign
#   instructions  how to speak (1.7B), or the voice itself (VoiceDesign)
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '8,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
model=$(realpath "$1")
python=$2
voice=${3:-ryan}
instructions=${4:-}
text="The quick brown fox jumps over the lazy dog. It was a bright cold day in April."
root=$(cd "$(dirname "$0")/.." && pwd)
dumps=$(mktemp -d)
trap 'rm -rf "$dumps"' EXIT

"$python" -P -W ignore - "$model" "$dumps/greedy.safetensors" "$text" "$voice" "$instructions" <<'PY'
import json, sys
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel
from safetensors.torch import save_file

model_dir, out, text, voice, instructions = sys.argv[1:6]
kind = json.load(open(f"{model_dir}/config.json"))["tts_model_type"]
m = Qwen3TTSModel.from_pretrained(model_dir, device_map="cuda:0", dtype=torch.float32)
core = m.model
ids = m._tokenize_texts([m._build_assistant_text(text)])
instruct = [m._tokenize_texts([m._build_instruct_text(instructions)])[0]] if instructions else [None]
speaker = voice if kind == "custom_voice" else None
codes, _ = core.generate(input_ids=ids, instruct_ids=instruct, languages=["English"], speakers=[speaker],
    non_streaming_mode=True, do_sample=False, subtalker_dosample=False, repetition_penalty=1.0, max_new_tokens=600)
wav = core.speech_tokenizer.decode([{"audio_codes": codes[0]}])[0][0]
save_file({"ids": ids[0].cpu().to(torch.int64), "codes": codes[0].cpu().to(torch.int64),
           "wav": torch.from_numpy(np.asarray(wav)).float()}, out)
print(f"reference: {codes[0].shape[0]} frames")
PY

cd "$root"
cargo build -q -p inference-models-speech --example qwen3_tts_parity --features cuda
args=(--model "$model" --dump "$dumps/greedy.safetensors" --text "$text" --language english --voice "$voice")
[[ -n $instructions ]] && args+=(--instructions "$instructions")
# f32 greedy decoding equals the reference's codes on every frame (runs 3, 5 and 6 of the investigation)
target/debug/examples/qwen3_tts_parity "${args[@]}"

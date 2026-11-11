#!/usr/bin/env bash
# One-off parity check of our Qwen3-TTS against the reference `qwen-tts` package: greedy decoding in f32 must give the
# reference's prompt ids and codes, and our codec decoder must reach a minimum SNR on the reference's codes; for a Base
# checkpoint also the clip's x-vector and, with its transcript, the clip's codes from the codec encoder.
# Not part of local_ci.sh: it needs torch and qwen-tts 0.1.1 with transformers 4.57.3 (a venv over the system torch:
# `python3 -m venv --system-site-packages venv && venv/bin/pip install transformers==4.57.3 accelerate==1.12.0 einops
# && venv/bin/pip install --no-deps qwen-tts==0.1.1`), the checkpoint, and a CUDA build.
#
# Usage: scripts/qwen3_tts_parity.sh <checkpoint dir> <python> [voice | clip.wav] [instructions | clip transcript]
#   checkpoint    a Qwen3-TTS-12Hz CustomVoice, VoiceDesign or Base directory
#   python        the venv's python
#   voice         a preset voice (CustomVoice), default ryan; for Base, the WAV clip to clone
#   instructions  how to speak (1.7B), or the voice itself (VoiceDesign); for Base, the clip's transcript (in-context)
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '9,13p' "$0" | sed 's/^# \{0,1\}//'
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
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from safetensors.torch import save_file

# full precision: CUDA convolutions default to TF32, which moves the codec's and the encoders' outputs
torch.backends.cudnn.allow_tf32 = False
model_dir, out, text, voice, instructions = sys.argv[1:6]
kind = json.load(open(f"{model_dir}/config.json"))["tts_model_type"]
m = Qwen3TTSModel.from_pretrained(model_dir, device_map="cuda:0", dtype=torch.float32)
core = m.model
ids = m._tokenize_texts([m._build_assistant_text(text)])
greedy = dict(do_sample=False, subtalker_dosample=False, repetition_penalty=1.0, max_new_tokens=600)
dump = {"ids": ids[0].cpu().to(torch.int64)}
prefix = None
if kind == "base":
    wav, sr = sf.read(voice, dtype="float32")
    ref_text = instructions or None
    items = m.create_voice_clone_prompt(ref_audio=(wav, sr), ref_text=ref_text, x_vector_only_mode=ref_text is None)
    prompt = m._prompt_items_to_voice_clone_prompt(items)
    ref_ids = [m._tokenize_texts([m._build_ref_text(ref_text)])[0]] if ref_text else [None]
    codes, _ = core.generate(input_ids=ids, ref_ids=ref_ids, voice_clone_prompt=prompt, languages=["English"],
        non_streaming_mode=False, **greedy)
    dump["spk"] = prompt["ref_spk_embedding"][0].float().cpu()
    prefix = prompt["ref_code"][0]
    if prefix is not None:
        dump["ref_codes"] = prefix.cpu().to(torch.int64).contiguous()
else:
    instruct = [m._tokenize_texts([m._build_instruct_text(instructions)])[0]] if instructions else [None]
    speaker = voice if kind == "custom_voice" else None
    codes, _ = core.generate(input_ids=ids, instruct_ids=instruct, languages=["English"], speakers=[speaker],
        non_streaming_mode=True, **greedy)
c = codes[0]
full = torch.cat([prefix.to(c.device), c], 0) if prefix is not None else c
wav = core.speech_tokenizer.decode([{"audio_codes": full}])[0][0]
if prefix is not None:
    wav = wav[int(prefix.shape[0] / full.shape[0] * wav.shape[0]):]
dump["codes"] = c.cpu().to(torch.int64).contiguous()
dump["wav"] = torch.from_numpy(np.asarray(wav)).float()
save_file(dump, out)
print(f"reference: {c.shape[0]} frames")
PY

cd "$root"
cargo build -q -p inference-models-speech --example qwen3_tts_parity --features cuda
args=(--model "$model" --dump "$dumps/greedy.safetensors" --text "$text" --language english)
if [[ $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['tts_model_type'])" "$model/config.json") == base ]]; then
    args+=(--reference-audio "$voice")
    [[ -n $instructions ]] && args+=(--reference-text "$instructions")
else
    args+=(--voice "$voice")
    [[ -n $instructions ]] && args+=(--instructions "$instructions")
fi
# f32 greedy decoding equals the reference's codes on every frame (runs 3, 5, 6 and 8 of the investigation)
target/debug/examples/qwen3_tts_parity "${args[@]}"

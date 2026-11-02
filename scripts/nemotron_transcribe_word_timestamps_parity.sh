#!/usr/bin/env bash
# One-off parity check of Nemotron 3.5 Transcribe's word timestamps against transformers': each clip transcribed by a
# running inference.rs server (`verbose_json`, word granularity), then transformers' word timestamp head and processor
# (`get_word_timestamp_logits`, `align_word_timestamps`) time the same transcript; the words must match and each start
# and end lie within a tolerance. A transcript with `<spk:N>` tags aligns with the speaker logits, as ours does.
# Not part of local_ci.sh: it needs torch, torchaudio, scipy, a transformers with nemotron3_5_transcribe, and about
# 90 GB of memory for the checkpoint on the CPU.
#
# Usage: scripts/nemotron_transcribe_word_timestamps_parity.sh <checkpoint dir> <server url> <wav>... [--tolerance s]
#   dir        NVIDIA's checkpoint (config.json, model.safetensors, alignment_tokenizer/), which transformers loads
#   url        an inference.rs server of the model, e.g. http://localhost:1234
#   tolerance  seconds a start or end may differ; default 0.041 (a head frame is 40 ms)
#   PYTHON     the interpreter with that transformers; default python3
set -euo pipefail

DEFAULT_TOLERANCE=0.041

if [[ $# -lt 3 ]]; then
    sed -n '9,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
model=$(realpath "$1")
url=$2
shift 2
wavs=()
tolerance=$DEFAULT_TOLERANCE
while [[ $# -gt 0 ]]; do
    case $1 in
        --tolerance) tolerance=$2; shift ;;
        *) wavs+=("$(realpath "$1")") ;;
    esac
    shift
done
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

for wav in "${wavs[@]}"; do
    name=$(basename "$wav" .wav)
    curl -sf -m 1800 "$url/v1/audio/transcriptions" -F "file=@$wav" -F model=default \
        -F response_format=verbose_json -F 'timestamp_granularities[]=word' > "$work/$name.json"
done

"${PYTHON:-python3}" -P - "$model" "$work" "$tolerance" "${wavs[@]}" <<'EOF'
import json, re, sys
import torch
from transformers import AutoProcessor, Nemotron3_5TranscribeForConditionalGeneration
from transformers.audio_utils import load_audio

model_dir, work, tolerance, wavs = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4:]
torch.set_grad_enabled(False)
processor = AutoProcessor.from_pretrained(model_dir)
model = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu")
failed = False
for wav in wavs:
    name = wav.rsplit("/", 1)[-1][: -len(".wav")]
    ours = json.load(open(f"{work}/{name}.json"))
    inputs = processor.apply_transcription_request(load_audio(wav, sampling_rate=processor.feature_extractor.sampling_rate))
    features, mask = inputs["input_features"], inputs["input_features_mask"]
    audio = model.get_audio_features(features, mask)
    logits = model.get_word_timestamp_logits(audio.asr_last_hidden_state, mask)
    tagged = re.search(r"(?i)<spk:\d+>", ours["text"]) is not None
    ref = processor.align_word_timestamps(ours["text"], logits, mask, audio.speaker_logits if tagged else None)
    expected = [(c["text"], *c["timestamp"]) for c in ref["chunks"]]
    got = [(w["word"], w["start"], w["end"]) for w in ours["words"]]
    if [w[0] for w in expected] != [w[0] for w in got]:
        print(f"{name}: words differ ({len(expected)} against {len(got)} of ours)")
        failed = True
        continue
    gaps = [max(abs(a[1] - b[1]), abs(a[2] - b[2])) for a, b in zip(expected, got)]
    over = sum(gap > tolerance for gap in gaps)
    print(f"{name}: {len(got)} words, {sum(gap < 1e-6 for gap in gaps)} exact, largest gap {max(gaps, default=0):.3f} s,"
          f" {over} past {tolerance} s")
    failed |= over > 0
sys.exit(1 if failed else 0)
EOF

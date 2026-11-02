#!/usr/bin/env bash
# One-off parity check of Nemotron 3.5 Transcribe's word timestamps against transformers', in two steps so the
# inference.rs server is stopped before transformers loads the checkpoint (the two do not fit in memory together):
# `transcribe` sends each clip to a running server (`verbose_json`, word granularity) and keeps the answers; `compare`
# has transformers' word timestamp head and processor (`get_word_timestamp_logits`, `align_word_timestamps`) time the
# same transcripts, and fails unless the words match and each start and end lies within a tolerance. A transcript with
# `<spk:N>` tags aligns with the speaker logits, as ours does.
# Not part of local_ci.sh: `compare` needs torch, torchaudio, scipy, a transformers with nemotron3_5_transcribe, and
# about 90 GB of memory for the checkpoint on the CPU.
#
# Usage: scripts/nemotron_transcribe_word_timestamps_parity.sh transcribe <server url> <out dir> <wav>...
#        scripts/nemotron_transcribe_word_timestamps_parity.sh compare <checkpoint dir> <out dir> <wav>... [--tolerance s]
#   url        an inference.rs server of the model, e.g. http://localhost:1234
#   out        where `transcribe` keeps the answers and `compare` reads them
#   dir        NVIDIA's checkpoint (config.json, model.safetensors, alignment_tokenizer/), which transformers loads
#   tolerance  seconds a start or end may differ; default 0.041 (a head frame is 40 ms)
#   PYTHON     the interpreter with that transformers; default python3
set -euo pipefail

DEFAULT_TOLERANCE=0.041

if [[ $# -lt 4 ]]; then
    sed -n '11,17p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
step=$1
source=$2
mkdir -p "$3"
out=$(realpath "$3")
shift 3
wavs=()
tolerance=$DEFAULT_TOLERANCE
while [[ $# -gt 0 ]]; do
    case $1 in
        --tolerance) tolerance=$2; shift ;;
        *) wavs+=("$(realpath "$1")") ;;
    esac
    shift
done

case $step in
    transcribe)
        for wav in "${wavs[@]}"; do
            curl -sf -m 1800 "$source/v1/audio/transcriptions" -F "file=@$wav" -F model=default \
                -F response_format=verbose_json -F 'timestamp_granularities[]=word' > "$out/$(basename "$wav" .wav).json"
        done
        ;;
    compare)
        "${PYTHON:-python3}" -P - "$(realpath "$source")" "$out" "$tolerance" "${wavs[@]}" <<'EOF'
import json, re, sys
import torch
from transformers import AutoProcessor, Nemotron3_5TranscribeForConditionalGeneration
from transformers.audio_utils import load_audio

model_dir, out, tolerance, wavs = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4:]
torch.set_grad_enabled(False)
processor = AutoProcessor.from_pretrained(model_dir)
model = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu")
failed = False
for wav in wavs:
    name = wav.rsplit("/", 1)[-1][: -len(".wav")]
    ours = json.load(open(f"{out}/{name}.json"))
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
    over = [(a, b) for a, b, gap in zip(expected, got, gaps) if gap > tolerance]
    print(f"{name}: {len(got)} words, {sum(gap < 1e-6 for gap in gaps)} exact, largest gap {max(gaps, default=0):.3f} s,"
          f" {len(over)} past {tolerance} s")
    for a, b in over:
        print(f"  {a[0]!r}: transformers {a[1]:.3f}-{a[2]:.3f}, ours {b[1]:.3f}-{b[2]:.3f}")
    failed |= bool(over)
sys.exit(1 if failed else 0)
EOF
        ;;
    *)
        echo "unknown step $step: transcribe or compare" >&2
        exit 2
        ;;
esac

#!/usr/bin/env bash
# One-off parity check of our Kokoro against the reference `kokoro` package on fixed phoneme input: the reference's
# source noise is recorded and replayed, so durations must match exactly and the waveform reach a minimum SNR.
# Not part of local_ci.sh: it needs torch, the `kokoro` package (pip install --no-deps kokoro loguru) and the release.
#
# Usage: scripts/kokoro_parity.sh <Kokoro-82M dir> [voice] [min SNR dB] [--cpu]
#   dir        holds config.json, kokoro-v1_0.pth and voices/
#   voice      default af_heart
#   min SNR    default 30 (the reference itself, CPU against CUDA, reaches 25-27)
set -euo pipefail

if [[ $# -lt 1 ]]; then
    sed -n '6,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
model=$(realpath "$1")
voice=${2:-af_heart}
min_snr=${3:-30}
cpu=()
[[ ${4:-} == --cpu ]] && cpu=(--cpu)
root=$(cd "$(dirname "$0")/.." && pwd)
dumps=$(mktemp -d)
trap 'rm -rf "$dumps"' EXIT

LOGURU_LEVEL=INFO python3 -W ignore - "$model" "$voice" "$dumps" <<'PY'
import importlib.util, os, sys, types
import torch
from safetensors.torch import save_file

model_dir, voice, out = sys.argv[1:4]
# kokoro/__init__ pulls in the misaki G2P; the model needs none of it
spec = importlib.util.find_spec("kokoro")
pkg = types.ModuleType("kokoro")
pkg.__path__ = [os.path.dirname(spec.origin)]
sys.modules["kokoro"] = pkg
from kokoro.model import KModel

torch.manual_seed(0)
model = KModel(repo_id="hexgrad/Kokoro-82M", config=f"{model_dir}/config.json", model=f"{model_dir}/kokoro-v1_0.pth").eval()
pack = torch.load(f"{model_dir}/voices/{voice}.pt", weights_only=True)

recorded = []
randn_like = torch.randn_like
def record(t, *a, **k):
    r = randn_like(t, *a, **k)
    if t.shape[-1] == 9:
        recorded.append(r.clone())
    return r
torch.randn_like = record

# invented phoneme strings over Kokoro's vocabulary: short, long, punctuated, and one at a faster speed
cases = {
    "short": ("həlˈoʊ wˈɜːld.", 1.0),
    "sentence": ("ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ, ænd ðɛn ɹˈʌnz əwˈeɪ!", 1.0),
    "question": ("dˈuː juː nˈoʊ wˈʌt tˈaɪm ɪt ˈɪz?", 1.0),
    "fast": ("ðɪs ɪz ə fˈæstɚ ɹˈiːdɪŋ ʌv ðə sˈeɪm θˈɪŋ.", 1.3),
}
for name, (ps, speed) in cases.items():
    ids = [model.vocab[p] for p in ps if p in model.vocab]
    ref_s = pack[len(ps) - 1]
    recorded.clear()
    with torch.no_grad():
        audio, dur = model.forward_with_tokens(torch.LongTensor([[0, *ids, 0]]), ref_s, speed)
    save_file({
        "ids": torch.tensor(ids, dtype=torch.int64),
        "ref_s": ref_s.float().contiguous(),
        "noise": recorded[0].contiguous(),
        "speed": torch.tensor([speed], dtype=torch.float32),
        "durations": dur.to(torch.int64).flatten(),
        "audio": audio.float().flatten().contiguous(),
    }, f"{out}/{name}.safetensors")
print(f"reference: {len(cases)} cases")
PY

features=()
if [[ ${#cpu[@]} -eq 0 ]] && command -v nvcc >/dev/null 2>&1; then
    features=(--features cuda)
fi
cargo run --manifest-path "$root/Cargo.toml" -q -p inference-models-speech --example kokoro_parity "${features[@]}" -- \
    --model "$model" --dumps "$dumps" --min-snr "$min_snr" "${cpu[@]}"

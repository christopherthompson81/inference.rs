#!/usr/bin/env bash
# One-off parity check of our VoxLingua107 language identifier against SpeechBrain's EncoderClassifier: the fbank,
# the 256-d embedding and the language posterior of each clip, read at 16 kHz on both sides.
# Not part of local_ci.sh: it needs torch and SpeechBrain (`python3 -m venv --system-site-packages venv &&
# venv/bin/pip install speechbrain==1.1.2`), the checkpoint directory, and a CUDA build.
#
# Usage: scripts/voxlingua_parity.sh <checkpoint dir> <python> <clip.wav>...  (VOXLINGUA_CPU=1 runs ours on the CPU)
set -euo pipefail

if [[ $# -lt 3 ]]; then
    sed -n '7p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
model=$(realpath "$1")
python=$2
shift 2
root=$(cd "$(dirname "$0")/.." && pwd)
dumps=$(mktemp -d)
trap 'rm -rf "$dumps"' EXIT

cd "$root"
cargo build -q -p inference-models-speech --example voxlingua_parity --features cuda
for clip in "$@"; do
    "$python" -P -W ignore - "$model" "$(realpath "$clip")" "$dumps/clip.wav" "$dumps/ref.safetensors" <<'PY'
import sys
import soundfile as sf
import torch
import torchaudio
from safetensors.torch import save_file
from speechbrain.inference.classifiers import EncoderClassifier

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
model_dir, clip, out_wav, out = sys.argv[1:5]
wav, sr = sf.read(clip, dtype="float32", always_2d=True)
wav = torch.from_numpy(wav.mean(1))
wav = torchaudio.functional.resample(wav, sr, 16000) if sr != 16000 else wav
wav = wav[: 60 * 16000]
sf.write(out_wav, wav.numpy(), 16000, subtype="FLOAT")
m = EncoderClassifier.from_hparams(source=model_dir, savedir=model_dir, run_opts={"device": "cuda:0"})
batch = wav.unsqueeze(0).to("cuda:0")
with torch.no_grad():
    feats = m.mods.compute_features(batch)
    feats = m.mods.mean_var_norm(feats, torch.ones(1, device="cuda:0"))
    emb = m.mods.embedding_model(feats, torch.ones(1, device="cuda:0"))
    logp = m.mods.classifier(emb).squeeze()
save_file({"fbank": feats[0].cpu().contiguous(), "embedding": emb.flatten().cpu().contiguous(),
           "probabilities": logp.exp().cpu().contiguous()}, out)
top = logp.exp().topk(3)
print("reference:", [(m.hparams.label_encoder.decode_ndim(int(i)), round(float(p), 4)) for p, i in zip(*top)])
PY
    target/debug/examples/voxlingua_parity --model "$model" --wav "$dumps/clip.wav" --dump "$dumps/ref.safetensors" ${VOXLINGUA_CPU:+--cpu}
done

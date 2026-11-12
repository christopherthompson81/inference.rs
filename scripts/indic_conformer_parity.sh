#!/usr/bin/env bash
# One-off parity check of our IndicConformer (from `ai4bharat/IndicConformer`'s `.nemo`) against AI4Bharat's own
# ONNX export (`ai4bharat/indic-conformer-600m-multilingual`): the encoder output and the greedy RNN-T and CTC token
# ids of each clip, the reference run through onnxruntime as the repo's `model_onnx.py` drives it (its TorchScript
# preprocessor, the language's CTC columns, its RNN-T loop), on the CPU.
# Not part of local_ci.sh: it needs torch and onnxruntime, both checkpoints, and a CUDA build (INDIC_CUDA=1 runs ours
# on the GPU).
#
# Usage: scripts/indic_conformer_parity.sh <IndicConformer.nemo> <onnx repo dir> <language> <clip.wav>...
set -euo pipefail

if [[ $# -lt 4 ]]; then
    sed -n '9p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
nemo=$(realpath "$1")
onnx=$(realpath "$2")
language=$3
shift 3
root=$(cd "$(dirname "$0")/.." && pwd)
dumps=$(mktemp -d)
trap 'rm -rf "$dumps"' EXIT

cd "$root"
cargo build -q -p inference-models-speech --example indic_conformer_parity --features cuda
for clip in "$@"; do
    python3 -P -W ignore - "$onnx" "$language" "$(realpath "$clip")" "$dumps/clip.wav" "$dumps/ref.safetensors" <<'PY'
import json, sys
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
from safetensors.torch import save_file

onnx_dir, lang, clip, out_wav, out = sys.argv[1:6]
BLANK, SOS, MAX_SYMBOLS, LAYERS, HIDDEN = 256, 5632, 10, 2, 640
wav, sr = sf.read(clip, dtype="float32", always_2d=True)
wav = torch.from_numpy(wav.mean(1))
wav = torchaudio.functional.resample(wav, sr, 16000) if sr != 16000 else wav
sf.write(out_wav, wav.numpy(), 16000, subtype="FLOAT")
assets = f"{onnx_dir}/assets"
session = lambda name: ort.InferenceSession(f"{assets}/{name}.onnx", providers=["CPUExecutionProvider"])
pre = torch.jit.load(f"{assets}/preprocessor.ts", map_location="cpu")
signal = wav.unsqueeze(0)
with torch.no_grad():
    feats, length = pre(input_signal=signal, length=torch.tensor([signal.shape[-1]]))
enc, enc_len = session("encoder").run(["outputs", "encoded_lengths"],
                                      {"audio_signal": feats.numpy(), "length": length.numpy()})
masks = json.load(open(f"{assets}/language_masks.json"))
vocab = json.load(open(f"{assets}/vocab.json"))
logits = session("ctc_decoder").run(["logprobs"], {"encoder_output": enc})[0]
logp = torch.from_numpy(logits[:, :, masks[lang]]).log_softmax(-1)[0, : int(enc_len[0])]
path = logp.argmax(-1).tolist()
ctc = [t for i, t in enumerate(path) if t != BLANK and (i == 0 or t != path[i - 1])]

joint_enc = torch.from_numpy(session("joint_enc").run(["output"], {"input": enc.transpose(0, 2, 1)})[0])
dec, pred, pre_net = session("rnnt_decoder"), session("joint_pred"), session("joint_pre_net")
post = session(f"joint_post_net_{lang}")
hyp = [SOS]
state = (np.zeros((LAYERS, 1, HIDDEN), np.float32), np.zeros((LAYERS, 1, HIDDEN), np.float32))
for t in range(joint_enc.size(1)):
    f = joint_enc[:, t, :].unsqueeze(1)
    added = 0
    while added < MAX_SYMBOLS:
        g, _, s0, s1 = dec.run(["outputs", "prednet_lengths", "states", "162"],
                               {"targets": np.array([[hyp[-1]]], np.int32), "target_length": np.array([1], np.int32),
                                "states.1": state[0], "onnx::Slice_3": state[1]})
        g = pred.run(["output"], {"input": g.transpose(0, 2, 1)})[0]
        joint = pre_net.run(["output"], {"input": (f + g).numpy()})[0]
        token = int(torch.from_numpy(post.run(["output"], {"input": joint})[0]).log_softmax(-1).argmax(-1).item())
        added += 1
        if token == BLANK:
            break
        hyp.append(token)
        state = (s0, s1)
rnnt = hyp[1:]
save_file({"features": feats[0].T.contiguous(), "encoded": torch.from_numpy(enc[0].T).contiguous(),
           "rnnt": torch.tensor(rnnt, dtype=torch.int64), "ctc": torch.tensor(ctc, dtype=torch.int64)}, out)
text = lambda ids: "".join(vocab[lang][i] for i in ids).replace("▁", " ").strip()
print("reference rnnt:", text(rnnt))
print("reference ctc: ", text(ctc))
PY
    target/debug/examples/indic_conformer_parity --nemo "$nemo" --language "$language" --wav "$dumps/clip.wav" \
        --dump "$dumps/ref.safetensors" ${INDIC_CUDA:+--cuda}
done

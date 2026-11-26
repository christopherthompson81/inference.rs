#!/usr/bin/env bash
# One-off parity check of our DocLayout-YOLO against the reference package's predictor on one page: the letterboxed
# input, the three head features and every kept box.
# Not part of local_ci.sh: it needs torch and the `doclayout-yolo` package (its own venv: pip install doclayout-yolo).
#
# Usage: scripts/doclayout_yolo_parity.sh <checkpoint .pt or its folder> <page image> [--python PY] [--cpu]
#   PY   the interpreter with doclayout-yolo installed (default python3)
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '6,7p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
model=$(realpath "$1")
image=$(realpath "$2")
shift 2
python=python3
cpu=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu) cpu=(--cpu) ;;
        --python) python=$2; shift ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
ckpt=$model
[[ -d $ckpt ]] && ckpt=$(find "$ckpt" -maxdepth 1 -name '*.pt' | head -1)
root=$(cd "$(dirname "$0")/.." && pwd)
dumps=$(mktemp -d)
trap 'rm -rf "$dumps"' EXIT

"$python" -P -W ignore - "$ckpt" "$image" "$dumps" <<'PY'
import sys
import cv2
from safetensors.torch import save_file
from doclayout_yolo import YOLOv10

ckpt, image, out = sys.argv[1:]
png = f"{out}/page.png"
cv2.imwrite(png, cv2.imread(image))  # decoded once, so both sides read the same pixels
model = YOLOv10(ckpt)
imgsz = model.ckpt["train_args"]["imgsz"]
model.predict(png, imgsz=imgsz, conf=0.2, device="cpu", verbose=False)  # builds and fuses the predictor's model
net = model.predictor.model.model
seen = {}


def keep(name):
    def hook(module, inputs, output):
        seen.setdefault(name, inputs[0])

    return hook


net.model[0].register_forward_hook(keep("input"))
net.model[-1].register_forward_hook(keep("feats"))
result = model.predict(png, imgsz=imgsz, conf=0.2, device="cpu", verbose=False)[0]
dump = {"input": seen["input"][0].contiguous(), "boxes": result.boxes.data.contiguous()}
for i, feat in enumerate(seen["feats"]):
    dump[f"feat{i}"] = feat[0].contiguous()
save_file(dump, f"{out}/reference.safetensors")
PY

cd "$root"
cargo run -q -p inference-layout --example doclayout_yolo_parity -- \
    --model "$model" --reference "$dumps/reference.safetensors" --image "$dumps/page.png" "${cpu[@]}"

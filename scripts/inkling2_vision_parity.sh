#!/usr/bin/env bash
# One-off parity check of our Inkling 2 image rows (preprocessing, the tower, the 2x2 merge and projector) against an
# f32 torch rendering of vLLM's encoder (`vllm/models/inkling2/nvidia/vision.py` and `ops/vision.py`: the CLIP pixel
# round trip, -1 padding, bilinear half-pixel position resize, adjacent-pair 2D RoPE on x + 1 then y + 1) over the
# checkpoint's `model.visual.*`. Not part of local_ci.sh: it needs torch and safetensors (INKLING_PYTHON names the
# interpreter that has them) and the 0.6 GiB tower on the GPU.
#
# Usage: scripts/inkling2_vision_parity.sh <checkpoint> <image>... [--cpu]
#   checkpoint  the official Inkling 2 directory (config.json with `vision_config`, the safetensors holding the tower)
#   image       PNG or JPEG files, read at their own size
#   --cpu       run ours on the CPU
set -euo pipefail

# the f32 tower matches the reference to rounding; bf16 sits about where llama.cpp's does (Run 2 of the investigation)
F32_MAX_REL_L2=1e-4
BF16_MAX_REL_L2=5e-2

if [[ $# -lt 2 ]]; then
    sed -n '8,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
checkpoint=$(realpath "$1")
shift
images=()
ours=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu) ours+=(--cpu); shift ;;
        *) images+=("$(realpath "$1")"); shift ;;
    esac
done
[[ ${#images[@]} -gt 0 ]] || { echo "no images" >&2; exit 2; }
root=$(cd "$(dirname "$0")/.." && pwd)
python=${INKLING_PYTHON:-python3}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

features=()
if [[ ! " ${ours[*]} " =~ " --cpu " ]] && command -v nvcc >/dev/null 2>&1; then
    features=(--features cuda)
fi
cargo build --manifest-path "$root/Cargo.toml" -q -p inference-models-other --example inkling2_vision_parity \
    "${features[@]}"
example="$root/target/debug/examples/inkling2_vision_parity"
rows_wide=$("$python" -P -c "import json, sys; print(json.load(open(sys.argv[1]))['vision_config']['text_hidden_size'])" \
    "$checkpoint/config.json")

for image in "${images[@]}"; do
    name=$(basename "${image%.*}")
    size=$("$python" -P -c "
import sys, numpy as np
from PIL import Image
im = Image.open(sys.argv[1]).convert('RGB')
np.asarray(im).tofile(sys.argv[2])
print(*im.size)" "$image" "$work/$name.rgb")
    read -r width height <<<"$size"
    "$python" -P -W ignore - "$checkpoint" "$work/$name.rgb" "$width" "$height" "$work/$name.ref.f32" <<'PY'
import json, math, sys
import numpy as np, torch, torch.nn.functional as F
from safetensors import safe_open

checkpoint, rgb, w, h, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
cfg = json.load(open(f"{checkpoint}/config.json"))["vision_config"]
P, M, E, H = cfg["native_patch_size"], cfg["adapter_merge_size"], cfg["hidden_size"], cfg["num_attention_heads"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
index = json.load(open(f"{checkpoint}/model.safetensors.index.json"))["weight_map"]
files = {}
def t(name):
    name = "model.visual." + name
    path = index[name]
    if path not in files:
        files[path] = safe_open(f"{checkpoint}/{path}", "pt", device=dev)
    return files[path].get_tensor(name).float()
clip_mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=dev)
clip_std_processor = torch.tensor((0.26862954, 0.2613026, 0.2757771), device=dev)
clip_std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=dev)
img = torch.from_numpy(np.fromfile(rgb, np.uint8).reshape(h, w, 3)).float().to(dev)
cell = P * M
x = torch.full((-(-h // cell) * cell, -(-w // cell) * cell, 3), -1.0, device=dev)
x[:h, :w] = img
x = 2.0 * (((x / 255 - clip_mean) / clip_std_processor).bfloat16().float() * clip_std + clip_mean) - 1.0
gh, gw = x.shape[0] // P, x.shape[1] // P
y = x.reshape(gh, P, gw, P, 3).permute(0, 2, 1, 3, 4).reshape(gh * gw, -1) @ t("backbone.embeddings.patch_embedding.weight").T
pos = t("backbone.embeddings.position_embedding")
g = math.isqrt(pos.shape[0])
if (gh, gw) != (g, g):
    pos = F.interpolate(pos.T.reshape(1, E, g, g), size=(gh, gw), mode="bilinear", align_corners=False)
    pos = pos.reshape(E, -1).T
cls = t("backbone.embeddings.class_embedding") + t("backbone.embeddings.cls_position_embedding")
y = torch.cat([cls[None], y + pos])
def ln(v, n): return F.layer_norm(v, (E,), t(n + ".weight"), t(n + ".bias"), cfg["layer_norm_eps"])
y = ln(y, "backbone.ln_pre")
dh = E // H
px = torch.cat([torch.zeros(1), torch.arange(gw).repeat(gh) + 1.0]).to(dev)
py = torch.cat([torch.zeros(1), torch.arange(gh).repeat_interleave(gw) + 1.0]).to(dev)
inv = 10000.0 ** (-torch.arange(0, dh // 2, 2, device=dev).float() / (dh // 2))
def rope(v):
    halves = []
    for half, p in ((v[..., : dh // 2], px), (v[..., dh // 2 :], py)):
        a = p[:, None] * inv[None]
        c, s = a.cos()[:, None], a.sin()[:, None]
        e, o = half[..., 0::2], half[..., 1::2]
        halves.append(torch.stack([e * c - o * s, e * s + o * c], -1).flatten(-2))
    return torch.cat(halves, -1)
n = y.shape[0]
for i in range(cfg["num_hidden_layers"]):
    l = f"backbone.layers.{i}."
    z = ln(y, l + "ln_1")
    q, k, v = [(z @ t(l + f"self_attn.{p}_proj.weight").T + t(l + f"self_attn.{p}_proj.bias")).reshape(n, H, dh) for p in "qkv"]
    a = F.scaled_dot_product_attention(rope(q).transpose(0, 1), rope(k).transpose(0, 1), v.transpose(0, 1))
    a = a.transpose(0, 1).reshape(n, E) @ t(l + "self_attn.out_proj.weight").T + t(l + "self_attn.out_proj.bias")
    y = y + t(l + "ls_1.gamma") * a
    z = F.gelu(ln(y, l + "ln_2") @ t(l + "mlp.c_fc.weight").T + t(l + "mlp.c_fc.bias"))
    y = y + t(l + "ls_2.gamma") * (z @ t(l + "mlp.c_proj.weight").T + t(l + "mlp.c_proj.bias"))
y = y[1:].reshape(gh // M, M, gw // M, M, E).permute(0, 2, 1, 3, 4).reshape(-1, M * M * E)
y = F.gelu(y @ t("proj_in.weight").T) @ t("proj_out.weight").T
y.cpu().numpy().astype(np.float32).tofile(out)
PY
    for dtype in f32 bf16; do
        "$example" --checkpoint "$checkpoint" --rgb "$work/$name.rgb" --width "$width" --height "$height" \
            --out "$work/$name.$dtype.f32" --dtype "$dtype" "${ours[@]}" >/dev/null
        limit=$F32_MAX_REL_L2
        [[ $dtype == bf16 ]] && limit=$BF16_MAX_REL_L2
        "$python" -P - "$work/$name.$dtype.f32" "$work/$name.ref.f32" "$name ($width x $height) $dtype" "$limit" "$rows_wide" <<'PY'
import sys, numpy as np
ours, ref, label, limit, wide = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
a = np.fromfile(ours, np.float32).astype(np.float64).reshape(-1, wide)
r = np.fromfile(ref, np.float32).astype(np.float64).reshape(-1, wide)
if a.shape != r.shape:
    sys.exit(f"{label}: {len(a)} rows, the reference {len(r)}")
rel = np.linalg.norm(a - r) / np.linalg.norm(r)
cos = ((a * r).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(r, axis=1)).min()
print(f"{label}: {len(r)} rows, rel L2 {rel:.3e} (<= {limit:.0e}), lowest row cosine {cos:.6f}")
sys.exit(0 if rel <= limit else 1)
PY
    done
done

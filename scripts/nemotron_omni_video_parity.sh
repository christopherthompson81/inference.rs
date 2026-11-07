#!/usr/bin/env bash
# One-off parity check of our Nemotron 3 Nano Omni video path (frame resize, tubelets through `video_embedder`, pixel
# shuffle, mlp1, the timestamped prompt) against transformers' own NemotronH Omni video processor and RADIO model:
# each tubelet's patches, tower features and projected embeddings, the `<video>` replacement text, and EVS's kept
# tokens at the checkpoint's `video_pruning_rate`. Not part of local_ci.sh: it needs torch and transformers (5.19+), so
# OMNI_PYTHON names the interpreter that has them.
#
# Usage: scripts/nemotron_omni_video_parity.sh <checkpoint> <frames_dir> [--fps F] [--stride S] [--cpu] [--bf16]
#   checkpoint  a Nano Omni, Bolt or Super VL directory: config.json, preprocessor_config.json and the shards
#               holding `vision_model.*` and `mlp1.*` (the rest of the model is not read)
#   frames_dir  the video's sampled frames as PNG or JPEG files, in name order
#   --fps       the source video's frame rate (default 30), --stride its frames between samples (default 15)
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '8,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
checkpoint=$(realpath "$1")
frames=$(realpath "$2")
shift 2
fps=30
stride=15
ours=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --fps) fps=$2; shift 2 ;;
        --stride) stride=$2; shift 2 ;;
        --cpu) ours+=(--cpu); shift ;;
        --bf16) ours+=(--dtype bf16); shift ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
root=$(cd "$(dirname "$0")/.." && pwd)
dumps=$(mktemp -d)
trap 'rm -rf "$dumps"' EXIT

"${OMNI_PYTHON:-python3}" -P -W ignore - "$checkpoint" "$frames" "$dumps" "$fps" "$stride" <<'PY'
import json, os, sys, types
import torch, torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import RadioConfig, RadioModel
from transformers.models.nemotron_h_omni.modeling_nemotron_h_omni import compute_retention_mask
from transformers.models.nemotron_h_omni.processing_nemotron_h_omni import NemotronH_Omni_Reasoning_V3Processor
from transformers.models.nemotron_h_omni.video_processing_nemotron_h_omni import (
    NemotronH_Omni_Reasoning_V3VideoProcessor,
)

checkpoint, frames_dir, out, fps, stride = sys.argv[1:]
fps, stride = float(fps), int(stride)
cfg = json.load(open(f"{checkpoint}/config.json"))
pre = json.load(open(f"{checkpoint}/preprocessor_config.json"))
vision = cfg["vision_config"]
args = vision["args"]
# RADIO's prefix: one cls token per teacher, registers padding it to `register_multiple`
cls = len(args["teachers"]) if args.get("cls_token_per_teacher") else 1
registers = args["register_multiple"] - cls % args["register_multiple"] if args.get("register_multiple") else 0
temporal = vision["video_temporal_patch_size"]
radio = RadioConfig(
    max_img_size=args["cpe_max_size"],
    num_cls_tokens=cls,
    num_registers=registers,
    video_temporal_patch_size=temporal,
)
model = RadioModel(radio)
model.make_preprocessor_external()
index = json.load(open(f"{checkpoint}/model.safetensors.index.json"))["weight_map"]
st = {}
TOWERS = ("vision_model.", "mlp1.", "vision_projector.")
for shard in sorted({f for k, f in index.items() if k.startswith(TOWERS)}):
    with safe_open(f"{checkpoint}/{shard}", "pt") as h:
        for k in h.keys():
            if k.startswith(TOWERS):
                st[k] = h.get_tensor(k).float()
# the released checkpoint's timm names onto transformers' RadioModel, as its conversion mapping does
P = "vision_model.radio_model.model."
sd = {
    "embeddings.patch_projection.weight": st[P + "patch_generator.embedder.weight"],
    "embeddings.video_patch_projection.weight": st[P + "patch_generator.video_embedder.weight"],
    "embeddings.position_embedding": st[P + "patch_generator.pos_embed"],
    "embeddings.cls_register_token": st[P + "patch_generator.cls_token.token"],
}
for i in range(radio.num_hidden_layers):
    b, o = f"{P}blocks.{i}.", f"encoder.layer.{i}."
    for part, w, bias in zip("qkv", st[b + "attn.qkv.weight"].chunk(3), st[b + "attn.qkv.bias"].chunk(3)):
        sd[f"{o}attention.{part}_proj.weight"], sd[f"{o}attention.{part}_proj.bias"] = w, bias
    sd[o + "attention.o_proj.weight"] = st[b + "attn.proj.weight"]
    sd[o + "attention.o_proj.bias"] = st[b + "attn.proj.bias"]
    for n in ["norm1", "norm2", "mlp.fc1", "mlp.fc2"]:
        for p in ["weight", "bias"]:
            sd[f"{o}{n}.{p}"] = st[f"{b}{n}.{p}"]
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not unexpected and all("layer_scale" in k or k == "summary_idxs" for k in missing), missing
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()
mlp = {k: v.to(device) for k, v in st.items() if k.startswith(("mlp1.", "vision_projector."))}

processor = NemotronH_Omni_Reasoning_V3VideoProcessor(
    norm_mean=pre["norm_mean"],
    norm_std=pre["norm_std"],
    patch_size=pre["patch_size"],
    downsample_ratio=pre["downsample_ratio"],
    # Nano Omni's vision config states the frame rule; Bolt's and Super VL's keep it in the preprocessor config
    video_target_num_patches=vision.get("video_target_num_patches", pre.get("video_target_num_patches", 1024)),
    video_maintain_aspect_ratio=vision.get("video_maintain_aspect_ratio", pre.get("video_maintain_aspect_ratio", True)),
    do_sample_frames=False,
)
names = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
images = [Image.open(f"{frames_dir}/{name}").convert("RGB") for name in names]
video = torch.stack([torch.from_numpy(__import__("numpy").asarray(im)).permute(2, 0, 1) for im in images])
inputs = processor(videos=[video], return_tensors="pt")
pixels = inputs["pixel_values_videos"].to(device).float()
indices = [i * stride for i in range(len(images))]

# the reference's own label text: `replace_video_token` on a stand-in for the processor's state
state = types.SimpleNamespace(
    image_start_token="<img>", image_end_token="</img>", image_token="<image>", video_temporal_patch_dim=temporal
)
metadata = types.SimpleNamespace(fps=fps, frames_indices=indices)
replacement = NemotronH_Omni_Reasoning_V3Processor.replace_video_token(
    state, {"num_tokens": inputs["num_tokens"]}, 0, num_frames=len(images), video_metadata=metadata
)

# NVIDIA's `pixel_shuffle` (ps_version v2) and `mlp1`: RMSNorm, linear, ReLU^2, linear
def pixel_shuffle(x, s=0.5):
    n, w, h, c = x.size()
    x = x.view(n, w, int(h * s), int(c / s)).permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h * s), int(w * s), int(c / (s * s)))
    return x.permute(0, 2, 1, 3).contiguous()

# the LayerNorm on RADIO's features a decoder with MTP layers brings (Super VL), before the shuffle
def final_norm(x):
    norm = "vision_projector.vision_final_layernorm."
    if norm + "weight" not in mlp:
        return x
    return F.layer_norm(x, x.shape[-1:], mlp[norm + "weight"], mlp[norm + "bias"], vision["layer_norm_eps"])

def project(x):
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * mlp["mlp1.0.weight"]
    return F.linear(torch.relu(F.linear(x, mlp["mlp1.1.weight"])).square(), mlp["mlp1.3.weight"])

n, c, h, w = pixels.shape
if n % temporal:
    pixels = torch.cat([pixels, pixels[-1:].expand(temporal - n % temporal, -1, -1, -1)])
packed = pixels.reshape(-1, temporal * c, h, w)
with torch.no_grad():
    features = model(packed).features
    rows, cols = h // radio.patch_size, w // radio.patch_size
    shuffled = pixel_shuffle(final_norm(features).reshape(packed.shape[0], rows, cols, -1))
    embeds = project(shuffled.reshape(packed.shape[0], -1, shuffled.shape[-1]))
# EVS as the reference applies it, on a (T, 1, N) grid: it reshapes to a square of sqrt(N) a side, which only a square
# tubelet has, and the comparison is per position either way
rate = cfg.get("video_pruning_rate", 0.0)
keep = compute_retention_mask(
    video_embeds=embeds.reshape(-1, embeds.shape[-1]),
    thw=(embeds.shape[0], 1, embeds.shape[1]),
    spatial_merge_size=1,
    q=rate,
).reshape(embeds.shape[0], embeds.shape[1])
os.makedirs(f"{out}/frames")
for name, image in zip(names, images):
    image.save(f"{out}/frames/{os.path.splitext(name)[0]}.png")
json.dump({"fps": fps, "indices": indices, "replacement": replacement, "rate": rate}, open(f"{out}/video.json", "w"))
save_file({
    "patches": model.embeddings._image_to_patches(packed).cpu().contiguous(),
    "features": features.cpu().contiguous(),
    "embeds": embeds.cpu().contiguous(),
    "kept": keep.to(torch.uint8).cpu().contiguous(),
}, f"{out}/video.safetensors")
print(f"reference: {len(images)} frames, {packed.shape[0]} tubelets of {rows} x {cols} patches")
PY

features=()
if [[ ! " ${ours[*]} " =~ " --cpu " ]] && command -v nvcc >/dev/null 2>&1; then
    features=(--features cuda)
fi
cargo run --manifest-path "$root/Cargo.toml" -q -p inference-models-other --example nemotron_omni_parity \
    "${features[@]}" -- --checkpoint "$checkpoint" --video "$dumps" "${ours[@]}"

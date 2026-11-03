#!/usr/bin/env bash
# One-off parity check of our Nemotron-H Omni image path (NVIDIA's processor resize, C-RADIO, pixel shuffle, mlp1)
# against the checkpoint's own transformers code: patches, tower features and projected embeddings per image. Not part
# of local_ci.sh: it needs torch and transformers (5.8+), so OMNI_PYTHON names the interpreter that has them.
#
# Usage: scripts/nemotron_omni_parity.sh <checkpoint> <image>... [--gguf <text.gguf> <mmproj.gguf>] [--cpu] [--bf16]
#   checkpoint  a NemotronH_Omni_Reasoning_V3 directory: config.json, preprocessor_config.json, the remote code and
#               the shards holding `vision_model.*` and `mlp1.*` (the rest of the model is not read)
#   image       PNG or JPEG files
#   --gguf      run ours from the text GGUF and its mmproj instead of the checkpoint's shards
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '6,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
checkpoint=$(realpath "$1")
shift
images=()
ours=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --gguf) ours+=(--gguf "$(realpath "$2")" "$(realpath "$3")"); shift 3 ;;
        --cpu) ours+=(--cpu); shift ;;
        --bf16) ours+=(--dtype bf16); shift ;;
        *) images+=("$(realpath "$1")"); shift ;;
    esac
done
[[ ${#images[@]} -gt 0 ]] || { echo "no images" >&2; exit 2; }
root=$(cd "$(dirname "$0")/.." && pwd)
dumps=$(mktemp -d)
trap 'rm -rf "$dumps"' EXIT
# the checkpoint's remote code, imported as a package from its own copy
mkdir "$dumps/omni_ref"
touch "$dumps/omni_ref/__init__.py"
for module in configuration_radio modeling_radio image_processing_nemotron_h_omni; do
    cp "$checkpoint/$module.py" "$dumps/omni_ref/"
done

"${OMNI_PYTHON:-python3}" -P -W ignore - "$checkpoint" "$dumps" "${images[@]}" <<'PY'
import json, os, sys
import torch, torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

checkpoint, out, *images = sys.argv[1:]
sys.path.insert(0, out)
from omni_ref.configuration_radio import RadioConfig
from omni_ref.modeling_radio import RadioModel
from omni_ref.image_processing_nemotron_h_omni import NemotronH_Omni_Reasoning_V3ImageProcessor

cfg = json.load(open(f"{checkpoint}/config.json"))
vision = dict(cfg["vision_config"])
vision.pop("args", None)
radio = RadioConfig(**vision)
model = RadioModel(radio)
model.make_preprocessor_external()
index = json.load(open(f"{checkpoint}/model.safetensors.index.json"))["weight_map"]
st = {}
for shard in sorted({f for k, f in index.items() if not k.startswith("language_model")}):
    with safe_open(f"{checkpoint}/{shard}", "pt") as h:
        for k in h.keys():
            if not k.startswith("language_model"):
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
    for part, w, bias in zip(["query", "key", "value"], st[b + "attn.qkv.weight"].chunk(3), st[b + "attn.qkv.bias"].chunk(3)):
        sd[f"{o}attention.attention.{part}.weight"], sd[f"{o}attention.attention.{part}.bias"] = w, bias
    sd[o + "attention.output.dense.weight"] = st[b + "attn.proj.weight"]
    sd[o + "attention.output.dense.bias"] = st[b + "attn.proj.bias"]
    for n in ["norm1", "norm2", "mlp.fc1", "mlp.fc2"]:
        for p in ["weight", "bias"]:
            sd[f"{o}{n}.{p}"] = st[f"{b}{n}.{p}"]
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not unexpected and all("layer_scale" in k or k == "summary_idxs" for k in missing), missing
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()
mlp = {k: v.to(device) for k, v in st.items() if k.startswith("mlp1.")}
processor = NemotronH_Omni_Reasoning_V3ImageProcessor(**json.load(open(f"{checkpoint}/preprocessor_config.json")))

# NVIDIA's `pixel_shuffle` (ps_version v2) and `mlp1`: RMSNorm, linear, ReLU^2, linear
def pixel_shuffle(x, s=0.5):
    n, w, h, c = x.size()
    x = x.view(n, w, int(h * s), int(c / s)).permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h * s), int(w * s), int(c / (s * s)))
    return x.permute(0, 2, 1, 3).contiguous()

def project(x):
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * mlp["mlp1.0.weight"]
    return F.linear(torch.relu(F.linear(x, mlp["mlp1.1.weight"])).square(), mlp["mlp1.3.weight"])

for n, path in enumerate(images):
    image = Image.open(path).convert("RGB")
    pixels = processor(images=[image], return_tensors=None)["pixel_values"]
    pixels = torch.as_tensor(pixels[0]).unsqueeze(0).to(device).float()
    with torch.no_grad():
        features = model(pixels).features
        rows, cols = pixels.shape[-2] // radio.patch_size, pixels.shape[-1] // radio.patch_size
        shuffled = pixel_shuffle(features.reshape(1, rows, cols, -1))
        embeds = project(shuffled.reshape(1, -1, shuffled.shape[-1]))
    image.save(f"{out}/{n}.png")
    save_file({
        "patches": model.embeddings._image_to_patches(pixels)[0].cpu().contiguous(),
        "features": features[0].cpu().contiguous(),
        "embeds": embeds[0].cpu().contiguous(),
    }, f"{out}/{n}.safetensors")
print(f"reference: {len(images)} images")
PY

features=()
if [[ ! " ${ours[*]} " =~ " --cpu " ]] && command -v nvcc >/dev/null 2>&1; then
    features=(--features cuda)
fi
source_args=(--checkpoint "$checkpoint")
[[ " ${ours[*]} " =~ " --gguf " ]] && source_args=()
cargo run --manifest-path "$root/Cargo.toml" -q -p inference-models-other --example nemotron_omni_parity \
    "${features[@]}" -- "${source_args[@]}" --dumps "$dumps" "${ours[@]}"

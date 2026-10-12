#!/usr/bin/env bash
# One-off parity check of our HiDream-O1-Image against HiDream's own pipeline (github.com/HiDream-ai/HiDream-O1-Image)
# for one prompt: each step's clean-image prediction from the reference's own input (teacher forced) must reach a
# minimum cosine, and the image sampled from the reference's noise is compared pixel by pixel.
# Not part of local_ci.sh: it needs a CUDA GPU, the reference checkout and a Python with its requirements
# (transformers 4.57, diffusers, torch).
#
# Usage: scripts/hidream_o1_parity.sh <checkpoint dir> [--dev] [--prompt <text>] [--size <w>x<h>] [--steps <n>] [--seed <n>] [--ref-image <file>]... [--keep-aspect] [--min-cosine <x>] [--slack <x>] [--f32 <k,...>] [--gguf <file>] [--keep-dumps <dir>]
#   dir   holds config.json, tokenizer.json and the model*.safetensors shards (HiDream-O1-Image or -Dev)
#   ref-image  a reference image: one edits it, two or more are subject references
#   keep-aspect  with one reference, the output takes the reference's size (the reference's keep_original_aspect)
#   f32   reruns those forwards of the reference in F32 on CPU (~40 GB RAM, minutes each); ours must then be at least
#         as close to F32 as the reference's BF16 is, the bar at noisy steps where BF16 itself drifts
#   gguf  ours from that GGUF instead of the checkpoint's shards, held to the quantized bar unless the bars are given
#   env   HIDREAM_REPO is the reference checkout, HIDREAM_PYTHON its interpreter (default python3)
set -euo pipefail

if [[ $# -lt 1 ]]; then
    sed -n '8,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
: "${HIDREAM_REPO:?set HIDREAM_REPO to the HiDream-O1-Image checkout}"
python=${HIDREAM_PYTHON:-python3}
model=$(realpath "$1")
shift
variant=full
prompt="A red fox sitting in a snowy forest at dawn, soft light, detailed fur."
size=1024x1024
steps=
seed=7
keep=
f32=
refs=()
keep_aspect=0
gguf=()
tolerance=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev) variant=dev ;;
        --prompt) prompt=$2; shift ;;
        --size) size=$2; shift ;;
        --steps) steps=$2; shift ;;
        --seed) seed=$2; shift ;;
        --ref-image) refs+=("$(realpath "$2")"); shift ;;
        --keep-aspect) keep_aspect=1 ;;
        --min-cosine | --slack) tolerance+=("$1" "$2"); shift ;;
        --f32) f32=$2; shift ;;
        # ours from a GGUF of the checkpoint, against the same reference
        --gguf) gguf=(--gguf "$(realpath "$2")"); shift ;;
        # reuse (or fill) this directory's reference dumps, so a rerun pays for the reference once
        --keep-dumps) keep=$(realpath -m "$2"); shift ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
# Dev's noisy steps move ~0.008 between two valid BF16 runs (swapping only the SDPA kernel), so 0.999 is out of reach;
# a quantized GGUF's bar admits Q8_0 (0.989, 0.014 behind F32) and Q6_K with down_proj at Q8_0, not plain Q6_K
if [[ ${#gguf[@]} -gt 0 && ${#tolerance[@]} -eq 0 ]]; then
    tolerance=(--min-cosine 0.98 --slack 0.03)
elif [[ $variant == dev && ${#tolerance[@]} -eq 0 ]]; then
    tolerance=(--min-cosine 0.99)
fi
root=$(cd "$(dirname "$0")/.." && pwd)
if [[ -n $keep ]]; then
    dumps=$keep
    mkdir -p "$dumps"
else
    # the dumps run to GBs at full size; TMPDIR picks the disk
    dumps=$(mktemp -d)
    trap 'rm -rf "$dumps"' EXIT
fi

# the reference checkout's commit too, so dumps from another revision are not reused
revision=$(git -C "$HIDREAM_REPO" rev-parse HEAD 2>/dev/null || echo unknown)
manifest=$(printf '%s\n' "$model" "$revision" "$variant" "$prompt" "$size" "$steps" "$seed" "$f32" "$keep_aspect" "${refs[@]}")
if [[ -n $keep && -f $dumps/manifest && $(cat "$dumps/manifest") == "$manifest" ]]; then
    reference=(true)
else
    rm -f "$dumps"/*.safetensors "$dumps"/*.json "$dumps"/*.png "$dumps/manifest"
    reference=("$python" -P -W ignore - "$HIDREAM_REPO" "$model" "$dumps" "$variant" "$prompt" "$size" "${steps:-0}" "$seed" "$f32"
        "$keep_aspect" "${refs[@]}")
fi
"${reference[@]}" <<'PY'
import json, sys
import numpy as np, torch
from safetensors.torch import save_file

repo, model_dir, out, variant, prompt, size, steps, seed, f32, keep_aspect = sys.argv[1:11]
refs = sys.argv[11:]
replayed = {int(k) for k in f32.split(",") if k}
sys.path.insert(0, repo)
from transformers import AutoProcessor
from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
from models import pipeline as P
from models import flash_scheduler as F

width, height = (int(v) for v in size.split("x"))
steps, seed = int(steps), int(seed)
processor = AutoProcessor.from_pretrained(model_dir)
tokenizer = processor.tokenizer
tokenizer.boi_token, tokenizer.tms_token = "<|boi_token|>", "<|tms_token|>"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()
# the reference hardcodes 151673; reading it lets a tiny test checkpoint stand in
model.model.tms_token_id = tokenizer.convert_tokens_to_ids("<|tms_token|>")

# the parity size need not be one of the model's presets
P.find_closest_resolution = lambda w, h: (w, h)
tensors, calls, noises = {}, [], []
forward = model.forward

def recording_forward(**kwargs):
    # flash_attn is not installed: the reference's 4D-mask path computes the same attention
    kwargs["use_flash_attn"] = False
    outputs = forward(**kwargs)
    n = kwargs["vinputs"].shape[1]
    k = len(calls)
    # with references, vinputs is the target's patches then theirs
    target = n - total_ref_len(kwargs)
    tensors[f"z_in.{k}"] = kwargs["vinputs"][0, :target].float().cpu().contiguous()
    tensors[f"x_pred.{k}"] = outputs.x_pred[0, -n:][:target].float().cpu().contiguous()
    if k < 2 and refs:
        # each sample's own layout once: its tokens and positions, and the reference inputs both share
        tensors[f"input_ids.{k}"] = kwargs["input_ids"][0].cpu().contiguous()
        tensors[f"position_ids.{k}"] = kwargs["position_ids"][:, 0].cpu().contiguous()
        tensors["ref.patches"] = kwargs["vinputs"][0, target:].float().cpu().contiguous()
        tensors["ref.pixel_values"] = kwargs["pixel_values"].float().cpu().contiguous()
        tensors["ref.image_grid_thw"] = kwargs["image_grid_thw"].cpu().contiguous()
        # the ViT's own outputs, so the vision path is checked apart from the denoiser
        if outputs.cond_image_embeds is not None:
            tensors["ref.vit"] = outputs.cond_image_embeds.float().cpu().contiguous()
            for i, layer in enumerate(outputs.cond_deepstack_image_embeds):
                tensors[f"ref.deepstack.{i}"] = layer.float().cpu().contiguous()
    call = {"timestep": float(kwargs["timestep"].reshape(-1)[0]), "target": target}
    if k in replayed:
        # the exact inputs, so the F32 pass replays this forward (an uncond one included) without the pipeline
        # the ViT's cached BF16 outputs stay out, so the F32 pass encodes the references in F32 itself
        replay = {name: v for name, v in kwargs.items() if not name.startswith("precomputed_")}
        call["kwargs"] = {name: v for name, v in replay.items() if not isinstance(v, torch.Tensor)}
        for name, v in replay.items():
            if isinstance(v, torch.Tensor):
                tensors[f"kwargs.{k}.{name}"] = v.cpu().contiguous()
    calls.append(call)
    return outputs

def total_ref_len(kwargs):
    if "pixel_values" not in kwargs:
        return 0
    # the positions cover the ids, then the target's patches, then the references'
    return kwargs["position_ids"].shape[-1] - kwargs["input_ids"].shape[-1] - target_len

model.forward = recording_forward
randn = F.hack_randn_tensor

def recording_randn(*args, **kwargs):
    value = randn(*args, **kwargs)
    tensors[f"noise.{len(noises)}"] = value[0].float().cpu().contiguous()
    noises.append(len(noises))
    return value

F.hack_randn_tensor = recording_randn
if variant == "full":
    settings = dict(num_inference_steps=steps or 50, guidance_scale=5.0, shift=3.0, timesteps_list=None,
                    scheduler_name="default")
elif len(refs) == 1:
    # inference.py's default for Dev editing
    settings = dict(num_inference_steps=28, guidance_scale=0.0, shift=1.0, timesteps_list=P.DEFAULT_TIMESTEPS,
                    scheduler_name="flow_match")
else:
    settings = dict(num_inference_steps=28, guidance_scale=0.0, shift=1.0, timesteps_list=P.DEFAULT_TIMESTEPS,
                    scheduler_name="flash", noise_scale_start=7.5, noise_scale_end=7.5, noise_clip_std=2.5)
keep = keep_aspect == "1" and len(refs) == 1
if keep:
    # the size the pipeline will take from the reference
    from PIL import Image
    from models.utils import resize_pilimage
    width, height = resize_pilimage(Image.open(refs[0]).convert("RGB"), 2048, P.PATCH_SIZE).size
target_len = (height // P.PATCH_SIZE) * (width // P.PATCH_SIZE)
image = P.generate_image(model=model, processor=processor, prompt=prompt, ref_image_paths=refs or None,
                         height=height, width=width, seed=seed, keep_original_aspect=keep, **settings)
image.save(f"{out}/reference.png")
# the pipeline's initial noise, before its scale, in patch layout
initial = torch.randn((1, 3, height, width), generator=torch.Generator("cpu").manual_seed(seed + 1))
tensors["noise.initial"] = (
    initial.reshape(3, height // 32, 32, width // 32, 32).permute(1, 3, 0, 2, 4).reshape(-1, 3072).contiguous()
)
tensors["image"] = torch.from_numpy(np.asarray(image)).contiguous()
save_file(tensors, f"{out}/reference.safetensors")
json.dump({"prompt": prompt, "width": width, "height": height, "variant": variant, "calls": calls,
           "references": refs, "keep_aspect": keep,
           "guidance": settings["guidance_scale"] > 1.0, "steps": settings["num_inference_steps"]},
          open(f"{out}/reference.json", "w"))
print(f"reference: {len(calls)} forwards, {len(noises)} step noises, {out}/reference.png")
PY
[[ -n $keep ]] && printf '%s' "$manifest" > "$dumps/manifest"

against=()
if [[ -n $f32 ]]; then
    if [[ ! -f $dumps/f32.safetensors ]]; then
        "$python" -P -W ignore - "$HIDREAM_REPO" "$model" "$dumps" "$f32" <<'PY'
import json, os, sys
import torch
from safetensors.torch import load_file, save_file

repo, model_dir, dumps = sys.argv[1:4]
sys.path.insert(0, repo)
from transformers import AutoTokenizer
from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

meta = json.load(open(f"{dumps}/reference.json"))
tensors = load_file(f"{dumps}/reference.safetensors")
torch.set_num_threads(os.cpu_count())
model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
model.model.tms_token_id = AutoTokenizer.from_pretrained(model_dir).convert_tokens_to_ids("<|tms_token|>")
wanted = [int(k) for k in sys.argv[4].split(",")]
missing = [k for k in wanted if k >= len(meta["calls"]) or "kwargs" not in meta["calls"][k]]
if missing:
    sys.exit(f"--f32: the reference ran {len(meta['calls'])} forwards and recorded none of {missing}")
out = {}
for k, call in enumerate(meta["calls"]):
    if "kwargs" not in call:
        continue
    kwargs = dict(call["kwargs"])
    prefix = f"kwargs.{k}."
    for name, v in tensors.items():
        if name.startswith(prefix):
            v = v.float() if v.is_floating_point() else v
            kwargs[name[len(prefix):]] = v
    with torch.no_grad():
        x_pred = model(**kwargs).x_pred
    n = kwargs["vinputs"].shape[1]
    out[f"f32.{k}"] = x_pred[0, -n:][: call["target"]].float().contiguous()
    print(f"f32: forward {k} done", flush=True)
save_file(out, f"{dumps}/f32.safetensors")
PY
    fi
    against=(--against "$dumps/f32.safetensors")
fi

cargo run --manifest-path "$root/Cargo.toml" -q -p inference-models-diffusion --features cuda --example hidream_o1_parity -- \
    --model "$model" --dumps "$dumps" "${tolerance[@]}" "${against[@]}" "${gguf[@]}"

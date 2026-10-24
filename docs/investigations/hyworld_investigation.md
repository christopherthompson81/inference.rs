# HY-World 2.0 investigation

Native support for Tencent's HY-World 2.0 models (WorldMirror 2.0 first, then WorldStereo-2), with the Tencent Python
code as the reference. Local branch only: the model license limits use and distribution to a territory that excludes
the EU, UK and South Korea, so nothing derived from it is pushed.

## Run 1 - 2026-10-24 12:12

Question: does the reference run here, and what does it cost on the 3090?

Reference: github.com/Tencent-Hunyuan/HY-World-2.0 at df9988e, in a uv venv at /mnt/data/venvs/hyworld (Python 3.11,
torch 2.7.1+cu128, flash-attn 2.8.3 wheel, gsplat 1.5.3 from its pt27cu128 index; the attention module imports
flash-attn unconditionally, though its F32 path runs SDPA). Weights: HY-WorldMirror-2.0/model.safetensors (5.05 GB F32,
sha256 9fff06539d3d...), skyseg.onnx for the sky mask, both fetched with curl.

The model: DINOv2 ViT-L/14 with 4 registers as patch embedding, 24 alternating frame/global attention blocks at 1024
wide with normalized 2D RoPE (base 100), camera, depth (with mask), normal, point and Gaussian heads; prior conditioning
(camera token, depth via pow3r, intrinsics token). The checkpoint covers every parameter; the 48 keys it lacks are the
blocks' RoPE `periods` buffers, computed from the base.

`python -m hyworld2.worldrecon.pipeline --pretrained_model_name_or_path /mnt/data/models/HY-World-2.0 --subfolder
HY-WorldMirror-2.0 --strict_output_path <out> --no_interactive [--enable_bf16]`, peak nvidia-smi above idle (100 ms):
```
sample (repo examples)          wall    case    forward   peak above idle
Park, 3 x 960x545 JPEG, F32     21.5 s  6.7 s             14,833 MiB
Park, BF16                      22.0 s                    11,057
Building, 32 PNG frames, F32    51.7 s  39.1 s  7.80 s    21,915   (adaptive resolution 504)
Building, BF16                  53.3 s  37.5 s  7.50 s    21,438
```
The forward is a fifth of a 32-frame case; saving the 3.16M voxel-pruned Gaussians (211 MB) and the point cloud (30 MB)
takes most of the rest. Peak VRAM barely moves with BF16 on 32 frames, so activations, not weights, set it.

## Run 2 - 2026-10-24 12:33

Question: do the ported backbone and trunk (new crate `inference-recon`, `worldmirror::{dinov2, trunk, layers, rope}`)
match the reference, and where does any gap come from?

Reference dump (`dump_ref.py`, scratch): the Park sample at target 518 (3 frames of 518 x 294, 777 patches each),
run as the pipeline runs it (CUDA, bf16 autocast outside, the trunk's own autocast inside); hooks save the input, the
backbone's patch tokens, the trunk at layers 0/4/11/17/23, block 0's insides and every prediction. Ours: bf16 matmuls
and attention, f32 norms and residuals (the reference's autocast split), RoPE table built with the reference's per-op
bf16 rounding, the DINOv2 position embedding resampled with PyTorch's antialiased bicubic (a = -0.5) on the CPU.
```
relative rms (max |diff|) against the reference
backbone patch tokens      3.6e-3 (2.6e-2)
trunk layer 0 frame/global 1.8e-2 / 2.4e-2
layer 4                    2.2e-2 / 2.3e-2
layer 11                   2.0e-2 / 2.1e-2
layer 17                   2.9e-2 / 2.2e-2
layer 23                   1.7e-2 / 7.1e-3
block 0 on the reference's own input: norm1 8e-8, q after q_norm 3.6e-3, after RoPE 4.0e-3, attention 3.3e-3,
norm2 3.9e-4, MLP 3.1e-3, output 1.25e-2
```
The block's parts all agree to bf16 rounding, and rebuilding the reference's output from its own parts with our
formula gives it exactly; the output is small next to its terms (rms: input 0.215, ls2 x MLP 0.184, output 0.055), so
the MLP's 0.3% becomes ~1.2% of the output. That is the reference's own bf16 noise, not a port error.

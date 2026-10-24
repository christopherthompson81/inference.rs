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

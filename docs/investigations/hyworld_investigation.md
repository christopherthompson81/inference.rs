# HY-World 2.0 investigation

Native support for Tencent's HY-World 2.0 models (WorldMirror 2.0 first, then WorldStereo-2), with the Tencent Python
code as the reference. The model license's territory terms cover its Materials (the model, its outputs and derivative
models); this engine implementation is not treated as one of them. Weights, sample outputs and reference dumps are
never committed.

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

## Run 3 - 2026-10-24 12:40

Question: do the camera head, the four DPT heads and the Gaussian parameters match, and is any end-to-end gap theirs
or the trunk's?

Camera head (f32; 4 refinement steps of adaptive-modulation blocks attending across frames) and decode
(`decode_cameras`: quaternion scaled by 2/|q|^2, world-to-camera inverted in closed form, intrinsics from the two
fields of view, principal point at the centre):
```
camera params rel rms 1.2e-3 (max 1.6e-3); camera-to-world max err 1.8e-5 / 4.2e-4 / 6.2e-4; fx rel err <= 1.8e-3
frame 0 decodes to the identity pose (t ~ 1e-4, q ~ (0, 0, 0, 1)): the world frame is the first camera's
```
DPT heads (`dpt.rs`, f32; the residual units' skip is relu(x), as the reference's in-place ReLU makes it) and the
Gaussian parameters (`gaussians.rs`), rel rms against the reference:
```
                     whole model   heads on the reference's trunk features
depth                1.1e-2        2.2e-4
depth mask logits    1.8e-2        3.9e-4
points               2.4e-2        6.4e-4
normals              1.4e-2        1.2e-4
gs depth             6.8e-3        1.4e-4
splat means          6.4e-3        1.3e-4
splat quats          3.9e-2        1.2e-3
splat scales         4.9e-2        8.4e-4
splat opacities      1.6e-2        2.3e-4
splat SH DC          4.0e-3        2.3e-4
```
On the same features the heads agree to f32 convolution order (PyTorch's cuDNN path uses TF32 on Ampere); the
whole-model gap is the trunk's bf16 noise (Run 2) carried through, largest in the quaternions and scales.

## Run 4 - 2026-10-24 12:57

Question: does the forward fit and run at the repo's 32-frame size (504 x 504, 41,696 global-attention tokens), and
where does its time go?

`thirty_two_frames_forward` (ignored bench, random input, trunk and heads timed apart):
```
first try         heads OOM even at 16 frames: every head returned its full-resolution fused features for all
                  frames (4.2 GB a head at 32 frames) and the splat convs ran on all frames at once; the CUDA conv2d
                  (no cuDNN in our builds) is im2col, so a 3x3 conv over 8 full-resolution frames needs a ~9 GB column
fixed             only the Gaussian head keeps its fused features, the splats are built per chunk of frames, chunks of 2
                  32 frames: trunk 9.5 s, heads 18.6 s, peak 13.0 GB (chunks of 1: heads 32.7 s, peak 10.5 GB)
position embed    built from two separable sin/cos tables on the GPU, not 32M CPU sin/cos per call: heads 11.9 s
1x1 conv, convT   pointwise convs and the stride-equals-kernel transposed convs as one matmul per frame: heads 9.2 s
bilinear kernel   inference-tensor's upsample_bilinear2d computed in double (1/64 rate on GA102) with 64-bit
                  divisions; now PyTorch's CUDA formula in float: heads 8.1 s (kernel 1.9 -> 0.47 s)
3x3 convs in bf16 heads 7.3 s, but head parity 8x worse (depth rel rms 2.2e-4 -> 1.7e-3); reverted
```
Parity on the reference's trunk features is unchanged by the kept changes (depth 2.2e-4, normals 1.2e-4, splat means
1.3e-4). nsys per 32-frame forward (18.1 s of kernels): fattn's flash_attn_ext_f16 5.8 s (72 calls, mostly the 24
global attentions over 41.7K tokens, ~40% of the 3090's bf16 tensor peak), im2col_f32 3.8 s (its layout reads across
channel planes, five 64-bit divisions an element), f32 sgemm 1.9 s, trunk bf16 GEMMs 1.1 s. Reference: 7.8 s forward
(FlashAttention-2, cuDNN convs). Next for speed: a long-sequence non-causal attention path and a better im2col or
direct 3x3 conv.

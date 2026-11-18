# Idol (image to Gaussian splats) investigation

Port of speridlabs/idol: DINOv3 ViT-H+/16 and FLUX.2 VAE image tokens, a rectified-flow transformer over an 8192-token
object latent plus a camera token, an octree decoder to level 8 and a Gaussian decoder (32 per anchor). Reference: the
idol-sdk repository in a local venv, run eagerly on the 3090. The log was written up at the end of the port from the
runs' recorded outputs; times are approximate.

## Run 1 - 2026-11-17 ~14:00

Question: do the image encoders match? `ref_cond.py` dumps DINOv3 `last_hidden_state` + plain layer norm (feature1
[1,4101,1280]) and the FLUX.2 VAE mean latent, 2x2-patchified to 128 channels and bn-normalized (feature2), on the
gallery armchair.

- DINOv3 ViT-H+/16 needed `Dinov3Spec` to grow `gated` (SwiGLU MLP) and `final_norm`; ViT-L/16 keeps both off.
- FLUX.2 VAE encoder: diffusers names mapped onto the BFL encoder (`down_blocks.i.resnets.j` -> `down.i.block.j`, the
  mid attention's 1x1 convs from linears), latent channels 32.
- f32: DINOv3 rms 3e-6, VAE 5e-4. Pass.

Next: the denoiser on dumped inputs.

## Run 2 - 2026-11-17 ~16:00

Question: one velocity evaluation against `ref_model.py`'s f16 and f32 dumps (same latent, camera, t, condition).

- f32 path: matches the f32 dump.
- f16 blocks with fattn: 4.3% rms off the f16 dump. Not rope or norms: f32 attention gives 0.23%, and the reference's
  own f16-vs-f32 gap is 0.41%. The 12,294-token f16 value sums are the drift.
- Fix: fattn `f32_values` extended to head 64 (fattn_f32_values Run 5). Velocity 0.23%.

Dead end kept: f32 Sdpa for the attention is right but takes the run to 425 s.

## Run 3 - 2026-11-17 ~18:00

Question: decoders and activations. Octree logits and Gaussian decoder features against the reference on dumped
latents: 0.2-0.4% rms; the activations (softplus scale with its inverse-softplus offset and 0.0009 filter, rotation
lr 0.1 plus identity, sigmoid opacity, tanh xyz offsets) exact on equal features.

## Run 4 - 2026-11-17 ~20:00

Question: preprocessing. Short side to 1024 (round-half-even), BiRefNet when there is no real alpha, 3x3 erode,
bbox crop with Python's int truncation, Lanczos to 1024, PIL BLEND onto black: byte-exact against the reference's
`preprocess_image` on the gallery cut-out.

## Run 5 - 2026-11-17 ~22:00

Question: end to end, seed 42, 50 steps, 262,144 Gaussians, against the reference's `armchair.ply`.

- Camera: ours yaw 0.906, radius 4.06, fov 26.6; the reference's seed 42 differs, but its seeds 1 and 2 land on ours.
  Sampling noise differs (torch's generator), so per-seed equality is not expected; the camera is a sample.
- Gaussian statistics (scale, opacity, color histograms) close to the reference's; point renders comparable.
- 68 s against the reference's eager 49 s. Open: where the 19 s go.

## Run 6 - 2026-11-18 00:10

Question: does the job/CLI path hold on an opaque photo, and is its camera usable for a viewer start?

- `inference image-to-splat --image armchair_white.jpg --mode real` (the gallery image flattened on white):
  69.3 s, 262,144 Gaussians, camera yaw 0.804 pitch 0.134 radius 2.664 fov 44.4.
- Prepared image vs the reference's `preprocess_image` on the same jpg (its own BiRefNet run in f16): 4.7% of bytes
  differ, none by more than 6 (BiRefNet's f16 mask edge), mean 0.08.
- Rendering the reference's PLY from its predicted camera: eye at `radius * (sin yaw cos pitch, cos yaw cos pitch,
  sin pitch)` in the z-up frame reproduces the input view; the opposite sign shows the back. The exported PLY is
  `(x, -z, y)`, y down, so the web viewer's OpenCV frame stands it up and the start camera is `(ex, ez, -ey)`.
- GPU memory sampled with nvidia-smi at 200 ms: 755 MiB before, 8830 MiB peak, so ~8.1 GB. The job asks for 10 GiB.

Next: the real-weight job test (`image_to_splat_job`, 12 steps, 65,536 Gaussians): passes in 19 s.

## Run 7 - 2026-11-18 01:00

Question: where do the 20 s over the reference's 49 s go?

`nsys profile -t cuda` on a 10-step run (17.0 s wall, 14.8 s of kernels), `cuda_gpu_kern_sum`:

    41.7%  6.17 s   346  flash_attn_ext_f16<64,64,64,1,...,f32 values>
     9.0%  1.33 s   568  cutlass f16 gemm 256x128
     6.1%  0.90 s 10269  bmul_f32
     5.2%  0.77 s  2970  badd_f16
     4.8%  0.71 s   240  cutlass f16 gemm 128x256
     3.9%  0.58 s  3249  cast_f16_f32
     3.6%  0.54 s  1152  layernorm_f32
     ... bmul_f16, cast_f32_f16, bdiv_f32, fast_sum_f32, copy2d_f32, badd_f32, usqr_f32, ugelu_f16, bsub_f32

- Attention: ~31 TFLOP of attention per 10 steps in 6.2 s is ~50 TFLOPS, ~70% of the 3090's f16 (f32 accumulate)
  peak. Not where the gap is.
- The elementwise tail (~5 s per 10 steps): the f32 rotation (casts, narrows, 4 muls, sub, add, stack, cast), the
  f32 head RMS norms, the f32 LayerNorm round trips, modulation and gated residuals, ~60 launches per block.
- TRELLIS.2 already fuses the same block shape in bf16 (`trellis2/dit_cuda.rs`). Templated `dit.cu` over bf16/f16
  (entry points `<kernel>_bf16`, `<kernel>_f16`), and `head_norm_rope` gained per-head rotation strides and a
  rotate-then-normalize order (`Rotation { per_head, first }`). Idol's blocks use them on CUDA.
- Kernel cases (`dit_cuda::cuda_cases`, now with an f16 per-head rotate-first case) pass; velocity against the f16
  dump 0.22% rms (0.23% unfused); TRELLIS.2's and Idol's real-weight job tests pass.
- Default run: 52.9 s (69.2 s before), camera yaw 0.804 pitch 0.132 (0.134 before), GPU peak 8.1 GB unchanged.

Left: the learned rope's own ops (f32 LayerNorm, three small f32 GEMMs, index_select, cos, sin) run unfused every
block; ~4 s over the reference remains.

# Hi3D v1.0 (Sparc3D) investigation

Porting [hi3d-official/Hi3Dv1.0](https://huggingface.co/hi3d-official/Hi3Dv1.0), an image-guided coarse-mesh refiner
(Sparc3D, arXiv 2505.14521), into `inference-recon`. The model ships under the Hi3D Open License v1.0, which covers its
inference code and architecture as well as its weights; the port carries that license and its NOTICE beside the code.

## The model, as read from the reference (`sparc3d/`)

- **Ladder:** up to three passes. Pass 1 voxelizes the coarse mesh on a 64^3 grid (adaptive: the grid shrinks 5% at a
  time until at most 80,000 cells are occupied), samples latents with `transformer_512` and decodes a 512^3 surface.
  Passes 2 and 3 revoxelize that surface at 128^3 (90,000 cells) and 192^3 (110,000), sample with `transformer_1024` and
  decode at 1024^3 and 1536^3.
- **Condition:** DINOv2-large (`facebook/dinov2-large`, transformers layout) on the prepared image bilinearly resized to
  756 (2917 tokens, L2-normed), concatenated channel-wise with CLIP ViT-L/14's pooled embedding (L2-normed, repeated per
  token): `[1, 2917, 2048]`. The image is cropped to its alpha bounding box, squared, composited on white and padded so
  the subject fills 1/1.2 of the side.
- **Transformer** (both checkpoints the same shape, f16 torso): input linear 32 -> 256; one sparse res block, then a res
  block that mean-pools 2x2x2 to 2112 channels; 24 TRELLIS cross blocks (22 heads of 96, shared modulation, RoPE then
  RMS norm on self-attention q/k, plain cross-attention, tanh GELU MLP x4); mirrored output res blocks with skip
  concatenation and an upsample that scatters back by the pooling's inverse; f16 layer norm; output linear 256 -> 32.
  Res blocks: LN, SiLU, conv3 (submanifold), LN * (1 + scale) + shift, SiLU, conv3, SiLU, linear (zero-init), skip.
  RoPE frequencies are rescaled when a coordinate reaches 64 (pass 3's 96^3 token grid).
- **Sampler:** flow matching with Heun's second-order step, 25 updates. Timesteps are `linspace(1, 0)` warped twice by
  `f t / (1 + (f - 1) t)`: f = 2, then f = clamp((19 n - 20000) / (3 n + 60000), 1, 5) for n tokens. Each step evaluates
  at a churned sigma (1.04x above warped t 0.985, else 1.002x). Guidance only from t = 0.7: scale 4 at t >= 1, else
  10 t^2, against a zero condition. Latents denormalized by the configs' mean/std.
- **VAE decoder** (PTv3 half): submanifold conv 32 -> 512, res block, 2 serialized-attention blocks (z-order and
  transposed z-order, patches of 48, 32 heads, conditional position encoding by a conv), res block; four stages of two
  res blocks each, three of them followed by an unpooling that predicts each cell's 8 children by a conv head and keeps
  those with a positive logit; GroupNorm-Swish-conv to 8 sign logits + 32 attributes (8 corner SDFs, 8 corner
  deformations x3, tanh). Res blocks pass through untouched above 12M cells.
- **Extraction:** per-cell corner values averaged onto shared grid vertices, signs from the sigmoid of the sign logits,
  vertices moved by `0.5 * deform` and normalized by the decoded cells' centroid and max radius, then FlexiCubes' dual
  marching cubes (no learned weights). Holes up to 10,000 edges are ear-clipped. The final mesh drops components under
  0.5% of the faces, is simplified to 1M faces (2M at 1536) with pyfqmr, and is centered and scaled to the unit sphere.

## Run 1 - 2026-11-25 12:17

Reference on the 3090: `/mnt/data/lato2-venv` (torch 2.11 cu130, spconv 2.3.8, transformers 4.57) with an SDPA
stand-in for flash-attn's two varlen calls (flash-attn is not installed). Inputs: the repo's `assets/example.png`
(dragon bust, RGBA 436x640) and a coarse mesh from our own TRELLIS.2 (`inference image-to-3d --variant 512
--vertex-colors --faces 200000`: 188K faces in 24 s). Script and outputs in `/mnt/data/models/hi3d-ref`.

```
ref.py example.png coarse/model.glb ref512    # 512 target, 25 steps, seed 0
```

Question: does the reference run on a 24 GB card, and what does each stage produce?

- 7,926 cells at 64^3, 79 transformer calls (25 Heun updates, guided steps doubled), sampling 24.1 s, whole pass
  29.6 s, peak 7.2 GiB (the conditioner and both encoders resident in f32).
- Decoded 563,575 cells at 512^3, extracted 555,816 vertices and 1,111,256 faces.
- First attempts ran out of memory: calling the pipeline's private steps outside its `inference_mode` kept autograd
  graphs (21 GB by block 10). The dump script disables grad.

Next: port the conditioner, the transformer and the sampler, checked against the dumped stages.

## Run 2 - 2026-11-25 12:33

Conditioner, transformer and sampler against Run 1's dump (`hi3d::parity`, ignored; HI3D_DIR, HI3D_DINO, HI3D_CLIP,
HI3D_REF), run with `scripts/deep_checks.sh --ignore-default-filter --run-ignored only -E 'package(inference-recon) &
test(/hi3d::parity/)'`. Question: does each stage match from the reference's own inputs?

| stage | rel rms | note |
|---|---|---|
| DINOv2 tokens | 7.1e-6 | transformers weights renamed to the hub layout, `Bicubic { offset: 0 }` position resampling |
| CLIP pooled | 1.8e-2, then 4.2e-6 | fast_image_resize's Lanczos differed from PIL by up to 23 levels; `pil_resize` is exact |
| transformer, first call | 0.49, 0.39, 0.11, then 9.3e-3 | three bugs, below |
| 25 Heun steps (79 calls, as the reference) | 3.1e-2 | latents, f16 rounding on both sides |

The transformer bugs, found layer by layer (`ref_layers.py` hooks):

- `SparseDownsample`'s `scatter_reduce(zeros, ..., reduce="mean")` keeps the default `include_self=True`, so each
  pooled cell is its cells' sum over count + 1 (TRELLIS has the same quirk; the weights were trained under it).
- Hi3D stores each block's `modulation` as `[1, 12672]`; the shared block now takes `[6c]` or `[1, 6c]`.
- The fused CUDA head kernel (`head_norm_rope`) assumed a head dim that is a multiple of 64: at 96 each lane held 3
  values, so rotation pairs straddled lanes. Lanes now hold an even share (4 at 96, over 24 lanes); the unfused block
  matched at 7.2e-4 before the fix, the fused one 0.108, both 7.2e-4 after.

The remaining 5e-3 at the first residual block is the reference's: spconv's f16 submanifold conv is 3.2e-3 from its own
f32 result on the same input (max 0.36 on values up to 38). Our sampling loop takes about 15 s for the 79 calls (the
reference 24 s).

Next: the VAE decoder, FlexiCubes extraction and the mesh voxelizer.

## Run 3 - 2026-11-25 13:14

Decoder, extraction, voxelizer and the whole ladder. Question: does each match from the reference's inputs, and does
the full ladder fit a 24 GB card?

| stage | result |
|---|---|
| voxelizing the coarse GLB at 64^3 | 7,926 cells, identical to the reference's and in its order (13 ms) |
| decoding the reference latents (f32, then f16 below the coarsest level) | 563,581 cells vs 563,575, 563,482 shared; sign logits 1.6e-2, attributes 1.4e-2 rel rms |
| extraction from the reference's cells | 555,816 vertices and 1,091,368 faces, every face identical; vertices within 5.7e-6 |
| hole filling | 1,111,256 faces as the reference; 82 of its 19,888 new faces differ (f32 near-ties in the ear clipper) |

Getting extraction exact took the reference's orders: dual vertices numbered by how many a cube makes (ones first),
quads sorted by their edge's vertex pair, and the quads whose edge starts outside emitted before the rest
(`torch.cat((quads[flip], quads[~flip]))`). Hole filling traces loops by face order, so a different order closed
different loops (8, then 10 extra faces).

End to end (`a_refinement_runs_end_to_end`, our seed, same image and coarse mesh), per stage start:

| target | passes (cells -> decoded) | total | note |
|---|---|---|---|
| 512 | 7,926 -> 547,297 | 19.7 s | 972,596 faces after the debris filter |
| 1024 | + 36,149 -> 2,507,309 | 130.8 s | sampling pass 2 85 s; the reference 158.8 s (pass 2 ~107 s), peak 14.5 GiB |
| 1536 | + 86,790 -> 5,949,597 | 485 s | first OOM in the last decode; 17.5 GB peak after the fixes below |

The 1536 decode ran out of memory at first. The unpooling projected every parent to all 8 children (`[parents, 8c]`,
~7.8 GB at the last level) before keeping some: it now projects each kept child through its slot's block of the
weight. The levels below the coarsest run in f16 (as the reference's autocast stores them), group norms gather their
statistics over row chunks, and the transformer is dropped before the last decode. Renders of ours beside the
reference (`render.py`, `/mnt/data/models/hi3d-ref/ours/compare{512,1024}.png`) show the same refinement: scales on the
neck, teeth and horn ridges the coarse mesh lacks.

Next: the CLI, job and web UI, chaining TRELLIS.2's geometry, then speed (pass 3 samples for 297 s).

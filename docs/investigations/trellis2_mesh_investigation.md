# TRELLIS.2 meshing investigation

Phase 2 of image to 3D: TRELLIS.2's second stage meshes JoVox's 64^3 structure, as JoVox's `img2mesh.py --stage mesh`
does (`voxelgen/interfaces/trellis2.py`). The reference is TRELLIS.2 at 5565d24 (MIT) with its CUDA extensions
o-voxel-gpu at fa9ae23, FlexGEMM at 6f4147d and CuMesh at c4ad612 (MIT), and the TRELLIS.2-4B checkpoint at af44b45
(MIT). Weights and dumps are never committed.

## Run 1 - 2026-11-10 12:30

Question: what does the second stage compute, and what has to be ported?

`voxels_to_mesh(pipeline, prepared_png, seed, coords64)`:
1. `preprocess_image` on the 512^2 RGBA from JoVox's prep: crop to the square around alpha > 204 (no margin), color
   times alpha, RGB.
2. `get_cond(.., 1024)`: DINOv3 ViT-L/16 at 1024^2 (4096 patches + 5), the zero condition beside it.
3. `sample_shape_slat`: N(0, 1) noise of 32 channels on JoVox's 64^3 cells, 12 Euler steps of the 1024 shape flow
   (`slat_flow_img2shape_dit_1_3B_1024_bf16`: TRELLIS.2's `SLatFlowModel`, 1536 wide, 30 blocks, MLP 8192, shared AdaLN
   modulation, RoPE on the integer cell coordinates), `rescale_t` 3, CFG 7.5 only for t in [0.6, 1] with guidance
   rescale 0.5; then de-normalized by `shape_slat_normalization`.
4. `decode_shape_slat(.., 1024)`: `FlexiDualGridVaeDecoder` (channels 1024/512/256/128/64, ConvNeXt blocks 4/16/8/4/0
   around 3^3 submanifold sparse convolutions, four channel-to-space upsamplings that each predict which of the 8
   children exist), 64^3 -> 1024^3; per voxel a dual vertex, 3 edge-crossing flags and a quad split weight.
5. `flexible_dual_grid_to_mesh`: a quad per crossed edge whose 4 neighbouring voxels exist, split into two triangles
   by the split weights.
6. CuMesh `fill_holes` (boundary loops under a perimeter of 0.03) and `simplify` to 1M faces.

Already in the engine: the sparse DiT (JoVox's flow is the same model with a replaced input layer) and DINOv3. To port:
the plain `SLatFlowModel` input/modulation, the Euler CFG-interval sampler, submanifold sparse convolution (FlexGEMM:
a neighbour hash and implicit GEMM; here a gather and matmul per kernel offset), channel-to-space, the dual-grid mesh
extraction (hash lookups, CPU) and CuMesh's hole filling and simplification.

## Run 2 - 2026-11-10 13:40

Question: does the reference's second stage run here, and what does it produce on a JoVox structure?

The extensions build against torch 2.11+cu130 with CUDA 13.4 after two local patches in the scratch clones: `-std=c++20`
(torch 2.11's headers need it; all three setup.py files pin c++17) and `#include <thrust/tuple.h>` in o-voxel's
`flexible_dual_grid_cuda.cu` (CUDA 13's thrust no longer pulls in `make_tuple`). trimesh, plyfile, zstandard, easydict
installed to the same scratch target. `trellis2ref/mesh_ref.py` (scratch) builds the pipeline from the local checkpoint
(the shape flow and decoder only, both put in eval mode: `models.from_pretrained` leaves them training), DINOv3 from
its local folder, SDPA in place of flash-attn, and runs `voxels_to_mesh` on the reference's own robot structure (JoVox
seed 0, 6270 cells), recording every stage:
```
cond (DINOv3 at 1024, 4101 tokens)   0.49 s
shape flow, 12 steps (SDPA shim)     11.61 s   (~1.1 s a guided step, ~0.7 s an unguided one)
decode                                1.25 s   cells 6270 -> 27248 -> 110938 -> 447215 -> 1801342 (x4 a level)
mesh                                            1801342 vertices, 3619600 faces
fill_holes                            0.02 s    1804121 vertices, 3636268 faces
simplify to 1M                        0.36 s    475653 vertices, 958750 faces
peak allocated 7.24 GB
```
The 1024^3 level holds 1.8M cells at 64 channels; its 3^3 convolution's im2col is 6.2 GB in f16, so the port chunks
it.

## Run 3 - 2026-11-10 14:30

Question: do the ported shape flow, decoder and mesh extraction (`inference_recon::trellis2::{dit, shape_flow,
sparse, decoder, mesh}`) match?

JoVox's DiT block moved to `trellis2::dit` and gained shared modulation (`share_mod`: the model's one SiLU-linear of
the timestep, plus each block's learned f32 offset) and a configurable MLP width; JoVox's parity tests still pass
unchanged (f32 shared-draw structure equal, recorded forwards at rms 4-5e-3). The sparse convolution gathers each
cell's 27 neighbours (CPU hash, zero row for a missing one; tap v = (v/9, v/3%3, v%3) - 1 on (x, y, z), FlexGEMM's
neighbour order) into an im2col chunked to 512 MB and runs one matmul; a unit test checks it against a dense conv3d.

`TRELLIS2_DIR=... TRELLIS2_REF=.../robot0.safetensors cargo nextest run -p inference-recon --features cuda --lib
--run-ignored all -E 'test(/trellis2::parity/)'`, the reference's latents into our f16 decoder:
```
stage 0:   27247 cells vs   27248 (27247 shared), features on shared rms 1.8e-3
stage 1:  110934 cells vs  110938 (110933 shared),                     rms 8.6e-3
stage 2:  447224 cells vs  447215 (447178 shared),                     rms 1.6e-2
stage 3: 1801366 cells vs 1801342 (1800803 shared),                    rms 1.4e-2
decode 1.44 s (reference 1.25 s)
```
A few subdivision logits near 0 flip under f16; the mesh from our decoder has 3619670 faces against 3619600. The mesh
extraction on the reference's own decoder output first gave 3619460 of 3619600 faces equal: the quad split compares
weight products, and softplus as `log(exp + 1)` on the device flipped near-ties; computed as torch does
(`log1p(exp(x))`, x above 20) in f32 on the host, all 3619600 faces are equal and the vertices within 6e-8.

Shape flow, bf16 against bf16: one forward at rms 7e-3, the 12-step latents at 6.5e-2. To tell drift from a bug, both
in f32 (the checkpoint is bf16, so the weights are the same; the reference converted with `convert_to(float32)`): a
first attempt still showed 1e-2 from block 0's first LayerNorm on, traced to the test loading our flow in bf16 (an
edit that never applied after rustfmt rewrapped the line). Corrected:
```
input layer 0, shared modulation 2.5e-7, block 0 norm1 5.5e-8, q after rope 2.1e-7, self-attn 2.4e-6,
cross-attn 6.0e-6, mlp 1.8e-6, block 29 1.3e-6; first step v 1.2e-6 (both branches)
12-step latents: rms 2.6e-5 (max 1.3e-2)
```
So the flow and the CFG-interval Euler sampler (with guidance rescale) are the reference's; the reference's own bf16
latents differ from its f32 ones by 7.6%, more than ours do from its bf16 ones (6.5%).

Next: hole filling and simplification (CuMesh), then the mesh into the voxels job.

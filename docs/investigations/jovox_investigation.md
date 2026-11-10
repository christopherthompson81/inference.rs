# JoVox investigation

Native support for JoVox (an image-conditioned 64^3 voxel structure from discrete flow matching over a 4^3 -> 16^3 ->
64^3 occupancy pyramid), with its Python code as the reference. Phase 1 is the structure (DINOv3 ViT-L/16, the flow,
the sampler); TRELLIS.2's second stage, which meshes the structure, is a later phase. Inputs must carry their own
transparent background: the reference's background remover (RMBG-2.0) is non-commercial and not ported. Weights and
reference dumps are never committed.

## Run 1 - 2026-11-10 07:10

Question: what does the reference need, and does its flow run here without its CUDA extensions?

Reference: github.com/yuyujunjun/JoVox at 6354b0e, with TRELLIS.2's `trellis2/{models,modules}` at 5565d24 (fetched
file by file, an empty `trellis2/__init__.py` in place of the pipeline imports); torch 2.11+cu130 from the user site.
Weights: xiangjuntang/JoVox-64 `model.safetensors` (6.16 GB, all F32: 678 tensors under `sparse_structure_flow_model.`
plus `_fm_step`; the DINOv3 encoder is not in it). DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m) is gated: the
download was refused until access is granted.

The model, from `voxelgen/network/{backbone,sampler,schedule,hierarchy}.py`: TRELLIS.2's `SLatFlowModel` at 1536 wide,
30 blocks, 12 heads (head dim 128), cross-attention to 1024-wide DINOv3 tokens (all 1029 at 512^2: class, 4 registers,
32 x 32 patches, through a LayerNorm with no affine), q/k RMS norms in both attentions, blocks in bf16. Its input layer
is replaced: each token's 8 slots (a byte of child occupancy each, or 256 = masked) embed through a shared table of
child states and a `grand_proj` over the grandchild bits, plus a Fourier point embedding of the cell center in the unit
cube and a level embedding; RoPE runs on those centers times 256 (21 frequencies a axis, the 64th pair identity). Two
timestep embedders take each level's known share of slots. 64 root tokens (4^3 cells) and one per occupied 16^3 cell.

The sampler (B=1): 72 grid steps then 8 cleanup steps. Each step remasks the earliest-revealed known slots down to the
plan's keep share (ties random), draws every slot from one guided forward (cfg 1.5 against a zero condition; greedy at
temperature 0, the released setting), then reveals `ceil(masked * h)` random masked slots. Its randomness is torch's
CUDA generator, so samples can't match draw for draw; the plan is deterministic numpy.

flash-attn and the conv/mesh extensions are not installed; the flow needs only attention, so `ref/flow_ref.py`
(scratch) replaces `sparse_scaled_dot_product_attention` with SDPA per sequence and runs `sample()` on a synthetic
condition (`randn(1, 1029, 1024)`, seed 1234), recording forwards 0, 30 and 60 (inputs and logits), the plan and the
voxels, as safetensors:
```
seed 0: 80 forwards, 16040 voxels, 8.8 s (first run, cold)
seed 1: 80 forwards,  2273 voxels, 6.0 s
```
The voxel counts mean nothing on a random condition; the recorded forwards are what the port is checked against.

## Run 2 - 2026-11-10 08:05

Question: do the ported flow (`inference_recon::jovox::flow`) and plan (`sampler::Schedule::plan`) match?

`JOVOX_DIR=.../jovox-64 JOVOX_REF=.../flow_seed0.safetensors cargo nextest run -p inference-recon --features cuda
--lib --run-ignored all -E 'test(/jovox/)'` (the parity tests in `jovox/parity.rs`):
```
forward 0:    64 tokens, max 2.991e-1 rms 4.141e-3, argmax agreement 1.00000
forward 30:  292 tokens, max 5.385e-1 rms 4.481e-3, argmax agreement 0.99957
forward 60: 1044 tokens, max 1.026e0  rms 4.946e-3, argmax agreement 0.99988
plan (0.8, 72, 1.0) and (0.7, 40, 0.3): equal within 1e-12
```
Logits are the guided batch's two rows (conditional and zero condition) over all 256 values of every slot. A relative
RMS of ~5e-3 is bf16's level; the greedy pick agrees on all but a few slots in ten thousand. The slot embedding folds
`grand_proj` into a 216-row table at load (exact up to f32 summation order).

Next: DINOv3 once access is granted (parity of the condition on a real image, then end-to-end structures against the
reference's on the same image, by IoU against its seed-to-seed spread).

## Run 3 - 2026-11-10 08:30

Question: does the ported sampler (`sampler::run`, driven as `JoVox::sample` drives it) finish and land where the
reference does on the same synthetic condition?

`a_structure_samples_under_the_recorded_condition` (parity.rs), the recorded condition, 72 steps, cfg 1.5, greedy:
```
seed 0: 12330 voxels, 9.43 s
seed 1:  6281 voxels, 5.04 s
seed 2:  1592 voxels, 2.71 s
```
The reference gave 16040 and 2273 for its seeds 0 and 1 on this condition: a random condition leaves the structure
to the remask and reveal draws, and both spread across the same range. Time follows the voxel count (more cell
tokens a forward) and sits at the reference's 6-9 s; the per-forward CPU work (slot counts, point features, rope
tables) is not yet measured against the forwards. The draws come from `StdRng`, not torch's CUDA generator, so no
seed reproduces the reference's sample.

Next: a real image's condition, which needs DINOv3 (access requested, still refused at 08:25).

## Run 4 - 2026-11-10 10:05

Question: with DINOv3 (access granted), does an image go through the port as through the reference?

`ref/e2e_ref.py` (scratch): the reference's prep stage (alpha kept), `load_rgba_image`, TRELLIS.2's
`DinoV3FeatureExtractor` (f32; transformers 5.19 here moved the layers to `model.model.layer`, aliased back), the flow
in bf16 and `sample()` at the released settings, on the repo's example images (512^2 RGBA). Reference structures:
```
apple: 18245 18378 18150 18140 voxels   9.4-9.9 s each
robot:  6270  7740  6428  6629          6.3 s
sofa:  16476 16852 26985 16426          8.9-9.5 s
```
Port, same images (`an_image_matches_the_reference_end_to_end`): the prepared image equal byte for byte (PIL's
Lanczos through premultiplied alpha), DINOv3's input equal, its 1029 tokens at rms 1.0-1.2e-6. Structures, seeds 0-3
each side, voxel IoU mean (min):
```
         reference-reference   ours-reference
apple    0.221 (0.147)         0.268 (0.109)
robot    0.222 (0.144)         0.068 (0.050)
sofa     0.071 (0.049)         0.130 (0.046)
```
64^3 voxel IoU between two samples of one image is low even within the reference. Robot falls outside it: our robots
spanned x 9-55 and y 17-47, the reference's x 18-45 and y 11-54, every seed.

## Run 5 - 2026-11-10 10:35

Question: is the robot's difference the port's, or the sampler's randomness?

Shared draws: both samplers draw from a splitmix64 counter stream instead of their own generators (the reference's
`schedule.torch.rand` swapped for it; the port's sampler now takes a `Uniform` source and picks by the reference's
keys, `born + 0.5 * u` for remasks and `u` for reveals, over the same full-level draws, skipped where the reference
skips them). Tracing every forward's tokens: the bf16 runs agreed for 10 steps, then one revealed slot held 15 here
and 0 there. Each step had 1-4 slots whose guided argmax differed, all near-ties (gaps 0.001-0.1 between logits of
3-13); a flip that lands on a revealed slot sends the samples apart. Logits rms 3.7e-3 (bf16's level).

Both in f32 first showed rms 1.2e-3, too large: the reference builds the flow in bf16 (`dtype="bfloat16"`), so its
weights were rounded on load and `convert_to(float32)` after does not undo that. Converted before loading: logits rms
4e-7, no argmax differences, and the shared-draw structures equal voxel for voxel:
```
robot seed 0: 6450 = 6450        apple seeds 0-1: 17882, 18428 equal        sofa seeds 0-1: 16712, 16160 equal
```
So the port is the reference; in bf16 the rounding of different kernels decides near-ties. The robot's spread is the
model's: x/y extents over seeds come in two modes, x-wide or y-wide (a front/side ambiguity). The reference's torch
generator over 10 seeds: 5 y-wide (25-35 x 43-49), 5 x-wide; the port's StdRng over 6: 5 x-wide, 1 even; shared draws
over 6: 3 and 3. The 4-of-4 split of Run 4 was the draw, not the port.

The end-to-end test (`JOVOX_F32=1` against an `F32=1 DET=2` dump) now asserts the prepared bytes, the condition (rms
< 1e-4) and, in f32, equal structures under shared draws.

# LATO.2 investigation

LATO.2 (LoHhhha/LATO.2, MIT; weights 0x4c48/LATO.2) generates a mesh in two flows: V-Flow places vertices (a sparse
DiT over a 64^3 voxel scaffold, decoded by a vertex VAE that subdivides to 1024^3) and T-Flow connects them (a dense
DiT over the vertices, decoded by a topology VAE into pairwise edge logits; faces are the edge triangles, small open
rings filled). From an image the scaffold comes from TRELLIS-image-large's sparse-structure flow and decoder under
DINOv2 ViT-L/14-reg features; the image is matted with BiRefNet first.

## Run 1 - 2026-11-24 15:20

Question: does the official image pipeline run here, and what does each stage hand the next?

Reference environment: `/mnt/data/lato2-venv` (system site packages: torch 2.11+cu130; added spconv-cu126, trimesh,
kornia, timm, matplotlib). Two stand-ins on `PYTHONPATH`, nothing else changed: `flash_attn` as per-segment SDPA (the
only attention kernel the reference calls) and an inert `torch_scatter` (encoder-only import). Harness `ref.py` (session
scratch) runs `scripts/image_e2e_inference.py`'s stages with its defaults (SS 25 steps cfg 5 interval 0.5-1 rescale 3;
V-Flow 24 steps cfg 3; T-Flow 50 steps; vert_num 2000; seed 42) and dumps every input and output.

```
PYTHONPATH=shim /mnt/data/lato2-venv/bin/python -P ref.py <image> a_bf16.safetensors          # the shipped precision
PYTHONPATH=shim /mnt/data/lato2-venv/bin/python -P ref.py <image> a_f32.safetensors --f32    # no autocast, fp16 blocks in f32
```

Image: one of the repo's own example images (a stylized armored figure holding a cup; 1010^2 after the 1024 cap).

| stage | bf16 (shipped) | f32 |
|-|-|-|
| scaffold voxels (64^3) | 24491 | 24512 |
| vertices after the VAE (1024^3) | 2577 | 2535 |
| edges | 7184 | 7234 |
| faces (raw / after ring fill) | 4515 / 4759 | 4576 / 4825 |

The f32 mesh renders as the figure (body, cup, halo). Vertex count follows the density knob loosely: 2000 asked, ~2550
made. The reference takes ~4 min here, almost all of it the stand-in's Python loop over windowed-attention segments.

Configs read from the checkpoints (`torch.load(weights_only=True)` works: config dicts + f32 state dicts):
- vflow: 768 wide, 12 blocks, 12 heads, cond 1024 (DINOv2 tokens: cls + 4 registers + 37^2 patches, pre-norm, then an
  affine-free LayerNorm), RoPE (head 64: 10 frequencies an axis, last 2 pairs unrotated) applied *before* the q/k RMS
  norm (TRELLIS.2 normalizes first), no cross q/k norm, density embedder (a second timestep MLP on vert_num/10000*1000).
- vvae decode: latent expander (8 blocks, 512, shifted 8^3 windows, RoPE, no q/k norm), vertex pruning head at 0.5,
  four upsample levels 64->1024 (GroupNorm32 + SiLU, subdivide, two submanifold convs + 1x1 skip, a self-context cross
  transformer with APE, pruning head keeping at least one child per parent) each followed by a cross transformer (RoPE,
  full attention) over the 64^3 latent.
- tflow: 768, 16 blocks, 12 heads, adaLN with 7 chunks (cross gate), q/k RMSNorm (eps 1e-5, variance in the input
  dtype), cross-attention to the voxel encoder's 8^3 tokens; sampled from t=0 to 1 (not 1 to 0), 50 steps, no CFG.
- tvae decode: 768, 8 heads (head 96: 16 frequencies an axis, all pairs rotated), 2 stages of (graph block without
  adjacency = full attention, no RoPE + 3 plain blocks), then 8 connection blocks with RoPE, then a pair MLP
  (1536 -> 256 erf-GELU -> 1) symmetrized over (u, v) and (v, u), edge where the logit > 0.
- TRELLIS SS flow v1: 1024 wide, 24 blocks, 16 heads, APE, q/k RMS norm on self-attention only, f16 blocks.

Next: port stage by stage, each checked against the f32 dump (`a_f32.safetensors`) from the stage's own inputs.

## Run 2 - 2026-11-24 16:05

Question: does each ported stage match the f32 reference from the reference's own inputs?

`lato2::parity` (ignored; LATO2_DIR, TRELLIS_LARGE_DIR, DINOV2_PTH, LATO2_REF=a_f32.safetensors), run with
`scripts/deep_checks.sh --ignore-default-filter --run-ignored only -E 'package(inference-recon) & test(/lato2::/)'`:

| stage | rel rms | exact outputs |
|-|-|-|
| conditioning image (crop, Lanczos, premultiply) | - | all 518^2 x 3 bytes equal |
| DINOv2 tokens (pre-norm + affine-free LN) | 3.5e-6 | |
| structure flow, 25 steps | 3.8e-5 | |
| structure decoder logits | 7.3e-7 | 24512 voxels equal |
| V-Flow first velocity (cond / uncond) | 8.2e-6 / 5.1e-6 | |
| V-Flow after 24 steps | 2.1e-3 (max abs 1.35) | |
| vertex decoder levels 1-4 | 4.1e-6, 1.5e-6, 6.9e-7, 1.4e-6 | kept cells equal at every level (2527, 2535 x 3) |
| offsets | 4.5e-7 | |
| voxel conditioner | 1.2e-6 | |
| T-Flow, 50 steps | 1.2e-5 | |
| T-VAE vertex features | 1.4e-6 | |
| edges | - | 7234 = 7234, none missing or extra |
| faces (raw and ring-filled) | - | 4576 raw equal; 4825 filled, all shared |

V-Flow's end state drifts while its first step matches to 1e-5: guidance at s=3 scales each step's rounding difference
by 1 + 2s = 7, so 24 steps grow 1e-5 to 1e-3. Gated at 5e-3 with the first step at 1e-4 (LATO2_VSTEP, `vstep.py`).

Dead end on the way: the first decoder level kept 3349 cells where the reference kept 2527. Hooks on the reference
(`vvae_ref.py`) showed everything matching through the 3^3 convolutions and the upsample's transformer input already
off (rel 0.36): the 1x1 skip convolution. A least-squares fit of the reference's skip output from its input matched the
checkpoint's weight read as `weight.view(in, out)` (residual 5e-7), not as its declared `[out, 1, 1, 1, in]` layout
(cosine 0.003 with the fit): spconv's 1x1 fast path multiplies `features @ weight.view(in_channels, out_channels)`
whatever the layout, and the model was trained through it. The 3^3 convolutions use the declared layout.

The ring filling walks Python sets of ints; ours walks sorted sets. On this mesh the filled faces are identical, so the
order did not matter here; it could pick a different equal-length ring elsewhere.

Next: the whole pipeline at the shipped precision (f16 structure blocks, bf16 V-Flow/VAEs, f32 T-Flow) on the GPU,
its time, and the CLI.

## Run 3 - 2026-11-24 23:00

Question: why does a served mesh look about half holes (a gnome figure, from the LAN web UI), when the reference's
demo shows a few?

The faces are written in index order (`u < v < w`, as the reference writes them), so their winding is arbitrary. A
check of edges two faces share on the robot mesh: 5343 of 7580 walked the same way by both faces (inconsistent), while
only 143 edges border a single face. The viewer's single-sided material culls the inward half; the matplotlib previews
used `|n . light|` shading and hid it. The reference's own output carries the same arbitrary winding.

Measured with a z-buffered render that draws back faces black (`zbuf.py`, scratch), as the share of the object's pixels:

| winding | robot (az 0.5 / 3.6) | gnome |
|-|-|-|
| as generated (the reference's order) | 0.479 / 0.518 | 0.479 / 0.490 |
| spread across two-face edges, each piece flipped by its signed volume | 0.248 / 0.221 | |
| the same, flipped by normals against the piece's center | 0.248 / 0.221 | |
| per piece, ray parity along a few normals | 0.147 / 0.148 | |
| spread stopping at folds (normals past 120 degrees), escape vote per piece | 0.139 / 0.145 | |
| every face with a clear escape vote decides itself, the rest spread from them | 0.006 / 0.001 | 0.021 / 0.007 |

Dead ends: spreading the winding is consistent across every edge (0 inconsistent afterwards) yet still turns regions
inside out, because the predicted connectivity folds back on itself and joins nested shells: the largest robot piece
(1565 faces) had 376 faces whose normal escapes the mesh outward and 354 inward. Ray parity counts crossings of
interpenetrating parts (the arms through the torso) and flipped the torso wrong.

What works: per face, a ray from its center along its normal and one against it; a face where exactly one escapes the
mesh takes that side (2199 of 5129 robot faces decide this way); the enclosed rest take the winding of a decided
neighbour across an edge only the two share. The GLB is now also double-sided, so a face this still gets wrong shows
instead of leaving a hole. Cost: brute-force ray tests on all cores, well under a second at these face counts.

## Run 4 - 2026-11-24 23:40

Question: after the winding fix the served gnome figure still shows hole defects; what closes them?

Measured on that figure (its job input, background removed, seed 0) with `holes.py` (scratch: boundary edges, their
connected pieces and sizes) and renders with the open edges drawn red:

| fill | faces | boundary edges | pieces (largest, in edges) |
|-|-|-|-|
| rings up to 12 (the reference) | 4226 | 380 | 11 (160, 94, 25, 22, 18) |
| rings up to 24 | 4273 | 335 | 10 (137, 94, 25) |
| rings up to 64 or 200 | 4273 | 335 | unchanged from 24 |
| rings 24 + CuMesh fan fill (perimeter 0.15 / 0.4 / 1 / 5) | 4278-4394 | 330 / 286 / 214 / 214 | 10 / 7 / 4 / 4 |
| rings 12 + boundary loops closed by min-area triangulation (vertex walk) | 4297 | 297 | 5 (160, 94, 25) |
| the same, walking edges and cutting sub-loops at revisited vertices | 4345 | 237 | 4 (154, 65, 9, 9) |

Only 2 of 218 boundary vertices branch; the open edges sit in the beard and lower body, as long winding cracks. Longer
rings stop helping at 24: the ring search rejects rings with chords and avoids vertices adjacent to both ends, and
face-less predicted edges cross the cracks. The fan fill (a new vertex at each loop's center) needs a perimeter as large
as the whole mesh to reach the cracks and would fan across them. A minimum-area triangulation of each boundary loop
zips a crack shut without new vertices; walking edges (not vertices) and cutting a loop off wherever the walk returns
to a vertex handles rims that touch themselves. In the z-buffered render the shaded surface then shows almost no gaps
(back faces 1.2% / 0.3% of the object): the red edges left lie on the rims of nested inner layers of the beard, under
other surface, and end at edges of three or more faces rather than closing.

Kept: loop closing up to 64 edges by default (`--close-loops`, job `close_loops`).

# Baked-texture GLB investigation

TRELLIS.2's export (o-voxel's `to_glb`, remesh=False path first): CuMesh cleanup, UV unwrapping (CuMesh's normal-cone
clustering, then xatlas), rasterized baking of the PBR volume into textures, inpainted seams, a GLB with a PBR
material. Approach agreed with the user: vendor xatlas now (exact parity with the reference's unwrapper), then port
it to Rust piece by piece with the vendored copy as the oracle. The narrow-band dual-contouring remesh is discussed
and deferred (it adds an inner shell taking ~half the face budget, inflates the surface by a voxel, and needs BVH
distance queries); its cleanup chain is ported either way.

## Run 1 - 2026-11-10 19:10

Question: is CuMesh's xatlas upstream xatlas, and does it build here behind its C API?

CuMesh c4ad612's `third_party/xatlas/xatlas.{cpp,h}` against jpcy/xatlas master f700c77: identical apart from the
namespace (`xatlas` -> `cumesh_xatlas`) and a trailing newline. CuMesh's binding adds each cluster as a mesh
(positions, u32 indices), runs `ComputeCharts` and `PackCharts` with xatlas's default options (CuMesh's Python
defaults equal xatlas's: normal deviation 2, roundness 0.01, straightness 6, normal seam 4, texture seam 0.5, max cost
2, 1 iteration; bilinear, rotate charts, rotate to axis, no resolution or texels-per-unit, so a ~1024^2 estimate) and
normalizes uvs by the atlas size.

Vendored as `crates/inference-xatlas` (upstream f700c77, unmodified, built by `cc` with `XATLAS_C_API=1`), with a safe
`unwrap(&[MeshInput]) -> AtlasOutput` doing what CuMesh's `uv_unwrap` does with the clusters. A unit cube unwraps into
one atlas: 12 triangles mapped, no invalid charts, uvs in [0, 1].

## Run 2 - 2026-11-10 (evening)

Question: does a CPU port of CuMesh's `compute_charts` (to_glb's options: 90 degree cone, no refinement, 1 global
iteration, area penalty 0.1, perimeter^2/area penalty 1e-4) give the reference's clusters?

Reference: `charts_ref.py` (scratch) on the 512 run's robot mesh (462K vertices, 944K faces): CuMesh
`remove_degenerate_faces` (removes none here) then `compute_charts`, 0.38 s on the 3090, 67139 charts. Three reruns
give identical ids (deterministic despite its atomic perimeter sums).

Port `trellis2::charts` (`compute_charts`, `remove_degenerate_faces`, `chart_meshes`), test
`charts_match_the_reference` (TRELLIS2_CHARTS_REF):

    charts in 4.17 s: 67145 vs 67139, 772911/944179 faces in the same chart; seams 566.111 vs 565.564;
    faces per chart p50/p90/p99/max [2, 8, 59, 13492] vs [2, 8, 58, 19067]

Not bit-exact, and can't be cheaply: the semantics match line by line (cone = normalized sum of unit face normals and
the widest angle to one; cost = merged half-angle + area + perimeter^2/area terms; a pair collapses when under the
threshold and cheapest at both charts, packed (cost bits, edge id) compare; cones recomputed every round; final split
into connected pieces), but CuMesh normalizes face normals with `rsqrtf` (approximate) and sums areas, normals and
lengths in CUB's tree order, so exact cost ties on the voxel-flat faces break differently and the merge order
cascades (18% of faces end in a differently-cut chart; one large chart split differently, 13.5K vs 19K faces). The
clustering's measures agree: chart count within 0.01%, seam length within 0.1%. The test asserts those (1e-3).

Implication: parity for the clustering is statistical; xatlas's output gets compared the same way (atlas size,
chart counts) rather than per vertex. CPU time 4.2 s (vs 0.38 s GPU) is per-round serial sums over faces; worth
parallelizing once the whole export is timed.

## Run 3 - 2026-11-10 (evening)

Question: does `charts::uv_unwrap` (our clusters, each added to the vendored xatlas, gathered in cluster order as
CuMesh does) unwrap like CuMesh's `uv_unwrap`?

Reference `unwrap_ref.py` (scratch), same 944K-face mesh: 16.2 s (xatlas on the CPU is nearly all of it), 908074
vertices. Test `the_unwrap_matches_the_reference` (TRELLIS2_CHARTS_REF + TRELLIS2_UNWRAP_REF):

    unwrapped in 20.60 s into 1511x1513: 914264 vertices / 944179 faces vs 908074 / 944179; uv coverage 0.3975 vs
    0.4008; faces without uv area 1126 vs 1126

First run panicked: a cluster xatlas gives no charts has a null chart array, which `slice::from_raw_parts` rejects;
the wrapper now treats a zero count as empty. Every face is mapped; seam-split vertices +0.7% and coverage -0.8%
follow from the 18% of faces clustered differently (Run 2); the zero-area faces are the same count. Asserted within
1e-2. The extra 4 s over the reference is our CPU clustering.

Next: the remove_duplicate_faces / repair_non_manifold_edges / remove_small_connected_components /
unify_face_orientations chain, then baking.

## Run 4 - 2026-11-10 (evening)

Question: do host ports of CuMesh's `remove_duplicate_faces`, `repair_non_manifold_edges`,
`remove_small_connected_components(1e-5)` and `unify_face_orientations` match it?

Reference `topology_ref.py` (scratch) runs them in sequence on the 944K-face mesh, dumping each stage; test
`topology_passes_match_the_reference` (TRELLIS2_TOPOLOGY_REF) runs each of ours from the reference's previous stage:

    dedup  in 0.16 s: 462167 vertices / 943959 faces, identical
    repair in 0.22 s: 556529 vertices / 943959 faces, identical (corners joined across manifold edges, numbered by
                      lowest corner, as CuMesh's min-root union)
    small  in 0.20 s: 474820 vertices / 886361 faces, identical
    unify  in 0.16 s: same vertices, 793694 faces identical, the other 92667 all flips;
                      inconsistently wound manifold edges 1406 (ours) vs 7855 (CuMesh), input 247078

Repair splits 94K vertices and the component pass then drops 58K faces: the dual-grid mesh has many pieces touching
only at vertices or through 3+-face edges. Unify differs only where a piece has conflicting loops: there the result
depends on union order, and CuMesh's concurrent hooks (`atomicMin` of root<<1|parity, parity 0 winning a tie at the
same root) race, leaving 5.6x more disagreeing edges than a sequential union-find. The test asserts only flips differ
and ours has no more inconsistent edges.

The input's 247K inconsistent edges (of ~1.3M) are inherited from the extracted mesh (whose faces matched the
reference's exactly, trellis2_mesh Run 3), so `to_glb`'s final unify matters for the baked GLB's normals.

## Run 5 - 2026-11-10 (evening)

Question: does a port of OpenCV's Telea inpainting (`inference_vision::inpaint_telea`) give cv2's bytes? `to_glb` fills
every texel no face covers: base color radius 3 (3 channels), metallic / roughness / alpha radius 1.

Reference: the installed cv2 is 5.0.0 (build info: baseline SSE SSE2 SSE3, GCC 14.2); its `photo/src/inpaint.cpp` at
the 5.0.0 tag equals the 5.x branch (the 4.x branch has a SIMD rewrite meant to be bit-identical). Ported from it,
keeping its quirks: FIFO ties in the heap, the band and the distance ring marched outward first (negated `t`),
central image differences doubled, `km`/`lp` edge clamps, and `round_cast`'s `saturate_cast<uchar>(sat + 0.5)`
(round half to even of sat + 0.5).

`bake_ref.py` (scratch) runs `to_glb` on the 1024-cascade reference dump with `cv2.inpaint` wrapped to record its
inputs and outputs. First comparison on those textures:

    color inpainted in 4.06 s: 22778 of 12582912 values differ   (all in unknown texels, 18859 of them by 1)
    metallic 14473 / roughness 140278 / alpha 12903 of 4194304 differ

A float32 Python transcription of the 5.0.0 source (scratch `inpaint_sim/sim.py`) agreed with the Rust port, not
cv2, on 200 generated small cases (36 failing; smallest 5x4, one value: 101 vs 102). The failing pixel sat on an
exact tie: sat = 100.99999 here vs >= 101.0 there, and 101.5 rounds to even, 102. Not FMA (the baseline has none;
contracting the accumulations made it worse: 50 failing) and not `sat`'s precision. Sweeping which unqualified math
calls take double: `lev = 1/(1 + fabs(dt))` computed in double, with 1 + |dt| never rounded to float, makes all 200
cases match. Under `using namespace cv` the unqualified `fabs` binds the C library's double `fabs`, while `sqrt` in
`sat` stays float (taking it in double broke 2 cases).

Implication: kept as a one-line comment at the `lev` line; pinned unit tests for a flat hole and that border tie,
both with cv2's outputs. The real-texture comparison is rerun in Run 6.

## Run 6 - 2026-11-10 (evening)

Question: does the whole bake (`trellis2::bake::bake`: `to_glb`'s cleanup chain, our clustering + vendored xatlas,
UV rasterization, BVH projection onto the extracted mesh, trilinear sampling, Telea) match `to_glb` end to end?

`bake_ref.py` (scratch, needed nvdiffrast built into scratch: CUDA 13.4, `-std=c++20`) on the 1024-cascade reference
dump (4.3M extracted faces, 2.1M texture voxels): 303 s, of which ~287 s in xatlas's ComputeCharts (1374 clusters,
one huge chart dominating). Test `the_bake_matches_the_reference` (TRELLIS2_FULL_REF + TRELLIS2_BAKE_REF):

    color / metallic / roughness / alpha inpainted: 0 differing values each (2048^2)
    baked in 64.6 s: 686384 vertices / 963916 faces vs 696700 / 970159, 2358795 texels covered vs 2346119
    surface color over 334439 texels: mean abs diff 0.57 levels, p50 0.3 p90 1.3 p99 6.0

Surface color: each reference texel's 3D point (its atlas, rasterized by our rasterizer) looked up on our baked mesh
through our atlas. The tail (p99 6) is at chart borders and where the simplifications differ. Our xatlas run is
~4.5x faster than the reference's on the same input: our clusters differ (Run 2) and none is the reference's
outlier chart.

End to end through the CLI (`inference image-to-3d`, 1024-cascade, robot photo, background removed): 135.4 s wall
(generation ~69 s, export 66.5 s), 3.76 GB peak RSS, GLB 710518 vertices / 986697 faces with 2048^2 base color and
metallic-roughness PNGs. Rendered with nvdiffrast beside the reference's `to_glb` output (scratch `render.py`): both
cleanly textured, no black seams; poses differ as separate generations do.

Both GLBs are now y-up as `to_glb` writes them ((x, y, z) -> (x, z, -y)); the vertex-color GLB used to keep the
z-up frame. The extracted-mesh PLY keeps TRELLIS's frame.

## Run 7 - 2026-11-10 (night)

Question: does a CPU distance query (the bake's BVH, now with an allocation-free stack and a batched
`unsigned_distance`) and a port of CuMesh's `remesh_narrow_band_dc` (`trellis2::remesh`) match cuBVH and CuMesh?

`remesh_ref.py` (scratch): the 1024 reference mesh hole-filled (4296850 faces), cuBVH distances at 1M random points in
[-0.55, 0.55]^3, and the remesh as `to_glb` runs it (1024^3, band 1, project back 0.9): 0.27 s on the GPU, 4141500
vertices / 8284084 faces; CuMesh's simplify to 1M then takes 1.0 s. Test `distances_and_the_remesh_match_the_reference`:

    bvh over 4296850 faces in 1.70 s; 1000000 distances in 5.11 s, worst difference 1.19e-7
    remeshed in 7.56 s: 4141502 vertices / 8284088 faces vs 4141500 / 8284084; nearest ours->ref mean 4.61e-8 p99
    1.41e-7

Two extra band voxels (a distance on the band threshold), positions otherwise the same. The remesh is the level set of
distance - one voxel, so it is a closed shell around both sides of the surface; `--remesh` bakes it single-sided as
`to_glb` does.

Bug found on the way: `uv_unwrap`'s vertex normals were never filled (an edit that did not apply), so the textured GLB
declared a NORMAL accessor with no data. Fixed on the PR branch with a test that every accessor fits its buffer view.

## Run 8 - 2026-11-10 (night)

Question: where does the export's time and memory go, and what can be cut without changing its output?

Phase timings (temporary instrumentation, robot, 1024-cascade, CLI on the 3090 box, 16 threads):

    default:  cleanup 16.0 s, unwrap 50.6 s (clusters 3.5 s, xatlas charts ~42 s, pack 0.9 s), raster 0.03 s,
              projection 0.5 s, sampling 0.6 s, inpainting 9.5 s
    --remesh: remesh + simplify 39.5 s, unwrap 42.2 s, inpainting 8.9 s

xatlas ran 355 s of CPU in 28.5 s wall: fully parallel, just a lot of work. `perf`: UniformGrid2 (the chart boundary
self-intersection check, run for every face added to a chart, each time rebuilding a grid of up to 512 cells a side)
~36%, `addFaceToChart` 10-15%. Changes, each checked by the atlas hash being unchanged:

- xatlas built without `NDEBUG` had its debug checks on (`XA_DEBUG`): defining it, charts 42 -> 30 s.
- Per-cell tails in `createGrid`'s cell lists: no change (lists are short); dropped.
- Stamped cells instead of rewriting every offset: 33.5 -> 29.1 s on the 1024 mesh; kept (README lists it). The
  rest of `createGrid` is the cell loads themselves (`perf annotate`: one compare on the cell stamp takes 43%).
- The four inpainting fills run side by side: 9.5 -> 4.7 s.
- The pipeline stopped filling holes, simplifying to 1M and sampling vertex attributes when the result is baked
  (the bake redoes the cleanup from the extracted mesh); vertex colors do it on request (`Generated::vertex_colored`).
- Covered texels go straight into byte planes (no 2048^2 array of f32 attributes).

Result: 135 s -> 114.8 s end to end (export ~47 s), `--remesh` 127.7 s. Memory, sampled every 0.5 s: anonymous RSS
~1.1 GB through generation, peak 2.3 GB at the end of the export; file-backed peak 2.0 GB at 2.5 s (checkpoints
mapped while loading), which is what `/usr/bin/time`'s 3.9 GB maximum resident counts with the anonymous part.

Next candidates: xatlas's chart growth is the floor here (quadratic in chart size, its output depends on it); our
simplify (13 s; 25+ s on the remesh's 8.3M faces, against CuMesh's 1 s on the GPU) is the largest piece we own.

## Run 9 - 2026-11-11

Question: the export ran entirely on the CPU while the GPU sat idle; how much does a CUDA port of our largest owned
step, `cleanup::simplify`, buy, and can it give the host port's exact result?

Benchmark `simplifying_the_remesh_is_timed` (TRELLIS2_REMESH_REF: the remesh's 8.3M-face shell to 1M), host:
36 rounds (20 at threshold 1e-8, 16 at 1e-7), each 0.1-1.1 s: 13.75 s.

`trellis2::simplify_cuda` + `kernels/cuda/mesh.cu` (the crate's first kernels, fatbin like inference-layout's): the
mesh stays on the device for every round. Per round: vertex -> face lists (count, scan, fill, then each list sorted
to the host's ascending order), edges per vertex (each vertex dedupes and ranks its few higher neighbours, so the
edge list comes out vertex-major and neighbour-ascending like the host's, no global sort), quadrics, costs, the
per-face cheapest edge (64-bit atomicMin of cost bits << 32 | edge), winners, collapses (winners share no face or
vertex, so they apply in parallel), compaction by prefix scans. Built with `--fmad=false` so every product rounds as
on the host.

    simplified on Cpu  in 13.75 s to 465634 vertices / 931728 faces, hash 7c739de25cc0a125
    simplified on Cuda in  0.49 s to 465634 vertices / 931728 faces, hash 7c739de25cc0a125

Bit-identical, 28x, and faster than CuMesh's own 1.0 s. A GPU case pins it on a bumpy grid (vertices and faces equal).
`cleanup::simplify_on(mesh, target, device)` dispatches; `bake` and `Generated::vertex_colored` take the device.

End to end (CLI, robot, 1024-cascade): default 115 s -> 87.0 s (export 33.2 s), `--remesh` 128 s -> 93.0 s (export
38.8 s); same output sizes as before. What is left of the export: xatlas ~28 s (CPU in the reference too), the
remesh's distance queries ~8 s, chart clustering 3.5 s, inpainting 4.7 s.

## Run 10 - 2026-11-11

Question: the same for the distance queries: the remesh's ~8 s is BVH closest-point queries (band levels, grid
corners, project back), and the bake's texel projection another 0.5 s.

`bvh_closest` (mesh.cu): the host traversal (nearer child first on a fixed stack, a box skipped when no nearer than
the best) and Ericson's closest point, `--fmad=false`; the tree uploaded once as flat arrays (`Bvh::flat`,
`mesh_cuda::DeviceBvh`). `bake::Distances` queries on the device when given one; the remesh and the texel projection
(now one batched query) take it. `distances_and_the_remesh_match_the_reference`, now also on the device:

    host:   1000000 distances in 5.39 s; remeshed in 7.93 s
    device: distances in 0.50 s, remeshed in 3.22 s; same distances: true, same shell: true

Bit-identical again; the remesh's remaining 3.2 s is host work (corner keys sorted, value lookups by binary search,
dual contouring). End to end: `--remesh` 93.0 s -> 86.3 s (export 32.2 s); default unchanged at 87.3 s (its
projection was 0.5 s).

## Run 11 - 2026-11-11

Question: is generation (the CUDA half) near optimal? Timed against the reference pipeline on this box
(`time_ref.py`, scratch: PyTorch SDPA attention, no flash-attn installed; robot, 1024-cascade):

    reference, warm: 42.4 s (cond 0.5, structure 6.7, shape cascade 23.0, texture 11.0, decode 1.2); cold 101 s
    ours (synchronized stage timings, temporary): structure 8.8, shape 512 4.5 + 1024 19.9, texture 11.9,
    decoders 3.8, ~2.5 s of loads: 53.7 s

nsys over our run (~50 s of kernels): attention (`flash_attn_ext_f16`) 13.0 s, bf16 GEMMs 14.3 s, and ~16 s of
elementwise passes: `bmul_f32` 3.0, bf16<->f32 casts 4.2, `badd_bf16` 2.4, `ucopy_bf16` 1.5, layernorm 1.3,
`bdiv_f32` 1.3, sums/squares 1.5... The DiT block ran its norms, q/k normalization, rotations, modulation and
gated residuals as ~30 separate passes, f32 round trips included. (The guidance interval was already honoured: steps
outside it run the condition alone.)

`kernels/cuda/dit.cu` + `trellis2::dit_cuda`, used by `Block::forward` for bf16 on CUDA (`forward_inspect`, which the
parity hooks use, keeps the unfused path):

- `ln_mod_bf16`: LayerNorm in f32 (affine for the cross-attention norm), rounded, then `h * (1 + scale) + shift` in
  bf16 as the ops round it.
- `head_norm_rope_bf16`: F.normalize times the gain, rounded, then the pair rotations in f32, rounded, written in
  attention's [batch, heads, tokens, d] layout (no transpose copies; v goes through it as a plain copy).
- `gate_residual_bf16`: x + y * gate with both roundings.

GPU cases compare each to the unfused ops: residuals exact, norms within one bf16 rounding (reduction order). First
end-to-end run produced an empty structure: the guided pass broadcasts one modulation row over its batch of two,
and the kernels read a row per batch; a zero row stride for a one-row modulation fixed it (cases added).

    structure 8.8 -> 5.8 s, shape 512 4.5 -> 2.8 s, shape 1024 19.9 -> 14.8 s, texture 11.9 -> 8.6 s
    generate 53.7 -> 40.6 s (reference warm 42.4 s); end to end 87 -> 76.1 s

Rendered: the same robot and texture quality (bf16 runs differ in detail anyway). Left: the decoders (3.8 s against
the reference's 1.2 s: sparse convolutions) and attention, whose kernel converts bf16 to f16 first.

## Run 12 - 2026-11-11

Question: the decoders took 3.8 s against the reference's 1.13 s (`decode_shape_slat` 0.57 + `decode_tex_slat` 0.56).
`decoding_is_timed` (parity.rs, TRELLIS2_FULL_REF's 1024 latents, both decoders, warm, hashed outputs; dev build):

    baseline: shape 2.00 s, texture 1.96 s, 1733367 cells, hashes 478f9538bb6fae22 547ee5961b606464

nsys: `is_u32_f16` (the im2col `index_select`, one element a thread, 64-bit div/mod per element) 1.34 s of the
decoders' ~2.7 s of kernels; GEMMs ~1.2 s; 2.6 GB of pageable neighbour-table uploads; the rest host gaps (hashing
1.7M cells for each level's table).

- `gather_rows` (`kernels/cuda/sparse.cu`): whole rows in 16-byte vectors, an inactive tap's index read as zeros (no
  padded copy of the features): 3.96 -> 2.67 s, same hashes.
- `child_neighbors`: a subdivided level's table from its parents' (a child's neighbour lies in its parent or one of
  the parent's 26; `slots` maps a parent's child slot to the kept child): no hash, no upload. GPU case against a
  freshly hashed table. 2.67 -> 2.03 s, same hashes.
- Fused passes (`ln_f16` bias + LayerNorm + silu with the tensor kernel's block size and sums, `bias_silu_f16`,
  `bias_residual_f16`; the tensor kernels build with FMA contraction and these without, so those products are
  `fmaf`): 2.03 -> 1.83 s, same hashes.

Left per run: GEMMs 1.22 s at ~72 TFLOPS (the 3090's f16-in/f32-accumulate peak), gather 0.24 s, elementwise 0.23 s.
Active taps per level 0.56/0.73/0.62/0.44/0.42; FLOP-weighted 0.65, so skipping inactive taps (FlexGEMM's masked
implicit GEMM) would save at most ~0.4 s, at the cost of a per-tap GEMM or a custom implicit GEMM and changed
roundings. Not taken.

End to end (robot, 1024-cascade): 76.1 -> 68.7 s (generation ~41.5 -> 37.3 s with the cascade's upsampling decode,
export 31.4 s).

## Run 13 - 2026-11-11

Question: is attention (13 s, `flash_attn_ext_f16`, bf16 converted to f16 first) worth a kernel of our own? nsys over
the current build, all kernels 34.0 s:

    flash_attn_ext_f16 12.4 s, bf16 GEMMs ~15 s, badd_bf16 1.52 s (9000 calls), gate_residual 0.57, head_norm_rope
    0.55, ugelu_bf16 0.48, ln_mod 0.43, convert_to_f16 0.38, ...

The shape-1024 self-attention call (batch 2 x 12 heads x 8259 tokens, d 128) takes 10.9 ms for 838 GFLOP: ~77
TFLOPS, at the card's tensor-core rate. The f16 conversion is 0.38 s. No kernel of our own, then.

The bias adds of the block's six linears were the largest pass left. Each now rides a fused kernel: qkv's and the
cross q's in `head_norm_rope_bf16`, the three output projections' in `gate_residual_bf16`, fc1's in a new
`bias_gelu_bf16` (the tanh GELU, its one contracted product an `fmaf`; the GPU case is exact against the ops). End to
end: 68.7 -> 68.2 s (generation ~36.3 s against the reference's warm 42.4 s), the same mesh (687104 vertices,
958912 faces).

## Run 14 - 2026-11-11

Question: can the unwrap skip xatlas's chart growth by taking our clusters (CuMesh's `compute_charts`) as the charts,
and how different is the result? `ChartOptions::singleChart` (vendored patch, off by default) takes each added mesh
as one chart, still parameterized (LSCM) and, when invalid, recomputed piecewise. `single_chart_unwrap_is_compared`
(parity.rs) on TRELLIS2_CHARTS_REF's 944K-face export mesh; L2 stretch area-weighted and normalized (1 = isometric):

    mode               time     islands  boundary  coverage  L2 stretch (worst 1%)
    default (CuMesh)   14.2 s   83537    1291      0.398     1.032 (1.132)
    single, cone 90    14.6 s   92990    1453      0.378     1.031 (1.146)
    single, cone 60    11.9 s   88382    1404      0.385     1.045 (1.125)
    single, cone 45     9.5 s   87034    1384      0.385     1.023 (1.116)
    single, cone 30     7.8 s   89747    1401      0.377     1.022 (1.095)
    single, cone 20     7.8 s  102688    1500      0.359     1.023 (1.095)

At the clusters' 90 degree cone it is no faster: many clusters do not flatten without overlaps and go piecewise,
whose growth runs the same boundary checks. Tighter cones get there (30 degrees: -45%, slightly less stretch, +7%
islands, +8% seams, -5% coverage).

xatlas's own timers (`XA_PROFILE`, temporary) on the default path, summed over threads: compute charts 114 s (7.1 s
wall); placing seeds 54.4 (boundary checks 37.2), growing 22.6 (22.7), merging 10.0, LSCM 10.4, piecewise 14.4
(14.0). Boundary checks are 65%. Growth re-projects the whole chart on its refit basis for every face, so its grid
has to be rebuilt; the piecewise patch places vertices once, so its grid can grow instead. Two changes, each keeping
the atlas hash (8e229b13923a3026):

- `PatchBoundaryGrid` for the piecewise patch (edges added as they join the boundary, interior ones skipped when
  queried, cells by edge box): piecewise 14.4 -> 2.95 s.
- `UniformGrid2`'s self-test over the pairs sharing a cell (both orders, as before) instead of a traversal and
  sorted gather per edge: seeds' checks 35.7 -> 26.3 s, growth's 20.9 -> 16.4 s.

Compute charts 114 -> 82 s summed (7.13 -> 5.15 s wall). End to end (robot, 1024-cascade): 68.2 -> 64.4 s, export
31.9 -> 27.2 s, the GLB byte-identical. With the grid changes the single-chart lead narrows (cone 30: 8.2 s against
the default's 13-14 s in the same test, which includes our clustering).

## Run 15 - 2026-11-11

Question: what does the single-chart unwrap at a 30 degree cluster cone look like on a real bake, and what does it
save there? A temporary env toggle in `bake` (not committed) switched the robot's 1024-cascade export to it; the
two GLBs rendered with nvdiffrast (`render_pair.py`, scratch: four views each plus a 2x front close-up).

    default:          export 27.2 s, end to end 64.4 s, 687104 vertices, 958912 faces
    single, cone 30:  export 15.7 s, end to end 57.2 s, 658827 vertices, 958912 faces

At 1024 the saving is larger than on the 944K-face test mesh (11.5 s of export), and the GLB has fewer seam vertices
here. Rendered, the two are indistinguishable at full views and in the close-up: no visible seams or blur from the
lower atlas coverage. Not wired in: it gives up matching the reference's atlas, so it would be an option or a
default change to decide on.

## Run 16 - 2026-11-11

Decision (user): the single-chart unwrap at a 30 degree cone becomes the default (`charts::uv_unwrap`;
`uv_unwrap_reference` keeps CuMesh's settings for the parity test against CuMesh's unwrap). `the_bake_matches_the_reference`
(surface color looked up through our atlas, so atlas-independent): mean abs diff 0.54 levels (0.57 with CuMesh's
unwrap), p99 5.7. Robot, 1024-cascade, CLI:

    default          58.6 s (export 15.3 s)
    --remesh         60.6 s (export 17.0 s)
    --vertex-colors  44.2 s (export 0.8 s)

Against Run 8's 114.8 / 127.7 / 62 s.

## Run 17 - 2026-11-11

Question: where do time and memory go now, and does the export's shape-only work fit beside the texture stage?
Temporary `PROFILE` marks (device synchronized) at each stage and export phase, and a sampler (`measure.sh`,
scratch: nvidia-smi used memory and the process's RssAnon every 0.25 s; 1756 MiB were in use by other processes and
are subtracted). Robot, 1024-cascade, sequential:

    condition 0.95, structure 6.35, shape 512 3.06, upsample 0.44, shape 1024 15.20, texture 9.03,
    shape decode 0.94, texture decode 0.93, extract 0.99;
    export: clean 3.61, clusters 2.43, xatlas 1.96, raster + projection 0.56, sample 0.40, inpaint 4.33, glb 0.27
    total 52.5 s; GPU peak 6.46 GB (shape 1024 sampling), host anon peak 2.4 GB (inpainting)

Clean, clusters, unwrap, rasterization and projection need only the shape (~8.6 s, mostly CPU). The pipeline now
decodes the shape and extracts its surface before the texture stage (`generate_with` hands the mesh to a callback;
the noise draws keep their order), and `bake` splits into `prepare` (the shape half) and `finish` (sampling,
inpainting); the job runs `prepare` on a scoped thread while the texture flow and decoder run:

    shape decode 26.89, extract 27.93, [prepare beside texture: clean 32.08, clusters 34.59, unwrap 36.40,
    raster + project 37.25], texture 37.24 (9.3 s, was 9.0), texture decode 38.15, sample 38.43, inpaint 42.49

The GLB is byte-identical to the sequential run. Without the marks:

    default          43.0 s, GPU peak 6.07 GB, host 2.34 GB (was 52.5-58.6 s, 6.46 GB)
    --remesh         44.9 s, GPU 7.05 GB (its distance queries now share the GPU with the texture stage), host 2.66 GB
    --vertex-colors  38.1 s, GPU 6.06 GB, host 1.49 GB

The GPU peak dropped because the shape decoder is freed before the texture flow loads. Left on the critical path
after the texture: inpainting 4.1 s (Telea, four planes side by side, each a serial fast march over ~2M unknown
texels; the color plane's radius-3 window dominates).

## Run 18 - 2026-11-11

Question: inpainting (4.1 s) is the critical path after the texture stage. Can it shrink without changing a value?

First, how far the unknown texels sit from the charts (`coverage.py`, scratch: the GLB's UV triangles rasterized
with nvdiffrast, a Euclidean distance transform): 2048^2 texels, 0.488 covered, 2145817 unknown; within 1 texel
0.261, 2: 0.507, 3: 0.753, 4: 0.893, 8: 0.999, max 13. The gaps are narrow, so filling only a band would save
nothing.

The port computes a pixel's value once, when it enters the band, from pixels that entered before it (or were
known), at their final values and arrival times, which never change after entry. So `TeleaSchedule` splits it: the
sequential march records each entry's pixel, arrival time and gradient; each entry gets a level, one past the deepest
earlier entry within the window plus one pixel; then each level is evaluated in parallel with the same arithmetic.
The inward march does not depend on the radius (it reads only the band, at -0 as the outward march leaves it, and
the region), so `telea_schedules` shares it between the radius-3 and radius-1 schedules, with each radius's outward
march running beside it.

Against cv2 5.0.0's planes in `the_bake_matches_the_reference`: 0 of 12582912 color values and 0 of 4194304 for each
material plane differ. Per plane alone (radius 3): setup 0.22 s, outward march 0.45, inward march 0.8, levels 0.64
(6583 levels for 1848185 entries), values 0.84; color 2.96 -> 2.04 s with the shared, overlapped marches and
fixed-size accumulators.

The marches depend only on the coverage, known once the atlas is rasterized, so `prepare` builds them beside the
texture stage and `finish` only evaluates. Robot, 1024-cascade: export after generation 4.65 -> 2.52 s, total 43.0 ->
39.5 s, the GLB byte-identical; host peak 2.46 GB (the schedules are held through the texture stage). `--remesh`
44.7 s, GPU 6.49 GB: its remesh makes `prepare` outlast the texture stage.

## Run 19 - 2026-11-11

Question: an image-to-3D job on the LAN server (a cut-out of a house, seed 1) failed with
`DriverError(CUDA_ERROR_INVALID_VALUE)` right after the decode stage. Why?

- `CUDA_LAUNCH_BLOCKING=1` on the CLI with the same image and seed: the failing launch is the generic `index_select`
  in the decoder's channel-to-space step. Level sizes printed: the 1024 decode's first level kept 0 of 18171 cells,
  and an empty `index_select` launches an empty grid (`LaunchConfig::for_num_elems(0)`), which CUDA refuses.
- The shape 1024 latents were all NaN (581472 of 581472); the 512 ones were clean. Same with the fused DiT kernels
  turned off, so not those.
- Attention on random inputs at 8192 to 70000 tokens: no NaN, so not length alone.
- A finiteness check after every block of every step: the first NaN is at t = 375 (an unguided step), block 0,
  13 tokens. Stage by stage inside the block: q/k/v finite (|q| <= 90, |v| <= 194), the self-attention output NaN in
  13 query rows, all in head 2.
- The call dumped and checked against f32 attention in PyTorch: those rows' logits are all near 0 (nearly even over
  18171 keys), and the unnormalized sums sum_j exp(s_j - max) v_j reach 472444. fattn's mma kernel keeps that
  accumulator in f16 (`T_C_VKQ = tile<16, 8, half2>` on Turing and later; llama.cpp's choice for speed), 65504 at
  most. Most heads exceed it too in total but split their keys across blocks combined in f32; head 2's partial sums
  overflowed.

Fix (in TRELLIS.2, not the shared kernel): the values go into attention divided by a power of two and the output is
multiplied back. sum_j p_j |v_j| <= sum_j |v_j| per (batch, head, channel), since p_j <= 1, so the largest such sum
bounds the accumulator; the scale keeps it under 32768 (1 when it fits). The fused path computes it with a kernel from
qkv (`value_abs_sums_bf16`, `value_scale_finish`) and folds the division into `head_norm_rope_bf16` and the
multiplication into `gate_residual_bf16`, both exact. A first version bounded by n * max|v| with `max_all` (12.6 ms a
call over the whole V) scaled the robot's every call and cost 12 s; the per-channel bound with the kernel costs
nothing measurable.

- On the dumped call: no NaN; error against f32 2.87e-2 relative L2, identical to the unscaled kernel's on its finite
  rows. That error is fattn's f16 accumulation itself (perfectly even attention over 8192 keys sits ~2% off with or
  without the scale): a follow-up if TRELLIS.2 ever needs closer agreement, an f32 V accumulator in the kernel.
- The robot also gets scaled (2 to 64 on most self-attention calls), so its GLB is no longer byte-identical; the
  reference-parity stages keep their cells, and the latent drifts move both ways within bf16's noise (shape 1024 rms
  0.099 against 0.105, shape 512 0.047 against 0.039, the chaotic structure flow 0.82 against 0.45 with identical
  logits and cells). Robot 1024-cascade 39.0 s; the house, seed 1: 1M faces in 86.8 s.

Also: a CUDA `index_select` of no rows now returns an empty tensor (a GPU case reproduces the failure without the
guard), and the model inventory, whose divide by zero on attention-free GGUFs (Mamba, RWKV) made the Models list fail
on the restarted LAN server, skips the KV profile for a model without attention heads.

## Run 20 - 2026-11-11

Run 19's value scaling is superseded: the DiT's attention now sums the values in f32 on CUDA (fattn's f32-values
variant), which neither overflows nor strays ~3% from the reference on diffuse rows; the scaling is removed. See
`fattn_f32_values_investigation.md` (Runs 1-4): robot 41.5 s, the house 97.6 s, every stage closer to the reference.

## Run 21 - 2026-11-11

Question: an image-to-3D job on the LAN server (a flat illustration, a cut-out, seed 0) sat in its export stage until
the server was killed. Reproduced with the CLI: four threads at ~90% CPU, the GPU idle; perf: xatlas's OpenNL
conjugate gradient (`nlCRSMatrixMult` 59%, `runCreateAndParameterizeChartTask` 24%).

The model drew the flat illustration as a flat card (extent ~0.9 x 0.0002 x 1.0; the PyTorch reference on the same
image: 0.73 x 0.00008 x 1.0, also a card). Its cleaned mesh, 963554 faces, clusters at 30 degrees into 20 clusters,
the largest 963520 faces: one legitimate cluster of nearly the whole sheet (the robot: 9006, the largest 12639).
The single-chart unwrap hands it to xatlas as one chart, whose LSCM is one least-squares system over a million faces.
CuMesh's settings (90 degree clusters, xatlas growing charts) on the same mesh: still running when a 10-minute limit
stopped it, so the reference path stalls the same way.

Fix (the single-chart default only; the reference path stays CuMesh's): `split_large_charts` cuts any cluster over
`MAX_SINGLE_CHART_FACES` (20000) into connected pieces of at most that many faces, grown breadth first over shared
edges. The flat-sheet unit case checks the cap, connectedness and that small charts pass through. The image: 24.5 s
end to end (export 3.5 s), its texture whole; the robot's GLB byte-identical (no cluster over the cap), 41.3 s.

## Run 22 - 2026-11-11

Question: another export stuck on the LAN server (the robot at 512, seed 324980, texture 1024), and its Cancel did
nothing. perf on the server: `compute_charts` (its rounds' closures, the pair sort, `acosf`). A round counter on the
same mesh (968023 faces): 10315 rounds; the first merge 293772, 157240, 102542, ... pairs, then from ~35K charts one
merge a round for ~10000 rounds, each a full pass over the mesh. A pair merges only as each chart's cheapest, so a
large chart takes one of the fragments around it a round. CuMesh runs the same rounds on the GPU.

- Once a round merges fewer than 1 in 50 charts, `merge_incrementally` carries on from that round's graph (pairs,
  shared boundaries, cones, areas) with the same rule, recomputing only the merged charts and their neighbours.
  The mesh: 7.3 s for the whole unwrap at 30 degrees, 10.9 s with CuMesh's 90; `charts_match_the_reference` against
  CuMesh still 67145 vs 67139 charts, seams 566.1 vs 565.6; the robot's GLB byte-identical, 40.4 s. The job end to end:
  20.7 s.
- Cancel never reached the export (it is checked between generation steps). `bake::prepare` now takes a stop check
  (the job's cancel): between its steps, in every clustering round and incremental round, and in xatlas through its
  progress callback, whose false return stops it (`inference_xatlas::STOPPED`; a case checks it gives up).

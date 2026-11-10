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

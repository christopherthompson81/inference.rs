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

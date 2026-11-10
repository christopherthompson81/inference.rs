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

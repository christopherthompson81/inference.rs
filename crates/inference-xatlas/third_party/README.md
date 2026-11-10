# Third-party provenance

| in-tree path | upstream | revision | license | status |
|---|---|---|---|---|
| `xatlas/xatlas.cpp`, `xatlas/xatlas.h`, `xatlas/xatlas_c.h` | [jpcy/xatlas](https://github.com/jpcy/xatlas) `source/xatlas/` | `f700c7790aaa030e794b52ba7791a05c085faf0c` | MIT (`xatlas/LICENSE`; carries thekla_atlas, NVIDIA texture tools and Fast-BVH code, MIT) | modified (below) |

sha256 (upstream): `xatlas.cpp` 0ed0283aad005c94738cb0cc4612dba264379d29dea5b3c9b242f2d4752d5df4, `xatlas.h`
e7675335ad8ab1c1cc9060ad153cf6b8ba2ee914282044eb5f02c49590218fbd, `xatlas_c.h`
3aef0438ca395a7c4c215de9fe40460a4530d19654bd21a3f849c76775fbc972.

This is the copy CuMesh (c4ad612) vendors for TRELLIS.2's UV unwrapping, apart from CuMesh's rename of the namespace to
`cumesh_xatlas`. It is built with `XATLAS_C_API=1` and called through its C API.

Changes, each marked `inference-xatlas:` in the source and leaving the output unchanged (an atlas of TRELLIS.2's
1024 robot mesh hashes the same before and after):

- `UniformGrid2::createGrid` empties its cells by bumping a stamp instead of rewriting every cell offset. The grid
  (up to 512 cells a side) is rebuilt for every face xatlas tries to add to a chart; chart computation on that mesh
  went 33.5 s -> 29.1 s.

The build defines `NDEBUG`, without which xatlas compiles its debug checks (`XA_DEBUG`): 42 s -> 30 s there.

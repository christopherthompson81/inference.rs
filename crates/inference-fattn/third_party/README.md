# Third-party provenance

Kernel sources in this crate that come from other projects. They are adapted and maintained here, so they live in
`kernels/`; this file records where they came from.

| in-tree path | upstream | revision | license | status |
|---|---|---|---|---|
| `kernels/cuda/fattn-common.cuh`, `fattn-mma-f16.cuh`, `fattn-tile.cuh`, `fattn-vec.cuh` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `ggml/src/ggml-cuda/` | `4617ccc1a` | MIT | adapted: Q and the result may be bf16 as well as f32 (runtime flags on every kernel, so no extra instances); with a bf16 result the unnormalized stream-k partial goes to an f32 slot per block; the f16 K/V copies come from the pool instead of space after dst. The vec kernel's quantized-K path (`Q_q8_1`) still reads f32 Q; no quantized-K instance is built |
| `kernels/cuda/fattn.cu*`, `fattn-tile.cu`, `fattn-swizzle.cuh`, `common.cuh`, `mma.cuh`, `convert.cuh`, `vecdotq.cuh`, `cp-async.cuh`, `vendors/cuda.h`, `instances/` | llama.cpp `ggml/src/ggml-cuda/` (`template-instances/` for `instances/`) | `4617ccc1a` | MIT | copied unchanged; `instances/` holds the mma and tile instances and the f16-f16 / bf16-bf16 vector instances |
| `kernels/cuda/ggml/` | llama.cpp `ggml/include/` (`ggml.h`, `ggml-backend.h`, `ggml-alloc.h`, `ggml-cuda.h`, `gguf.h`) and `ggml/src/` (`ggml-impl.h`, `ggml-common.h`) | `4617ccc1a` | MIT | copied unchanged, for the declarations the kernels include |
| `kernels/cuda/ggml_compat.cu` | functions from llama.cpp `ggml/src/ggml.c` and `ggml/src/ggml-cuda/ggml-cuda.cu`, `convert.cu` | `4617ccc1a` | MIT | adapted: the subset fattn links against; a stream-ordered pool on the caller's stream; f16 converters for f32 and bf16 only |
| `kernels/cuda/entry.cu` | (ours) | | | C entry points over raw pointers, wrapping them in ggml tensor descriptors |

# inference-tensor-kernels

The CUDA kernels inference-tensor's backend loads at runtime (affine, binary, cast, conv, fill, indexing, quantized,
reduce, ternary, unary), built ahead of time into compressed SASS fatbins for every listed arch when the `cuda`
feature is on.

Derived from candle-kernels 0.11.0 of [candle](https://github.com/huggingface/candle) at rev `e65eb1de3cfe41008fc3acc99ed3f1c72a8d250f` (MIT OR
Apache-2.0, see `LICENSE-MIT` and `LICENSE-APACHE`), and maintained here; upstream is no longer tracked.

## Changes from candle-kernels

- `src/lib.rs`: the `Module` items are statics instead of consts. Each use of a const embedded its own copy of the
  module's PTX string, so a release `libinference_ffi.so` carried 38 copies of the 11 modules (50 MiB of PTX for 10.6 MiB of
  modules).
- `build.rs`, `src/lib.rs`: the 11 runtime-loaded modules ship as compressed SASS fatbins built from the `.cu` sources
  for every listed arch (`KernelBuilder::build_fatbin`), so loading them needs no driver JIT. The build reads each
  cubin's entries with `cuobjdump -symbols`: `Module::entries` replaces `Module::ptx`, `Module::is_optional` marks an
  entry some arch lacks, and `ARCHS` lists the built archs.
- `src/mmvq_gguf.cu`, `src/mmq_gguf/` and their FFI declarations are removed. inference-quant exports the same 46
  launcher names, so the two archives collided at link time; its 10 MMQ launchers also take an extra `type_dst`
  argument, so calls through candle's declarations were undefined behaviour.
- `src/moe/`, `src/ffi.rs`, the static `libmoe.a` build, the `cutile` feature and the `indexed_moe_forward` kernels in
  `src/quantized.cu` are removed. Only candle-nn's MoE module (removed in the vendored candle-nn) and candle-core's
  `indexed_moe_forward` (removed) called them; inference-nn's `moe_gemm_wmma` shared a C name with candle's.
- `src/quantized.cu`: the `mul_mat_vec_*`, `mul_mat_q*`, `dequantize_mul_mat_vec_*` and `quantize_q8_1` kernels are
  removed with their candle-core launchers; inference-quant runs GGUF matmul. `dequantize_block_*` and `get_rows_*`
  stay.
- `src/fill.cu`: adds `const_set_i16`/`_i32` and `copy2d_i16`/`_i32`, which candle-core's CUDA backend maps but
  upstream never defined, so filling or concatenating an I16/I32 CUDA tensor failed at kernel load.
- `src/reduce.cu`: the `rope`, `rope_i` and `rope_thd` kernels are removed with candle-nn's `rotary_emb`.
- `src/reduce.cu`: the `rmsnorm` kernels are removed with candle-nn's CUDA `rms_norm`.
- `src/sort.cu` is removed with candle-core's CUDA argsort; its kernel lives on as inference-nn's `argsort_rows`.

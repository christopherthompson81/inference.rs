# inference-tensor

The tensor layer of inference.rs: `Tensor`, devices and their CPU/CUDA/Metal backends, GGUF and quantized storage,
safetensors/pickle/npy loading (from candle-core), and neural-net building blocks under `nn` (from candle-nn; its
`loss` and `optim` moved to inference-nn). CUDA kernels for the backend live in `inference-tensor-kernels`.

Derived from [candle](https://github.com/huggingface/candle) 0.11.0 at rev `e65eb1de3cfe41008fc3acc99ed3f1c72a8d250f`
(MIT OR Apache-2.0, see `LICENSE-MIT` and `LICENSE-APACHE`), and maintained here; upstream is no longer tracked.

## Changes from candle-core

- `Cargo.toml`: the workspace-inherited fields and dependencies are written out with the values of candle's root
  manifest at that rev, `inference-tensor-kernels` replaces `candle-kernels`, candle-nn's dependencies are merged in,
  and the bench and example targets are dropped.
- `src/cuda_backend/device.rs`: `get_or_load_func` loads a module's SASS fatbin (`Module::image`) with
  `Ptx::from_binary` instead of its PTX, so loading needs no driver JIT.
- `Cargo.toml`: cudarc uses `dynamic-loading` instead of `dynamic-linking`, so the driver and every CUDA library are
  `dlopen`ed on first use rather than linked. cuBLAS loads with each `CudaDevice`; cuRAND, NVRTC, cuDNN and NCCL
  only when a code path calls them, so a bundle can leave out the ones it never uses.
- `src/cuda_backend/device.rs`: the cuRAND generator is created on first use (from the stored seed, so `set_seed`
  keeps its sequence). A missing driver, cuBLAS or cuRAND is an error rather than cudarc's panic.
- `src/cuda_backend/device.rs`: `get_or_load_custom_image` loads a custom module from a cubin or fatbin, beside
  `get_or_load_custom_func`'s PTX.
- `src/quantized/`: `fast_mmq.rs` and `fast_mmvq.rs` are removed with their call in `QCudaStorage::fwd`, and so are
  the matvec, MMQ and dmmv launchers with `quantize_q8_1` and `set_force_dmmv`. GGUF matmul on CUDA goes through
  inference-quant (`gguf::qmatmul_forward`); a `QMatMul` call that still lands here dequantizes and runs cuBLAS.
- `src/quantized/`: `QTensor::indexed_moe_forward` and `QMatMul::indexed_moe_forward` are removed with their CUDA
  implementation; inference-quant's GGUF MoE paths replace them.
- Removed as unused by inference.rs: `src/streaming.rs`, `src/test_utils.rs`, `src/quantized/tokenizer.rs` (and the
  `tokenizers` dependency) and `src/cuda_backend/cutile.rs` (and the `cutile` feature and dependency).
- `src/cpu/erf.rs`: drops `use std::f64;`, which made `f64::INFINITY` resolve to the module constants that Rust 1.99
  deprecates (a path dependency's lints are not capped like a git dependency's).
- `src/cuda_backend/mod.rs`: kernel names spell F8E4M3 `f8_e4m3`, as candle-kernels defines them (upstream asked for
  `f8e4m3`, so every F8E4M3 cast, binary and comparison kernel failed to load), and a strided F8E4M3 copy runs
  `ucopy_u8` (`ucopy_f8_e4m3` is sm89+ only, the casts sm80+).
- `src/sort.rs`: the CUDA arm of `arg_sort_last_dim`/`sort_last_dim` is removed with candle-kernels' `sort.cu`;
  inference-nn's `ArgSortOp` sorts on CUDA.

## Changes from candle-nn

Paths are candle-nn's, now under `src/nn/`.

- `Cargo.toml`: merged into inference-tensor's (the bench target was dropped before that).
- `src/moe.rs` and `src/moe/` are removed with `pub mod moe`, the `cutile` feature and the `candle-kernels`
  dependency they needed. The MoE kernels they called are gone from candle-kernels; inference.rs runs MoE through
  inference-nn and inference-quant.
- Removed as unused by inference.rs: the `attention` and `cpu_flash_attention` modules (inference-nn has its own CPU
  flash attention), `kv_cache`, `rnn`, `encoding`, `sequential`, `func` and `sampling` with their re-exports;
  `src/rotary_emb.rs` (inference-quant's `rotary` is the one RoPE); and the npz, pth, routing, sharded and
  renaming backends in `src/var_builder.rs` (inference-quant has its own sharded loader); `rms_norm_slow` and
  `layer_norm_slow` in `src/ops.rs`; the CUDA arm of `ops::rms_norm`, so `ops::rms_norm` and `RmsNorm` fail on
  contiguous CUDA input (inference-nn's `ops::rms_norm` is the CUDA one, f32/f16/bf16; candle's f64 CUDA kernel is
  gone).

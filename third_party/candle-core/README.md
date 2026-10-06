> **Vendored copy for inference.rs.** This is candle-core 0.11.0 from [candle](https://github.com/huggingface/candle) at
> rev `e65eb1de3cfe41008fc3acc99ed3f1c72a8d250f` (MIT OR Apache-2.0, see `LICENSE-MIT` and `LICENSE-APACHE`), patched in
> through `[patch."https://github.com/huggingface/candle.git"]` in the workspace `Cargo.toml`. Only `src/`, `README.md`
> and the license are vendored; the tests, benches and examples are not. Changes:
> - `Cargo.toml`: the workspace-inherited fields and dependencies are written out with the values of candle's root
>   manifest at that rev, `candle-kernels` points at `../candle-kernels`, and the bench and example targets are dropped.
> - `src/cuda_backend/device.rs`: `get_or_load_func` loads a module's SASS fatbin (`Module::image`) with
>   `Ptx::from_binary` instead of its PTX, so loading needs no driver JIT.
> - `Cargo.toml`: cudarc uses `dynamic-loading` instead of `dynamic-linking`, so the driver and every CUDA library are
>   `dlopen`ed on first use rather than linked. cuBLAS loads with each `CudaDevice`; cuRAND, NVRTC, cuDNN and NCCL
>   only when a code path calls them, so a bundle can leave out the ones it never uses.
> - `src/cuda_backend/device.rs`: the cuRAND generator is created on first use (from the stored seed, so `set_seed`
>   keeps its sequence). A missing driver, cuBLAS or cuRAND is an error rather than cudarc's panic.
> - `src/cuda_backend/device.rs`: `get_or_load_custom_image` loads a custom module from a cubin or fatbin, beside
>   `get_or_load_custom_func`'s PTX.
> - `src/quantized/`: `fast_mmq.rs` and `fast_mmvq.rs` are removed with their call in `QCudaStorage::fwd`, and so are
>   the matvec, MMQ and dmmv launchers with `quantize_q8_1` and `set_force_dmmv`. GGUF matmul on CUDA goes through
>   inference-quant (`gguf::qmatmul_forward`); a `QMatMul` call that still lands here dequantizes and runs cuBLAS.
> - `src/quantized/`: `QTensor::indexed_moe_forward` and `QMatMul::indexed_moe_forward` are removed with their CUDA
>   implementation; inference-quant's GGUF MoE paths replace them.
> - Removed as unused by inference.rs: `src/streaming.rs`, `src/test_utils.rs`, `src/quantized/tokenizer.rs` (and the
>   `tokenizers` dependency) and `src/cuda_backend/cutile.rs` (and the `cutile` feature and dependency).
> - `src/cpu/erf.rs`: drops `use std::f64;`, which made `f64::INFINITY` resolve to the module constants that Rust 1.99
>   deprecates (a path dependency's lints are not capped like a git dependency's).
> - `src/sort.rs`: the CUDA arm of `arg_sort_last_dim`/`sort_last_dim` is removed with candle-kernels' `sort.cu`;
>   inference-nn's `ArgSortOp` sorts on CUDA.
>
> Everything else is upstream. To re-sync, copy `candle-core/src` from the new rev and reapply the changes above.

# candle
Minimalist ML framework for Rust

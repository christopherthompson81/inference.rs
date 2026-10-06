> **Vendored copy for inference.rs.** This is candle-nn 0.11.0 from [candle](https://github.com/huggingface/candle) at
> rev `e65eb1de3cfe41008fc3acc99ed3f1c72a8d250f` (MIT OR Apache-2.0, see `LICENSE-MIT` and `LICENSE-APACHE`), patched in
> through `[patch."https://github.com/huggingface/candle.git"]` in the workspace `Cargo.toml`. Only `src/`, `README.md`
> and the license are vendored; the tests, benches and examples are not. Changes:
> - `Cargo.toml`: the workspace-inherited fields and dependencies are written out with the values of candle's root
>   manifest at that rev, `candle` points at `../candle-core`, and the bench target is dropped.
> - `src/moe.rs` and `src/moe/` are removed with `pub mod moe`, the `cutile` feature and the `candle-kernels`
>   dependency they needed. The MoE kernels they called are gone from candle-kernels; inference.rs runs MoE through
>   inference-nn and inference-quant.
> - Removed as unused by inference.rs: the `attention` and `cpu_flash_attention` modules (inference-nn has its own CPU
>   flash attention), `kv_cache`, `rnn`, `encoding`, `sequential`, `func` and `sampling` with their re-exports;
>   `src/rotary_emb.rs` (inference-quant's `rotary` is the one RoPE); and the npz, pth, routing, sharded and
>   renaming backends in `src/var_builder.rs` (inference-quant has its own sharded loader); `rms_norm_slow` and
>   `layer_norm_slow` in `src/ops.rs`; the CUDA arm of `ops::rms_norm`, so `ops::rms_norm` and `RmsNorm` fail on contiguous CUDA input (inference-nn's
  `ops::rms_norm` is the CUDA one, f32/f16/bf16; candle's f64 CUDA kernel is gone).
>
> Everything else is upstream. To re-sync, copy `candle-nn/src` from the new rev and reapply the changes above.

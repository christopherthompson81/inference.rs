> **Vendored copy for inference.rs.** This is candle-nn 0.11.0 from [candle](https://github.com/huggingface/candle) at
> rev `e65eb1de3cfe41008fc3acc99ed3f1c72a8d250f` (MIT OR Apache-2.0, see `LICENSE-MIT` and `LICENSE-APACHE`), patched in
> through `[patch."https://github.com/huggingface/candle.git"]` in the workspace `Cargo.toml`. Only `src/`, `README.md`
> and the license are vendored; the tests, benches and examples are not. Changes:
> - `Cargo.toml`: the workspace-inherited fields and dependencies are written out with the values of candle's root
>   manifest at that rev, `candle` points at `../candle-core`, and the bench target is dropped.
> - `src/moe.rs` and `src/moe/` are removed with `pub mod moe`, the `cutile` feature and the `candle-kernels`
>   dependency they needed. The MoE kernels they called are gone from candle-kernels; inference.rs runs MoE through
>   inference-nn and inference-quant.
> - `src/attention/cpu_flash/standard.rs`: drops `use std::f32;`, which made `f32::NEG_INFINITY` resolve to the module
>   constant that Rust 1.99 deprecates.
>
> Everything else is upstream. To re-sync, copy `candle-nn/src` from the new rev and reapply the changes above.

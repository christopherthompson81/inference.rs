> **Vendored copy for inference.rs.** This is candle-core 0.11.0 from [candle](https://github.com/huggingface/candle) at
> rev `e65eb1de3cfe41008fc3acc99ed3f1c72a8d250f` (MIT OR Apache-2.0, see `LICENSE-MIT` and `LICENSE-APACHE`), patched in
> through `[patch."https://github.com/huggingface/candle.git"]` in the workspace `Cargo.toml`. Only `src/`, `README.md`
> and the license are vendored; the tests, benches and examples are not. Changes:
> - `Cargo.toml`: the workspace-inherited fields and dependencies are written out with the values of candle's root
>   manifest at that rev, `candle-kernels` points at `../candle-kernels`, and the bench and example targets are dropped.
> - `src/cuda_backend/device.rs`: `get_or_load_func` loads a module's SASS fatbin (`Module::image`) with
>   `Ptx::from_binary` instead of its PTX, so loading needs no driver JIT.
>
> Everything else is upstream. To re-sync, copy `candle-core/src` from the new rev and reapply the changes above.

# candle
Minimalist ML framework for Rust

> **Vendored copy for inference.rs.** This is candle-kernels 0.11.0 from [candle](https://github.com/huggingface/candle)
> at rev `e65eb1de3cfe41008fc3acc99ed3f1c72a8d250f` (MIT OR Apache-2.0, see `LICENSE-MIT` and `LICENSE-APACHE`), patched
> in through `[patch."https://github.com/huggingface/candle.git"]` in the workspace `Cargo.toml`. Changes:
> - `src/lib.rs`: the `Module` items are statics instead of consts. Each use of a const embedded its own copy of the
>   module's PTX string, so a release `libinference_ffi.so` carried 38 copies of the 11 modules (50 MiB of PTX for 10.6 MiB of
>   modules).
> - `build.rs`: the statically linked MoE/mmq/mmvq kernels compress their fatbin (`KernelBuilder::compress_fatbin`).
> - `build.rs`, `src/lib.rs`: the 11 runtime-loaded modules ship as compressed SASS fatbins built from their PTX
>   (`KernelBuilder::build_fatbin`), so loading them needs no driver JIT. The PTX stays a build intermediate, and each
>   `Module` carries the entry names parsed from it (`Module::entries`) in place of `Module::ptx`.
>
> Everything else is upstream. To re-sync, copy `candle-kernels/` from the new rev and reapply the changes above.

# candle-kernels

This crate contains CUDA kernels used from candle. Some of these implementations
come from the [dfdx crate](https://github.com/coreylowman/dfdx).

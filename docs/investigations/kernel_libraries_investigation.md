# Kernel libraries investigation (#425)

Downloadable CUDA kernel libraries, per GPU architecture and per quantization type. The size measurements behind the plan are in `build_artifacts_investigation.md`, Runs 11-13.

## Run 1 - 2026-10-11

Question: what does phase 1 (runtime-loaded kernel libraries) have to convert, and how are kernels linked today?

- **Two mechanisms:**
  - inference-tensor's kernels are SASS images embedded as bytes and loaded through the driver API (`load_module`, `get_or_load_func`). They are already runtime-loaded.
  - The C++ kernel crates (quant, fattn, nn's CUDA ops, paged attention) carry host-side launch code. Rust calls them through `extern "C"` blocks.
- **Linking:**
  - `inference-kernel-build::build_and_link` builds a shared library in dev builds on Linux (`build_shared_lib`: `-shared`, `--cudart shared`, SONAME = absolute path, no rpath) and a static archive elsewhere and in release.
  - CUTLASS extras link as their own static libraries: `inferencenvfp4`, `inferencefa3paged`, `flashattentionv3`.
- **FFI surface:** 23 files with `extern "C"` blocks, ~370 functions.
  - The largest: `inference-nn/src/cuda/ffi.rs` 96, `inference-quant/src/gguf/ffi.rs` 91, `quant/utils/ffi.rs` 45, `blockwise_fp8/ffi.rs` 33, `paged-attn/cuda/ffi.rs` 18.
  - The rest have 1-13 each.
- **Design:** a macro takes an `extern "C"` block as written and generates a function table, a loader (dlopen on first use, every symbol resolved at once), and same-named wrappers, so call sites do not change.
  - A missing library or symbol is an error naming the library.
  - The library path comes from an environment override, else the dev build's absolute path, else beside the host library.
- **Next:** prototype on paged attention (18 functions). Its tests pass, and launch overhead is measured against the linked build.

## Run 2 - 2026-10-11

Question: does the loader work on a real kernel crate?

- **Implementation:**
  - `inference_tensor::kernel_library!` takes an `extern "C"` block. It generates a function table and a `OnceLock` loader that resolves every symbol at once, same-named `pub unsafe fn` wrappers, and `ensure_loaded()`.
  - Library lookup order: `INFERENCE_KERNEL_DIR`, then beside the binary or library (`dladdr`), then its `kernels/`, then the build's absolute path.
  - Targets without dlopen (Windows) get the plain block, linked as before.
  - `KernelBuilder::build_loadable_lib` builds the shared library (the dev `build_shared_lib` path, shared cudart) for every profile on Unix. It emits `INFERENCE_KERNEL_LIB_<NAME>` instead of link directives.
- **Paged attention** (18 functions) is converted. Its FA3 FP8 functions (sm90, own CUTLASS archive) stay a static block.
- **Result:**
  - `cargo nextest run -p inference-paged-attn --features cuda`: 13 passed.
  - `ldd` on the test binary shows no `libinferencepagedattention`.
  - `LD_DEBUG=files` shows `libinferencepagedattention.so` "dynamically loaded" by the test binary, and that library's `libcudart.so.12` resolved as its own dependency.
  - Loader unit tests cover the missing-library error (it names the file and `INFERENCE_KERNEL_DIR`) and the lookup order.
- **Overhead:** each call costs one initialized `OnceLock` read and one indirect call, nanoseconds against a microsecond-scale launch. It gets measured end to end once the quant and nn kernels go through the loader.
- **Open for phase 1:**
  - Packaging must ship the libraries: the wheel builder and `--size` need to find the build's libraries, and the C ABI library no longer embeds them.
  - The CUTLASS archives (`inferencenvfp4`, `inferencefa3paged`, `flashattentionv3`) still link statically.

## Run 3 - 2026-10-11

Question: does every kernel crate run through the loader?

- **Converted:**
  - inference-fattn (2 functions), inference-nn (`cuda/ffi.rs`, `dflash_context.rs`), and inference-quant (17 FFI files).
  - Static blocks kept for the separate CUTLASS archives: FA3 FP8 (paged-attn), FlashInfer GDN sm90 (nn), DeepGEMM (quant). `nvfp4/cutlass.rs` is left as is.
- **Macro-generated declarations:** `gguf/ffi.rs` (8 kinds, 305 launchers), `afq/ffi.rs` (9 kinds, 189, built with `paste!`) and `hqq/ffi.rs` (one kind per submodule).
  - Each `declare_<kind>!` becomes `<kind>_kernels!`, emitting one `kernel_library!` call (wrapped in `paste::paste!` where the names are pasted) in a private module that is re-exported. HQQ's calls stay in place, per module.
  - Done by scratch scripts (`convert_ffi.py`, `fold_decl_macros.py`); the output is plain source.
- **Macro fixes found on the way:**
  - The wrappers take any visibility (`pub(crate)`, private).
  - They have the C ABI (`unsafe extern "C" fn`) because some call sites pass them as `extern "C"` function pointers. Without `#[no_mangle]` they do not collide with the library's symbols.
  - `non_snake_case` is allowed for C names like `launch_mmq_quantize_q8_1_D4`.
- **Dead ends:**
  - A first pass skipped `afq/ffi.rs` and `hqq/ffi.rs`, counting 0 functions because their declarations come from macros; the test link failed with 60 undefined symbols.
  - The converter kept `fused_split_glu_quantize_bf16` static because its `cfg(any(...))` mentions the DeepGEMM cfg by substring; it lives in the main library.
- **Result:** `local_ci.sh --cuda`: 3027 passed, 39 skipped. Every CUDA test now reaches its kernels through `dlopen`.
- **Found:** `install.sh` builds with `cargo install`, which deletes its temporary target dir, so a library built there would vanish with it. Release-type builds therefore need the libraries in a persistent location.

## Run 4 - 2026-10-11

Questions: does a packaged build find its kernels, does a missing one fail clearly, and what do size and speed do?

- **Persistent libraries:**
  - Release-type builds (`PROFILE != debug`) put their kernel libraries in `$INFERENCE_KERNEL_BUILD_CACHE`, else `~/.cache/inference-rs/kernels/built/<name>-<hash>/`, so `cargo install`'s deleted target dir cannot take them.
  - Every build writes `target/<profile>/cuda-kernels/<name>.lib` naming the library it made.
  - Release builds link the libraries with `-s`.
- **Model load:**
  - `inference_nn::cuda::ensure_kernels_loaded()` opens all four.
  - Core calls it (`loading::ensure_cuda_kernels`) from the text/multimodal, speech and diffusion load paths whenever a device is CUDA.
- **Packaging:**
  - `build_wheels.py` copies the pointed-to libraries into `inference_rs/_lib/kernels/`, and the package data takes `_lib/kernels/*`.
  - CUDA is now detected from the kernel libraries too: the host library no longer names `cudart`, and the first wheel came out tagged CPU (`0.9.3`, manylinux). Fixed, it is `0.9.3+cu12.sm86`, `linux_x86_64`.
  - `bundle_size.py` sums sections over the host and kernel libraries and records each file.
- **Sizes** (bundle, sm_86, CUDA 12.8):
  - Before the split: one 108.40 MiB library.
  - Now: host 64.47 MiB, quant 26.49 MiB, fattn 17.00 MiB, nn 1.40 MiB, paged ~0.2 MiB; 109.55 MiB in all.
  - Unstripped it was 111.14 MiB; `-s` saved 1.6 MiB.
  - The remaining +1.15 MiB (1%) is the cost of separate libraries: `.text` +0.71 MiB (helpers no longer deduplicated by the final link) plus dynamic symbol tables.
- **Wheel end to end:** `build_wheels.py --library target/bundle/libinference_ffi.so`, unzipped, with `~/.cache/inference-rs/kernels/built` renamed away.
  - Python `JsonEngine` on the tiny checkpoint, `device: cuda:0`, one chat request: ok.
  - `LD_DEBUG=files` shows all four libraries "dynamically loaded" from the wheel's `_lib/kernels/`.
- **Missing libraries:** with `_lib/kernels/` removed as well, `inference_engine_load` fails with "CUDA kernel library `libinferencecuda.so` is not installed (looked in ...)", naming every location. No crash.
- **Speed:** `inference bench -f Qwen3-1.7B-UD-Q4_K_XL.gguf --prompt-len 512 --gen-len 128 --iterations 3`, dev CUDA build, two runs each:

| build | prefill tok/s | decode tok/s |
|---|---|---|
| master | 18293 +- 38, 16594 +- 1110 (noisy) | 385.6, 383.5 |
| runtime-loaded | 18414 +- 21, 18318 +- 42 | 385.5, 385.1 |

  - No measurable overhead.
- **Side finding (pre-existing, not changed here):** the shipped host library already embeds 1,084 home-directory paths (`~/.cargo/registry`, `~/.rustup`) from panic locations, so published wheels carry the builder's username. A `--remap-path-prefix` in the wheel build would remove them.

## Run 5 - 2026-10-11

Question: what did the phase 1 review find, and do the fixes hold?

Findings and fixes:
- **Relative library paths:** for a binary found on PATH, `dladdr` reports `argv[0]`, so "beside the binary" became the working directory, and a planted `libinferencequant.so` there would have loaded. A relative path now falls back to `current_exe()`, and a test pins that only absolute paths are searched (unless `INFERENCE_KERNEL_DIR` is relative).
- **`Library::symbol` soundness:** it took a `&str` and passed it to `dlsym` unchecked; it now takes `&CStr`.
- **No ABI check:** a library found anywhere but the build's own path was trusted by symbol name, so a changed signature was silent UB. Each library now links a one-line C object exporting `inference_kernel_build_id`, a hash of its compiled objects' contents, written beside it as `lib<name>.so.id`.
  - The build passes it to the crate as `INFERENCE_KERNEL_ID_<NAME>`, and `open` refuses a library whose id differs.
  - Dev and bundle builds of the same kernels report the same id (`183e3aac4d2fe5a8`), since the objects are identical.
  - A copy with one id character changed is refused at engine load: "CUDA kernel library `.../libinferencecuda.so` is from another build (id 183e3aac4d2fe5a0, this binary expects 183e3aac4d2fe5a8)".
- **Stale pointer files:**
  - The CPU wheel test picked up the CUDA debug build's pointer files and came out tagged CUDA (`test_wheel` failed on the tag).
  - `build_wheels.py` and `bundle_size.py` now take only the pointed-to libraries whose path the host library itself records.
  - Pointer files are written atomically.
- **The up-front check missed loaders:** the transcription loaders (3) and GGML now call `ensure_cuda_kernels` too.
  - Each library's other tables still load lazily, but they open the same file, so once the id check passes their symbols exist.
- **Build-script tracking:** `rerun-if-changed` on the built library (a deleted one is rebuilt), and `rerun-if-env-changed` for `HOME`/`XDG_CACHE_HOME`.
- **Dead code and leftovers:**
  - Removed `build_and_link` and `build_shared_lib`, which nothing calls now.
  - Removed a redundant `strip` hash input, a stale comment, a misplaced doc line, and an empty AFQ extern block.
- **Open:** the build cache under `~/.cache/inference-rs/kernels/built` is never pruned. A later phase (the download cache) should own its cleanup.

## Run 6 - 2026-10-11 14:20

Question (phase 2): how much of `libinferencequant.so` is per-GGUF-type code, and what does splitting it into one library per type leave in the base?

Before the split (dev build, sm_86 only), objects by size:
- `mmvq_gguf.o` 2.45 MiB, `mmvq_iqk.o` 0.98 MiB, `mmvq_kt.o` 0.50 MiB: every type's matvecs in three files.
- 33 `mmq_instance_<type>.o` at 0.54-0.78 MiB each.
- Everything else (Marlin, ops, LoRA, GEMV, AFQ, indexed MoE, ...) about 4 MiB.
- The library: 28.9 MiB, 3,730 exported symbols.

Change:
- The three MMVQ sources became headers (`mmvq_gguf.cuh`, `mmvq_iqk.cuh`, `mmvq_kt.cuh`) plus one `mmvq_instance_<type>.cu` per type, holding its instantiation lines moved verbatim (34 types: IQ1_M has a matvec but no MMQ). The Q8_1 quantize kernels moved to `mmvq_quantize.cu`, which stays in the base.
- `KernelBuilder::split_lib(name, pattern)`: one compile pass over every source, then the objects matching a pattern link into their own `lib<name>.so` with its own build id and pointer file. The quant build adds `inferencequant_<type>` for each `*_instance_<type>.cu`, so a type's MMQ, MMVQ and (for IQ types) dequantize kernels share one library.
- Windows still links one static archive of everything.
- Rust: `gguf_type_library!` declares each type's functions in a `kernel_library!` of its own. `GgufMatMul::from_parts`/`from_qtensor` (now fallible) and `RawGgufTensor::to_device` open the type's library when a layer lands on a CUDA device, so a missing type fails at model load.
- Renamed `inferencecuda` to `inferencenn`: it holds only inference-nn's ops.

After (same dev build):
- Base `libinferencequant.so`: 4.03 MiB.
- Per type: 0.60-1.02 MiB (IQ1_M 0.12 MiB). Standard K-quants are 0.87-1.02 MiB, since they also carry the fused gate/up and Q/K/V matvecs.
- Total 29.1 MiB, against 28.9 MiB before: each library carries its own copy of the shared tables and headers.
- A Q4_K_M model (Q4_K, Q6_K) needs about 5.8 MiB of quant kernels instead of 28.9 MiB.

Not split: the indexed-MoE and MoE GEMV kernels (per standard type, ~0.4 MiB in all) and LoRA stay in the base.

## Run 7 - 2026-10-11 14:50

Question: do the split libraries cost speed, fail clearly when one is missing, and load only what a model needs?

- **Full local CI** (`--lint --tests --cuda --slim --bindings --docs --size-update --sweep`): 3,027 tests passed, 40 skipped.
- **Bundle (`--size-update`):**
  - `libinferencequant.so` 27.77 MB to 3.99 MB.
  - The 34 type libraries are 0.12-1.04 MB each.
  - Bundle total 114.98 MB to 116.56 MB (+1.4%): each library has its own copy of the shared tables, plus its own dynamic symbol table and cudart registration.
- **Speed:** `inference bench -f Qwen3-1.7B-UD-Q4_K_XL.gguf --prompt-len 512 --gen-len 128 --iterations 3`, dev CUDA builds of master and this change, alternated three times:

| build | prefill tok/s | decode tok/s |
|---|---|---|
| master | 15954, 16026, 15965 | 349.3, 351.1, 350.2 |
| split | 16087, 16877, 16899 | 352.9, 365.7, 360.2 |

  - No regression.
  - Both builds are below Run 4's 18.3k/385 tok/s, so the machine is slower today (a browser holds the GPU at ~29% utilization).
- **Missing type library:** with `libinferencequant_q6_k.so` moved aside, the bench fails at model load with exit 1: "CUDA kernel library `libinferencequant_q6_k.so` is not installed (looked in ...); set INFERENCE_KERNEL_DIR to its directory".
- **What a model loads:** `LD_DEBUG=files` on that model shows the base plus `libinferencequant_{iq4_xs,q4_k,q5_k,q6_k}.so` and no other type. These are the four types the UD-Q4_K_XL file holds: 7.3 MiB of quant kernels instead of 28.9 MiB.

# Build artifact size investigation

Motivation: running the local checks had long single-core stretches, and `target/debug` grew to 105 GB and filled
the disk twice. A workspace of this size should not need tens of GB of artifacts to test.

Machine: i7-10700K (8C/16T), RTX sm_86, rustc 1.98.1, CUDA 12.8, ccache.

## Run 2 - 2026-09-25 11:40

Question: why did `target/debug` reach 105 GB?

Finding:

- `incremental/` was 43 GB, `examples/` 36 GB and `deps/` 24 GB.
- `cargo test --workspace` links every example of every package (58 in the SDK crate alone), once per feature
  combination, and it refilled `examples/` to 60 GB in one run.
- A linked test binary was 1.35 GB, of which `.text` was 66 MB and `.rodata` 30 MB. The rest was DWARF
  (`.debug_info` 367 MB, `.debug_loc` 316 MB, `.debug_str` 215 MB, ...).
- The cause is `[profile.dev] debug = true` on an `opt-level = 3` profile. Each statically linked binary re-embeds
  the full debug info of inference-core and all its dependencies.

## Run 3 - 2026-09-25 12:05

Question: which debug-info setting gives backtraces without the bulk?

Command: `cargo test -p inference-core --lib --no-run` from empty, separate target dirs, via
`CARGO_PROFILE_DEV_DEBUG` / `CARGO_PROFILE_DEV_SPLIT_DEBUGINFO`.

Finding:

| setting | build | test binary | target dir |
|---|---|---|---|
| `debug = true` (was) | 301 s | 1053 MB | 7.4 GB |
| `debug = "line-tables-only"` | 244 s | 352 MB | 4.3 GB |
| `debug = true`, `split-debuginfo = "unpacked"` | 310 s | 261 MB | 5.9 GB |
| line tables + `split-debuginfo = "unpacked"` | 257 s | 213 MB | 4.1 GB |

With line tables plus split debug info, the binary is `.text` 50 MB + `.debug_line` 49 MB + `.debug_ranges`
48 MB + `.debug_addr` 12 MB. The line data is what file:line backtraces need, so this is the floor for an
opt-level 3 test binary of this crate without dropping backtraces.

Change: `[profile.dev] debug = "line-tables-only"`, `split-debuginfo = "unpacked"`. A new `[profile.debugging]`
(dev + `debug = true`) keeps full DWARF one flag away for debugger sessions.

The 58 SDK examples moved from `inference/examples/` into a top-level `examples/rust` package
(`inference-examples`), which is not a default member. Clippy compile-checks all of them, and the tests mode links a
three-example smoke set. That smoke set is built under `--workspace`, so it keeps the workspace's feature unification
and reuses the test build's dependencies.

## Run 4 - 2026-09-25 15:30

Question: is the incremental cache worth its size? `target/debug/incremental` had reached 29 GB.

Method: warm the `inference-core` test build, append one line to `ops/topk.rs` twice, and time each rebuild with
`CARGO_INCREMENTAL=1` and then `=0`.

Finding: 27.4 s / 30.0 s with incremental, and 124.1 s / 124.7 s without. Incremental is a 4x win on the edit loop
and stays on.

The 29 GB was mostly history. inference_core alone had more than ten incremental directories (0.6-2.7 GB each, up to
four sessions each), one per build variant and profile setting tried that day. Cargo never garbage-collects the
directories of units it no longer builds. Run 5 measures the steady state.

## Run 5 - 2026-09-25 16:30

Question: what does a normal run of the local checks leave behind once history is gone?

Command: `rm -rf target/debug/incremental`, then `scripts/local_ci.sh --lint --tests --cuda` (760 s, both suites green)
and a warm rerun (136 s). A cold `deps/` would have cost a full rebuild, so live artifacts were separated from stale
ones afterwards. Each mode's build was replayed with `--message-format=json`, and a file counts as live if its hash
appears in the artifacts cargo reports.

Finding: `target/debug` is 30 GB, of which ~4 GB is stale `deps/` from earlier variants. The live set:

| part | all modes | CPU only (`--lint --tests`) |
|---|---|---|
| `deps/` + `examples/` | 11.8 GB | 5.6 GB |
| `incremental/` | 10.9 GB | ~5 GB |
| `build/` (kernel objects) | 2.9 GB | small |

- `deps/` is 5.8 GB of 59 linked test binaries, 2.2 GB rlibs, 2.1 GB split `.dwo`, 1.1 GB rmeta.
- A CPU test binary of inference-core is 196 MB (`.text` 50 MB, line tables and ranges ~110 MB). A CUDA one is
  423 MB, of which 177 MB is `.nv_fatbin`: 106 sm_86 SASS objects and no PTX, re-embedded in every CUDA test binary.
- Consolidating the integration tests (8 files in inference-quant, 2 in inference-paged-attn) would save only ~0.5 GB.
  The large binaries are per-crate lib test harnesses and the CLI binary.
- The flash-attn and paged-attn build scripts run once under clippy and once under the test build with `--features
  cuda`. Each gets its own `build/` output (475 MB and 242 MB). ccache makes the second compile cheap in time but not
  in disk.
- `incremental/` holds ~1.4 GB per inference_core variant (CPU/CUDA x lib/test). Run 4 showed it pays for itself.

Implication: 105 GB came from examples, full DWARF and history, and all three are addressed or understood. What is
left scales with the number of CUDA-linked binaries and feature variants, and cargo never deletes the stale ones.

## Run 6 - 2026-09-25 16:45

Question: can the CUDA kernels live once on disk instead of once per cargo variant and once per test binary?

Change: cudaforge gets `build_and_link(name, archive)`. On dev-profile Linux builds it links the kernel set into
`target/debug/cuda-kernels/<name>-<hash of compile inputs>/lib<name>.so`, with the absolute path as SONAME so no
rpath or `LD_LIBRARY_PATH` is needed. Release builds and Windows keep the static archive. It is used for the four
kernel sets every CUDA build compiles (inference-core, flash-attn, paged-attn, quant). The SM90/SM100-only sets (FA3,
FlashInfer GDN, DeepGEMM, NVFP4) stay static: FA3 builds with `-fvisibility=hidden`, which a shared library would not
export, and none of them can be verified on this sm_86 card.

Command: `scripts/local_ci.sh --cuda`, then `--lint --tests --cuda`. Both suites green (2453 CUDA, 2133 CPU).

Finding:

- An inference-core CUDA test binary drops from 423 MB to 309 MB. It lists `libinferencecuda.so` and
  `libinferencepagedattention.so` as NEEDED by absolute path, and `--as-needed` drops the kernel sets it never calls.
- The remaining 58 MB `.nv_fatbin` is mostly candle-kernels' `libmoe.a` (an upstream git dependency, left alone).
- `cuda-kernels/` is 861 MB (objects plus `.so`), shared by clippy and tests. Before, each variant held its own
  objects plus an archive copy of them.

## Run 7 - 2026-09-25 17:00

Question: can stale artifacts be removed without a clean rebuild?

Change: `scripts/local_ci.sh --sweep` replays each selected mode as a no-op build with `--message-format=json`, and
`scripts/sweep_target.py` deletes what no replay reports.

Findings while validating the dry runs:

- Uplifted binaries (examples, CLI bins) are reported by their unhashed hardlink, so the hashed original and its
  `.dwo` files looked stale. Fixed by matching inodes and promoting the original's hash.
- The clippy replay must pass `-- -D warnings`. The flag is part of clippy's fingerprint, and a replay without it
  rebuilt every lint unit (48 s).
- `cargo test --doc` cannot be replayed (`--no-run` is rejected). Its units turned out to be covered by the other
  replays; the only unreported artifacts were from a hand-run `cargo check`, which is not a canonical mode.
- A unit shared by several replays was counted once per replay, so crates kept too many incremental dirs.
- `inference-ffi` rebuilds four times per full run: `FeaturesChanged { old: ["cuda"], new: [] }`. A cdylib gets no
  metadata hash, so its CPU and CUDA variants share one slot. This predates this work and costs a relink per mode.

Result, all modes twice with `--sweep`: 190 s and 134 s, green both times. The first sweep freed 5.6 GiB, and the second
under 0.1 GiB (21 entries). The rerun recompiled only the inference-ffi churn, so nothing live was deleted.
`target/debug` was 34.5 GB, of which `incremental/` was 19.5 GB.

`incremental/` at 19.5 GB was the next target. Each crate dir held two finalized sessions (rustc deletes the older
one only at the start of the next compile), each ~1.45 GB for an inference-core build unit: 782 MB query cache,
358 MB `pre-lto.bc`, 225 MB objects, 81 MB `.dwo`. The sweep now keeps only the newest session, which frees another
12 GiB. The bitcode comes from `lto = false`, which still means thin local LTO at opt-level 3.

## Run 8 - 2026-09-25 17:30

Question: what does `lto = "off"` (no thin local LTO) cost and save in the dev profile?

Command: `cargo test -p inference-core --lib --no-run` in two empty target dirs, one with `CARGO_PROFILE_DEV_LTO=off`,
then two one-line edits to `ops/topk.rs` in each, and the lib test binary run twice.

| | cold build | edit rebuild | incremental dir | test binary | lib tests (16 threads) |
|---|---|---|---|---|---|
| `lto = false` (default) | 252 s | 28.8 s / 27.9 s | 1299 MB | 213 MB | 7.00 s / 7.06 s |
| `lto = "off"` | 186 s | 10.8 s / 10.8 s | 939 MB | 193 MB | 7.47 s / 7.46 s |

Change: `[profile.dev] lto = "off"`. The edit loop is 2.6x faster and a cold build 26% faster, for ~6% on CPU-bound
test code. GPU kernels are compiled by nvcc and unaffected.

## Run 9 - 2026-09-25 18:10

Question: the first full run with `lto = "off"` took 87 s per suite instead of 55 s. Where did that come from?

Finding:

- Per-test times, compared within each suite, put almost all of it on `inference-ffi::layout_abi
  detections_through_the_abi`: 13-16 s became 47-53 s in both suites, and it is the tail. PaddleOCR-VL CPU tests were
  ~10% slower. The rest were unchanged. (A first comparison joined the CPU and CUDA suites by test name and blamed
  PaddleOCR by mistake.)
- In isolation the layout ABI test takes 8.1 s with thin local LTO and 42 s without it. PaddleOCR `page_00` takes
  15.8 s and 18.4 s.
- `profile.dev.package."*".codegen-units = 1` (deps as one CGU) changes neither: 41.8 s and 18.5 s. The hot code is in
  the workspace crates.
- `profile.dev.package.inference-layout.codegen-units = 1` brings the layout test back to 8.9 s. Its CPU conv kernels
  need inlining across the 256 incremental CGUs.

Change: keep `lto = "off"` and add `[profile.dev.package.inference-layout] codegen-units = 1`. PaddleOCR's ~16% on CPU
is the remaining cost of the 2.6x faster edit loop.

## Run 10 - 2026-09-25 18:30

Command: `scripts/local_ci.sh --lint --tests --cuda --sweep`, twice.

Result: 257 s, then 142 s warm, green both times. CPU suite 63 s and CUDA suite 59 s (55 s and 57 s with thin local LTO).
The first sweep freed 10.4 GiB; the second found only the inference-ffi churn. `target/debug` is 18.5 GB for all
three modes:

| part | size |
|---|---|
| `incremental/` | 8.5 GB |
| `deps/` | 7.8 GB |
| `cuda-kernels/` | 0.9 GB |
| `build/` | 0.7 GB |
| `examples/` | 0.6 GB |

From 105 GB at the start: examples and full DWARF were the bulk of it, then stale variants, per-variant kernel copies,
superseded incremental sessions and pre-LTO bitcode.

## Run 11 - 2026-10-11

- Question: how much of the shipped library does each model family cost, i.e. what would a one-model package save?
- Command: `RUSTFLAGS= cargo build --profile bundle -p inference-ffi` (the wheels' build, as `local_ci.sh --size`) with `--features cuda` and with `--no-default-features --features cuda,code-execution[,models-<family>]`. Scratch target on /mnt/data, sm_86, CUDA 12.8.
- Library sizes:

| families | file | `.text` | `.nv_fatbin` | link wall |
|---|---|---|---|---|
| all five (default) | 108.4 MiB | 58.4 MiB | 30.1 MiB | 162 s |
| none | 96.5 MiB | 48.5 MiB | 30.1 MiB | 71 s |
| qwen only | 100.1 MiB | | 30.1 MiB | 68 s |
| llama only | 99.5 MiB | | 30.1 MiB | 69 s |

- **Families:** together 12.0 MiB (11%), one family 3-4 MiB.
- **Always shipped:**
  - 48.5 MiB of host code: the engine, plus speech, diffusion, layout and the agent, which have no features;
  - 30.1 MiB of kernels, the same for every family set;
  - ~18 MiB of tables, relocations, unwind and exception data.
- **Source lines:** text families 102k; always-built model crates 27k (speech 12k, layout 5k, diffusion 4k, agent and code-exec 6k); engine 305k (quant 104k, nn 89k, core 66k, tensor 46k).
- **Kernels are per architecture:** a release built with several values in `CUDA_COMPUTE_CAP` carries ~30 MiB per SM.
- Implication: model-family features cannot make a small package. The split has to come from the kernels (by architecture, and optional for CPU-only use) and from the engine's own optional parts (quantization formats, speech/image/layout).

## Run 12 - 2026-10-11

- Question, from the user: if capabilities (text generation, audio, image, layout, agent) were plugins, what would the core and each plugin weigh?
- Command: the Run 11 bundle build (`--features cuda`, all families) with `--config profile.bundle.strip=false`; 160 s wall, 134.8 MB unstripped, the same 61.3 MB `.text` as the stripped file.
  - `nm -S --size-sort -C --defined-only`: 78,785 symbols. Each text symbol goes to the first `inference_*`/`candle_*` crate in its demangled name, else the first non-std crate.
  - `inference_core` and `inference_nn` are split by module: decoder/scheduler/KV/paged/vision pipelines and nn's attention/kv/decoder/moe/gdn count as text; speech/transcription as audio; diffusion as image.
  - Third-party crates serving one capability go with it: symphonia, rustfft and the phonemizer to audio; image codecs to image; html5ever to the agent; tokenizers, minijinja and llguidance to text.
  - Caveats: under fat LTO, code inlined across crates is counted where it ends up, and generics are counted against the crate that instantiates them, so the split is approximate.
- Host code (58.1 MiB of text symbols):

| bucket | MiB | largest parts |
|---|---|---|
| engine | 18.3 | inference-tensor 10.25 (dtype-generic CPU ops), quant 2.48, api 1.31, protocol 1.12, core engine/pipeline 0.8 |
| text generation | 11.4 | qwen 1.53, gemma 1.22, llama 1.12, tokenizers 0.98, other 0.83, core text pipelines 0.70, minijinja 0.58, llguidance 0.53 |
| shared third-party | 10.3 | drop glue 1.18, serde_json 0.71, sort 0.59, rustls 0.44, h2 0.32 |
| CUDA kernel host | 10.1 | C++ launch code: launch_fattn 2.70, instantiate_mmq 2.32, mmq fixups 0.48, marlin 0.42 |
| audio | 4.7 | phonemizer 2.43, inference-models-speech 1.11, rustfft 0.58, symphonia ~0.3 |
| agent | 1.5 | inference-agent 0.30, mcp 0.28, html5ever 0.28, code-exec 0.25, html2text 0.17 |
| image | 1.4 | image crate and codecs ~1.0, inference-models-diffusion 0.28 |
| layout | 0.4 | inference-layout 0.41 |

- Kernels (`.nv_fatbin` per static archive, sm_86): inference-quant 18.0 MiB (MMQ/MMVQ/Marlin), inference-fattn 11.2 MiB, inference-nn 0.6 MiB.
- The other sections (rodata, unwind/exception tables, relocations; ~18 MiB) scale roughly with the code, about +30% on each bucket.
- Implications for a plugin split:
  - **Core (engine plus shared): ~29 MiB of host code, ~37 MiB with its tables.** It needs no kernels beyond inference-nn's 0.6 MiB.
  - **Text generation is most of the GPU weight:** 11.4 MiB host, plus FlashAttention and quantized-matmul kernels (29 MiB) and their 10 MiB of launch code.
  - **Small plugins:** audio ~6 MiB, image ~2 MiB plus quant kernels when it loads a GGUF, agent ~2 MiB, layout ~0.5 MiB, each with tables.
  - **These sizes assume plugins share the core** (one build's Rust dylibs). A self-contained plugin behind a C ABI would also carry the ~29 MiB engine.

## Run 13 - 2026-10-11

- Question, from the user: how well does text generation divide internally; would shipping one family be smaller?
- Host code, from Runs 11 and 12:
  - A family's own crate is 0.5-1.5 MiB of `.text` (qwen 1.53, gemma 1.22, llama 1.12, other 0.83, phi 0.49).
  - A one-family library is 3-4 MiB larger than a no-family one.
  - The rest of text generation's ~11 MiB (tokenizers, minijinja, llguidance, core decoder/scheduler/KV pipelines, nn attention/KV/decoder) is shared by every family.
- Kernels, per source object in the dev build's shared kernel sets (`target/debug/cuda-kernels/*/*.o`, `size -A`, sm_86):

| group | GPU (`.nv_fatbin`) | host launch code |
|---|---|---|
| MMQ, the IQ/ik GGUF types (23 instance files) | 9.53 MiB | 2.61 MiB |
| FlashAttention MMA f16 (one file per ncols1 x ncols2, every head size inside) | 8.68 MiB | 2.49 MiB |
| MMVQ (decode, all types) | 2.07 MiB | 0.87 MiB |
| Marlin (GPTQ/AWQ/affine) | 1.52 MiB | 0.52 MiB |
| MMQ, the 11 standard GGUF types (q2_k...q8_0) | 0.33-0.57 MiB each, ~4.8 MiB together | 0.11 MiB each |
| FlashAttention tile, per head size | 0.15-0.35 MiB each | 0.16-0.29 MiB each |
| other quant formats and ops | 0.62 MiB | 0.70 MiB |
| inference-nn kernels (GDN, MoE, SSM, sort, ...) | 0.67 MiB | 0.53 MiB |

- Implication: text generation divides along kernels, not families.
  - **Quantization format** is the biggest lever. The IQ/ik formats alone are ~12 MiB with their launch code, more than all five families together. Standard GGUF types are ~0.4 MiB each; Marlin ~2 MiB.
  - **Head size** comes next. A family needs only its own: Llama/Qwen 128, Gemma 256, DeepSeek MLA 576/512, ...
    - The tile kernels are already split per head size.
    - The MMA kernels hold every head size in each file, so a per-family split would need the instance files regrouped by head size.

## Run 14 - 2026-11-05

- Question, from the user: get `target/debug` down to its bare essentials, with the build itself removing stale parts and
  build-only artifacts. Step one: what a freshly swept `target/` holds.
- Context: the root disk filled (0 MB free) during a full CI run. Freed: incremental dirs older than a day and all but
  the newest 20 per crate (~45 GB), the npm and uv download caches (~29 GB), then `scripts/local_ci.sh --lint --tests
  --cuda --slim --bindings --sweep` (green; the sweep freed 101.8 GiB). After it `target/` is 45 GB:

| part | size |
|---|---|
| `debug/incremental` (623 dirs) | 21-22 GB |
| `debug/deps/*.dwo` (489,896 files) | 10.23 GiB |
| `debug/deps` linked test binaries (108) | 8.59 GiB |
| `debug/deps` rlib (598) / rmeta (1303) / so (47) | 2.13 / 1.03 / 0.61 GiB |
| `debug/examples` | 975 MB |
| `debug/build` | 541 MB |
| `bundle` (`--size`) / `release` | 7.0 / 4.8 GB |

- Linked test binaries per mode, from `cargo nextest list` over `--workspace --lib --bins --tests`: CPU 61 (4.1 GiB),
  CUDA 61 (4.7 GiB). 15 and 10 of them hold no tests, but together only 0.1 GiB: the empty harnesses cost link time,
  not disk.
- `.dwo` liveness: a codegen unit is live when a live rlib holds its `<crate>-<hash>.<cgu>.rcgu.o` member, or a linked
  binary or shared object names its `.rcgu.dwo` in a skeleton unit (scanned as bytes: decoding `.debug_info` with
  readelf took ~30 s a binary). 48 s on 16 processes.

| `.dwo` | files | size |
|---|---|---|
| all | 489,896 | 10.23 GiB |
| referenced | 35,046 | 0.90 GiB |
| referenced by nothing | 454,850 | 9.33 GiB |

- `inference_ffi` alone held 217,121 (2.41 GiB), dated back to 2026-10-29. A unit keeps its hash while an incremental
  rebuild renames its codegen units, and the old units' `.dwo` stay: Run 7's sweep works per unit (its hash) and
  cannot see them.
- Also: 12 leftover `rmeta*` temp dirs from interrupted rustc runs (48 KB); 733 `.dwo` names referenced but absent
  (to check before deleting anything).
- Implication: the sweep needs a codegen-unit pass. Next: quarantine the unreferenced `.dwo`, rebuild, and check
  that backtraces still resolve file:line before deleting them.

## Run 15 - 2026-11-05

- Question: are the 454,850 unreferenced `.dwo` safe to delete, and can the sweep find them on every run?
- Check: moved them to a quarantine dir on the same filesystem; the canonical test build (`cargo nextest run --no-run
  --workspace --lib --bins --tests`) stayed a no-op (one unit: Run 7's inference-ffi relink churn, 2.4 s), and
  `addr2line -i` still resolves a test binary's code to file:line (`host.rs:127`). Deleted: root 120 -> 129 GB free.
- Mistake on the way: `cargo nextest -p inference-drafters` resolves one package's features and built a variant; only
  the canonical replays keep the target dir's variant set.
- `sweep_target.py --superseded` (no replay): `.dwo` no live rlib member or binary names, plus rustc's interrupted
  `deps/rmeta*` temp dirs. First version: 52 s serial, 47 s of it a regex over each binary's string sections. A fixed
  `.rcgu.dwo` search back to the previous NUL: 1.7 s for the whole dir, the same 522 entries. Its verdicts matched a
  whole-file scan of every binary in `deps/` and `examples/` (none of the flagged names appear anywhere).
- `local_ci.sh` runs it after every invocation that builds, before reporting failure (stale codegen units pile up
  whether or not the checks pass); `--sweep` includes it.

## Run 16 - 2026-11-05

- Question: what is left in `incremental/` (23.3 GiB in 632 dirs), and which of it is stale?
- By crate: inference_core 21 dirs 3.61 GiB, the `integration` test binaries 25 dirs 3.00 GiB, nn 13 dirs 1.48 GiB,
  quant 1.23 GiB, server-core and api ~1.15 GiB each, the model crates 0.5-0.8 GiB each.
- inference_core's 21 dirs: four codegen variants at 465-516 MB (512 objects and their `.dwo`, ~140 MB query cache:
  the CPU and CUDA rlib and lib-test builds), one at 279 MB, and 16 check-only variants at 73-91 MB (clippy with and
  without cuda, the six `--slim` family checks, ...). These are the cache that makes a touch rebuild 17 crates in
  10.9 s, so they stay.
- Stale inside them: 86 dirs held a superseded session (5.67 GiB). rustc loads only the newest finalized session and
  deletes older ones when that unit next compiles; until then only `--sweep` removed them. `--superseded` now does too,
  keeping each dir's newest finalized session (an interrupted `-working` one is never kept). Freed 5.7 GiB in 172
  entries; a touch of inference-drafters then rebuilt 17 crates in 10.9 s, so no cache was lost.
- Also: the `--tests` mode no longer links three SDK examples as a smoke build (`target/debug/examples` was 975 MB);
  clippy's `--examples` still type-checks every example, and `--bindings` still builds `tiny_checkpoint`, which its tests
  run.

## Run 17 - 2026-11-05

- Question: after the cleanup, what remains, and should local CI stop building the CPU-only variant?
- `scripts/local_ci.sh --lint --tests --cuda --slim --bindings --sweep` with the `--superseded` pass and no smoke
  examples: green, the sweep freed 13.5 GiB; root 135 GB free (0 MB at the start of Run 14).
- Left in `target/debug` is live: the incremental cache per build variant (Run 16) and 61 + 61 test binaries, the CPU
  and the CUDA suite each linking their own (4.1 + 4.7 GiB).
- Dropping the CPU build locally would save ~4 GiB of binaries and the CPU codegen variants' incremental dirs. What it
  catches: the CUDA suite runs nearly every CPU-device test too (about 4 tests run only without `cuda`); the 109
  `cfg(not(feature = "cuda"))` items are compiled and linted by GitHub's `Clippy` on every PR. GitHub's CPU `Test`
  job runs only after a merge, 11.5-17.2 min a run (8 runs on 2026-11-05, warm cache). Its one failure on record had
  been red on every master push since #634 (2026-10-25): `openapi_matches_committed`, a default rendered from the
  generating machine's `$HOME` (fixed separately).
- Decision (user): keep the CPU build and suite local; a 12-17 min post-merge job is too slow to be the only check.

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

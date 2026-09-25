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

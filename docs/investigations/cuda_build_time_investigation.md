# Cold CUDA build time investigation

Issue #1: a cold `--features cuda` build is bound by two single-TU kernel files, `inference-core/src/cuda/gdn.cu` and
`inference-paged-attn/src/cuda/flashinfer_decode.cu`, which each compile on one core while the other 15 idle.

Machine: i7-10700K (8C/16T), 125 GB RAM, RTX sm_86, CUDA 12.8. All timings in this doc come from compiling one file with
the real nvcc (no ccache), with the same flags cudaforge passes (`-gencode=arch=compute_86,code=sm_86 -c
--default-stream per-thread -std=c++17 -O3 -U__CUDA_NO_HALF*... --expt-relaxed-constexpr --expt-extended-lambda
--use_fast_math -Xcompiler -fPIC`, plus `-DENABLE_FP8 -I src/cuda` for paged-attn). The wrapper that runs them is
`/usr/bin/time -f "wall=%es maxrss=%MKB" nvcc ...`. The helper scripts named below (`nvtime.sh`, `split_gdn.py`,
`sass_compare.py`, `sampler.sh`) were local scratch tools and are not in the repo. Each one's method is described
in the run that uses it.

## Run 1 - 2026-09-24 22:35

Question: what are the standalone compile times of the two files, as a baseline for the split?

Command: both files compiled concurrently (2 of 16 threads busy).

Finding: gdn.cu took 1584 s (26.4 min) at 1.46 GB, with ~10 other compiles running (Run 7). The FlashInfer
baseline was stopped at 28 min in that setting, and was re-run for Run 9: 3821 s (63.7 min) at 14.9 GB.

## Run 2 - 2026-09-24 22:37

Question: where inside gdn.cu does the compile time go?

Method: `split_gdn.py` parses gdn.cu into top-level items. Constants, types and `__device__` helpers go into
`gdn_common.cuh`, and kernels plus their host launchers go into one `.cu` per family (by the line ranges of the
original file). A whitespace-insensitive multiset compare confirmed the split preserves every character of the
original (188681 non-whitespace chars on both sides).

Finding, first attempt (9 families): several files failed to compile, and their sub-second "times" were errors.

- `gdn_spec_recurrence.cu` launches `gdn_speculative_recurrence_rmsnorm_gate_value_major_128_kernel`, which is
  defined in the fused-speculative family, so those two families must share a TU.
- `gdn_conv1d.cu` lost `causal_conv1d_update_kernel` to the recurrence family, because the splitter assigned items by
  the line of their leading comment rather than of their first code line.

The ones that did compile were all cheap: decode 12.9 s, spec_commit 3.3 s, transition 3.0 s, spec_fused 2.7 s (its
kernels are templates instantiated elsewhere), rmsnorm 2.4 s, packing 2.1 s. The recurrence family was still in
`cicc` after 60+ s, while the whole file had been running for 120 s.

Implication: the cost is concentrated, not spread across 25 entry points. Fix the two splitter bugs, then re-time.

## Run 3 - 2026-09-24 22:40

Question: what does one FlashInfer decode instantiation cost, and does head dim matter?

Command: a TU with `INFERENCE_FLASHINFER_DECODE_INSTANTIATE(__half, __half, HD)` (4 variants: sliding window x
soft cap) for HD = 64 and HD = 512, compiled concurrently.

Finding: hd64 23.2 s, hd512 30.2 s, both with a 435 MB peak RSS. The original file has 24 (dtype pair, head dim)
combinations, and the issue measured ~37 min of cicc+ptxas for the whole file (Run 9 later measured 63.7 min
standalone). One f16 combination is a small fraction of that, so either the cost is superlinear or some pairs are
much heavier than f16 (Run 6: the fp8 ones are).

Implication: a split by head dim (6 dtype pairs x 4 variants per TU) is worth timing before going finer.

## Run 4 - 2026-09-24 22:42

Question: after fixing the splitter (families assigned by first code line; spec fused + spec recurrence merged into one
TU; `launch_gated_delta_rule_recurrence` declared in the header and explicitly instantiated for `__half`,
`__nv_bfloat16` and `float` in its own TU, because the warp and chunked launchers fall back to it when those are
split apart), where does the gdn cost sit? In the final layout, all three launchers share `gdn_recurrence.cu`, so
that declaration was dropped in review.

Finding (8 families, all compiled concurrently):

| TU | wall |
|---|---|
| packing | 2.8 s |
| conv1d | 3.1 s |
| rmsnorm | 3.8 s |
| transition | 4.6 s |
| spec_commit | 5.1 s |
| decode | 14.6 s |
| spec_recurrence (fused + checkpoints) | 26.1 s |
| recurrence | still in cicc after 18 min |

The recurrence family split three ways: plain 5.1 s and warp 3.1 s, so the chunked kernel is the whole cost.

Implication: splitting gdn.cu into families alone does nothing for the tail. The chunked recurrence has to be split
inside itself.

## Run 5 - 2026-09-24 22:45

Question: does the chunked family's cost split by state dtype?

Method: three copies of the chunked TU, each rewritten so all dispatch branches cast `state` to one type (so only that
StateT is instantiated). A first attempt that only forced `state_dtype` at runtime was discarded, because templates
are instantiated regardless of runtime dead code.

Finding: f32 669 s, f16 650 s, bf16 629 s (each ~825 MB RSS, run concurrently with ~10 other compiles). So one
StateT costs ~11 min, and the family is ~3 x 11 min in one TU. Per StateT the kernel is instantiated three times:
(BK 128, key-major), (BK 64, key-major) and (BK 128, value-major).

Implication: a per-StateT split gives an ~11 min tail on its own. Next, time single kernel configurations.

## Run 6 - 2026-09-24 22:45

Question: is FlashInfer decode cost uniform across dtype pairs?

Finding (one (dtype, cache, 512) combination per TU, 4 variants each): f16/f16 30.2 s, f32/f32 37.4 s,
bf16/fp8_e4m3 **427.6 s** with 1.3 GB RSS. The fp8 KV-cache instantiations are ~14x the cost of a same-dtype one.
The per-head-dim TUs (6 dtype pairs each) were all in ptxas at ~3.5 GB RSS after ~8 min of cicc.

Implication: the fp8 cache pairs dominate. Split those finest, and group the cheap same-dtype pairs.

## Run 7 - 2026-09-24 22:58

Question: within one StateT, which chunked configuration is the long pole, and what do the fp8 FlashInfer head dims
cost?

Method: one TU per chunked kernel config (the kernel template plus a function returning a pointer to one
instantiation), f32 state, and one TU per `(bf16, fp8_e4m3, HD)`. All ran concurrently with ~8 other compiles.

Finding:

| TU | wall | peak RSS |
|---|---|---|
| chunked f32 BK 64 | 108.7 s | 374 MB |
| chunked f32 BK 128 | 224.9 s | 730 MB |
| chunked f32 BK 128 value-major | 292.0 s | 727 MB |
| FlashInfer bf16/fp8 hd128 | 383.1 s | 1.3 GB |
| FlashInfer bf16/fp8 hd512 | 427.6 s | 1.3 GB |

The per-head-dim FlashInfer TUs from Run 6 (6 dtype pairs x 4 variants each) all took 1216 to 1314 s at ~3.9 GB,
independent of head dim, which fits the fp8 pairs dominating each of them. The whole gdn.cu baseline finished in
1584 s (26.4 min) at 1.46 GB under the same load. The FlashInfer baseline was still in cicc at 5.1 GB after 28 min
when it was stopped.

Why the chunked kernel is so expensive: `float s[BK]` and `float delta[BT]` live in registers, and phase 5 fully
unrolls BK copies of a runtime loop over the chunk. That is a code-size problem inside cicc/ptxas. I am not changing
it here, since reworking the unrolling is a performance decision for that kernel, not a build fix.

Decision:

- gdn: family TUs plus one TU per (StateT, chunked config) under `src/cuda/gdn_chunked/`, 9 files. The launchers
  get the kernel through `gdn_chunked_kernel<...>()`, a declared template explicitly instantiated in each file, so
  the launch code is unchanged (it already called the kernel through a function pointer).
- FlashInfer: `flashinfer_decode.cu` keeps the `extern "C"` entry points and the cheap reshape/gather kernels.
  `dispatch_flashinfer_decode_softcap<DType, CacheType, HEAD_DIM>` is declared in `flashinfer_decode.cuh` and
  instantiated under `src/cuda/flashinfer_decode/`: one TU per same-dtype pair (all head dims) and one per
  (dtype, fp8 cache, head dim), 15 files. The 28-argument call chain now passes a `DecodeArgs` struct.

## Run 8 - 2026-09-24 23:05

Question: what does a real cold build look like now?

Command: `cargo clean -p inference-core -p inference-paged-attn -p inference-quant -p inference-flash-attn`, then
`NVCC=/usr/local/cuda-12.8/bin/nvcc CUDAFORGE_THREADS=16 cargo check -p inference-cli --features cuda`, which is the
issue's command, run with the real nvcc so nothing comes from ccache. It was sampled every 10 s with
`sampler.sh` (CPU busy % from /proc/stat, plus cicc/ptxas counts and names).

Finding: **Finished in 15m 47s (948 s wall), versus 50m 32s for the same command in the issue.**

| Time | CPU | Compiles |
|---|---|---|
| 0 - 560 s | 93-99% | 15-41 cicc across the four kernel crates |
| 560 - 840 s | 96% falling to 68% | cicc drains, then 7-12 FlashInfer fp8 ptxas |
| 840 - 900 s | 54% falling to 15% | the last 4, then 1, fp8 ptxas (f32_fp8_hd64 last) |
| 900 - 948 s | 8% | Rust check of the dependents |

The machine is now saturated for ~10 of the ~15 minutes. The tail is the fp8 ptxas group (~5 min), and no single
file sets the wall time anymore.

Verification:

- SASS: `sass_compare.py` diffs the instruction lines of every device function (addresses and encodings stripped),
  comparing the baseline gdn.o with the 17 new core objects. **187 functions on both sides, 0 missing, 0 differing.**
  A first version flagged 141 as differing, because its regex let each body run into the next fatbin header. Those
  were parser artifacts.
- The `extern "C"` `T` symbols (`nm`) of gdn.o and the new objects are identical (25 entry points).

## Run 9 - 2026-09-25 00:15

Question: are the FlashInfer kernels identical after the split and the `DecodeArgs` refactor, and what did the
original file cost on its own?

Command: master's `flashinfer_decode.cu` compiled standalone (nvtime.sh, mostly alone on the machine after Run 8).
Then `sass_compare.py` compared it with the new `flashinfer_decode` object plus the 15 instance objects from Run 8's
build, followed by an `nm` diff of the `extern "C"` symbols.

Finding: the original file took **3821 s (63.7 min) with a 14.9 GB peak RSS** as a single process. SASS: **1752
functions on both sides, 0 missing, 0 differing.** The 3 entry points (`flashinfer_decode`,
`reshape_and_cache_flashinfer`, `gather_kv_cache_flashinfer`) are unchanged.

Runtime checks: `cargo test -p inference-paged-attn --features cuda --lib` passed 13/13, including
`mixed_bf16_fp8_hnd_decode_matches_bf16_cache`. `cargo test -p inference-core --features cuda --lib gdn --
--ignored` passed 25/25 of the GPU GDN tests (chunked/value-major prefill, decode variants, speculative commit and
checkpoints, conv1d, rmsnorm, pooled state). The non-ignored 54 passed as well.

Not done: an end-to-end model generation comparison. Identical SASS for every device function, with unchanged launch
code, makes it redundant for this change.

Follow-ups outside this change: the fp8 decode TUs now form the tail (~5 min of ptxas), and the chunked recurrence's
full BK unroll makes it the most expensive kernel in the tree to compile. Both are kernel-design questions.

## Review notes - 2026-09-25

- Instantiations shared between instance TUs, such as FlashInfer's merge-states kernels (they depend only on DType)
  in both `bf16.cu` and each `bf16_fp8_hd*.cu`, appear as weak host stubs in several objects. The linker keeps one,
  and each TU registers its own identical copy of the device code. This is the case nvcc's
  `-static-global-template-stub` addresses. It is harmless here because device code and flags are identical in every
  TU, which the SASS compare confirms.
- The chunked tile sizes (`GDN_CHUNKED_BT`/`GDN_CHUNKED_BV`) are hoisted, so the launchers and instantiations cannot
  drift. A mismatch would only fail at link time.
- `inference-paged-attn/build.rs`'s per-file `rerun-if-changed` list was removed. It was already incomplete, and
  cudaforge's `.watch(["src/cuda"])` emits `rerun-if-changed=src/cuda` for the whole tree.

## Run 10 - 2026-09-25 08:10

Question: why is a CUDA build oversubscribed (a desktop screenshot showed load ~55 on 16 threads), and can the kernel
crates share cargo's `-j`?

Finding (cause): five crates compile CUDA through cudaforge (inference-core, -quant, -paged-attn, -flash-attn and
candle-kernels). Each build script sizes its own rayon pool from `CUDAFORGE_THREADS`, or half the cores by default,
and none of them use cargo's jobserver. So with several running at once, the build has up to ~16 x N nvcc processes
plus cargo's own 16 rustc jobs.

Change: cudaforge 0.1.6 is vendored as `third_party/cudaforge` and patched in with `[patch.crates-io]`, which also
covers candle-kernels.

- New `src/jobserver.rs`: every nvcc compile holds a slot, either the build script's implicit token or one from
  cargo. The final `nvcc --lib` archive step runs on the build script's own slot.
- Tokens are requested through jobserver's helper thread, and waiters block on a condvar that wakes on either a token
  or the implicit slot freeing, so `-j1` cannot deadlock.
- Polling with `try_acquire` was rejected: it busy-waits, and it returns `Unsupported` on non-Linux Unixes, or when
  reopening the inherited pipe fails (on Linux, jobserver 0.1.34 reopens `/dev/fd/N` non-blocking).
- The default pool is raised to all cores, since the jobserver is now what bounds concurrency.

Command: `cargo clean -p inference-core -p inference-paged-attn -p inference-quant -p inference-flash-attn -p
candle-kernels`, then `NVCC=/usr/local/cuda-12.8/bin/nvcc cargo check -p inference-cli --features cuda`, with no
`CUDAFORGE_THREADS`, sampled every 10 s (load average and the nvcc/rustc/cicc/ptxas counts).

Finding: **nvcc never exceeded 16 concurrent processes, and load average peaked at 18.7** (it was ~55 before). CPU
was at 96-99% until ~670 s, then the long-pole files (the fp8 FlashInfer and chunked GDN TUs in ptxas) formed a tail
to ~990 s. The run took 17m 09s. That is not comparable to Run 8's 15m 47s, because this clean set also rebuilds
candle-kernels, so Run 11 re-runs the upstream cudaforge on the same set.

## Run 11 - 2026-09-25 08:50

Question: is Run 10's wall time a regression? It needs the same clean set with upstream cudaforge.

Command: Run 10's command with master's `Cargo.toml`/`Cargo.lock` (registry cudaforge) and `CUDAFORGE_THREADS=16`, which
is the setting used before this change.

Finding: 17m 27s, with **load average up to 57.5 and up to 58 concurrent nvcc processes**. Compared with Run 10:

| | upstream cudaforge | jobserver cudaforge |
|---|---|---|
| wall | 17m 27s | 17m 09s |
| peak load | 57.5 | 18.7 |
| peak nvcc | 58 | 16 |

The jobserver change removes the oversubscription at no wall-time cost. A fully busy machine finishes the same work
at the same rate whether it runs 16 or 58 compilers.

## Run 12 - 2026-09-25 09:05

Question: what do warm rebuilds cost?

Method: Run 11's state (every kernel built) and the same env, timing `cargo check -p inference-cli --features cuda`.
The "Compiling N of M kernels" lines that cargo replays from cached build-script output are not compiles. Only new
ones count.

Finding:

| change | time | kernels compiled |
|---|---|---|
| none | 11.0 s | 0 |
| `touch` a Rust file in inference-core | 46.5 s | 0 |
| edit a Rust file inside `inference-core/src/cuda/` | 10.2 s | 0 |
| edit one `.cu` | 12.1 s | 1 |
| revert that `.cu` | 12.0 s | 1 |

Kernel incrementality is right. cudaforge rebuilds a `.cu` only when its own content hash changes, and rebuilds the
whole crate only when a watched `.h`/`.cuh`, or the args (including our header-hash define), change. The 11 s no-op
was a bug. `CARGO_LOG=cargo::core::compiler::fingerprint=info` showed the inference-core build script stale on
`missing ".../inference-core/.git/HEAD"`. `set_git_revision` emitted `rerun-if-changed=.git/HEAD`, which cargo
resolves relative to the package directory. That file never exists, so the script reran and inference-core plus every
dependent rebuilt on each invocation. In a full `cargo build`, that means recompiling the biggest crate every time.

Fix: ask git for the real paths (`rev-parse --path-format=absolute --git-path` for `HEAD`, the current branch ref and
`packed-refs`, which also covers worktrees), and emit only paths that exist.

Finding: the first build after the fix reran the script once (10.6 s). After that, **no-op builds take 0.32 s** and
nothing is marked dirty.

Review follow-ups:

- An arrived token is now used before the implicit slot. Previously, a waiter woken by a token could take the freed
  implicit slot instead and leave the token unreturned, which cost a `-j` slot for the rest of the build.
- A helper error wakes its waiter and stops limiting instead of stalling.
- `from_env_ext(true)` validates the inherited fds.
- The git-revision watch now also covers a packed branch ref (the nearest existing ref directory plus
  `packed-refs`, only in that case), so the first commit after `git gc` or a fresh clone is picked up. Verified in a
  scratch repo: `pack-refs --all`, then a commit writes the loose ref under the watched directory.

## Run 13 - 2026-09-26 11:00

- Question: after the reorg, where does a cold build run on few threads, and is an inference-core split warranted?
- Command: `CARGO_TARGET_DIR=<scratch> cargo test --no-run --features cuda --workspace --lib --bins --tests --timings`
  (nvcc through ccache, so CUDA TUs were mostly cache hits; rustc was cold).
- Result: 297 s, 870 units. 16 active units until ~t=85 s, then 2-4 active from t=130 s to the end (170 s). The tail
  holds 579 unit-seconds over 167 s (~3.5 average on 16 cores). Total 2157 unit-seconds, so ~135 s ideal.
- Critical path, all inference-core:
  - `inference-core` lib: t=104-231 s (126 s: frontend 68 s single-threaded, codegen 58 s).
  - `inference-core` lib test: t=118-297 s (179 s). It ends the build.
  - `inference-server-core` lib waits on core's rmeta (t=172, 80 s), then its lib test (t=231, 66 s).
  - candle-core (36 s) and inference-quant/layout (~20 s) sit just before core.
- inference-core is 322k lines: vision_models 84k, pipeline 58k, models 23k, paged_attention 17k, cuda 16k,
  gguf 14k, speculative 10k, ops/xlora/kv_cache ~8k each. 1609 unit tests.
- Coupling: the model trees import model-facing pipeline items (`ModelForwardContext`, `EitherCache`, `IsqModel`,
  `NormalLoadingMetadata`, `text_models_inputs_processor`), `speculative` mixins, `layers`, `ops`, `paged_attention`,
  and `attention`. Vision models also carry their input processors (`Processor`, chat-template use). Reverse deps
  into vision_models come from loaders (15 files), gguf (3) and pipeline (2).
- Implication: a split is warranted. The win comes from sibling crates that compile in parallel after a shared base,
  not from a serial base -> core chain, and from splitting the 179 s lib-test build.

## Run 14 - 2026-09-26 12:00

- Question: after moving the base modules (layers, attention, caches, GDN, MoE, CUDA/Metal, utils; ~78k lines) into
  `inference-nn`, where is the single-threaded stretch?
- Command: `cargo test --no-run --features cuda --workspace --lib --bins --tests --timings` after a one-line change
  in inference-nn (incremental rebuild of nn and everything above it, not cold).
- Result: 186 s. From t=20 s to t=70 s only two units run, `inference-core` lib and `inference-core` lib test (rustc
  frontend is single-threaded, so this is the one-thread stretch seen in htop).
  - `inference-nn` lib 12.3 s (frontend 6.2 s), lib test 16.9 s.
  - `inference-core` lib 113.8 s (frontend 54.7 s, was 68 s), lib test 167.4 s and still ends the build.
  - `inference-server-core` lib 86.9 s, lib test 61.3 s, both waiting on core.
- Implication: the base layers were a small share of core's compile. The time sits in what remains (vision_models
  84k, pipeline 58k, models 23k) and in core's lib-test build, which compiles all of it a second time. Splitting the
  models into sibling family crates (step 3) is where the tail breaks up; step 1 only enables it. Cold numbers to
  follow for a like-for-like comparison with Run 13.

## Run 15 - 2026-09-26 12:15

- Question: cold, like-for-like with Run 13, does moving the base modules into `inference-nn` shorten the build?
- Command: same as Run 13 (`CARGO_TARGET_DIR=<scratch> cargo test --no-run --features cuda --workspace --lib --bins
  --tests --timings`), load average ~17 on 16 cores during the run (browsers open), so +-10% noise.
- Result: 321 s (Run 13: 297 s). Total 2358 unit-seconds (was 2157).
  - `inference-nn` lib t=114-142 s, 28 s (frontend 10.9 s); its lib test 28 s runs in parallel with core.
  - `inference-core` lib t=125-257 s, 132 s (frontend 72.8 s, was 68 s); lib test 179 s, still ends the build.
  - 2-4 units active from t=160 s to the end again.
- Negative result: taking 78k lines out of core did not shrink core's frontend, and nn adds ~11 s of serial frontend
  ahead of it.

## Run 16 - 2026-09-26 12:25

- Question: what does inference-core's single-threaded compile time go to?
- Command: `RUSTC_BOOTSTRAP=1 cargo rustc -p inference-core --lib --features cuda -- -Z time-passes` (scratch
  target, only core rebuilt, quiet machine).
- Result: 94 s total. Serial passes: type_check_crate 10.7 s, MIR_borrow_checking 13.8 s,
  monomorphization_collector 14.3 s, generate_crate_metadata 17.3 s, macro expansion 2.6 s, coherence 2.7 s,
  resolve ~2 s (~65 s together). codegen_to_LLVM_IR 35.7 s is also generated on the main thread; only
  LLVM_passes (43.5 s wall) fans out across codegen units.
- Implication: ~100 s of core's build is single-threaded, and a third of it is monomorphization, IR generation
  and metadata for generic code. Since removing the base modules did not move these numbers, the cost is
  concentrated in core's own code. Next: `cargo llvm-lines` on core to find the generic functions that dominate IR,
  and `-Z self-profile` for typeck/borrowck by item, before choosing where to cut.

## Run 17 - 2026-09-26 12:40

- Question: which code makes up inference-core's LLVM IR (the 36 s serial IR generation, 14 s monomorphization and
  43 s of LLVM passes in Run 16)?
- Command: `cargo llvm-lines -p inference-core --lib --features cuda` (scratch target).
- Result: 7.91M lines, 126k function copies. No single hot function (largest is `Engine::run` at 0.5%).
  - By crate of origin: inference_core 40.1%, core 13.6%, rustfft 10.3%, alloc 9.1%, serde_json 4.0%, rav1e 2.4%,
    std 2.3%, inference_nn 2.0%, candle_core 1.9%, hashbrown 1.7%, tokio 1.5%, tokenizers 1.3%.
  - rustfft (818k lines): `FftPlanner::<f32>` in the Gemma 3n and Gemma 4 audio processors and `FftPlanner::<f64>` in
    Phi-4-MM's instantiate every SIMD butterfly inside core, for both float types.
  - rav1e (192k lines, the AVIF encoder): `DynamicImage::write_to(&mut Cursor, ImageFormat::Png)` in
    `engine/agentic_session.rs` and `pipeline/response.rs` is generic over the writer and dispatches on the format
    at runtime, so every enabled encoder is instantiated in core. `image`'s default features (including avif) come
    in through openai-harmony.
  - serde ~588k lines, almost all `serde_json` `StrRead` visitors for ~186 config structs; one deserializer path, so
    it only shrinks by moving the configs out with their models.
  - Trait default methods monomorphized per model: `create_anymoe_layers` x63 (80k lines), and
    `load_tensors_from_path` x96 (64k).
  - The rest is broad: model `forward` 5.0%, constructors 4.7%, iterator adapters and drop glue.
- Implication: ~13% of core's IR (rustfft, rav1e) can go with local fixes: FFT planning behind non-generic
  functions in inference-audio, and a direct `PngEncoder`. `create_anymoe_layers` can delegate to one non-generic
  body. The remaining bulk is spread across the models and their configs, which is what family crates split.

## Run 18 - 2026-09-26 12:50

- Change: FFT planning moves behind `inference_audio::fft::plan_forward_{f32,f64}` (non-generic, so rustfft's
  kernels instantiate in inference-audio); the Gemma 3n, Gemma 4, Voxtral and Phi-4-MM audio processors call those.
  The three PNG encodes (`write_to(Cursor, Png)` twice, `save_with_format(path, Png)`) use `PngEncoder` directly.
- Command: `cargo llvm-lines` and `-Z time-passes` as in Runs 16 and 17, quiet machine (load ~2).
- Result: core IR 7.91M -> 6.33M lines (-20%), 126k -> 104k copies. rustfft and rav1e/ravif are gone from core;
  dropping `save_with_format` also removed the other image encoders it instantiated.
  - `-Z time-passes`: total 94.0 -> 84.7 s. codegen_to_LLVM_IR 35.7 -> 27.7 s, monomorphization 14.3 -> 12.2 s,
    generate_crate_metadata 17.3 -> 14.9 s, LLVM_passes 43.5 -> 37.0 s. Typeck (11.1 s) and borrowck (13.9 s) are
    unchanged, as expected.
- Implication: ~12 s less single-threaded time per build of core, and the lib-test build gets the same cut.

## Run 19 - 2026-09-26 14:30

- Question: cold, like-for-like with Runs 13 and 15, after the codegen trim (Run 18), the model interface move and
  the five text-model family crates?
- Command: same as Run 13 (`CARGO_TARGET_DIR=<scratch> cargo test --no-run --features cuda --workspace --lib --bins
  --tests --timings`), load average ~7.5 during the run.
- Result: 275 s (Run 13: 297 s, Run 15: 321 s). 2147 unit-seconds.
  - Family crates start together at t=117 s once inference-nn's metadata is out and take 1.2-8.3 s each.
  - `inference-core` lib t=121-220 s, 99.5 s (frontend 57.3 s, was 72.8 s); lib test 142 s (was 179 s).
  - 2-3 units active from t=150 s to t=230 s: core lib, core lib test, inference-server-core.
- Implication: core is still the critical path. What remains in it is the vision models (~84k lines), the pipeline
  (~58k) and loaders; the vision families are the next split.
- Side finding: target/debug reached 103 GB (71 GB of it incremental caches) after checking each family crate
  alone with and without CUDA; `local_ci.sh --sweep` brings it back to 21 GB.

## Run 20 - 2026-09-26 16:20

- Question: after moving the Gemma vision models (Gemma 3, 3n, 4, DiffusionGemma; ~21k lines) into the Gemma family
  crate, does core's single-threaded stretch shrink?
- Command: same cold build as Run 13, load average ~3-6.
- Result: 277 s (Run 19: 275 s). 88 s of the build have 3 or fewer units active.
  - `inference-models-gemma` lib 15.3 s (frontend 5.4 s), in parallel after inference-nn.
  - `inference-core` lib 98.0 s (frontend 56.6 s, was 57.3 s); lib test 129.7 s (was 142 s).
  - `inference-server-core` lib 82.9 s still follows core.
- Negative result: moving ~21k lines of model code took under 1 s off core's frontend. What remains in core
  (pipeline, loaders, engine, request/response and the input processors) and the generic instantiations it drives
  are what make it slow, not the model code. The family moves still pay off for modularity and core's lib-test
  build; for the critical path the next lever is inside core itself (a pass-level profile of what's left) and
  inference-server-core, which starts only after core.

## Run 21 - 2026-09-26 16:45

- Question: what dominates inference-core's serial frontend now that the model code is out of it?
- Command: `RUSTC_BOOTSTRAP=1 cargo rustc -p inference-core --lib -- -Z self-profile -Z self-profile-events=default,args`,
  events >= 5 ms exported with `crox --minimum-duration 5000` and grouped by query and item.
- Result: the largest items are the async sampling wrappers every pipeline implements (`sample_causal_gen`,
  `try_sample_causal_gen_batched`, `try_sample_speculative_causal_gen`, `sample_block_gen` in normal, multimodal,
  ggml, gguf, embedding, speech, AnyMoE). Each costs ~200 ms in each of mir_borrowck, check_coroutine_obligations,
  optimized_mir and items_of_instance, although its body is one call: awaiting `sample_and_add_toks` (already
  `&dyn Pipeline`) embeds that function's whole state machine in every wrapper's coroutine, and rustc re-checks it
  each time. Next largest: `Engine::add_request` (319 ms borrowck), `Engine::run`, `agentic_loop`.
- Change: `sample_and_add_toks`, `sample_and_add_toks_batched`, `finalize_block_gen` and the speculative driver's
  entry point return a `BoxFuture`, so the wrappers await a small boxed future (one allocation per sampling step).
- Result (`-Z time-passes`, CUDA features, `CARGO_INCREMENTAL=0` for both): total 69.0 -> 64.6 s;
  codegen_to_LLVM_IR 15.9 -> 14.2 s, generate_crate_metadata 12.0 -> 12.1 s, borrowck 9.0 -> 8.8 s.
- Side finding: the same compile with incremental on (the dev default) takes ~85 s cold; incremental costs ~15 s on
  a clean build of core but makes an edit rebuild ~28 s, so it stays.

## Run 22 - 2026-09-26 16:55

- Question: inference-server-core starts after core's metadata and spends ~65 s in codegen; what is in it?
- Command: `cargo llvm-lines -p inference-server-core --lib`.
- Result: 2.10M lines. rav1e (the AVIF encoder) is 9.1%: `encode_agentic_tool_images` in `chat_completion.rs`
  used `DynamicImage::write_to(Cursor, Png)`, which instantiates every enabled encoder, as core's did in Run 17.
  Also large: serde (serialize/visit_map ~13%) and utoipa's OpenAPI generation (`compose`, `operation`, `schemas`
  ~8%).
- Change: encode with `PngEncoder` directly. Result: 2.10M -> 1.48M lines (-29%), rav1e gone.

## Run 23 - 2026-09-26 17:10

- Question: does boxing the request-handling futures shrink the engine's big async functions?
- Change: `dispatch_prepared_request` awaits `Box::pin(handle_request(..))`, and `handle_request` boxes its
  `agentic_loop` and `add_request` calls, so `Engine::run` no longer carries request handling in its state machine.
- Result (self-profile, per item): `add_request` mir_borrowck 319 -> 149 ms; `agentic_loop` 201 -> 179 ms (plus
  136 ms of coroutine obligations of its own); `Engine::run` 202 -> 192 ms (its cost is its own 1134-line body).
  Whole-crate `-Z time-passes` moved within noise (73.3/66.1 s before vs 70.7 s after, load average 8-10).
- Implication: sub-second. Splitting `add_request` validation into a sync function would take another ~0.1 s and
  is worth doing for readability, not build time. The larger serial phases left in core are
  generate_crate_metadata (~12 s) and the type/borrow checking spread over many items.

## Run 24 - 2026-09-26 19:30

- Question: did the inference-nn / family-crate splits (cross-crate inlining with LTO off) or the boxed engine
  futures cost inference speed?
- Binaries: `cargo build --release -p inference-cli --features cuda` at `c159663f` (before the splits, separate
  worktree and target dir) and at `8dddd57c` (current). RTX 3090.
- Command: `inference bench -f <model.gguf> --iterations 5` (512-token TTFT, 128-token decode at depth 4), 4 rounds,
  alternating base/current per model. Each GGUF symlinked into its own dir so the loader doesn't pick up an unrelated
  mmproj sitting beside it in the models dir.
- Raw (prefill T/s | decode T/s):

| round | tiny base | tiny cur | qwen3b base | qwen3b cur |
|---|---|---|---|---|
| 1 | 13514 / 650.8 | 13127 / 648.8 | 7333 / 256.8 | 7359 / 256.3 |
| 2 | 13417 / 646.4 | 13335 / 648.2 | 7463 / 255.7 | 7218 / 255.2 |
| 3 | 13339 / 643.1 | 13086 / 645.8 | 7312 / 254.4 | 7194 / 254.9 |
| 4 | 13308 / 641.8 | 13301 / 643.3 | 7418 / 254.2 | 7546 / 253.8 |
| mean | 13394 / 645.5 | 13212 / 646.5 | 7382 / 255.3 | 7329 / 255.1 |

- Finding: decode is flat (+0.15% TinyLlama, -0.1% Qwen 3B). TinyLlama decode runs at 1.5 ms/token, so it is
  dominated by host overhead and is where an engine-path or inlining regression would show first. Prefill is -1.4% / -0.7%, inside the
  per-run spread (current's r1/r3 TinyLlama runs had +/-570 T/s within-run stddev, base's rounds span 200 T/s).
  Rounds drift down together for both binaries (thermal).
- Implication: no measurable speed cost from the splits or the boxing. Nothing to chase. Release LTO stays off.

## Run 25 - 2026-09-26 21:45

- Question (user: "clippy-driver appears to exhibit low concurrency"): where is clippy serial, and does rustc's
  parallel front end help?
- Incremental: after a one-line edit in core, `cargo clippy --workspace --tests --examples --timings -- -D warnings`
  takes 12 s, 8.8 s of it with <= 2 units active (core check-test 8.1 s, then server-core, pyo3, CLI). Serial by
  dependency order, but short.
- `--slim` (6 family sets of `cargo clippy -p inference-core`): ~5.2 s each warm, run one after another because they
  shared `target/` and cargo locks a target dir per invocation. After a rebase each re-checks core cold, in series.
- Cold `cargo clippy -p inference-core --lib` in a fresh target dir, serial vs `RUSTC_BOOTSTRAP=1 RUSTFLAGS=-Zthreads=8`:

| | total | inference-core | aws-lc-sys build | syn |
|---|---|---|---|---|
| serial | 51 s | 18.4 s | 13.7 s | 5.5 s |
| -Zthreads=8 | 40 s | 7.5 s | 11.2 s | 8.5 s |

- Finding: core's cold check is 18.4 s now (the crate moves took the front end from ~53 s); the parallel front end
  cuts it 2.5x. Each check-only target dir is 1.4 GB.
- Implication: run the slim sets concurrently in their own target dirs (`target/slim/<family>`). `-Zthreads` needs
  RUSTC_BOOTSTRAP on stable and applies to every build via RUSTFLAGS, so it is the user's call, not a script default.
- Parallel `--slim` in per-set target dirs: first run 159 s (one-time dependency check into six new dirs), then
  10 s after a one-line core edit (was ~31 s serial). The six dirs hold 11 GB.
- Decision: rejected. 11 GB of extra target dirs is not worth ~21 s per slim run; `--slim` stays serial in `target/`.

## Run 26 - 2026-09-26 23:20

- Question: cold, like-for-like with Runs 13-23, after the vision, DFlash and loader moves (#35-#44)?
- Command: `CARGO_TARGET_DIR=<scratch> cargo test --no-run --features cuda --workspace --lib --bins --tests --timings`,
  load average ~17 at the end (other work on the machine).
- Result: 241 s (Run 13: 297 s, Run 23: 255 s), 884 units, 2131 unit-seconds. 47 s of the build has <= 3 units
  active (was 86 s).
- Critical path: candle-kernels build script t=13-77 s (64 s), candle-core t=77-119 s, `inference-core` lib
  t=119-199 s (80 s, was 93.6 s), core lib test t=137-239 s (102 s, was 118 s), server-core lib t=168-206 s, its lib
  test and pyo3's lib test t=200-240 s.
- Core's LLVM IR (CUDA lib) is 4.42M lines, from 5.68M at the start of this series.
- Off the critical path but large: `image` 44.5 s, `ravif` 36 s (still pulled in by something even though the server
  no longer encodes AVIF), `tokenizers` 37 s twice (two feature sets), `rust-mcp-schema` 36 s.
- Traced the two off-path extras; neither can be dropped from this workspace:
  - `ravif`: `openai-harmony` 0.0.8 depends on `image` with default features (so AVIF) but never uses `image` in its
    source; its own `default` feature is empty, so only a patched or vendored harmony would drop it.
  - `tokenizers` twice: we use 0.21.4, shared with `toktrie_hf_tokenizers` (even its latest, 1.8.0, pins 0.21); the
    pinned candle rev needs 0.23.2 unconditionally (`candle-core/src/quantized/tokenizer.rs`). Bumping us to 0.23
    breaks `add_special_tokens` (now takes owned tokens) and llguidance's `ByteTokenizer::from_tokenizer` (a 0.21
    type), and would still leave two versions.
- Implication: core's lib test (102 s) is still the last unit and the remaining wall-time lever.

## Run 27 - 2026-09-27 01:10

- Question: core's lib test (102 s cold) ends the cold build; is it worth attacking, e.g. by moving test code out of
  core or building core at a lower opt-level for tests?
- Findings:
  - Core carries ~27k lines of unit tests (pipeline 9.6k, gguf 3.6k, scheduler 3.3k, vision_models 2.9k). The lib
    test is a full second compile of core with `cfg(test)`, so moving tests out saves only their share (~20 s of the
    102 s); integration tests would avoid the second compile but the unit tests lean on `pub(crate)` internals.
  - Incremental, after a one-line edit in core: `cargo test --no-run -p inference-core --lib` rebuilds in 6 s at
    opt-level 3 and 6 s at opt-level 1 (`--config profile.dev.package.inference-core.opt-level=1`); core's 906 unit
    tests run in 7.0 s vs 7.9 s.
- Implication: the cold 102 s only matters for cold builds; for the edit loop it is not the bottleneck, and a lower
  opt-level buys nothing. The cycle's wall time is in `local_ci.sh` itself; measure its phases next.

## Run 28 - 2026-09-27 01:45

- Question: where does a `local_ci.sh --lint --tests --cuda` cycle spend its time after a one-line core edit, and can
  the phases overlap?
- Serial phase times (warm target, edit to a function body in `pipeline/normal.rs`): fmt check 4.3 s, clippy 11.0 s,
  nextest build 16.1 s, nextest run 9.4 s, doctests 9.9 s, smoke examples 1.4 s, clippy (cuda) 11.3 s, nextest build
  (cuda) 17.5 s, nextest run (cuda) 54.6 s. Total ~135 s.
- The CUDA suite is GPU-bound: 2451 tests summing 834 s of per-test time on 16 threads; the inference-nn CUDA tests
  take ~11 s of wall time at 16, 8 or 4 test threads alike. One nn kernel test takes 0.24 s alone and 2.8 s in the
  full suite (GPU sharing). Raising `CUDA_CACHE_MAXSIZE` to 4 GiB changed nothing (0.23-0.24 s), so it is not PTX JIT.
  Leaving out the real-checkpoint tests (PaddleOCR-VL, the layout ABI detection) takes the suite from 53 s to 47 s.
- Change: `local_ci.sh` lints and builds the CUDA suite, runs it in the background, and runs the CPU clippy, CPU
  suite, doctests and smoke build meanwhile; the RLIMIT_NPROC sandbox test (it counts every process the user runs)
  is left out of both suites and runs alone at the end.
- Result: 103 s after the same kind of edit (135 s serial). An intermediate version that overlapped only the CPU test
  run and doctests reached 124 s. In the overlapped run the CPU suite takes 9.9 s and the CUDA suite 60.7 s.

## Run 29 - 2026-09-27 (afternoon)

- Question: how much of core's IR leaves with the speech (Dia) and diffusion (FLUX, T5, CLIP) models moved into
  `inference-models-speech` and `inference-models-diffusion`?
- Command: `cargo llvm-lines -p inference-core --lib --features cuda` after `local_ci.sh --lint --tests --cuda --slim
  --sweep` (all green: 2127 CPU and 2445 CUDA tests).
- Result: 4,134,514 lines (79,045 copies), from ~4.356M after #54: -221k, 5.1%. The module table had put
  diffusion_models at 86k and speech_models at 49k (135k); the rest is presumably their generic instantiations of
  shared helpers that are now codegenned in the new crates. Core keeps the loaders, `SpeechLoaderType`/
  `DiffusionLoaderType` (core impls `arch_metadata` on them) and the diffusion request processor. The Flux stepper's
  hub access is inverted through a `RepoFileFetcher` closure built by core's loader, so the new crate has no hf_hub dep.
- Next: X-LoRA models (xlora_models, ~150k) into their family crates.

## Run 30 - 2026-09-27 (afternoon)

- Question: what does moving the eight non-GGUF X-LoRA models into their family crates take out of core?
- Change: ScalingsMaker, XLoraClassifier and XLoraConfig to `inference_nn::xlora`; gemma/gemma2, llama/mistral/mixtral,
  phi2/phi3 and starcoder2 to `xlora/` in their family crates. The two GGUF ones stay in core (they read gguf Content).
- Command: `cargo llvm-lines -p inference-core --lib --features cuda` after a green `local_ci.sh --lint --tests --cuda
  --slim` (2127 CPU, 2445 CUDA tests).
- Result: 4,047,334 lines (78,650 copies), from 4,134,514: -87k, 2.1%. Less than the 150k the module table gave
  xlora_models; the GGUF pair and their instantiations of core generics stay behind.

## Run 31 - 2026-09-27 (afternoon)

- Question: what is left in core by module after #56, and is any of the biggest share duplicated?
- Command: `cargo llvm-lines -p inference-core --lib --features cuda`, grouped by `inference_core::<module>` prefix.
- Finding: vision_models (the inputs processors kept in core) is now the largest group, spread over ~20 processors of
  4-22k each. Qwen2-VL and Qwen2.5-VL `process_inputs` were 7331 and 7327 lines, and a name-normalized diff of the two
  files showed the 2.5 processor was a copy of the 2-VL one: identical bodies, identical vision args structs; the only
  non-rename difference was that 2.5 built the text metadata after the media block instead of before. That block only
  rewrites tokens when `has_changed_prompt` is false, and prompt planning (add_request and every prompt step) sets it
  first, so the order is moot.
- Change: Qwen2.5-VL uses Qwen2VLProcessor and Qwen2VLVisionSpecificArgs; the 1235-line copy is deleted.
- Result: 4,028,130 lines (78,444 copies), from 4,047,334: -19k. No Qwen VL checkpoint on this machine, so no
  end-to-end run; CPU and CUDA suites green.
- Next: Qwen3-VL and muse_glimmer also import qwen2vl helpers; a normalized diff of qwen3_vl against qwen2vl showed
  1374 differing lines, so less of it is shared.

## Run 32 - 2026-09-27 (afternoon)

- Question: how much of Qwen3-VL's processor duplicates Qwen2-VL's?
- Finding: a per-function comparison showed ten Qwen3-VL helpers identical to the qwen2vl ones (find_sequences,
  video_hashes, grid_patch_count, split_media_pixels, select_media_view, shift_media_spans, media_data_cached_offset,
  select_media_batch, apply_mrope_position_delta(s)) and `Qwen3VLVisionSpecificArgs` field-identical to
  `Qwen2VLVisionSpecificArgs`; packed_layout, prompt_mrope and the processor bodies really differ (video timestamps).
- Change: Qwen3-VL (and Qwen3-VL-MoE, Qwen3.5, Qwen3.5-MoE, which reuse its processor) use the qwen2vl helpers and
  args struct; four duplicate tests removed.
- Result: 4,021,774 lines, from 4,028,130: -6k. Small in IR; the gain is ~360 fewer lines to keep in sync.

## Run 33 - 2026-09-27 (evening)

- Question: the top-function list for engine/gguf/loaders showed dozens of `isq_layer_regexes` impls at 2.4-4.3k lines
  each. What do they cost in total, and why so much?
- Finding (before): isq_layer_regexes 72,435 + immediate_isq_predicates 20,770 + isq_layer_regexes_moqe 5,005 +
  immediate_isq_predicates_moqe 327 = ~98.5k lines over ~54 loaders. Each list element was an inline
  `Regex::new(..)?`, so every pattern carried its own error-conversion and drop path.
- Change: `isq_regexes(&[..])` in pipeline/isq.rs builds the list; a script rewrote the 142 `vec![Regex::new(..)?, ..]`
  lists in the loaders. A review re-extracted every pattern and comment per file in order: identical.
- Result: 3,925,599 lines (78,435 copies), from ~4.028M: -102k, 2.5%. The same functions now total ~6.3k.

## Run 34 - 2026-09-27 (evening)

- Question: does the same inline-error-path pattern explain `GgufDeviceMapLoaderInner::layer_sizes_in_bytes` (13.7k)?
- Finding: yes; 90 `tensor_info_size_in_bytes!(self.model.tensor_info(..)?)` sites, each expanding its own lookup and
  error conversion.
- Change: `tensor_bytes(name)` / `tensor_bytes_as(name, dtype)` methods replace the macro; a review matched all 90
  sites in order (same name, same arm/dtype).
- Result: layer_sizes_in_bytes 13.7k -> 3.3k, non_mapped_size_in_bytes 3.3k -> 1.0k; core 4.022M -> 4.009M on the
  pre-#59 base (-13k). The rest of gguf (~128k) is spread over 1-3k straight-line config builders and bindings with
  no single repeated construct, so it is left as is.

## Run 35 - 2026-09-27 (evening)

- Change (readability, not IR): `Engine::admit_request` (790 lines, 10.7k IR) split into step helpers: request-kind
  validation, message extras, prompt rendering, context fitting, stop criteria, KV preallocation, reasoning setup,
  multimodal prompt prep, recurrent slot assign/release, prefix-cache hit. admit_request is now 231 lines; a review
  found no behavior change (validation order, error variants, lock scopes, early returns).

## Run 36 - 2026-09-28

- Question: what do the embedding models still in core cost, and does splitting core's lib.rs change anything?
- Command: `cargo llvm-lines -p inference-core --lib --features cuda` before and after, on the same tree otherwise.
- Baseline: 3,918,365 lines (78,652 copies). embedding_models:: held 35.3k, of which embedding_gemma 14.8k and
  qwen3_embedding 13.9k; xlora_models:: 32.1k and models::quantized_llama 28.7k.
- Change: EmbeddingGemma and Qwen3-Embedding move to inference-models-gemma / -qwen; their loaders split into
  `embedding_loaders/{gemma,qwen3}.rs`, gated on the family features like the normal loaders (an unbuilt family is
  a load error naming the feature; `EmbeddingLoaderBuilder::build` returns a Result for it). The 2.2k-line
  `impl InferenceRs` and its builder and configs leave lib.rs for `inference_rs/{mod,builder,config,lora,models,
  sessions}.rs` (readability, not IR).
- Result: 3,902,517 after the move (-15.8k): 15.3k of the models' code was still instantiated in core, serde
  deserialization of their configs (generic, so emitted where core called `serde_json::from_str`) and the
  AnyMoE mixin defaults emitted with the trait-object vtables core builds. With the configs parsed through the
  family crates' `json_config!` (as the normal loaders do): 3,892,700, -25.7k (0.66%); 7.8k of vtable defaults stay.
- Not moved: the GGML llama and the two quantized X-LoRA models (60.8k together) read GGUF content, LoRA and the
  model-config traits, which live in core; moving them needs those in inference-nn first.

## Run 37 - 2026-09-28

- Question: where does inference-core's LLVM IR sit per function, and do long functions have outsized IR (so splitting
  them would pay)? Compare IR lines against source lines, and group generic instantiations by their base function.
- Command: `CARGO_TARGET_DIR=<scratch> cargo llvm-lines -p inference-core --lib --features cuda --sort lines`, at
  `b0c0c9b9` (after the last three models left core, #98, and the dead-code cleanups #100-#104). Source lengths from
  brace-matching `fn` bodies; IR per source line computed for core functions with one definition and >= 40 lines.
- Result: 3,924,564 lines, 79,536 copies (Run 36 end: 3,892,700; this build keeps the moved models' GGUF/X-LoRA
  construction instantiated in core through `from_gguf::<File>`, 10.6k). No function passes 0.3%.
  - Long sync functions sit at the crate's typical ~15 IR lines per source line: `MultimodalLoader::
    load_model_from_path` 12.7k IR / 814 lines, `NormalLoader::load_model_from_path` 10.9k / 651. Splitting them
    moves IR between functions rather than removing it; only deduplicating copies removes it.
  - The highest ratios (25-70 per line) are short async functions whose state machines carry their awaits
    (`snapshot_paged_recurrent_prefix` 69, `handle_daemon_request` 51, `execute_extraction` 32, `execute_search`,
    `do_reload_model`, the NCCL replicators) and JSON-building GGUF config synthesizers. Each is a few thousand lines
    at most.
  - Grouped by base function across instantiations: serde config visitors ~265k (`visit_map` 134k + `visit_seq`
    70k + `deserialize_struct` 60k over ~140 structs, 6.8%); drop glue 82k over 2,361 types; `create_anymoe_layers`
    70.7k over 50 impls (1.8%) plus `finish_training` 23.1k over 59; `process_inputs` 59.1k over 23;
    `load_model_from_path` 38.9k over 9; `layer_sizes_in_bytes` + `non_mapped_size_in_bytes` 39.9k over 55 each.
- Implication: function length is not the IR lever; per-model monomorphization is. The largest fixable item is still
  the one Run 17 named: the AnyMoE mixin defaults (`create_anymoe_layers`, `finish_training`) instantiated once per
  model, which could delegate to one non-generic body (~90k, ~2.3%). The loader consolidation removes duplicated
  `load_model_from_path` code (worth it for maintenance; IR follows the lines it deletes). serde's per-struct
  visitors are the biggest block but come with the configs; `visit_seq` exists only for sequence-form input, which
  configs never use, so a hand-written or map-only deserializer would be the only way to cut it.

## Run 38 - 2026-09-28

- Change: the AnyMoE mixin defaults stay thin. `create_anymoe_layers` gathers the MLP list, LoRA targets and a
  `&dyn Fn` for fine-tuned experts from `self`, then calls non-generic `build_anymoe_experts` and
  `install_anymoe_layers`; `finish_training` passes `get_mlps_mut()` to `finish_anymoe_training`. The bodies now
  compile once, in inference-nn.
- Command: same `cargo llvm-lines` as Run 37, scratch target.
- Result: core IR 3,924,564 -> 3,821,632 (-102.9k, -2.6%), 79,536 -> 78,556 copies. `create_anymoe_layers`
  92.4k -> 19.6k (what is left per model is the wrapper and its closure), `finish_training` 28.1k -> 2.1k.
- Implication: a little more than Run 17 estimated, since the trait-object vtables core builds instantiate every
  default once per model. The same shape (mixin default whose body only needs `self` for a few accessors) is worth
  checking in `DeviceMappedModelLoader`'s `layer_sizes_in_bytes` / `non_mapped_size_in_bytes`, although those are
  per-loader overrides rather than shared defaults.

## Run 39 - 2026-09-28 (evening)

- Question: cold, like-for-like with Run 26, after the last models left core, the dead-code sweeps, the AnyMoE
  bodies and the step splits (#92-#114)? Is anything outside our crates worth attacking?
- Command: `CARGO_TARGET_DIR=<scratch> cargo test --no-run --features cuda --workspace --lib --bins --tests --timings`
  at `19e3ac2b`. Load average ~17 by the end: other work on the machine, so treat this as +-10%.
- Result: 254 s (Run 26: 241 s), 832 units, 2318 unit-seconds. 36 s of the build has <= 3 units active (was 47 s).
  - Critical path unchanged in shape: candle-kernels build script t=14-80 s, candle-core t=80-117 s, core lib
    t=117-193 s (76 s, was 80 s), core lib test t=134-253 s (119 s, was 102 s) ends the build.
  - Core's lib test also ends the build now (inference-api lib test ends at 244 s, the flux and CLI test binaries at
    253 s). The core lib test grew with the tests added since (#106-#113) and the machine load.
- External crates are all off the critical path: they finish before core's lib starts. Dropping one saves CPU
  (which matters on a loaded machine), not wall time on an idle one. The inventory with exclusive costs is in
  `dependency_inventory_investigation.md`.
- Side finding: `inference-audio` is 265 source lines, but its lib takes 39 s (37.7 s of that codegen) and 938k IR
  lines. The cause is rustfft's planner, instantiated for f32 and f64 with its SSE, AVX and scalar kernels:
  - f64: 522k lines, used only by Phi-4-multimodal's mel front end, which mirrors numpy's float64 FFT.
  - SSE: 404k lines, a fallback for x86 CPUs without AVX.
  - Scalar: 167k lines.
- Implication: the cold wall time is core's lib test, as in Runs 26-27. Everything else is CPU hygiene, and the
  largest single item there is our own audio crate's FFT features.

## Run 40 - 2026-09-28 (evening)

- Question: what is left in inference-core by responsibility, and which parts could leave it? Core's lib test (119 s)
  ends the cold build, so whatever leaves core comes off the critical path.
- Commands:
  - `CARGO_TARGET_DIR=<scratch> cargo llvm-lines -p inference-core --lib --features cuda`, with each function
    grouped by the first `inference_core::<module>` in its name.
  - Source and test lines per module (tests counted from the first `#[cfg(test)]`, or the whole of a `tests.rs`).
  - Module coupling from the non-test `crate::<module>` references.
- Result: 3,812,237 IR lines, 78,388 copies; 141k source lines, of which 38k are tests. 44.4% of the IR names no core
  item at all (external generics instantiated from core, such as collections, tokio and serde over foreign types).
  | Responsibility | Modules | Lines (tests) | IR |
  |---|---|---|---|
  | Pipeline machinery and loading | pipeline (loaders 21.6k, cuda_graph, multimodal, normal, isq, sampling, chat_template...) | 55.7k (16.9k) | 20.9% (loaders 4.5%) |
  | Multimodal request preprocessing | vision_models | 24.2k (2.8k) | 8.8% |
  | GGUF metadata to configs, bindings, tokenizer | gguf | 14.3k (4.4k) | 3.4% |
  | Engine runtime | engine, scheduler, sequence, prefix_cacher, speculative, distributed, adapter | 26.4k (8.0k) | 13.1% |
  | Protocol types | request, response, files, tools, reasoning_parsers, chat_collector | 10.3k (3.5k) | 5.0% |
  | Selection and tuning | selection, tuning, diagnostics, resource_plan | 6.1k (0.9k) | 1.9% |
  | Services | search, remote_fetch, video_input, block_diffusion, agent_approval | 1.4k (0.1k) | 1.3% |
  | Facade | inference_rs | 3.0k (1.3k) | 1.4% |
- Coupling:
  - The protocol types reference no engine or pipeline module (`request` -> response, files, tools; `tools` ->
    reasoning_parsers), so they are a leaf.
  - gguf references vision_models 4 times, pipeline 3 times and models once.
  - search references pipeline, embedding_models, remote_fetch and engine.
  - Engine, scheduler, sequence and pipeline reference each other in cycles (scheduler -> engine -> scheduler;
    pipeline <-> sequence), and vision_models leans on pipeline 24 times and sequence 21 times through the
    `InputsProcessor` trait.
- Implication, in order of cost to move:
  1. The protocol types can become a crate below core as they stand (about 5% of IR and 3.5k test lines).
  2. gguf needs its few upward references cut.
  3. The normal loaders need `NormalModelLoader`, `DeviceMappedModelLoader`, `IsqModelLoader` and their helpers
     moved into inference-nn first; then each family crate could host its own loader rows.
  4. vision_models and the multimodal loaders need a preprocessing interface that does not take `Sequence`.
  5. The engine runtime and the pipeline machinery stay together.

## Run 41 - 2026-09-28 (evening)

- Question: before splitting "pipeline machinery and loading" out of core, what inside it can shrink?
- Command: Run 40's `cargo llvm-lines` output, restricted to functions naming `inference_core::pipeline`. Trait
  methods are grouped across implementers.
  - Method bodies were compared across the 55 loaders, with the config type name normalized.
  - Pipeline methods were compared between `normal.rs`, `multimodal.rs` and the smaller pipelines, using
    whitespace-normalized similarity ratios.
- Result: pipeline-named IR is 950k lines. The largest groups:
  - `Loader::load_model_from_path` 97.6k (1316 copies, over 9 impls and their closures).
  - `InputsProcessor::process_inputs` 94.3k (the vision processors).
  - `LoadTensors::load_tensors_from_path` 58.6k (1092 copies).
  - `load_model_from_hf` 32.3k.
  - `layer_sizes_in_bytes` 23.5k and `non_mapped_size_in_bytes` 21.0k.
  - `get_device_for_tensor` 24.7k over about 600 copies (the normal, multimodal and embedding defaults).
- Findings:
  1. `load_tensors_from_path` is instantiated once per call site of the safetensors var-builder loader, for both the
     plain and X-LoRA backends, because the predicate is `impl Fn`. Every `|_| true` is its own type. A `&dyn Fn`
     predicate leaves two copies (one per backend), in inference-nn; the family crates' call sites benefit too.
  2. `get_device_for_tensor`'s default builds its `Regex` and closure inside the trait default, so every loader
     instantiates it. Delegating to a non-generic function of `num_layers` leaves one copy.
  3. The loaders' `DeviceMappedModelLoader` / `NormalModelLoader` methods are mostly the same body with a different
     config type. Identical once the config type is normalized:
     - `is_gptx`: 26 of 57 are `Ok(true)`.
     - `non_mapped_max_act_size_elems`: 29 of 55 are `Ok(0)`.
     - `num_layers`: 19 of 55 read `num_hidden_layers`.
     - `mapped_max_act_size_elems`: 19 of 55 compute batch * heads * min(seq, chunk)^2.
     - `get_config_repr`: 24 of 57.
     - `non_mapped_size_in_bytes`: 15 of 55 compute embed + untied head + norm.
     - `layer_sizes_in_bytes`: llama, mistral and smollm3 are identical, and more share the dense-decoder shape.
     Defaults derived from `model_config()`, plus a dense-decoder size helper, would remove about 1-1.5k source lines.
  4. `NormalPipeline` and `MultimodalPipeline` duplicate each other:
     - 30 methods, 586 lines, are identical apart from whitespace: speculative sampling and attach, CUDA-graph
       capture bookkeeping and reclaim, calibration, hybrid recurrent snapshots, cache clone, AnyMoE layers,
       `re_isq_model`.
     - The CUDA decode-graph path is 74-89% similar: `try_cuda_decode_graph_forward` 187 lines, `forward_step` 129,
       `capture_cuda_decode_graph_step` 106, `replay_cuda_decode_one_token` 28.
     - The GGUF, GGML and embedding pipelines repeat about 70 lines each of the short ones.
     - A shared runtime struct owning the CUDA-graph, speculative and calibration state, with these methods
       implemented once, would remove roughly 1k lines.
- Implication: findings 1 and 2 are small edits worth about 75k IR (about 2% of core). Findings 3 and 4 are mainly
  source (2-2.5k lines) with some IR. All four also shrink what a later move of the loaders or pipelines out of core
  would carry.
  - Not cheap: `load_model_from_path` (the loader consolidation concluded, #110), `process_inputs` (per-model vision
    code), and the config deserializers.

## Run 42 - 2026-09-28 (night)

- Change:
  - Run 41's findings 1 and 2. The safetensors loader erases its predicate into `Arc<dyn Fn>` before the
    non-generic body, and its four spawn blocks become one. The `get_device_for_tensor` defaults and lfm2vl call
    one `layer_indexed_device`.
  - The self-contained dependency-inventory findings:
    - rubato and rustfft without default features (scalar FFT kernels);
    - variantly replaced by two methods;
    - tqdm replaced by the indicatif-backed `with_progress`;
    - tokio-test dropped, and futures-util replaced by futures;
    - strum 0.28, and scraper 0.26 with html2text 0.17 (one html5ever).
- Command: the same `cargo llvm-lines -p inference-core --lib --features cuda` as Run 40 (scratch target), plus
  `cargo llvm-lines -p inference-audio --lib`.
- Result:
  - Core: 3,812,237 -> 3,585,564 IR lines (-226.7k, -5.9%), 78,388 -> 73,983 copies.
    - `load_tensors_from_path` no longer appears in core at all (was 58.6k over 1092 copies). Its closures,
      thread-spawn shims and drop glue went with it, which is why the drop is about three times Run 41's estimate.
    - `get_device_for_tensor` is 24.7k -> 4.6k.
  - inference-audio: 938,164 -> 197,540 IR lines (-79%).
  - Gone from the build graph: syn 1, darling 0.11, uuid 0.8, tqdm, crossterm 0.25 and realfft; strum and
    html5ever each down to one version.
  - Full `local_ci.sh --lint --tests --cuda --slim --bindings --docs` green (2188 CPU, 2508 CUDA tests).
- Not done: moving reqwest 0.13 off aws-lc. With `rustls-no-provider`, reqwest falls back to the process-wide rustls
  provider and panics if none is installed, so every client path would need an install first. That includes the 8
  `reqwest::get` calls in the examples, which users copy. That is not worth ~30 s of CPU off the critical path.

## Run 43 - 2026-09-28 (night)

- Question: Run 41 finding 4 (`NormalPipeline` / `MultimodalPipeline` duplication). What would sharing it take?
- Finding:
  - Of the 57 methods identical between the two files, most are 3-line trait accessors. The larger ones
    (`amoe_create_layers`, `try_sample_speculative_causal_gen`, `attach_speculative_with_runtime`, `apply_calibration`,
    `re_isq_model`, the CUDA-graph state methods) are glue over helpers that are already shared: AnyMoE weight
    loading, the speculative driver, `isq_flow`, `cuda_graph`.
  - The two model traits share only `cache`, `device`, `config`, `model_config`, `max_seq_len` and the capability
    flags directly, plus their `IsqModel` / `AnyMoeBaseModelMixin` / `SpeculativeTargetMixin` supertraits.
  - Sharing the glue needs one of two things. Option a: a common model supertrait, which re-splits every model impl
    in the five family crates. Option b: a macro stamping the same methods into both impls, which saves no IR and
    reads worse.
  - The near-identical CUDA decode-graph path differs exactly at the model forward calls.
- Implication: the net is roughly 300-400 lines, not the ~1k Run 41 estimated. Not worth doing before unbundling.
  Revisit if the model traits get a common supertrait for another reason.

## Run 44 - 2026-09-28 (night)

- Change: a new `inference-protocol` crate with no candle dependency holds the wire-protocol half of core:
  - `tools/`, including the call parsers and grammars;
  - `reasoning_parsers/` and `files/`;
  - the data structs of `response.rs` and `request.rs`;
  - `TopLogprob`, moved from inference-nn.

  Core keeps the `Request`/`Response` channel types and re-exports the rest, so `inference_core::` paths keep working.
- Question: does the new crate stay off the critical path, even with inference-nn now depending on it for
  `TopLogprob`? What does core lose?
- Command: the Run 39 cold `cargo test --no-run --features cuda --workspace --lib --bins --tests --timings`, then
  `cargo llvm-lines -p inference-core --lib --features cuda` (scratch target).
- Result:
  - inference-protocol's lib builds at t=80-83 s and its lib test at t=80-84 s, after its deps (mcp at 56 s, image
    and harmony) and well before inference-nn's lib (t=99 s) and core's lib (t=119 s). It adds nothing to the
    critical path, and nn's new dependency on it does not delay nn.
  - Core IR: 3,585,564 -> 3,485,212 lines (-100.4k, -2.8%), 73,983 -> 71,868 copies. About 184 unit tests move out of
    core's lib test.
  - Wall: 259 s (Run 39: 254 s). The measurement is noisy: load reached ~18 because lint runs overlapped the end of
    the build, and core's lib test still ends it (t=137-258 s).
- Implication: the split does what it should structurally, but at ~3% of core the wall-time gain is inside the
  noise. The larger step is moving inference-api's request types (`openai.rs`, `responses_types`) here too. That
  takes inference-api's lib test (t=201-254 s) off the tail as well.

## Run 45 - 2026-09-28 (night)

- Question: `local_ci.sh --docs` became noticeably slow. Is it serialized, and what does it cost?
- Command: `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps [--timings]`, timed after `touch`ing one crate's
  lib.rs; `CARGO_INCREMENTAL=0 cargo check -p <crate> --lib` for comparison.
- Result, re-documented crates and wall time per touched crate:
  | Touched crate | Wall | Crates re-documented |
  |---|---|---|
  | inference-protocol (PR #118 as merged) | 163 s | 15, including inference-nn and all 7 model crates |
  | inference-protocol (after the fix below) | 83 s | 7 |
  | inference-core | 71 s | 6 |
  | inference-api | 47 s | 4 |
  | inference-cli | 0.3 s | 0 |
  | nothing | 0.3 s | 0 |
- Not serialized: `--timings` shows the 7 rustdoc jobs after a protocol touch all start within 5 s, since each needs
  only its deps' check metadata (1-3 s). Each job is single-threaded and never incremental:
  - server-core documents in 36.5 s alone and 80.5 s beside the other six (contention roughly doubles each job);
  - server-core's non-incremental check alone is 28.4 s, core's 20.4 s.

  Running it in the background beside the GPU suite would only queue it on the target-dir lock behind clippy and the
  CPU tests.
- Cause of the 163 s: inference-nn depended on inference-protocol only for `TopLogprob`. Any tool-parser edit
  re-documented, and recompiled, nn and every model crate.
- Change:
  - nn keeps its own `TopLogprob`; core converts at the two places it builds response logprobs.
  - `AdapterGenerationId` stays in nn; the wire `AdapterGenerationSelection` carries the 64-hex ID as a `String`
    (its schema was already a string), parsed where the API converts it to the engine's selection.
  - `--docs` now documents only the crates whose files differ from the merge base with origin/master, where a
    workspace manifest or lockfile change means all of them (`scripts/doc_targets.py`, unit-tested under
    `--lint`); `--docs-all` documents every crate.

## Run 46 - 2026-09-28 (night)

- Change:
  - inference-api's `openai.rs` and `responses_types/` (with `TextConfig`/`TextFormat`) move into inference-protocol
    behind an `openai` feature.
  - The API converts the wire `AdapterSelection` in `lora_routing::core_adapter_selection`, since the orphan rule
    rules out the old `From` impl.
  - Run 45's fix: inference-nn no longer depends on inference-protocol.
- Command: the Run 39 cold build, now on an idle machine (load 1.5 at the start), plus
  `cargo llvm-lines -p inference-core` and `-p inference-api --lib --features cuda`.
- Result:
  - Wall 243 s (Run 39: 254 s; Run 26: 241 s); 2,245 unit-seconds (Run 39: 2,318).
  - inference-protocol lib t=76-85 s, its lib test t=76-89 s: still before inference-nn (t=94 s) and core (t=113 s).
  - inference-api's lib test t=205-241 s (36 s; Run 39: t=201-254 s, 53 s). It now finishes with the server-core
    test (t=201-237 s) and the CLI (t=211-243 s), no longer after them.
  - Core's lib test (t=131-243 s, 112 s) ends the build, as in every run since Run 26.
  - IR: core 3,486,102 lines, inference-api 1,372,394.
- Implication: the protocol split is done. The cold build's tail is now core's lib test alone; the next wall-time
  lever is still shrinking what core compiles (the loaders or the vision preprocessing, Run 40) or its tests.

## Run 47 - 2026-09-29

- Change: the multimodal input processors left core for their family crates (#122-#127; the log is in
  `multimodal_preprocessing_investigation.md`). Core also dropped `ordered-float`, `rand_distr`, `inference-vision` and
  `rubato`.
- Command: the Run 39 cold build on an idle machine (load 0.9 at the start), then
  `cargo llvm-lines -p <crate> --lib --features cuda`.
- Result:
  - Wall 237.6 s (Run 46: 243 s); 799 units (was 832), 2,238 unit-seconds (was 2,245).
  - The family crates build together from t=104 s and take 22-29 s each. Their lib tests end by t=155 s.
  - Core's lib takes t=113-187 s (73.5 s) and its lib test t=133-237 s (104.5 s; Run 46: 112 s). The lib test still
    ends the build; inference-api's lib test ends at 229 s and the CLI test at 237.5 s.
  - IR: core 3,025,589 lines (Run 46: 3,486,102, -13%); qwen 754,607, llama 716,135, gemma 695,276.
- Implication: moving the processors was worth about 7 s of core's lib test, and about 5 s of cold wall time. Core's
  lib test is still the tail, now tied with the CLI test binary. The next lever is still the size of core itself; the
  loaders are the largest part left.

## Run 48 - 2026-09-29

- Question, from watching local CI: why does rustdoc run for so long on a single core after the concurrent part of the
  run, cycling through crates one at a time?
- Commands:
  - `cargo doc --no-deps <targets> --timings`;
  - `/proc` sampling of the live rustdoc processes and the load average;
  - `cargo test --workspace --doc`, timed.
- Result:
  - The doctests are not it. The whole workspace's doctests take 15.7 s wall and 20 s CPU when warm.
    - `--merge-doctests yes`, which edition 2024 gives by default, would compile each crate's doctests as one
      binary. On this toolchain it is unstable, and there is little to win.
    - An earlier 98 s reading for `-p inference --doc` was a rebuild: `-p` alone resolves features differently.
  - `--docs` is. With the source changed, `scripts/local_ci.sh --docs` took 40.5 s wall and 81.6 s CPU.
    - 16 rustdoc processes were alive at once, but the load average stayed at about 2.
    - Their `%CPU` kept falling as they slept, so the runs were serialized.
    - An earlier run with `--timings` had 6 crates finishing one after another 15-35 s apart: 107 s wall for 94 s of
      CPU.
    - The cause is rustdoc's default cross-crate merge. Each crate reads and rewrites the shared files in
      `target/doc` (search index, implementor lists) under the doc root's lock.
  - Documenting all 26 crates into an empty `target/doc` took 21.3 s; into a full one (248 MB), 30.2 s.
  - `--merge none`, which skips the merge, is unstable.
  - `--emit dep-info` is stable. rustdoc still runs every doc lint (a planted broken intra-doc link fails with
    `unresolved link`), but writes no HTML, and so takes no lock. The whole workspace takes 16.7 s wall and 57 s
    CPU, with every crate in parallel.
- Change: `local_ci.sh --docs` passes `--emit dep-info`. Rendered docs come from `cargo doc`.
  - The same changed-source `--docs` run now takes 17.6 s wall (was 40.5 s).
  - A branch that touches `Cargo.lock` documents `--workspace`, which is now a 17 s pass instead of 30-107 s.

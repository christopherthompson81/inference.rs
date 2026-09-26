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

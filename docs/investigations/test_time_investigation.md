# Test time investigation

`scripts/local_ci.sh --lint --tests --cuda --bindings --size --sweep` takes about 10 minutes. Where does it go, what
grew, and what can bring it down?

## Run 1 - 2026-10-28 16:38

Question: which phases make up the 10 minutes, and which tests grew since 2026-10-26?

Command: nextest's per-test `PASS [ n s]` lines and the summaries of 17 saved `local_ci.sh` logs (2026-10-26 to
2026-10-28), comparing the oldest (`ci55`) and the newest (`ci_gs`); then three of the grown kernel tests alone:
`cargo nextest run --profile cuda --features cuda -p inference-quant -E '...' -j 1`.

Finding:
- Phases in the newest log, which run one after another apart from the overlaps the script sets up:
  - lint builds: 35 s + 1 min 26 s
  - CUDA test build: 26 s + 1 min 55 s
  - CUDA suite: 209 s, with the CPU suite (106 s), the bindings and the slim/docs checks running beside it
  - `--size` bundle build: 2 min 08 s
- The CPU suite holds at 95-120 s. The CUDA suite grew from 143-168 s (2026-10-26) to 168-234 s (2026-10-27/28),
  noisy from run to run; its summed per-test time grew from 2102 s to 2703 s for 129 more tests.
- Most of the growth is in old tests running slower, not in new tests: `marlin_matches_dequantized_q4k` 4.6 -> 12 s,
  `indexed_mmvq_matches_the_dequantized_experts` 3.1 -> 11.9 s, `marlin_matches_dequantized_all_affine_formats`
  6.3 -> 15.3 s, `image_edit_route::the_form_and_the_json_body_edit_alike` 14.5 -> 36.4 s. A few grew from real
  work: `qwen4exp_text_tiny::flash_next_isq_load...` 0.5 -> 20.8 s, `qwen3_5_text_tiny::an_isq_load...` 1.6 -> 16.5 s.
- Alone, the three kernel tests take 0.80, 0.75 and 0.74 s: 15-20x less than in the suite. The suite is contention
  bound. It runs 16 test processes (nextest's default, one per hardware thread) on one RTX 3090, while the CPU suite
  runs another 16 on the same 16 threads.

Implication: measure the CUDA suite alone, at the default parallelism and at fewer threads, to see whether GPU
contention or CPU oversubscription costs the wall time.

## Run 2 - 2026-10-28 16:45

Question: alone, how long does the CUDA suite take, and does fewer threads help it?

Command: `cargo nextest run --no-fail-fast --profile cuda --features cuda --workspace --lib --bins --tests
-E 'not (<the script's ALONE filter>)' -j N` under `/usr/bin/time`, N = 16 then 8, with nothing else running.

Finding:
- j16: summary 92.6 s (wall 112 s, the rest nextest listing the binaries); j8: summary 87.9 s (wall 88.6 s).
- In the full `local_ci.sh` run the same suite took 209 s: overlapping it with the CPU suite (106 s), the bindings
  and the slim/docs checks more than doubles it. Run back to back, the two suites would take about 194 s, so the
  overlap saves nothing.
- The CPU suite's wall time is its longest test: `hidream_o1_tiny::hidream_o1_dev_samples_its_fixed_schedule_only`
  at 102 s, with three more HiDream tests and two Kokoro ones at 63-94 s each.

Implication: two levers. Split the 16 threads between the overlapped suites instead of giving each all of them;
and shorten the CPU long poles (the HiDream and Kokoro tiny tests), which bound the CPU suite whatever its threads.

## Run 3 - 2026-10-28 16:55

Question: does splitting the threads between the overlapped suites help, and what bounds them when overlapped?

Command: the CUDA suite (`--profile cuda --features cuda`) in the background and the CPU suite in the foreground,
started together, at `-j 16` / `-j 16` and then `-j 8` / `-j 8`; per-test times compared with Run 2's lone `-j 8`.

Finding:
- 16 + 16: wall 168 s (CUDA 164.8 s, CPU 109.9 s). 8 + 8: wall 165 s (CUDA 161.6 s, CPU 115.9 s). The split changes
  nothing; and without the bindings and slim/docs checks the full script adds, the overlap is 168 s against
  ~198 s back to back, so it does save ~30 s.
- The CUDA suite is CPU bound: its summed test time is 590 s alone (`-j 8`) and 1756 s overlapped. Its slowest tests
  under overlap are host-heavy ones: `kokoro_tiny::kokoro_reads_english_text` 5.0 -> 52.2 s, `kokoro_speaks_phonemes`
  3.5 -> 37.6 s, `parakeet_tiny::each_head_repeats...` 13.1 -> 25.4 s, `hidream_o1_tiny::...dev_samples...` 7.3 ->
  24.6 s, `local_attention_tiny::llama4_decodes_alike_on_cpu_and_gpu` 3.4 -> 22.0 s.
- So the phase costs about the work of both suites over 16 threads (~1244 + 590 core-seconds, ~115 s ideal) plus
  the long poles' tail. The thread split cannot reduce the work.
- A flaky test under load: in the 8 + 8 run `qwen4exp_text_tiny::flash_next_autotuned_mtp_varies_its_depth_and_keeps_
  greedy_output` failed with "every step drafted one depth: 90 draft tokens over 45 verified sequences". The
  autotuner picks its depth from timings, which contention flattens.

Implication: cut the work, starting with the CPU suite's heaviest tests (HiDream ~380 core-seconds over five tests,
Kokoro ~190 over three); check what each spends it on. Make the autotuned-MTP test independent of timing.

## Run 4 - 2026-10-28 17:10

Question: is per-process thread-pool oversubscription the cost? Each test process sizes its compute pools to the
physical cores (8 here), so 16-32 concurrent test processes run ~256 compute threads on 16 hardware threads, and
the barrier pool's completion waits stall on any worker the scheduler took off its core.

Command: Run 3's overlapped suites with `RAYON_NUM_THREADS=2`, then `=1` (the barrier pool follows it), alone first:
`cargo nextest run -p inference -E 'test(/^hidream_o1_tiny::/) | test(/^kokoro_tiny::/)' -j 1`.

Finding:
- Alone, with full pools: `kokoro_reads_english_text` 3.5 s (63 s in the suite), HiDream dev 31.1 s (102 s), HiDream
  edit 22.8 s (80 s); the ten tests 95.7 s in all.
- Overlapped, 2 threads a process: wall 144 s (CUDA 135.5 s, CPU 106.8 s), against 168 s with full pools.
- 1 thread a process: wall 164 s (CUDA 162.9 s, CPU 143.2 s): the long poles (the HiDream tests, alone 7-31 s
  with 8 threads) then run single threaded and set the wall.
- Oversubscription costs ~25 s of the phase, but no fixed per-process size wins both ways: small pools starve the
  long poles once the short tests are done, large ones oversubscribe while they run.

Implication: size each op's parallelism from what the machine has free, not per process. Plan (the user's
direction): a machine-wide lease of worker slots that every process's pools take from per parallel op, growing
again on later ops when slots come back.

## Run 5 - 2026-10-28 17:15

Question: which pools would a machine-wide lease have to govern, and can rayon's take part?

Command: a survey of `ThreadPoolBuilder`, `BarrierPool`, `par_iter` / `rayon::scope` and `gemm::Parallelism` uses in
inference-tensor, inference-nn and inference-quant; rayon-core 1.13's builder API.

Finding:
- A process starts four compute pools, each sized to the physical cores (8 here): rayon's global pool
  (`init_global_threadpool`), `candle_pool` (a second rayon pool, `with_threadpool`), the `BarrierPool` (7 workers,
  spinning 25k iterations before parking) and the CPU attention backend's rayon pool. That is ~31 compute threads
  a process before anything leases.
- gemm takes its parallelism per call (`Parallelism::Rayon(n)`, n from `get_num_threads()`), and so do the barrier
  pool and the chunked CPU kernels: those can size each op from a lease.
- rayon-core 1.13 has only `start_handler` / `exit_handler`; the acquire/release hooks once meant for a jobserver
  are gone, so a rayon pool cannot lease per wake-up. Its `par_iter` sites (spread over many crates, mostly cold
  paths) run at the pool's fixed width.

Implication: lease at the op for everything that takes an explicit thread count, and fold the extra rayon pools
into fewer, narrower ones for the rest. Open choices for the user before building it (slot mechanism, lease
granularity, how far to retire rayon).

## Run 6 - 2026-10-28 17:24

Question: with the machine-wide lease (user's choices: lock-file slots, held and refreshed, the hot paths leased),
how long do the overlapped suites take, and does anything change its results?

Command: `cpu_lease` in inference-tensor (one `flock`ed file per slot under `/dev/shm/inference-rs-cpu-slots-<uid>`,
physical cores less one slots; a short process tries for more every 2 ms, an idle one gives its slots back after 20 ms,
a holder gives them back once every 50 ms), the barrier pool running each op on the caller plus the leased workers,
and first gemm's thread count leased too; Run 3's overlapped suites; the Kokoro and HiDream tests alone.

Finding:
- First version (barrier pool and gemm leased, chunks sized by the leased count): wall 123 s (CUDA 115.6 s, CPU
  97.6 s), but four Kokoro tests failed in each suite, and one even alone: they compare two generations' bytes, which
  now differed late in the audio.
- Two causes. The barrier pool's chunks followed the leased count, so a kernel that reduces per range summed in
  another order when the lease changed between calls; and gemm's own result follows its thread count (with gemm back
  on the fixed width and the lease on, the five Kokoro tests pass; with `INFERENCE_CPU_SLOTS=0` they pass too).
- Fixed: the chunk boundaries follow the pool's width and only the number of threads pulling them follows the lease
  (a unit test checks the ranges at every worker count); gemm and the layout conv through it keep the fixed width.
- Then: wall 121 s (CUDA 118.4 s, CPU 96.1 s), every test passing, against 168 s with per-process pools (Run 3)
  and 144 s with a fixed cap of 2 (Run 4). The Kokoro and HiDream tests together: 57.7 s.

Implication: the barrier pool alone recovers the overlap's cost; gemm's rayon pools stay unleased (making them
lease would need gemm's split to stop following its thread count). Next: the full `local_ci.sh` time, then the
remaining long poles (the HiDream tiny tests run 4.2 MP images).

## Run 7 - 2026-10-28 17:35

Question: what does the whole `local_ci.sh` run take with the lease?

Command: `/usr/bin/time scripts/local_ci.sh --lint --tests --cuda --bindings --size --sweep` (after an edit to
inference-tensor, so the lint, test and bundle builds all rebuild from it).

Finding:
- Wall 442.6 s (7 min 23 s), every check passing, against about 10 minutes before (Run 1).
- CUDA suite 125.7 s (209.4 s in Run 1's log), CPU suite 94.5 s (106.4 s); lint builds 37.8 s + 11.2 s and
  33.6 s + 15.0 s; CUDA test build 33.4 s; the `--size` bundle build 2 min 25 s, now the largest single step.

Implication: the suites are no longer the bulk. Left: the bundle build (`--size`, needed only for kernel or
dependency changes), and the long-pole tests (the HiDream tiny tests make 4.2 MP images).

## Run 8 - 2026-10-28 17:56

Question: does the `--size` bundle build (2 min 25 s, last and alone in Run 7) fit beside the suites?

Command: `local_ci.sh` starting `cargo build --profile bundle -p inference-ffi --features cuda` in the background once
the CPU test binaries are built (its own `target/bundle` lock, so no cargo lock is shared), and checking the size at
the end; the same flags as Run 7, after touching `crates/inference-tensor/src/utils.rs`.

Finding:
- Wall 369.4 s (6 min 9 s), every check passing, against 442.6 s.
- The bundle build took 3 min 31 s beside the suites (its crate compiles share the cores before the single-core fat
  LTO), and slowed them: CPU suite 140.5 s (94.5 s), CUDA suite 179.6 s (125.7 s). The run still saves 73 s.

Implication: keep it. Left: the HiDream tiny tests, the slowest of the CPU suite (31 s, 23 s alone), which make
4.2 MP images: 3600-4000 patch tokens through quadratic attention at every step.

## Run 9 - 2026-10-28 18:11

Question: can the HiDream tiny tests (the CPU suite's long poles, 41-96 s each beside the CUDA suite in Run 6) stop
making 4.2 MP images, without changing what a real checkpoint does?

Command: an `image_edge` config field (default the trained 2048, which every real checkpoint keeps, since none sets
it) that the size table and the matched-reference size scale from, rounded up to whole 32 px patches; the tiny
fixture sets 256. `cargo nextest run -p inference -p inference-server-core -E 'test(/^hidream_o1_tiny::/) |
test(/^image_edit_route::/)'`.

Finding:
- Why they were slow: at 2560x1440 a tiny model (hidden 128, 2 layers) still sees 80 x 45 = 3600 patch tokens through
  quadratic attention at every step (Dev's fixed 28, twice); an edit at 2336x1760 about 4000 plus the reference.
- At edge 256: generation snaps to 320x192 and a 640x480 reference matches at 288x192. The seven tests take
  0.06-0.64 s each (7-31 s alone before), all passing. At the trained edge the scaled table is the table itself (a
  unit test checks each entry), so real-checkpoint sizes and the parity script are unchanged.

Implication: the CPU suite's long poles are gone; measure the whole run again.

## Run 10 - 2026-10-28 18:16

Question: what does the whole `local_ci.sh` run take now, with the lease, the overlapped bundle build and the small
HiDream images?

Command: `/usr/bin/time scripts/local_ci.sh --lint --tests --cuda --bindings --size --sweep`, the same flags as
Runs 7 and 8.

Finding:
- Wall 284.4 s (4 min 44 s), every check passing: against about 10 minutes (Run 1), 442.6 s (Run 7) and 369.4 s
  (Run 8).
- CPU suite 24.2 s (140.5 s in Run 8, 94.5 s in Run 7); CUDA suite 101.6 s (179.6 s, 125.7 s); the bundle build
  2 min 09 s beside them.

Implication: the CPU suite no longer sets the pace; what remains is the builds (lint, test and bundle, after an edit
low in the stack) and the CUDA suite, whose slowest tests are now Parakeet and Kokoro on the GPU.

## Run 11 - 2026-10-29 10:44

Question: where does the full local CI stand after the Granite and OLMo ports? The OLMo PR's run had the CUDA suite at
140.2 s against Run 10's 101.6 s.

Command: `/usr/bin/time scripts/local_ci.sh --lint --tests --cuda --bindings --size --sweep`, the same flags as
Run 10, then `scripts/housekeeping.sh --slow-tests` on the log (and on the OLMo PR's log).

Finding:
- Wall 312.0 s (Run 10: 284.4 s), every check passing. CPU suite 22.8 s (OLMo run 20.2 s, Run 10 24.2 s); CUDA suite
  126.8 s (OLMo run 140.2 s, Run 10 101.6 s). Two runs of the same tree differ by 13 s on the CUDA suite, so part of
  the growth is noise; the rest is ~25 s more than Run 10.
- CPU: Kokoro tiny is 67.2 s of test time in 5 tests (repeats-exactly 22.4 s, reads-English 18.9 s, phonemes 14.4 s,
  GGUF 7.4 s), then Kokoro's g2p unit tests 15.4 s (21), Parakeet tiny 13.1 s (11).
- CUDA, slowest tests: Parakeet each-head-repeats 25.7 s, qwen4exp DFlash / MTP / autotuned MTP / sparse prefix 9.4-12.2
  s each, llama4 CPU vs GPU 10.4 s, GDN speculative commit 10.1 s, doclayout batched 9.3 s, Qwen2-VL prefix cache and
  Qwen3.5 (MoE) MTP 8.7-8.9 s.
- CUDA, slowest modules by summed time: inference-nn `ops` 146.7 s (40 tests), `cuda::gdn` 127.9 s (26),
  `qwen3_5_text_tiny` 112.9 s (26), `qwen_vl_tiny` 96.2 s (14), paged attention `fattn_tests` 86.3 s (17),
  quant `gguf::packed_affine` 74.5 s (18), `qwen4exp_text_tiny` 71.9 s (10), fattn parity 68.7 s (17).

Implication: the CUDA suite's wall is set by its long single tests (Parakeet at 25.7 s alone is a fifth of it) more
than by its sum; Parakeet's repeat test, Kokoro's three long CPU tests and the qwen4exp speculative tests are the
first to look at. Run 10 kept no per-test times, so the ~25 s is not attributed to specific tests; this run's lists
are the baseline for the next round.

## Run 12 - 2026-10-29 20:38

Question: where does test time stand against Run 11, across the five full local CI runs of the #702-#708 PRs?

Command: `scripts/housekeeping.sh --slow-tests` on each run's `scripts/local_ci.sh --lint --tests --cuda --bindings
--size --sweep` log (no `/usr/bin/time` wall for these).

Finding:
- CUDA suite wall / summed test time per run: Responses 101.9 s / 1452 s, devices 97.9 / 1391, jobs 98.4 / 1380,
  media 98.5 / 1383, decision formats 137.2 / 1956 (Run 11: 126.8 s). The last run is 40% slower in every test alike
  (Parakeet repeat 28.0 s against 21.9-23.1, `qwen3_5_text_tiny` 163 s against 100-134), so it was load on the machine,
  not the change; the steady CUDA wall is ~98-102 s.
- CPU suite: 21.0 s for 3005 tests (Run 11 22.8 s). Kokoro tiny 61.2 s in 5 tests (repeats-exactly 20.7, reads-English
  16.3, phonemes 13.1, GGUF 7.2), Kokoro g2p unit tests 15.8 s (21), Parakeet tiny 13.2 s (11).
- CUDA, steady runs: Parakeet each-head-repeats 21.9-23.1 s alone (a quarter of the wall), llama4 CPU vs GPU 8.3-13.5
  s, the qwen4exp speculative tests ~10 s each; modules `qwen3_5_text_tiny` 100-134 s (26), inference-nn `ops` and
  `cuda::gdn` ~140-160 s each.

Implication: the CUDA wall is set by Parakeet's repeat test: cutting its repeats or its checkpoint size is the first
test-time item, then Kokoro's three long CPU tests. A run on a loaded machine is not comparable; recheck a slow
outlier against the neighbouring runs before attributing it.

## Run 13 - 2026-10-29 22:12

Question: Run 12's first test-time items: Parakeet's long repeat test (22-28 s of the CUDA suite) and Kokoro's CPU
tests (13-21 s each in the CPU suite). What do they spend, and what can be cut without losing a check?

Commands: `cargo nextest run --profile cuda --features cuda -p inference -E 'test(/parakeet_tiny::each_head_repeats/) |
test(/kokoro_tiny::/)'` (alone); the same on CPU; per-head timing printed from the Parakeet test (reverted); `perf
record -g` of `kokoro_is_detected_from_its_config`; the CPU suite (`cargo nextest run --workspace --lib --bins
--tests`) under `CANDLE_BARRIER_POOL_SPIN_LIMIT` default / 0 / 2000, and with a nextest `priority` override.

Finding:
- Alone, the Parakeet repeat test takes 11.0 s on CUDA and 3.0 s on CPU; in CI 22-28 s is contention. Per head on
  CUDA (load, then two 45 s transcriptions): TDT 0.74 s, RNN-T 2.51 s, CTC 0.31 s, streaming 2.53 s, prompted streaming
  3.22 s; on CPU 0.34 / 0.50 / 0.33 / 0.53 / 1.22 s. The three heads that decode with an RNN-T greedy loop take ~1.2 s
  a transcription on the GPU against ~0.25 s on the CPU: each step reads the argmax back to the host (`to_scalar`),
  ~1.5 ms a step over ~800 steps. A product cost too (an hour of audio is ~45k frames); left as a follow-up.
- The test becomes one test per head (a const assert keeps the five in step with `HEADS`), so they run side by side:
  alone on CPU the longest is 1.8 s (prompted streaming).
- Kokoro alone on CPU: is-detected 1.8 s, GGUF 2.9, phonemes 3.8, English text 5.6, repeats 7.2. Profile of the
  shortest: 24% idle pool workers spinning (`BarrierPool`), ~18% elementwise Mul/Sqr/Div (instance norms and the sine
  source over 24 kHz samples; weight norm is already folded at load), 8.7% gemm, 7% `regress` regex compilation
  (vernacula's data load, which Kokoro's loader warms once per process on purpose), ~7% the fixture's random weights.
  The inputs were larger than their checks need: the 30-word phoneme string split into ~7 chunks of the tiny 64-token
  context, and the long English text was 20 copies of a 44-phoneme sentence (880) to pass the 508-phoneme chunk limit.
  Now 10 words (2-3 chunks) and 14 copies (616). Alone: 1.7 / 2.7 / 2.7 / 4.1 / 3.4 s, sum 21.2 -> 14.7 s, longest
  7.2 -> 4.1 s.
- The Kokoro g2p unit tests (21) finish in 1.0 s wall alone; each pays the phonemizer's data load in its own process
  (0.4-1.0 s), which is their 15.8 s summed in CI. Not changed: the load is vernacula's.
- Spin limit (negative): CPU suite wall 10.1 / 9.2 s at the default, 9.7 / 9.1 s at 0, 9.0 s at 2000; noise.
- Priority (negative): starting the speech tests first gave 9.2-9.3 s; the Kokoro tests take ~8 s inside the suite
  wherever they start. Reverted.
- The CUDA suite is throughput-bound, not tail-bound: its last tests to finish are 1-1.5 s tensor tests, and the long
  ones (llama4 CPU vs GPU 20.6 s, qwen4exp 5-8 s) finish within the first 160 of 3448. So splitting Parakeet shortens
  that test, not the suite's wall, which follows summed test time (1,380-1,950 s); the CPU suite runs beside it.

- Full local CI with the change (`/usr/bin/time scripts/local_ci.sh --lint --tests --cuda --bindings --size --sweep`, a
  warm build): wall 190.9 s. CPU suite 15.8 s (Run 12: 18-23 s), Kokoro tiny 46.4 s summed in 5 tests (Run 12: 61-67 s),
  longest 15.4 s (20.7 s); CUDA suite 96.2 s, its longest Parakeet test 11.4 s (one test of 22-28 s before).

Implication: done for these tests. The suite-level levers are summed CUDA test time (inference-nn `ops` and
`cuda::gdn`, `qwen3_5_text_tiny`) and the RNN-T greedy loop's per-step host sync.

## Run 14 - 2026-10-30 06:17

Question: Run 13 put the RNN-T greedy loop's GPU cost on the per-step argmax readback (`to_scalar`, ~1.5 ms a step
over ~800 steps). Is that right, and what does a step cost?

Command: timers around the joint, the readback and the predictor advance in `Transducer::greedy_from` (reverted),
`cargo nextest run --profile cuda --features cuda -p inference -E 'test(/parakeet_tiny::(rnnt|tdt)_repeats/)'
--no-capture`.

Finding:
- Run 13's diagnosis was wrong. A 45 s transcription with the tiny random RNN-T head takes 5,630 steps, not ~800: its
  random weights emit a token at almost every step, up to `max_symbols_per_step` a frame, so each step also advances
  the predictor. Per transcription: total 1.14 s, joint 0.19 s (33 us a step), readback 0.04 s (7 us), predictor
  0.90 s (160 us an advance). TDT: 282 steps, 0.06 s. The sync is cheap; the predictor's launches are the cost.
- Two bit-exact cuts in the advance: a single token reads its embedding row as a view (`narrow`) instead of uploading
  the id and gathering, and one sigmoid over all four gates replaces three over their slices. Predictor 0.90 -> 0.72 s
  (128 us an advance), RNN-T test 3.0 -> 2.6 s alone. All 27 Parakeet tests pass on CUDA and CPU (pins unchanged).
- Then the gate matmuls: one matmul of `[x, h, 1]` against `[w_ih; w_hh; b_ih + b_hh]` (transposed, stacked at load)
  replaces two matmuls, their add and the bias add. Not bit-exact (one sum over the stacked row), so checked on real
  checkpoints: `scripts/parakeet_parity.sh` on CUDA, F32, against transformers.
  - parakeet-tdt-0.6b-v3 on the 8 clips of the #414 sweep (3 LibriSpeech, the 97.6 s demo, two AMI and two
    VoxConverse 300 s; the kept reference): emissions 15/15, 16/16, 17/17, 619/619, 1207/1207, 714/714, 1160/1160,
    1676/1676, WER 0.000% over 2928 words.
  - nemotron-speech-streaming-en-0.6b (RNN-T), first 4 clips, fresh reference: 18/18, 16/16, 21/21, 639/639.
  - nemotron-3.5-asr-streaming-0.6b (RNN-T, prompted), same clips: 28/28, 23/23, 27/27, 774/774.
  - All 27 Parakeet tests pass on CUDA and CPU; no tiny pin moved.
  Predictor 0.72 -> 0.64 s on CUDA (113 us an advance); on CPU an advance is ~10 us, launch-free.

Implication: the RNN-T loop is launch-bound on token-dense output, not sync-bound. Real checkpoints emit far fewer
tokens than frames, so the product cost there is the per-token advance; what is left is ~9 launches per layer (the
`[x, h, 1]` concat, the matmul, sigmoid, tanh and the five-op cell update) and the projector. A fused LSTM cell kernel
is the remaining lever if speech decode ever shows up in a profile.

## Run 15 - 2026-10-30 07:06

Question: Run 13 left the CUDA suite's summed test time (inference-nn `ops` 95.6 s over 40 tests, `cuda::gdn` 87.9 s
over 26, `qwen3_5_text_tiny` 94.8 s over 26) as the suite-level lever. What does a GPU test spend?

Commands: `scripts/housekeeping.sh --slow-tests` on the last full CI log; the `ops` and `gdn` CUDA tests alone; `nsys
profile -t cuda,osrt` and `/usr/bin/time` of one test binary run (`ops::tests::cuda_add_rms_norm_matches_separate_ops`);
1, 4, 8 and 16 concurrent copies of it; a 10-line C program (`cuInit`, primary context, one `cuMemAlloc`) the same
way; that program under an MPS daemon; the CUDA suite alone under `/usr/bin/time`.

Finding:
- In CI the `ops` CUDA tests take 2.7-3.4 s each, all started together (positions 2120-2135). Alone, one takes 0.35 s;
  run as a group of 69 (`ops` + `gdn`), most take ~3 s. The tests' own CUDA API time is ~25 ms (module loads 4.4 ms,
  allocations 9.4 ms); the binary starts in under 10 ms (`--list` 0.00 s). The rest is CUDA context creation.
- Context creation is serialized in the driver. Ours: 1 / 4 / 8 / 16 concurrent copies in 0.36 / 0.65 / 1.19 / 2.42 s
  (~0.15 s a process). The bare C program: 0.14 s alone (cuInit 0.036 s, context and first allocation 0.09-0.14 s),
  16 in 1.97 s (~0.12 s a process). So ~0.12 s a GPU process is the driver's (RTX 3090, persistence mode on), and our
  startup adds ~0.03 s of serialized time.
- The CUDA suite alone: nextest 75.9 s, `/usr/bin/time` user 285 s, sys 285 s, so ~7.5 of 16 cores busy and half of
  it kernel time (driver ioctls). 645 of its 3452 tests take >= 0.15 s (the rest are CPU tests or skips under 0.15 s);
  645 x 0.12 s = 77 s. The suite's wall is the serialized context creation of its GPU processes; summed test time is
  that queue, counted once in each waiting test.
- MPS (negative): `nvidia-cuda-mps-control -d` with user pipe and log directories fails to start ("Failed to
  initialize MPS control daemon"); it would also be a machine setup change. Dropped.

Implication: the CUDA suite's wall falls only with fewer GPU processes, ~0.12 s each. Speeding a test's body does not
move it, and splitting a GPU test (Run 13's Parakeet split) adds a process. GPU processes (>= 0.15 s) by package:
inference-nn 153, inference-quant 95, server-core 38, inference-fattn 36, inference-ffi 21, then model crates. The
candidates are the families of small kernel tests, counted as GPU processes: inference-nn `ops` (34), `cuda::gdn`
(26), paged attention layers (19), `speculative_rejection` (11), `moe::experts` (10); inference-quant `utils::ops`
(17), `packed_affine` (15), `raw` (9); inference-fattn `parity` (17), `paged` (9), `fp8` (5), `varlen` (5). Merging
each family into one process that runs every case and reports every failing one would cut ~165 processes, ~20 s.

Follow-up, the merge: `inference_tensor::gpu_cases!(cuda_cases, [...])` turns a module's GPU tests into cases of one
test, which runs each (a panic fails only its case) and fails naming every failing one; `INFERENCE_TEST_CASE` narrows
it to matching cases. The 12 families above: 173 cases in 13 processes (`gguf::raw` keeps its device-generic pair
apart from its CUDA-gated cases). An injected panic in one `ops` case reported that case and ran the rest.
- CUDA suite alone: 3452 -> 3292 tests, nextest 75.9 -> 51.7 / 51.8 s, user + sys 570 -> 455 s.
- Full local CI (`--lint --tests --cuda --bindings --size --sweep`): the first run after the change rebuilt every
  crate above inference-tensor (wall 401 s, CUDA suite 103.8 s beside the builds); warm, wall 107.5 s with the CUDA
  suite at 62.0 s and the CPU suite at 11.6 s (Run 13: 190.9 s, 96.2 s, 15.8 s).

## Run 16 - 2026-10-30 10:25

Question: after Run 15 (merged kernel families, CUDA suite 51.7 s alone), which test processes really open the GPU,
what do they cost alone, and what does the suite leave idle? The user noted VRAM sits mostly unused.

Commands: the CUDA suite through a cargo target runner (`CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER`) that runs each
test under `strace -f --seccomp-bpf -e trace=openat` and records whether it opened `/dev/nvidiactl` or
`/dev/nvidia0` / `nvidia-uvm`; the suite again with `--test-threads 1` for solo times; the suite a third time with
`nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw -lms 200` sampling.

Finding:
- 480 of the 3,294 test processes open the GPU device; none initializes the driver without opening it (so no CPU test
  pays for a needless probe).
- Serially the suite takes 185.7 s: the 480 GPU tests 148.5 s, the CPU tests 37.0 s. In parallel it takes 52.7-62.7 s,
  ~3x on 16 threads.
- Solo, the GPU tests split cleanly: 245 take >= 0.15 s (sum 138.1 s; a real context, ~0.55 s mean), 235 take under
  0.1 s (median 0.02-0.06 s; they open the device without the context cost). By module (count, solo sum):
  `qwen3_5_text_tiny` 26 / 14.5 s, `llama_tiny` 15 / 6.8 s, `qwen_vl_tiny` 14 / 7.6 s, `routed_lora_cuda_tests`
  11 / 5.0 s, `qwen4exp_text_tiny` 10 / 6.5 s, `cuda_graph::tests` 10 / 1.9 s, `gemma3_tiny` 8 / 4.0 s,
  `parakeet_tiny` 5 / 9.9 s; the cheap ones `engine_abi` 21 / 2.2 s, `chat_route` 18 / 1.0 s,
  `bert_embedding_tiny` 17 / 1.3 s.
- During the parallel suite (53.6 s): GPU utilization mean 25% (>= 90% in 9% of samples), VRAM mean 3.9 GB of 24 GB
  (idle 1.7 GB; peak 20.1 GB, the card-filling eviction test, which already runs alone), power mean 163 W. With the CPU
  at ~7.5 of 16 cores (Run 15), every resource is under half used while all 16 slots stay occupied: each GPU test
  spends most of its ~5x-solo time queued behind the others' context creation. 245 real contexts x ~0.12 s = ~29 s
  of the wall.

Implication: more processes in parallel cannot help (the queue is in the driver), and `gpu_cases!` serializes its
cases, which is right for millisecond kernel tests but would lengthen the model modules (seconds each). The lever is
the user's: run a module's tests concurrently inside one process, sharing its context, with a VRAM budget gating how
many run at once. Kernel tests that hold `CUDA_TEST_LOCK` (process-global pools and graph scopes) stay serial, and
tests that read the device's free memory keep running alone.

## Run 17 - 2026-10-30 10:32

Question: Run 16's plan, piloted on `qwen3_5_text_tiny`: can a module's model tests run as concurrent cases of one
process under a VRAM budget?

Commands: per-test peak VRAM, the module run serially through a target runner logging each test's pid beside
`nvidia-smi --query-compute-apps=pid,used_memory -lms 50`; a `concurrent_cases!` test (tokio multi-thread runtime,
each case a spawned task claiming its measured MiB from a 4096 MiB semaphore) over the module's 20 tests that touch no
process-global state (the graph-replay rounds, the verify-graph and drafter tests that install a metrics recorder, and
the one that sets an environment variable stay separate processes); then the same with
`INFERENCE_RS_CUDA_GRAPHS=0`.

Finding:
- Peak VRAM per test process, alone: 348-620 MiB on the GPU, 0 for the five CPU-only tests. That includes the
  context, kernel modules and library workspaces, which cases sharing a process pay once.
- Concurrent, graphs on: 9 of the 20 cases fail, either at load (`DriverError(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
  "operation not permitted when stream is capturing")`) or with "Internal server error" mid-request. Engines capture
  in `CU_STREAM_CAPTURE_MODE_RELAXED` on `per_thread_stream()`, a per-thread default stream, which still orders
  against the legacy default stream: a synchronous copy or free on another thread (another engine loading) touches
  the capturing stream.
- Concurrent, graphs off: all 20 pass, the whole test in 1.60 s (the cases alone, serially, as separate processes:
  14.5 s).

Implication: running cases side by side works and pays well, and it found what the one-test-per-process layout hid:
two engines in one process (the server's model manager hosts several, loading them on request) break each other
while one captures a CUDA graph. That is a product question before a test one.

## Run 18 - 2026-10-30 11:01

Question: Run 17's failure with graphs on: which work on one engine's thread breaks another engine's capture in the
same process, and what fix holds?

Commands: backtraces at every CUDA error carrying a capture code (temporary prints in inference-tensor's error
conversions); a two-thread probe in `graph_capture`'s tests (thread A holds a relaxed capture open, thread B runs one
step: device creation, allocation, copy, own-stream sync, cuBLAS matmul, a first-use kernel, flash attention, a
device drop, a pinned free, a cuBLAS-only device drop, a graph destroy); a log of captures begun without the shared
scope (temporary, in cudarc's `begin_capture`); the pilot's concurrent test five times after each step.

Finding:
- First errors: a load's footprint measurement (`device_used_bytes`: context synchronize, pool trim) and its memory
  pool retention (pool attribute change) fail with `STREAM_CAPTURE_UNSUPPORTED` during another engine's capture,
  and invalidate it.
- The probe: allocation, copies, own-stream sync, cuBLAS, a kernel's first use, flash attention and freeing pinned
  memory on B leave A's capture intact. Dropping a device with loaded kernel modules (`cuModuleUnload`) and dropping
  a cuBLAS-only device (`cublasDestroy`, whose synchronizing free succeeds) invalidate it silently: cudarc records
  the unload's error in the context, and the destroy reports success. An invalidated capture left open then fails
  every later context-wide call in the process, so one silent invalidation cascades.
- Fix: a process-wide lock in the vendored cudarc (`driver::safe::capture_lock`): captures hold it shared (the three
  production sites: core's decode graph, Qwen3.5's MTP graph, DFlash's), context-wide calls exclusively (context
  synchronize in core and `MemoryUsage`, pool trim and high-water reset, pool attribute changes in the graph pool
  guards and retention, `cuDeviceGraphMemTrim`, peer access setup), and cudarc's `CudaModule`, cuBLAS, cuBLASLt,
  cuRAND and cuDNN drops take it too. A thread inside its own capture takes no exclusive lock (it would deadlock;
  CUDA reports the call's error as before), and a nested capture on one thread takes no second shared lock.
- No capture began without the shared scope. The pilot's 20 concurrent cases, graphs on: 20/20 in each of 5 runs,
  1.58-1.88 s. A regression test (`context_wide_work_on_another_thread_waits_for_an_open_capture`) passes with the
  lock and fails without it (`STREAM_CAPTURE_UNSUPPORTED`).

Implication: several engines in one process (the server's model manager) no longer break each other's graph captures;
model test modules can run their cases concurrently in one process with graphs on.

## Run 19 - 2026-10-30 11:28

Question: with Run 18's fix, how far do concurrent cases go across the model test modules, and what do they save?

Commands: per-test peak VRAM for 14 modules, serially (the target-runner pid log beside `nvidia-smi
--query-compute-apps`); each module's tests that touch no process-global state, carry no `cfg`/`ignore`
attribute and ran under CUDA moved into its `concurrent_cases!`; the shared tests on CUDA (3 runs, then 6) and CPU;
the CUDA suite alone twice and the CPU suite.

Finding:
- Peaks: the card-filling eviction test 22,218 MiB (kept apart, as are the memory-report test, the graph-replay
  rounds and the recorder tests), every shared test 610 MiB or less.
- 14 modules share 133 tests in 14 processes: `qwen3_5_text_tiny` 20, `bert_embedding_tiny` 17, `parakeet_tiny`
  14, `qwen_vl_tiny` 14, `gguf_embedding_tiny` 13, `llama_tiny` 12, `rope_encoder_embedding_tiny` 10,
  `qwen4exp_text_tiny` 6, `gemma3_tiny` 5, `paddleocr_vl_tiny` 5, `hidream_o1_tiny` 5, `embedding_tiny` 4,
  `decision_tiny` 4, `granite_tiny` 3. Left as they were: `qwen3_sliding_window_tiny` (its futures are not
  `Send`, so they cannot be spawned) and one gemma3 test gated on `cfg(cuda|metal)`.
- First runs: 1 of 3 aborted in `fattn-common.cuh:1310`, `GGML_ASSERT(max_blocks_per_sm > 0)` (the occupancy
  query found the kernel's shared memory over its limit). The mma f16 path raises each kernel's limit through a
  per-instantiation `static size_t` that two threads read and write unsynchronized; a smaller request landing after
  a larger one lowers the attribute under the larger launch. The quant decode path had the same raise and an occupancy
  cache whose count and size another thread could pair wrongly. Now one serialized, never-lowering
  `ggml_cuda_raise_shared_memory_limit` for the four f16 sites and the quant one, and the occupancy cache under a
  mutex. Then 6 of 6 runs pass, 8.6-9.2 s for all 14 shared tests together (`parakeet_tiny` the longest).
- CUDA suite alone: 3292 -> 3177 tests, 51.7 -> 45.2 / 43.7 s. CPU suite 2899 tests, 10.2 s (Run 15: 11.6 s).

Implication: ~7 s off the CUDA suite, less than Run 16's ~29 s of context queue would allow: many shared tests were
the cheap kind that never paid for a context. The two bugs found are product bugs for any process running several
engines (the server's model manager). Next lever if wanted: the server-core route tests (`chat_route` 18, `keyed`
10) and the C ABI's `engine_abi` 21, each a model per test.

## Run 20 - 2026-10-30 11:54

Question: the full local CI after Run 19 failed once: `jobs::a_device_job_evicts_the_idle_model_and_holds_the_device`
passed and its process died with SIGSEGV at exit. Did this branch's drop-time locks (cudarc's module, cuBLAS,
cuBLASLt, cuRAND and cuDNN drops now waiting out open captures) cause it?

Commands: the test alone 6 times; `--stress-count 20` of it, then of every `jobs::` test, with 16 busy-loop shells
(the suite's CPU load); the binary under `gdb` in a loop with that load (40 iterations); then an A/B, 5 x 20
iterations of the five jobs tests under the same load, A the branch, B with only the drop-time locks removed.

Finding:
- Alone: 6/6 pass (2.2 s). Under load: the one test 20/20; all five, 1 SIGSEGV in 9 iterations. Under gdb: none in
  40 (gdb's timing hides it, as in job_scheduler Run 6).
- A/B: A 3 crashes in 100 iterations (500 test processes), B 2 in 100. All five at exit after a pass:
  `a_sized_job_runs_beside_the_resident_model` 4, `a_job_that_may_not_displace_waits_for_the_device_to_empty` 1.

Implication: not this branch; an existing exit-time race in the jobs tests, the class job_scheduler Run 6 fixed for
`unload_model` (an engine thread still tearing down while the process exits), apparently still open on a job path.

## Run 21 - 2026-10-30 12:23

Question: after Runs 19-20, which GPU test processes still pay a real context, and what does merging them save?

Commands: Run 16's traced and serial logs, minus the modules converted since; the families left merged into
`gpu_cases!` with the merge script (each case's `#[test]` dropped, one invocation per `cfg` group); the merged tests on
CUDA; the CUDA suite alone twice.

Finding:
- 169 processes still paid a context (solo >= 0.15 s, ~20 s of queue). The server-core route tests and the C ABI's
  engine tests the plan named are the cheap kind (`chat_route` 18 in 1.0 s solo, `keyed` 10 in 0.7 s,
  `engine_abi` 2 of 21 over 0.15 s), so the target became the kernel families: `routed_lora_cuda_tests` 11,
  `fast_mmq` 6, `hqq` 6, core `cuda_graph` 5, `attention` 5, `blockwise_fp8::ops` 5 (two `cfg` groups),
  `qsa` 4, `conv` 4, `cutlass_moe_tests` 4, `indexed_moe_lora_cuda_tests` 4, `layout::layers` 3,
  `gguf` 3, `unquantized` 3: 59 cases.
- Kept separate: tests that measure process-global state (the allocator-baseline tests in core and nn, the graph pool
  threshold and retention tests) and `cuda::preload` (module loading itself); `graph_capture` had one left, not
  worth a merge.
- The merged tests: 27 pass, 236 cases ok. CUDA suite alone: 3177 -> 3128 tests, 44.5 -> 36.5 / 36.2 s.

Implication: since Run 15 (75.9 s), 36 s. The rest of the queue is spread thin (families of 1-3).

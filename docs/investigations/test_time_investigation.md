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

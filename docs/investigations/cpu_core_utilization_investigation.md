# CPU core utilization

The user saw a CPU load of Nemotron 3.5 Transcribe use one core, then four to six of the sixteen, and asked whether
processing could use the cores more. The machine has 8 physical cores (16 threads); the compute pool defaults to the
physical count, and a machine-wide lease (`cpu_lease.rs`, lock files in `/dev/shm/inference-rs-cpu-slots-<uid>`)
shares 7 extra-worker slots among inference processes.

## Run 1 - 2026-11-01 ~17:20

**Question:** why was a CPU BF16 load (prompt scoring through the Python binding, `device: cpu`) on one core for
minutes?

**Command:** `perf record -t <busiest thread> -g` during the load.

**Finding:** 94 % of samples in `copy_strided_src_` (`StridedIndex::next`, its zipped iterators): the NeMo stacked
experts' `transpose(1, 2).contiguous()` at load. A swapped last pair of dims makes the source's innermost stride
non-unit, so the generic copy visits every element through a strided index, on one thread. `serve` takes the same
path (the earlier CPU calibration server loaded for ~8.5 minutes); the binding is not the cause.

**Fix:** `copy_transposed_`: a batch of matrices with stride 1 down the rows is copied in 32 x 32 tiles, in parallel
over each batch's row bands past 64 K elements. Test against elementwise reads (partial tiles, a start offset, a
broadcast batch). The same load then took ~97 s.

## Run 2 - 2026-11-01 ~17:45

**Question:** after loading, the forward pass showed 3.5-6 cores busy; where are the rest?

**Commands:** `perf record --latency -p <pid> -g -- sleep 15`, then `perf report --sort parallelism`, and by symbol
at parallelism 1 and 2.

**Findings:**
- Wall time by threads running: 69.5 % at 2, 29.7 % at 1, under 1 % at 3 or more. The lease was not the limit (only
  this process held the slots).
- At both levels the time is `gemm_f32` microkernels and lhs packing, `widen_bf16` (a BF16 weight widened to f32 per
  call) and `BarrierPool` workers waiting.
- The unquantized experts' grouped path (`ungated.rs::forward_grouped`) runs one expert after another, two small
  matmuls each: a 677-token prompt gives each of the 128 experts ~32 rows, too few for gemm to split an 1856 x 2688
  weight over the cores. The experts are ~1.9 of the prompt's ~4.7 TFLOP.

**Change:** on the CPU the experts run side by side (rayon), each returning its hidden and gated output rows; the
statistics and the `index_add` back into the tokens stay sequential, in expert order.

**Baseline (per clip, BF16 reference run):** clips 0 / 2 / 4 / 6: 65.0 / 62.7 / 53.7 / 87.6 s (677 / 651 / 515 /
952 prompt tokens).

**Also:** the lease (c18a7a472) was meant for test runs, where nextest's per-test processes each sized their pools to
the machine; it applied to every process, the application too. It now leases only under nextest (`NEXTEST_RUN_ID`
set) or with an explicit `INFERENCE_CPU_SLOTS`; an application's ops take every worker they ask for. The run below
still has the lease (its library predates the change).

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

## Run 3 - 2026-11-01 ~18:45 (baseline tag `cpu-baseline-2026-11-01`, 660e3a207)

**Question:** with nothing else running, how do `serve --cpu` loads of Nemotron 3.5 Transcribe use the cores?

**Command:** `scratchpad/bench/cpu_bench.sh`: `inference serve --cpu`, wait for the listener (load time), five
calibration clips as chat requests with the transcription instruction and `max_tokens` 1 (prompt processing only),
CPU% sampled every 2 s and `perf record --latency` over the prompt phase; machine idle (load average < 1.5).

**Findings:**
- BF16 checkpoint: load 63.2 s (was ~8.5 minutes before Run 1's copy). 396-818 tokens at 11.4-14.0 tok/s, 540 %
  CPU; wall time at 8 threads 64.8 %, at 15 9.3 %, at 1 15.3 %.
- GGUF pair (IQ4_NL up, Q5_0 down, Q8_0 rest): load 28.1 s; 1.2-1.7 tok/s, 147 % CPU, 78 % of wall time on 2
  threads; the 818-token prompt was OOM-killed at 119 GB. 43 % of samples in `memmove` and byte gathering: the
  experts took the gather path, copying a matrix per routed row. The CPU expert kernel takes a layer only when both
  projections are its types, and its 32-value set was IQ4_NL and Q8_0; the Q5_0 down stack sent every MoE layer to
  the gather.

**Fix:** Q4_0 and Q5_0 in the CPU expert kernel's 32-value dots (AVX2 and generic; Q5_0's fifth bits expanded to
bytes and `0xf0` ORed into the nibbles whose bit is clear, so the signed dot applies). Tests: AVX2 against generic,
and the kernel against Candle's dequantized weights for IQ4_NL / Q5_0 and Q4_0 / Q5_0, gated and ungated.

**After:** GGUF pair load 15.0 s, 396-818 tokens at 14.4-15.1 tok/s, 583 % CPU; wall time at 9 threads 91.2 %.
Samples: `vec_dot_q8_0_q8_0` 35 % (the dense Q8_0 layers' prefill, a row dot per output), `q5_0_32` 13 %,
`iq4_nl_32` 11 %, barrier waits 7 %, f32 gemm 8 %.

**Next:** per-core efficiency: the dense Q8_0 matmuls in prefill (a tiled kernel or dequantize-then-gemm past some
rows), then the 8-thread pool on a 16-thread machine (hyperthreads were measured slower for the conv kernels).

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

## Run 4 - 2026-11-01 ~19:05

**Question:** why do the dense Q8_0 layers' prefill matmuls take a dot product per output row (35 % of samples)?

**Finding:** the tiled x86 path (`repack_x86::select`) required `k % 256 == 0` for every type, Q8_0 included, though
Q8_0's tiles are per 32-value block. This model's inputs are 2688 wide (attention q/k/v, `ssm_in`, shared up) and
3712 (shared down); only the 4096-wide ones tiled.

**Change:** Q8_0 takes the tiles at any whole number of 32-value blocks: its rows pad to whole superblocks with
zero-scale tiles (+128, a zero weight) and zero activations, which add nothing; the superblock count rounds up in
every path (matmul, fused and indexed gemv), exact for the K-quants, which still need whole superblocks. The test
against ggml's `vec_dot_q8_0_q8_0` (1e-4) runs at k = 1024, 1056 (33 blocks) and 2688.

**Result (GGUF pair, same benchmark):** 20.2-21.7 tok/s (was 14.4-15.1), 640 % CPU; wall time at 9 threads 47 %,
at 8 39 %. Samples: the expert dots `q5_0_32` 19.8 % and `iq4_nl_32` 16.7 % (a row dot per route, each expert's
weights decoded again for every token routed to it), `q8_0_tile2_avx2` 10.7 %, f32 gemm 10.4 % (the audio tower
in f32), barrier waits 10.2 %, elementwise mul 9.0 %, `expf` 7.2 %.

**Next:** tiles for the 32-value expert types (decode an expert's rows once for its tokens, as IQ4_XS / IQ3_S do).

## Run 5 - 2026-11-01 ~19:30

**Question:** what are the ~20 % of samples in generic elementwise ops (`binary_map` Mul, `expf`, `vec_add`) and the
10 % barrier waits? DWARF unwinding gave no callers; `perf record --call-graph lbr` did.

**Finding:** the engine thread's time sat under `granite::MambaLayer::forward_full` (5.1 % of all samples, as much as
the whole MoE): on the CPU the selective scan ran one timestep at a time as tensor ops, expanding each head's decay
(one scalar) over the 64 x 64 x 128 state and passing the 2 MB state through ~6 ops per step, each split over the
barrier pool, for every token of every Mamba layer; the causal convolution was a per-step loop of small ops too.

**Change:** `inference_nn::ops::{selective_scan_cpu, causal_conv1d_cpu}`: the scan as `selective_scan_cuda` defines
it (softplus of dt + bias clamped, `state = state * exp(dt a) + dt b x`, `y = state . c + d x`), parallel over
(batch, head) with each head's 32 KB state in cache across the sequence; the convolution over plain slices, parallel
over output steps. Granite's CPU path takes them; other devices keep theirs. The tiny Nemotron-H test (step-by-step
decode, a separate code path, against one prefill at 1e-4) and the Granite suites pass.

**Result (GGUF pair):** 27.8-29.9 tok/s (was 20.2-21.7), wall time at 9 threads 86.5 %, barrier waits 3.9 %. Samples:
expert dots `q5_0_32` 26.2 % and `iq4_nl_32` 22.3 %, `q8_0_tile2_avx2` 14.5 %, f32 gemm 13.5 % (the audio tower),
the scan 5.2 %.

**Next:** the expert dots (48 %): a token-reusing tile for the 32-value types.

## Run 6 - 2026-11-01 ~19:50

**Question:** do the 32-value expert dots (48 %) gain from decoding each weight block once for several routes, as the
256-value types' `vec_dot_x4` does?

**Change:** `vec_dot_32_x4` (IQ4_NL, Q4_0, Q5_0, Q8_0): each block decoded to its 32 signed weights once and dotted
with four routes' activations, every route's sums in the single-row kernel's order (the test checks the four results
equal the single-row one bit for bit). Each type's single and four-row kernels come from one decoder through a macro.
The expert kernel's 32-value gate/up and down loops take routes four at a time.

**Result (GGUF pair):** 31.1-34.1 tok/s (was 27.8-29.9). Samples: `q5_0_32_x4` 18.0 %, `iq4_nl_32_x4` 17.3 %, Q8_0
tiles 16.9 %, f32 gemm 16.0 % (the audio tower), the scan 6.1 %, barrier waits 5.7 %; 9 threads 84 % of wall time.

## Run 7 - 2026-11-01 ~20:00

**Question:** with the cores busy, does a 16-thread pool (the hyperthreads) beat the default of one per physical
core?

**Command:** the same benchmark with `RAYON_NUM_THREADS=16` (sizes rayon, the barrier pool and gemm).

**Finding:** 33.0-34.3 tok/s at 1,161 % CPU against 31.1-34.1 at 607 %: no gain for twice the CPU; the SIMD kernels
are compute-bound per core. The default stays at the physical core count.

**Where it stands (GGUF pair, CPU, from the `cpu-baseline-2026-11-01` tag):** prompt processing 1.2-1.7 -> 31-34
tok/s, nine threads busy 84 % of the wall time; load 28 -> 4-15 s; the 818-token prompt no longer runs out of
memory. BF16 checkpoint: load ~8.5 min -> 63 s, 11-14 tok/s. Left: the expert dots (~40 %), the Q8_0 tiles (17 %),
the audio tower's f32 gemm (16 %).

## Run 8 - 2026-11-01 ~20:20

**Question:** how far are we from llama.cpp, on the same text GGUF?

**Commands:** `CUDA_VISIBLE_DEVICES= llama-bench -m ...-text-IQ4_NL-down-Q5_0.gguf -p 512 -n 0 -t 8 -ngl 0 -r 2`;
ours: `serve --cpu` with the text GGUF alone, a 539-token wikitext prompt as a chat message, `max_tokens` 1, three
times. Both on the idle machine.

**Findings:**
- llama.cpp pp512: 70.95 +- 1.10 tok/s. Ours, decoder only: 48.7, 51.1, 51.1 tok/s. About 1.4x behind (from ~9x
  at the baseline tag's 1.2-1.7 with audio).
- After the 4 x 4 register tiles for the 32-value expert rows (each block decoded once for four routes, each route's
  activations loaded once for four rows, as llama.cpp's tinyBLAS), audio prompts run 33.5-36.6 tok/s (was
  31.1-34.1); the audio tower and speaker branch are ~30 % of an audio request.
- Samples now (audio prompts): Q8_0 tiles 18.2 %, f32 gemm 17.3 % (audio side), expert tiles 28.6 % + single-route
  remainders 5.2 %, the scan 6.5 %, barrier waits 6.1 %.

**Next:** profile llama.cpp and ours op by op on the text prompt to place the remaining 1.4x; the audio side's f32
encoders (two of them: speech and the speaker branch) on audio prompts.

## Run 9 - 2026-11-01 ~20:40

**Question:** where do the remaining ~2.7 s per 512 tokens go, against llama.cpp on the same file?

**Commands:** `perf record` of `llama-bench -p 512 -t 8 -ngl 0` and of our `serve --cpu` on the 539-token text prompt.

**Per 512 tokens (ours ~10.0 s, llama.cpp 7.3 s):** dense Q8_0 2.5 s vs 2.4 s (tinyBLAS 4xN); Q5_0 down experts
2.35 s vs 2.5 s (llama.cpp's plain `ggml_vec_dot_q5_0_q8_0`); IQ4_NL up experts 2.2 s vs 1.46 s (llama.cpp repacks
IQ4_NL to 8 interleaved rows, `gemv_q4_b32_8x8_q8_0_lut_avx`); the Mamba scan 0.9 s vs 0.1 s.

**Changes:**
- The scan's inner loop fused the state update and the dot with C into one serial sum, which kept it scalar; the
  update now runs on its own and the dot keeps 8 partial sums. Scan samples 8.8 % -> ~4.3 % (with the dot).
- A group's last partial route quad pads with its final route into the 4 x 4 tile (only the real routes stored),
  where it took the single-route dots (7 % of samples).

**Result:** text prompt 54.3-55.1 tok/s (was 48.7-51.1); llama.cpp 70.9. Left: IQ4_NL's interleaved repack
(~0.5 s per 512 tokens), the scan (~0.3 s), barrier waits (3.5 %), many small items.

## Run 10 - 2026-11-01 ~21:10

**Questions:** what is the audio side's 30 % of an audio request; what is left in the decoder besides the three big
kernels?

**Audio side (temporary probes, reverted):** per 540-token clip the speech encoder takes 3.43 s and the speaker
branch 0.81 s. Over its 63 layers (both encoders) at 489 steps: FFN 2.08 s, q/k/v + RoPE 0.98 s, attention 0.53 s,
output projection 0.26 s, GELU 0.26 s. A probe of the same f32 matmul shapes (350 x 1280 x 5120 and the like) runs at
393-422 GFLOP/s, about what the encoders' linear layers get: they are gemm-bound in f32. Faster would mean other
arithmetic (Q8_0 weights, an accuracy question for the audio KLD), not less overhead.

**Decoder, grouped (539-token text prompt):** Q8_0 tiles 26.4 %, Q5_0 experts 25.4 %, IQ4_NL experts 24.5 %, Mamba
scan + conv 6.9 %, barrier waits 3.7 %, activation quantization 2.5 %, elementwise 2.4 %, expert glue 2.4 %,
attention 2.3 %.

**Scan:** `perf annotate` showed the 8-lane dot compiled to scalar `vmulss` / `vaddss` with the partial sums spilled
to the stack. The row step (update, then the dot with C) is now AVX2 intrinsics with a portable fallback, the update
as a multiply and an add as the CUDA scan rounds it; B and C reach the CPU scan by group (the 64 heads read their 8
groups) instead of expanded to every head (~35 MB a layer). Scan 4.3 % -> 2.8 %; text prompt 55.1-55.7 tok/s.

**IQ4_NL against llama.cpp:** per 32 multiply-adds (one row, one token, one block) both spend 4 integer ops (sign,
sign, maddubs, madd); we spend 3 more on the scale (convert, broadcast, fused multiply-add) where llama.cpp's
interleaved weights and activations (8 rows' and 4 tokens' blocks woven together) land one row per lane and scale 8
outputs with one fused multiply-add (~0.5 per output). That is the 1.6x on those kernels. Interleaving in registers
at decode time does not recover it (broadcasting each 8-value activation chunk costs what the scale saves); it takes
the repacked layouts, at load for the weights and per prompt for the activations, as llama.cpp does.

## Run 11 - 2026-11-01 ~22:00

**Question:** can a row-per-lane kernel reach llama.cpp's interleaved rate without repacking the weights at load?

**Kernel:** `vec_dot_32_8x4` takes eight rows in their GGUF layout. Rows i and i + 4 load as one 256-bit pair (low
and high lane), split into values 0..16 and 16..32, and two in-lane 4x4 transposes of 32-bit words leave each row's
values of a word position in its own lane. Each token's 16-byte activation halves are broadcast and their words
shuffled across the lanes, so one convert and one fused multiply-add scale the block for all eight rows. The
transposes cost ~16 shuffles per eight-row block, shared by the four tokens. Run 10's "interleaving in registers
does not recover it" was wrong: it assumed the activation side was interleaved per row, not the weights.

**Command:** `cargo test --release -p inference-quant --lib tile_rate_32 -- --ignored --nocapture` (8 rows x 2688
columns x 32 tokens, one thread).

| type | 4x4 tiles GMAC/s | 8x4 lanes GMAC/s | ratio |
|------|------------------|------------------|-------|
| IQ4_NL | 34.1 | 49.3 | 1.45 |
| Q4_0 | 35.2 | 51.3 | 1.46 |
| Q5_0 | 30.8 | 40.3 | 1.31 |
| Q8_0 | 37.4 | 50.7 | 1.36 |

Agrees with the one-row kernel within 1e-4 relative (`block32_dots_match_the_generic_kernels`). Next: run the
expert items in eight-row groups through it and measure the text prompt.

## Run 12 - 2026-11-01 ~23:00

**Question:** what does the eight-row kernel buy end to end (expert items now at least eight rows)?

**First try, a regression:** the text prompt fell from ~60 to 36 tok/s. The server build is the dev profile (debug
assertions and overflow checks off), and there the kernel ran at 10 GMAC/s against 49 under `--release`: `perf`
showed `<[__m256i; 2]>::map` and the array zips' iterators as their own functions, so every vector went through
memory. Plain index loops instead: 47 GMAC/s in the dev profile too.

**A/B, alternating servers (539-token text prompt, warm runs):** HEAD 59.0-60.5 tok/s, eight-row kernel 66.9-68.8
tok/s (+13 %). llama.cpp's pp512 on the same file: 70-71.

## Run 13 - 2026-11-01 ~23:40

**Question:** Q4_0 and Q5_0 values fit 0..32 once their offset is added back; does dotting them unsigned pay?

**Change:** for those two the eight-row kernel takes the stored values (nibbles, Q5_0's fifth bit ORed in as 0x10)
straight into maddubs' unsigned side, no sign ops. Four word positions' pair sums add in i16 (at most 4 x 2 x 31 x 127
= 31496) before one madd, and each block takes `offset * sum(y)` out in i32, with the sum stored in `Q8Block32` at
quantize time. Integer sums are the same as the signed path's, so the results are too.

**Kernel (`tile_rate_32`, dev profile):** Q4_0 48.6 -> 68.8 GMAC/s, Q5_0 37.2 -> 50.4. IQ4_NL and Q8_0 keep the sign
path (their values overflow i16 pairs when offset).

**A/B, alternating servers (539-token text prompt, warm runs):** previous commit 79.0-80.7 tok/s, unsigned 84.8-87.8
(+8 %). Both ran ~20 % faster than in Run 12, so the machine was slower then (the A/B pairs are what compare).
llama.cpp's pp512 measured right after: 72.0 +- 0.4 tok/s.

## Run 14 - 2026-11-02 ~00:10

**Question:** what is left after the expert kernels, and where do audio prompts stand?

**Profile (539-token text prompt, `perf record`):** dense Q8_0 tiles 35.3 %, IQ4_NL experts 21.2 %, Q5_0 experts
18.8 %, barrier waits 5.5 %, scan row step 1.3 %, conv 1.3 %, attention 1.2 %, the rest below 1 % each. A 2.5 %
symbol named for the GDN conv update (`causal_conv1d_update_cpu`'s zipped sum) was the Mamba conv's tap sum (the two
closures' code merged under one name): an index loop inlines it, and it is gone from the profile. Text prompt
78.6 / 86.4 / 86.5 tok/s (noise-level change). `CpuExpertProjection::row` (0.8 %, out of line) is `#[inline]` now.

The dense Q8_0 AVX2 tile already spends one sign, one maddubs and one madd per 32 multiply-adds with the weight
work shared over the tokens; past that it takes AVX-VNNI, which this CPU lacks.

**Audio prompts (`cpu_bench.sh`, 5 clips, same build):** 56.0-60.7 tok/s (Run 7's 33.5-36.6 was a slower machine
state too). At ~86 tok/s for the decoder, a 540-token clip spends ~6.3 s there and the rest (~3.4 s) in the speech
encoder and speaker branch, f32 gemm-bound (Run 10): the audio side is now ~35-40 % of an audio request.

## Run 15 - 2026-11-02 ~03:40

**Question:** where does CPU decode stand (all runs so far measured prefill)?

**Commands:** `serve --cpu` on the text GGUF, 128 tokens with `ignore_eos` twice; `CUDA_VISIBLE_DEVICES=
llama-bench -m <text gguf> -p 0 -n 128 -t 8 -ngl 0`.

**Result:** ours 7.7 / 7.7 tok/s, llama.cpp tg128 10.71 +- 0.04. Decode is memory-bound: ~3.5B active parameters at
~4.5-5 bits plus the Q8_0 dense parts is ~2-2.5 GB a token against ~40 GB/s of dual-channel DDR4, a ceiling near
16-20 tok/s. We are 28 % behind llama.cpp here, the opposite of prefill. Not yet profiled.

## Run 16 - 2026-11-02 11:06 (local clock)

**Question:** where does CPU decode lose to llama.cpp (Run 15: 7.7 against 10.71 tok/s)?

**Commands:** `serve --cpu` on the text GGUF, scratch `bench/decode.py` (128 tokens, `ignore_eos`) under `perf record`
(then `--call-graph lbr`, since the build has no frame pointers); the GGUF's tensor sizes from `gguf-py`.

**Budget:** a token reads ~1.99 GB of dense Q8_0 (`ssm_in` 677 MB, `output` 374, `ssm_out` 269, the shared expert
488, attention 148) and ~0.86 GB of routed experts (6 of 128: Q5_0 down, IQ4_NL up), ~2.85 GB. llama.cpp's 10.7 tok/s
is ~30.5 GB/s; ours at 7.1 is ~20 GB/s.

**Profile (master `56ef9b6c8`, 7.1 tok/s):** `vec_dot_q8_0_q8_0` 46.8 %, barrier-pool spin 15.2 %, `q5_0_32_8x4`
11.3 %, `iq4_nl_32_8x4` 10.8 %, then `binary_map_vec_par<Mul>` 2.8 %, `expf` 2.2 %, `vec_mul` 1.9 %, `vec_add` 1.9 %:
on the CPU a decode step ran the Mamba update as tensor ops over `(batch, 64, 64, 128)` (dA = exp(dt A) over 524K
values, dB, dB x, state * dA + dB x, the matmul with C), each a pass over the 2 MB state per layer. CUDA decode steps
already run the scan kernel as a one-token chunk.

**Change:** a CPU decode step runs `selective_scan_cpu` as a one-token chunk, the same call the CUDA step makes
(`scan_step`, B and C by group); the tensor-op step stays for devices with neither kernel.

**Result:** alternating master and branch servers, two runs each: master 7.0 / 7.2 / 7.2 / 7.2 tok/s, branch 8.3 /
7.9 / 8.0 / 8.1 (+13 %). The elementwise ops leave the profile. Greedy 200-token CPU completions against master: one
identical, two diverge at tokens 49 and 50, logprobs within 0.10 before (F32 summation order). llama.cpp in the same
session: tg128 9.73 +- 0.09 (Run 15's 10.71 was another day's machine state), so the gap is now ~20 %.

**Negative results:**
- Advancing the CPU scan's state in place (an `InplaceOp1`, as the CUDA scan writes its state) instead of copying the
  2 MB out and back: no measurable change (7.9-8.3). The remaining `memmove` (~0.5 %) is the recurrent pool's gather
  and scatter (`RecurrentStatePool::{gather,scatter}_recurrent_state`), not the scan.
- An AVX2 four-row Q8_0 dot (each activation block loaded once, four accumulators): 7.6-8.1, no change; the Q8_0
  GEMV runs at ~29 GB/s and is memory-bound.

**What is left (perf, LBR stacks):** per 125 ms token, the Q8_0 matmuls ~69 ms (~29 GB/s), the routed experts ~37 ms
(~23 GB/s), and ~19 ms of small ops spread through the layers while the pool spins (16.5 %). The experts' bandwidth
is the larger lever.

## Run 17 - 2026-11-02 11:34 (local clock)

**Question:** why do the routed experts read at ~23 GB/s in decode when the dense Q8_0 GEMV reads at ~29?

**Finding:** a decode step routes one token to each of its 6 experts, and `padded_quads` pads a group's tail to four
routes by repeating the last, so the 8-row tiles (`*_32_8x4`) dotted every weight block against four copies of the one
activation row: four times the integer work for the same bytes.

**Change:** the tile kernel is generic over its token count (`sum_blocks_32_8xt`), with `*_32_8x1` beside the four-
route form; a group of one route takes it, for the up and down projections. Unit case: the 8x1 rows equal the 8x4
tile's lanes bit for bit.

**Result:** alternating master (`effdb8251`) and branch servers, two runs each: master 7.6 / 7.7 / 7.9 / 8.0 tok/s,
branch 8.3 / 8.4 / 8.3 / 8.2 (+6 %). Experts 25 -> 21 % of samples (~33 GB/s); greedy CPU completions bit-identical to
#776's. Per ~122 ms token now: Q8_0 ~63 ms (~31.5 GB/s), experts ~26 ms, the pool's spin 18 % (~22 ms, the small
ops between matmuls). With both kernel families near the machine's bandwidth, the gap to llama.cpp (9.73 tok/s,
~103 ms) is that serial time.

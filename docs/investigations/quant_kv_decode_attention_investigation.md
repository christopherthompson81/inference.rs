# Quantized-KV decode attention investigation

Question: why does our draft-verify step grow 13.6 ms from 2K to 125K context on Qwen3.8 27B (q3 cache, 16 full
attention layers) against tqLLM's 9.4 ms (tqllm_comparison_investigation.md Run 7), and how much of it can the fattn
kernel give back?

## Run 1 - 2026-10-19 16:40

Question: the fattn kernel alone at the model's shape (24 Q heads, 4 KV heads, head 256), q3/q3 paged cache, 1 row
(plain decode) and 8 causal rows (a 7-draft verify), against a bf16 cache.

Command: new bench `bench::verify_long_context` (`cargo nextest run -p inference-fattn --features cuda --run-ignored
only -E 'test(/bench::verify_long_context/)' --no-capture`), RTX 3090, LAN server stopped. GB/s counts the q3 cache.

```
kv        rows 1 q3   GB/s    bf16       rows 8 q3   bf16
2048         26.4      70       30.1        58.0       89.8
8192         53.3     138       72.6        98.8      114.7
32768       159.6     184      212.4       236.5      310.8
65536       299.8     196      399.3       417.0      547.7
131072      592.5     198      774.8       817.0     1059.0
262144     1111.8     211     1519.5      1645.0     2099.9
```

Finding: 817 us x 16 layers at 128K is 13 ms a step, the whole end-to-end growth. bf16 streams ~690 GB/s; q3 reads
4.6x fewer bytes but runs only 1.3x faster, at ~200 GB/s: not bandwidth bound. At bandwidth 128K would be ~130 us.

## Run 2 - 2026-10-19 17:00

Question: what bounds the quantized path: the loads, the dequant math, or the attention math?

Variants of `fattn_load_quant_tile` / the kernel (each a full fattn rebuild, ~9 min), 128K:

```
variant                                              rows 1    rows 8
as is                                                 592.5     817.0
dequant math removed (loads kept, synchronous)        436.5     683.8
raw rows copied with cp.async into smem staging,
  dequantized smem -> f16 tile (pipelined like bf16)  583.2     779.1
no copy and no dequant (attention math only)          145.5     442.3
staging + dequant with compile-time bits and the
  0x6400 | (2c + 1) half trick (~3.5 ops a value)     479.8     681.7
```

Finding: with the loads overlapped, the dequant itself is the cost: ~440 us of the 583 at 1 row, ~340 of 779 at 8
rows, and the old unpack (runtime bits, 64-bit shifts, a float conversion a value) ran ~25 instructions a value.
The compile-time unpack takes back ~100 us at both widths; the floor at 8 rows (442 us) is the f16 MMA over 64
padded columns (6 heads padded to ncols2 8, x 8 rows).

## Run 3 - 2026-10-19 19:30

Question: of the staged path's remaining cost, how much is the copy and how much the dequant, and what does the
dequant's code look like?

Variants at 128K (staged, compile-time unpack as the base; the bench now passes `causal: true` without a mask
tensor, as the paged decode does, which reads ~3-4% faster than an explicit mask):

```
variant                                              rows 1    rows 8
base, explicit mask                                   479.8     681.7
base, implicit causal                                 463.4     658.4
copy kept, dequant skipped                            289.0     609.2
dequant kept, copy skipped                            411.5     601.3
attention math only (Run 2)                           145.5     442.3
dequant forced inline                                 544.2     763.0
one 8-value chunk per thread (contiguous stores)      803.3     913.5
ld.shared reads, 4 groups' reads before converting    428.5     722.5
```

SASS of the noinline dequant (`cuobjdump -sass`, ncols1 1 x ncols2 8, head 256): ~3.4 instructions a value
(LOP3/SHF/HADD2/HMUL2) as designed, but the staged reads compile to generic `LD`, not `LDS`: through a noinline
call the compiler cannot tell the pointer is shared.

Findings:
- The copy is not hidden at 8 rows (+167 us alone): one tile ahead at nbatch_fa 32 is ~3.6 KB in flight per SM,
  too little to cover DRAM latency. A deeper ring of raw tiles is cheap in smem (the raw tile is ~1/4.6 of the
  f16 tile).
- The dequant costs ~3x its instruction count: few warps per SM (the 1-row config's f16 tiles take 67 KB, one CTA
  of 4 warps; the 8-row config at most two), so each group's dependent loads and math have little to overlap them. Batching reads helps 1 row
  and hurts 8 rows (register pressure next to the 64-column accumulators); inlining hurts both.
- The 8-row floor (442 us) is the f16 MMA itself: KQ accumulates in f32, which GA102 runs at half the f16 rate, over
  64 columns of which 16 are padding (6 heads to ncols2 8). tqLLM scores with int8 MMA (4x the f32-accumulate rate)
  on codes it never widens to f16; that, not the loads, is what a further step needs.

## Run 4 - 2026-10-20 00:15

Question: what the staged loader (Run 2's compile-time unpack, noinline, generic loads: the best variant so far,
463 / 658 us at 128K) gives end to end.

Command: correctness first: `cargo nextest run -p inference-fattn -p inference-nn --features cuda -E
'package(inference-fattn) | test(/fattn_tests::/)'`: 53 passed (decode and 8-row verify over q3/q3 and q4/q2 caches
at head 256, head 128 through the 4-byte copy). Then the model's verify graph (`INFERENCE_RS_CUDA_PHASE_TIMINGS=1`,
specbench ui, 1 run each), master's binary and this branch's back to back in one session:

```
prompt    master target verify   branch target verify
1988      24.04 ms               23.91 ms
31919     26.90                  26.60
124730    35.68                  34.83
```

Finding: -0.85 ms a step at 125K (decode 66.4 -> 67.8 tok/s), within noise below 32K. Run-to-run conditions move a
lot: Run 7 of the tqllm doc read 40.4 ms at 125K for the same master binary, so only same-session pairs compare.
The kernel bench predicted ~2 ms; the model's 16 layers recover less than the bench.
Implication: incremental fixes inside the f16-tile mma kernel are worth ~2-3% at long context. The attention at
125K still costs ~10.9 ms a step against a ~2 ms bandwidth floor (16 x ~130 us); what tqLLM does differently is
structural: K codes go to int8 in smem and score with an int8 MMA against an int8 Q in K's rotated basis (no f16
K tile, no f32-accumulate f16 MMA), with only V widened to f16.

## Run 5 - 2026-10-20 03:00

Question: does scoring Q.K with int8 MMA (tqLLM's approach) beat the staged f16 path?

Implementation (branch quant-kv-int8-qk-wip, inside the mma kernel, multi-stage head-256 instances): K codes unpacked
to an int8 tile as c - 2^(bits-1); a stored value is (2 k8 + 1) * scale / 2^bits, so a 32-value group's Q.K is
scale / 2^bits * (2 qs dot(k8, q) + qs sum(q)), one m16n8k32 per group. Q as one int8 per value missed the f16
tolerance (decode 0.0041 over 0.004); Q as hi + lo / 254 (two int8 MMAs, ~16 bits) matches the f16 path's errors
exactly (0.0020 decode, 0.0157 verify) and fits the f16 Q fragments' registers.

Bugs on the way: an `else` before `#pragma unroll` let the f16 Q.K loop run as well (3.2 max error); writing the
int8 Q into tile_V overwrote unread f16 Q rows at 64 columns (tile_Q is larger than the K tile there). Fixed by bracing
and by quantizing each group in place (its 64 f16 bytes become its 32 hi + 32 lo bytes).

```
128K                     rows 1    rows 8
staged f16 (Run 4)        463.4     658.4
int8 hi/lo Q.K            474.6     695.5
```

Finding: no gain; the f16 MMA was not what bounds this kernel once it carries the quantized loads.

## Run 6 - 2026-10-20 03:40

Question: where does a quantized tile's time go? `clock64()` probes at the iteration's phase boundaries, thread
(0, 0) of block 0, per launch (temporary; ncu has no counter access on this machine).

128K, rows 1 (ncols 8, nbatch_fa 64, one 4-warp CTA per SM), cycles over 102 tiles:

```
phase                                    int8 Q.K   + 8 lookups batched   + issue inlined   + no table lookups
wait for the copies (A)                    16K        12K                   16K               16K
issue V's cp.async                        139K       260K                  249K              163K
unpack K to int8                          178K       176K                  173K              174K
Q.K (int8 MMA)                            128K       124K                  133K              134K
softmax                                    34K        33K                   33K               33K
wait (B) + issue next K's cp.async        158K       282K                  272K              184K
unpack V to f16                           232-446K   459K                  241K              236K
P.V                                        49K        42K                   45K               46K
bench (us)                                 588        867                   743               639
```

(the unpack-V row also absorbs the barrier wait for warps still issuing, so it moves with them)

Findings:
- Issuing the cp.async copies is the largest cost, ~1,400-2,700 cycles a tile for ~14 warp-wide 16-byte copies:
  the issuing warp stalls once the SM's outstanding requests fill, so the DRAM time lands at issue, where nothing
  overlaps it (one CTA of 4 warps per SM: the head-256 1-row config takes 82 KB of smem with staging).
- Block-table lookups add ~600-900 cycles a tile (identity table vs the lookup); batching them did not help.
- The K unpack (~1,700 cycles a tile, ~8x its instruction count) and V unpack are latency bound for the same reason.
- Q.K and P.V together are ~17%: the MMA side, int8 or f16, is not where this kernel's time goes.

Implication: the llama.cpp mma kernel's tiling (nbatch_fa 64 at 1 row, f16 K and V tiles double-buffered) leaves too
few warps per SM to overlap a quantized cache's copies and unpacks. tqLLM's kernel uses 32-key tiles and small smem,
so several CTAs share an SM. The next step is a dedicated decode/verify kernel with that shape (or warp-specialized
producer warps), not more changes inside this one.

## Run 7 - 2026-10-20 06:30

Question: does a dedicated decode/verify kernel with tqLLM's shape (small tiles, two blocks an SM) beat the mma
kernel over a quantized cache?

Implementation (`fattn-quant.cuh`, branch quant-kv-int8-qk-wip): one block per (split, K/V head, sequence), 4 warps,
32-key tiles; raw K/V rows cp.async'd into two slots sized by the layer's bits; K unpacked to int8, V to f16 in
smem; Q quantized from global straight into int8 hi/lo A fragments (amax and sums over the 4 lanes sharing a
row); Q.K by m16n8k32 int8, online softmax in exp2, P.V by m16n8k8 f16 from the score fragments with
ldmatrix.trans of V; every warp's partial is a split of a workspace merged by a combine kernel. Paged, batched,
causal and window only; anything else falls to the mma kernel. ~45 KB smem at q3, so two blocks an SM. Passed the
quantized-cache fattn tests on its first working build.

```
128K (us)                         rows 1    rows 8
mma kernel, staged (Run 4)         468.1     682.3
dedicated kernel                   513.2     766.3
  unpack skipped                   401.7     665.5
  Q.K, softmax and P.V skipped     418.9     438.2
2K                                  88.7      91.6    (mma kernel ~23 / ~52)
```

Findings:
- With no math at all the kernel streams the q3 cache at ~280 GB/s; the copy, not the arithmetic, is the ceiling.
  The bf16 mma path streams ~780 GB/s through the same cp.async, with ~32 KB tiles in flight against our ~7 KB.
- At 2K the fixed costs dominate: 4 x SMs / (K/V heads x sequences) splits regardless of length (device-side lengths
  leave the host blind under graph capture), and the combine pass.
Implication: the quantized path needs much more data in flight per SM than these tiles give (deeper raw pipeline,
larger raw tiles unpacked in sub-tiles, or producer warps), and lengths the host can use to size splits. Without
counter access (ncu) each variant is a ~10 minute rebuild and an inference from timings; enabling profiling
(NVreg_RestrictProfilingToAdminUsers=0) would cut that loop.

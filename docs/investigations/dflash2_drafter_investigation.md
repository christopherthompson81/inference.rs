# DFlash 2 drafter investigation

Does the DFlash 2 block-diffusion drafter for Qwen3.8 27B (incoai/Qwen3.8-27B-DFlash2) decode faster than the
checkpoint's built-in MTP head on a 24 GB card, and what does it cost in VRAM?

## Run 1 - 2026-10-19 04:00

Question: what is the net VRAM cost of DFlash 2 over built-in MTP?

Command: the drafter's config.json and safetensors; the 27B GGUF's tensor sizes (gguf-py); the loaders.

- Drafter: `DFlash2DraftModel`, 5 Qwen3 layers at the target's width (hidden 5120, 32/8 heads, head 128,
  intermediate 17408), taps target layers 5/19/33/47/61, block 8, selector top-k 16; no embeddings or head of its
  own. 3.85 GB in BF16. Its context cache is a 2048-token sliding window over 5 layers, ~40 MB at any length.
- Built-in MTP: one layer, 254 MiB in the IQ4_XS GGUF, plus its own attention layer's KV (~300 MB at 256K, q4).
  `inference-selection`'s loader loads it only when MTP is built in (`MtpConfig::is_builtin`), so an attached
  drafter gives it back.
- The drafter takes the target's ISQ type (`draft_lm_head_isq`), none for a GGUF target, so it loads in BF16;
  `INFERENCE_RS_DFLASH_ISQ` overrides it (q4k ~1.1 GB).

Net over built-in MTP: about +3.3 GB with a BF16 drafter, about +0.5 GB with q4k.

## Run 2 - 2026-10-19 04:20

Question: decode speed and acceptance, built-in MTP against DFlash 2 at several draft counts and precisions?

Command: the model manager loads only built-in MTP (`mtp.model` is refused: "an assistant model is not loaded by
the manager"), so DFlash runs through `inference serve --format gguf -f Qwen3.8-27B-IQ4_XS.gguf --pa-context-len
32768 --pa-cache-type q4 --mtp-model <drafter> --mtp-n-predict N` with `INFERENCE_RS_DFLASH_ISQ`, one server per
setting; built-in MTP and no speculation through the manager. RTX 3090, the LAN server stopped; chat completions,
256 tokens with `ignore_eos`, three runs each; greedy, and the web UI's defaults (temperature 0.6, top_p 0.95,
top_k 20, repetition penalty 1.1, on the GPU since PR #490).

```
decode tok/s                    greedy              UI sampling          acceptance (len, from the log)
no speculation                  50.2 50.2 50.2      50.2 50.2 50.2
built-in MTP, 3 drafts          77.9 79.2 78.2      61.8 70.3 62.1
DFlash 2 bf16, 7 drafts         74.0 73.8 73.5      54.5 47.1 62.8       3.49 2.56 2.45
DFlash 2 bf16, 3 drafts         96.1 96.1 95.5      77.1 87.8 68.5       3.02 2.69 2.49
DFlash 2 q8_0, 7 drafts         70.6 71.3 70.9      47.3 52.3 55.3       3.50 2.35 2.69
DFlash 2 q4k, 7 drafts          73.6 74.2 73.9      47.8 49.9 49.4       3.48 2.34 2.61
DFlash 2 q4k, 2 drafts          93.1 97.4 96.7      76.4 74.6 91.6       2.62 2.09 2.13
DFlash 2 q4k, 3 drafts          95.4 97.9 98.6      87.2 78.1 77.4       3.05 2.55 2.36
DFlash 2 q4k, 4 drafts          92.9 93.8 94.8      68.5 70.0 66.8       3.35 2.52 2.36
DFlash 2 q4k, 5 drafts          76.3 77.5 77.7      59.6 66.4 53.9       2.99 2.79 2.57
```

z-lab's card measures 7 drafts (block 8) on an H200, where a wide verify step is nearly free; on the 3090 the
8-row verify costs more than the extra accepted tokens return, and 2-3 drafts are best. A q4k drafter keeps the
BF16 drafter's acceptance and speed.

Implication: DFlash 2 with a q4k drafter and 3 drafts is ~23% faster than built-in MTP greedy (96-99 vs 78-79)
and ~20% with the UI's sampling, for ~0.5 GB more VRAM. To serve it from the web UI the model manager needs to
load an assistant model (`mtp.model`), and the drafter's precision needs a setting in place of the environment
variable.

## Run 3 - 2026-10-19 05:10

Question: why does our acceptance trail the published numbers, and do other engines do better on this card?

Published on an RTX 3090: llama.cpp (z-lab discussion 9: UD-Q4_K_XL target, DFlash2 Q8_0 drafter, a FastAPI-app
prompt, temperature 0.7, top_p 0.8, top_k 20, presence penalty 1.5, thinking off, 10-12K tokens at 160-176K
context) mean length 4.09 at 4 drafts, 5.51 at 7 (64% of drafts); ExLlamaV3 (r0b0tlab, EXL3 4.0 bpw target and
drafter, GSM8K greedy, 8K context) 5.66 against the MTP head's 4.12, 163 tok/s.

Code review against z-lab's `dflash/model.py` and vLLM PR 52816 (a subagent, every item cited): no divergence in the
tapped states (residual after layer i, 0-based, all 64 layers), the context path (fc then hidden_norm, k_norm and
NeoX RoPE at absolute positions, theta 1e7, 2048 window), the noise block (bonus anchor, mask 248070, drafts from
positions 1..n), the dynamic convolutions (causal, after each sublayer's norm and projection), the selector (top-16,
pairwise codebooks, probabilistic under `auto`), bidirectional masking, the target's head, or verification.
Open: drafts under n=7 run a shorter block ([anchor, n masks]) than the 8 rows it was trained on; a missing
prefill context would degrade silently (a fallback appends only the verify rows), unconfirmed; the noise block's
mask embedding comes from the Q4_K `token_embd` (cosine 0.9975 to BF16).

Command: the llama.cpp workload on ours (same sampling, `reasoning_effort: off`, `tool_choice: none`, a FastAPI
prompt told to answer directly: without that the model calls invented tools and stops), 2048 tokens, two runs,
acceptance from `/v1/models/speculative_stats`; IQ4_XS target, q4 cache at 32K, DFlash2 drafter at q8_0.

```
                      decode tok/s     accepted / proposed     mean length    ms per step
built-in MTP, 3       90.4  89.0       1459/1764 1438/1827     3.48 3.36      38
built-in MTP, 7       77.9  71.9       1661/2702 1616/3010     5.30 4.76      68
DFlash 2, 3           102.3 98.9       1412/1908 1434/1839     3.22 3.34      32
DFlash 2, 4           109.5 106.4      1522/2104 1514/2136     3.89 3.84      36
DFlash 2, 5           96.6  92.1       1566/2410 1546/2510     4.25 4.08      44
DFlash 2, 7           81.3  83.3       1599/3143 1605/3094     4.56 4.63      56
```

On this workload DFlash 2 at 4 drafts is the fastest (106-110 tok/s, +20% over built-in MTP at 3). Two gaps:
acceptance at 7 drafts is 4.6 against llama.cpp's 5.51 (and below our own built-in MTP at 7, 4.8-5.3, though
z-lab measures DFlash 2 above MTP); and the step cost grows fast with drafts, 56 ms at 7 against ExLlamaV3's ~35 ms
at the same width.

Implication: next, (1) draft the full 8-row block and keep the first n (the trained geometry), (2) log the drafter's
context length on each request's first draft to rule out the silent fallback, (3) profile an 8-row verify step.

## Run 4 - 2026-10-19 06:30

Question: what does ExLlamaV3 do per speculative step that we don't (its 3090 numbers: ~35 ms per 7-draft step,
163 tok/s at mean length 5.66)?

Command: two code studies (subagents) of exllamav3 v1.6.3 (native DFlash 2) and r0b0tlab/qwen38-exl3-dflash2
against ours, one with `INFERENCE_RS_CUDA_PHASE_TIMINGS=1` runs (IQ4_XS target, q8_0 drafter, q4 cache at 32K, one
GSM8K question, greedy, 240 tokens) and llama.cpp's `test-backend-ops perf` on a 4096x14336 matmul; SASS from our
sm_86 mmvq objects.

```
per step, GPU time            1 row (no spec)    5 rows (4 drafts)    8 rows (7 drafts)
target verify graph           20.3 ms            31.6 ms (1.56x)      45.4 ms (2.24x)
DFlash drafter graph          -                  4.3 ms               5.8 ms
GDN transition apply          -                  0.01 ms              0.01 ms
tok/s, mean length            50.0               122, 4.45            126, 6.2-6.6
llama.cpp mmvq, 8 rows / 1:   IQ4_XS 2.08x, Q4_K 2.5x, Q6_K 2.15x (5 rows: 1.41x, 1.67x, 1.51x)
```

On a GSM8K prompt our acceptance at 7 drafts (6.2-6.6) is above ExLlamaV3's 5.66; Run 3's gap was the prompt.
Host time is near zero (step wall time ~ summed graph time); our verifier stays on the GPU (theirs syncs per draft
position from Python); GDN rewind is at parity (their two state rows with a batched replay, our deferred transition
log); we capture the whole verify forward and drafter where they capture per block. ExLlamaV3's plain decode is
slower than ours (42.8 tok/s), but its 7-draft step is ~1.3x its one-token step against our 2.2x.

The difference is the small-batch quantized matmul. Ours (mmvq, `MMVQ_MAX_BATCH = 8`; mmq only from 9 rows)
decodes each weight block once but repeats the dp4a dot products and activation loads per row, and above 4 rows
drops to 2 warps x 2 output rows per block at 139-193 registers (IQ4_XS cuda8 151), so it turns compute and
latency bound: ~4.5 ms per extra row. ExLlamaV3's `exl3_gemv_kernel.cuh` (`EXL3_GEMV_MAX_M 8`) decodes each 16x16
weight tile once into a tensor-core fragment and covers all 8 rows with one fp16-accumulate mma, a block reading
activations once per 32 output columns; its int8 dp4a GEMV runs only for <= 2 rows at <= 5 bpw.

Implication, ranked:
1. A tensor-core small-batch GEMV for 3-8 rows per GGUF type in the target (IQ4_XS 10.7 GB, Q5_K 2.8 GB, Q6_K
   1.0 GB head), with the fused GLU/QKV variants: verify 45 -> ~24-27 ms at 8 rows, ~6-8 ms saved at 5. IQ4_XS
   maps onto `mma.m16n8k32.s8` after the existing table lookup (one 4-byte qs load = one thread's two k-slices).
   First a cheap test: route 5-8 rows to the existing mmq (int8 mma) per type where it beats mmvq.
2. The drafter at q4k through the same kernel: ~1 ms (q4k alone) to ~2.5-3 ms.
3. fp16 GDN recurrent state: ~0.1-0.2 ms. Not worth porting: their host loop, per-block graphs, int8 GEMV.

## Run 5 - 2026-10-19 07:30

Question: does sending 4-8 row batches to the existing MMQ kernel (int8 tensor-core tiles) where it beats MMVQ
close the verify gap?

Command: a new ignored bench, `fast_mmq::bench::small_batch_mmvq_vs_mmq` (`GGUF_BENCH_FILE`, optional
`GGUF_BENCH_TENSORS`), over the 27B's IQ4_XS/Q5_K/Q6_K tensors, Qwen2.5-Coder 3B's Q4_K/Q6_K and Qwen3.5 0.8B's
Q8_0; then Run 3's code workload.

```
time / 1-row MMVQ      rows:  1     2     3     4     5     6     7     8      MMQ (flat)
IQ4_XS ffn_up 17408x5120     1.00  1.02  1.07  1.28  1.31  1.57  1.72  1.98   1.36-1.45
IQ4_XS ffn_down 5120x17408   1.00  1.02  1.19  1.09  1.34  2.15  1.88  2.15   1.42-1.48
Q5_K attn_qkv 10240x5120     1.00  1.02  1.20  1.40  1.58  1.89  2.11  2.54   1.33-1.43
Q5_K ssm_out 5120x6144       1.00  1.04  1.23  1.46  1.68  1.93  2.05  2.39   1.45-1.48
Q6_K output 248320x5120      1.00  1.02  1.16  1.33  1.54  1.97  2.18  2.28   1.24-1.29
Q6_K ffn_down 2048x11008     1.00  1.00  1.04  1.21  1.54  1.63  1.71  1.85   1.60-1.67
Q4_K ffn_up 11008x2048       1.00  1.15  1.40  1.78  1.85  2.13  2.38  2.80   1.45-1.48
Q8_0 attn_qkv 6144x1024      1.00  0.94  1.10  1.23  1.19  1.35  1.46  2.65   1.43-1.45
```

MMQ costs the same from 1 to 8 rows; MMVQ crosses it at a per-type row count. `fast_mmvq::prefers_mmvq` now keeps
MMVQ up to 5 rows for IQ4_XS, 3 for Q4_K/Q5_K, 4 for Q6_K (its crossover moves with the output width), 7 for Q8_0,
and 8 for unmeasured types, at every plain, fused GLU/FFN and fused QKV dispatch.

```
code workload, 2048 tokens   before (Run 3)          after
no speculation               -                       49.7 45.5
built-in MTP, 3              90.4 89.0               91.8 94.2
DFlash 2 q8, 4               109.5 106.4             112.9 116.5
DFlash 2 q8, 5               96.6 92.1               136.1 124.6
DFlash 2 q8, 7               81.3 83.3 (56 ms/step)  142.9 135.3 (~34 ms/step)
```

A 7-draft step now costs about ExLlamaV3's (~35 ms) and 7 drafts is the best setting, as z-lab and llama.cpp find.
MMQ's flat cost sits at 1.25-1.6x the 1-row MMVQ, so a tensor-core GEMV built for 1-8 rows (Run 4's item 1) can
still take the 8-row step toward the 1-row cost.

## Run 6 - 2026-10-19 16:30

Question: can a tensor-core matvec for 2-8 rows (ExLlamaV3's design, Run 4 item 1) take an 8-row verify step near
the 1-row cost on recipe A (IQ4_KS bulk; qwen38_27b_weight_quant_investigation.md)?

Step breakdown before (`INFERENCE_RS_CUDA_PHASE_TIMINGS=1`, recipe A, q4k drafter at 7 drafts, 32K, q3 cache): target
verify graph 31.7 ms, DFlash drafter graph 3.8 ms, GDN transition apply 0.2 ms; a 1-row decode is 20 ms.

Kernel (`mmvq_gguf/mmvq_tc.cuh`, IQ4_KS only, sm_80+): a warp covers 16 weight rows with int8 mma.m16n8k32, 8 warps
split the row's blocks; activations quantized straight into the mma B fragment order (`mmvq_tc_quantize`). Bench on
recipe A's tensors (time / 1-row MMVQ, 2-8 rows flat):

```
version                                         ffn_up 17408x5120   ffn_down 5120x17408   attn_qkv 10240x5120
MMQ (reference)                                 1.46-1.51           1.41-1.45             1.37-1.41
1. per-lane 4-byte A loads, q8_1 activations    1.53-1.66           1.66-1.73             1.56-1.68
   (asm not volatile: no change; contiguous per-warp block runs: ~5x, reverted)
2. warp-tile copy, row-contiguous loads         1.26 (2 rows)-1.62  1.37-1.63             1.28-1.62
3. + activations in B fragment order            1.33 flat           1.53 flat             1.30-1.34
4. + register double buffer                     1.65                1.96                  1.54   (76 regs, occupancy 5 -> 3 blocks; reverted)
```

IQ4_KS rows sit at 12 mod 16 bytes (a 4-byte row scale, 136-byte blocks), so 16-byte loads fault; ExLlamaV3
repacks its weights into fragment order. Short, wide weights (ffn_down) lose: every block rereads the whole
activation (156 KB at K=17408), about as much traffic as the weights. With the kernel only where rows >= 2 x
columns, the verify graph measures 31.3 ms against 31.7: ~1%.

Implication: the tall matrices gain ~10% each but the step barely moves. Reaching ~1.1x needs a load-time repack
into fragment order (16-byte loads) and activation sharing across row tiles, best done with Nsight Compute counters
(blocked here for non-admin users: ERR_NVGPUCTRPERM). Parked on branch perf/small-batch-tc-gemv.

## Run 7 - 2026-10-20 05:00

Question: with GPU counters available (`NVreg_RestrictProfilingToAdminUsers=0`), where does the tensor-core matvec
lose to a 1-row MMVQ, and where does the rest of the 8-row verify step go?

Command: `ncu` on the small-batch bench (recipe A's ffn_up 17408x5120 at 8 rows); `nsys --cuda-graph-trace=node` over
recipe A with the q4k drafter at 7 drafts (1024 tokens x 2); phase timings for the step.

```
ffn_up, 8 rows          time     DRAM read   DRAM GB/s   L2 hit   occupancy   registers   top stall
1-row MMVQ (1 row)      61 us    47.4 MB     790         -        91%         39          long scoreboard 13.5
tensor-core matvec      80 us    48.3 MB     600         53%      62%         64          long scoreboard 9.6
MMQ                     91 us    -           549         27%      17%         128         -

per verify step (8 rows, 388 steps): 35.2 ms of kernels: MMQ 16.2 (235 calls: ffn_down, ssm_out, attn_gate,
attn_output, Q5_K attn_v, Q6_K head, the drafter), tensor-core matvec 13.1 (194: ffn gate/up, attn_qkv, attn_q),
GDN 1.7, the drafter's BF16 GEMV 1.6, attention 0.9, the rest ~2
```

The matvec reads exactly the weights (no DRAM over-fetch) but keeps fewer bytes in flight: 4 blocks/SM at 64
registers. Shared-memory stores conflicted (32% excess wavefronts; per-lane float stores at a 4-word stride): one
vector store per lane takes ffn_up 1.34x -> 1.29x. Tried and reverted: `__launch_bounds__` for 5-6 blocks/SM
(48/40 registers, spills: 3.5-5.9x); one 16-row tile per warp with split-K across blocks (L2 hit 53% -> 26%, DRAM
read 48 -> 70 MB, 2.7x: IQ4_KS rows are 4-byte aligned, so a warp's lone 64-byte row chunks waste sectors that
neighbouring warps shared before); 8-block groups (ffn_down 1.53x -> 1.45x, ffn_up 1.29x -> 1.72x: 20 groups over
8 warps).

```
8-row verify step (phase timer)   MMQ only 31.7 ms   first matvec 30.9 ms   + vector smem stores 30.4 ms
```

Implication: about 4% of the step, IQ4_KS only. Taking the matvec to a 1-row cost needs 16-byte loads, so a weight
layout with aligned blocks (a load-time repack of IQ4_KS, or ExLlamaV3's fragment order), which every kernel
reading the type would have to accept; and Q5_K/Q6_K/Q4_K versions for the rest of MMQ's 16 ms.

## Run 8 - 2026-10-20 07:00

Question: does prefetching more groups (cp.async, no registers) fix the matvec's latency bound, and what does
ExLlamaV3's kernel actually do differently (read first-hand this time, exllamav3 a3bfbfd,
`exllamav3_ext/quant/exl3_gemv_kernel.cuh`)?

cp.async 3-stage pipeline into the warp tile: ffn_up 1.29x -> 1.27x at 108 registers (2 blocks/SM); capped for 3
blocks (80 registers) 1.31x, for 4 (64) 1.77x. Reverted.

ExLlamaV3's small-m GEMV: the EXL3 format is tile-major, a 16x16 weight tile one run of 8 x bits words (32 at
4 bpw), so a warp reads a tile with one coalesced 128-byte load (a word per lane) and a lane gets its decode
inputs by one shuffle; no shared-memory staging, no barrier in the main loop. A register prefetch ring 2-4 k-slices
deep costs a few registers because a tile is one word per lane (64 registers, two 512-thread blocks per SM).
Warps split K with no syncs; a block covers 32-64 output columns; one shared-memory reduction at the end; fp16
m16n8k16 with the weights as the B operand, two per tile.

Implication: the row-major IQ4_KS layout (rows 4-byte aligned) is what forces our staging and costs the prefetch
its registers. Next: prototype a tile-ordered IQ4_KS (same bytes, regrouped per 16 rows in mma fragment order) and
a ring-prefetch matvec for 1-8 rows, in the bench, before any integration.

## Run 9 - 2026-10-20 09:30

Question: does a tile-ordered IQ4_KS with ExLlamaV3's kernel structure (coalesced word loads straight into the int8
mma fragments, a register prefetch ring, warps splitting K with no syncs) reach the 1-row cost at 2-8 rows?

Layout (`fast_mmvq::iq4_ks_to_tiled`, the same bytes): per 16-row group the rows' f32 scales, then per superblock
the block scales as [block][row] bytes and per 32-value block two 128-byte halves in A-fragment lane order. Kernel
`mmvq_tc2_iq4_ks_kernel<KSPLIT, GROUPS, PREFETCH>`: per lane per block 3 word loads (2 A halves, 1 scale word), 2 shuffles,
1 mma. Bench on recipe A, rows 1-8, results against MMQ within 5e-3 (q8_1 activations):

```
time / 1-row MMVQ at 8 rows     MMQ     tiled 8x2x4 (rows 1 -> 8)   best (KSPLIT, GROUPS, PREFETCH)
ffn_down 5120x17408             1.45    1.06 -> 1.08                1.03 (16, 1, 4)
attn_qkv 10240x5120             1.41    1.12 -> 1.13                1.06 (8, 4, 2)
attn_gate 6144x5120             1.47    1.38 -> 1.31                1.17 (16, 1, 4)
ffn_up 17408x5120               1.51    1.21 -> 1.25                1.20 (8, 1, 4)
deeper prefetch (PF 8) or 4-warp blocks: 1.25-1.85, worse everywhere
```

Implication: the gate passes; per-shape configs put the IQ4_KS matvecs at 1.03-1.20x against MMQ's ~1.45x (about
-5 ms of a 35 ms verify step). At 1 row the tiled kernel trails the dp4a MMVQ on some shapes (ffn_up 1.21x), so the
repack belongs where decode is multi-row (speculation, batching). Integration: repack at load, choose the config per
shape, an MMQ loader for the tiled layout (prefill), unpack on readback.

## Run 10 - 2026-10-18 05:55

Question: integrated (tiled IQ4_KS loaded when speculation is attached, configs timed per shape at load, MMQ reading
the tiled layout for prefill), what does the repack buy end to end, and at what memory?

Integration: `LoadOptions::multi_row_decode` (set with an MTP/DFlash config) has the GGUF weight source repack 2-D
IQ4_KS weights on the host before upload (`RawGgufTensor::tiled`, type `Iq4KsTiled`); `fast_mmvq::tune_tiled` times
the 4 kernel shapes at 8 rows once per (device, rows, cols); 1-8 rows take the tiled matvec, more take MMQ's tiled
loader; embedding and readback untile. The row-major tensor-core matvec (Runs 6-7) is gone, superseded.

A/B on one binary (temporary switch, removed), recipe A, q4k DFlash 2 at 7 drafts, 32K q3 cache, code workload
2048 tokens (`spec_accept.py`, UI sampling):

```
                      decode tok/s     mean length   target verify graph   GPU memory   load
row-major (MMQ)       123.3 128.9      4.46 4.63     32.6 ms               20378 MiB    48.3 s
tiled                 146.2 144.9      4.59 4.52     27.6 ms               20364 MiB    49.6 s
```

First version tiled on the GPU (allocate the tiled copy, launch a tile kernel, drop the row-major copy): same speed
but +754 MiB, the freed row-major buffers staying in the CUDA pool. Repacking on the host removes it.

Implication: +15% decode at equal acceptance, no memory cost. Q5_K/Q6_K tensors (MMQ at 2-8 rows, ~1.45x) are the
next candidates for a tile-ordered layout of their own.

## Run 11 - 2026-10-18 06:40

Question: how does decode speed decay as the prompt grows (an 8K session, up to ~60K)?

`ctx_sweep.py` (scratchpad): distinct repo sources per size plus a design-review question, UI sampling, recipe A,
q3 cache, `--pa-context-len 65536`, master after #494. DFlash 2 q4k at 7 drafts, 1024 tokens; ms/step = mean length
/ tok/s:

```
prompt   decode tok/s   mean length   ms/step   prefill tok/s
    54   161.2          4.68          29.0       761
  2007   104.7          3.14          30.0      1568
  3832   116.8          3.64          31.2      1500
  7267   111.6          3.72          33.3      1439
 15016    89.9          3.44          38.3      1339
 29055    69.1          3.26          47.2      1186
 52590    57.4          3.53          61.5       998
```

Acceptance drops from the short prompt (a different task) to ~3.3-3.7 and then holds; the step cost does the decaying.
Split with phase timings (512 tokens per size) and against no speculation:

```
prompt   no-spec tok/s (ms/step)   DFlash tok/s, mean length   target verify graph   drafter graph
    54   50.1 (20.0)               140.3, 4.38                 26.6 ms               3.91 ms
  7267   47.2 (21.2)                90.7, 3.20                 30.9 ms               3.93 ms
 29055   45.0 (22.2)                65.4, 3.21                 44.5 ms               3.89 ms
 58699   41.2 (24.3)                43.2, 2.90                 62.6 ms               3.86 ms
```

Implication: at 59K a 1-row step gains 4.3 ms but the 8-row verify gains 36 ms (~8x): the verify's attention scales
with query rows, as if each row reads and dequantizes the whole KV cache by itself. The drafter is flat. At 8K the
loss is small (31 vs 27 ms); past 16K speculation's lead shrinks, and at 59K it barely beats plain decode. Next: find
the attention kernel the 2-8 row verify takes with a quantized paged cache and whether its rows share KV tiles.

## Run 12 - 2026-10-18 07:30

Question: why does the verify's attention scale with its rows, and does sharing K/V tiles across them fix it?

Cause (read from the code): the verify reaches `try_run_fattn_decode` with one block-table row and context length
per query row (`BlockTableSnapshot::from_sequence_tables` repeats each sequence's table), so fattn runs 8 one-query
decodes, instance `ncols1=1, ncols2=8` (8 Q heads of one KV group per CTA); every row's CTAs page in and dequantize
the whole quantized prefix. Fix: with several rows a sequence (and no chunked window), pass Q as (sequences, rows,
heads, dim), the first table row and the last row's length per sequence, causal; fattn then picks `ncols1=8`, one
CTA per KV head covering 8 rows x 8 heads, each K/V tile loaded once.

Phase timings, 512 tokens per size (same workload as Run 11):

```
prompt   target verify graph, per row -> per sequence   drafter
    54   26.6 -> 26.4 ms                                3.9
  7267   30.9 -> 27.1 ms                                3.9
 29055   44.5 -> 29.3 ms                                4.0
 58699   62.6 -> 32.3 ms                                4.0
```

Run 11's sweep again (1024 tokens), ms/step = mean length / tok/s:

```
prompt   before: tok/s, ms/step   after: tok/s, mean length, ms/step
    54   161.2, 29.0              136.8, 3.94, 28.8
  2007   104.7, 30.0              131.6, 3.83, 29.1
  3832   116.8, 31.2              117.8, 3.47, 29.5
  7267   111.6, 33.3              129.0, 3.84, 29.8
 15016    89.9, 38.3              108.7, 3.30, 30.4
 29055    69.1, 47.2               95.7, 3.06, 32.0
 52590    57.4, 61.5               92.8, 3.17, 34.2
```

Greedy A/B on one binary (temporary switch, removed), 3 prompts x 400 tokens: outputs split at a near-tie token
after 160-280 characters in both directions, acceptance 4.12/4.55/5.80 per sequence against 5.41/3.88/5.87 per row:
summation order, no direction. `verify_rows_over_a_quantized_cache_match_attention_over_its_values` pins the path at
8 rows over q3 and windowed caches; giving each sequence its first row's length fails it and the existing
multi-token test.

Implication: the step cost is now flat to within 5 ms out to 53K (was +32 ms); tok/s at long context follows
acceptance. At 53K DFlash runs 93 tok/s against 41 without speculation.

## Run 13 - 2026-10-18 10:30

Question: speed out to the full 262K context, for the model card, at the serving settings (recipe A, q3 cache,
`--pa-context-len 262144`, DFlash 2 q4k at 7 drafts; master after #495).

`ctx_sweep.py`, 1024 tokens with DFlash, 512 without:

```
prompt    DFlash tok/s   mean length   prefill tok/s     no speculation tok/s   prefill tok/s
    54    133.2          3.94          771               45.2                   635
  7267    130.1          4.02         1401
 15016    105.9          3.37         1307
 29055     96.8          3.18         1160               40.5                  1130
 58699     95.9          3.42          929
121221     94.4          3.97          667               31.7                   627
173804     68.2          3.15          552
261062    CUDA_ERROR_OUT_OF_MEMORY mid-request (HTTP 500)  24.6                   378
```

Peak GPU memory: 24,054 MiB with DFlash (the load measures 23,178), 20,989 without.

Implication: the 262K DFlash setup loads but cannot serve a ~261K-token request: something on the speculative path
grows with context past the load-time reservation, while plain decode serves the full window in 21 GiB. Run 5's "fits
262K" was a load-time measurement. Until that growth is found, the usable DFlash context is somewhere between 174K
and 261K. Next: find what allocates with context on the DFlash path (prefill chunks capture target hidden states for
the drafter; its context KV is bounded), then retest at the full window.

## Run 14 - 2026-10-18 13:00

Question: what fills the card in Run 13's out-of-memory, and how does the 262K DFlash setup keep enough headroom?

Findings, in order:

- Not prompt length alone: the same 261K request passed on a fresh server under Nsight, and a replay of Run 13's
  whole sequence (8 requests, 1024 tokens each) passed too, memory flat between requests (23,635-23,960 MiB).
- Not a context-sized allocation: Nsight memory tracking on a 121K request shows net allocations during the request
  peak at 156 MiB and stay flat through prefill; the stream-ordered pool is trimmed to zero after each chunk.
- Other GPU clients move: the desktop (compositor, Plasma, Chrome's GPU process) holds 470-1,260 MiB, and the screen
  locker's greeter adds 301 MiB while the screen is locked. The card's usable capacity is 24,145 MiB (PyTorch's
  `mem_get_info`), not nvidia-smi's 24,576, so about 440 MiB was really free after the old load.
- The DFlash windowed pool over-allocated: 9 live slots (one per GDN pool slot, the pool's initial capacity) where
  the reservation counted `max_seqs + 1` = 5, and 17 prefix checkpoints (`--prefix-cache-n + 1`) where the GDN pool
  can hold at most 8 snapshots, which a restore also needs. 12 slots x 41 MiB never usable.
- `--max-seqs 1` saves 75 MiB (the pool stayed at 9 slots); `--prefix-cache-n 4` 480 MiB.

Fix: `dflash::windowed_kv_capacities` sizes live slots as served sequences + the graph pad slot and retained
prefixes as `min(--prefix-cache-n, GDN pool capacity - pad)`, for both the reservation and the allocation.

```
                                 live / checkpoint slots   our process at load
before                           9 / 17                    22,441 MiB
after                            5 / 9                     21,949 MiB (-492)
```

Verification at the full window: a 260,929-token prompt plus 1,024 generated tokens, DFlash at 7 drafts, with a
PyTorch process holding 650 MiB beside the server (its 256 MiB context plus 400 MiB, about the locker plus Chrome
growing): passed, card free 233 MiB at the request's start, peak 24,007 of 24,145 MiB; decode 50.6 tok/s at mean
length 2.76, prefill 826 tok/s (the earlier serve had cached part of this prompt). Without the hog it peaks at
23,345 MiB. Before the fix the same hog would have left the request about 490 MiB short.

Implication: the 262K DFlash setup now serves the full window with ~650 MiB to spare for other GPU clients. More
headroom: move the desktop to the integrated GPU, or lower `--prefix-cache-n` (8 retained prefixes cost ~41 MiB
each in the drafter beside the GDN snapshots).

## Run 15 - 2026-10-18 21:00

Question: after #494/#495, where does a DFlash step's time go, and is a Q5_K/Q6_K tensor-core matvec the next lever?

Nsight (`--cuda-graph-trace=node`), recipe A, q4k DFlash 2 at 7 drafts, the code workload; decode phase only
(prefill chunks excluded), per draft-and-verify step (~34 ms):

```
tile-ordered IQ4_KS matvec + its activation quantize   19.9 ms   (~13 GB at ~70% of peak bandwidth)
MMQ Q6_K, output head, 2 calls (verify + drafter)       2.9 ms   (1.46 ms per 1 GB pass, ~75% of peak)
MMQ Q4_K, drafter projections, 20 calls                 1.7 ms   (small, latency bound)
bf16 GEMV, 96 calls (ssm_beta, ssm_alpha per GDN layer) 1.3 ms   (~13 us each, latency bound)
GDN speculative recurrence                              1.5 ms
MMQ Q5_K, attn_v                                        0.14 ms
```

Implication: a Q5_K/Q6_K tensor-core matvec saves at most ~0.7 ms (the head already reads at ~75% of peak); the levers
are the head read twice, the IQ4_KS matvec's last 15-30% of bandwidth, and latency-bound small calls.

A vocabulary-pruned drafting head (the first N token ids, a free row slice of the Q6_K head): emitted tokens outside the
first N ids on four workloads (code, explanation, story, Spanish; UI sampling), 2303 tokens:

```
N        16K     32K     65K     100K    152K
outside  27.0%   16.9%   9.4%    3.4%    3.4%   (152K still misses <|im_end|>, id 248046)
```

Qwen's ids are not frequency ordered enough: a 32K head could not draft one emitted token in six. A frequency-ranked
set would need a gathered copy (~137 MB at 32K, tight at 262K) and probabilistic-draft plumbing for at most ~1.2 ms per
step before its acceptance cost: dropped.

The 96 GEMVs: GGUF weights load through a weight source, which `load_packed_weights` skips, so a GDN layer's dense
`ssm_beta`/`ssm_alpha` stayed two matmuls. `MergedDenseProjection::merge_unquantized` concatenates loaded dense
projections and rebuilds them as views (keeping any dynamic-LoRA wrapper through `preserve_dynamic_lora`), so the
originals' memory goes. Verify graph 26.4 -> 25.8 ms (phase timer, 409 steps); drafter unchanged at 3.9 ms.

## Run 16 - 2026-10-18 21:40

Question: how much bandwidth does the tile-ordered IQ4_KS matvec leave, and what limits it?

Bench (`small_batch_mmvq_vs_mmq`, recipe A tensors, 8 rows; the 1-row dp4a MMVQ for comparison), now with a column per
kernel config:

```
shape                     tiled, 8 rows      dp4a MMVQ, 1 row   configs 8x2x4 / 16x1x4 / 8x4x2 / 8x1x4 (us)
ffn_up    17408x5120      677 GB/s           779 GB/s           77.8 / 80.4 / 75.5 / 72.0
ffn_down  5120x17408      704                724                73.2 / 70.4 / 77.1 / 73.3
attn_qkv  10240x5120      673                714                44.5 / 48.0 / 41.2 / 44.7
attn_gate 6144x5120       539                621                35.3 / 31.3 / 37.3 / 32.2
ssm_out   5120x6144       603                655                28.2 / 29.4 / 29.8 / 29.1
```

Nsight Compute, ffn_up at 8 rows (tuned config 8x4x2): 664 GB/s (73% of peak), 127 registers per thread, 2 blocks per
SM, theoretical occupancy 33%, achieved 29%, 1.66 waves; issue slots 43% busy, IPC 1.92; stall samples long scoreboard
1649, wait 873, selected 653, not selected 452, math pipe throttle 326. attn_gate: 481 GB/s, 96 blocks, 0.59 waves,
achieved occupancy 19%. But configs with 4x the blocks (one row group per warp) are slower there (35-37 us vs 31), so
grid fill is not the limit; per-warp latency and instruction count are (~100 warp instructions per 16x32 tile).

A register cap for 3 blocks per SM (`__launch_bounds__(threads, 768 / threads)`): 8x4x2 spills (ffn_up 128 us), the
others unchanged (best per shape 71.2 vs 72.0 us): reverted.

Implication: the remaining ~10-20% needs a redesign of the inner loop (fewer instructions per tile, e.g. decoding the
4-bit table once per two halves, or a different split of decode and mma across warps), worth ~1.5-2.5 ms of a ~31 ms
step at best.

## Run 17 - 2026-10-18 23:30

Question: can the tile-ordered IQ4_KS matvec's inner loop close its gap to the 1-row MMVQ (~780 GB/s)?

SASS of the bf16 8x4x2 kernel: ~100 instructions per 16x32 tile, 128 `LDG.E.CONSTANT` in the unrolled loop: the
4-bit table (`iq4k_values`, a `__device__` array) loaded as four words per decode, two decodes per tile.

Changes, each benched (recipe A tensors, 8 rows, best config per shape) and checked by
`tiled_iq4_ks_matches_the_row_major_weight` (random scale bytes, so both table halves):

1. The table in immediates: IQ4_KS's upper table half is the lower plus 4 at every index (-127/-123 ... 113/117), so
   rows on the upper half decode with the lower table and add 4 x the block's activation sum, which
   `mmvq_tc_quantize` now emits per row (8 int32 per 32-value block). Table loads gone; best times within noise to
   -4% (ffn_up 72.0 -> 70.4 us, attn_qkv 41.2 -> 39.7).
2. Activations prefetched with the weights (B fragment, row scales, row sums in the same ring): ncu ffn_up DRAM
   throughput 72.9% -> 78.9%, "wait" stalls 873 -> 467; ffn_up 68.6 us.
3. Offsets within a row group (the same for every group, under 32 bits) computed once per block instead of a 64-bit
   multiply per load, and the ring slot used in place (refilled after the block's math): IMAD.MOV 142 -> 54 per
   function, registers 8x2x4 104, 8x1x4 77 (3 blocks per SM).

```
shape                   before (Run 16)       after
ffn_up    17408x5120    72.0 us (677 GB/s)    64.4 us (735 GB/s)
ffn_down  5120x17408    70.4                  64.5    (734)
attn_q    12288x5120    51.5                  46.9    (713)
attn_qkv  10240x5120    41.2                  38.8    (718)
attn_gate 6144x5120     31.3    (539)         25.9    (645)
ssm_out   5120x6144     28.2                  25.0    (668)
```

Deeper prefetch (8x1x8, 8x2x6, 8x1x6) matches but does not beat these; slot 16x1x4, now worst everywhere, is
replaced by 8x1x6. Target verify graph (phase timer, 441 steps) 26.4 -> 24.34 ms.

Drafter (Run 15's 20 Q4_K calls per step, 1.7 ms): already fused per layer (`qkv_proj`, `gate_up_proj`), 4 dependent
matmuls x 5 layers, ~1 GB at ~600 GB/s: bandwidth, not launch latency; a tensor-core Q4_K matvec might save 0.2-0.3 ms.

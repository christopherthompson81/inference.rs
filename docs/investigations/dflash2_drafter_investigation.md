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

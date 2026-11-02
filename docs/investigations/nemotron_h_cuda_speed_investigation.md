# Nemotron-H CUDA speed

Nemotron 3.5 Transcribe's text GGUF (IQ4_NL up, Q5_0 down, Q8_0 rest, 19.30 GiB) on an RTX 3090, against llama.cpp
970b51ce7 on the same file. Builds are the dev profile with debug assertions and overflow checks off.

## Run 1 - 2026-11-02 ~03:20

**Question:** where do we stand on the GPU?

**Commands:** `llama-bench -m <text gguf> -p 512,2048 -n 256 -ngl 99 -fa 1`; ours `inference serve --max-seq-len 8192
--format gguf -m <dir> -f <text gguf>` with a script sending a 539-token prompt three times, a 2436-token one twice
(max_tokens 1) and 256 decoded tokens twice (`ignore_eos`).

**llama.cpp:** pp512 4342 +- 136, pp2048 4345 +- 42, tg256 198.2 tok/s.

**Ours:** 539-token prompt 4856 tok/s warm (651 cold). The 2436-token prompt was refused: "CUDA memory pressure
prevented prompt admission requiring 379584512 bytes of transient workspace", with 23.7 GiB in use.

**Why:** Granite's model reported plain `ModelConfigMetadata`, so the paged cache (a) held KV for all 29 layers, the 23
Mamba ones included (2775 MB for 97,952 tokens, ~29 KB a token where the 6 attention layers need ~6 KB), and (b) knew
no prefix-prefill features, so a prompt reusing a cached prefix (the chat template's first block) planned the gather
path, whose workspace grows with query x context. The planner also reserved only 64 MB for activations (the GGUF
estimate counts one attention chunk's scores), leaving no room for it.

## Run 2 - 2026-11-02 ~03:50

**Change:** Granite (and the Transcribe, Granite Speech and Granite 4 Vision wrappers) report a
`HybridPagedKvCacheConfig`: only attention layers paged, their recurrent state bytes, and uniform prefix-prefill
features when no layer has sinks.

**Result:** the same 2669 MB now holds 455,456 tokens. 539-token prompt 4856 tok/s; 2436-token prompt 721 / 724
tok/s (admitted, but 6.7x slower per token than the short one and 6x slower than llama.cpp's pp2048); decode 55.2 /
55.1 tok/s (llama.cpp 198). The 64 MB activation estimate is unchanged.

**Correction (Run 3's profile):** the 539-token rate (4856 tok/s) was a prefix-cache hit, the script sending the same
text with a different last line. A fresh 959-token prompt runs at 642 tok/s: prefill was ~650-720 tok/s at every
length, 6-7x behind llama.cpp. Later runs open every prompt with a distinct line (`gpu_fresh.py`).

## Run 3 - 2026-11-02 ~04:30

**Question:** where does a prompt's GPU time go?

**Command:** `nsys launch --trace=cuda,nvtx --cuda-graph-trace=node inference serve ...`, then `nsys start` / `stop`
around one request each (a plain `nsys profile` stopped by SIGINT kept only the first 1.26 s of CUDA records), and
`nsys export --type sqlite`.

**959-token fresh prompt (1.52 s of kernels, 1.25 s busy):** `indexed_moe_forward_q5_0_q8_1` (the Q5_0 down
projection) 45.5 % at 8.2 ms a layer per chunk, IQ4_NL up as `gather_rows_u8` + `dequantize_superblocks` + BF16 gemm
per expert 21 %, elementwise (bmul, copy2d, fast_sum) 13 %, the SSM scan 3.8 %. The 2436-token prompt looks the
same over six prefill chunks.

**Why:** ungated experts on CUDA always took the gather path: the indexed matvec per (token, expert) pair, decode's
kernel, at every batch size. Gated experts have had grouped mmq for prefill.

## Run 4 - 2026-11-02 ~05:00

**Change:** ungated experts take grouped mmq for prefill (32+ tokens, no LoRA, not calibrating): routes sorted by
expert on the GPU, up through `grouped_moe_mmq`, the activation, down through `grouped_moe_mmq`, the weighted reduce.
The dispatch-table setup is shared with the gated path (`grouped_dispatch`). GPU case
`ungated_grouped_prefill_matches_gather` (Q4_0 / Q5_0 and Q4_K / Q8_0 against the gather path) passes.

**Result (`gpu_fresh.py`, no prefix reuse):** prompts of 592 / 530 / 1149 / 2321 / 2345 / 4398 tokens at 1298 / 1468
/ 1526 / 1721 / 1757 / 1788 tok/s (was ~650-720); decode 54.0 / 54.7 tok/s (unchanged). llama.cpp: 4342 / 198.

## Run 5 - 2026-11-02 ~05:40

**Question:** does a tiny Nemotron-H score on CUDA as its decode does (the CPU tests never ran it on the GPU)?

**Finding:** no, by up to 2 nats, with or without quantization, prefix caching or the grouped path. Bisecting the
tiny model's layer list: attention-only models agree with the CPU to 0.001, Mamba ones differ by up to 1.25 nats.
`ssm_scan_kernel` gives each warp lane `ceil(d_state / 32)` state items with no bounds check, so a `d_state` below 32
(the tiny model's 8) has lanes 8-31 read and write past the head's state row and add other heads' B and C into the
sum. The real model's 128 never hit it. With the guard: Mamba-only models match the CPU to 1.3e-6, the whole tiny
model to 2.5e-4.

Quantized (Q8_0) on CUDA, prefill and decode differ by 0.002-0.035 nats whether or not the experts run grouped (the
dense projections' prefill dequantizes to cuBLAS while decode quantizes the activations to Q8_1), so the CUDA case
`a_long_prompt_on_cuda_scores_as_its_decode` is unquantized at 1e-3; grouped mmq is pinned against the gather path by
the experts' GPU case.

**Real model (`audio_kld.py cmp`, 24 clips, 2801 positions, GPU pair against the BF16 CPU reference):** mean KLD
0.005451, same top-1 99.536 % (before Runs 2 and 4: 0.004948, 99.357 %). IQ4_NL up now quantizes its activations to
Q8_1 inside mmq, as llama.cpp does, where it dequantized the weights to BF16 for cuBLAS before.

## Run 6 - 2026-11-02 ~06:30

**Question:** what is left after grouped mmq (2436-token prompt, 1.61 s of kernels, 1.12 s busy)?

**Profile:** `mul_mat_q` 39 % (experts and dense), then 56,028 each of `bmul_bf16`, `fast_sum_small_bf16` and
`copy2d_bf16` (~406 per Mamba layer per chunk, one per token): on CUDA the Mamba conv ran per timestep (narrow the
window, multiply by the weight, sum), only the CPU having a conv kernel. The scan 10 %.

**Change:** GDN's fused conv kernels (prompt, width-4 tiled, decode step) take an optional bias before their SiLU, and
Granite's Mamba conv runs them on CUDA for prompts and decode steps. GPU case `causal_conv1d_bias_matches_the_cpu_conv`
(widths 3 and 4, prompt and step); tiny model `a_long_prompt_scores_on_cuda_as_on_the_cpu` (the same checkpoint's
logprobs on both devices, 1e-3).

**Result (`gpu_fresh.py`):** 592 / 530 / 1149 / 2321 / 2345 / 4398 tokens at 2092 / 3252 / 2623 / 3158 / 3326 / 3302
tok/s (Run 4: ~1300-1790); decode 55.6 / 55.8 (unchanged). llama.cpp pp512 4342, tg256 198.

## Run 7 - 2026-11-02 ~07:10

**Question:** where does a decode step go (64 tokens, 1.61 s of kernels, 0.98 s busy)?

**Profile:** the IQ4_NL up ran `gather_rows_u8` + `dequantize_superblocks` + a BF16 gemm once per routed expert per
layer (~158 a token, 31 % of busy time): the gather path, since IQ4_NL (raw storage) has no `indexed_moe_forward`
kernel. Q5_0 down took `indexed_moe_forward_q5_0_q8_1`. The lm head `mmvq_gguf_q8_0_f32_plain` 0.43 ms a token, then
~1.7 ms idle before the next step's embedding (`get_rows_q8_0`).

**Change:** `fast_mmvq::moe_decode_ungated`: the expert-indexed matvec gated decode uses, with the up projection
alone (no fused GLU), the activation as tensor ops on its F32 rows, Q8_1 quantization, then down; `moe_decode` and it
share one body. Ungated experts take it for up to 8 tokens. GPU case `ungated_indexed_decode_matches_gather`.

**Result (`gpu_fresh.py`):** decode 91.0 / 91.4 tok/s (was 55.6 / 55.8); prefill unchanged (2027-3317). llama.cpp
tg256 198.

## Run 8 - 2026-11-02 ~07:40

**Question:** the decode profile after Run 7: 1157 ms over 64 tokens under nsys, 625 ms busy, ~1750 kernels a token.
What are they?

**Profile:** `bmul_f32` 160 a token, `cast_bf16_f32` 208, `uexp_f32`, `badd_f32`...: the Mamba decode step ran as ~20
F32 tensor ops a layer (dA, dB, the state update, the dot with C, the D skip), the scan kernel only taking prompts.

**Change:** a decode step runs the scan kernel as a one-token chunk (B and C expanded to the heads as for prompts),
at decode's dt floor (`time_step_min`; prompts clamp at `prefill_time_step_min`).

**Result (`gpu_fresh.py`):** decode 114.5 / 113.5 tok/s (was 91); 4398-token prompt 3294 tok/s. Tiny-model CUDA cases
pass (decode against one prefill, CPU against CUDA).

## Run 9 - 2026-11-02 ~07:55

**Question:** what is left of a decode step at 114 tok/s (8.8 ms a token; llama.cpp ~5 ms)?

**Profile (64 tokens under nsys, so slower):** per token 12.1 ms wall, 6.6 ms of kernels, 1348 kernels. Of the kernel
time: dense Q8_0 matvecs (`mmvq_gguf_q8_0_bf16_plain`, ~96 a token at 21 us: Mamba in/out, shared experts,
attention) 2.0 ms, the experts' indexed matvecs (Q5_0 down, IQ4_NL up) 1.1 ms, the F32 lm head 0.43 ms, ~230 BF16-F32
casts 0.5 ms, then small elementwise ops and Q8_1 quantizations. Between steps ~1.6 ms idle from the lm head to the
next embedding (the logits' copy to the host, 0.6 ms a call including the wait, then sampling and scheduling).

**Implications:** the GPU idles ~45 % of a step under nsys (less without it): launch overhead over ~1350 kernels.
llama.cpp runs its decode as a CUDA graph. Granite does not opt into decode graphs (`supports_cuda_decode_graphs`),
and its Mamba cache replaces its conv and SSM state tensors every step where a captured graph needs fixed addresses,
so graphs need the recurrent state updated in place in a pool, as Qwen3.5's GDN layers do. Short of that: fewer
casts (the Mamba step's F32 round trips) and fusing the elementwise ops.

## Run 10 - 2026-11-02 ~09:00

**Question:** do Granite's hybrid decode steps capture as CUDA graphs as they are?

**Finding:** yes. On CUDA the recurrent pool's gather is `index_select` and its scatter the `copy_rows` kernel, both
by the device slot indices the graph machinery swaps, so opting in (`supports_cuda_decode_graphs`) was the whole
change for the real model: 4 graphs (buckets 1-4) captured in 194 ms, decode 163.5 / 163.7 tok/s (was 114).
200-token greedy completions of three prompts are identical with graphs on and off (ids and logprobs to 3 decimals);
three concurrent requests diverge from single ones at near-ties as much with graphs off (batched BF16 rounding).
Four "CUDA error with no caller" logs after capture are evicted, never-replayed graph outputs whose free fails as
`drop_cuda_graph_entry_output` expects: the first prompt's memory check evicts the precaptured graphs (the KV cache
leaves the planner's 64 MB activation estimate free, Run 1), and they are captured again lazily.

**Opt-in limits:** a tiny checkpoint's capture failed on a host copy: unquantized ungated experts group their routes
on the host, as do Granite's own experts (`GraniteMoE`). Granite opts in only when every MoE layer's experts decode on
the device (`MoEExperts::decode_is_device_only`: quantized ungated stacks the indexed matvec reads, or any gated
backend).

**Two flakes on the way:**
- The fused MoE router, with fewer experts than a warp's 32 lanes, scored the padding lanes as sigmoid(-inf) = 0;
  a selection bias leaving every expert below 0 made a padding lane the pick (expert 4 of 4), about one tiny run in
  six. Fixed in the kernel, with a GPU case.
- Decode-against-prefill on the CPU (Q8_0) differed by up to 1.4e-3 about once in 30 full-group runs, never alone:
  the scoring request could reuse the prefix cache's state from the decode steps. The checks now run without a
  prefix cache: 0 failures in 40 runs of the group.

**Tests:** `a_long_prompt_scores_as_its_graph_decode` (BF16, Q8_0 experts, head dim 64 so FlashInfer takes the
attention) replays graphs and scores within 0.5 of one prefill; over 30 random checkpoints the gap is p90 0.054,
worst 0.29 with graphs and p90 0.056, worst 0.18 without, so it bounds stale-state replays, not rounding.

**Result (`gpu_fresh.py`):** prompts 2107-3307 tok/s, decode 163.7 tok/s (llama.cpp pp512 4342, tg256 198).

## Run 11 - 2026-11-02 ~10:00

**Question:** why are the precaptured decode graphs evicted (Run 10), and only for this model?

**Finding (`/metrics` after load, before any prompt):** `inference_cuda_memory_pressure_total{level="reserve"}` 5,
`inference_cuda_graph_evictions_total{reason="memory_pressure"}` 4, 350 MB available against a required 512 MB. The
default paged cache holds one sequence at the model's context, or what is free when that is less; for Nemotron-H's
context it is less, so the cache took everything but the cap's 512 MB reserve, and what loaded after it (graphs,
workspaces) left free memory under the engine's floor (total / 128 within 512 MB-2 GB, the same 512 MB on a 24 GB
card). Every idle check and prompt then reclaimed the graphs. Qwen2.5-Coder-3B's full context fits in 1152 MB, so its
graphs stay.

**Change:** a default-sized cache on CUDA reserves the engine's free floor as a post-load component
(`cuda_free_floor_bytes`, now shared by the planner and the engine's thresholds). A cache that fits beside it, or one
sized by the user, is unchanged.

**Result:** 512 MB reserved, the cache at 2157 MB (368,096 tokens, was 455,456), 880 MB available after the run;
4 captures, 510 replays, no pressure events or evictions over `gpu_fresh.py`. Decode 167.3 tok/s, prompts unchanged.

## Run 12 - 2026-11-02 ~11:00

**Question:** what does a decode step's Mamba layer launch now (167 tok/s)?

**Profile (one layer between two scans, from Run 9's trace):** 57 kernels a Mamba + MoE block. Before the scan:
`a_log` cast, exp and neg, `D` and `dt_bias` casts every step; B and C cast and copied out to every head; the state
gathered from the recurrent pool (`is_u32_f32`, 11.8 us) and scattered back after (`indexed_row_copy`), the conv
state too. After it, the grouped gated RMSNorm as ~11 elementwise ops.

**Change:** the scan kernel reads B and C by group and takes optional slot indices, updating those rows of a
recurrent pool in place (a negative slot is a padding row: no state write, output 0); its wrapper takes the CPU scan's
`SelectiveScan`. Granite keeps the scan's F32 constants from load. A CUDA decode step with an F32 state pool and conv
pool in the activation dtype runs the conv update and the scan on the pool's rows, without the gather and scatter.
GPU case `cuda_selective_scan_matches_the_cpu_scan_gathered_and_in_a_pool` (40-wide state, a padding row, untouched
rows checked).

**Result:** decode 200.8 / 200.8 tok/s (was 167; llama.cpp tg256 198); prompts 3463-3469 at 2.3-4.4K tokens. Greedy
completions of the three prompts identical to Run 10's eager reference (ids and logprobs to 3 decimals).

**Found on the way, not fixed:** `a_long_prompt_scores_as_its_graph_decode` fails about once in 20-60 runs, master
too (1 in 60). Scoring the sequence twice gives the same prefill both times, so the decode is off, and only at one
step (step 1, 2 or 4 of five, by 0.5-1.9 nats, the steps after it within 0.03), every step a graph replay (4 replays,
cache hits). A one-step error that does not carry into later steps is not the recurrent state or the KV cache; it
looks like the step's reported logprob, possibly read after the next replay (the engine runs one step ahead) has
overwritten the graph's logits. The tokens match, so greedy decoding is unaffected.

## Run 13 - 2026-11-02 ~12:00

**Question:** is Run 12's one-step decode error a graph-replay bug or test noise?

**Code reading (subagent, read-only):** with logprobs requested no lookahead step launches: `Sampler::cuda_sampling_plan`
returns `None` when `return_logprobs` is set (inference-nn/src/sampler.rs ~937), so `try_submit_cuda_token_batch`
declines and the step samples eagerly, the logits copied to the host (`to_vec1`, a blocking copy on the graph's
stream) and the token and logprob both taken from that copy. The graph's logits buffer is reused across replays, but
the next replay cannot launch before that copy returns. So it is not the logprob read racing the next step.

**Command:** the graph case with `INFERENCE_RS_CUDA_GRAPHS=0`, 100 runs (replay assertion skipped).

**Result:** 0 failures in 100 with graphs off, against about 1 in 30 over the loops with graphs on (master included):
at that rate 100 clean runs happen ~3 % of the time, so replays are implicated. Eager decode and replays run the same
kernels, which argues against routing near-ties. A step that is off by 0.5-1.9 nats while the steps after it are
within 0.03 fits a replay reading one stale piece of per-step attention metadata (a position, context length or slot)
better than stale recurrent indices, which would carry into every later step. Not yet tested; next is a debug mode
that runs each replayed step eagerly too and compares the logits, to catch the bad step and diff its inputs.

## Run 14 - 2026-11-02 ~13:00

**Question:** what does a failing replay read wrong?

**Probes (temporary):** after each replay, synchronize and read back the graph's inputs (token ids, positions, state
indices, and every metadata buffer the staging wrote: slot mappings, block tables, context lengths, FlashInfer
indptr / indices / tile plans, rope positions) against the values staged, and the step's top token and logprob from
the graph's own logits; the capture and replay threads. A failing run: every input as staged, capture and replays on
one thread, all kernels and copies on one stream, and the graph's own logits already give the wrong logprob (so not a
later read of them).

**Decisive check:** the test keeps a failing checkpoint; rerun on it, it fails 10 of 10 with identical values, and with
`INFERENCE_RS_CUDA_GRAPHS=0` it fails identically, bit for bit (step 5: decode -2.5467, prefill -3.6523). Run 13's "0 in
100 with graphs off" was an unlucky draw of checkpoints: the test made new random weights every run, and about one draw
in 30 has a step where decode (indexed matvec, Q8_1-quantized activations) and prefill (grouped mmq) land on opposite
sides of a near-tie, most likely in the experts' routing. Not a graph bug; Run 13's conclusion is withdrawn.

**Change:** the tiny Nemotron-H's weights come from a seeded `StdRng`, as `support/recording.rs` seeds its
checkpoints, so every run checks the same weights. On them the BF16 graph case differs from one prefill by 0.023, and
its tolerance goes from 0.5 to 0.1; the other cases: CUDA F32 1.4e-6, CPU quantized 2.4e-7, CPU 4.0e-5, CPU against
CUDA 4.4e-4. 0 failures in 10 runs of the Nemotron-H and Granite tiny group.

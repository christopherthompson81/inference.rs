# Concurrent prefill on Inkling 2 Mini

Question: why do concurrent requests on the LAN chat load (Inkling 2 Mini recipe C, q8 KV, LoopCD-Logits 0.5, DFlash 2
drafter, 160K context) finish no sooner in total than the same requests one at a time? A prefill-heavy workload (2K
shared system prompt, 1-4K user messages, short JSON answers) took 22.5 s for 12 requests serially and 25.3 s at 4
concurrent (`prefix_cache_investigation.md` Run 9).

Bench (scratchpad `pf.py`, `sweep.sh`): a dev server on 18083 per setting, the chat load, N unique ~1.9K-token prompts
at once (max_tokens 1), alone or beside 2 decoders looping 48-token replies; prefill tok/s is new prompt tokens over
the batch's wall time, decoder tok/s their tokens finished in that window.

## Run 1 - 2026-11-27 11:50

Prefill chunk (`--max-prefill-chunk-tokens`, default 512) against prompts at once, 160K and 32K context:

```
context chunk  decoders  N=1    N=2    N=4   (prefill tok/s)
160K    512    0         1500   1823   1657
160K    512    2         1092    888   1651
160K    1024   0         1717   1744   1783
160K    1024   2         1274   1182   1265
160K    2048   -         CUDA_ERROR_OUT_OF_MEMORY in `moe experts forward`, every case
32K     512    0         1543   1813   1836
32K     512    2          863   1035   1823
32K     2048   0         1864   1956   1965
32K     2048   2         1805   1824   1714
32K     4096   0         1929   1935   1937
32K     4096   2         1873   1825   1892
32K     8192   0         1836   1976   1862
```

The chunk is per prompt: four prompts at 512 are a 2,048-token step, which is why N=4 nears the ceiling even at 512.
Prefill tops out near 1,950 tok/s (about 8 GFLOP a token: 48 layer visits of ~83M active parameters), reached from
2,048 tokens a step. The loss is a lone prompt: 512-token steps give 1,500-1,550 alone and 863-1,092 beside decoders,
where the scheduler runs up to 8 decode steps between its chunks. At 160K a larger chunk does not fit.

Implies: cut the prompt activations so a larger chunk fits, and look at the interleave.

## Run 2 - 2026-11-27 12:20

Card memory over one 1,898-token prompt (nvidia-smi at 20 ms, after a warm request), 32K load: the earlier
measurement was about 625 KB a token one at a time (+320 MB over 512-token chunks, `server_memory_investigation.md`).

`BlockStacks` kept every block stack a forward built for the depth mixes: with n attention-residual blocks that is
n(n+1)/2 block states a token alive to the forward's end, though each layer reads only its own block's stack. Keeping
only the latest:

```
chunk 2048: +738 MiB, ~400 KB a token, 1,877 tok/s
chunk 512:  +163 MiB, ~326 KB a token, 1,549 tok/s
```

At 160K chunk 2048 now runs prompts alone (1,837 / 1,868 / 1,900 tok/s for N = 1, 2, 4), but beside decoders it still
runs out of memory: admission (`admit_cuda_prompt`) splits a batch of prompts by their activations, never one prompt's
chunk. What is left is about the MoE estimate (~342 KB), transient per layer. The Inkling tiny tests pass.

Decode steps between prefill steps (`--max-decode-steps-before-prefill`), 160K, chunk 512, 2 decoders:

```
steps  N=1 prefill  decoders   N=2 prefill  decoders
8      1036         26         980          49
2      1403          0         1316         17
1      1426          0         1409          0
```

Prefill priority gains ~35% prefill throughput and stalls the decoders for the prompt's duration. No step mixes
decode rows with a prompt chunk in this engine, so the two always trade.

Implies: options are a chunk limit admission can lower per step (raise the default, shrink one prompt's chunk when its
activations do not fit), the prompt activation estimate brought down to the new measurement, a prefill-first
interleave while prompts wait, and mixed decode + prefill steps (the larger change).

## Run 3 - 2026-11-27 12:55

Change: a lone prompt's step now counts its activations too, and under memory pressure its chunk halves
(block-aligned, down to 256 tokens; `inference_cuda_prompt_chunk_reductions_total`) where it used to be rejected or
run out of memory; one that cannot shrink further runs on its attention workspace as before. Inkling's estimate comes
down to the measurement (moe + 32 bytes a hidden unit, ~416 KB, from 170: ~733 KB).

160K load, prefill tok/s (decoders = 2 looping 48-token replies):

```
chunk  words  decoders  N=1    N=2    N=4
2048   1800   0         1805   1844   1839
2048   1800   2         1591   1550   1506
4096   1800   0         1862   1846   1853
4096   1800   2         1753   1511   1559
4096   6000   0         1758   1766
4096   6000   2         1435   1447
```

No `out of memory` in any log (chunk 2048 beside decoders failed every case in Run 2). The 6K-word run halved 12
chunks and split 2 batches. Against the 512 default beside decoders (1,092 / 888 / 1,651) a lone prompt prefills
~45% faster; the decoders stall for the longer steps (0-20 tok/s in these windows).

Implies: memory no longer pins the chunk at 512; what the chunk should be is now the latency trade, the adaptive
scheduling's job.

## Run 4 - 2026-11-27 13:40

Change: a recurrent model's lone prompt with nothing decoding beside it took `max_prefill_chunk_tokens` (512) per step
anyway (`select_recurrent_prompt_batch`); where admission sees its activations (the model reports a per-token figure,
on CUDA) it now takes the whole token budget (4096), admission halving it under pressure. 160K, default flags: one
1.9K-token prompt 1,543 -> 1,820 tok/s, two at once 1,813 -> 1,866.

The shared-system-prompt workload (12 requests, ~2K new tokens each, json_schema answers) with that build:

```
flags                         serial wall  4-concurrent wall
default                       18.4 s       19.8 s
chunk 2048                    19.4 s       18.0 s
chunk 2048, 1 decode step     18.5 s       18.3 s
```

Little from the interleave: the workload is prefill-bound at the GPU ceiling. Split by streaming one request: TTFT
0.86-2.59 s (prefill ~1,400-1,550 tok/s from a 1,984-token cached prefix), then 10 completion tokens
(`{"choice":"test"}`) in 0.33-0.40 s, ~28 tok/s: the grammar-constrained decode of mostly forced tokens costs as much
as ~600 prompt tokens.

Decode-heavy (dec.py: 300-token essays, ignore_eos), aggregate tok/s:

```
load                   1 stream  2 streams  4 streams
DFlash 2, auto depth   90.5      122.2      150.2
DFlash 2, depth 3      85.8      122.4      175.9
no drafter             86.1      125.9      181.4
```

At 4 streams the drafter costs 17% against none, though the autotuner has a plain step (depth 0) among its
candidates per batch size; a plain step with a drafter loaded still feeds it the context, and the tuner exposes no
metric of what it chose.

Implies: by load, the levers left are speculation (draft less, or not at all, as streams grow, with the plain step's
real cost), forced grammar tokens (jump ahead without a model step), and mixed decode + prefill steps.

## Run 5 - 2026-11-27 13:15

Question: why does auto depth lose to a fixed depth at 4 streams (Run 4)?

A new counter, `inference_speculative_depth_decisions_total{depth}`, and the tuner's debug log: at 4 streams it
picked depth 2 (340 of 419 steps) and predicted ~190 tok/s, while fixed depth 2 ran 164-180 and auto 127-150. The
cost it misses is CUDA graphs: per 17 s run, auto evicted 16 graphs under memory pressure and captured 40 (fixed depth
2: none and 22), and a warm 16-request run kept evicting (123 evictions, 127-136 tok/s). The tuner drops every sample a
capture spans, so the thrash never reaches its costs. Auto keeps the full depth's graphs at every batch plus depth 2's
up to batch 8; the load reserved only the full depth's (356 MB), and the cache fills the card to its floor.

(A first theory, that a drafted step's timing missed the work after verification, was wrong: the next step is planned
before the verification is recorded, so both kinds span one cycle. Reverted.)

Fixed depths, 152K, aggregate tok/s, 300-token essays:

```
                 1 stream  4 streams
auto (0/2/7)     82-90     127-150
depth 2          81        164-180
depth 3          86        159-176
depth 7          76        121-131
no drafter       86        165-181
```

Changes: the load also reserves an auto-depth target's shallower verify graphs (Inkling: 356 -> 458 MB; evictions
123 -> 69, 144-150 tok/s), and Inkling's auto depth drafts its full depth for a lone sequence only, keeping to depth 2
or a plain step beside others, with the full depth's graphs captured for batch 1 alone. Warm runs after:

```
auto, 1 stream    84.1 / 77.7 tok/s
auto, 4 streams   166.3 / 172.9 tok/s   (2520 steps at depth 2, 498 plain, 34 at 7; no evictions)
```

Implies: at 4 streams auto now sits with the best fixed depth (+25% over before), one stream unchanged.

## Run 6 - 2026-11-27 14:00

Question: why does a 10-token json_schema answer (`{"choice":"test"}`) take ~0.35 s with the drafter?

Token arrival times (streamed, one request, 152K load) showed a ~100-160 ms gap after the first token, then verify
steps of ~35 ms landing 1-2 drafts each. Without the drafter the same answer streams a token every ~13 ms (0.11-0.20 s
in all). A constrained row is verified on the host under the grammar (each row's logits read back, a mask a row), so
a draft that may not land costs more than a plain step. llguidance reports the tokens a grammar forces next
(`compute_ff_tokens`), which the engine never asked for.

Changes: a step of only grammar-constrained sequences drafts as deep as their grammars force (the deepest captured
depth within it), else takes a plain step; the drafts' leading tokens are replaced by the forced ones; and when every
draft is forced the drafter only takes the context. The step that samples its next token before drafting counts one
forced token fewer (that token is the first forced). After:

```
                        answer (after the first token)   drafts landed
before                  0.23-0.35 s                       9 of 53
forced-only drafting    0.19-0.20 s, ~13 ms a token       10 of 10
no drafter (reference)  0.11-0.20 s
```

The gap after the first token stays (~100 ms, with or without schema or drafter). Not the prefix write-through: that
runs inside the prompt step (~50 ms a prompt, ~30 ms of it exporting Inkling's recurrent state with its window rings;
batching its host copies saved ~6 ms, reverted). Open.

Also: a lone prompt's chunk with nothing decoding is capped at 2048 (from the whole budget): one 2,944-token step ran
out of memory in the MoE experts at 152K though admission saw no pressure (graph pools claim room after it looks), and
prefill throughput levels off from 2048 (Run 1). Three runs of five after: no `out of memory`.

Test: `a_grammar_forced_reply_drafts_only_forced_tokens` (tiny Qwen3.5 + tiny DFlash, a regex forcing the reply): 0
of 46 drafts landed before, all after.

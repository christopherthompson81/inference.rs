# tqLLM comparison investigation

Question: tqLLM (Apache-2.0 C++/CUDA, github.com/turbo-tan/tqllm) reports much higher Qwen3.8-27B decode on an RTX 3090
than ours, mostly at long context. Which of its techniques explain the gap, and which are worth taking?

Our baseline (same card, recipe A IQ4_KS 14.06 GiB, q3 KV, DFlash 2 drafter q4k, 7 drafts, chain verify): no spec ~50
tok/s; DFlash ~146 tok/s on a code workload at short context, ~93 at 53K; verify step ~23.9 ms.

## Run 1 - 2026-10-19

Source read of tqllm at d617986 (cloned to ~/Programming/tqllm) plus its published numbers. No runs yet.

Published claims (tqllm.com, README; their bench excludes prefill and the first token, includes drafter/verify/host):

| Workload | tqLLM | Ours |
|---|---|---|
| No spec, 2K summary | 55.5 | ~50 |
| DFlash2, 2K / 8K / 32K wikitext summary (draft acceptance 35-40%) | 155.2 / 149.9 / 131.0 | not measured |
| DFlash2, C++ code prompt at 32K / 128K / 192K / 256K, chain verify, T4KV | 202.8 / 157.1 / 144.4 / 106.6 | ~146 short, ~93 at 53K |
| LocalMaxxing reasoning-v1 / code-v1, 32K, median of 5 | 273.1 / 199.5 | not measured |
| Prefill 2K / 8K / 32K | 2,258 / 2,426 / 2,155 | not measured |

Quality: no KLD published for TQ3_4S-v2 (14.5 GB, the file behind every 3090 speed number); only task scores. Their
TQ_XS/S/M/L files: KLD 0.079 / 0.055 / 0.045 / 0.022 at 11.85-14.72 GB, measured with tq3 KV on GB10, dataset and
context unstated. Ours: recipe A 0.0169 (15.1 GB), recipe B 0.0192 (14.5 GB), wikitext c512 x 40, weights only.

ExLlamaV3 (r0b0tlab/qwen38-exl3-dflash2, 4.00 bpw, 3-bit KV): 162.9 tok/s DFlash2 on GSM8K at 8K (acceptance 5.66),
25.3 tok/s at 150K depth (acceptance 1.82). Not ahead of us; dropped.

### Weights and matvec
- TQ3_4S: 32 weights per 16 B (4.0 bpw), fixed sign flip + 32-point Hadamard, 8-level N(0,1) code, four E3M5 scales.
  TQ3 (3.125 bpw) / TQ4 (4.125 bpw): 128-point Hadamard, one FP16 scale per 128. Quantizer not in the repo; no imatrix code.
- Activations are rotated inside the kernel that produces them (norm, swiglu, gated norm), never as a separate pass.
- Matvec decodes weights in registers to FP16 and runs FP16 mma.m16n8k16 against FP16 activations; 1-8 columns share
  one accumulator tile, up to 32 columns before GEMM. Variants autotuned per shape at load plus a per-arch rule table.
- gate|up and qkv|z|b|a concatenated at load; residual add fused with the next norm's sum of squares.
- Two-stage LM head: 4-bit pass picks top 16, exact int8 rescore with argmax on GPU.
- Prefill: per chunk, dequant to FP16 scratch, requant per row to int8, cuBLASLt int8 GEMM (chunks >= 1024 tokens;
  1640 -> 2123 tok/s on a 20.7K prompt). Matvec/GEMM crossover measured at load on the real weights.
- Loads K-quants and IQ types; no ik_llama types, so our IQ4_KS file won't load.

### Speculation
- Default is a greedy token tree (1 root + up to 7 nodes, best-first on the GPU from the drafter's top 16 per position
  and a learned pair-transition score); chain verify when sampling. One host sync per step.
- GDN rollback: two state slots per layer; full accept flips the slot, partial accept replays the accepted tokens from
  projections stashed during verify. Tree GDN kernel rebuilds each node's state from its ancestors in shared memory.
- 8-column cap comes from a 16-slot shared buffer in the replay kernel (previous path + new tree).
- CUDA graphs: verify keyed by column count and a power-of-two context bucket; commit + drafter inject captured as one
  graph (replaces ~110 launches). No adaptive draft length; drafter and verify run serially.
- Prompt lookup (`--lookup`) can add a branch under the root within the same column budget; MTP chosen per request by
  prompt length.

### Attention and KV
- Decode attention: split-K with a fixed one-wave split count (2 x SMs / kv heads), tiles interleaved across splits.
  One block per (kv head, split) serves every GQA head and every verify column (up to 48 rows) from one smem tile.
- Quantized KV decoded while copying to smem; Q rotated into K's basis and quantized to int8, scores on int8 MMA;
  P.V in FP16 MMA; V stays rotated and one inverse Hadamard runs after the merge.
- Plain decode, chain and tree verify share one attention arithmetic (bit-identical greedy, tested).
- TqCache: old KV positions in pinned host memory, staged back two attention layers ahead via CUDA VMM remapping.
- Memory fit ladder: embeddings/head to host, smaller prefill buffers, smaller KV format, KV tail to RAM, shorter context.
- Other: trellis-coded KV (tcq2 V, tcq3 K), idle-time KV compaction (not exact), MoE experts in host RAM with a VRAM
  cache, `/v1/responses` and `/v1/messages`, single-slot prompt cache only.

Caveat on their greedy verify: the target argmax is taken over a 4-bit-head top-16 shortlist re-scored in int8, which
the README says is not a certified full-vocabulary argmax.

Tree verify is greedy-only in tqLLM, and the agent workload samples, so it is out unless paired with sampled tree
verification (SpecInfer's multi-step speculative sampling: recursive rejection over a node's children, exact in
distribution). Not taken for now; the long-context attention work comes first.

Implications: the long-context gap points at decode attention (fixed-wave split-K with K/V shared across verify rows,
int8 scoring), not at layer count; the short-context gap at the matvec (FP16 MMA with fused rotation) and per-step
overhead (graphs over commit/inject). Tree verify should help most on low-acceptance prompts. Next: run tqLLM here with
its TQ3_4S-v2 file to confirm the numbers on this card, then the same prompts through our server, with per-step
draft/verify/commit times from both (`TQLLM_DEBUG=profile`).

## Run 2 - 2026-10-19 15:20

Question: our numbers on tqLLM's own benchmark prompts (relbench.py's wiki summaries), with one client clock for both
engines.

Command: our LAN binary (master 1220f2a33), `inference serve --format gguf -f Qwen3.8-27B-recipe-A.gguf
--pa-context-len 262144 --pa-cache-type q3 --mtp-model Qwen3.8-27B-DFlash2 --mtp-n-predict 7 --mtp-draft-isq q4k`;
`specbench.py` (scratchpad): the same prompt slices as relbench.py, streamed, decode = (completion_tokens - 1) / (last
delta - first delta), 400 tokens, thinking on, 3 runs, median. "ui" = temperature 0.6, top_p 0.95, top_k 20.

```
prompt (tok)   greedy decode   runs                  ui decode   runs                  first-run prefill
1988           194.6           194.6 199.3 190.9     116.4       115.9 168.0 116.4     ~1,300 tok/s
7707            96.0            96.0  87.2  97.6      97.8        97.8  97.8 105.8     ~1,290
31919           96.4            96.5  96.4  96.4      96.1        88.0  99.5  96.1     ~1,310
```

Log: DFlash acceptance 33-35% (mean length 3.3-3.4) on these summaries. Repeats hit the prefix cache, so prefill is
from the first run only.

Finding: at 2K greedy we beat their published 155.2; from 8K we sit at ~96 against their 149.9 (8K) and 131.0 (32K).
The drop is a cliff between 2K and 8K, then flat to 32K, which points at something switching with context length
(attention split / kernel choice or a graph bucket) rather than gradual KV read cost. Prefill is ~1,300 against their
~2,250. Next: the same client against tqLLM on this card.

## Run 3 - 2026-10-19 15:21

Question: where between 2K and 8K does our decode drop?

Command: specbench.py greedy at 3K/4K/5K/6K, 2 runs each, same server.

```
prompt (tok)   2948    3929   4813    5748
decode         116.4   96.9   100.7   88.7
```

Finding: no clean threshold; the numbers move with the prompt (each length is a different wiki slice, so a different
summary). Plain decode barely moves with context on this setup (Run 13 of the dflash2 doc: 45 tok/s at 54 tokens,
40.5 at 29K) and the verify step is flat to 59K after #495, so the 2K figure is most likely a high-acceptance prompt,
not a context cliff. Our engine logger prints only cumulative acceptance (33-35% overall, mean length 3.1-3.7 per
5 s window); tqLLM returns `draft_n_accepted` / `draft_n` per request in `timings`, which we lack. Per-request
acceptance and step time are what the comparison needs, so the tqLLM run should log them (`TQLLM_DEBUG=profile`).

## Run 4 - 2026-10-19 15:35

Question: what is tqLLM's 65,536-token draft vocabulary (`atx_65536.txt`), and does it cover the model's output better
than the first 65,536 ids (the pruning we tried and dropped in dflash2 Runs 15-17)?

Profile: header `llama-mtp-vocab-v1 248320 65536`, sorted ids 0..248076; ids 0-546 all kept, then a frequency-style
pick: 14,876 of ids <16K, 12,234 in 16-32K, 17,529 in 32-64K, 14,052 in 64-100K, 3,298 in 100-152K, 3,547 above, all
33 special/control ids 248044-248076 (im_end, think, tool_call) included.

Command: `vocab_cover.py` (scratchpad): ten UI-sampled outputs from our server (700 tokens each, thinking on),
re-tokenized with the Qwen3.8 tokenizer; share of tokens inside each set.

```
output        tokens  atx_65536  first 65536
code-rust        701      98.9%        98.9%
code-python      701      99.6%        97.1%
code-ts          701      96.7%        97.0%
explain          701      99.9%        99.4%
story            698      99.6%        99.1%
spanish          699      86.4%        87.1%
chinese          698      82.1%        35.4%
math             457     100.0%        97.6%
agent            701      99.7%        99.1%
summary          701      97.3%        96.1%
all output      6758      95.9%        90.5%
input: wikitext 96.2 / 97.1, repo rust 98.4 / 98.5, repo svelte 95.7 / 98.4
```

Misses: Chinese compounds, Spanish words (ocurre, luz, proceso, energia), identifiers (Props, pageSize, sortable).

Finding: on English prose, code and tool-call output it covers 97-100%, but so do the first 65,536 ids; its edge is
Chinese (82% against 35%), which is where our prefix test lost. Neither covers Spanish well (86-87%). A token outside
the set ends the draft there, so at ~1-3% misses per token a 7-draft chain loses a few percent of acceptance on
English/code, more elsewhere. The gain is the drafter head: 65,536 of 248,320 rows, ~0.26x the read (at 4 bits
~170 MB instead of ~635 MB per draft pass, ~0.6 ms of a ~24 ms step on this card, consistent with our earlier
"at most ~1.2 ms" estimate). A small, workload-dependent win, not the source of a 1.5x gap.

Held idea (user, 2026-10-19: keep it for pairing with other speedups): a per-request draft vocabulary. A fixed
English base set plus the prompt's own token ids (identifiers and names the output reuses, most of the English misses
above), gathered by index from the full head rather than copied (one 2.5 KB row per id, no extra memory), padded to a
fixed length so captured graphs stay valid. Verification still runs on the full vocabulary, so output is unchanged.
Worth ~0.6-0.7 ms of a ~24 ms step alone; more as a share once the step shrinks.

## Run 5 - 2026-10-19 15:45

Question: tqLLM on this card with its benchmark settings, through the same client as Run 2.

Command: `TQLLM_DEBUG=profile tqserve --model Qwen3.8-27B-TQ3_4S-v2.gguf --context 40960 --np 1 --reasoning on
--kv tq3 -md drafters/Qwen3.8-27B-DFlash2-TQ3.gguf --draft-vocab mtp-vocab/atx_65536.txt` (d617986, CUDA 12.8,
sm86; 21.8 GB on the card); specbench.py as in Run 2.

```
                 greedy (tree verify)                       ui sampling (chain verify)
prompt   decode  steps  ms/step: draft verify commit   decode  steps  ms/step: draft verify commit   prefill
1988     139.6   108          0.03  25.5   1.0         104.6   143          2.7   23.0   1.0          ~1,900
7707     134.0   110          0.03  26.1   1.0          98.6   148          2.7   23.6   1.0          ~2,100
31919    117.6   119          0.03  27.8   1.0          96.4   144          2.7   25.0   1.0          ~1,890
```

(tree mode queues the draft inside the verify launch, so its draft time shows inside "verify".)

Side by side with Run 2 (ours: chain, 7 drafts, q3 KV, 262K context reserved):

```
prompt   ours greedy  tq greedy   ours ui   tq ui    ours prefill  tq prefill
1988     194.6        139.6       116.4     104.6    ~1,300        ~1,900
7707      96.0        134.0        97.8      98.6    ~1,290        ~2,100
31919     96.4        117.6        96.1      96.4    ~1,310        ~1,890
```

Findings:
- Their published 155.2 / 149.9 / 131.0 reproduce about 10% lower here (139.6 / 134.0 / 117.6); the desktop shares
  this card.
- With sampling (the agent workload) we are level or ahead: 116 vs 105 at 2K, 98 vs 99 at 8K, 96 vs 96 at 32K. Their
  sampled path is chain verify at ~2.8 tokens per step (400 / 143) and a 26.7 ms step; ours is in the same range.
- Their greedy lead from 8K comes from tree verify: 3.4-3.7 tokens per step (400 / 108-119) at a similar step cost.
  Greedy-only, so not usable for the agent workload (sampled tree verify is the held option).
- Prefill is a real 1.5x gap: ~1,900-2,100 tok/s against our ~1,300 (int8 cuBLASLt GEMMs per chunk).
- Their verify step grows 9% from 2K to 32K (25.5 -> 27.8 ms greedy, 23.0 -> 25.0 sampled); the long-context claim
  (157 at 128K) is still untested here.
Next: long context, both engines, sampled, ~64K and ~128K prompts.

## Run 6 - 2026-10-19 16:00

Question: does tqLLM's long-context claim (157 tok/s at 128K) hold on this card, and how do we compare there?

Command: tqserve as in Run 5 but `--context 262144`; ours as in Run 2; specbench.py ui at 64K/128K (2 runs), greedy at
128K (1 run).

```
prompt (tok)   tq ui   tq greedy   ours ui   ours greedy   tq prefill   ours prefill (first run)
62486          87.6               79.8                    ~975         ~850
124730         91.7    95.9        71.4      59.9          ~770         ~930
```

tq per step at 124730, ui: draft 2.7, verify 31.4-32.4, commit 1.1 ms over 122 steps; greedy (tree): verify 34.3 ms
over 118 steps. tqLLM's prompt cache did not hit on repeats (same TTFT), ours did.

Finding: their 157 at 128K does not reproduce on summaries (92 sampled, 96 greedy); it was a C++ code prompt, which
drafts far better. At 125K they lead us by ~1.3x sampled. Their prefill falls off with context (1,900 -> 770 tok/s),
ours less (1,300 -> 930), so we prefill faster past ~100K.

## Run 7 - 2026-10-19 16:05

Question: our step split by context, to set against tqLLM's profile.

Command: our server with `INFERENCE_RS_CUDA_PHASE_TIMINGS=1` (the timer syncs, so decode reads ~5-15% low: 112 at
2K against 116); specbench.py ui, 1 run each at 2K/32K/128K; medians per request from the log.

```
prompt    steps (400 tok)  tok/step   target verify   drafter   sum      | tq: steps  tok/step  verify  draft+commit  sum
1988      107              3.74       26.8            3.95      30.7     |    143     2.80      23.0    3.7           26.7
31919     133              3.01       30.0            4.19      34.2     |    144     2.78      25.0    3.7           28.7
124730    145              2.76       40.4            4.19      44.6     |    122     3.28      32.4    3.8           36.2
```

Findings:
- Sampled acceptance is similar (ours higher at 2K/32K, lower at 125K; single runs, content dependent).
- Short-context verify: ours 26.8 ms against 23.0 (+3.8 ms, +17%). Part is file size (14.06 GiB against 13.5 GiB,
  ~4%); the rest is the matvec/fusion path (their FP16-MMA TQ3_4S with rotation fused into producers).
- Context growth 2K -> 125K of the verify step: ours +13.6 ms, theirs +9.4 ms. Decode attention over 3-bit KV costs us
  ~1.45x theirs at depth; the fixed-wave split-K with K/V tiles shared across the 8 verify rows and int8 QK scoring is
  the candidate.
- Drafter: ours 4.0-4.2 ms against their 2.7 draft + 1.0 commit/inject; the vocab subset (~0.6 ms) is part of it.
- Host gap: our wall time per step at 2K is ~32.7 ms against 30.7 ms of graphs (~2 ms between graphs).
Next candidates, in order of size at 125K: decode attention (~4 ms/step at 125K), verify matvec (~3.8 ms at every
context), drafter + host gap (~1-3 ms).

## Run 8 - 2026-10-19 16:15

Question: the quality of TQ3_4S-v2 (14.5 GB, the file behind all their 3090 speed numbers), which they don't publish.

Command: `tqkld --model <gguf> --llama-base bf16-wiki-c512-ch40.kld` (their llama.cpp-standard mode against our saved
BF16 logits: wikitext, 512 context, 40 chunks), with Unsloth IQ4_XS as a calibration (0.0175 by llama-perplexity).

```
file                 mean KLD   median   99%     99.9%   same top p   RMS dp
TQ3_4S-v2 (14.5 GB)  0.0878     0.0371   0.877   3.166   88.0%        8.62%
IQ4_XS (calibration) 0.0167     0.0067   0.182   0.591   94.2%        3.74%
ours, recipe A       0.0169 (llama-perplexity)
```

Finding: the calibration agrees with llama-perplexity (0.0167 against 0.0175), so the tool reads our reference
correctly. TQ3_4S-v2 is ~5x the KLD of our recipe A at about the same size, worse than their own 11.85 GB TQ_XS
(0.079 on their method), with 88% top-1 agreement against 94%. Its 4.0 bpw is 3-bit codes plus a quarter of the bits
in scales. Their speed numbers come from a much lossier file; decode is bandwidth bound, so the per-byte speed
comparison in Runs 5-7 still stands, but there is no quality reason to adopt the format.

## Run 9 - 2026-10-20 15:00

Question: after #509, where does our draft-and-verify step still lose to tqLLM's, by kernel?

Command: same session, both engines at their Run 2 / Run 5 settings (ours: recipe A IQ4_KS, q3 cache, DFlash 2 q4k at
7 drafts, 262K; tqLLM: TQ3_4S-v2, tq3 cache, DFlash2 TQ3, 262K). Step level: specbench ui with our phase timers /
`TQLLM_DEBUG=profile`. Kernel level: each server under `nsys launch -t cuda --cuda-graph-trace=node`, collection
started at the 8th streamed token and stopped 2 s later (`nsys_decode.py`, scratch), buckets per step from the sqlite
export (`nsys_step.py`, steps = main-model attention launches / 16). nsys inflates the gaps between graphs.

Step level (no profiler):

```
ms a step      tq verify  draft  commit  total     ours verify  drafter  wall
1988           20.63      2.46   0.94    24.0      24.10        3.96     29.6
31919          22.51      2.47   0.92    25.9      26.63        3.97     31.8
124730         28.33      2.52   1.01    31.9      33.93        3.95     39.3
```

Kernel level, ms a step:

```
                                  2K: ours   tq      125K: ours   tq
GPU idle (nsys)                       4.69   1.13          3.94   0.99
body weights matvec (+ tq drafter)   17.45  16.83         17.55  17.00
drafter weights (Q4_K MMQ)            1.78      -          1.82      -
LM heads (Q6_K MMQ x2 / u4h+i8h)      2.99   1.21          3.02   1.21
bf16 gemv (ssm beta/alpha)            0.96      -          0.97      -
activation quantize                   0.69      -          0.70      -
attention                             1.11   1.97         10.88   9.50
gdn / conv                            1.75   1.75          1.76   1.86
norm / glu / rope / rotate            0.63   1.34          0.64   1.44
sample / draft select                 0.21   0.29          0.20   0.30
step                                 32.72  24.88         42.02  32.66
```

Findings:
- We are ~23% slower a step at every length; attention is now a minor part of it (+1.4 ms at 125K, and ahead at 2K).
- The gaps, largest first: time between kernels (~1.5 ms without the profiler: wall minus graph times; tqLLM queues
  draft, verify and commit with one host sync); weights incl. the drafter (+2.4 ms: our file is ~4% larger, the
  Q4_K drafter is 1.8 ms); the LM heads (+1.8: a full-vocab Q6_K head for both verify and drafter against tqLLM's
  4-bit shortlist plus int8 re-score); unfused small kernels (+1.7: 48 bf16 gemvs and ~385 activation-quantize
  launches a step, which tqLLM folds into its producers).
- Not all of tqLLM's savings carry over: its verify head shortlist is not an exact argmax, and its sampled path
  takes only the top 16. The drafter's head is the safe place for a cheaper head (it only proposes).
Next, by size: the host gap between graphs; the drafter's head and weights (4-bit head over a draft vocabulary, the
held idea from Run 4); fusing beta/alpha into the GDN projection and the activation quantize into its producers.

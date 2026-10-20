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

## Run 10 - 2026-10-20 22:00

Question: what do the time between graphs and the small unfused kernels from Run 9 really cost, and what do the
cheap fixes take back?

Graph-level nsys (`--cuda-graph-trace=graph`, which records a graph as one item; node tracing inflates
`cuGraphLaunch`), 2K: the step is verify graph 24.06 ms, then 0.93 ms outside graphs (sampling, the drafter's input
projection and K/V writes, ~0.45 ms of host gaps between tiny metadata copies), drafter graph 3.95 ms, 0.06 ms back.
So the host gap is ~0.5 ms a step, not Run 9's 1.5; the bigger cheap items sit inside the verify graph.

- GDN gate projections (bf16, M 96, K 5120, 48 a step): `bench_bf16_gdn_gate_projection_cuda_graph_crossover` read
  14.3 us at batch 8 (cuBLAS 6.3). A row-tile kernel (4 rows a block, 2 past batch 4, warps split K, 16-byte loads
  reusing each input chunk across rows) reads 5.3 us at batch 8, 4.2 at batch 4; batch 1 keeps the old kernel
  (3.5 against 6.1). nsys: 15.0 -> 6.2 us a launch.
- Activation quantize: each tile-ordered matvec quantized its own input, so gate/up, qkv/z and q/k,v requantized
  one activation (384 launches a step). A first attempt keyed reuse on the tensor id and never fired (the
  projections see reshaped views, and `ensure` cleared the tag before the claim); the kept version is an explicit
  `with_shared_input` scope keyed by the input's address within it.

Interleaved A/B (master, branch, master, branch; three 2-3K prompts each; target verify graph, all steps):

```
             steps   verify median
master       640     24.23 ms
branch       688     23.62 ms
```

-0.61 ms a step (2.5%). Left from Run 9: the drafter's head and Q4_K weights (~1.5-2 ms against tqLLM), the LM
head, the remaining ~0.5 ms of host gaps.

## Run 11 - 2026-10-20 06:00

Question: how much of the drafter's ~4.0 ms graph is its LM head, and what does restricting proposals to a
vocabulary prefix cost in acceptance? Experiment behind `INFERENCE_RS_DFLASH_DRAFT_VOCAB=base_rows:tail_start`: the
drafter reads a row slice of the target's Q6_K head (ids `[0, 65536)` plus the special tail `[248044, vocab)`) and
fills the gap with -inf, so top-k, the DFlash2 selector and its full-vocab codebooks see unchanged ids. The draft
distribution is the one sampled from, so rejection sampling stays exact; only acceptance can drop.

Interleaved A/B (`ab_vocab.sh`: base, 64k, base, 64k; one 2K and one 32K prompt, UI sampling, 400 tokens):

```
                 dflash_graph   2K steps  ms/step   2K tok/s   32K steps  ms/step   32K tok/s
base (x2)        3.96 / 3.99    117       28.6-29.0 116.8/115.3 119      30.7-30.8  107.1/106.8
64k + tail (x2)  2.89 / 2.90    114       27.6-27.7 124.3/123.6 125      29.6-29.7  105.6/105.7
```

The head is 1.08 ms of the drafter graph, and the wall step drops ~1.1 ms at both lengths. Sampling is seeded, so each
variant repeats its step count exactly; one prompt per length is a single acceptance sample (2K got 3 steps fewer,
32K 6 more). Next: acceptance over a mixed prompt set (12 wiki, 4 code, 2 Chinese) before deciding.

Mixed prompt set (`accbench.py`: 12 wiki summaries at spread offsets, 4 code tasks, 2 Chinese essays; UI sampling,
400 tokens; one server per variant, steps paired to requests by log gaps):

```
            base: tok/step  ms/step  tok/s     64k + tail: tok/step  ms/step  tok/s
wiki        3.793           28.34    133.9     3.471                 27.29    127.2
code        3.200           28.18    113.6     3.059                 27.10    112.9
zh          2.149           28.14     76.4     1.346                 27.06     49.7
```

Negative: the step gets 1.05 ms (3.7%) cheaper, but acceptance drops 8.5% on wiki, 4.4% on code and 37% on Chinese,
so every set is slower. The 97-99% per-token coverage measured in Run 8 does not carry over: a draft chain ends at
its first out-of-set token, so a 1-3% miss rate over 7 positions removes far more accepted tokens than the miss rate
suggests. A bigger static prefix would save proportionally less and still miss CJK. Dropped; the code is reverted.
Only a per-request set (prompt ids on top of a base set) could recover wiki's rare tokens, and it still needs
CJK coverage. Next on the list: the drafter's Q4_K weights and the verify LM head.

## Run 12 - 2026-10-20 09:00

Question: the Q4_K drafter (1.8 ms a step) and the Q6_K LM head (two ~1.5 ms reads a step) run MMQ at 8 rows; how
far is that from their bandwidth floor, and does a tensor-core matvec over the row-major blocks close it?

`small_batch_mmvq_vs_mmq` on recipe A's `output.weight` (Q6_K) and `token_embd.weight` (Q4_K), 248320x5120:

```
            rows 1 dp4a   rows 2 dp4a   rows 8 dp4a   rows 8 mmq
Q6_K        1212 us       1206          1519          1515
Q4_K         811           943          1095          1099
```

8 rows cost ~25% over the 1-row read (MMQ's tiles, or dp4a's per-row integer work). IQ4_KS's tile-ordered kernel
(mmvq_tc.cuh) holds its 1-row time at 8 rows, but needs its own storage type, repack, untile and MMQ variant. Tried
instead: `mmvq_tc_k.cuh`, the same mma scheme over row-major blocks. Lane (g, t) reads words 4t and 16+4t of each
32-byte chunk, which are the A fragment's values for both nibble blocks; Q4_K is one k32 mma per block with the min
term from the activation row sums, and Q6_K two k16 mmas per block (one per 16-value scale) with 16-bit loads, since
its 210-byte superblocks leave every other word 2-byte aligned. Configs (KSPLIT warps over the superblocks, GROUPS
16-row groups, register prefetch of the next superblock):

```
248320x5120, rows 8   mmq     tc 4x2    8x1     16x1    8x1 no prefetch
Q6_K                  1510    1572      1367    1562    1429
Q4_K                  1110    1012       874     953     904
```

ncu on Q6_K 8x1: 1.33 ms, DRAM 92.6% (843 GB/s; 7% over the weight bytes), 128 registers, 33% occupancy. Near its
floor; the 16x1 shape is register-bound to one CTA an SM and loses 43% of issue to barrier waits on the uneven K split.

Drafter shapes (synthetic weights, `row_major_tc_shapes`), rows 8, us:

```
                 Q4_K mmq  tc best (cfg)      Q6_K mmq  tc best
6144x5120        41.6      30.1 (8x1 nopf)    52.7      41.3
5120x4096        30.8      19.8 (4x1 nopf)    37.9      28.1
34816x5120      175.5     133.2 (8x1)        235.0     199.0
5120x17408       99.6      88.4 (8x1)        135.5     140.0
5120x25600      145.1     127.5 (8x1)        202.3     211.9
```

The wide-K, short-N shapes gain least (Q6_K loses): 320 CTAs of 16 rows each reread the whole activation (24 bytes a
lane a block against 9 of Q4_K weight), and neither more warps nor fewer registers helped. Kept: Q4_K from 2 rows
(4x1 no prefetch at K <= 4096, else 8x1), Q6_K from 3 rows up to K 8192 (dp4a wins at 2). Test:
`row_major_tensor_core_matvecs_match_a_dense_matmul` (37 rows, 3 and 10 superblocks, batch 1-8, every config, all
input dtypes; cosine > 0.9999 against the dequantized matmul).

Expected a step: drafter 1.88 -> 1.49 ms, heads 2 x (1.51 -> 1.37) ms, ~-0.7 ms.

End to end (interleaved master, branch, master, branch; one 2K and one 32K prompt each; phase timers):

```
              dflash_graph   2K steps  ms/step   tok/s       32K steps  ms/step   tok/s
master (x2)   3.98 / 3.99    117       28.9-29.0 115.8/115.4  119      30.7-30.8  107.4/106.9
branch (x2)   3.48 / 3.49    112       28.1-28.4 124.0/123.1  122      30.1-30.2  106.6/106.4
```

The drafter graph drops 0.5 ms as expected, but the verify graph barely moves (23.7 -> 23.5 at 2K, flat at 32K), so
the head's share was smaller than the nsys split suggested. A step is 0.55-0.75 ms cheaper.

Acceptance: the 18-prompt set read lower on the branch (wiki 3.79 -> 3.66 tokens a step, code 3.20 -> 3.14, Chinese
2.15 -> 1.94), so a second set of 24 new wiki offsets ran on both (`WIKI_ONLY=1 WIKI_START=30000 WIKI_N=24`):
master 3.796, branch 3.744 tokens a step; per-prompt paired difference -0.05 +- 0.07 (SE), i.e. noise, at
28.40 -> 27.88 ms a step (133.7 -> 134.3 tok/s). Against the dequantized f32 matmul the new kernel is no less
accurate than MMQ (rms rel err at 8 rows: Q4_K 5.6e-3 vs MMQ 0.6-1.7e-2; Q6_K equal at 5.6e-3), so no numerics
regression explains a drop. Single-sample prompts swing +-0.6 tokens a step; acceptance A/Bs need 24+ prompts.

## Run 13 - 2026-10-20 07:40

Question: before building sampled tree verify (tqLLM's greedy lead from 8K comes from its tree), how many tokens a
step would a sampled tree add within the same 8 verify rows? The fast matvec paths stop at 8 rows
(`MMVQ_MAX_BATCH`), as do tqLLM's (`kGdnReplayCols`), so a tree trades depth for width.

Exactness: the DFlash2 selector's q at depth d is a softmax over the drafter's top 16 for that position, scored
against the predecessor token, so a node's children are draws from a known q_d(.|parent). Siblings drawn from it
without replacement and checked by recursive rejection (accept with min(1, p/q); on reject p <- norm(max(p - q, 0)),
q <- q without the rejected token) keep the output distribution exact.

Method: an env-gated dump in `verify_staged_batch` (not kept; patch in the scratchpad) logged each step's 7 drafts,
the 16-candidate q per position and the top 64 target logits per verify row; recipe A, q3 cache, DFlash 2 q4k at 7
drafts, UI sampling (temperature 0.6, top_p 0.95, top_k 20), the 18-prompt set plus 24 more wiki offsets (4672
steps). `tree_sim.py` (scratchpad) rebuilds p exactly as `prepare_topk_distribution` does. A leaf sibling at depth d
hangs off chain node d-1, whose verify row is already in the log, and its q is the logged q_d, so for the "chain
plus leaf siblings" family the expectation is exact.

```
4672 steps; observed 3.567 tokens a step, chain-7 expectation 3.576
P(depth d accepted)   0.777 0.573 0.421 0.307 0.223 0.160 0.116
depth                 1     2     3     4     5     6     7
P(sibling accepted | first draft rejected)
                      0.477 0.375 0.308 0.256 0.227 0.188 0.189
target mass inside q's 16
                      0.973 0.926 0.876 0.817 0.769 0.720 0.689

8 rows (best shapes): chain 7 3.576 | chain 6 + sibling at 1 3.556 | chain 6 + sibling at 2 3.541 | chain 5 + 2 sib 3.478
more rows, chain 7 + best leaf siblings: 9 rows 3.672 (+2.7%), 10 rows 3.754 (+5.0%), 12 rows 3.853 (+7.7%),
16 rows 3.970 (+11.0%)
```

Negative: at 8 rows the chain is the best tree. A node is worth the probability that its path is accepted. The
seventh chain draft is worth 0.116 tokens. A sibling at depth 1 is worth P(reject at 1) x 0.477 = 0.106, and every
deeper sibling is worth less. Subtrees under a sibling would score lower still, since each of their nodes is that
0.106 times a continuation probability below 1. Best-first allocation of 7 nodes therefore picks the chain. A tree
only pays when rows are added. Leaf siblings give +11% tokens a step at 16 rows (subtrees under them maybe a few
points more), but every 8-row fast path (tensor-core matvecs, fattn-quant's `FQ_MAX_ROWS`, the GDN fused verify)
would have to double. A step at 16 rows has to cost under ~11% more than at 8 to come out ahead. Not pursued; the
sample is at temperature 0.6, and a hotter workload flattens q and could move the balance toward width.
tqLLM's greedy tree most likely gains on near-ties, where a greedy chain rejects and the second-ranked sibling is often right (not measured here).
That does not carry over to sampled verification.

## Run 14 - 2026-10-20 08:30

Question: what is in the time between our verify and drafter graphs, and how much decode does removing it buy?

Graph-level nsys (`nsys launch -t cuda,osrt --cuda-graph-trace=graph`, 2 s of a 2K decode; `nsys_gap.py`,
`nsys_gap2.py` in the scratchpad), then a temporary probe without the profiler. The probe was CUDA events after the
verify graph, before and after the drafter graph and at the next verify launch, plus host `Instant`s at the
rejection wait and named points in the driver (not kept; `gap_probe.patch` in the scratchpad). 64-step medians:

```
GPU   verify end -> drafter start 1.175 ms (0.34 ms of it kernels: rejection 0.13, drafter context projection 0.2)
      drafter end -> next verify start 0.026 ms
host  rejection result -> drafter launch 0.78 ms:
        target recurrent commit (transition publish)   0.267
        drafter context K/V append                     0.263
        drafter graph inputs + launch                  0.114
        sequence bookkeeping                           0.073
      verify end -> host awake ~0.24 ms past the rejection result (C-states: C2 120 us, C3 1034 us exit latency)
```

The drafter does not need the target's recurrent commit, so a variant ran it after the drafter launch (only for
batches with no finished sequence, since finishing caches the prefix with the committed state). The GPU gap dropped
1.175 -> 0.935 ms, and output was identical, but decode did not move. A valid interleaved round: master 147.99,
deferred 148.46 tok/s. (A second round was invalid: `pgrep -x inference` missed a binary named differently, so it
measured a leftover server. The bench helpers now stop servers by PID.) Polling the rejection event every 100 us
instead of sleeping keeps the core out of deep C-states. It cut the gap to 0.43 ms and the host work 3-4x, but
decode fell 4-8% (135.7 vs 146.2 tok/s), because the verify graph itself got slower on the GPU (23.7 -> 24.6 ms).
A plain spin wait on the same event was 4% slower too (143.3 vs 149.3 tok/s, two interleaved rounds).

Why: decode on this card is power-limited. nvidia-smi during decode reads 355-359 W against a 370 W limit, SM clocks
1590-1740 MHz against a 2115 MHz max at 63 C, with `clocks_event_reasons.sw_power_cap` Active throughout and no
thermal or hardware slowdown. The ~1 ms gap a step is time the card spends under the cap, not lost throughput:
shrinking it raises average power and the clocks drop to match. Negative: dropped, code reverted.

Implications: on this card, decode speedups have to cut energy a token (fewer bytes or instructions per step, as the
tensor-core matvec did), not idle time. Host-gap and launch-overhead work pays only on a card with power headroom.
The TDP is a user setting (`nvidia-smi -pl`, max 390 W here).

## Run 15 - 2026-10-20 09:40

Question: tqLLM's prefill is ~1.5x ours (Run 5). Its large chunks dequantize each weight, requantize it to int8 with
one scale per row, and run a cuBLASLt int8 GEMM. Would that pay here, and what does it cost in KLD?

Where our prefill goes (nsys, one cold 7987-token prefill, 512-token chunks, 1341 tok/s): kernels 5844 ms of a
5925 ms span. MMQ (IQ4_KS) 3952 ms (68%), GDN recurrence (`gated_delta_rule_recurrence_kernel_vmajor_grouped`) 789,
attention 477, GDN prepare 121, activation quantize 102, the rest about 400. Chunk size barely matters:
`--max-prefill-chunk-tokens` 512 / 1024 / 2048 gives 1348 / 1378 / 1398 tok/s.

GEMM throughput at prefill shapes (sustained ~1.5 s per point, random data, so the card sits at its power cap as in
real prefill; constant data reads ~15% higher):

```
MMQ, recipe A weights (new ignored bench prefill_mmq_shapes, not kept)
  IQ4_KS 17408x5120 / 5120x17408 / 10240x5120 / 5120x6144, rows 512-2048: 93-100 TOPS
MMQ by type, synthetic 17408x5120 and 5120x17408, rows 512 and 2048
  Q8_0 97-100, Q4_0 98-102, Q4_K 91-94, Q6_K 77-79 TOPS
upstream llama.cpp (8bb3cf0d2) test-backend-ops, IQ4_XS 4096x14336 n=512: 88 TOPS
cuBLASLt i8 x i8 -> i32 (scratchpad i8gemm/bench.cu), rows 512 / 1024 / 2048
  34816x5120 207 / 203 / 208, 5120x17408 150 / 150 / 217, 14336x5120 192 / 173 / 186, 5120x6144 137 / 152 / 197 TOPS
```

MMQ sits near 100 TOPS on every weight type, Q8_0 included, and matches upstream, so IQ4_KS decoding is not the
limit. ncu on MMQ Q8_0 at 512 rows: tensor pipe 36.5%, issue slots 39%, 254 registers and one block per SM (16.7%
occupancy), stalls spread over math-pipe throttle, wait, global and MIO. A plain int8 GEMM runs 1.5-2.1x faster.

KLD of the int8 numerics, emulated exactly inside MMQ's entry point (dequantize, round to int8 with the given scales,
multiply in f32; prefill batches of 256+ rows; everything but the 16 Q5_K K/V projections and the Q6_K head, which
have no GPU dequantizer). Measured against our own MMQ output: wikitext, 40 windows of 512 tokens, the second half of
each scored as llama-perplexity does, KLD from the baseline's top 64 plus a tail term (temporary `kld_cmp` example;
patch and source in the scratchpad).

```
weights int8 per row, activations as             KLD vs MMQ  median   p99     same top-1  PPL
  MMQ itself                                     0           0        0       100%        5.8265
  bf16 (weights only)                            0.0017      0.0009   0.013   98.0%       5.8274
  int8 per 32 (MMQ's activation blocks)          0.0018      0.0010   0.013   98.0%       5.8288
  int8 per 128                                   0.0026      0.0013   0.022   97.7%       5.8278
  int8 per 512                                   0.0048      0.0022   0.046   96.6%       5.8456
  int8 per token (what a plain int8 GEMM takes)  0.0278      0.0109   0.289   92.7%       5.9072
```

Negative: a plain int8 GEMM needs one activation scale per token, and that alone adds KLD 0.028, 1.6x recipe A's
whole quantization cost (0.0169). Outlier channels in the activations are the likely cause. Finer scales recover the
quality, but then a scale applies inside the K loop, which cuBLASLt does not offer. A custom kernel with 128-wide
groups would cost +0.0026 KLD (about 15% on top of the file's own), and its speed against MMQ is unknown until
written. Requantizing the weights to one scale per row costs 0.0017 of that on its own. Not taken.

The lossless lead in prefill is the GDN recurrence, at 13.5%. It is a token-sequential scan, 0.97 ms per layer per
512-token chunk, about 1.6 TFLOPS. A chunked (WY) form on tensor cores would cut it while keeping fp32 state math.

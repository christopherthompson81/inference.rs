# Nemotron 3.5 Lightning on the CPU

nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (`nemotron_h`: 23 Mamba-2, 6 NoPE attention and 23 MoE layers,
128 experts of which 6 route per token plus a shared one, one MTP layer) on a Ryzen 7 7840U (Zen 4, 8 cores / 16
threads, AVX-512 with VNNI and BF16) with 60 GB of LPDDR5. The BF16 checkpoint is 65.8 GB, more than the RAM, so the
model runs quantized. Goals: run it here at the best speed the machine allows, with its MTP head drafting.

## Run 1 - 2026-11-02 ~19:50

**Question:** what can the hardware do, and what does llama.cpp do on it?

**Commands:** a 4 GiB AVX-512 read loop over 1-16 OpenMP threads; llama.cpp 2cbca389f built `-DGGML_NATIVE=ON`,
`convert_hf_to_gguf.py --outtype bf16` (the MTP layer kept, folded in as `blk.52` with `nextn_predict_layers = 1`),
`llama-quantize`, `llama-bench -p 512 -n 128 -t 8,16 -r 3`.

**Findings:**
- Read bandwidth: 41.7 GB/s on one thread, 55.0 GB/s at four, 53-55 GB/s from four to sixteen.
- No k-quant fits this model: its rows are 2688 (hidden) and 1856 (expert) wide, neither a multiple of 256, so
  `llama-quantize Q4_K_M` falls back to Q5_0 / Q8_0 for nearly every matrix (25.4 GB). Only the 32-block types
  apply: Q4_0, IQ4_NL, MXFP4 (4.25-4.5 bpw), Q5_0 and Q8_0.
- Experts IQ4_NL, everything else Q8_0 (as the Transcribe recipe): 19.68 GB. A decode token reads about 3.2 B weights
  (six experts and the shared one over 23 layers 1.84 B, Mamba 0.89 B, attention 0.14 B, `lm_head` 0.35 B), about
  2.7 GB at this mix: a ceiling of ~20 tok/s at 55 GB/s.
- llama.cpp on that file: pp512 79.47 +- 1.51 / tg128 17.57 +- 0.18 tok/s at 8 threads; 61.50 / 8.82 at 16 (the
  SMT siblings contend for bandwidth).

## Run 2 - 2026-11-02 ~20:40

**Question:** where does our engine stand on the same file?

**Command:** `inference bench --cpu --format gguf -f <IQ4_NL-exps-Q8_0> --prompt-len 512 --gen-len 128
--iterations 3`, release build of master 017a7635f.

**Result:** prefill 106.3 +- 1.1 tok/s (34 % ahead of llama.cpp), decode 14.9 +- 0.1 tok/s (66.9 ms a token, 15 %
behind llama.cpp, 25 % below the bandwidth ceiling).

## Run 3 - 2026-11-02 ~20:40

**Question:** the BF16 reference logits for KLD, on a machine whose RAM the BF16 file exceeds.

**Command:** `llama-perplexity -m <BF16 GGUF> -f wiki.test.raw -c 512 --chunks 40 -t 8 --kl-divergence-base`.

**Finding:** at `-b 512` every chunk is one pass over the 65.8 GB mmap, which thrashes the page cache (I/O pressure
~38 %, 100 s a chunk and climbing). `-b 4096 -ub 4096` scores eight chunks a pass: 64 s a pass, 10.3 minutes in all.
PPL(BF16) = 6.5402 +- 0.17076; the logits file is 2.67 GB.

## Run 4 - 2026-11-02 ~21:00

**Question:** built-in MTP needs PagedAttention (`attach_speculative` refuses without a cache engine), which was
CUDA and Metal only. Does a CPU PagedAttention serve the hybrid model?

**Change:** `inference-paged-attn` CPU kernels with Metal's signatures over the cache engine's vLLM layout
(`reshape_and_cache`, one-query `paged_attention` with ALiBi, softcap and sinks, `gather_kv_cache`, `copy_blocks`),
the CPU-only builds' backend; `paged_attention.rs` compiles in every build. CPU paged attention stays opt-in
(`--paged-attn on`); `paged_attn_supported_on(device)` says which devices a build's kernels serve.

**Tests:** kernel cases against plain softmax attention (GQA, sinks, softcap, ALiBi; F32 and BF16 caches), a gather
round trip, a block copy; `nemotron_h_tiny::a_long_prompt_scores_as_its_cpu_paged_decode` (decode steps through the
block tables score as one prefill), with a probe confirming the CPU decode kernel runs.

## Run 5 - 2026-11-02 ~21:30

**Question:** can Nemotron-H's MTP head draft on the CPU without changing greedy output?

**Reference:** llama.cpp's `nemotron-h-moe.cpp` (transformers ignores `mtp.*`): `e = RMSNorm(embed(x[p+1]))`,
`h = RMSNorm(target hidden after norm_f at p)`, `eh_proj(concat(e, h))`, a NoPE attention sublayer
(`mtp.layers.0`), the trunk's MoE (`mtp.layers.1`), a weight-only mean-subtracting LayerNorm (`final_layernorm`),
the shared `lm_head`; the next draft chains the post-norm hidden.

**Change:** `MtpHead` takes any `MtpBlock` (the decoder's own layer by default); Granite's hybrid model loads the
head's block as its attention blocks, as the paged layer after the stack's, and stashes each Mamba layer's pre-step
state and conv input / dt on a verify step. A commit that keeps fewer rows than ran re-advances those layers'
states over the kept rows from the stash, skipping the input projection. GGUF `blk.52.nextn.*` and the folded block
bind to the checkpoint's `mtp.layers.{0,1}` names.

**Test:** `nemotron_h_tiny::an_mtp_head_drafts_on_the_cpu_without_changing_greedy_output` (random weights: 0 of 42
drafts accepted, every verify step replays). With the replay disabled the greedy output diverges at once.

## Run 6 - 2026-11-02 ~22:10

**Question:** is MTP on the real model correct, and what does it buy?

**Commands:** `inference serve --cpu --max-seqs 1 --format gguf -f <IQ4_NL-exps-Q8_0>` plain, with `--paged-attn on`,
and with `--paged-attn on --mtp --mtp-n-predict {1,2,3}`; four chat prompts (code, a list, prose, counting), greedy,
thinking off, 160 tokens for correctness and 256 for speed (`usage.avg_compl_tok_per_sec`), machine idle.

**Correctness:** two of four prompts byte-identical with MTP; the other two part where the plain run's top two
tokens are 0.107 and 0.113 nats apart (its `top_logprobs`), as paged attention alone does at 0.042 and 0.313:
near-ties a multi-row forward rounds the other way, not drafts leaking into the output.

**Speed (decode tok/s per prompt, mean):**

| config | prompts | mean | MTP accept (len) |
|---|---|---|---|
| plain | 15.55 / 15.17 / 15.37 / 15.47 | 15.4 | |
| paged | 15.44 / 15.88 / 15.14 / 15.27 | 15.4 | |
| MTP 1 | 16.71 / 16.66 / 15.50 / 18.43 | 16.8 | 70-93 % (1.70-1.93) |
| MTP 2 | 19.95 / 17.59 / 14.85 / 20.93 | 18.3 | 58-83 % (2.15-2.67) |
| MTP 3 | 20.10 / 16.95 / 13.24 / 23.81 | 18.5 | 38-61 % (2.15-2.84) |

**Finding:** CPU paged attention costs nothing; MTP 2 gives +19 %. A verify step costs ~1.85x a decode step
(2.2 tokens a step at 18.3 tok/s is 120 ms, against 65 ms), where its weight reads predict ~1.5x: three tokens
route to ~17 distinct experts a layer instead of 6, but the non-expert 1.9 GB (Q8_0) is read once either way.

**Next:** the quant sweep (what the non-expert tensors, 70 % of a token's bytes, tolerate), a tolerance that accepts
a draft within a few hundredths of a nat of the target's top token, and a profile of the verify step's multi-row
path (dense m=3 matmuls, the experts' row kernel, the Mamba gather / scatter and replay).

## Run 7 - 2026-11-02 ~23:10

**Question:** does accepting a draft that ties the target's top token (within a few tenths of a nat) buy speed?

**Change:** `--mtp-greedy-tolerance <NATS>` (`MtpSpec.greedy_tolerance`): host greedy verification also accepts a
draft whose logit is within that many nats of the argmax, for argmax sampling over raw logits (no penalties, bias
or processors), unconstrained, without returned logprobs. Unit case `a_draft_within_the_tolerance_of_the_top_logit_ties`
(which first caught the row's `[1, vocab]` shape).

**Result (Run 6's prompts, decode tok/s mean):** MTP 2 exact 18.33, 0.10 nats 18.06, 0.25 nats 18.04; MTP 3 exact
18.53, 0.25 nats 18.67. One prompt of four (0.10) or three (0.25) took a tie and changed text, so the tolerance runs,
but the speed is within run-to-run noise: ties are too rare to matter, and a verify step's 1.85x cost is the limit.

## Run 8 - 2026-11-03 ~09:40

**Question:** which of the 32-block types does each tensor group tolerate?

**Commands:** `llama-quantize --imatrix` (wiki.train, 160 x 512 tokens, from a Q8_0 build) from the BF16 GGUF, base
Q8_0 with one group lowered per variant; `llama-perplexity --kl-divergence` against Run 3's BF16 logits.

**Result (KLD vs BF16; the cost is over all-Q8_0's 0.0040; decode MB is a token's weight reads saved):**

| group | Q5_0 KLD (+cost) | IQ4_NL KLD (+cost) | other | decode MB saved (Q5_0 / IQ4_NL) |
|---|---|---|---|---|
| routed `ffn_up_exps` | 0.0063 (+0.0023) | 0.0110 (+0.0070) | Q4_0 0.0117, MXFP4 0.0201 | 258 / 344 |
| routed `ffn_down_exps` | 0.0061 (+0.0021) | 0.0088 (+0.0048) | Q4_0 0.0101, MXFP4 0.0168 | 258 / 344 |
| shared expert | 0.0055 (+0.0015) | 0.0075 (+0.0035) | | 172 / 230 |
| `ssm_in` | 0.0080 (+0.0040) | 0.0130 (+0.0090) | | 239 / 318 |
| `ssm_out` | 0.0105 (+0.0065) | 0.0111 (+0.0071) | | 95 / 127 |
| attention q/k/v/o | 0.0066 (+0.0026) | 0.0096 (+0.0056) | | 52 / 70 |
| `output` | 0.0063 (+0.0023) | 0.0121 (+0.0081) | | 131 / 175 |
| `token_embd` | 0.0052 (+0.0012) | 0.0063 (+0.0023) | | 0 / 0 |

**Findings:**
- IQ4_NL is the best 4-bit type everywhere; Q4_0 is a little worse at the same size and MXFP4 is twice the cost.
- The experts' down projection tolerates 4 bits better than the up projection.
- `ssm_out` is the most sensitive matrix per byte: Q5_0 already costs +0.0065 for 95 MB. Keep it at Q8_0.
- For decode speed, a lowered matrix pays in proportion to how often it is read. The shared expert at IQ4_NL
  (66 MB per 0.001 KLD), `output` at Q5_0 (57) and `ssm_in` at Q5_0 (60) pay best. Attention (20 at Q5_0) and
  `ssm_out` (15) do not.
- `token_embd` has no decode cost (one row per token), so IQ4_NL there only saves RAM.

## Run 9 - 2026-11-03 ~10:30

**Question:** what do recipes built from Run 8 score, and what do they decode at?

**Recipes (imatrix, from BF16):** r0 = experts IQ4_NL, the rest Q8_0 (the Run 1 mix); r1 = r0 plus the shared expert
and `token_embd` IQ4_NL and `ssm_in` and `output` Q5_0; r2 = r1 plus `ssm_in` IQ4_NL and `ssm_out` and attention Q5_0.

**First speed pass:** r1 decoded at 5.7 tok/s and r2 at 2.8. A flat profile put 45 % in `dequantize_iq4_nl` and
`dot_f32`: a dense IQ4_NL matrix (a type Candle has no `QTensor` for) dequantized all of itself on every CPU call.
**Change:** `gguf::cpu_dense` dots a 32-value block type's dense rows in place with the expert path's 8x4 / 8x1
kernels, for up to 32 activation rows; longer batches keep the dequantized matmul.

**Result (Run 6's prompts and serve setup, decode tok/s mean):**

| recipe | size | KLD | top-1 | PPL | plain | MTP 2 |
|---|---|---|---|---|---|---|
| Run 1 file (no imatrix) | 18.3 GiB | 0.0155 | 95.0 % | 6.582 | 15.4 | 18.3 |
| r0 | 18.3 GiB | 0.0150 | 95.2 % | 6.584 | 16.3 | 18.6 |
| r1 | 17.6 GiB | 0.0238 | 93.5 % | 6.607 | 18.3 | 19.6 |
| r2 | 17.4 GiB | 0.0351 | 92.0 % | 6.672 | 18.7 | 18.6 |

**Findings:** Run 8's single-group costs add up to within 15 % of a recipe's KLD (r1 predicted 0.028). r1 buys 12 %
plain decode for 1.6x the KLD; r2 adds only 2 % more for another 1.5x, and its MTP is no faster than plain. MTP's
gain shrinks as the non-expert bytes shrink (r0 +14 %, r1 +7 %), since the verify step's extra cost is in the
routed experts and fixed per-step overhead, not in the matrices every token reads once.

## Run 10 - 2026-11-03 ~13:00

**Goal:** the fastest decode with KLD under 0.01. MTP verification is exact, so the MTP head and the draft path cost
no KLD.

**Recipes:** A = routed and shared experts Q5_0, the rest Q8_0: 21.7 GiB, KLD 0.0090, top-1 96.2 %, PPL 6.544.
B = routed up Q5_0, down IQ4_NL: KLD 0.0103, and slower than A (15.5 against 16.2 tok/s plain) despite fewer
bytes. llama.cpp decodes A at 18.35 tok/s (tg128, 8 threads).

**Where an MTP 2 step goes (A, 131 ms):** propose 22 ms (two drafts, each reading the 375 MB Q8_0 `lm_head`),
commit / Mamba replay 4 ms, the verify forward the rest.
**Change:** `--mtp-draft-isq` (`MtpConfig.drafter_isq`) also gives a built-in MTP head a draft-only `lm_head` copy
at that type; verification keeps the target's. Q5_0: propose 16.6 ms, MTP 2 18.3 -> 20.0 tok/s, acceptance
unchanged. Q4_0 is no faster (no repacked kernel) and accepts less; depth 3 does not beat depth 2.

**Plain decode overhead:** a pool-call timer showed 84 % of a token inside barrier-pool calls (the kernels, near
bandwidth) and 16 % (10 ms) between them on the main thread. The Mamba scan and conv step dispatched on rayon,
whose threads wake beside the pool's spinning workers for every layer.
**Change:** both run on the barrier pool, and the scan writes its output in place. Between-call time fell to 9 %.

**A/B (same thermal state, recipe A):** plain 15.95 -> 16.84 tok/s; MTP 2 with the Q5_0 draft head 19.02 -> 19.39.
The machine ran ~4 % slower than at Run 9 by then; MTP's per-prompt spread (15.8 to 22.1) is the prompts' acceptance.

## Run 11 - 2026-11-03 ~15:00

**Question:** where do plain decode's remaining ~5 ms against llama.cpp go?

**Method:** a temporary timer on every barrier-pool call, keyed by the call's closure type, over 130 decode tokens on
recipe A. Per token: dense tile matmuls 33.0 ms (1.79 GB, ~54 GB/s), routed experts 21.4 ms (0.95 GB, ~44 GB/s),
router / scan / conv / attention 2.6 ms, 4.8 ms between calls (runs of small tensor ops of 10-37 us each).

**Experts:** the 8x1 kernel dots eight rows in lockstep, eight streams 1.8 KB apart per thread. A 256 MB DRAM
microbench at eight threads: Q5_0 8x1 44 GB/s, one row at a time 56.5 GB/s (Q8_0 51 against 58); one thread runs
both at the same rate, so it is DRAM stream interleaving, not compute. With four tokens the 8x4 kernel's shared
decoding still wins (Q5_0 40 GB/s against 31 for 1x4). An AVX-512 VNNI rewrite of the 8-row kernels (no transposes,
`vpdpbusd`) ran no faster in cache or from DRAM, so it was dropped.
**Change:** a lone route (a decode step's, and most experts' in a verify step) dots its rows one at a time, as does
`cpu_dense` for its sub-quad tokens; the 8x1 kernels are gone. Expert calls: 459 -> 420 us, 21.4 -> 19.6 ms a token.

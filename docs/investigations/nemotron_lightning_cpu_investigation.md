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

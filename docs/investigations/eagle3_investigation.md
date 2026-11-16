# EAGLE-3 drafters on the shared decoder

Issue #584. EAGLE-3 (arXiv 2503.01840) drafters are one Llama layer trained per target. `fc` fuses the target's
hidden states at three layers, the layer reads `concat(norm(embed(next token)), norm(fused))` with the fused state as
its residual, and a reduced-vocab `lm_head` maps back through `d2t`. Chained drafts feed the layer's pre-norm output
back in place of the fused state. Reference: llama.cpp 970b51ce7 (`src/models/eagle3.cpp`, `common/speculative.cpp`),
the speculators library's `eagle3.py` shipped with RedHatAI's drafters.

The shared decoder (`crates/inference-nn/src/decoder/`) already had everything a chained drafter needs: multi-row
verify, paged K/V trimmed on rejection, prompt-chunk prefill with the last row held until its token is sampled, CUDA
graphs and device draft sampling, all in `MtpHead` (ERNIE 4.5's built-in head). EAGLE-3 runs as an `MtpHead` over its
own block:
- the target's forward records `fc(concat(taps))` where an MTP head records its final hidden state, so the head's
  refresh rows and chain rows both read the drafter's width;
- the head hands its block `concat(embedding, hidden)` unprojected (`MtpInput::Concat`);
- draft logits gather the reduced vocabulary into the target's (`DraftVocab`);
- core reserves the drafter's paged layer as an auxiliary layer (CUDA), as DFlash's are.

Setup for the runs: Qwen3-8B Q4_K_M (Qwen/Qwen3-8B-GGUF) on the RTX 3090, `inference serve` (dev profile, CUDA),
`--pa-context-len` default. Drafters: RedHatAI/Qwen3-8B-speculator.eagle3 (speculators format, Apache 2.0, own
embedding copy and `t2d` unread) and AngelSlim/Qwen3-8B_eagle3 (SpecForge `midlayer`, pickle). Bench (`e3bench.sh`
in the session scratchpad): a warm-up, then three 400-token chat requests (code, explanation, story), thinking off,
tok/s from each response's usage and acceptance from the server's `inference_speculative_*` counters over the three.

## Run 1 - 2026-11-16 04:00

**Question:** does the first build draft at all, and how fast?

**Finding:** it loads, attaches (3 drafts from the speculators config) and drafts. `inference bench` (synthetic
tokens) 65 tok/s against 145 plain. Real text, t=0.6: 65-74 tok/s against 150-154 plain, 2.07 tokens a step: ~29 ms a
step against plain's 6.6.

nsys over a 100-token request: `ia_u32_bf16` (`index_add`, scattering the draft vocabulary's logits into the
target's) 823 ms of 1466 ms busy, 5.8 ms a call. Replaced with a gather: the draft logits padded with one `-inf`
column, then `index_select` of a precomputed target-to-draft index. After: 151-172 tok/s (plain 150-154).

## Run 2 - 2026-11-16 04:20

**Question:** do the configs' layer ids name the layer whose output is the state, or the one it feeds? (Our taps are
layer outputs; vLLM and llama.cpp record the state entering the listed layer, so tap = id - 1.)

```
TMP_EAGLE3_SHIFT (tap = id + shift), three requests each, tokens a step
                 t=0.6     t=0 (greedy)
shift -1         2.10      2.27
shift 0          2.27      2.30
shift +1         2.20
```

Within noise: neighbouring layers' residual streams are close, so one layer either way barely moves acceptance. Kept
vLLM's convention (-1), which the drafters were trained against. The card's own acceptance lengths at 3 drafts are
coding 2.39, math 2.48, summarization 2.13 (BF16 target, vLLM); ours on a mixed set with a Q4_K_M target, 2.1-2.3.

## Run 3 - 2026-11-16 04:40

**Question:** where does the step go now? nsys, 47 verify steps, 13.9 ms GPU a step: the target's Q4_K linears at 4
rows 6.4 ms, its Q6_K head 1.0 ms, `sparse_rejection_categorical_f32_kernel` 1.2 ms, bf16 drafter GEMMs and GEMVs
~1.7 ms, top-k and sort over the 152K target vocabulary ~1.4 ms. The bf16 drafter (~400M parameters with its head)
reads ~0.8 GB per chained step.

`--mtp-draft-isq` now quantizes the drafter's linears (q/k/v/o, MLP, `fc`, head) after load:

```
t=0.6           tok/s (three requests)     tokens a step
bf16            181 / 156 / 156            2.15
q8_0            206 / 174 / 176            2.19
q4k             228 / 193 / 190            2.29
plain           154 / 150 / 150
```

## Run 4 - 2026-11-16 05:00

**Question:** what is the 1.3 ms rejection kernel? nsys of the q4k drafter: `sparse_rejection_categorical_f32_kernel`
1.29 ms a step, one block per sequence over the whole vocabulary for every draft row. The verifier takes it when the
request samples by temperature alone. With a top-k or top-p (Qwen3's recommended settings, or a generation config
that sets them, as the 27B's DFlash runs had) it takes `sparse_rejection_topk_f32_kernel`, ~0.05 ms a step. The GGUF
here ships no generation config, so plain t=0.6 requests went the slow way. A general cost of temperature-only sampled
verification for every drafter, not EAGLE-3's: left for its own change.

Qwen3's recommended non-thinking sampling (t=0.7, top-p 0.8, top-k 20):

```
                tok/s (three requests)     tokens a step
plain           156 / 154 / 154
EAGLE-3 q4k     257 / 223 / 209            2.28
```

## Run 5 - 2026-11-16 05:10

**Question:** best draft depth, same sampling, q4k drafter:

```
n_predict       tok/s                      tokens a step
2               241 / 219 / 220            2.11
3               257 / 223 / 209            2.28
4               245 / 204 / 214            2.33
5               256 / 198 / 186            2.31
```

3 (the speculators config's proposal length, the default) is as good as any. Acceptance stops rising past 3, as on
the card's summarization row.

Left: target-side tap capture under multi-GPU device maps is untested (taps move to the head's device); Qwen3.5/3.6
targets (their own tap path, DFlash's) and the specdrift variant's per-tap norms (`fcs.*`, `fc_norm`), gpt-oss
(`norm_before_fc`), and an `eagle3` GGUF binding.

## Run 6 - 2026-11-16 05:40

**Question:** does the SpecForge layout (`midlayer.*`, pickled `pytorch_model.bin`, no layer ids in its config) load
and draft? AngelSlim/Qwen3-8B_eagle3, q4k, Qwen3's recommended sampling. The memory reservation sized external
drafters from safetensors only; a pickled one now reserves its size on disk.

```
                         tok/s (three requests)     tokens a step
RedHat (speculators)     257 / 223 / 209            2.28
AngelSlim (SpecForge)    225 / 202 / 159            1.91
```

It loads at vLLM's default layers (2, 18, 33 of 36) and drafts. Its lower acceptance on these requests is the
drafter's (other training data), not the format: the same code path serves both past the tensor prefix.

## Run 7 - 2026-11-16 07:30

**Question:** can temperature-only sampled verification (`sparse_rejection_categorical_f32_kernel`, 1.29 ms a step in
Run 4) cost what the top-k path does? A general cost for every drafter: a request with no top-k / top-p (or a GGUF
without a generation config) verifies this way.

The kernel ran one 256-thread block per sequence. For each draft row it read the 152K-wide logits twice (max, then
exp sum), then sampled the residual over thread-contiguous segments (uncoalesced) with one thread walking ~600 tokens;
thread 0 also read the 128-wide q row slot by slot, in doubles, per draft, and insertion-sorted it on rejection.

```
ncu, wide-vocabulary test case (4 sequences, 3 drafts, vocab 151936), DRAM clocks, caches flushed
q width 3                                          q width 128
256 threads, two passes (master)       134 us
1024 threads, one online pass          (in-model 593 us avg: q width 128 dominates)
+ chunk stats kernel, 128 chunks       18 + 22 us                   18 + 851 us
+ q staged, validated, sorted in block                              18 + 38 us
```

At 1024 threads in one block the kernel was instruction-bound on its SM (~20 instructions a vocabulary entry, 11
cycles a warp issue), not memory-bound: the row sums had to leave the block. Now a first kernel computes each
row's (max, exp sum) per chunk of 1187 tokens over 128 x rows blocks; the verify kernel combines a row's 128 partials,
takes the residual's chunk masses as the chunk's exp sum less `min(p, q)` at its q tokens (a residual under 1e-5 of
the chunk's mass is floored to zero: q covering it all), picks a chunk and walks only that one with a block scan. The
q row is staged in shared memory and checked, normalized and sorted by the block.

The wide case at q width 3 draws the same tokens as the old kernel; the bench below accepts exactly as master. The
tiny-vocabulary case is exact against the CPU reference; the wide case is checked against f64 cumulative mass, since
an f32 serial sum over 152K tokens lands 1-2 tokens off for both kernels.

In-model (nsys, 100-token request): 7 us + 28 us a step against 1.29 ms. Same seeded requests, t=0.6 only, q4k
drafter, no throttling:

```
                tok/s (three requests)     tokens a step
master          227 / 194 / 190            2.29
this change     255 / 216 / 212            2.29 (674 of 1563 accepted, same as master)
```

Temperature-only now runs as fast as Qwen3's top-k / top-p settings did in Run 4 (257 / 223 / 209).

## Run 8 - 2026-11-16 08:50

**Question:** do Qwen3.5 / Qwen3.6 targets draft with EAGLE-3, and how do the available drafters compare with the
built-in MTP head and DFlash 2 on Qwen3.6-35B-A3B?

Qwen3.5 / Qwen3.6 run on their own text model, not the shared decoder, but already capture DFlash's taps (layer outputs,
in layer order, over the logits' rows and over every prompt row). The EAGLE-3 head there is the shared `MtpHead`:
attach sets the tap layers to the drafter's, and propose / prefill fuse the captured taps (reordered to the drafter's
order), `record` them and call the head's own `propose` / `prefill`. Core now reserves `drafts + 1` recurrent
checkpoint lanes for an EAGLE-3 drafter, as for DFlash, so a rejected tail on the GDN layers rolls back from lanes.

Drafters for the 35B: jiapingW/Qwen3.5-35B-A3B-Eagle3-Specforge (MIT, plain EAGLE-3, trained on Qwen3.5-35B-A3B) and
Dogacel/specdrift-qwen3.6-35b-a3b-eagle3 (no license given; local evaluation only), an EAGLE 3.1 drafter: `fc_norm`
(`fcs.{0,1,2}.weight`, one RMSNorm a tap before `fc`, vLLM's layout) and `norm_output` (the chained hidden is the final
norm's output: the head's existing `chain_normed`). Semantics from NVIDIA NeMo AutoModel's and AMD ATOM's EAGLE 3.1
docs. PatchyTisa's caption drafter (M-RoPE, partial rotary) is still refused.

Qwen3.6-35B-A3B-IQ4_KSS-mtp.gguf, `--pa-context-len 16384`, e3bench (t=0.6, seed 7), no throttling:

```
                              tok/s (three requests)     tokens a step   accepted
plain                         200 / 200 / 200
EAGLE-3 jiapingW q4k          257 / 243 / 226            2.38            46.0%
EAGLE 3.1 specdrift q4k       251 / 260 / 209            2.31            43.8%
built-in MTP (2 drafts)       277 / 241 / 207            2.47            73.3%
DFlash 2 (default depth)      276 / 243 / 142            3.43            34.8%
```

Both drafters work; neither is a clear win over the built-in head here. A drafter trained on Qwen3.5 accepts as well
as one trained on 3.6. EAGLE-3's single chained layer degrades least on the story request, where DFlash 2's longer
blocks lose most. Test: the tiny Qwen3.5 text target with a tiny EAGLE 3.1 drafter keeps the graph fixture's greedy
traces, drafts, and replays the head's graphs.

## Run 9 - 2026-11-16 10:30

**Question:** does gpt-oss-20b draft with EAGLE-3, and is it faster?

Drafter: RedHatAI/gpt-oss-20b-speculator.eagle3 (Apache 2.0, speculators format, `norm_before_fc`: one RMSNorm over
the three concatenated taps, `input_norm.weight` 8640 wide, ahead of `fc`). gpt-oss runs on its own model file, not the
shared decoder: it now records the drafter's taps in its forward and drives the shared head through the speculative
mixin. Attach (loading, depth, the attach info) is one `Eagle3Drafter::attach` the shared decoder, Qwen3.5 and gpt-oss
call. Target: ggml-org/gpt-oss-20b-GGUF MXFP4 (and openai/gpt-oss-20b safetensors for parity).

Four problems on the way, three of them gpt-oss's own and older than this work:

1. **Windowed verify rows (paged fattn).** The tiny gpt-oss test with a drafter matched plain decode for 21 steps,
   then not (a window of 8; with a window of 512 it held). A new kernel case (head dim 64, f16, window 8 / 40, sinks,
   3-row verify) failed by 1.05 max abs where the rows of one sequence straddle a block edge (lengths 33, 34, 66, 97
   failed; 35, 64 passed), even with a window of 200 that never binds. The multi-row decode path runs a sequence's
   verify rows as one causal block over the first row's table with the last row's length; a window's tables are cut
   to each row's own blocks, so the last rows read past the first row's table. Windowed tables now stay per row.
   Any drafter verifying on a sliding-window layer through this path was affected.
2. **MXFP4 experts at verify width.** 3 drafts ran at 85-96 tok/s against 325 plain (as measured then, item 3 aside):
   `mxfp4_moe_grouped_gemm_wmma_kernel` 0.28 ms a call, 48 a verify step, for 4 rows. Up to 8 tokens now take the
   per-token GEMV (`mxfp4_moe_gemm`, whose grid already scales with tokens): 85-96 to 188-218 tok/s. A CUDA case now
   checks both against the dequantized experts at gpt-oss's K (1, 4, 12 tokens).
3. **gpt-oss ran one expert per token.** Output was nonsense on both the GGUF and the safetensors checkpoint
   ("The capital of Germany is a city, but the city is ...", llama.cpp: "Berlin. The capital of Italy is Rome ...").
   Per-layer comparison with transformers (MXFP4 dequantized, bf16, CPU): embeddings equal, layer 0's attention equal
   (cos 1.0000 every row), router logits equal, MoE output cos 0.17-0.97 by row. Our gate_up output was
   `[tokens, 1, 5760]`: the MoE hands its experts `[tokens, 1, hidden]` to broadcast over the top-k, and MXFP4's gather
   read top-k from that middle dim, so only each token's first expert ran, at the full weight. The dense layers
   broadcast it; MXFP4's gather now does too. After: GGUF and safetensors both decode llama.cpp's continuation token
   for token. The tiny gpt-oss checkpoint loads its experts dense, which is why it never caught this.
4. **Harmony stopped on its first token.** With the model fixed, every chat ended after `<|channel|>`: Harmony's
   `<|channel|>`, `<|message|>`, `<|start|>` and `<|end|>` were end tokens. A Harmony turn now ends on `<|return|>` or
   `<|call|>`, as llama.cpp and the checkpoint's generation config end it.

Then, the same e3bench requests (t=0.6), MXFP4 GGUF:

```
                          tok/s (three requests)     tokens a step   accepted
plain                     200 / 200 / 200
EAGLE-3 q4k, 3 drafts     173 / 171 / 179            2.33            44.2%
EAGLE-3 q8_0, 3 drafts    179 / 154 / 163            2.29            43.0%
EAGLE-3 bf16, 3 drafts    151 / 151 / 157            2.29            43.0%
EAGLE-3 q4k, 2 drafts     201 / 193 / 204            2.17            58.5%
EAGLE-3 q4k, 1 draft      213 / 210 / 202            1.69            68.6%
```

Acceptance is the card's (k=3: 1.9-2.4). The step is what costs: nsys, 3 drafts, 14.3 ms a step, of which the MXFP4
GEMV is 7.4 ms against 2.0 ms a plain token, at ~850-900 GB/s either way. A plain token reads its 4 experts; 4 verify
rows read 16 expert slots, ~12.6 distinct experts of 32 if routing were random, so even a kernel sharing an expert
across rows would save under a fifth. Per committed token that is ~5.4 experts read against plain's 4: on this
bandwidth-bound card a sparse MoE with 4 of 32 experts active gains from drafting only at 1 draft (1-6%). The earlier
"greedy 2.60 tokens a step, 447 tok/s" was the broken model repeating itself.

## Run 10 - 2026-11-16 13:30

**Question:** does Gemma 4 draft with EAGLE-3, and by how much on the dense 31B and the MoE 26B-A4B?

Gemma 4 decoding was checked against llama.cpp first (gemma4_investigation.md Runs 1-2; the fixes there: KV sizing in
the device map, CPU-spilled layers, BOS on raw prompts, text-only GGUF). Its text model now records an attached
drafter's taps and its mixin drives the shared head ahead of an assistant model, which still loads as before. With a
drafter attached the KV-sharing models' prefill keeps every row (its taps cover them all) rather than narrowing to the
logits rows. Gemma scales its embeddings by sqrt(hidden); the drafter's `input_layernorm` is an RMSNorm, so the
target's scaled embedding reads the same as the raw table.

Drafters: RedHatAI/gemma-4-31B-it-speculator.eagle3 (taps 2, 30, 57) and RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3
(taps 2, 15, 27), Apache 2.0, speculators format, q4k. Targets: google's QAT q4_0 GGUFs, text file alone. e3bench,
t=0.6, no throttling:

```
                              tok/s (three requests)     tokens a step   accepted
31B  plain                     42 /  42 /  42
31B  EAGLE-3, 3 drafts         74 /  67 /  57            2.20            40.0%
26B  plain                    200 / 199 / 200
26B  EAGLE-3, 3 drafts        195 / 200 / 167            2.01            33.7%
26B  EAGLE-3, 1 draft         232 / 237 / 206            1.58            58.3%
```

Greedy, the 31B's ISO-8601 reply with the drafter matches plain decode for 149 tokens (mean |d logprob| 0.008, max
0.133), then splits at a near tie the 4-row verify rounds the other way (plain: '    ' -0.53 / '        ' -0.89;
drafting: '        ' -0.66 / '    ' -0.73). A tiny Gemma 4 (the 31B's layout in four layers: three sliding, one global
with K as V and proportional partial RoPE) with a tiny drafter keeps paged decode's output.

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

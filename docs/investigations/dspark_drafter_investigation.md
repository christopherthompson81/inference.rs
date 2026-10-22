# DSpark drafter investigation

Issue #585: DSpark draft models (DFlash backbone plus a Markov head and a confidence head). Target here: the
Qwen3.8 27B drafter `RadixArk/Qwen3.8-27B-DSpark` against the DFlash 2 drafter we serve today
(`incoai/Qwen3.8-27B-DFlash2`, 7 drafts, q4k), on the RTX 3090 with the recipe-A IQ4_KS target.

## Run 1 - 2026-10-22 16:56

Question: what does the DSpark checkpoint need beyond our DFlash path, and is it likely to beat DFlash 2 here?

Read: the checkpoint's `config.json`, `README.md`, `dflash.py` and `dspark.py` (SpecForge), its safetensors header,
llama.cpp 970b51ce7 `src/models/dflash.cpp` (`build_dspark_markov_head`), and our `crates/inference-models-qwen/src/dflash.rs`.

Licensing: the architecture code is DeepSeek's DeepSpec (MIT) via TorchSpec and SpecForge, so implementing it is
fine. The RadixArk card says `license: other` and gives no terms; that bears on downloading the weights, not on our code.

The checkpoint: 1.86B BF16, five full-attention Qwen3 layers (hidden 5120, 32 q heads, 8 kv heads), taps 5/19/33/47/61
(same as DFlash 2), `block_size` 7, mask 248070, no embedding or head of its own (it uses the target's). Extra
tensors: `markov_head.markov_w1` [248320, 256], `markov_head.markov_w2` [248320, 256], `confidence_head.proj` [1, 5376]
plus bias.

What differs from our DFlash 1/2 path:
- Row semantics. SpecForge drafts from every row: the noise block is `[anchor, mask x 6]` and row j predicts the token
  at start+j+1, so 7 rows give 7 drafts and the verify is 8 wide. Our z-lab path runs `[anchor, mask x n]` and drafts
  from rows 1..n. Same draft positions, one fewer row.
- Markov head. Draft i's logits get `markov_w2 . markov_w1[prev]`, where prev is the anchor for draft 0 and the
  (biased, argmax) pick of draft i-1 after that, so the picks chain position by position. llama.cpp does this
  in-graph and greedy. Each step reads `markov_w2` (127 MB BF16) once per draft: about 0.9 GB, roughly 1 ms on the
  3090 for 7 drafts, against a ~30 ms step.
- Confidence head. Predicts each draft's acceptance from [hidden; markov_w1[prev]] for an early cutoff. Optional.
- Draft-side YaRN. `rope_parameters` is yarn, factor 32 over 8192 (DFlash 2 uses default RoPE, theta 1e7). Our
  `validate_rope_type` rejects any draft-side scaling today.
- Full attention. All five layers are `full_attention` (non-causal, no window); DFlash 2's are 2048-token windows.
  Our windowed KV pool, and with it the CUDA graphs and paged prefix state, need every layer windowed, so this
  drafter would run the eager path with a context cache that grows with the whole prompt.

That last point is the expensive one here. The drafter keeps K and V for every context token: 5 layers x 2 x 8 heads
x 128 x 2 bytes = 20 KB per token in BF16, 5.2 GB at 262K. The target's whole q3 KV cache is about 14 KB per token
(16 full-attention layers), and the 262K setup leaves ~650 MB free, so at the same memory the context would drop to
roughly 100K. Each draft step would also read the drafter's whole context KV (2.6 GB at 128K, ~3 ms).

Published acceptance (SGLang, NVFP4 target, temperature 1.0, thinking on, gamma 7): code 3.3-4.1, chat 3.2-3.7,
math 3.9-4.5, against DSpark v1's 2.6-3.6. The card doesn't compare against DFlash 2. Our DFlash 2 on the agentic
coding workload gives 3.4-4.2 tokens per step at 7 drafts (model card, #545), so the two may be close.

Implication: measure before building. llama.cpp runs both drafters (DFlash 2 and DSpark share its `dflash` arch), so
an A/B there on the same target, prompts and sampling answers whether DSpark accepts more, independent of our code.
If it doesn't clearly win, the Markov head alone isn't worth a full-attention drafter's memory on a 24 GB card.

## Run 2 - 2026-10-22 17:11

Question: does DSpark accept more than DFlash 2 on the same target, prompts and sampling?

Command: llama.cpp 970b51ce7 `llama-server`, Unsloth IQ4_XS target (mainline can't load IQ4_KS), both drafters
converted with `convert_hf_to_gguf.py --target-model-dir` to Q8_0, `--spec-type draft-dflash` or `draft-dspark`,
`--spec-draft-n-max 7`, `-fa on -c 16384 -ctk q8_0 -ctv q8_0 -b 1024 -ub 256` (f16 KV at 16K didn't fit beside the
desktop's 1.3 GB). The cardbench prompts in one turn each: coding (write tests) and design review, short and at 7K
tokens of `read_file` results; model default sampling (temperature 1.0, top_k 20, top_p 0.95, thinking on), seeds 1
and 2, 1,024 tokens max, two alternating passes per drafter. Acceptance from the response `timings`. Script:
`ab.py` and `run_ab.sh` in the session scratchpad. The DSpark log shows `draft-dspark`, `block_size=7`,
`sample_from_anchor=true`, and the GGUF carries `markov_w1`, `markov_w2` and `conf_proj`; the confidence cutoff
(`--spec-draft-conf-min`) stays at its default, off.

```
workload   ctx drafter runs tokens    acc/prop  rate tok/s mean
code         0 dflash     4    472   374/658    0.57       90.6
code         0 dspark     4    472   362/742    0.49       78.5
code      7000 dflash     4   2328  1694/4406   0.38       66.5
code      7000 dspark     4   2328  1686/4458   0.38       62.0
review       0 dflash     4   4096  2474/11290  0.22       47.5
review       0 dspark     4   4096  2376/11976  0.20       43.6
review    7000 dflash     4   1260   902/2492   0.36       62.3
review    7000 dspark     4   1638  1208/2982   0.41       66.9

all:  dflash 8156 tokens, 5444/18846 accepted (0.289), 3.03 tokens per step
      dspark 8534 tokens, 5632/20158 accepted (0.279), 2.96 tokens per step
pass means (tok/s): dflash 67.2 66.3, dspark 62.5 63.0
```

GPU at 54-63 C and 1950-1965 MHz at each server start, without the 0x20 software thermal slowdown bit (reasons 0x0, 0x400, 0x600).

Finding: DSpark is no better than DFlash 2 here. Tokens per step are 2.96 against 3.03, and it wins only the 7K
review case, whose outputs differ (1,638 vs 1,260 tokens). Decode is 5-7% slower in llama.cpp, matching the Markov
head's sequential per-draft projections plus a drafter that attends to the whole context. The workloads with
identical outputs (both code cases, short review) make the comparison exact there: DSpark proposed more drafts for
the same tokens, so it took more steps.

Implication: don't build DSpark for the Qwen3.8 27B target. On this card it would cost about 60% of the 262K
window (Run 1) for slightly lower acceptance. The Markov head on its own is cheap, so if a DSpark drafter appears
for a target with no DFlash 2 drafter (Gemma 4, LFM2.5, Ling), it is worth revisiting, with the full-attention
backbone as the cost to design around (quantized or windowed drafter KV).

## Run 3 - 2026-10-22 17:35

Question: does our DSpark implementation (anchor-row drafting, the chained Markov head, draft-side YaRN, the eager
full-attention path) accept what llama.cpp's does?

Command: `target/debug/inference serve --format gguf -m /mnt/data/models -f Qwen3.8-27B-IQ4_XS.gguf --pa-context-len
32768 --pa-cache-type q8 --mtp-model <drafter> --mtp-n-predict 7 --mtp-draft-isq q8_0`, the same prompts, seeds and
sampling as Run 2 (`ab_ours.py`, acceptance from the `/metrics` proposed and accepted counters), two alternating
passes per drafter. The load log shows `DSpark ... (block of 7, depth 7, greedy draft, taps [5, 19, 33, 47, 61])` and
the drafter's own YaRN (factor 32 over 8192).

```
            runs  tokens  proposed  accepted  rate   tokens/step  tok/s (wall)
DFlash 2     16    7784    15554      5560    0.357     3.50          75.4
DSpark       16    8152    19040      5428    0.285     3.00          49.8
llama.cpp (Run 2): DFlash 2 3.03, DSpark 2.96 tokens/step
```

Finding: our DSpark accepts what llama.cpp's does (3.00 against 2.96 tokens per step), so the row semantics, the
Markov chain and the RoPE match the reference. Our DFlash 2 accepts more than llama.cpp's (3.50 against 3.03); its
probabilistic selector is a likely reason, unconfirmed. DSpark's wall speed is two thirds of DFlash 2's: no CUDA
graphs on the full-attention path, and every draft attends the drafter's whole context cache.

Implication: DSpark is supported for Qwen3.5/3.8 targets. Left for later: a paged cache for full-attention drafters
(graphs, memory accounting), the confidence cutoff, the speculators format, Qwen3 dense targets.

# Inkling 2's own drafter (DSpark over DFlare features)

The official checkpoint's `mtp.safetensors` (0.64 GB BF16): a 3-layer dense drafter (local layers 0 and 2, window
1024) drafting blocks of 8 (the anchor, the bonus token, then 7 masks each predicting its own position) from the
target's residual stream at layer visits 5, 23 and 41 (of 48, the looped core counted per pass) and its normed output.
Each draft layer mixes the four taps (softmax of `fuse_weight`), RMS-normalizes them (`dflare_norm`) and projects its
context K/V (`k_proj_ctx`, `k_norm_ctx`, RoPE at theta 1e6; the target has none); a recurrent (`rnn_hidden`) Markov
head chains the drafts from the anchor token. Reference: vLLM's `vllm/models/inkling2/nvidia/dspark.py` and
`ops/dspark.py`, over its DFlash/DSpark speculators.

## Run 1 - 2026-11-21 13:30

Question: is the drafter worth building, before choosing it over a custom recipe? Two measurements.

Acceptance, offline: the official target's greedy continuations of 6 prompts (prose, code, a story, an explainer, a
chat turn), 512 tokens each; a local patch (not committed) dumped the four taps from one prefill over each (the token
ids from `quantize kld --resident --text`'s base header), and a torch port of vLLM's drafter (`drafter_spike.py`, f32
math over the BF16 weights, kept in `/mnt/data/models/.profile-tmp/kld/`) drafted at every anchor, a draft accepted
while it equals the target's greedy token there.

```
text            mean accepted (of 7)   P(>= 1 .. 7)
1 prose         2.49                   0.72 0.49 0.38 0.29 0.24 0.20 0.16
2 code          4.92                   0.94 0.86 0.78 0.69 0.61 0.55 0.48
3 explainer     1.57                   0.68 0.41 0.25 0.12 0.07 0.04 0.02
4 story         4.91                   0.92 0.84 0.78 0.72 0.64 0.56 0.45
5 hash map      1.99                   0.68 0.48 0.32 0.21 0.15 0.10 0.06
6 chat          2.23                   0.69 0.50 0.38 0.28 0.20 0.12 0.06
all (2928)      3.02                   0.77 0.60 0.48 0.39 0.32 0.26 0.20
```

Verify cost: the IQ4_NL GGUF's eager prompt of t tokens (`inference bench --prompt-len t`, TTFT), against the
graph-replayed decode step's 12.4 ms (80.4 tok/s): 1: 16.2 ms, 2: 21.6, 4: 26.9, 8: 38.1. Less the eager step's ~3.8 ms
over the graph's, a verify of 8 is ~34 ms (2.7 decode steps: the experts' union of eight tokens' routes).

Estimate (a draft pass guessed at ~4 ms, not measured): k drafts verified give 1 + sum P(>= j) tokens a cycle; k = 7:
4.02 tokens for ~38 ms, 9.5 ms a token (1.3x); k = 3: 2.85 for ~27 ms, 9.5 ms (1.3x); k = 1: 1.77 for ~22 ms, no
gain. By text: code and the story ~1.8x, the explainer ~1.0x. The cost of a verify, not the drafter's acceptance, is
what limits it: an 8-token step's experts read most of the union of their routes.

## Run 2 - 2026-11-21 14:00

Design, from the drafter host and the Granite hybrid target (the engine's verify batches, paged rollback and graph
replay need no change):

- Drafter (`inference-drafters`): a config synthesized from the target checkpoint's `mtp_config` (the official
  directory as `--mtp-model`, its `mtp.safetensors` alone); taps by layer visit, 48 the normed output (so the target
  has 49 tap points); the DFlare context (each layer's softmax mix of the four taps, its `dflare_norm`, `k_proj_ctx` /
  `v_proj_ctx`, `k_norm_ctx`) in place of `fc` + `hidden_norm`; the `rnn_hidden` Markov head (state, gate, latent
  through `hidden_out` folded into the target's head, the last committed row's mixed taps as `f_rnn`); logits
  softcapped and cut to the unpadded vocabulary. DFlash layout: the anchor plus 7 masks (`mask_embed`).
- Target (`inkling2.rs`): `DrafterTarget` (taps through `ctx.logits`, the normed embedding) and the
  `SpeculativeTargetMixin` as Granite's; a verify (`SpeculativeDecode`, t > 1) stashes each slot's pre-pass conv state
  and the conv inputs `[k | v | attn | ffn]`, and commit rebuilds a rejected row's state as the window of
  `[state, inputs]` that ends at its last kept row (no replay: the state is the last `width - 1` inputs).
- Then graphs for verify (`SpeculativeGraphState`), then the verify step's expert kernels.

## Run 3 - 2026-11-21 16:30

The drafter runs end to end (`--mtp-model <the official directory>`): `inference-drafters/src/inkling.rs` (config from
`mtp_config`, the DFlare context, the recurrent head), `inkling2_speculative.rs` (taps, rollback, the mixin).

Bugs on the way: the DFlare norms took f32 mixes against BF16 weights (now f32, as the reference); folding the head by
dequantizing the Q8_0 `lm_head` (1.85 GB f32) ran out of the 655 MB the paged cache leaves, so the fold runs through
the head's forward over `hidden_out`'s columns and keeps `[vocab, rank]` at the model dtype; the speculative sizing
counted every safetensors file of the directory (the target's 20 GB of shards) as the drafter's.

Parity: the drafter's eager path over the spike's dumped taps drafts the torch port's tokens at anchors 40, 100 and 200
(at 300 until a near tie at draft 5); the engine's own drafts at the anchors of a run match the torch port's at the same
anchors, and the engine's taps match the prefill taps at every committed position (0.5-5%, BF16). So the drafter and
the target are right, and Run 1's acceptance was wrong: it averaged over every anchor of a text, where a run drafts
from the row after its last rejection and skips the spans it accepted, which are where acceptance is high.

```
inference serve, IQ4_NL GGUF target, greedy, 256 tokens (tok/s)   prose   code    story   explainer
plain (decode graphs)                                               76.5    79.0    80.9    81.0
drafts 7 (accepted 1.35 a verify; P(>= k) .57 .32 .19 .11 ...)      37.0    101.1   39.5    51.9
drafts 3 (accepted 1.19)                                            56.9    104.0   61.2    69.3
drafts 1 (accepted 0.63)                                            62.4    80.1    64.5    66.4
```

Verify runs eagerly (Inkling hands no graph state yet; decode graphs stay off while a drafter is attached, since a
captured forward would replace, so free, the last forward's taps and its replays would read stale ones), and the
draft pass costs ~10 ms (three dense layers, then seven chained steps over the 201,024-row vocabulary). At drafts 3 a
cycle is ~38 ms for 2.19 tokens. Only code gains. What each next step would buy, at drafts 3: verify graphs ~4 ms a
cycle; a cheaper draft pass (its graph, one fused Markov step) ~7 ms; cheaper multi-token experts what they save of
the ~23 ms verify. With all three, prose reaches about plain's speed and code ~1.6x.

Tests: the tiny official checkpoint with a random `mtp.safetensors` drafts on CUDA (BF16, the drafter's paged
attention reads half precision) and keeps the greedy output up to a near tie; with the rollback's window started
before the kept rows instead of after them it fails at step 2.

## Run 4 - 2026-11-21 17:30

Verify graphs: Inkling hands its taps and conv replays through a `SpeculativeGraphState` (the captured forward's,
reinstalled after each replay; the warm forward's state, taken first, keeps the tensors a captured forward's store would
otherwise free mid-capture), so decode graphs stay on with a drafter. Greedy output is the eager run's, token for token.

The commit's rollback was 48 slots x (row selects, the window, a copy, the scatter): ~300 eager launches and ~250
small uploads a cycle. One kernel (`inkling_conv_rollback`) now cuts every slot's pool rows from its stash in one
launch (a table of the slots' pointers and the rows' kept counts uploaded). With its window off by one the tiny drafter
test fails.

```
inference serve, IQ4_NL GGUF, --pa-context-len 8192, greedy, 256 tokens (tok/s)   prose  code   story  explainer
plain (decode graphs)                                                             76.5   79.0   80.9   81.0
drafts 3, eager verify (Run 3)                                                    56.9   104.0  61.2   69.3
drafts 3, verify graphs                                                           65.6   119.6  70.7   79.5
drafts 3, verify graphs + one-launch rollback                                     69.3   122.9  74.7   83.4
drafts 7, verify graphs + one-launch rollback                                     42.2   114.1  45.0   59.3
```

nsys at drafts 3 (prose): a cycle 31.8 ms, the GPU busy 26.8 ms (84%): the graphed verify ~24 ms, the eager draft
pass 2.7 ms (200 launches: the head over the drafts 0.56 ms, the fold's matvecs, the argmaxes, the three layers), the
host ~5 ms. The verify's experts (IQ4_NL indexed matvec) are 7.5 ms of it: 4 rows x 8 routes reach ~29 distinct of
the 128 experts a layer, so a kernel reading each expert once for all its routes saves ~10% of their reads; the verify
is near its weight-read floor.

Memory: at the default cache sizing a speculative run fails with OOM once its graphs are captured. With the cache
capped (8192 tokens) a run peaks at 20.8 GB against plain's 19.7 GB, ~1.1 GB more where the drafter's reservation is
607 MB (its weights): the folded head (103 MB) and the verify graphs' wider activations, replay stash and logits are
not reserved. The cache planner needs a verify row's runtime bytes (per row beyond a decode row, times the widest
graph's rows) reserved when a drafter is attached.

## Run 5 - 2026-11-21 18:30

Memory at the default cache sizing. Peaks at a capped cache (8192 tokens), `nvidia-smi` sampled:

```
plain, one sequence               19517 MB  (1 graph)
drafts 3, one sequence            20503 MB  (2 graphs): +986, the drafter's weights 640
drafts 3, four sequences          20701 MB  (8 graphs): +198 more
drafts 7, four sequences          20835 MB
```

A drafting run's fixed cost past its weights (~346 MB) was the first draft's fold: the head's forward over all 256 of
`hidden_out`'s columns at once, a 206 MB f32 output beside the 103 MB fold it keeps. It folds 32 columns at a time
now, and the fold is reserved with the drafter's weights (`DFlashConfig::runtime_bytes`). The rows a verify step runs
past a decode step's hold their logits, taps and conv stash in each captured graph: the speculative config records
the verify width (`PagedAttentionConfig::speculative_query_rows`) and the load reserves a loader's
`speculative_row_bytes` times the extra rows of every graph bucket up to the serving capacity (Inkling: ~5.4 MB a row,
144 MB at drafts 3 and 335 MB at drafts 7 over buckets 1-4).

At the default sizing both now serve (the cache 4854 / 4663 MB against 5094 before); drafts 7 peaks at 23841 of
24576 MB.

```
default sizing, greedy, 256 tokens (tok/s)   prose  code   story  explainer
plain                                         76.5   79.0   80.9   81.0
drafts 3                                      67.7   120.4  73.7   82.3
drafts 7                                      41.1   111.1  44.6   58.6
```

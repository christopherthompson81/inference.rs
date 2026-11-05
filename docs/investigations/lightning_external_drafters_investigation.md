# Nemotron 3.5 Lightning with its DFlash and DSpark drafters

NVIDIA publishes two drafters for Nemotron 3.5 Lightning (30B-A3B), both NVFP4 only:
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash` and `-NVFP4-DSpark`. The community BF16 repos are those
weights upscaled, so they carry nothing extra. Goal: run the drafters' NVFP4 weights as stored on the RTX 3090 (no
upscale), drafting for the Nemotron-H target, and in doing so find what an external drafter needs from any target, so
the drafter side becomes model-agnostic (Qwen3.5 is the only target today).

Baseline (nemotron_h_cuda_speed_investigation.md Run 25): plain decode ~246 tok/s, built-in MTP 3 with a q4_0 draft
head ~295 tok/s.

## Run 1 — 2026-11-05

**Question:** what is in NVIDIA's DFlash drafter, and can our GGUF NVFP4 kernels (which run on sm_86) read its weights
without re-quantizing?

**Command:** `hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash` (1.18 GB) into
`/mnt/data/models/nemotron-lightning-dflash-nvfp4`; the safetensors header read by hand.

**Config:** `DFlashDraftModel`, `model_type` qwen3, 6 full-attention layers, hidden 2688, 32 heads x 128, 2 KV heads,
intermediate 6144, yarn RoPE (factor 128 over 8192, theta 1e4), non-causal (`dflash_config.causal: false`),
`mask_token_id` 990, no SWA. `has_lm_head: false` (the target's head drafts), `has_embed_tokens: true` with
`train_embed_tokens: false` (a frozen copy of the target's table). Taps: `target_layer_ids` [1, 5, 19, 29, 41, 51]
and `eagle_aux_hidden_state_layer_ids` [2, 6, 20, 30, 42, 52], the same layers counted as layer outputs and as
`hidden_states` indices (embeddings at 0); to confirm against the Qwen3.5 path.

**Weights:** ModelOpt `W4A16_NVFP4`, group 16, attention and embeddings excluded.

| tensors | dtype |
|---|---|
| `embed_tokens` [131072, 2688] | BF16 (704 MB of the 1.18 GB) |
| `layers.N.self_attn.{q,k,v,o}_proj`, `q_norm`, `k_norm` | BF16 |
| `fc` [2688, 16128] (the six taps fused), `layers.N.mlp.{gate,up,down}_proj` | U8 (two E2M1 a byte), `weight_scale` F8_E4M3 a 16-value group, `weight_scale_2` F32 per tensor |
| norms, `hidden_norm` | BF16 |

**NVFP4 against our GGUF NVFP4 block** (`mmvq_gguf.cuh` `block_nvfp4`: 64 values, 4 UE4M3 scale bytes, 32 nibble
bytes; ModelOpt decode in `nvfp4/mod.rs` `dequantize_block_scaled`):
- Value codes: the same 4-bit E2M1 code (bit 3 sign); our table holds the values doubled and the scale decode halves.
- Scales: ModelOpt's E4M3 bytes are our UE4M3 bytes (positive E4M3).
- Nibble order differs: ModelOpt packs adjacent values (low nibble = even index); our sub-block's byte `j` holds values
  `j` (low) and `j + 8` (high). A bit shuffle.
- `weight_scale_2` has no slot in the block: one F32 multiply on the matvec's output, exact.
- Every quantized width (2688, 3072, 6144, 16128) is whole 64-value blocks.

**Implication:** a load-time repack into GGUF NVFP4 is lossless, and the existing sm_86 matvec/MMQ kernels then run
the drafter at 4 bits. Embeddings may share the target's table instead of a second 704 MB copy (to check that they
are equal). Next: map the Qwen3.5 DFlash/DSpark integration for what a target must supply.

## Run 2 — 2026-11-05

**Question:** is the drafter's `embed_tokens` the target's table, so it can be shared instead of loaded twice?

**Command:** a byte compare of the drafter's `embed_tokens.weight` against the BF16 target's
`backbone.embeddings.weight` (`/mnt/data/models/nemotron-lightning-bf16`), both [131072, 2688] BF16, through mmap.

**Finding:** 2688 of 352,321,536 values differ, all in row 990, the `mask_token_id`: L2 norm 1.279 in the drafter
against 0.877 in the target. Every other row is the target's, bit for bit.

**Implication:** the drafter needs the target's embedding table plus one trained row (the mask token, 5 KB), not a
second 704 MB table. With a quantized target (Q4_0 GGUF here) the drafter would read the target's quantized rows, which
is a change from what it was trained on; whether that costs acceptance is a measurement for later. A drafter loader can
keep its own copy of only the rows that differ from the target's table.

## Run 3 — 2026-11-05

**Question:** how is the Qwen3.5 DFlash/DSpark integration built, and which of it is really Qwen3.5's?

**Command:** a read of `inference-models-qwen/src/dflash.rs`, `qwen3_5/{speculative,text}.rs`,
`inference-nn/src/speculative/` and `inference-core/src/speculative/` (an Explore agent's map; the tap indexing checked
at `qwen3_5/text.rs:3197`).

**Finding:**
- The drafter (`dflash.rs`, ~5.5K lines: config, layers, DSpark Markov head, DFlash2 selector, eager / windowed /
  paged context KV, its own graph cache, prefix-state export) imports only `inference_nn`, `inference_quant` and
  `inference_tensor`. It sits in the Qwen crate but needs nothing from Qwen.
- A tap is a layer's output (residual added), 0-based: `tap_layers.contains(&i)` after layer `i`. So NVIDIA's
  `target_layer_ids` [1, 5, 19, 29, 41, 51] are Nemotron-H layer outputs, and `eagle_aux_hidden_state_layer_ids` the
  same as `hidden_states` indices.
- The target supplies: tap captures of every row on prefill (`last_full_capture`) and of the logits rows on decode and
  verify (`last_spec_capture`), carried through CUDA graphs in its `SpeculativeGraphState`; its embedding table (noise
  block `[anchor, mask x (block-1)]`) and lm_head; a paged-cache layer offset for a paged drafter.
- Rejected drafts are never appended to the drafter's context, so a commit needs no drafter rollback.
- Glue that is generic but written in `qwen3_5/speculative.rs`: the DFlash propose / prefill / terminal-commit paths and
  the DFlash branches of the `SpeculativeTargetMixin` impl (plan, graph plans, precapture, bypass, prefix state).
- Really Qwen3.5's: depth tied to GDN checkpoint lanes (`resolve_dflash_n_predict`), retained prefixes sized by GDN
  snapshot slots, the GDN -> PLE -> DFlash commit order, the hyper-connection stream-mean tap.
- Blocks for a second target: `inference-core`'s reservation is gated on `models-qwen` ("DFlash drafters exist only
  for Qwen3.5 targets"), Granite rejects external drafters, drafter weights load from safetensors only (BF16 or ISQ),
  and Granite (`inference-models-other`) cannot reach a drafter living in `inference-models-qwen`.

**Implication:** the drafter must move out of the Qwen crate to somewhere both target crates reach, and the glue
splits into a generic drafter driver plus a small per-target surface (taps, rollback, prefix sizing). Where it goes
is a structure decision for the user.

## Run 4 — 2026-11-05

**Question:** does NVIDIA's NVFP4 DFlash drafter draft for Nemotron-H once the drafter glue is target-agnostic
(`inference_drafters::host::ExternalDrafter`, Qwen3.5 moved onto it with its 55 DFlash/DSpark tests passing)?

**First attempt:** `DFlash taps target layer 51 but the target has 29 layers`. Our Nemotron-H blocks pair a mixer with
the MoE after it; the checkpoint counts them apart (`layers_block_type`, 52 entries). The taps land on
1 = moe, 5 = attention (followed by moe 6), 19 = attention (moe 20), 29 = moe, 41 = mamba (followed by attention, so
a block without FFN), 51 = moe. A tap is now a point in a block, from the layout's `layers.N` names: after the
block, or between its mixer and FFN (a per-block `MixerTap` keeps that residual when tapped). The drafter's layer
check counts checkpoint layers (52), not blocks.

**Command:** `mtp/run.sh df1 256 --paged-attn on --pa-context-len 8192 --mtp-model
/mnt/data/models/nemotron-lightning-dflash-nvfp4` (4 generic prompts, greedy, thinking off).

Attach line: `DFlash ... (block of 15, depth 7, greedy draft, paged context, taps [1, 5, 19, 29, 41, 51])`, YaRN from
the drafter (factor 128 over 8192), checkpoint lanes 1 (Granite rolls back by Mamba replay).

| config | per prompt tok/s | mean | accept (len) |
|---|---|---|---|
| plain | 249.8 / 250.5 / 248.4 / 249.0 | 249.4 | |
| MTP 3, q4_0 draft head | 329.5 / 248.3 / 225.8 / 395.7 | 299.8 | 73.0 % (3.19) |
| DFlash, depth 7 (its default) | 231.3 / 130.4 / 133.9 / 514.1 | 252.4 | 36.1 % (3.53) |
| DFlash, depth 3 | 192.5 / 137.8 / 153.4 / 355.1 | 209.7 | 47.6 % (2.43) |

NVIDIA's card gives SPEED-Bench acceptance length 3.16 on average at draft length 7 (2.49 writing to 3.75
multilingual, temperature 1, BF16 target); greedy here 3.47-3.53, so the NVFP4 repack, the taps and the drafter's
mask row are right. A step costs ~14 ms at depth 7 (3.5 tokens) against MTP 3's ~10.6 ms (3.2 tokens): the built-in
head accepts nearly as much at depth 3, where an 8-row verify does not have to be paid.

## Run 5 — 2026-11-05

**Question:** speculative text departs from plain decoding (DFlash 7 at chars 248 / 57 / 111, MTP 3 at 38 / 55 /
944, counting the same). Near-ties, or a verify bug?

**Command:** `LOGPROBS=1 mtp/run.sh ...` (top-2 logprobs per token) for plain, MTP 3, DFlash 3 and 7; then
`mtp/tf.sh 1 4,14,30,53,100`: the chat prompt plus plain decoding's first k tokens sent as one prompt, so token k is
scored by a multi-row prefill instead of a one-row decode.

At the departures, plain's top two: 0.015 and 0.013 nats apart on prompts 0 and 2 (ties), but 0.44 (DFlash 7) and
0.95 (DFlash 3) on prompt 1. Over the shared prefixes the chosen token's logprob moves 0.08-0.35 nats under every
drafter. Plain decoding's own prefill moves it as far:

| position | one-row decode, top two | multi-row prefill, top two |
|---|---|---|
| 4 | -1.283 / -1.361 | -1.030 / -1.265 (order flipped) |
| 14 | -0.650 / -1.086 | -0.668 / -0.998 |
| 30 | -0.000 / -9.588 | -0.000 / -11.069 |
| 53 | -0.533 / -1.482 | -0.880 / -1.502 |
| 100 | -0.056 / -4.292 | -0.069 / -4.053 |

**Implication:** the departures are this stack's multi-row against one-row numerics (Q4_0 experts, BF16 activations,
the chunked Mamba scan against its recurrent step), which every verify inherits; not a drafter or verify bug. Whether
0.3 nats between prefill and decode is right for the plain model is its own question (the tiny parity tests hold at
0.06 on random weights).

## Run 6 — 2026-11-05

**Question:** NVIDIA's DSpark drafter (`Qwen3DSparkModel`: 6 causal SWA-1024 layers, block 8, a rank-512 Markov head,
an attention sink per head; card: SPEED-Bench acceptance length 3.75 on average at draft length 7, against DFlash's
3.16) on the same target.

**First attempt:** `shape mismatch for markov_head.markov_w2.weight, expected: [131072, 512], got: [131072, 256]`: the
Markov head's `w2` is NVFP4 too (`markov_w1` and attention stay BF16), and each layer has `attention_sink_bias` [32].
The drafter now loads `w2` through the same repack, and passes per-head F32 sinks to fattn (`FattnOptions::sinks`,
both the paged and the windowed path) and as an extra softmax logit on its eager path.

**Command:** `mtp/run.sh dsN 256 --paged-attn on --pa-context-len 8192 --mtp-model
/mnt/data/models/nemotron-lightning-dspark-nvfp4 [--mtp-n-predict N]`. Attach: `DSpark ... (block of 7, depth 7,
greedy draft, windowed context, ...)`, its CUDA graphs captured.

| config | per prompt tok/s | mean | accept (len) |
|---|---|---|---|
| DSpark, depth 3 | 295.3 / 189.6 / 206.3 / 362.1 | 263.3 | 66.1 % (2.98) |
| DSpark, depth 4 | 298.0 / 178.1 / 177.0 / 414.2 | 266.9 | 52.7 % (3.11) |
| DSpark, depth 5 | 307.3 / 163.7 / 156.3 / 424.5 | 263.0 | 51.5 % (3.58) |
| DSpark, depth 7 (default) | 281.0 / 148.6 / 149.4 / 490.4 | 267.4 | 41.9 % (3.93) |
| (Run 4) MTP 3, q4_0 head | 329.5 / 248.3 / 225.8 / 395.7 | 299.8 | 73.0 % (3.19) |

DSpark's acceptance is above its card's (3.93 at depth 7), and its text departs from plain decoding at the same
characters as DFlash's (Run 5's multi-row numerics). At depth 3 it verifies the same 4 rows as MTP 3 and accepts 2.98
against 3.19, yet a step costs 11.3 ms against 10.6: its drafting pass (6 layers over the block and the Markov chain
over a 131K vocabulary) costs more than the built-in head's.

**Implication:** both NVIDIA drafters run on Nemotron-H from their NVFP4 weights at NVIDIA's acceptance, through the
target-agnostic `ExternalDrafter`; on the 3090 neither beats the built-in MTP head (~300 tok/s). Next for speed: profile
DSpark's step (drafter forward, Markov chain, context append) against MTP's.

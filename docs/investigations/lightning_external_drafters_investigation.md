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

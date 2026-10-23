# DFlash full-attention drafter cache investigation

Issue #585, the first follow-up to #596. Every published DSpark drafter, and one layer of z-lab's Qwen3.5-27B DFlash
drafter, uses full attention. Our windowed drafter KV pool (`inference-nn/src/paged_attention/windowed_pool.rs`)
needs every drafter layer windowed, so these drafters take the eager path today.

## Run 1 - 2026-10-22 17:55

Question: what does the eager full-attention path cost, and where should a full-attention drafter's context cache live?

What the eager path does (`DFlashDraftModel::draft_hidden_batch`, `ctx_cache`):
- Each sequence keeps its context K/V as one growing tensor per sequence, `[layers, kv_heads, len, head_dim]` in BF16,
  appended by concatenation, so every append copies the whole context, and each draft pads and stacks the batch's
  contexts again. For Qwen3.8's DSpark drafter that is 20 KB per token (5 layers x 2 x 8 heads x 128 x 2 bytes).
- None of it is in the KV planner's budget, so a long session can run the card out of memory after the planner has
  handed the rest to the target's cache.
- No CUDA graphs (they need the windowed pool), and `dflash_prefix_replay` returns `Full` for a window-less layer, so
  a prefix-cache hit replays the whole cached prefix through the target to rebuild the drafter context.
- Measured in #596 Run 3 (Qwen3.8 27B, 7K-token prompts): DSpark 49.8 tok/s wall against DFlash 2's 75.4, at
  similar acceptance.

Read: how the target's paged cache is built and used (a subagent, every item cited):
- The built-in MTP head already has a paged layer of its own after the target's layers (`mtp_layer_idx`,
  `paged_kv_layers`), so an extra paged layer beside the target's is an existing pattern.
- `CacheEngine::allocate_gpu_cache` loops over `num_layers()` with per-layer geometry (`num_kv_heads_for_layer`,
  `k_head_dim_for_layer`); the KV planner sizes tokens from the same config.
- The prefix store's host and disk tiers copy every paged layer with blocks (`write_prefix_through`,
  `write_imported_blocks`), so extra layers save and restore with the target's blocks.
- The scheduler allocates verify slots only on the step after the propose; the MTP propose reserves
  `base_len + n_predict` slots itself through `PagedAttentionMeta.kv_cache_manager`, which the DFlash propose
  context also carries (`SpeculativeProposeBatchCtx.cache`).
- The drafter is attached after the target's cache exists, but `inference-core/src/speculative/config.rs` already
  peeks at the drafter's config before load (checkpoint lanes, memory reservation).

Design: a drafter with any full-attention layer keeps all its layers as auxiliary paged layers of the target's cache.
- Before load, the speculative config declares them on `PagedAttentionConfig` (count, KV heads, head dim); the
  pipelines wrap the model's cache config so the planner and the cache engine see them, always at the model dtype,
  and outside the KV profiles' fingerprint.
- The drafter writes its projected context K/V for committed positions into those layers at the target's slots, and
  writes each draft's noise K/V into the slots the propose reserves for the verify positions (later overwritten by the
  real context once those positions commit).
- Attention is the same paged varlen kernel the windowed pool uses, over the target's block tables, non-causal, with a
  left window for any sliding layer.
- Prefix replay becomes `NotRequired`: the blocks carry the drafter's context. CUDA graphs follow in a second step,
  replaying with the target's block tables.

Implication: build it in that order (config and planner, then context writes and eager paged attention, then graphs),
and measure DSpark against Run 3's 49.8 tok/s after each.

## Run 2 - 2026-10-22 18:14

Question: with the drafter's layers in the target's paged cache (eager attention over the target's block tables, no
graphs yet), does DSpark keep its acceptance, and what does it gain?

Change: `PagedAttentionConfig::with_auxiliary_layers` declares the drafter's layers before load, the pipelines wrap
the model's cache config (`AuxiliaryPagedKvCacheConfig`), and the drafter writes committed context and each draft's
noise rows into those layers and attends them with `flash_attn_paged_varlen`. First real run failed: the cache engine
allocated every unquantized FlashInfer layer at one engine-wide element type, u8 under a quantized cache, and refused
to mix quantized and BF16 layers. Now each layer allocates at its own type and only the model's own layers must share
one. A second failure: a one-token context write was `[1, 8, 128]` with a 128 stride on the unit dimension after the
transpose, which `reshape_and_cache_flashinfer` rejects, so the write forces a row-major copy.

Command: Run 3 of the DSpark investigation again (`run_paged.sh`: the IQ4_XS target, `--pa-context-len 32768
--pa-cache-type q8`, the DSpark drafter at q8_0, the same 8 prompts x 2 seeds, two passes).

```
                         tokens  rate   tokens/step  tok/s (wall)  KV cache at 32K
DSpark, eager (Run 3)     8152   0.285     3.00          49.8        1088 MB (drafter outside it)
DSpark, paged             8198   0.278     2.94          64.0        1728 MB
DFlash 2 (Run 3)          7784   0.357     3.50          75.4        1088 MB
```

The load log reads `DSpark ... paged context`. Acceptance holds (the outputs differ run to run, so 2.94 against 3.00
is sampling). Wall speed is up 29% with no graphs, the context no longer being re-concatenated and re-padded each step.
The KV planner now counts the drafter: 640 MB more for the same 32,736 tokens, exactly 32,768 x 20 KB.

Implication: next, CUDA graphs for the paged path (the windowed path's graphs already take block tables and slot
mappings as replayed inputs), then a test that a quantized target cache and BF16 drafter layers coexist.

## Run 3 - 2026-10-22 18:17

Question: are CUDA graphs for the paged path worth building now?

Arithmetic from Run 2: DSpark paged runs 2.94 tokens per step at 64.0 tok/s, about 46 ms a step; DFlash 2 with its
drafter graphs runs 3.50 at 75.4, also about 46 ms. The gap is acceptance, not step time, so graphs would save only the
drafter's launch overhead. They would also need a block-table width bucket per context length: the paged kernel sizes
its KV extent from the table width (`paged_kv_len` = width x page size), so one graph at the full 262K width would plan
every replay over the whole window.

Tests added: the planner keeps auxiliary layers at the model dtype under `q4` with per-layer overrides and leaves the
KV profile fingerprint unchanged; the cache config reports their geometry and bytes; on the tiny Qwen3.5 model, a
full-attention DSpark drafter keeps greedy output beside a `q8` target cache, and a stored prefix restores with the
drafter's context after a restart.

Implication: ship the paged cache without graphs; graphs (with width buckets) stay open in #585.

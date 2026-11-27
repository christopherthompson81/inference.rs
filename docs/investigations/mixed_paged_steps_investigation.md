# Mixed paged steps (decode rows riding prompt chunks)

Goal: one forward carries prompt-chunk rows and the decode rows of sequences already generating (vLLM's chunked
prefill), so decodes progress during other sequences' prefill instead of alternating with it
(`concurrent_prefill_investigation.md` Runs 2-4: prefill-first interleave gains ~35% prefill throughput and stalls the
decoders; no step mixes the two).

## Run 1 - 2026-11-27 14:30

Question: what stands between today's engine and a mixed step? (Code map, no runs.)

Today every step is all-prompt or all-completion: `schedule` (paged_scheduler.rs) returns early with prompts unless
`completion_is_due`, and the engine infers the kind from `scheduled[0].is_prompt()` (engine/mod.rs). The flag drives
CUDA admission, timing, progress, the prefix snapshots, `is_final_prompt_chunk`/`needs_logits` (batch-wide), the
decode lookahead, logits kept per row vs batched, and sampling for every row.

Kernels are not the blocker: the prefix-prefill attention path (FattnPaged/FA3 `try_prefix_gather_prefill`) takes
packed ragged queries with cached prefixes (`num_cached_tokens`, `query_lens`, `cu_seqlens_kv`); a decode row is
`cached = len - 1, query = 1`. What blocks it:

- `inputs_processor.rs::prompt_chunk_inputs` packs only rows with no prefix hit, no chunk offset, final and causal.
- CUDA decode graphs refuse steps with cached tokens or a first chunk (decoder.rs): mixed steps run eager, no
  lookahead.
- Hybrid recurrent caches: the batch kind is per forward (prefill gathers/scatters state, decode updates in place),
  and freshness is one batch-wide `is_first_prompt_chunk`; packed GDN has only run on fresh rows.
- Speculative verify rows need k+1 logits a row (`last_n_context_len` is batch-wide) and a homogeneous staged batch;
  drafter taps and follow assume one row shape.
- Inkling 2 implements no packed prefill: its forward is `[b, t]` with one `past`, batch-wide decode/verify/in-place
  flags and window rings keyed on `t`.

Plan, by phase:

1. Plain decode rows, packed-prefill models, no proposer, not hybrid; behind a flag (default off) for A/B. Scheduler
   output gains decode rows; decode slots reserved before prompts admit; the prompt budget less the decode rows.
   Engine: a step kind, decode rows in admission (dropped first under pressure), per-row progress, finality over
   prompt rows, lookahead off. Pipeline: a mixed input path (decode rows as `prefix_cache_len = len - 1`), packed
   inputs allowing cache hits, sampling only decode rows and final prompt rows.
2. Speculative verify rows ride along (not hybrid): per-row logit counts, verify rows as `(len - 1, 1 + k)` chunks,
   the staged-batch and drafter-tap paths per row subset.
3. Hybrid GDN models with packed prefill (Qwen3.5, Qwen3-Next): per-row fresh state, decode rows through packed GDN
   with existing state, snapshots per row kind, the recurrent prompt batch reserving decode rows.
4. Inkling 2 and its DFlash drafter: packed prefill in `inkling2.rs` (per-token positions and tau, biased attention
   over `cu_seqlens` with a per-row past, window rings per row, per-row recurrent mixer modes), then phases 1-3.

Risks: a decode row through the prefill kernel is not bit-identical to the decode kernel (tests need tolerances);
mixed steps lose graph replay and lookahead, so they pay only where the prefill is compute-bound (MoE helps);
preemption order and `defer_prompt_tail` must know decode rows; rewritten `prefix_cache_len` must be restored;
sliding-window models need checking or excluding in phase 1; packed `FlashParams` must carry KV lengths once cached
prefixes are present.

Implies: the Inkling LAN load benefits only at phase 4, which needs packed prefill in Inkling first. Phase 1 is
measurable on its own on a packed-prefill model (llama tiny for correctness, a dense GGUF for the A/B).

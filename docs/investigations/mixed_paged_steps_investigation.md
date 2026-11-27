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

## Run 2 - 2026-11-27 15:40

Phase 1 built, behind `INFERENCE_RS_MIXED_PAGED_STEPS=1`. A decode row is the final one-token chunk `[len - 1, len)`
of its sequence, appended by the scheduler to a scheduler-visible prompt batch (`add_mixed_decode_rows`, slots
reserved as a decode step's are, a row that does not fit waiting a step). Packed prefill now takes rows with cached
prefixes (not on sliding-window models: Gemma 3's tiny tests failed with it), the paged metadata carries each row's
finality (`final_prompt_rows`), a step wants logits where any row is final, and a mixed step samples only its final
rows. Gate: `supports_mixed_paged_steps` (packed prefill, no proposer, not hybrid, no sliding window); packed prefill
needs F16/BF16 activations, so the tiny test runs BF16.

Test `a_decode_riding_a_prompt_step_decodes_as_alone` (llama tiny, BF16): a 256-token decode with a long prompt
arriving 2 ms later; a step carried both (`(0, 148)` and `(38, 39)`), and each trace matched its run alone within
0.06. (Two dead ends on the way: an f32 load never mixed, since packed prefill refuses f32; and a 48-token decode
finished inside the 30 ms delay. A debug line that locked the pipeline twice in one statement hung the engine at
startup, which read as a slow test.)

Measured on Qwen3-8B Q4_K_M (dense, 32K), `itl.py`: two streaming decoders, then eight ~2K-token prompts (max_tokens 1)
one or four at a time; decoder tok/s while the prompts run, longest gap between their tokens:

```
                                  prompts (1 / 4 at a time)  decoders during   max gap
off                               4.32 s / 4.93 s            70 / 90 tok/s     116 ms
mixed, decode-only steps dropped  3.32 s / 3.23 s            29 / 21 tok/s     111 ms
mixed, interleave kept            4.43 s / 5.05 s            83 / 101 tok/s    112 ms
```

(242 tok/s before the prompts arrive.) Dropping decode-only steps while prompts remain makes every step a prefill
step: prompts finish 23-35% sooner and each decoder gets one token a 512-token step. Keeping the interleave and letting
the decodes ride its prefill steps too gives them +12-19% at ~2% prefill cost. The longest gap is one prefill step
either way.

Implies: mixing widens the trade rather than settling it; where to sit on it is the adaptive policy's choice (load,
queue depth, a latency target), with the chunk size setting the gap.

## Run 3 - 2026-11-27 16:40

The adaptive policy: `--decode-latency-target-ms` (runtime `decode_latency_target_ms`), the longest a running
decode should wait between tokens while prompts prefill. With it set (and no `--max-prefill-chunk-tokens`), the
prompt chunk beside decodes moves with the gaps decodes actually see: each sequence now records when its latest token
landed and the wait before it, and each schedule looks at the longest wait since the last look while prompts are in
flight, shrinking the chunk x0.75 past the target and growing it x1.15 under 70% of it (block-aligned, 256 tokens to
the step budget). Prefill and decode steps alternate one for one under a target; with nothing decoding a prompt takes
the full chunk as before.

Two dead ends first, both from CUDA prompt steps returning at submission (~9 ms, the work showing in the next step):
a prefill rate from each step's execution time read ~200K tok/s, so the chunk took the whole budget (370 ms steps,
380-650 ms gaps); and a gap from wall time between schedule calls misattributed the same way (340-655 ms with mixing).
Tokens landing on sequences is the measure that holds.

Qwen3-8B Q4_K_M, `itl.py` (two streaming decoders, eight ~2K-token prompts, one / four at a time), target 50 ms:

```
                      prompts          decoders during   max gap
no target, mixing off 4.32 / 4.93 s    70 / 90 tok/s     116 ms
target, mixing off    4.09 / 3.74 s    35 / 33 tok/s     138 / 72 ms
target, mixing on     4.02 / 4.06 s    57 / 52 tok/s     126 / 75 ms
```

The four-at-a-time gap settles near the 256-token floor (~58 ms a step here, plus the decode rows); the one-at-a-time
maximum includes the first prompt, before the chunk has moved down from 512. Mixing keeps more of the decoders' rate
under the same bound.

Implies: the target bounds the gap at some decoder throughput, mixing recovers part of it. The floor sets the
tightest bound a model can hold; a smaller one is a fixed-cost question per model.

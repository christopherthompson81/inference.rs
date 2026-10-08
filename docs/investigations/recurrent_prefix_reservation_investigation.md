# #254: recurrent prefix snapshots reserved up front on hybrid models

## Run 1 - 2026-10-08 08:38

Question: after #239's shared budget, what does the up-front snapshot reservation cost in served sequences?

Code today (`crates/inference-core/src/pipeline/mod.rs`, `add_recurrent_prefix_memory_reservations` and
`paged_attention_memory_reservations`): bytes per recurrent snapshot x (`prefix_cache_n` + 1), added to the primary
device's reservation that both the paged KV planner and the recurrent pool fit read (so "both see the same number" and
"size from the configured slots" are already done).

Command: `target/debug/inference run --format gguf -m /mnt/data/models -f Qwen3.8-27B-IQ4_XS.gguf --max-seqs 32
--prefix-cache-n {16,0}` (CUDA build, RTX 3090 24 GB), reading the load logs.

Raw:
- `--prefix-cache-n 16` (default): "Reserving 2560 MB on the primary device for runtime components and activations";
  "serving fewer sequences at once max_num_seqs=32 fitted_seqs=7"; KV cache 1249 MB.
- `--prefix-cache-n 0`: "Reserving 48 MB ..."; "fitted_seqs=23"; recurrent pool capacity 24.

So 17 snapshots (~150 MB each, about one live sequence's recurrent state) cost 16 concurrent sequences: the default
config serves 7 instead of 23. Lazy reservation is the remaining part of #254 and the one that matters.

## Run 2 - 2026-10-08 08:43

Owner chose lazy, budget-aware. Mapped the snapshot lifecycle (read-only survey) to place it.

Raw findings:
- A snapshot is a fresh gather of the slot's committed lane per recurrent layer (`snapshot_recurrent_state`,
  hybrid_cache.rs), held on device by the prefix cacher; paged entries are keyed by block hashes and die with their KV
  block leases (`prune_revoked_paged_recurrent_entries`), non-paged entries follow KV-on-device FIFO eviction.
- The pool has no free list and no slot-to-slot copy; when every slot is owned, `allocate_slot` doubles the pool
  (`resize_recurrent_storage`), which bumps the storage generation and makes the engine drop and re-capture CUDA graphs.
- Requests take a recurrent slot on arrival (`assign_recurrent_slot`, engine/add_request.rs), not on admission;
  `max_num_seqs` = `fitted_serving_capacity` limits only the running set.
- No engine test checks a hybrid prefix-cache hit (`cached_tokens`) at all.

A plain "allocate snapshots lazily from free memory" does not work: without the reservation the planner gives the
memory to the KV cache and the pool, so nothing is free. The workable lazy form keeps snapshots in idle pool slots.

Plan (three PRs):
1. Baseline test: a tiny Qwen3.5 engine test that a repeated prompt hits the prefix cache with a recurrent snapshot
   (paged and non-paged), pinning output equality with the cold run.
2. Pool: a `PrefixSnapshot` slot owner, a slot-to-slot row copy (gather + indexed copy, committed lane to lane 0),
   `allocate_slot` evicting the oldest snapshot slot before doubling and reporting it; prefix cacher entries hold
   a slot handle, both eviction directions free/invalidate; capture and restore copy rows.
3. Drop `add_recurrent_prefix_memory_reservations` from the budget; re-pin the reservation tests; measure the 27B
   (expect ~23 sequences) and check snapshot hits still happen while slots are idle.
Known costs: FullCheckpoints lanes leave N-1 rows of a snapshot slot unused; snapshot slots also count toward DFlash's
capacity-sized auxiliary state.

## Run 3 - 2026-10-08 09:07

Implemented step 2 (snapshot slots in the pool). Checked with
`cargo nextest run --profile cuda --features cuda --workspace --lib --bins --tests -E 'test(prefix) | test(snapshot) | test(recurrent) | test(hybrid) | test(qwen3_5)'`:
254 passed, including the step 1 test `a_repeated_prompt_restores_the_recurrent_prefix`, which still restores the
paged prefix and decodes the same tokens as the cold run, now through a handle instead of tensor copies.

Raw findings while wiring it:
- The non-paged `search_for_matching_cache` clones the entry, so a hit leaves the handle with the entry; only drops
  release it. A stale handle there sends the request through a cold prefill (no hit recorded); the entry lingers
  until FIFO eviction or a replace, as it can never match again with its snapshot.
- Paged validation drops a stale entry and retries the next-longest held prefix, so a reclaimed long snapshot does
  not hide a shorter live one.
- `checkpoint_bytes` now counts auxiliary state only (the recurrent bytes live in the pool); the byte-accounting test
  re-pinned from 72/8 to 64/0.
- `snapshot_recurrent_state` had no caller left and is removed; `restore_recurrent_state` survives as a test seeding
  helper. The checkpoint-lane test now stores and restores through the handle and still sees the active lane land
  in lane 0 with the other lanes zeroed.
- Releases drain once per engine step, after scheduling; a queued release whose slot has since been reused by a newer
  snapshot is a no-op because the id no longer matches.

Next: step 3, dropping the up-front reservation, then the 27B measurement.

Review pass on the same change, raw findings and what was done:
- Eviction was by store age, so a hot shared prefix (one system prompt reused by every request) became the first
  victim. Snapshot owners now carry `last_use`, bumped on restore; eviction takes the least recently used.
- A non-paged hit matched before its request took a slot, and that allocation could evict the matched snapshot.
  The request now touches its matched snapshot before allocating. The paged path still allocates before the
  scheduler validates, so there a matched snapshot can give way if it is the least recently used one; it costs one
  miss, after which the next boundary stores a fresh one.
- A stale non-paged entry now gets dropped on the miss instead of shadowing shorter entries.
- A failed auxiliary capture released nothing, leaving a snapshot slot with no handle until it aged out; it is now
  released, and the block-count check moved ahead of the copy.
- `inference_recurrent_state_slots_used` now includes idle snapshots, so a new gauge
  `inference_recurrent_state_snapshot_slots` tells them apart.

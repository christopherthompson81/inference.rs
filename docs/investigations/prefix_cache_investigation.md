# Prefix cache for agent workloads

Goal: an agent resends its whole history each turn (previous prompt + reply + tool result), often minutes apart.
Each turn should cost only its new tokens, and survive pauses and restarts (a host-RAM / disk tier).

## Run 1 - 2026-10-18 14:00

Question: how does today's in-memory prefix cache serve an agent loop on Qwen3.8 27B (hybrid GDN + attention)?

`agent_sim.py` (scratchpad): a source-code context as the system prompt, a `read_file` tool, greedy 200-token
replies; each turn appends the reply, a tool result or a follow-up. Recipe A, q3 cache, 262K window, DFlash 2 q4k
at 7 drafts:

```
30K context, DFlash      prompt    cached   prompt time
turn 0                   39887     -        36.24 s
turn 1                   40183     -        36.88 s
turn 2-4                 ...       -        37.3-38.2 s
```

Every turn recomputes the whole history. Isolated at 8K:

```
                         agent loop: cached / prompt time per turn    same prompt twice
no speculation           7200 / 0.47 s, 7808 / 0.26 s                 7200 / 0.04 s
DFlash                   none / 5.43 s, none / 5.59 s                 7200 / 0.05 s
```

Cause (code survey): a hybrid prefix hit needs a GDN snapshot at the restore point, taken once per prompt at its
last full block. With DFlash, `HybridPagedPrefixValidator` looked for a snapshot exactly at the end of the cached
KV, else one at least the drafter's replay window (2048 tokens) earlier. An agent turn caches the reply's blocks
past the previous prompt's snapshot, so the exact lookup misses, and a reply shorter than 2048 tokens leaves that
snapshot too close for the replay rule: full recompute. Every snapshot taken with DFlash on also holds the
drafter's checkpoint, so restoring both at it needs no replay.

Also found: no snapshot is taken after a reply (the next turn recomputes the reply and tool result), no host or disk
tier exists, and block hashes use `DefaultHasher` with no model or cache-type salt (not stable across builds).

## Run 2 - 2026-10-18 15:00

Question: does restoring the longest held snapshot with a drafter checkpoint, at or before the end of the cached KV,
give agent turns their hits, and decode as a cold run does?

Change: the validator's auxiliary path uses `longest_held_prefix` up to the KV hit instead of an exact lookup.

```
8K, DFlash agent loop    cached / prompt time per turn
before                   none / 5.43 s, none / 5.59 s, none / 5.67 s
after                    7200 / 0.48 s, 7808 / 0.17 s, 8032 / 0.22 s
```

Correctness: each turn's exact message list replayed on a fresh server (cold cache), greedy: turns 2 and 3 match the
warm (cache-hit) replies exactly, tool calls included; turn 1 matches for its first ~110 characters and then splits at
a near-tie ("the code I have" vs "the code that was given"), the summation-order difference between resuming at a
snapshot and recomputing with other chunk boundaries. `a_dflash_agent_turn_resumes_at_the_previous_prompt_snapshot`
(token-exact turns on the tiny hybrid model, drafter window 128) fails on the old validator and passes now.

Next: a snapshot after each reply, then the host-RAM / disk tier with a stable, salted block key.

## Run 3 - 2026-10-18 17:00

Question: a disk tier for paged prefix entries (hybrid models, with or without DFlash): does a long agent prefix
survive a server restart, how fast does it restore, does it decode as a cold run, and what does write-through cost?

Design (see `crates/inference-core/src/prefix_store.rs`): blocks stored once under their chain hash (now SipHash-1-3
through `siphasher`, identical values with a stability guarantee) with their token ids; an entry is the GDN state per
recurrent layer plus the DFlash checkpoint at a block boundary, and its chain of block hashes, the key file written
last. Write-through when the engine takes a prefix snapshot (new blocks gathered per attention layer, state exported,
files written by a background thread); promotion at admission when disk holds a longer prefix than the GPU cache:
blocks allocated and filled in runs of consecutive ids, registered under their hashes, recurrent snapshot and drafter
checkpoint imported, the prefix entry added with its retention lease, so the scheduler's ordinary match hits.
Namespace per model setup: the loader's identity (selection, local weight files' sizes and times, dtype, ISQ, drafter)
plus the KV, recurrent and checkpoint layouts and block size.

Recipe A, q3 cache, 262K window, DFlash 2 q4k at 7 drafts, `agent_sim.py` with a ~28K-token context,
`--prefix-cache-dir` on the data drive:

```
server 1, cold (writes through)     prompt 28455   cached -       prompt time 23.41 s
server 2, after a restart           prompt 28455   cached 28448   prompt time  0.03 s   (restore 889 blocks, 0.23 s)
server 2, next turn                 prompt 29054   cached 28448   prompt time  0.61 s
store on disk                       583 MB, 1 entry
same prompt, no store               prompt 28455   cached -       prompt time 23.36 s
```

The restored server's greedy reply equals the cold server's, tool call and arguments included. Write-through adds
nothing measurable to the cold turn (23.41 s against 23.36 s without the store). Tests: store unit tests (round trip
across reopen, a block with other tokens drops the entry, LRU collection keeps shared blocks, half-written entries
removed on open) and `a_stored_prefix_restores_after_a_restart` with and without a DFlash drafter on the tiny hybrid
model, which fail with promotion disabled.

Seen in passing: Qwen3.8's tool-call grammar fails to build on this GGUF's tokenizer (`unknown special token:
"</tool_call>"`, logged per request), an existing issue unrelated to the cache.

## Run 4 - 2026-10-18 19:00

Question: does a snapshot after each reply (an engine-issued prefill of prompt + reply while the client acts on the
reply) let the next agent turn skip recomputing the reply, and is the generated token sequence what the next turn's
prompt re-renders?

Re-render check (`rerender_check.py`, Qwen3.8 tokenizer and chat template through `transformers`, reasoning off, a
`read_file` tool): over 4 turns of 54-58 generated tokens, each with two tool calls, prompt + generated token ids are an
exact prefix of the next turn's rendered prompt, every token; the rendered prompt lengths equal the server's (304,
412, 520, 629). A template that re-renders a reply differently (trimmed reasoning) only makes the reply snapshot miss,
falling back to the prompt's.

Change: when a hybrid paged sequence finishes naturally (text, no adapter) and its reply carries the snapshot past the
prompt's, and no other request waits, the engine queues a one-token greedy completion over prompt + generated tokens
(`NormalRequest::prefix_warmup`, response discarded), provided the prompt's snapshot is on the GPU, so it prefills only
the reply; the existing chunk cut snapshots its last full block, and the disk tier writes it through.

Tiny hybrid model, 400-token prompt + 40-token reply: the next turn resumes at 416 (the reply's snapshot) instead of 384
(the prompt's), with and without a DFlash drafter; with the warm-up disabled the same test resumes at 384 and fails.
Without a drafter one decode step's logprob differs from a cold run by 0.044 at the same tokens: the reply's first
block is reused as the original request decoded it, where the cold run prefills it. Control: without the warm-up
(resume at 384, all prefill) warm and cold agree to every bit; and tiny Llama, plain attention, reusing a decoded block
drifts by up to 0.0024 from a cold run. The test checks the resume point and tokens exactly, logprobs within 0.1.

Recipe A, DFlash 2 q4k, `agent_sim.py` at ~40K tokens, 3 s between turns:

```
turn   prompt   cached   prompt time
0      39366    -        35.22 s
1      39661    39392    0.30 s     (cached past the previous prompt's 39366: the reply is included)
2      39953    39680    0.30 s
3-4    ...      ...      0.29-0.31 s
```

Before (Run 2, 28K): follow-up turns resumed at the previous prompt's snapshot, 0.47-0.61 s, the reply recomputed.

## Run 5 - 2026-10-19 01:45

Report: the web UI Code tab's demo stopped after its first round of tool calls, with no error shown. Server log:
`step - Model failed with error: windowed KV checkpoint capacity 9 exhausted`.

Repro (recipe A + DFlash 2 q4k, the LAN config, freshly restarted; a script posting the Code tab's demo then sending
its prompt in `auto` mode, 4 runs back to back): runs 1 and 3 fail with "The model failed to process the request",
runs 0 and 2 pass.

Question: what holds the 9 drafter checkpoint slots (`retained_prefixes` 8 + 1 staging) when the cache keeps one
checkpoint per entry? Reading the code: `snapshot_paged_recurrent_prefix` skips quietly when the recurrent pool has no
idle lane, but turned a full DFlash checkpoint pool into a step error. Capacity eviction runs in
`add_paged_recurrent_prefix`, after the capture, and is bounded by the cacher's owner capacity, not the drafter's.

Change: DFlash's capture and import return `None` when the pool has no free slot; on `None` the pipeline evicts the
least recently used cached prefix holding a checkpoint (`evict_oldest_auxiliary_prefix`, eviction reason
`checkpoint_pool`) and retries, and once none holds one it caches the prefix without a checkpoint (a later hit replays
the drafter). A disk import that finds the pool full skips the entry, as one finding no recurrent lane does.

Same repro, 8 runs on the fix: all 8 finish and their tests pass, no `exhausted` in the log. Metrics after:

```
inference_paged_recurrent_prefix_owners_capacity              16
inference_paged_recurrent_prefix_checkpoints_used             9
inference_paged_recurrent_prefix_owner_evictions_total{capacity}         0
inference_paged_recurrent_prefix_owner_evictions_total{checkpoint_pool}  50
inference_speculative_prefix_cache_captures_total             61
inference_speculative_prefix_cache_hits_total                 60
inference_speculative_prefix_cache_misses_total               6
```

Cause: a sizing mismatch, not a leak. The cacher keeps up to 16 owners (`--prefix-cache-n`), the drafter backs 8
checkpoints, so the cacher never reached its own cap (0 capacity evictions) and the 10th capture found the pool full.
Run 4's reply warm-up roughly doubles captures per agent turn, which is why a few demo runs were enough to hit it. The
checkpoint pool is now its own LRU bound; aligning the two capacities would also work, but the eviction keeps
working whichever is set lower.

## Run 6 - 2026-10-30 14:28

Question (#723): on Qwen3.8 27B with the DFlash 2 drafter (q3 KV, 262K window), a question over the same `read_file`
context a coding request just used re-prefilled the whole prompt. Why, when the final-message snapshot exists?

Commands: `repro723.py` (scratchpad): a context of repository sources as `read_file` results, a coding request (one
32-token turn, or the multi-turn 1,024-token agent run), then a review question over the same context, both with the
same tools; cached tokens from `usage` and the moved `/metrics` prefix counters. Servers from the LAN copy (master
0e3d62ef5) without a host store and with `--prefix-cache-ram-gib 16`, then this branch's build with no store flags.

Finding:
- No store, one coding turn: review 13,216 of 13,246 tokens cached. The final-message snapshot
  (`CheckpointHints::final_message_start`) works with the drafter. (A first try sent the review without the tools,
  so its prompt differed from the start: 0 cached, `no_replay_boundary`.)
- No store, the 8-turn agent run: review 0 of 15,977; 30 `checkpoint_pool` evictions, the review's miss
  `checkpoint_unavailable` with cause `checkpoint_pool`. Without a store the drafter keeps up to `prefix_cache_n`
  prefixes on the GPU (8 here, bounded by recurrent snapshot slots); each turn adds snapshots, a hit promotes only the
  longest match, and `evict_oldest_auxiliary_prefix` drops the least recently used: the shared context's snapshot.
- `--prefix-cache-ram-gib 16`, a 10-turn agent run: review 9,792 of 9,840, `prefix_store_restored_blocks_total` +306.
- Change: an unset `prefix_cache_ram_gib` now gives an 8 GiB memory tier (at most an eighth of the machine's
  memory); 0 keeps none. The store only opens for hybrid models with paged attention. With a store the drafter keeps
  `HOST_BACKED_GPU_PREFIXES` on the GPU: this branch's server, no store flags, logs `retained_prefixes=2` and an
  8 GiB memory store, loads at 20,374 MiB (no store: 20,949), and the agent repro restores 350 blocks from memory.
- Test: `a_prefix_the_drafter_dropped_restores_without_a_configured_store` (tiny model, `PREFIX_CACHE_SEQS + 2`
  agent prompts, no store options): 384 of 420 cached with the default, 0 (and a failure) with it forced off.

Implication: the default server keeps a hybrid model's dropped prefixes in RAM and frees ~575 MiB of VRAM the drafter
held for them.

## Run 7 - 2026-10-30 16:04

Question: the #723 benchmark servers logged `Failed snapshotting recurrent prefix for sequence N: DFlash sequence N is
at position P, expected checkpoint boundary B` five times, P 3-8 short of B (1629/1632, 33692/33696, 146264/146272,
11834/11840). Which snapshot fails, and what does it cost?

Commands: the warnings against the server logs around them; a probe test over the tiny Qwen3.5 checkpoint with the
windowed DFlash drafter (`tiny_dflash(128)`, 2 drafts): a prompt, a greedy reply of R tokens (`ignore_eos`), then
prompt + reply + follow-up, printing the follow-up's cached tokens per R.

Finding:
- Every warning sits 1-3 ms after a `request completed`: the decode snapshot of a sequence's last verify step. The
  drafter appends a verify step's kept rows (anchor plus accepted drafts) to its context only when it drafts the
  next block (`dflash_propose`); a finished sequence never drafts again, so its drafter context stays at the step's
  start while the target sits at the boundary. B - P is the kept rows, at most the 8 of a 7-draft block.
- Probe, 400-token prompt, R 20-60: the follow-up always reaches the reply's last full block. The reply warm-up
  (`queue_reply_warmups`) prefills the reply on top of the prompt's snapshot, so the miss costs that prefill (up to
  the reply's length) rather than a re-prefill of the context.
- Probe, 20-token prompt (no block, so no warm-up), R 8-50: R 13 (ends at 32 computed) caches 0; R 45 (ends at 64)
  caches 0 though R 14-44 cached 32: a decode snapshot releases its predecessor before it captures, so the failed
  final one also loses the earlier one.
- Change: `SpeculativeCommitRow` carries the sequence id, and Qwen3.5's `speculative_commit` appends a terminal row's
  kept rows to the windowed drafter's context from the verify capture. Probe after: R 13 caches 32, R 45 caches 64,
  every other R as before.
- Test: `a_dflash_reply_ending_on_a_block_boundary_snapshots_there` (20-token prompt, R 45): 64 cached and decodes as
  a cold run; without the change, "resumed at 0, not the reply's final boundary (64)". Peak 420 MiB alone.

Implication: a reply ending on a block boundary keeps its final snapshot; an agent's next turn no longer depends on
the warm-up to cover it.

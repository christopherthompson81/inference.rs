# Agent session compaction

Goal (from the tqLLM feature triage): agents that run long can reach the model's context limit; the compaction that
then summarizes the session should succeed even near the limit and run quickly. Measure what compaction costs on this
stack first, then decide what the server can do about it (prefix reuse, KV offload or compaction, context editing).

## Run 1 - 2026-10-21 22:30

Question: what does a Claude Code style compaction cost against our server, by session size?

How Claude Code compacts (captured requests through `inference launch claude`): every request carries
`max_tokens` 32000, `thinking: adaptive` and `context_management: {edits: [clear_thinking_20251015]}`; compaction is
one more `/v1/messages` request with the whole history plus a summary instruction. The server accepts any prompt up to
the window (`fit_prompt_to_context`); one that leaves less than `max_tokens` is not refused, generation just stops at
the window.

Command: `compactbench.py` (scratchpad): an agent-shaped history (WikiText served as `Read` tool results, 12K
characters each) at a target size; one warm request (`Reply with OK.`, 16 tokens) to prefill and cache it, then the
compaction request (Claude Code's summary instruction, `max_tokens` 20000), streamed through `/v1/messages`. Server:
the LAN configuration (Qwen3.8 27B recipe A, q3 KV, DFlash 2 at 7 drafts, 262K, disk prefix store) on its own port.

```
history     prompt    warm (cold) TTFT   compaction TTFT   cached      summary tokens   decode     compaction total
64K         65,395    67 s               0.44 s            65,376      12,403           103 tok/s  121 s
128K        125,780   179 s              1.21 s            125,760     20,000 (cap)     85 tok/s   236 s
200K        193,826   364 s              364 s             0           4,435            86 tok/s   416 s
```

Raw findings:

- Prefix reuse works up to 128K: the compaction prefills only its instruction (0.4-1.2 s).
- At 194K the compaction reused nothing and re-prefilled the whole session: 6 minutes, at the point a compaction is
  most likely. No warning or eviction line in the log. The three sizes ran one after another on one server, so the
  earlier sessions' cached blocks (about 225K tokens) were in the 262K-token cache when the 194K one arrived.
- The summary itself is the rest of the cost: 4-20K tokens at 85-103 tok/s (2-4 minutes); one ran to the 20K cap.

Next: the 194K case alone on a fresh server, to tell a capacity interaction from a size limit.

## Run 2 - 2026-10-21 23:10

Question: was Run 1's 194K miss a size limit or the earlier sessions in the cache?

Command: the 194K case alone on a freshly started server (empty disk store), another WikiText offset.

```
history     prompt    warm (cold) TTFT   compaction TTFT   cached      summary tokens   decode     compaction total
200K        194,073   364 s              1.65 s            194,048     6,723            70 tok/s   98 s
```

It hit, so Run 1's miss came from the cache's other contents. The fresh server's metrics after this one session:
`paged_prefix_retention_pressure_evictions{allocation_pressure}` 6, `recurrent_prefix_owner_evictions{block_pressure}`
6, `{checkpoint_pool}` 6, `recurrent_prefix_validation{miss,no_replay_boundary}` 4, `{miss,checkpoint_unavailable}` 1.

A survey of the lookup paths (17 ways a lookup ends without reuse) found the existing
`inference_paged_recurrent_prefix_validation_total{outcome,reason}` covers only the recurrent validation, nothing is
logged per request, and two candidates for Run 1's miss:

- Block pressure revokes the oldest retention lease (`block_pool.rs:907`), and a matched prefix's lease is touched
  only at commit, after allocation, so the allocation for a request can revoke the lease of the prefix it reuses.
- A new request's recurrent slot (`add_request.rs:495`, paged mode) can evict the oldest snapshot, which can be the
  one it is about to restore; #536's smaller pool leaves fewer spare slots.

Also: a single long prompt has one recurrent snapshot, at its end (`recurrent_checkpoint_boundary`), so losing it costs
the whole prefill; a multi-turn session falls back to an earlier turn's snapshot (`longest_held_prefix`).

Next: a miss reason per request (logged and counted), then rerun Run 1 to name its cause.

## Run 3 - 2026-10-21 23:55

Question: with a miss reason per lookup (new: `inference_prefix_cache_misses_total{reason,cause}` and a log line,
eviction causes recorded by the block pool and the prefix cacher), why did Run 1's 194K compaction miss?

Command: Run 1 again (64K, 128K, 194K on one server, empty disk store), this build.

Same timings as Run 1 (194K compaction: TTFT 364 s, cached 0). Its lookup:

```
prompt_tokens=193862 matched=193792 reused=0 reason="checkpoint_unavailable" cause="unknown"
```

Every KV block matched; the recurrent snapshot (with the DFlash checkpoint) for the prefix was missing, and none of
the recorded eviction paths (block pressure, capacity, checkpoint pool, snapshot reclaimed) had removed it. So the
entry was never kept, or went by an unrecorded path. Two candidates had no record: a capture skipped for want of an
idle recurrent slot (`store_recurrent_snapshot` returning `None` returns silently) and an owner released when a later
snapshot of the same sequence replaces it. Both now record a cause (`no_snapshot_slot`, `replaced`).

The first version also logged ordinary hits (a hit reuses one block less than it matched, the drafter's replay) and a
shared 32-token preamble that cannot restore (`no_replay_boundary`); a lookup is now logged above debug only when it
loses 1,024 cached tokens or more, or a block was evicted.

## Run 4 - 2026-10-22 00:00

Question: the recorded paths did not explain the 194K miss (`cause="unknown"`, and neither `no_snapshot_slot` nor
`replaced`); where did the snapshot go?

Command: Run 3's sequence with temporary traces on every capture, entry add / remove and validation (not kept).

```
23:43:27.537  capture seq=8 tokens=193824 -> add, 6057 blocks, with the DFlash checkpoint
23:43:28.869  validate seq=9 cached=193792 valid=0 reason=checkpoint_unavailable   (6056 blocks matched)
```

The warm request's snapshot was taken and held. Its prompt ends `... tool result, "Reply with OK."`; the snapshot
sits at the prompt's last full block (block 6057), which already holds the first tokens of that final message. The
compaction prompt replaces that message with the summary instruction, so it diverges inside block 6057: its KV match
stops one block short of the only snapshot, and a single long prompt has no earlier one to fall back on. Not
capacity at all: the 64K / 128K prompts (and Run 2's offset) happened to end their last full block before the final
message began. The block-pressure revocations seen were the older sessions'.

A Claude Code compaction appends its instruction after the whole conversation (the previous prompt's snapshot is a
true prefix, and the reply warm-up adds one past the reply), so it likely hits; anything that replaces or edits the
final message (a retry, a regenerate, an edited prompt, a client that drops the last tool result before compacting)
diverges inside that block every time and loses the whole prefill. That is what progressive snapshots would bound.

The miss reason now names this case: with no recorded eviction, a lookup whose blocks lead into a longer entry
reports `snapshot_beyond_match`.

Also found: `run3.sh` backgrounded `cd ... && ./inference ... &`, so `$!` was the subshell and each run left its
server running; fixed. The server itself stops on SIGTERM in 2 s.

## Run 5 - 2026-10-22 01:30

Question: with a second recurrent checkpoint where a chat's final message starts (the template's turn-start token,
learned once from a probe render and accepted only when it is a special token), does Run 1's 194K compaction hit?

Command: Run 1's sequence again (64K, 128K, 194K on one server, empty disk store), this build.

```
history     prompt    warm (cold) TTFT   compaction TTFT   cached      summary tokens
64K         65,395    68 s               0.41 s            65,376      12,403
128K        125,780   182 s              1.29 s            125,760     20,000 (cap)
200K        193,826   363 s              0.53 s            193,792     4,042
```

The 194K compaction now resumes at the block before the final message (6056 blocks, the last one short of it) instead
of re-prefilling 6 minutes. No `checkpoint_unavailable` lookup in the log. The tiny-model test
`a_new_final_message_resumes_at_the_previous_one` reuses 0 tokens with the boundary disabled, the history with it.

Also fixed: `inference_prefix_cache_misses_total{reason="recurrent_and_auxiliary"}` (2 here) counted hits that reuse
one block less than they matched (the drafter's replay), which the log already skipped; the counter now takes only the
lookups that lose 1,024 tokens or more, find nothing cached, or lost a block to eviction. The two
`no_replay_boundary` / `snapshot_beyond_match` ones were the shared 32-token preamble of each new session.

Next: progressive snapshots (every 16-32K tokens to the host / disk tier), so a divergence earlier than the final
message (an edited earlier turn, context editing that clears old tool results) costs at most one interval.

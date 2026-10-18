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

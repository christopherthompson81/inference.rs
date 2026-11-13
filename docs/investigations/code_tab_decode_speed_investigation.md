# Code tab decode speed investigation

Question: after the 2026-11-12 Code tab and Rust 1.100 changes, did decode slow down? The user saw the Code tab report
about 50 tok/s at 200K context on Qwen3.8 27B (recipe A GGUF, q3 KV, DFlash 2 q4k at 7 drafts, RTX 3090), below what
the model was measured at. Reference: the card re-measure of 2026-10-30 (master #722), DFlash coding 151 tok/s at
0.4K down to 139 at 199K, verify step 27.9 -> 37.5 ms.

## Run 1 - 2026-11-12 22:58

The LAN server log's 5-second throughput lines from the user's own sessions since the 21:37 load, keeping intervals
with one sequence running and no prefill (283 of them): decode median 92.0 tok/s (p10 73.6, p90 115.8), MTP accept
median 34.2% (3.39 tokens a step).

A first benchmark against the LAN server overlapped one of the user's sessions (2 running; the user's decode fell to
25-45 tok/s while the benchmark prefilled up to 100K tokens beside it) and was cut off when the model was unloaded.
Discarded. Lesson: never benchmark against the LAN server while it is in use.

## Run 2 - 2026-11-12 23:20

`agentbench.py` (scratchpad): a Code-tab-shaped request (system prompt, `read` calls of `inference-core` sources as
tool results, a coding question, 1000 tokens, model-default sampling), once with the six tool definitions and once with
the reads as plain user text, each cold then warm, against the idle LAN slot (the user unloaded and stopped their
work; the slot reloaded with its saved settings). Warm runs (decode on a cached prefix):

```
depth    tools          plain          accepted/step   step (tools)
5K       125-128 tok/s  123-131        2.3-2.5         27.3 ms
86K      105-112        86-93          2.0-2.6         ~34 ms
164K     94-96          82-89          2.1-2.7         39.2 ms
```

Cold prefill: 1,606 tok/s at 4K, 823-882 to 86K, 471-475 from 86K to 164K.

The step time is level with the reference at short context (27.3 vs 27.9 ms) and 5-9% above it at 164K, inside the
card's session-to-session drift (thermal throttle reason 0x20 was set at the start; one short GPU test overlapped a
run). The rate gap to the reference is acceptance: ~2.5 accepted drafts a step here against ~3.2 on the card's coding
prompts. No sign that the day's changes slowed decode.

## Run 3 - 2026-11-12 23:25

Why the user's turns felt slower than the decode rate: the scheduler's prefix lines for the user's 156K-209K session.
Every turn reuses less of its prefix than matched:

```
prompt_tokens  matched   reused   recomputed  reason
176992         176960    168032   8960        recurrent (replaced)
179402         179360    176960   2442        recurrent (replaced)
182607         182464    177344   5263        recurrent (replaced)
186179         186144    180544   5635        recurrent (replaced)
188072         187840    185120   2952        recurrent (replaced)
191362         188928    186720   4642        recurrent (replaced)
```

The hybrid model needs a recurrent-state checkpoint at the point it resumes from; the one at the previous turn's end
was replaced in the checkpoint pool, so the turn resumes at an older checkpoint and re-prefills 2.4K-9K tokens at
~300 tok/s (prefill at that depth): 8-30 s before each turn's first token, on top of the new tokens. The log shows
the 5-second windows of those turns as `prefill 300-400, decode 0`.

Next: why the turn-end checkpoint is replaced before the next turn (pool size, eviction order, or the label requests
that run between turns taking pool slots).

## Run 4 - 2026-11-12 23:35

Trying to reproduce Run 3's per-turn re-prefill on the idle LAN slot (log lines `Prefix cache reused less than the
cached prefix`):

```
setup                                                      requests  reduced-reuse lines
3 tool turns at 17K context (twoturn.py, raw HTTP)        3         none
CLI agent fixing the demo package (8 rounds, auto)        8         none
Code tab demo over /ui/api/code (labels on, in process)   8 + 4     one, a label request (no_replay_boundary)
the user's session resumed from a copy, 62K, read-only    3         none (1,938 blocks restored from the disk store)
the same copy cut before its last compaction, 213K        3         none
```

The 213K replay prefilled cold in 432 s (the disk store did not hold the prefix); its next two requests took 171 s and
122 s because the agent compacted the session (213K is past 80% of the 262K context), and the request after that was
19,176 tokens.

None of these reproduce the `recurrent` / `replaced` fallback. Each turn resumed exactly where the previous one ended.
What the live session had that these lack is still open: the user's turns ran while the Code tab page was open
and polling, after earlier compactions in the same live server process, and with the server up for an hour of other
requests (the checkpoint pool was full at 22:18: "Recurrent prefix cache full at 16 sequences"). Next: check whether a
full checkpoint pool makes a decode snapshot replace the turn-end one early (the release in `snapshot_recurrent_prefixes`
runs before the new snapshot is known to fit).

## Run 5 - 2026-11-13 00:25

The user's live session hit it again after the redeploy (23:46-00:07): every turn reused 2K-10K tokens less than it
matched (`recurrent`, `replaced`), and from 23:58 reuse fell to 0 (`checkpoint_unavailable`, then `capacity` and
`block_pressure`), re-prefilling ~155K tokens (4 minutes) a turn. The prompt grew steadily from 115K to 155K with no
compaction in between, so compaction was not the trigger.

Reproduced on a test server (same slot settings, own prefix dir) by driving a copy of that session through the Code
tab's routes with labels on, read-only: each turn reused 2,052-2,066 tokens less than the previous prompt's end.
Temporary trace logging (`recurrent_trace`, removed afterwards) on every checkpoint add, replace, eviction and the
step boundaries showed:

- every checkpoint of the agent's sequence was added with `aux=false` (no drafter state); only the small label
  requests captured `aux=true`
- a decode step landed on a block boundary at 157,184 tokens and was checkpointed; the next turn matched exactly
  157,184 yet resumed at 155,040

Cause: `SpeculativePrefixCheckpointPolicy::uses_auxiliary_state` held drafter checkpoints to text-only sequences
(`modality_signature == 0`, from upstream ef8c2fd95, no reason given). The session holds 10 images (agent screenshot
reads since #933), so its checkpoints never carried the drafter's state, and a resume without it must replay the
drafter's window (`fallback_replay`, ~2K tokens): the boundary moved back past the newest checkpoints to the previous
prompt's second-to-last chunk. With the snapshot pool under pressure that degraded to no checkpoint at all.

Fix: drafter checkpoints for every sequence. The target verifies every token, so a drafter resumed from its checkpoint
can only change acceptance, not the output. Same replay with the fix (155K tokens, images in context, labels on):

```
turn  prompt   cached   prefill
1     155864   0        125.4 s (cold)
2     159376   156576   3.7 s
3     160875   159488   1.9 s
4     162670   160864   2.5 s
5     164799   162848   2.7 s
```

No reduced-reuse lines, 26 checkpoints with drafter state, no capture or restore errors, MTP acceptance 37.6-44.4%
(3.6-4.1 a step) against 34% (3.4) before. Prefixes stored on disk for image sessions before the fix lack the drafter
state, so they will not restore from disk once; new ones will.

On compaction and the disk store (the user asked whether compaction should wipe it): not needed for correctness. The
store is size-capped and least-recently-used, keyed by content, so stale pre-compaction prefixes age out; GPU blocks
of the old prefix are the cost that matters, and they leave under pressure. An explicit drop hint from the agent after
a compaction would free them sooner; not done.

## Run 6 - 2026-11-13 01:10

After #935 a turn of the user's 174K-token session still prefilled for 5 minutes (00:59-01:04):
`prompt_tokens=173904 matched=127680 reused=0 reason="checkpoint_unavailable" cause="snapshot_beyond_match"`. The
prompt itself changed at ~127.7K tokens. The session's image reads named files by path and every request read them in
again: eight of its `/tmp` screenshots were gone (each became a "no longer exists" note) and three game assets the agent
regenerates had been rewritten (mtime 19:04 local), so the history changed at the first of them, and the model was
shown pictures it never read there. With no checkpoint before the change, the whole prompt was prefilled again.

Fix: the session keeps a copy of each image when it is read (`Session::keep_image`, `<session>.images/`), and the image
turn names the copy; deleting the session deletes them. Later requests send the same bytes however the original
changes. Sessions from before keep their path references.

Not done: a change far back in a long prompt (an edited earlier message, say) still finds checkpoints only near the
end; periodic checkpoints through a long prefix would let such a turn resume nearby instead of from zero.

## Run 7 - 2026-11-13 02:30

A `/compact` of the user's session took about 4 minutes and the page's stream dropped ("TypeError: Error in input
stream"). Attaching with curl showed the run had finished on the server: two summary requests of 133,289 and 23,981
prompt tokens, then `compacted` (context ~8.9K tokens). The first one prefilled from zero.

Cause: the summary request shared no prefix with the cached conversation. It sent `enable_thinking: false`, and the
model's template (Qwen3.8, from the GGUF) writes a reasoning-effort line at the top of the system prompt only when
thinking is on, so the prompt differed from its first tokens; it also left out the tool list, clipped long tool
results and dropped images. The stream carried one `summary_text` event per streamed token (4,681 for this
compaction), so a page reattaching replayed them for seconds.

Fix: when the context, the summary prompt and a full reply fit the window, the summary request is the round's own
request (same tools, thinking setting, images, unclipped results) with the summary prompt appended; it falls back to
the chunked requests when it does not fit or the model calls a tool or runs out of tokens. Summary progress reaches the
page at most every 500 ms, and the page reattaches by itself (up to 5 times) when its stream drops. Not measured on the
GPU yet.

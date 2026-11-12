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

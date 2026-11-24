# Inkling 2 in a long agent session

A coding-agent session on Inkling 2 (chat slot: recipe C, DSpark drafter at 7 drafts with auto depth, q8 KV, 237,568
context) building a small Godot game from an empty folder: decode slowed as the context grew, and the prefix cache
seemed to miss. The question: what the engine and the agent do wrong over a 5K-45K token session, and how to fix it.

## Run 1 - 2026-11-23 21:00

The session's server log (a dev build of master serving port 1234 from 20:38; same config as the LAN deploy), the
agent's session file (`~/.local/share/inference-rs/agent/sessions/`, 51 entries: 21 tool rounds, 7 activity labels)
and the 5 s throughput lines.

Raw, per agent turn (prompt grows ~1-5K tokens a turn):

```
seq 3   prompt  1674  matched  1344  reused 0  checkpoint_unavailable  snapshot_reclaimed
seq 5   prompt  4104  matched  1696  reused 0  checkpoint_unavailable  snapshot_reclaimed
seq 7   prompt  4819  matched  4256  reused 0  checkpoint_unavailable  replaced
seq 15  prompt 10381  matched  8384  reused 0  checkpoint_unavailable  snapshot_reclaimed
seq 17  prompt 15368  matched 15360  reused 0  ...
seq 24  prompt 32599  matched 32576  reused 0  ...
seq 29  prompt 34886  matched 34880  reused 0  ...
seq 34  prompt 38001  matched 37984  reused 0  ...
seq 41  prompt 40483  matched 40480  reused 0  ...
seq 46  prompt 44050  matched 44032  reused 0  ...
```

Every turn's KV blocks matched to within a block of the prompt's end, and every turn reused nothing: the recurrent
state at the matched position was gone, so the whole prompt prefilled again. At 32-44K that is 40-50 s a turn
(prefill 1430 tok/s falling to 510 tok/s across a 40K prompt in 512-token chunks).

The load had `fitted_seqs=1` ("Recurrent state for every sequence slot does not fit beside the weights and KV floor"),
recurrent pool capacity 3 (one sequence, two spare slots), `max_num_seqs=1`; the LAN deploys earlier today, loaded
fresh, had capacity 6 and 4 sequences, and their sessions hit (75-85% hit rate, no reuse misses).

Decode: 115-165 tok/s in the first turns, 83-100 tok/s from 15K on, with MTP acceptance 30-45% (mean length 1.6-2.0)
while writing GDScript, 85-95% on repeated text.

Next: why the fit fell to one sequence, and why two spare slots lose the conversation's snapshot.

## Run 2 - 2026-11-23 21:10

Question: does reloading the chat slot in a running server fit fewer sequences than a fresh process?

LAN binary (master 400b6e3c2), `load_chat.py 237568 7 q8_0` through `/v1/manager/load`, process memory from
`nvidia-smi --query-compute-apps`:

```
fresh process, nothing loaded          268 MiB
first load                           21958 MiB   pool capacity 6, max_num_seqs 4
after unload                           838 MiB
load again (and every reload after)  21414 MiB   pool capacity 3, fitted_seqs 1, max_num_seqs 1
Replace (load over the loaded slot)              pool capacity 3, fitted_seqs 1
```

The process keeps 570 MiB once a model has run (stable over three cycles, so not a leak). The first load sizes its
recurrent pool before that memory is taken; every later load sees it as used and fits three fewer slots. The first
load's layout is the one that fits: with 6 slots the device still had 1.97 GB free at idle; the reload leaves 2.5 GB
free with 3. The user's session followed a Replace from the web UI, so it ran at one sequence and two spare slots.

The 570 MiB is not the kernel libraries (62 MB on disk for sm86). The largest per-thread stack among them is 1792 bytes
(`libinferencenn.so`), which at 82 SMs x 1536 threads is ~225 MB of context local memory the driver keeps once a
kernel needing it has launched; cuBLAS workspaces and module code make up the rest.

## Run 3 - 2026-11-23 21:20

Question: with two spare slots, why does the conversation's snapshot not survive from one turn to the next?

Read `hybrid_cache.rs` (`reserve_recurrent_snapshot`, `store_recurrent_snapshot`) and the store sites (prompt-chunk
captures in `pipeline/step.rs`, the paged prefix entry in `pipeline/mod.rs`, finished sequences in `sampling.rs`).
A snapshot takes a free slot, else evicts the least recently used snapshot (`last_use`, the snapshot id, bumped on a
restore). Per agent turn:

- the agent's request finishes; its end-of-turn snapshot is the newest, its live slot frees;
- the agent asks for the step's activity label (a ~300-token request, thinking off, 12 tokens): it takes the free slot,
  and when it finishes its own snapshot needs a slot while its live slot is still held, so the oldest snapshot goes,
  then the next; with two spare slots the conversation's end-of-turn snapshot is among them;
- the next turn matches the conversation's blocks but has no state at their end, and prefills from zero.

So recency alone ranks a 300-token label's snapshot above a 44K-token conversation's. Next: rank snapshots by what
they save as well as by recency (GreedyDual-Size: a snapshot's priority is the clock plus its token count; eviction
takes the lowest and advances the clock to it; a restore renews the priority), and fit reloads like fresh loads.

## Run 4 - 2026-11-23 21:15

Two changes, then the fit again.

- Snapshot eviction is GreedyDual-Size (`hybrid_cache.rs`): a snapshot carries the tokens it covers; its priority is
  the eviction clock plus those tokens, renewed on a restore; eviction takes the lowest priority (then the least recent)
  and advances the clock to it. A new test stores a 40x longer snapshot and then twice the spare slots of short ones:
  the long one stays, the first short one goes, and after 40 slots' worth of short evictions the clock has passed it
  and it goes too. The existing LRU tests pass unchanged with equal lengths. The four store sites pass their tokens:
  mid-chunk captures their boundary, the paged prefix entry its cached tokens, a finished sequence its length, a disk
  restore its blocks.
- The 570 MiB is the process-lifetime pool (#1064): cuBLASLt, GGUF MoE and matmul workspaces that outlive the model
  that first asked for them, reused by the next. A first load sizes before they exist; a reload saw them as used.
  `MemoryUsage::query_for_load` counts the pool's reserved bytes (`CudaDevice::process_lifetime_reserved_bytes`) as free
  where loads size against free memory (the recurrent budget, KV sizing, the auto device map), so a reload sizes as the
  first load did and ends at the same footprint.

Dev build, same flow as Run 2:

```
first load                 pool capacity 6
unload, load (twice)       pool capacity 6, 6
Replace over the loaded    pool capacity 6
process after the replace  21960 MiB (first load in Run 2: 21958 MiB)
```

Next: a session like the user's against this build, to see the snapshots survive the label requests.

## Run 5 - 2026-11-23 21:20

A baseline agent session against the Run 4 build: the `inference agent` CLI (`--url`, `--permission-mode auto`, `-p`)
in a scratch copy of the projects folder (the other Godot projects and the Godot 4.7.2 binary beside an empty `blocks`
directory), asked for a falling-block puzzle game in Godot. The CLI asks for no activity labels (only the web UI's
Code tab does), so this run has no side requests.

Raw: three tool calls (`ls`, a `find` over the Godot folder, `godot --headless --version`), then one reply that decoded
for 5 minutes (87-132 tok/s, MTP acceptance 33-62%) and ended at the agent's reply cap: "Error: a reply hit the token
limit (--max-reply-tokens)" (32,768 tokens). The session file keeps nothing of that reply. No prefix reuse misses.

So a second problem beside the cache: a reply that never ends. Next: the same run through a logging proxy, to read
the reply.

## Run 6 - 2026-11-23 21:30

The Run 5 session again through a logging reverse proxy (each request body and its streamed reply to a file), to read
the reply that ran to the cap. Request: `max_tokens` 32768, no sampling fields and no reasoning effort (so the template's
default effort 0.9 and the model's generation defaults, temperature 1.0).

Raw: the fourth reply (after `ls`, `find`, `godot --version`) was all reasoning: 53,537 characters at the point read,
no content and no tool call, holding 47 fenced code blocks, 30 "Actually", 5 "Let me write": the model drafting the
whole game (piece tables, rotation, board, UI) in its thinking, redrafting each part, never calling `write`.

The same request replayed (`replay.py`, the logged body with overrides), counting a reply that reaches 4096 tokens with
no tool call as a runaway:

```
default effort (high), temperature 1.0     1 of 12 run away (others 101-278 tokens, all a tool call)
effort medium                              3 of 12
temperature 0.6                            1 of 12
a system-prompt line ("keep your reasoning brief; write code with `write` instead of drafting it in your
reasoning")                                3 of 24
```

Neither effort, temperature nor the prompt moves the rate: a runaway is a mode the model falls into at a step that
calls for a design, about one time in ten, and a session has twenty such steps. The model card says as much: "In
internal testing the model sometimes became stuck in long reasoning loops, or repeatedly re-checked its own work
without making progress, particularly when debugging code. We recommend setting limits on output tokens and agent
turns." Past agent sessions (1,720 replies, several models): reasoning median 210 characters, p90 8.7K, p99 30K.

## Run 7 - 2026-11-23 21:45

A reasoning budget in the engine, from the user's suggestion: past the budget, bias the end-of-thinking token so the
model closes its thinking itself and goes on with the reasoning it has.

- `thinking_budget` (tokens) on a chat request; Anthropic Messages' `thinking.budget_tokens` maps onto it.
- `ReasoningBudget` (a logits processor in the sampler): inside a reasoning block (the last open marker with no close
  after it, in the prompt and output together) that has run `budget` tokens, the close marker's logit rises 0.5 per
  token past, capped at 30. Inkling 2's block is `<|content_thinking|>` ... `<|end_message|>` (200008, 200010); the
  markers come from the template's reasoning parser and must be single tokens.
- A processor takes sampling off the fast GPU paths, so the sequence swaps it into its sampler only within 16 tokens
  of the budget (more than a verify step's drafts) and drops it once the block closes. The host verify path builds
  each drafted position's context from the tokens accepted so far, so drafted positions are biased by their own
  count; the device verify paths need a plain sampling plan and step aside while it is armed.
- The agent sends `thinking_budget` 4096 (`--thinking-budget`, 0 for none; the Code tab sends the default).

Raw, the Run 6 request replayed with `thinking_budget` 2048, 16 samples: 14 ended within 108-515 tokens as before; two
reached the budget (2223 and 2226 tokens) and both closed their thinking and made a tool call. With 1024, a reply at
1232 tokens: its thinking stopped inside a drafted piece table ("J":) and the call was a sensible next step (checking
the Godot binary), where unbudgeted it would have drafted on to the 32K cap.

## Run 8 - 2026-11-23 22:30

The user's flow on the Run 4 + Run 7 build: the chat slot loaded twice (as a Replace), then the web UI Code tab driven
over its API (`/ui/api/code/open`, `/ui/api/code/send` with auto permissions and activity labels), in a scratch copy
of the projects folder, asked for the falling-block game.

Raw:
- Both loads: pool capacity 6 (was 3 after a reload).
- 40 min, 72 rounds, 71 tool calls, 22 activity labels; the game written (`main.gd`, `logic.gd`, `sfx.gd`, a scene, a
  logic test under `tools/`), context at the end 83,840 tokens.
- Prefix reuse: every turn reused its prefix; five turns reused less than matched, each from an earlier snapshot:
  31,904 of 36,096, 40,896 of 42,304, 56,480 of 60,896, 62,848 of 67,040, 74,016 of 83,808 (cause "replaced", the
  drafter checkpoint beside the recurrent snapshot). Before: 0 reused on every turn.
- The session sat for minutes twice: the model ran headless Godot scripts that never call `quit()`. The bash tool's
  timeout killed the shell but not Godot, which ran on (one copy 4 min after its call had timed out; another call
  wrote its script to `/tmp`). Fixed: the tool starts each command in its own process group and kills the group when
  the call times out or is dropped (an interrupt); a test checks a timed-out command's subshell does not survive.
- One call searched the whole filesystem for the Godot binary (`find / -name "godot*"`), which its own `ls -R` was
  listing; auto permissions ran it unasked.
- Decode, 5 s windows averaged per few minutes: ~110 tok/s early, 69-76 at 30-40K context, 35-56 at 60-84K.

Next: decode against context length, drafted and plain, outside the agent.

## Run 9 - 2026-11-23 22:45

Decode against context outside the agent (`ctx_decode.py`): one prompt of this repo's Rust source cut at each size
(each size's prefix cached from the last), asking for a new function, greedy, thinking off, 256 tokens; the server's
decode tok/s. The chat slot as served (DSpark at 7 drafts, auto depth, q8 KV), then reloaded without a drafter.

```
context    drafted (7, auto)    plain
  2.4K         100.5             94.7
 17.6K         111.9             97.3
 35.3K          93.4             88.8
 69.1K          70.8             72.5
102.4K          60.0             67.6
```

Plain decoding loses 30% from 17K to 102K: the per-token time grows ~4.5 ms over 100K tokens. The KV cache is 18.7
KB a token (4437 MB for 237,536), so at 102K a decoded token reads ~1.9 GB of KV, ~2.2 ms at the 3090's bandwidth:
attention runs at about half of it at long context. Drafting gains ~10% up to 35K and loses past ~64K (60 vs 67.6 at
102K), where auto depth should have chosen plain decoding.

So the session's decode slowdown is mostly attention over the growing context, with the drafter costing more than it
saves past ~64K. Next: where the decode step's time goes at 100K (attention kernel efficiency on q8 KV), and why auto
depth keeps drafting there.

## Run 10 - 2026-11-24 00:00

Where a 102K-context plain decode step's time goes (nsys, `--cuda-graph-trace=node`, one 129-token reply after the
prompt was cached): 15.7 ms a token, of which the paged attention decode (`biased_attention.cu` `decode_kernel`, q8
cache) 5.75 ms, 48 calls a token. Inkling 2 attends with 24 heads over 8 KV heads (head dim 128), 20 of its 24
layers windowed at 1024 keys; the 8 full-context visits a token (layers 5 and 23 once, 11 and 17 on each of the
core's 3 passes) took ~640 us each.

Changes, each checked by the kernel's reference cases (a new case at 24 heads over 8, on a view and a q8 cache, split
pasts and verify rows; and 8 over 1, which keeps the per-head kernel):

1. A block takes every head of its KV head (2-4 of them), reading each K and V row once instead of once a head.
2. Decode splits are sized by blocks of KV heads, up to 64 a row instead of 16 (62 at one query over 8 KV heads).

```
                         full-context call   102K plain   69K plain   102K drafted
before                        ~640 us          67.6         72.5         60.0
1 (grouped, 16 splits)         578 us          71.4         77.6         58.6
1 + 2 (62 splits)              540 us          71.5         81.1         68.4
```

(tok/s; drafted replies vary in length and acceptance, so their column is noisy.)

A full-context call reads ~223 MB of q8 K and V: 540 us is ~410 GB/s, under half the 3090's. Neither re-reads nor
the grid size was the limit: the inner loop loads q8 rows 4 bytes a lane and dequantizes every value for three dots,
so the kernel is bound by its instructions. Next step for long context, if wanted: 8-byte loads and int8 dot products
on the q8 keys (queries quantized per group), an inner-loop rewrite rather than a launch change.

# Code tab sampling: greedy vs the model's defaults

Question: can the Code tab's agent decode greedily (temperature 0) without hurting results, and what does it buy in
speed? Qwen's model cards advise against greedy decoding for Qwen3-family models (repetition, worse reasoning).

Setup for every run: the LAN config (Qwen3.8 27B recipe A GGUF + DFlash 2 q4k at 7 drafts, q3 KV, 262K), the Code
tab's Run demo task (a temp project with two planted bugs, prompt "The tests in test_stats.py fail. Fix stats.py so
they pass, then run them.", `auto` mode), driven through `/ui/api/code/demo` and `/ui/api/code/send`. A run passes
when the agent finishes and `python3 test_stats.py` exits 0. `temperature: 0` reaches the sampler as greedy
(`Sampler::new` maps temperatures below 1e-7 to `None`), which takes DFlash's greedy device verify.

## Run 1 - 2026-10-19 02:10

Command: a driver posting Run demo then the prompt, 10 runs at `temperature: 0`, then 10 with no temperature (the
model's default sampling), same server process. Recorded per run: pass, rounds (model replies), tool calls, failed
tool calls, repeated identical calls, completion tokens, decode tok/s (completion tokens over completion time, summed
over rounds), wall time.

Control: the same 4-line-poem chat request twice with no temperature gives two different poems; twice at
`temperature: 0` gives the same one. So "default" does sample and greedy is deterministic for a fixed prompt.

```
SUMMARY temp=0: 10/10 pass, rounds 4.0, tool errors 0, repeated calls 0, completion tokens 452.5, tok/s 199.8, wall 4.5 s
SUMMARY temp=default: 10/10 pass, rounds 4.2, tool errors 0, repeated calls 0, completion tokens 555.5, tok/s 132.9, wall 6.5 s
```

Raw runs:

```
greedy  {"run": 0, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 372, "tok_s": 193.1, "wall_s": 3.9}
greedy  {"run": 1, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 350, "tok_s": 212.9, "wall_s": 3.9}
greedy  {"run": 2, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 443, "tok_s": 204.1, "wall_s": 4.5}
greedy  {"run": 3, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 520, "tok_s": 206.4, "wall_s": 4.6}
greedy  {"run": 4, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 452, "tok_s": 199.6, "wall_s": 4.5}
greedy  {"run": 5, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "edit", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 537, "tok_s": 195.1, "wall_s": 5.0}
greedy  {"run": 6, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 432, "tok_s": 200.6, "wall_s": 4.4}
greedy  {"run": 7, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 384, "tok_s": 197.0, "wall_s": 4.1}
greedy  {"run": 8, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "edit", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 571, "tok_s": 187.9, "wall_s": 5.4}
greedy  {"run": 9, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 464, "tok_s": 201.5, "wall_s": 4.5}
default {"run": 0, "ok": true, "end": "done", "err": null, "rounds": 5, "tools": ["bash", "read", "read", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 748, "tok_s": 143.3, "wall_s": 8.1}
default {"run": 1, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 474, "tok_s": 131.0, "wall_s": 5.8}
default {"run": 2, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "write", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 416, "tok_s": 123.8, "wall_s": 5.6}
default {"run": 3, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "edit", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 684, "tok_s": 116.3, "wall_s": 8.3}
default {"run": 4, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 453, "tok_s": 132.6, "wall_s": 5.5}
default {"run": 5, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 482, "tok_s": 126.1, "wall_s": 6.0}
default {"run": 6, "ok": true, "end": "done", "err": null, "rounds": 5, "tools": ["bash", "glob", "read", "read", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 757, "tok_s": 136.4, "wall_s": 8.3}
default {"run": 7, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "write", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 406, "tok_s": 133.1, "wall_s": 5.2}
default {"run": 8, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "edit", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 724, "tok_s": 144.4, "wall_s": 7.3}
default {"run": 9, "ok": true, "end": "done", "err": null, "rounds": 4, "tools": ["read", "read", "write", "bash"], "tool_errors": 0, "repeats": 0, "completion_tokens": 411, "tok_s": 141.8, "wall_s": 5.2}
```

Findings:
- Failure rate: 0/10 greedy, 0/10 default (and 16/16 default passes in earlier runs after the checkpoint-pool fix). No
  repetition loops, no failed tool calls, no repeated identical calls in either.
- Greedy took the canonical path every time (read, read, edit, bash; twice two edits): 4 rounds in all 10 runs.
  Default wandered more: 2 runs opened with an exploratory `bash`/`glob` (5 rounds), 3 rewrote the file with `write`
  instead of `edit`.
- Speed: greedy decodes at ~200 tok/s against ~133 tok/s sampled (1.5x), from DFlash's greedy device verify accepting
  more of the drafter's tokens, and writes ~19% fewer tokens; wall time per task 4.5 s vs 6.5 s.
- Greedy runs still differ from each other (350-571 completion tokens): the system prompt carries the demo's unique
  temp path, so each run's prompt differs.

Limit: the task is easy and short (4 rounds, ~2K context). It shows the speed difference clearly, but it can't show
the long-session failure mode the model card warns about (greedy repetition in long reasoning); that needs a longer,
harder task before greedy becomes a default.

## Run 2 - 2026-10-19 02:25

Question: does greedy's pass rate hold on a task with more rounds, more context and reasoning across files?

Task (`stockroom`, built in the scratchpad, not shipped): a small Python package of three modules (`units`,
`inventory`, `report`), a README stating the intended behaviour, and a stdlib `unittest` suite of 19 tests. Planted:
two wrong unit factors (`dozen` = 10, `kg` = 100), an off-by-one stock check, a `KeyError` for an unknown SKU where
the spec says it holds nothing, `undo()` left as `NotImplementedError` (must skip failed ships), `low_stock` using
`<=` and sorting by quantity instead of SKU, 2 decimals instead of 3, and no singular "1 item". The buggy package fails
14 of 19 (7 failures, 7 errors); a reference fix passes all 19. Prompt: fix the package so
`python3 -m unittest discover -s tests -t .` passes, following the tests and README, without changing the tests. A run
passes when the agent finishes, the suite passes, and `tests/*.py` hash the same as the template (no editing tests).

Server: the LAN config plus `code_root` at the scratch runs directory, so each run opens a fresh copy as a project.
Runs alternate greedy and default (10 each), so drift over time hits both.

Smoke pair: both pass. Greedy 9 rounds, 16 tool calls, 6624 completion tokens, 161.5 tok/s, 14.9K peak context,
58.8 s; default 8 rounds, 16 calls (1 failed), 5792 tokens, 122.7 tok/s, 11.8K peak, 62.1 s. So roughly 2x the rounds,
4x the tool calls and 12x the generated tokens of Run 1's demo.

## Run 3 - 2026-10-19 02:57

Command: `hard_bench.py` with 10 alternating greedy/default pairs; stopped after the first run.

Greedy run 0 never finished. Server log: 02:23-02:25 normal rounds (read the files, 9 tool calls, 3 rounds, 7.8K
context); from 02:25:07 to 02:45:42 one reply decoded continuously, 265 falling to 147 tok/s as the context grew, with
DFlash at `MTP accept 100.0% (len 8.00)` the whole time (a drafter only matches every token of a loop); from 02:45:42
the sequence sat in prefill at ~300 T/s, decode 0, with 1 running and 0 waiting. Interrupted through
`/ui/api/code/interrupt` at 32.5 min (wall 1949 s); the suite still failed 7 + 7, untouched.

The interrupted reply, committed to the session by the interrupt: 25 chars of content and 843,562 chars of reasoning.
The last 200 chars occur 1,993 times: a 423-char cycle repeated ~1,992 times, the first full cycle at char 746
(0.09% in):

```
 `parse_quantity("2.5kg")` → 2500.0. The test expects `(MASS, 2500)`. `2500.0 == 2500` → True. So it passes. But
 for cleanliness, we could convert to int when the value is an integer. Actually, let me be careful — the tests pass
 either way. But the README says "mass is stored in grams, count in items" — the tests use `assertEqual` which works
 with float too. Still, it's cleaner to return int when possible. Let me think:
```

The reasoning had already found both units bugs; the loop is a deliberation it never resolves, which greedy decoding
can't leave: the same context picks the same continuation, and the cycle is self-reinforcing.

Conclusion: greedy is not safe for the Code tab's agent on this model. Run 1's easy task couldn't show it (short
reasoning); the first greedy run on the harder task looped within ~750 characters of its reasoning. The model's default
sampling stays the agent's decoding. The remaining 19 runs weren't needed for that answer.

Harness gap found on the way: the agent sends no `max_tokens`, so a looping reply runs to the full context (262K, ~20
min of GPU here, then the reply warm-up re-prefills it). Sampling makes this rare, not impossible.

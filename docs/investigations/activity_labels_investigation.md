# Activity labels

The Code tab folds each run of thinking and tool calls into one line. The line's summary is built from the calls
("Read 8 files, ran 1 command (9)"); the session's model can say what the run was for instead. The design: one
request after a prompt is answered (never alongside the agent's decoding), built from the agent's own next request so
the conversation's prefix cache carries it, asking for a JSON array of short past-tense phrases.

## Run 1 - 2026-10-22 03:29

Question: on the LAN model (Qwen3.8 27B recipe A, DFlash 2), how long after the answer do the labels arrive, and are
they useful?

Command: a bench server on this build (open mode, port 18761), `ui_labels.py` (scratchpad `labels/`): Playwright
opens the Code tab, sets **Run everything**, runs the demo, and times from the Send button's return to the activity
lines changing. The label request asked for no thinking (`enable_thinking: false`).

```
turn 53.2 s; labels 8.97 s after the turn
'Read 4 files, searched 1 pattern, ran 1 command (6)' -> 'Explored project files and read tests and README (6)'
'Read 4 files (4)'                                   -> 'Read the stockroom source modules (4)'
'Edited 4 files (4)'                                 -> 'Fixed unit factors, inventory logic, and report (4)'
'Ran 1 command (1)'                                  -> 'Ran the test suite to verify fixes (1)'
```

The labels read well, but 9 s is a re-prefill: the engine log shows about 11K tokens prefilled at 1,200-1,300 tok/s
after the turn. Cause: this model's chat template words its system prompt by the thinking setting (the "Reasoning
effort is set to xhigh" sentence appears only while thinking is on), so `enable_thinking: false` changed the very
first block, and the request matched nothing of the cached conversation.

## Run 2 - 2026-10-22 03:35

Change: the label request keeps the session's thinking setting, leaves the reply room to think (1,024 tokens on top of
24 a label) and asks it to answer at once. Same command.

```
turn 33.1 s; labels 2.36 s after the turn
'Read 8 files, ran 1 command (9)' -> 'Read project files, tests, and README (9)'
'Ran 1 command (1)'               -> 'Ran test suite to see failures (1)'
'Edited 6 files (6)'              -> 'Fixed bugs in units, inventory, and report (6)'
'Ran 1 command (1)'               -> 'Ran tests to confirm all pass (1)'
```

No lookup lost cached tokens; the labels were written to the session file and the checkbox switches back to the
summaries. The 2.4 s is mostly the model's brief thinking: a request continuing a pre-filled assistant message
(vLLM's `continue_final_message`, which this API lacks) could skip it, for about 0.5 s; not worth an API feature while
the labels arrive after the answer anyway. Leaving the request's template inputs alone is the general rule: any
setting a template reads can reword the prefix.

## Run 3 - 2026-10-22 03:57

Question: labelling each run as it closes instead of after the answer; what does a label request cost the agent if
the session's model serves it alongside the agent's own decoding?

Command: `conc.py` (scratchpad `labels/`) against the LAN configuration (dev build, idle otherwise): an 800-token
agent-style decode (model default sampling, as the agent sends none), alone and with a label-sized request (735-token
prompt, 300 tokens with thinking) started 1 s in; five pairs, alternating.

```
agent decode alone                   106.1 tok/s (98-115)
with one label request alongside      83.5 tok/s (82-84)
agent time added per label request    2.04 s, for a label that takes 3.08 s
```

The agent runs at about a third of its speed while a label request shares its steps. Two sequences double the verify
rows to 16 (past the 8 the tiled IQ4_KS kernels cover; dflash_tree_verify_investigation.md Run 2) and batch the
drafter, so a second sequence is far from free on this card. Labelling each run live with the session's model would
add about 2 s per run to the agent's time: the demo's 4-6 runs, 8-12 s on a 30-50 s task.

## Run 4 - 2026-10-22 04:05

Question: a small model on the CPU (no VRAM, no share of the agent's steps), labelling one run at a time from a
compact prompt (the task, the run's thinking trimmed to 1,200 characters, its calls), how fast and how good?

Command: `cpulabel.py` over the runs of a saved demo session, each model served alone with `--cpu` (dev build, so
debug assertions on: CPU work runs about 1.6x slower than a release build); thinking off, 24 tokens.

```
model (CPU)                 seconds a label   labels for the two runs                               the 27B's
Qwen3-0.6B q8_0             2.98 / 2.35       "- Run tests." / a list of the edited files           "Explored codebase and ran failing tests"
Qwen3.5-0.8B q8_0           2.47 / 1.91       "read the documentation" /                            "Fixed bugs and verified tests pass"
                                              "Fixed the unit and inventory coefficients."
Llama-3.2-1B q8_0           4.29 / 2.68       a rambling list / "Ran tests, fixed stockroom package."
Qwen2.5-Coder-3B Q4_K_M     13.6 / 8.29       a sentence cut at 24 tokens /
                                              "Fixed unit and inventory tests, updated report formatting."
```

Raw findings:

- Speed is fine up to 1B (2-4 s here, about 1.5-2.5 s in a release build); 3B at Q4_K_M takes 8-14 s on this CPU.
- Quality is not: the sub-1B models echo the calls or name the wrong thing about half the time; only the 3B and the
  0.8B's second label say what the work was for. None matches the session's model, which sees the whole conversation.
- Not measured yet: whether a CPU labeller slows the agent (host-side work of the GPU decode).

Implication: neither live option is free. Live labels from the session's model cost the agent about 2 s per run;
live labels from a small CPU model cost nothing measurable on the GPU but read worse, and a 3B is too slow on this
CPU. The labels after the answer (Run 2) are the only option that is both good and free.

## Run 5 - 2026-10-22 04:20

Question (the user's proposal): the session's model, but a small request: thinking off, at most 20 tokens, and only
the run itself as input (the task, the run's calls, with or without its thinking), sent while the agent decodes. What
does it cost the agent, and how good are the labels?

Command: `conc2.py` (scratchpad `labels/`) against the LAN configuration: the agent-style 800-token decode alone three
times (102.4 tok/s), then for each of the 9 runs of the three saved demo sessions, the decode again with one label
request started 1 s in, once with the run's thinking (its last 1,500 characters) and once with the calls only;
temperature 0, `enable_thinking: false`, 20 tokens.

```
variant            agent time added a label   label latency   prompt tokens
with thinking      +0.40 s                    0.60 s          475
calls only         -0.10 s                    0.41 s          197
(Run 3: full conversation, thinking on, 300 tokens: +2.04 s, 3.08 s)
```

Per trial the added time ranges -1.1 to +1.0 s: the sampled agent decode varies that much by itself, so the calls-only
cost is within noise and the thinking variant's is at most about 0.4 s. Labels against the full-context ones (Run 2):

```
with thinking                                        calls only                                         full context
Read code and tests to diagnose failures             Explored codebase and tests to diagnose failures   Explored codebase and ran failing tests
Fixed stockroom bugs and verified all tests passed   Fixed stockroom package bugs to pass failing ...   Fixed bugs and verified tests pass
Checked test file and ran test suite                 Examined test file and ran test suite              Ran test suite to see failures
Verified all tests passed successfully               Ran test suite to identify failures                Ran tests to confirm all pass
Ran test suite to verify fixes                       Ran test suite to identify failures                Ran test suite to confirm failures
```

Raw findings:

- Thinking off and a 20-token cap take a label from 3 s to about 0.5 s, and its cost to the agent from 2 s to under
  half a second: the prompt is a few hundred new tokens, and the reply shares only a handful of the agent's steps.
- Both variants read as well as the full-context labels. The calls alone can't tell a test run that finds failures
  from one that confirms the fix (the 4th row: calls-only says "identify failures" for the run that confirmed them);
  the run's thinking says which. Both slip into first person now and then ("I explored ..."), which the instruction
  can rule out.

Implication: label each run live with the session's model from a compact prompt carrying the run's thinking, thinking
off, 20 tokens, sent as soon as the run closes; no second model and no wait for the answer.

## Run 6 - 2026-10-22 04:40

Change: labels are made live. The Code tab's agent loop follows its own stream (`activity::RunTracker`: a run takes
the thinking and calls until visible text, the next prompt or the end of the turn), and as each run closes it sends
Run 5's request on a task of its own: the task, the run's thinking (its last 1,500 characters) and its calls, framed
as `<activity>` text to label (opencode's title request frames its input the same way, so a model doesn't carry on the
work instead), thinking off, 20 tokens. The label reaches the page as a `label` event on the prompt's stream and is
kept in the session file. The after-answer request and its route are gone.

Command: `ui_live.py` (scratchpad `labels/`) on a bench server with this build: the demo four times, Describe
activity on, off, on, off, logging every change of every activity line with the time since Run demo.

```
describe on                                                     describe off
  5.0 s  line 0: Read 8 files, searched 1 pattern, ran ...      4.9 s  line 0: Read 8 files, ran 2 commands (10)
 27.6 s  line 0: ... ran 2 commands (11)                        19.3 s line 1: Ran 1 command (1)
 30.0 s  line 0: Read source files and ran tests to confirm     ...
         bugs (11)                                              31.0 s answer done
 35.8 s  line 1: Edited 4 files, ran 1 command (5)
 37.0 s  line 1: Fixed stockroom package and ran tests (5)
 38.3 s  answer done
second pair: on, label 20.0 s after the run's last call at 4.3 s and the text that closed it; answer at 27.0 s;
             off, answer at 65.1 s (four runs instead of two)
```

Each label lands 0.5-2.5 s after its run closes, while the agent works on; the last run's label arrived before the
answer finished. The turn times are the demo's own variance (two to four runs a turn, 27-65 s), not the labels: Run 5
puts their cost at under half a second each.

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

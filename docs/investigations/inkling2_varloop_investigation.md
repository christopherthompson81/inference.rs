# Inkling 2 varloop investigation

Thinking Machines' `TML-d24-varloop-research` is Inkling 2 Mini's architecture (same tensor names and shapes, same
tokenizer) with `loop_config.is_varloop: true`: midtrained at 1 to 5 passes of the looped core (sampled 1:2:6:2:1),
post-trained at 3. The question: does it run usefully at other pass counts than 3, where Inkling 2 Mini degenerates at 2
(loop guidance investigation, Run 2), and what does each count cost in speed?

## Run 1 - 2026-11-25 06:45

Code: `LoopConfig::is_varloop` (checkpoint key, GGUF `loop.varying`), `run_passes` above `num_passes` accepted only on a
varloop config, `run_visit` maps the last trained pass to the last run one (drafter taps) and moves the coda by the
passes added or skipped. GGUF synthesis no longer refuses `loop.varying` (llama.cpp only lowers its pass minimum to 1
for it); the recipe load of a checkpoint goes through that GGUF, so the varloop checkpoint failed to load before.

Tiny test `other_loop_pass_counts_run_and_keep_drafting_on_cuda`: 2 passes (fixed), 1 and 4 (varloop) score differently
from 3 and keep greedy output with the drafter; 4 on a fixed loop fails the load. Inkling tiny suite 26/26 CPU and CUDA.

## Run 2 - 2026-11-25 07:10

`varloop/serve.sh N` (scratchpad): `inference serve -m tml-d24-varloop --pa-cache-type q8 --pa-context-len 32768
--hf-overrides '{"loop_config": {"run_passes": N}}'`, no drafter; the first load wrote its recipe C (15.58 GiB, 88 s,
a recipe of its own since the key hashes config.json). `probe.py`: three greedy chat replies with thinking off
(`reasoning_effort: none`), 512 tokens at most; tok/s is wall time over completion tokens (prefill included).

| passes | sky (tok/s) | Rust fn (tok/s) | ms/token (sky) |
|---|---|---|---|
| 1 | 206.8 | 206.8 | 4.8 |
| 2 | 135.9 | 119.6 | 7.4 |
| 3 | 103.0 | 92.9 | 9.7 |
| 4 | 81.9 | 73.9 | 12.2 |
| 5 | 67.9 | 62.6 | 14.7 |

Every count reads coherently (Rayleigh explanations alike in substance, the trip answer right at all five), where
Inkling 2 Mini at 2 passes repeated itself into the length limit. Time per token is linear in passes: ~2.3 ms fixed
(prelude, coda, head) + ~2.5 ms a pass.

Raw oddity: at 2 to 5 passes the model refuses the Rust Fibonacci function ("I can't provide a ready-to-use
implementation...") and offers a conceptual explanation; at 1 pass it writes it (correct, with doc tests). Post-training
ran at 3 passes only, so its refusal behavior may not carry to 1 pass; the card warns the model is not tuned for
deployment. Not an engine fault: the same reply text at 3, 4 and 5 passes.

Next: quality per count needs a benchmark (MATH-500 L4-5 with `thinking_budget`, as in the loop guidance runs).

## Run 3 - 2026-11-25 07:25

Question: does a derived anti-refusal control vector undo the code refusals of Run 2? `varloop/comply.py SCALES
[LAYERS]` (scratchpad) sets a `derived` vector from 8 pairs: ordinary coding requests (Python, C, JS, Go, SQL, bash,
Rust, Python class) in the chat format (`<|message_user|><|content_text|>Q<|end_message|><|message_model|><|content_text|>`
then the reply), positive = the first line of a code answer, negative = the model's own refusal wording. Three held-out
requests (the Rust Fibonacci one, a Python sorted merge, a TypeScript groupBy), greedy, thinking off, 400 tokens.

3 passes:
- scale 0: Rust and Python refused, TypeScript answered.
- 1.5 (all layers): all three answered, fenced, correct, stopped on their own (244 / 151 / 325 tokens).
- 2.5: Rust and TypeScript clean; Python opens with "The user wants a Python function... Let me think" (reasoning
  leaking into the reply), then the code.
- 4: no refusals, but fences dropped, Python ran on past `return result` into a second copy, Fibonacci written inside
  its doc comment, TypeScript wrote a wrong version then "Wait, this is wrong".
- 8: degenerate (`def merge(a, b): return a + b` repeated; wrong Rust).
- 2 and 4 on layers 6-17 only: like 2.5 and worse (4 leaks terse reasoning: "We need answer user asks write...").
2 passes: scale 0 refused Rust and TypeScript; 1.5 answered all three.

So the refusals sit on a direction a few pairs find, and 1.5 is the clean scale; from 2.5 up the vector also pushes
toward reasoning text (the positive replies start straight into an answer, as thinking-mode text does).

Found: the derived vector cache key is the model name plus the texts, not the config overrides, so the 2-pass server
read the vector derived on 3 passes (one file, d829bf47ae297af4.gguf). It still worked, but a vector derived at one
pass count is reused at another, whose layer outputs differ.
Fixed in the same change: the key now includes the load's config overrides (`derived_vector_key`), tested by an overridden tiny Llama load deriving a file of its own (fails without the fix).

## Run 4 - 2026-11-25 07:45

Question: LoopCD at reduced depth (the paper's Section 4.3: half the loops plus guidance matched or beat the unguided
full depth on Huginn, Parcae and Looped-Qwen3, multiple-choice only) at 2 of 3 passes. Coherence screen first:
`varloop/screen.sh` loads each model at `run_passes` 2 (with `--loop-guidance-logits` for the logits rows) and runs
`probe.py W` (sets `loop_guidance` W, then the three Run 2 prompts, greedy, thinking off).

- varloop, Logits 0 / 0.5 and Hidden 0.25: coherent prose; trip "2 hours and 35 minutes" (right) in all; Rust request
  still refused (guided: one line, "I can't provide a function that returns the n-th Fibonacci number."). tok/s
  (prefill included) sky 105.8 at Logits 0, 111.9 at 0.5, 126.9 Hidden.
- Mini, Logits 0 / 0.5 and Hidden 0.25: trip "3 hours and 40 minutes" (wrong) in all three; Rust broken in all
  (`let mut b: u  = 0;`, `let mut result: u 0;`, `let mut b: u a = 1;`) and running to the 512-token limit; prose
  slips ("nitrogen and nitrogen molecules" plain, "The reason the sunlight is blue" at Logits 0.5).

Guidance does not rescue Inkling 2 Mini at 2 passes (its third pass is not optional the way the paper's later loops
were); the varloop checkpoint passes the screen, so it goes on to MATH-500 L4-5 (`varloop/math_sweep.sh`): 2 passes
plain / Logits 0.5 (one logits load) / Hidden 0.25, then 3 passes plain.

## Run 5 - 2026-11-25 08:40

`varloop/math_sweep.sh`: `lato/math_budget.py` (MATH-500 levels 4-5, integer answers, 156 problems, greedy, thinking on,
`thinking_budget` 6144, 4 at a time) on the varloop checkpoint at 2 passes, strengths switched live on one
`--loop-guidance-logits` load.

- 2 passes plain: 152/156 (0.974), mean 1636 tokens, 2 cut off; 19 min.
- 2 passes LoopCD-Logits w_max 0.5: 153/156 (0.981), mean 1548 tokens, 0 cut off; 3 fixed, 2 broken against plain.

Stopped there (Hidden 0.25 and 3 passes plain not run): plain 2 passes nearly saturates this set, with 4 problems left
to gain and per-problem flips (3 fixed, 2 broken) as large as any gap the arms could show, so it cannot separate 2 from
3 passes or guided from plain. A harder set (MMLU-Pro subset, AIME) would be needed to measure what the third pass or
the guidance buys on this checkpoint.

Conclusion: the varloop checkpoint runs at 1 to 5 passes and at 2 passes with either guidance form, coherent and at
152-153/156 on MATH L4-5 at 2 passes (136 tok/s plain, ~112 with LoopCD-Logits, against 103 at 3 passes); Inkling 2
Mini at 2 passes stays broken with or without guidance.

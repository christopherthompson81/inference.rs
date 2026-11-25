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

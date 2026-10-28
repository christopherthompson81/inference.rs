# Decision models investigation (#660)

Decision models answer typed questions (choice, score, noul) about a `state` in one forward pass, generating no
tokens. #660 was filed for `ModernBertDecisionModel` (laya, Julia-1); this investigation scopes the whole class.

## Run 1 - 2026-10-27 19:18

Question: what does llama.cpp (970b51ce7, the reference) support as decision models, through what API, and which
checkpoints exist?

Command: read `tools/server/server-decision.{h,cpp}`, `common/common.h` (`common_decision_type`),
`src/models/{modern-bert,vllm-decision2,clef}.cpp`, `gguf-py/gguf/constants.py` (`Keys.Decision`),
`conversion/bert.py`, and the `/v1/systemone` section of `tools/server/README.md`; Hub searches by type name.

Finding: one endpoint, `POST /v1/systemone`, TypeSafe-compatible (typesafe.ai API): a `state` (text, object, array, or
chat messages with images), `files` / `images`, and `questions` keyed by id, each `choice` (criteria: option ->
description), `score` (2-10 level descriptions) or `noul` (optional true / false descriptions). The answer per
question: `choice` + `probabilities` + `confidence`; `score` (expected level) + `legend` + `probabilities` +
`confidence`; `noul` (P(true)). Probabilities are scaled by temperatures stored in the model
(`<arch>.decision.temperature.<type>[.<n-options bucket>]`, plus `score_bias`). Prompts come from a `systemone` chat
template in the model; a question can be evaluated in several option orders ("variants") and averaged.

Thirteen types, by how they read the answer:
- label logits at the last prompt token, on an ordinary causal LLM: openjev, lev (noul on a rating scale), nimble (all
  questions in the prompt), pplx-decider (1-2 letter labels), lfm2-d1 (labels by question type), granite-switch (a
  decision aLoRA), h2o-lightning (labels after an answer prefill), vllm-d3 (non-causal attention);
- hidden-state dot products: kev (last token against one end token per option);
- decision heads reading the embeddings output: laya / Julia-1 (ModernBERT plus `head_layers` pre-norm blocks, one
  marker token per option, three token types for the question types), lfm2-d1-omni (laya-like, another layout),
  vllm-decision2 (its own `vllm-decision2` architecture, options marked in the batch), clef (its own architecture, all
  questions in one prompt, decided jointly).

Checkpoints with real use (Hub downloads): clef-flash (Cloudflare, ~77K GGUF + 44K), laya (convaiinnovations, 58K;
multilingual 16K), kev-4b / 0.8b (52K GGUF), vllm-d3 (36K), h2o-lightning-4b (21K), Bespoke-Nimble-9B (21K),
OpenJev (14K), Julia-1 (12K), pplx-decider (11K), vllm Decision-2.0 (3.5K), granite-switch 4.1 (3K). Sizes run from
0.8B to 27B.

Implication: a scope decision. The endpoint, request parsing, answer formatting and temperatures are shared by all
types; the label-logit family rides on decoders we already run; laya / Julia-1, lfm2-d1-omni, vllm-decision2 and clef
each need a model head; kev needs hidden-state outputs.

## Run 2 - 2026-10-27 19:43

Question: does the first `/v1/systemone` path (decision checkpoints detected by their files, configs merged, the
ModernBERT decision head, prompts as laya's `build_sequence` / Julia-1's `sequence` build them) answer as the models'
own code does?

Command: `laya_ref.py` / `julia_ref.py` (laya's `RLAgent.system_one`, Julia-1's `TransformerEngine` +
`predict_typed`, CPU F32) over `decision_requests.py` (three states, eight questions: choice with 2/3/5 options,
score with 4/5 levels, noul with and without criteria, an object state, a state past laya's 512 tokens); then
`inference serve --cpu --dtype f32` on each checkpoint and `compare.py` (answers) / `compare_logits.py` (laya's
unrounded logits at its temperatures).

```
laya vs its rounded answers:      worst 4.70e-05, every choice the same (laya rounds to 4 decimals)
laya vs its logits + temperatures: worst 8.00e-07
Julia-1 vs its answers:           worst 9.63e-01, two choices differ
```

Finding: laya is exact. Julia-1 is not, though it runs the same model code, so the difference is in what reaches it
(its option rendering, its budgets, or its mmBERT tokenizer's encoding), not the head.

Implication: debug Julia-1 from its token ids (ours against `julia.data.sequence`) before its scores.

## Run 3 - 2026-10-27 19:48

Question: is Run 2's Julia-1 mismatch in its rendering, budgets or tokenizer?

Command: Julia-1's server log, then `ss -ltnp` on the port; rerun `compare.py` against `julia_ref.json` once the port
was free.

```
Error: Failed to bind the HTTP server to 0.0.0.0:18431 ... Address already in use (os error 98)
LISTEN 0.0.0.0:18431 users:(("inference",pid=561208)) - the laya server of the logits run, still up
Julia-1 vs its answers (its own server this time): worst 5.47e-06, every choice the same
```

Finding: none of the three. Run 2's Julia-1 run never served Julia-1: a laya server left from the run before still
held the port, so laya answered Julia-1's questions. Served on its own, Julia-1 matches its own engine to F32
precision, with its description-only options, its 256-token head budget and its 8192-token context.

Implication: both phase-1 checkpoints are exact. Next: the C ABI, the Python / C# / Rust SDK, tiny-checkpoint tests
and the docs, with `/v1/decisions` as the route and `/v1/systemone` as its TypeSafe alias.

## Run 4 - 2026-10-27 19:59

Question: do laya's variant folders (`multilingual/`, `typed-decisions/`, no root `config.json`) load by their files,
and does the multilingual one (its own encoder and tokenizer) match laya's code?

Command: `inference serve --cpu --dtype f32` on each folder, one request to `/v1/decisions`; `laya_ref.py` on
`multilingual/` (laya's code from the repository root on the path) and `compare_logits.py`.

```
multilingual:    loads, route -> billing (0.99999), angry 0.059
typed-decisions: loads, route -> billing (0.843),   angry 0.683
multilingual vs laya's logits + temperatures: worst 1.42e-06
```

Finding: both load from their folders alone, through `/v1/decisions` (the descriptive route; `/v1/systemone` stays
TypeSafe's alias), and the multilingual variant is exact. `typed-decisions/` was only loaded, not compared.

Implication: phase 1 is complete for the ModernBERT decision models: laya (three variants) and Julia-1.

## Run 5 - 2026-10-27 20:20

Question: phase 2, the label-logit types: what does llama.cpp's runtime do for them, where does each type's prompt
come from, and what does our engine lack?

Command: read `tools/server/server-decision.cpp` (`init`, `render_options`, `render`, `fill_task`,
`get_temperature`, `get_decision_confidence`, `format_answer`), `conversion/qwen.py` (`OpenJevModel`), openjev's
`helper/shim.py`; our `engine_logits.rs`, `pipeline/step.rs`, `pipeline/inputs_processor.rs`.

Finding:
- Prompts: llama.cpp's converter writes each type's prompt as a `systemone` chat template (ported from the model
  repo's shim; openjev's is a Qwen chat turn listing `[A] key: description` options and ending in an empty `<think>`
  block), its temperatures (`<arch>.decision.temperature.<type>[.<bucket>]`, buckets `2`, `3_5`, `6_10`, `11`; lev
  `small`/`mid`/`large`), score biases and `<arch>.decision.type`. The runtime renders the template with the state,
  the question and its options (each with a `label` for the coded types), tokenizes with special tokens parsed, and
  reads the logits of one label token per option at the last prompt position.
- Labels: openjev A-Z then a-z, each a single token; lev / granite-switch / nimble / pplx-decider / vllm-d3 the codes
  A-Z then AA-ZZ that are single tokens; h2o-lightning the `<arch>.decision.labels` texts; lfm2-d1 per question type,
  groups of tokens per option. Noul's options are `[true, false]` for openjev, granite-switch, lfm2-d1 and
  h2o-lightning. lev shows a choice in two orders and averages the reversed one back; its noul is read as a rating
  scale (9 labels) and answered as the expected rating.
- Answers: softmax at the temperature, plus the score bias for its level count; the variants averaged. Confidence is
  per type: TypeSafe's published formulas (choice `(p_max - 1/k) / (1 - 1/k)`, score one minus the mean distance to
  the mode over a uniform's) for most, normalized entropy for vllm-d3 and vllm-decision2. llama.cpp also uses
  TypeSafe's for laya, where laya's own code (and phase 1) uses normalized entropy; phase 1 follows laya.
- Our engine: `engine_logits` returns every position's full logits (`return_raw_logits` widens the logits window to
  the whole prompt in the inputs processor); a decision needs one position's.

Implication: (1) a last-position raw-logits mode: the forward returns logits instead of sampling, the inputs
processor keeps its usual last-token window; (2) the label-logit runtime ported from llama.cpp, driven by a decision
profile the GGUF's metadata gives (the safetensors checkpoints get built-in profiles, from the converter's templates,
later); (3) parity against llama.cpp's `/v1/systemone` on the same GGUF, starting with the smallest checkpoints.

## Run 6 - 2026-10-27 20:43

Question: does the label path (GGUF decision profile, `systemone` template rendered with a llama.cpp-style
`tojson`, label logits read by a capture logits processor on a one-token pass) match llama.cpp's `/v1/systemone` on
the nimble GGUF (Qwen3.5 9B, Q4_K_M)?

Command: llama.cpp `build_cuda/bin/llama-server -m Bespoke-Nimble-9B-v3-Q4_K_M.gguf -ngl 99 -c 8192 [-np 1]`;
ours `target/debug/inference serve --format gguf -m <dir> -f Bespoke-Nimble-9B-v3-Q4_K_M.gguf` (CUDA, bf16);
`scratchpad/decision/server_ref.py` (saves a server's answers over the synthetic `decision_requests.py` set),
`compare.py`, and `logit_check.py` (renders one prompt with jinja2, tokenizes it on llama.cpp, then both servers'
next-token top logprobs on those ids).

Finding:
- Engine: no new raw-logits mode was needed. A `CustomLogitsProcessor` sees the last position's logits before
  sampling (no penalties in deterministic sampling), so a `max_len: 1` completion over the rendered token ids with a
  capture processor yields the label logits; the prefix cache shares the request's common prompt.
- Prompt: token counts per request equal llama.cpp's (944, 1270, 1838), and both servers agree on the top labels.
- llama.cpp is not deterministic with its default 4 slots: the same request gave noul 0.398 then 0.373. With `-np 1`
  it repeats exactly, and its decision answer equals the one its own `/completion` logits give.
- Against the single-slot reference: same choice on every question; probabilities differ by up to 5.7e-2 (route
  3e-5, kind 9e-5; the near-even ones most). Our logprobs come in multiples of 1/16: the logits leave lm_head in bf16,
  whose step at magnitude 8-16 is 0.0625, so a label gap carries up to ~0.06 of rounding, which is the size of the
  differences. `--dtype f32` to rule it out fails on this model (`fused_decode_recurrence_cuda only supports
  f16/bf16`).

Implication: the label path matches up to bf16 logit rounding; f32 logits (lm_head in f32) would tighten every
label-type answer and is its own change across the models' heads. Next: a second family with the other options
(noul true first, canonical legend, entropy confidence), then tests.

## Run 7 - 2026-10-27 20:50

Question: does a second label family (vllm-d3-nano, Qwen3.5 2B, Q8_0: canonical legend, entropy confidence) match?

Command: as Run 6 with `ggml-org/vllm-d3-nano-GGUF` `vllm-d3-nano-Q8_0.gguf`; llama.cpp needs `-b 4096 -ub 4096`
here (`input (810 tokens) is too large to process. increase the physical batch size`), then `compare.py` and
`logit_check.py` on the `kind` question.

Finding:
- Same choices and prompt token counts (326, 388, 1621), but probabilities off by up to 0.12 and confidences far
  apart (kind 0.978 vs 0.702), more than bf16 rounding on a Q8_0 model explains.
- On identical token ids the raw next-token logprobs differ (llama.cpp top label -0.04, ours -0.39), so the gap is
  the forward, not the decision path.
- The GGUF sets `qwen35.attention.causal = false`: llama.cpp's `vllm_d3.py` converter writes it ("the full attention
  layers see the whole prompt", the checkpoint's `attention_mode: noncausal_full_attention`) and llama.cpp reads it
  into `hparams.causal_attn`, which also forces the whole prompt into one ubatch (the batch-size error above). We
  run the full-attention layers causally.

Implication: vllm-d3 needs a non-causal prompt pass in Qwen3.5's full-attention layers, with no prefix cache and no
chunked prefill for such a model; that is a model-side change of its own. Until then a GGUF whose decision profile
needs non-causal attention loads without the profile (with a warning), and `/v1/decisions` on a model without one
fails with a clear message instead of answering wrong.

## Run 8 - 2026-10-27 21:01

Question: can pplx-decider be checked against llama.cpp, and does OpenJev-Flash carry a profile?

Command: dumped the decision metadata of `perplexity-ai/pplx-decider-flash-gguf-v1.2` (`decider-flash-v1.2-q8_0.gguf`)
and `openjev/OpenJev-Flash-9B-GGUF` (`OpenJev-Flash-9B-Q4_K_M.gguf`); read pplx-decider-flash-v1.2's
`decision_config.json` and grepped llama.cpp's conversion and runtime for `noncausal` / `attention_mode`.

Finding:
- Neither GGUF has `<arch>.decision.*` or a `systemone` template (their publishers converted them, not llama.cpp's
  decision converter), so neither server serves them as decision models. Both deleted.
- Perplexity's GGUF sets `qwen35.attention.causal = false`, and the checkpoint's `decision_config.json` says
  `attention_mode: noncausal_full_attention` (with `pooling: last`).
- llama.cpp writes `causal = false` only in `VllmD3Model` (a subclass of `PplxDeciderModel`); `PplxDeciderModel`
  itself does not, and nothing else reads `attention_mode`. So a pplx-decider GGUF from llama.cpp's converter would
  run causally there (read from the code and config, not run). A candidate upstream report.
- The tiny decision test also passed under `nimble` after moving it off `pplx-decider`.

Implication: pplx-decider and vllm-d3 both need non-causal full attention; their families now load without a
decision profile whatever the GGUF's causal flag says. Supported and checked: nimble (Run 6); openjev next on
`ggml-org/OpenJev-GGUF` Q4_K_M (19.4 GB); lev has no GGUF to check against yet.

## Run 9 - 2026-10-27 21:41

Question: does openjev (letter labels, noul listed true first, per-type temperatures) match llama.cpp?

Command: `ggml-org/OpenJev-GGUF` `OpenJev-Q4_K_M.gguf` (Qwen3.5 27B, causal, `temperature.choice/score = 0.85`,
`noul = 1.555`); llama.cpp `llama-server -ngl 99 -c 8192 -np 1` alone on the GPU for `server_ref.py`, then ours
`serve --format gguf` alone, then `compare.py`.

Finding:
- First attempt: the llama.cpp server was still running (20 GB), so our auto device map put layers 2-63 on the CPU
  (17-19 prompt tok/s). Rerun with the GPU free: all 64 layers on cuda[0].
- With the default sequence count our third request failed with `CUDA_ERROR_OUT_OF_MEMORY`: the plan reserved 48 MB of
  activations (`max_seq_len: 4096, max_batch_size: 1`) for a 19.4 GB model on the 24 GB card, while a request's
  questions prefill together. `--max-seqs 1` serves it. A planning gap of its own (the concurrent prefill a decision
  request makes is not in the activation reserve), not a decision-path bug.
- With `--max-seqs 1`: token counts equal (239, 298, 1550), same choice on every question, probabilities within
  5.5e-2 (route 2e-5, kind 4e-5, repro noul 0.0332 vs 0.0330, which confirms the true-first order and noul's
  temperature); the near-even answers differ most, as in Run 6 (bf16 logits).

Implication: nimble and openjev match llama.cpp up to bf16 logit rounding. Left for later: f32 logits, non-causal
full attention (pplx-decider, vllm-d3), the activation reserve for concurrent prefill, lev (no GGUF to test).

## Run 10 - 2026-10-27 22:13

Question: with Qwen3.5's full-attention layers reading the prompt bidirectionally, does vllm-d3-nano match llama.cpp?

Command: Qwen3.5 `is_causal` (from the GGUF's `attention.causal`, or forced for the pplx-decider / vllm-d3 families):
prompt steps of the full-attention layers get an all-zero mask and `causal: false` flash params, the model refuses a
prompt with cached or earlier-chunk positions, packed prefill is off, and the pipeline sets no prefix cache and a
whole-context prompt chunk. `serve --format gguf ... vllm-d3-nano-Q8_0.gguf` with default flags, then `compare.py`
against Run 7's single-slot `d3_ref.json`.

Finding:
- First try (with `--prefix-cache-n 0 --max-prefill-chunk-tokens 8192`): `packed prefill requires causal varlen
  attention support` on the batched questions; packed prefill now reports unsupported for such a model.
- Then, with default flags: same choice on every question, worst probability difference 8.3e-3 (Run 7, causal:
  0.124); route 1.7e-4, kind 3.3e-4, repro 2.7e-4.
- Tiny Qwen3.5 checkpoint (CPU f32 and paged CUDA bf16): with `is_causal: false` the first row's logits move when
  the last token changes, with the causal default they do not.

Implication: vllm-d3 is served; pplx-decider takes the same path (forced non-causal, since llama.cpp's converter
does not write the flag for it) but has no llama.cpp-converted GGUF to check against.

## Run 11 - 2026-10-27 22:26

Question: how much of the nimble gap (Run 6, worst 5.7e-2) does an f32 lm_head output close?

Command: `ModelForwardContext::lm_head` feeds a GGUF head an f32 input (its kernels sum in f32 and write the input's
dtype, so the logits come out f32); nimble `compare.py` against `nimble_ref1.json`, then `logit_check.py` on
`repro` and `next` with llama.cpp (`-np 1`) beside ours.

Finding:
- Our logprobs no longer come in 1/16 steps.
- Decision answers: angry 1.0e-2 -> 4.1e-3, urgency 3.1e-2 -> 1.2e-2, route 3e-5 -> 6e-6; repro (5.7e-2 -> 5.1e-2)
  and next (3.5e-2 -> 5.3e-2) stay. Worst 5.25e-2.
- Raw logits on llama.cpp's token ids: next's A-B gap 0.864 vs llama.cpp 0.878 (close), so its decision gap comes
  from how the request's questions run (prefix reuse and batching on both sides), not the forward. repro's gap 2.06
  vs 1.50: the forward itself differs there, from bf16 activations over 32 layers (llama.cpp keeps f32
  activations); f32 activations are not an option for Qwen3.5 on CUDA (Run 6).

Implication: Run 6's "bf16 logit rounding is the size of the differences" was only part of it. The f32 head removes
the rounding at no measurable cost; the rest is activation precision and batching, not the decision path.
- Cost: `inference bench` on Qwen3.5-0.8B Q8_0 (CUDA), decode 128 tokens at depth 4: 553.1 / 553.2 tok/s before,
  551.5 / 549.6 after (about 0.4%, on a model whose 248K-row head is a large share of each step).
- Test: a tiny llama GGUF with a Q8_0 head, loaded in bf16 on the CPU, returns logits with nonzero low 16 bits; it
  fails without the change (an F32 GGUF head loads as a plain tensor and is left alone).

## Run 12 - 2026-10-27 22:55

Question: why does openjev 27B run out of memory on a request's batched questions (Run 9), and what should reserve
for it?

Command: `scratchpad/reserve/measure.sh`: a fresh `serve --prefix-cache-n 0 --max-prefill-chunk-tokens N
--max-num-batched-tokens N` per N, GPU memory sampled every 20 ms during one N-token completion, peak minus the
loaded level; Qwen3.5 0.8B Q8_0, nimble 9B and openjev 27B Q4_K_M. Then openjev's three requests twice in a row.

Finding:
- Peak prefill memory grows linearly: 0.8B 66 / 130 / 258 / 514 MiB at 512 / 1024 / 2048 / 4096 tokens; 9B 130 /
  226 / 482 / 962; 27B 194 / 354 MiB at 512 / 1024, and 1280 fails. The card exposes ~24143 MiB, so after loading
  the 27B (23689 MiB in use) about 450 MiB is left, against a planned activation reserve of 48 MB: the Qwen3.5
  estimate counts eager attention scores, which flash attention never materializes, and nothing else.
- The prompt pre-flight before each step (trim the pool, split the batch, or refuse) sized only the attention
  workspace, so a batch of the request's questions passed it and then failed in the forward. Request 2 alone
  succeeds on a fresh server; after requests 0 and 1 the pool holds ~300 MiB of other-shaped blocks.
- Per token, the most any length took: 135 / 266 / 397 KB (the 27B's 512-token point carries a fixed ~34 MiB). Dense
  Qwen3.5 now reports `66 KiB + 19 B x intermediate_size` (135.7 / 301 / 398 KB), which covers each.
- The pre-flight adds that estimate times the step's prompt tokens for a batch of more than one sequence. Applied to a
  single sequence too, it refused request 2's lone 512-token chunk (204 MB estimated, ~194 MiB real): one prompt's
  chunk is admitted on its attention workspace as before.
- Result: the six requests all succeed, `inference_cuda_prompt_batch_reductions_total 4`, no rejections, no OOM;
  openjev parity unchanged (worst 4.6e-2). A single sequence with a chunk past the headroom (`--max-prefill-chunk-tokens
  1536` on the 27B) still fails in the forward, as before.

Implication: a request's questions no longer exhaust memory; the load-time reserve is left alone (raising it would
take KV room from tuned setups), and the per-token figure is reported for dense Qwen3.5 only.

## Run 13 - 2026-10-28 08:22

Question: does kev (`ggml-org/Kev-0.8B-GGUF`, Q8_0) answer as llama.cpp's `/v1/systemone` does, with its pointer head
loaded in place of Qwen3.5's LM head and read from the raw-logits rows?

Command: `llama-server -m Kev-0.8B-Q8_0.gguf -np 1 -ngl 99` and `server_ref.py kev_ref.json`; then `inference serve
--format gguf -m .../kev -f Kev-0.8B-Q8_0.gguf` and `compare.py kev_ref.json` over the same three requests.

Finding:
- The GGUF is Qwen3.5 with `embedding_length_out 512`, `cls.output.{weight (Q8_0), bias}`, tied embeddings and no
  `output.weight`; one temperature (2.351) for all three types. llama.cpp's readout: the `[q | k]` rows of the last
  prompt token and of each `<|box_end|>`, `q_last . k_option / sqrt(256)`, then the shared softmax.
- Ours: same prompt token counts (106 / 156 / 1475), same choice on all 8 questions, worst difference 1.33e-2 (a noul),
  the rest 4.9e-4 to 8.5e-3. A chat request to the model is refused (its output is the head, not vocabulary logits).

Implication: kev is supported; the remaining gap is of the size the label families show (bf16 activations).

## Run 14 - 2026-10-28 08:48

Question: does h2o-lightning (`h2oai/h2o-lightning-4b-gguf`, Q4_K_M) answer as llama.cpp does, with its texts written
the way its Python shim writes them?

Command: `llama-server -np 1 -ngl 99` with `server_ref_h2o.py` over the shared set plus one h2o request (no type and
no instructions, `yes`/`no` noul aliases, a one-level score, `<|im_end|>` and `</think>` in the state, floats such as
1e20 and 1e-7, non-ASCII object instructions, a trailing-space last option); then ours with `compare_h2o.py`. Also
`h2o_long.py`: one ~140 KB log state past the 32000-token budget, against both (llama.cpp with `-c 40000`).

Finding:
- The GGUF (h2oai's own) carries `decision.labels` (255, each with a leading space, after the `Answer:` prefill), per-type
  temperatures (0.75 / 0.65 / 0.8) and `image` 1.0, a `qwen2` pre-tokenizer on `qwen35`.
- Prompt token counts equal on all four requests (297 / 343 / 1595 / 635) and on the long one (64158); same choice on
  all 12 questions; worst difference 5.85e-2 (a score), most under 1e-2; the long state within 3e-3.

Implication: h2o-lightning is supported for text. Image input (its mmproj) stays open with the other families'.

## Run 15 - 2026-10-28 09:11

Question: does lfm2-d1 (`LiquidAI/d1-3B-GGUF`, Q8_0, LFM2) answer as llama.cpp does, with labels chosen per question
and each scored by the best logit of its spellings?

Command: `llama-server -np 1 -ngl 99` with `server_ref_d1.py` over the shared set plus two d1 requests (a null state;
an object state with floats, letter keys, underscore keys, 30 options, a described noul); ours with `compare_d1.py`.

Finding:
- The GGUF has no temperatures (all 1) and a template that writes a non-text state with `tojson(indent=2)`, which our
  `tojson` did not take; it now takes `indent`, `separators` and `ensure_ascii`, and writes floats as Python's `repr`,
  as llama.cpp's jinja does.
- Prompt token counts equal on all five requests (152 / 239 / 1498 / 27 / 576), same choice on all 13 questions, worst
  difference 1.10e-2 (a noul), most under 5e-3.
- A first tiny-model check failed on a test bug, not the code: the tiny GGUF's choice temperature is 2.0 and the
  expected value used 1.0.

Implication: lfm2-d1 is supported for text. Its noul needs single-token `yes` / `no` (as in llama.cpp).

# CLI as a downstream consumer of the engine surfaces

The `inference` binary (CLI subcommands, `serve`, the web UI) should use the same surface as any outside consumer:
`inference_api::Engine`, which the C ABI and the HTTP server build on. Where the CLI needs something the API lacks,
the API gets it first.

## Run 1 — 2026-09-30 16:00

**Question:** how much of the CLI goes through `inference_api::Engine`, and what would it take for the rest to?

**Commands:** `cargo llvm-lines --bin inference -p inference-cli`, grouped by defining crate and CLI module; per-file
counts of `inference_core::` / `inference_api::` paths; a read-only feature-by-feature comparison of `run` and
`bench` against the API.

**Raw finding, size:**
- Binary IR 1,145,695 lines: the CLI's own code 205k (interactive mode 49k, clap argument types about 110k, config
  22k, uqff 22k), tokio 105k, serde_json 79k, the HTTP stack (axum, h2, hyper) about 100k, inference-webui's
  generics 61k.
- The one binary hosts `serve` and the web UI; the user wants to keep it as one demo binary.

**Raw finding, surfaces:**
- Every command loads through `inference_api::Engine::load(EngineSpec)`.
- `run` (interactive.rs, 2,160 lines) and `bench` bypass the API for requests: `engine.state().get_sender()` and
  hand-written `NormalRequest` literals (six in interactive.rs, one in bench.rs), reading core `Response`s. Ctrl-C
  uses the process-global `TERMINATE_ALL_NEXT_STEP`; approvals use a synchronous `AgentToolApprovalHandler`.
- The API already covers streaming text and reasoning, usage and throughput, sampling, reasoning controls, web search,
  code execution and shell, sessions, agent permission, approvals (`AgenticToolApprovalRequired` events answered with
  `resolve_approval`), produced files, adapters, image generation and speech.
- Gaps: no model category, modalities or generation defaults on `ModelObject` (the CLI picks its mode from them);
  `CompletionRequest.prompt` takes only a string (bench sends exact token ids); agentic progress and approval events
  are untyped JSON; `BlockDenoisingProgress` is filtered out of `ChatStream`; cancelling a stream by dropping it
  loses the final chunk and usage; no local-file media policy (the server policy rejects local paths); no
  encoder-cache counts in `Usage`.
- Logic the CLI duplicates: quant resolution three times (serve.rs `apply_quant_resolution`, quantize.rs
  `resolve_gguf_source`, and `inference_selection::quant::resolve_model_quant`, which `EngineSpec::resolve_quants`
  runs at load); sandbox policy built by the CLI because `AgenticSpec` only takes a mode; `serve` assembling the web
  UI and MCP routers itself, with the MCP router calling `parse_request` on raw state and so bypassing `ChatEngine`'s
  permission defaults and approval broker; UQFF selection parallel to inference-selection's; doctor probes (nvcc,
  driver, xcode) outside `run_doctor`, so the API's and ABI's doctor lacks them.

**Implication:** the load path is already right. The work is (1) the API additions above, (2) `run` and `bench` on
`Engine`, removing roughly 600 to 750 lines from interactive.rs and the core request types from the CLI, and (3) the
duplicated logic moved to inference-selection, inference-server-core and the API spec.

## Run 2 — 2026-09-30 16:45

**Question:** the MCP router's `chat` tool called `parse_request` on raw engine state. Which of the API's chat
policy did MCP clients skip, and does routing it through `Engine::chat` restore it?

**Finding:** `call_chat_tool` passed no approval handler, no tool-dispatch URL and no skill store, and skipped
`ChatEngine::prepare_inner`: the spec's `agent_permission` default and its strictest-merge with the request's, the
`max_tool_rounds` default, the refusal of `ask` without a stream, LoRA adapter model resolution and the response
model override. An MCP client on a server started with `agent_permission: ask` could therefore run tools with no
approval.

**Change:** `create_mcp_router` takes `&inference_api::Engine` and the tool calls `Engine::chat`. The CLI's
`spawn_mcp_server` passes the engine.

**Test:** `tests/integration/mcp.rs` on the tiny checkpoint: a plain call returns text content, and under a spec with
`agent_permission: ask` the blocking MCP call is refused with JSON-RPC invalid params, as HTTP refuses it.

**Review follow-ups:** under `ask` the tool could never succeed (the refusal asks for `stream=true`, which a blocking
MCP call can't give), so an `ask` server now lists no `chat` tool; the MCP guide says the server's agent policy
applies and `stream` is ignored. The review also found `/v1/responses` (server handler and `Engine::response`) outside
`ChatEngine`: `prepare_response_inner` hardcodes `agent_permission: None`, which the agentic loop treats as `auto`,
so a server run with `deny` or `ask` still auto-runs tools through the Responses API. That is the next fix.

## Run 3 — 2026-09-30 17:30

**Question:** Run 2's review found `/v1/responses` outside `ChatEngine`. What did it skip, and what should `ask`
mean on an API with no approval events?

**Finding:** `parse_openresponses_request` built its internal `ChatCompletionRequest` with `agent_permission: None`
(the agentic loop treats that as `auto`), no tool-dispatch URL and no server `max_tool_rounds` default. Both the HTTP
handler and `Engine::responses` (so the C ABI and bindings) took this path.

**Change:** `ChatEngine::apply_agent_policy`, factored out of `prepare_inner`, fills the `max_tool_rounds` default and
merges the server's permission with the request's by the strictest. `prepare_response` takes `&ChatEngine`, applies
it, and passes the server's tool-dispatch URL and skill store.

**Dead end, then the design:** the first version refused `ask` on Responses outright, on the premise that the API has
no event to carry an approval. The review pointed out that this dropped `/v1/responses` entirely for an `ask`
server, plain text included, and that the Responses stream already carries engine extensions
(`agentic_tool_call_progress`, `file_produced`). Now Responses matches chat: under `ask` a streaming request installs
the approval broker's handler and notifier and streams `agentic_tool_approval_required` events (answered with
`resolve_approval`); a blocking or background request is refused with chat's `ASK_REQUIRES_STREAMING`.

**Test:** `chat_route::responses_apply_the_servers_ask_permission`: under an `ask` spec, a blocking `Engine::responses`
and `POST /v1/responses` refuse on `agent_permission`, and `Engine::responses_stream` completes. Not covered: that a
`deny` server's permission reaches the core request for Responses (it flows through the same `parse_request` as chat).

## Run 4 — 2026-09-30 18:30

**Question:** the first API additions `run` and `bench` need: model category, modalities and generation defaults
(the CLI picks its mode from core's `get_model_category`), and token-id completion prompts (bench sends exact counts).

**Change:** `ModelObject` gains `category`, `modalities` and `generation_defaults`, filled for loaded models and for
the `default` entry, which carried nothing before. They are protocol types mapped from core's: core's
`ModelCategory::Multimodal` holds an `Arc<dyn MultimodalPromptPrefixer>`, which has no place on the wire.
`CompletionRequest.prompt` is a string or an array of token ids, routed to core's `RequestMessage::CompletionTokens`;
`echo` and a `best_of` above 1 are refused with token ids, since that message doesn't carry them.

**Raw finding:** the token path shares everything after prompt rendering with the text path (suffix, logprobs,
n, stop, grammar, truncation, adapter, streaming), except echo, best_of and BOS insertion. `generation_defaults`
fill only temperature, top-k, top-p, min-p and repetition penalty at request time (`fill_model_defaults`);
`max_new_tokens` and `max_length` don't reach the engine, so the docs say exactly that.

**Re-aggregation check (the user asked whether slimming core over-divided anything):** the mirrored model-info types
are a wire/engine boundary, not over-division; the one true duplicate is `GenerationDefaults`, whose removal would
make inference-nn depend on protocol and lengthen the critical path. Two real re-aggregation candidates: chat
request assembly (every surface could call the public `parse_request` and skip `ChatEngine`, which caused both
policy bugs; `ChatEngine::prepare` should be the only entry), and model selection held four ways (clap `ModelType`,
TOML `ModelEntry`, `ModelSelected`, `EngineSpec`) with about 800 lines converting between them.

**Follow-up:** echo with token prompts (what log-likelihood evaluation uses) needs `CompletionTokens` to carry
`echo_prompt` and `best_of`; core already decodes token prompts for the echo text.

## Run 5 — 2026-09-30 19:15

**Question:** Run 1 listed agentic progress and approval events as untyped JSON on `ChatStreamEvent`, so a Rust
consumer (the CLI) would have to parse JSON the engine had just built from typed data.

**Change:** `ChatStreamEvent::AgenticToolCallProgress(AgenticToolProgress)` and
`::AgenticToolApprovalRequired(AgenticToolApproval)` carry the typed round, tool, phase data (images as images),
approval id and arguments; each has `to_json()` built on the existing serializers, which the SSE route and
`ChatStreamEvent::to_json` (the C ABI) call, so the wire payloads are unchanged. `engine_chat` re-exports the core
types they carry. The Responses stream's own items stay JSON; nothing in Rust consumes them.

**Deferred:** `BlockDenoisingProgress` on chat streams (it would add an event to the C ABI stream) and per-request
cancellation that keeps the final chunk and usage. Core has no per-request cancel, only the process-wide
`TERMINATE_ALL_NEXT_STEP`, so that is a scheduler change of its own.

## Run 6 — 2026-09-30 20:30

**Question:** per-request cancellation that still delivers the final chunk and usage, replacing the process-wide
`TERMINATE_ALL_NEXT_STEP` the CLI uses for Ctrl-C.

**Dead end:** the first version marked a canceled request's sequences `Done(Canceled)` in the schedulers'
per-step `cancel_closed_response_groups` hook. The new test got an internal error instead of a final chunk: that hook
runs before scheduling, and the scheduler frees finished sequences without a step, which is right for a closed
client but sends nothing. `TERMINATE_ALL` works because it marks sequences inside `schedule()`.

**Change:** `NormalRequest.cancellation: Option<RequestCancellation>` (a shared atomic flag) is copied onto each
sequence. The same per-step hook latches it into a plain bool on the engine thread, and the stop check returns
`Canceled` after the EOS, stop-token and length checks. The sequence therefore ends on its next sampled token through
the normal done path. `ChatEngine` attaches a token to every chat request; `ChatStream::cancel()` fires it.

**Review findings fixed:** the latch exists because the CUDA decode path asks whether a step finishes a sequence
twice (before syncing the lookahead tail and while finishing), and an atomic another thread flips could answer the
two differently. The agent loop now checks the token after each round: a tool call that finished on the canceled
token is dropped rather than approved and run, and the final response is marked `canceled`.

**Known limits, for follow-up:** a sequence still waiting or mid-prefill runs its prefill and one token before it
ends; ring/NCCL workers don't receive the cancel (the same as a closed client today); the Responses and Anthropic
streams and the C ABI's stream handles can't fire it yet (an `inference_stream_cancel` would need C# and Python
bindings). Diffusion and speech requests never reach the stop check.

**Test:** `chat_route::a_cancelled_chat_stream_ends_with_its_usage`: cancelling after the first chunk ends the stream
with `finish_reason: canceled`, usage, and fewer tokens than the cap (CPU and CUDA suites).

## Run 7 — 2026-09-30 21:30

**Question:** can `inference bench` run entirely on `inference_api::Engine` now that completions take token ids and
models report `max_model_len`?

**Change:** bench's requests are `CompletionRequest`s (token-id `prompt`, `max_tokens`, `top_k: 1`, `adapter`)
through `Engine::completion_stream`; the adapter check uses `Engine::lora_adapters` and the context check
`Engine::models`. The post-warmup `IntervalLogger::reset` is dropped: it only affected the periodic log line
spanning warmup, not the results table. bench now imports only `initialize_logging` from core.

**Raw finding (a limit, not a regression):** a new test on the tiny checkpoint first asserted one streamed chunk per
generated token and got 2 chunks for 4 tokens. The streaming path holds back tokens whose text is empty or an
incomplete UTF-8 sequence, which random weights produce often. Bench divides decode time by `max_tokens - 1`, so its
TPOT stays right, but TTFT runs to the first chunk that carries text; the old core-channel path read the same
stream. The test instead checks that the request runs to `max_tokens` (`usage.completion_tokens`, finish reason
`length`) and that a measurement completes.

**Review notes left for later:** internal errors now reach bench as the API's generic message, since the API hides
internals from callers (a local consumer could be given the source); the completion path serializes the whole
request for its log even when logging is off, which grows with prompt length but stays well under 0.1% of prefill.

## Run 8 — 2026-09-30 23:30

**Question:** can `inference run` (interactive and one-shot; text, vision, image and speech) drive only
`inference_api::Engine`?

**Change:** a new `run/chat.rs` does chat turns through `Engine::chat_stream`: requests built as
`ChatCompletionRequest`s, typed agentic events (tool panels, approvals answered with `Engine::resolve_approval`),
Ctrl-C through the stream's `RequestCancellation`, stats from `usage`. The model's category and generation defaults
come from `Engine::models`, `/adapter` from `Engine::lora_adapters`, and image and speech modes call
`image_generation` (b64_json, written locally) and `speech_generation`. The text and vision REPLs are one loop.
Two opt-ins on `ChatStream`: `with_denoising_progress()` (block-diffusion progress; HTTP skips it) and
`cancellation()`. `run::require_adapter` replaces three copies. interactive.rs went from 2,166 lines to about 1,300
across two files; the run path imports only protocol value types and logging from core.

**Review regressions found and fixed:** media decoding moved into the API's request preparation, so a bad or
unreachable file failed the turn, which ended the session and left its message and attachment in the history;
a late approval answer (after the broker's 300 s auto-deny) returned an error that did the same; http(s) media
passed through got the server's public-only policy, losing the private-network URLs the old CLI fetched; and
Ctrl-C during request preparation exited. Now every reference is loaded once in the CLI with the API's
`load_media_source` under `MediaSourcePolicy::Local`; a failed turn rolls back its message and attachments and
names the user's path rather than `media://N`; approval errors are reported and the stream continues; and a
Ctrl-C while preparing cancels the turn as soon as it streams. TTFT starts once the request is dispatched.

**Known differences:** the prefix-cache line reports the turn's reused prompt tokens (not cumulative hits), and
agentic turns don't report it because the agent loop's usage aggregate drops `prompt_tokens_details`; the encoder
cache line is gone (no API for it); session media is re-decoded by the engine on every turn, as any API client's
would be; an approval unanswered for 5 minutes is denied.

**Tests:** CLI unit tests for sampling, request building, media parts and rollback; a text turn and a two-turn image
conversation (the second turn resends the first image by its index) on the tiny checkpoint.

## Run 9 — 2026-10-01 00:30

**Question:** the MCP and Responses policy bugs (Runs 2 and 3) had one cause: `inference_api::engine_chat::parse_request`
was public, so any surface could assemble a chat request without `ChatEngine`'s agent policy. Can it be made
private, leaving `ChatEngine::prepare` (and `Engine::chat`/`chat_stream`) as the only way in?

**Finding:** outside inference-api only two things used it: server-core re-exported it (and its crate docs example
taught the raw path), and the Anthropic `count_tokens` handler parsed a request to tokenize it. That handler is engine
logic in the HTTP layer.

**Change:** `inference_api::anthropic::count_tokens` holds the counting; the server handler maps its `Result`.
`parse_request` and `ChatCompletionParseContext` are `pub(crate)`; server-core re-exports `ChatEngine` and
`PreparedChat` instead, and its docs example prepares its custom route through `ChatEngine::prepare`. The compiler
now enforces what the two reviews had to catch. The `embed-in-axum` guide, which also taught `parse_request`,
points at `ChatEngine::prepare`. Still public, below the parsing layer: `dispatch::send_request` with a hand-built core
`NormalRequest`; closing that would mean hiding core's request types.

**Test:** `chat_route::anthropic_count_tokens_counts_the_rendered_prompt` (a count on the tiny checkpoint, and an empty
message list rejected with an Anthropic error body).

## Run 10 — 2026-09-30 (time approximate)

**Question:** quant resolution exists three times (Run 1). Can the one the engine runs at load,
`inference_selection::quant`, cover everything the CLI's `apply_quant_resolution` (serve.rs) and `resolve_gguf_source`
(quantize.rs) do, so the CLI just passes `quant` through?

**Finding (design pass, read-only):** the selection resolver only took `Run.quant`. The CLI additionally handled
`--quant` on explicit text, multimodal and embedding models; `--format gguf --quant` (mixed repos allowed, no ISQ
fallback); projector auto-selection for an exact `-f`; a required projector for multimodal GGUF; carrying dynamic LoRA
and multimodal limits into the GGUF it picks (`run_as_gguf` dropped them); and a GGUF-input-only rule for `quantize`.
None of these could be said in a `ModelSelected`, so non-CLI clients got less. The CLI's projector rule has three
behaviors, which two bools can't hold: pick when the repo is a GGUF artifact repo (an exact file in a mixed repo gets
none), pick any sibling for the direct local file shorthand (even in a mixed directory), and require one for an
explicit multimodal model.

**Change (first of two PRs):** `quant` on `Plain`, `Lora`, `MultimodalPlain`, `Embedding` and `GGUF`;
`GGUF.quantized_filename` may be empty when `quant` picks it; `GGUF.mmproj_selection: MmprojSelection`
(`given` default, `artifact_repo`, `any`, `required`). `quant::resolve_model_source(model, token, force_cpu,
QuantPolicy::{Weights, GgufInput})` resolves all of them; a resolved spec has `quant: None` and `mmproj_selection:
given`, and `ModelSelected::needs_source_resolution` tells whether one is still pending. The loader refuses unresolved
specs. `Engine::load` resolves every spec that needs it; `isq` with `quant` is refused as before, now for any kind.
OpenAPI, Python types and the Python spec reference regenerated.

**Tests:** 13 resolver tests in selection ported from the CLI's cases (vision+audio projectors, LoRA runtime kept,
mixed-dir ISQ fallback, multimodal needs a projector, embedding refuses GGUF, GGUF quant in a mixed repo, no ISQ
fallback for GGUF, quant+filename conflict, each `mmproj_selection` mode, the GGUF-input policy, loader refusal);
the engine test covers a GGUF `quant` and its `isq` conflict; a Python test that a GGUF `quant` spec omits the filename.

**Review:** no correctness bugs; resolved specs never stay pending. It found cases the follow-up PR could not say:
`--xlora`/`--legacy-lora` with `--quant` (now `quant` on `XLora`, `LoraGGUF`, `XLoraGGUF`, with X-LoRA switching to
`XLoraGGUF` in a GGUF repo and needing its base `model_id`); a LoRA-enabled multimodal model needing its projector
(now `mmproj_selection` on `Lora`, applied when `quant` resolves to GGUF); and `--tok-model-id`, which the GGUF a
`quant` resolves to can't carry. The last stays in the CLI: `--tok-model-id` implies GGUF format, as `--mmproj` already
does, since it means nothing for safetensors. `--quant` with `--format ggml` also stays a CLI check. JSON `null` for
`quantized_filename`/`mmproj_selection` is refused (Python omits `None`, so only hand-written JSON sees it).

**Next:** the CLI emits `quant` and `mmproj_selection` and deletes both resolvers. Known behavior changes there:
`--format plain --quant` on a GGUF-only repo loads the GGUF instead of erroring; a TOML model with both `quant` and
`isq`/`from_uqff` is an error instead of `isq` being dropped with a warning.

## Run 11 — 2026-09-30 (time approximate)

**Question:** with Run 10's resolver in selection, can the CLI drop `apply_quant_resolution` and `resolve_gguf_source`
and just put `--quant` in the spec it hands `Engine::load`?

**Change:** the CLI's conversion writes `quant` into every spec kind and `mmproj_selection` into the GGUF it emits
(`any` for the direct `-f` shorthand, `required` for an explicit multimodal model, including LoRA-enabled ones,
`artifact_repo` otherwise). `normalize_quant_flags` keeps only the flag rules: `--quant` with `-f`, `--quant` with
`--format ggml`, and `--tok-model-id` with `--quant` meaning GGUF. `quantize` resolves its spec with
`QuantPolicy::GgufInput` before loading. `serve` no longer computes an id override (the engine keeps the requested id
after a UQFF swap), and the TOML path no longer resolves per model. serve.rs lost about 600 lines, quantize.rs about
150; config's spec building is sync.

**Behavior changes:** `--format plain --quant` on a GGUF-only repo loads the GGUF (the CLI can't tell an explicit
`plain` from none once it is a `ModelSelected`); a TOML model with `quant` and `isq` fails with the engine's
`drop isq` error instead of a warning; an explicit multimodal GGUF without a projector fails at load, not at argument
conversion.

**Review:** one regression, `--legacy-lora --quant` without `--format gguf`: no safetensors spec carries a legacy
LoRA, so conversion refused it before the engine could resolve. `--legacy-lora` with `--quant` now means GGUF, like
`--tok-model-id`. Also: `quantize` resolved after creating its output directory, leaving an empty one on failure (now
resolves first); serve and quantize shared their GGUF file and projector rules as copies (now one helper each); the
TOML spec builders were async with nothing to await. Accepted too: TOML `quant` with `from_uqff` is an error.

**Tests:** the CLI's 12 resolution tests (now covered in selection) became 5 conversion tests: `quant` left for the
engine, `--tok-model-id` meaning GGUF, projector selection per how the file was given, multimodal LoRA requiring a
projector, the flag conflicts, legacy LoRA meaning GGUF, and a multimodal dynamic-LoRA GGUF keeping its runtime. The 5 quantize tests now run the selection resolver with the GGUF-input policy.

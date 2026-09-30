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

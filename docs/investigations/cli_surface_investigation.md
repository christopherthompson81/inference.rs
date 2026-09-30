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

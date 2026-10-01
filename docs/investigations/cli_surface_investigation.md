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

## Run 12 — 2026-09-30 (time approximate)

**Question:** Run 1 found the doctor's toolchain probes (nvcc, NVIDIA driver, the driver's CUDA version, Xcode) in the
CLI's `doctor.rs`, so `/system/doctor` on the server returned a report without them. Can they live in
`inference_selection::run_doctor`?

**Change:** `DoctorReport.toolchain: ToolchainInfo { nvcc, nvidia_driver, driver_cuda, xcode }`, probed in
`run_doctor` for the backends the build has (CUDA tools only with `cuda`, Xcode only with `metal`). The review moved
it off `SystemInfo`: `system_info` is also `/v1/system/info` and the C ABI's `inference_system_info`, which should
not spawn processes on each call. The CUDA build/driver check now reads the probed version rather than running
`nvidia-smi` again, and the `.exe` fallbacks are gone (`Command` resolves `.exe` on Windows, and under WSL they could
pick up the Windows toolkit's `nvcc.exe`). The
CLI prints it and lost its process-spawning helpers (~90 lines). The CLI's Metal line printed `Xcode Xcode 16.2`,
since the probe's value already starts with the product name; it now prints it once.

**Tests:** `toolchain_probe_outputs_parse` (nvcc release line and its fallback, deduplicated driver versions, Xcode
version and build).

## Run 13 — 2026-09-30 (time approximate)

**Question:** Run 1 found the CLI building its own `SandboxPolicy` (profile, memory/CPU/process caps, network) and
writing it into each tool config, because `AgenticSpec` only took a mode. Can the spec carry the profile and limits?

**Change:** `AgenticSpec.sandbox_profile: Option<SandboxProfile>` (default `developer`) and
`AgenticSpec.sandbox_limits: SandboxLimits { max_memory_mb, max_cpu_secs, max_procs, network }`, beside the existing
`sandbox` mode, so specs that only give a mode are unchanged. The engine builds the default policy from all three for a
code-execution or shell config that has no `sandbox_policy`. The CLI passes its `--sandbox`, `--sandbox-profile`,
`--sb-*` and `--sandbox-network` values through and lost `extract_sandbox_settings` and `default_sandbox_profile`.
The CLI's default profile was `restricted` unless the agent, code execution or shell was on; since the policy only
reaches those tools' configs, the effective default was always `developer`, which is what the engine now applies.

**Tests:** `a_sandbox_profile_and_limits_shape_the_default_policy` (engine: restricted profile, limit overrides,
network override); the CLI's spec test checks the flags land in `AgenticSpec` and tool configs carry no policy. The
CLI's five `extract_sandbox_settings` tests went with it.

**Review:** no regressions; confirmed every path builds its spec through `agentic_spec`, nothing else attached a
policy, and the effective-default claim. Fixed docs that still described the conditional default (TOML reference,
sandbox reference) and the sandbox reference's caution box, which said the Python SDK is unsandboxed by default when
the engine spec (and so Python) is sandboxed; only the Rust SDK is.

## Run 14 — 2026-09-30 (time approximate)

**Question:** Run 1 found `serve` assembling the web UI (its router, observability layer and tool flags), binding the
listeners with TCP_NODELAY, spawning the MCP listener and logging the API surfaces itself. Can that live below the
CLI?

**Finding:** `inference-webui` depends on `inference-server-core`, so a `with_ui` option on server-core's router
builder would be a cycle. The split follows the dependency: server-core runs a router, the UI crate mounts itself.

**Change:** `inference_server_core::serve::serve(app, &engine, ServeOptions { host, port, mcp_port })` binds with
TCP_NODELAY, spawns the MCP listener and logs the API surfaces and routes. `inference_webui::mount(app, &engine,
UiOptions, ObservabilityConfig)` nests the UI at `UI_ROUTE` with the API's request logging and metrics;
`UiOptions::from_agentic` reads the tools from the `AgenticSpec` rather than the CLI re-deriving them from its flags
(so the UI now reports the search embedding model the engine uses, default included). `serve_engine` is 20 lines and
axum is only a dev-dependency of the CLI now. `UI_ROUTE` lives in server-core's route registry (its metrics treat
the UI as housekeeping) and the UI crate re-exports it.

**Review:** no regressions; the UI's tool flags equal master's in every build and mode. `--mcp-port` equal to `--port`
was reported only after the model loaded; the CLI now refuses it first, with the flag names.

**Tests:** `serve::tests::accepted_connections_enable_tcp_nodelay` moved to server-core;
`the_ui_mounts_beside_the_api_with_the_engines_tools` mounts the UI on the tiny checkpoint's engine with its chat
cache in a tempdir and reads the tools back from `/ui/api/capabilities`.

## Run 15 — 2026-09-30 (time approximate)

**Question:** "which model to load" is written three ways: the clap `ModelType` groups, the TOML `ModelEntry`, and
`inference_selection::ModelSelected` (plus the per-model fields of `ModelSpec`/`RuntimeSpec`). Is the conversion
between them worth re-aggregating? (Read-only design pass.)

**Finding:** about 1,300 CLI lines turn model flags into `ModelSelected` + runtime settings: `convert_to_model_selected`
(230) and `convert_text_model` (312) in serve.rs, quantize's own pair (230), `model_spec` and `build_model_specs`
(two copies of the per-model runtime split), and TOML's `to_model_type`. Verbatim duplicates: two `model_format_mut`,
`extract_quantization` vs `model_quantization_mut`, two cpu-consistency loops. Unused `Deserialize` derives on
`ModelSourceOptions`, `DeviceOptions`, `CacheOptions`. Divergences: `arch` silently dropped for Auto (becomes `Run`),
GGUF, multimodal and embedding; TOML skips clap's declarative checks (quant vs isq/from_uqff,
`tgt_non_granular_index` without X-LoRA, matformer slice without its config); `tune` skips `normalize_quant_flags`.

**Bug:** `DeviceOptions` derives `Default`, so its `max_seq_len`/`max_batch_size` default to 0, and the TOML path fills
unset device fields from that default. A `[[models]]` entry without a `[models.device]` table (every documented
example) hands automatic device mapping `max_seq_len = 0, max_batch_size = 0`; clap gives 4096 and 1.
`FormatOptions` has the same shape (`gqa` 0 vs clap's 1).

**Plan:** PR 1 fixes the defaults (with a from-config test), drops the dead derives and merges the duplicates; PR 2 has
TOML reuse the clap groups and adds the missing TOML checks; PR 3 moves the conversion into inference-selection as a
flat `ModelRequest -> ModelSelected` (serve/run/bench/tune/from-config), making dropped `arch` explicit; PR 4 does the
same for quantize; PR 5 (optional) moves the SDK's hand-built `ModelSelected`s. `ModelSelected` and the OpenAPI/Python
schema stay unchanged throughout.

**PR 1 (defaults and duplicates):** confirmed the bug with
`a_model_without_device_or_format_tables_gets_the_cli_defaults` (it saw `(0, 0)` before the fix). `DeviceOptions` and
`FormatOptions` now implement `Default` with clap's values. The unused `Deserialize` derives on `ModelSourceOptions`,
`DeviceOptions` and `CacheOptions` and their serde default functions are gone; the two `model_format_mut`s and the
quantization accessors are `ModelType::{format_mut, quantization}`; the cpu-consistency check is one
`config::models_cpu`; `tune` runs `normalize_quant_flags`; the image-size defaults use the `AutoDeviceMapParams`
constants.

## Run 16 — 2026-09-30 (time approximate)

**Question:** PR 2 of the Run 15 plan was to have TOML reuse the clap groups and gain clap's checks. Reading
`ModelEntry` again: flattening `ModelSourceOptions` saves about 15 lines, needs back the `Deserialize` derives PR 1
removed, and PR 3 rewrites that conversion anyway; `DeviceOptionsToml` must keep `cpu: Option<bool>` for the
consistency check. So PR 2 is only the checks. Of Run 15's list, quant with isq/from_uqff is already refused by the
engine for every client; three were refused nowhere outside clap.

**Change:** `AdapterOptions::validate` refuses `tgt_non_granular_index` without X-LoRA (CLI and TOML both run it);
core's `load_matformer_slice` refuses a slice name without its config file (it used to load the model unsliced, for
every client); the selection loader's `resolve_ordering` refuses an empty ordering path with no inline ordering
(it used to fail with "Could not load ordering file at "), so the TOML/API user sees which field is missing.

**Tests:** `config_rejects_an_xlora_index_without_xlora`, `a_matformer_slice_without_its_config_is_refused`, and
`inline_ordering_is_used_over_the_order_path` now asserts the new ordering error rather than any error.

## Run 17 — 2026-09-30 (time approximate)

**Question:** Run 15's PR 3 was a public `ModelRequest` in inference-selection. That would be a fourth public way to
describe a model (clap, TOML, `ModelSelected`, and it), while most of the duplication it removes is inside the CLI:
quantize's own 230-line copy of serve's conversion. The user chose the smaller change: quantize maps its arguments onto
the `ModelType` serve converts, and dropped `arch` stops being silent.

**Change:** quantize's `convert_to_model_selected`/`convert_gguf_source` became `as_model_type` (a field mapping) plus
serve's `convert_to_model_selected` and `ModelSelected::write_uqff_mut` for the UQFF output. An auto model with
`--arch` now uses the text loader it names instead of auto-detection, which ignored it. `--arch` on a multimodal,
embedding, diffusion, speech, GGUF or GGML model (where it can't apply) logs that it is ignored.

**Review:** no field differs from master's quantize conversion for any variant or format. The arch reroute means
a multimodal checkpoint given a (text) `-a` now gets the text loader rather than auto-detection; the docs already say
`--arch` forces the text loader, so the flag's help now says so too. Also fixed from the review: `tune --emit-config`
writes `arch`, so the config loads the loader that was tuned; quantize applies `normalize_quant_flags` like the other
commands (`--tok-model-id --quant` means GGUF).

**Tests:** quantize's 5 conversion tests pass unchanged through the shared path;
`an_explicit_arch_picks_the_text_loader_for_an_auto_model`; `an_emitted_config_keeps_the_tuned_architecture`
round-trips an emitted config through the TOML parser.

## Run 18 — 2026-09-30 (time approximate)

**Question:** chat streams could be cancelled with their final chunk and usage (Run 8); what about completion,
Anthropic and Responses streams, background responses, and C ABI streams?

**Finding:** completion, Anthropic and Responses requests carried no `RequestCancellation`, so their streams could only
be dropped (no final event, no usage). `cancel_response` on a background response only relabelled the task: its
request ran to the end ("the abandoned request finishing does not undo the cancellation"). The C ABI had no stream
cancel; `inference_stream_free` abandons.

**Change:** every prepared request (completion, Responses; Anthropic already went through `PreparedChat`) carries a
cancellation, and `CompletionStream`, `AnthropicStream` and `OpenResponsesStreamer` have `cancel()` like `ChatStream`.
A cancelled Responses stream ends with `response.cancelled` (status `cancelled`, usage); Anthropic ends with
`message_delta` (usage) and `message_stop`. A background task holds its request's cancellation: cancelling or
deleting it stops generation, and the task keeps the partial response the stopped request returns. `BlockingStream`
carries its cancellation, and `inference_stream_cancel` in the C ABI (C# `EngineStream.Cancel`, Python
`Stream.cancel`) may run on another thread while one waits in `inference_stream_next`: the handle keeps the
cancellation in its own field and each call borrows only the fields it uses.

**Tests:** `cancel::` in server-core's integration tests (completion, Anthropic, Responses stream, background
response); Python and C# tests cancel a chat stream from another thread and read its final chunk.

**Review:** the FFI field-disjoint borrows are sound, and the leases keep cancel from racing free. Fixed: the ABI
patch version (0.0.12) was not bumped for the new entry point; a cancelled stream stored its truncated reply as history,
so `previous_response_id` could continue it while a cancelled background run could not (now neither can; both are
fetchable); a cancelled response's message item said `in_progress` after `output_item.done` said `completed` (both are
now `incomplete`, and so is a cancelled background partial's). Tests use a 512-token cap so a loaded runner cannot
finish before the cancel lands.

**Found, not fixed:** a Responses run stopped by its token cap ends `completed`, never `incomplete` with
`max_output_tokens`, in both the streaming and non-streaming paths, though the HTTP reference says otherwise.

## Run 19 — 2026-09-30 (time approximate)

**Question:** Run 18 found Responses runs stopped by their token cap reporting `completed`. Fix it on both paths.

**Change:** `finished_status` maps a run's finish reasons to `cancelled` (any `canceled`), `incomplete` (any `length`,
now core's `FINISH_REASON_LENGTH`) or `completed`, for the stream's terminal event (`response.incomplete`) and the
non-streaming/background resource; an incomplete one gets `incomplete_details.reason = max_output_tokens` and its
message item is `incomplete`. It stays continuable with `previous_response_id`, unlike a cancelled one.

**Tests:** `a_capped_response_is_incomplete_and_can_be_continued`; the Responses stream tests (server-core, Python,
C#) cap the tiny model's output, so they now expect `response.incomplete` and check its details.

**Review:** tool-call rounds report `tool_calls` (core overrides the finish reason when it parses tool calls), so they
never read as capped. Missed and fixed: the C ABI Responses test still expected `completed` (full CI caught it); the
stream's `Response::Done` fallback always sent `response.completed` (now `terminal_event` names the event from the
status for both paths); with several choices every message item took the run's status (now each takes its own
choice's); the non-streaming path stored a cancelled run's history (now neither path does); a cancelled background
partial kept `incomplete_details`. Left as it was: a cap that lands mid-reasoning reports a completed reasoning item.

## Run 20 — 2026-09-30 (time approximate)

**Question:** `inference run` stopped printing encoder-cache hits when it moved to the engine API (Run 8): the counts
lived on the core `IntervalLogger`, which the API did not expose. Per request or per model?

**Finding:** the counters are per model and cumulative: about 20 multimodal models bump shared `AtomicUsize`s inside
their forward passes, with no notion of which sequence an image belongs to. Per-request `Usage` would touch every one
of them; the user chose exposing the model counters instead.

**Change:** `Engine::cache_stats()` (`CacheStats { data: [ModelCacheStats { model, prefix_cache_hits,
prefix_cache_sequences, encoder_cache: Option<EncoderCacheStats { hits, misses }> }] }`), served at
`GET /v1/models/cache_stats`, in the C ABI as `inference_models_cache_stats` (ABI 0.0.13), in C# as `CacheStats()`
and in Python as `cache_stats()`. The CLI reads the encoder totals before and after each turn and prints the
difference as before; concurrent requests on the same engine would blur it, which a single-user CLI does not have.

**Tests:** `a_second_image_turn_resends_the_first_image_by_its_index` now checks the first turn misses and the
second hits; `cache_stats_list_each_loaded_models_counters` (HTTP); Python and C# read the stats.

**Review:** counters, route and ABI are right. Fixed: the docs said "cumulative" without since when (the prefix counters
live on the engine's logger and restart with it; the encoder ones are the model's and restart on reload);
`prefix_cache_sequences` counts every prompt sequence started, not only those that consulted the cache; media already
covered by a reused prefix is neither an encoder hit nor a miss; the observability page implied the Prometheus
counters were these (they are one process-wide series); the field is `model_id` like `/v1/models/status`; entries are
sorted by model id; the C ABI integration test calls the new entry point.

## Run 21 — 2026-09-30 (time approximate)

**Question:** Run 19 left one inconsistency: when the token cap (or a cancel) lands while the model is still reasoning,
the response is `incomplete`/`cancelled` but its reasoning item says `completed`.

**Change:** the stream records how its reasoning item ended: `completed` when text or a tool call followed it, the
run's item status (`incomplete` unless the run completed) when the run stopped mid-reasoning. The non-streaming and
background path applies the same rule per choice: reasoning with a reply or tool call after it is `completed`,
reasoning alone takes its choice's status.

**Tests:** `reasoning_cut_off_by_the_cap_is_incomplete_but_reasoning_before_a_reply_is_not` (resource conversion;
the streamer needs an engine state, and the tiny checkpoint does not reason).

**Review:** the non-streaming half missed the common case: a tag-based reasoning model's content is `Some("")`, not
`None`, when it produced no reply (sampling.rs keeps reasoning state that way), so "a reply followed" was always true.
It now needs non-empty content; the test covers `Some("")`. Found, not fixed (agentic loop, separate issues): a
streaming agentic run whose last allowed round ends in a tool call sends no terminal chunk, so the stream ends with
"Response channel closed before completion"; and across rounds the streamer keeps one reasoning item, so a second
round's reasoning streams into an item already marked done.

## Run 22 — 2026-09-30 (time approximate)

**Question:** #157: since generated images default to the file store, nothing bounds the bytes it holds (only 4096
entries, about 10 GB of 2-3 MB base64 PNGs in the worst case).

**Change:** `FileStore` tracks the bytes its bodies take (`resident_bytes`: base64 or text as held) and, after an
insert, evicts expired entries and then the oldest until it is under `MAX_FILES` and `MAX_STORE_BYTES` (1 GiB); the
newest file always stays, so an over-cap upload is still fetchable once. Replacing an entry keeps its place and counts
only its new size. Storing binary bodies as raw bytes was not done: `FileContent::Binary { data_base64 }` is matched
across the protocol, agent and code-exec crates (a serde adapter could keep the wire shape, so the cost is internal),
and it saves only the 4/3 base64 overhead, which bounds nothing on its own. A shorter TTL for generated images would
cover one producer; the cap also bounds uploads, agent outputs and input files.

**Review:** accounting holds on every path (all `by_id` mutation goes through `Inner::remove` or `insert`; stored
files are `Arc` and never mutated, so no underflow). The cap is per loaded model (one store per `EngineInstance`),
now said in the docs; evictions are logged at debug. Known: a session import over 1 GiB evicts its own earlier files,
and evicted agent outputs or uploads then 404, as expired ones already do.

**Tests:** `the_byte_cap_evicts_the_oldest_and_keeps_the_newest`.

## Run 23 — 2026-09-30 (time approximate)

**Question:** #158: `GET /v1/files` and the C ABI's `inference_files_list` list every engine's files, and
`GET /v1/containers/{id}/files` ignores the container id, so any client of a shared server can enumerate and download
the others' files.

**Finding:** the server has no client identity (no API keys; `Authorization` is only allowed through CORS), so there is
nothing to scope by. File ids are random: uploads and generated images are v4 UUIDs, agent outputs carry a 48-bit random
run id. The web UI never lists files. So an id works as a capability, and the leak is the listing.

**Change:** `GET /v1/files` lists only on a server built `with_file_listing(true)` (`--allow-file-listing`,
`allow_file_listing` in TOML); otherwise 403 naming the option. A Responses run tags the files it produces with its
container id (`InferenceRs::try_tag_file`, the store's session tags), and the container listing returns only files
carrying that tag (`try_list_tagged_files`). The C ABI's `inference_files_list` stays a full listing for the host
application, documented as such.

**Tests:** `only_an_opted_in_server_lists_files_and_a_container_lists_its_own` (403 by default, listing when opted in,
container listing empty until a file carries its tag, another container's still empty).

**Review:** found the bigger remaining path: the web UI's chat history (`/ui/api/list_chats`) returns every saved chat
to anyone who can reach `/ui`, and a chat carries file ids and its `session_id`, whose export (`GET /v1/sessions/{id}`)
includes the files' bodies; saved sessions also restore their files after the TTL. "The web UI never lists files" was
true but beside the point. The UI stays on by default (the user wants it in the demo binary); `serve` now warns when
it is mounted on a non-loopback address and the docs say it is single-user. Also fixed: a container's file and content
routes served any file under any container id (now 404 unless the container cited it); two docs still advertised the
listing; the 403 names the builder option too and carries `file_listing_disabled`. Not fixed, left on #158: a request
without `session_id` can be matched to another client's session by message prefix (same prompt and greedy reply), so
the agent's `list_files` sees that session's files; session import accepts client-chosen file ids, which can replace a
known file's body; agent output ids share their run's prefix (`file_<run>_r<round>_<idx>`), so one id reveals its
siblings'.

## Run 24 — 2026-09-30 (time approximate)

**Question:** two agent-loop bugs from the Run 21 review. (1) A streaming agentic run whose last allowed round, or a
round whose tool has no handler, ends on a tool call sends no terminal chunk, so the stream ends with "Response channel
closed before completion". (2) The Responses streamer keeps one reasoning item for the whole run, so a later round's
reasoning streams into an item already marked done.

**Finding (1):** the loop holds tool-call chunks back from the client; a round's final chunk goes to
`tool_call_final_chunk`, and only `held_final_chunk` (text-only finals) was ever sent at the end. The non-streaming loop
returns the tool call with `finish_reason: tool_calls` in the same cases.

**Change (1):** `stopped_round_final_chunk` sends the held final chunk, or else the tool call the run stopped on, with
the run's usage and session, on both stop paths (round limit, no handler).

**Change (2):** reasoning arriving after the current item closed starts a new item (its own id and output index); the
earlier one is kept in `earlier_reasoning_items` and listed before it in the resource. `accumulated_reasoning` still
spans every round for the stored history and `reasoning`. The streamer grew past clippy's enum-variant size gap in
server-core's `OpenResponsesResponder`, so its SSE and JSON variants are boxed.

**Tests:** `a_round_stopped_on_its_tool_call_ends_the_stream_with_that_call` (unit). An end-to-end test was tried and
abandoned: a forced `tool_choice` on the tiny random checkpoint fails with "Tool choice was required but no tools were
called" within a few ms, with no chunks at all, even with empty-object arguments, so no tiny-model run reaches a tool
call. `each_agentic_round_streams_its_reasoning_into_its_own_item` feeds the streamer two rounds of synthetic chunks
on the tiny engine's state.

**Review:** fix (1) is right (core sends each tool call whole in the final chunk, so the terminal chunk carries the
full call; the cancel path sends at most one terminal). Fix (2) had a real bug: output indices were recomputed from
item counts, so a new round's reasoning item took the open message item's index and the message's later events moved
to another. The streamer now fixes each item's `output_index` when it is added (`claim_output_index`) and builds the
resource in that order, which also removes the old gap where function calls were indexed as if a message existed. A
tool-progress event now marks a round boundary, so a round that reasons, calls a tool and reasons again also gets two
items (before, only reply text between them split them). The test streams both shapes and checks every item keeps one
index matching its position in the resource. Left: stored Responses history has no assistant tool-call message, so a
streamed run ending on a client tool call can't be continued with its `function_call_output`; a run stopped at its
round limit hands back a call to a server-side tool the client cannot run (as the non-streaming reply always has).

## Run 25 — 2026-09-30 (time approximate)

**Question:** Run 24 left stored Responses history without the assistant's tool call: a run that hands a client tool
call back stores no assistant message (streaming, with no text) or one without `tool_calls` (non-streaming), so a
follow-up with `previous_response_id` and a `function_call_output` answers a call the history never made.

**Change:** the streamer records each tool call it returns (`returned_tool_calls`) and `finish` stores an assistant
message carrying them, with or without text; the non-streaming/background path stores the choice's `tool_calls` the
same way. Calls keep the name the model emitted, as `convert_input_items_to_messages` rebuilds an input
`function_call`.

**Tests:** `a_reply_that_returns_a_tool_call_is_stored_with_it` (a streamed and a collected run, each ending on a
client tool call, then the stored history's last message is the assistant's call with its id and arguments).

**Review:** no bug in the diff, but two follow-ups went wrong once calls were stored. (1) Clients that resend the
`function_call` item alongside its `function_call_output` (the usual pattern when not relying on the store) got the
call twice in the merged prompt; the merge now drops input calls whose `call_id` the stored history already holds.
Disabling that drop makes the new follow-up test fail with `[user, assistant(1), assistant(1), tool]` instead of
`[user, assistant(1), tool]`. (2) A round-limit stop hands back calls to server-side tools; storing those would leave
a call no client will answer, so only calls to tools the request defined (`function`, Responses `function`, or a
namespace entry by its qualified name) are stored. Also from review: a streamed run that fails or errors now stores
no history (non-streaming already didn't), and the streamer's fallback `Done` branch records its text, reasoning and
calls before `finish`. Tests split into `a_streamed_reply_is_stored_with_the_client_calls_it_returns` and
`a_collected_reply_is_stored_with_the_client_calls_it_returns` (each returns a client call and a server-tool call;
only the client's is stored), plus `a_follow_up_answers_the_stored_call_without_repeating_it` (an end-to-end
`prepare_response` with `previous_response_id`, the resent call and its output, checking the merged turn order).

## Run 26 — 2026-09-30 (time approximate)

**Question:** #158's remainder, first part. The user wants both a tokenless server and a keyed one; this run is the
hardening both share, so an id works as a capability only when it can't be guessed or derived. Keyed mode (owners
behind API keys) and the web UI/MCP under keys follow as separate PRs.

**Finding (sweep of id sources and shared state):** agent output ids were `file_<run>_r<round>_<idx>` with a 48-bit run
id shared by the run's files, so one id named its siblings; web UI chats were `chat_<n>` from a counter seeded off
disk; the UI's fork ids were client-chosen, falling back to `Math.random()` where `crypto.randomUUID` is missing (plain
HTTP off localhost); session import kept client-chosen file ids, and `FileStore::insert` on a known id swaps the body and
keeps its tags, so an import could replace another session's or container's file; Responses always set
`session_id: None`, so a `previous_response_id` follow-up found its agent session only by content matching. The
Anthropic route already passes its request's `session_id` through (the `None`s the sweep found there were tests).

**Change:** agent outputs get `file_<uuid>` each (`File::make_output_id`; `run_id` is gone); UI chats are
`chat_<uuid>`; `fork_session` names the fork server-side and returns it. The response cache stores a
`StoredConversation` (messages plus the run's `session_id`, from the stream's chunks or the collected reply, or the
session the request continued when the run reports none) and a follow-up sends that id. Open mode keeps content
matching for clients that pass no id (decided: one trust domain); the guide now says so and that a session id is a
secret.

**First import fix (rejected in review):** refuse a file id that is live and not tagged with the importing session.
Review broke it twice. Tags pile up (the agent loop re-inserts every input file under the request's session, Responses
adds `cntr_<response>`), so citing a known upload as an input file under session `atk` and then importing `atk` with
that id passed the check and swapped the body: a read capability became a write. And an expired entry the 120 s reaper
hadn't reached counted as absent, while `insert` carried its old tags onto the new body.

**Import as merged:** an import never replaces a stored body. Each file is retagged if live in any engine's store
(`FileStore::retag_live`, under the engines lock), else inserted; `insert` over an expired entry starts fresh (new seq,
no tags). Restoring your own session still works, since its stored files are its own bodies.

**Also from review:** a follow-up on an older reply used to branch (the longer session failed content matching, so it
got a fresh one); with the id sent, it would have spliced and then overwritten the session, losing the newer turns'
server-tool messages and sharing the sandbox between branches. The cache now tracks each session's head (the response
last stored with it, `ResponseCache::session_head`) and only a follow-up on the head sends the id. Not fixed and
pre-existing: `splice_session_into_request` replaces the request's images with the stored ones, so a follow-up adding
an image can lose it. The fork-failure path in the UI kept writing into the source session; it now drops the id. The
web UI's random chat ids protect nothing yet: `/ui/api/list_chats` lists every chat on a keyless server (keyed-mode PR).

**Tests:** `a_session_import_keeps_the_body_of_a_file_already_stored` (an untagged upload and one tagged by a citing
session both keep their body through an import; a session restores a new file of its own),
`a_body_stored_over_an_expired_entry_leaves_its_tags_behind`, `only_a_follow_up_on_the_latest_reply_continues_its_session`,
`a_reply_without_an_agent_run_keeps_the_session_it_continued`, the Responses streamed/collected tests now also check the
stored session id, and `output_ids_are_random_on_their_own`.

## Run 27 — 2026-09-30 (time approximate)

**Question:** #158's remainder, keyed mode. With API keys, can each key's owner reach only what it stored, through the
engine API first (ABI-first), with the HTTP server mapping keys to owners on top?

**Design (decided with the user):** named keys file (`name = key`; `--api-keys-file`, `[server] api_keys_file`) plus
`INFERENCE_RS_API_KEY` for one key owned by `default`; the owner is the name, so a key rotates without orphaning data.
Visibility is exact-match on the owner, with `None` (open mode, or an unscoped ABI handle) as an owner of its own: a
lookup that forgets to pass its owner finds nothing rather than everything.

**Change:** owners on every per-client store. `FileStore` entries (`get`, `list_all`, `list_for_session`, `remove`,
`attach_to_session` filter by owner; `insert` leaves another owner's live file alone, found while writing the store
test); `AgenticSessionStore` entries (`get`, `find_by_messages`, `list_ids`, `delete`, `export`, `import`, `fork`;
`held_by_other` refuses a session id another owner holds, so it can be neither read nor taken over); `NormalRequest.owner`, which the agent loop uses for
session lookup, input and output files and the `read_file`/`list_files` tools; the Responses cache and background tasks
(`Owned<T>` entries; get/delete/cancel/`previous_response_id` by owner); skills (persisted `owner`, serde default);
approvals (the broker records the requesting owner per approval; another owner's decision is `NotFound`). The engine
API carries it as `ChatEngine.owner`, set by `Engine::for_owner(name)`, a clone acting for that owner. The server's
`auth::authenticate` layer maps `Authorization: Bearer` or `x-api-key` to an `Owner` extension (SHA-256 digests compared
in full; `/health` and CORS preflights pass without a key; 401 `invalid_api_key` otherwise) and every per-client
handler reads it. A keyed server always lists `GET /v1/files` (each owner sees its own). The C ABI gets
`inference_engine_for_owner` (ABI 0.0.14); Python `Engine.for_owner`, C# `InferenceEngine.ForOwner`, each holding a
reference on its parent handle so the parent's host callbacks stay registered. The Rust SDK stays unscoped.

**Not covered yet:** the web UI and the MCP server sit outside the router build and act unscoped, so `serve` refuses
keys with either (`--no-ui`, no `--mcp-port`) until the next PR. The KV prefix cache stays shared (time to first token
can reveal a shared prefix), documented.

**Tests:** `keyed::*` (401 without or with an unknown key, `x-api-key` accepted, `/health` open; files listed, read and
deleted only by their owner; a session id held by one owner is 404 to read and 400 to import or chat under for
another; a stored response read and continued only by its owner), `a_keys_file_maps_each_key_to_its_owner`,
`a_keys_file_with_a_repeat_or_a_bad_line_is_refused`, `only_the_requesting_owner_answers_an_approval`,
`a_session_is_found_used_and_removed_only_by_its_owner`, `a_file_is_reached_only_by_its_owner`,
`a_response_is_reached_and_deleted_only_by_its_owner`, `a_conversation_is_continued_only_by_its_owner`,
`an_owner_handle_reaches_only_what_it_stored` (ABI), and the Python and C# binding tests.

**Review:** the first version had six holes. (1) The `/health` exemption was `path.ends_with("/health")`, so every
route ending in an id ran keyless as owner `None` when the id was `health` (`PUT /v1/sessions/health` could import
files under chosen ids, filling the shared byte cap and probing whether an id existed elsewhere); now only exact
`GET /health` and `GET /`. (2) Code-exec and shell sandboxes, the agent loop's remembered approvals, the broker's and
the Responses session heads were keyed by session id alone; `held_by_other` only guards while the session entry lives,
and it goes on idle expiry (30 min, under the sandbox's 60), on DELETE, on the 128-session cap any key can force, and
before the first round saves it. In those windows another owner sending the same id joined the live interpreter. All
of them now key by `sandbox_key(owner, session_id)` (a SHA-256 prefix in hex, carried on `ToolCallContext.owner`),
which also fixes a pre-existing path traversal: the shell work dir was `root.join(session_id)` with a client-chosen id.
(3) Auth sat inside `observe_http`, which buffers the body for its model label, so a keyless request could make the
server read up to the body limit; auth is now the outermost layer. (4) `count_tokens` ran unscoped. (5) A non-UTF-8
`INFERENCE_RS_API_KEY` was silently ignored, starting the server open; now an error (and an empty one is unset).
(6) A 401 on `/v1/messages` used the OpenAI envelope; the `Bearer` scheme is now case-insensitive. Documented rather
than changed: keys don't gate server-wide operations (any key may unload a model), and owners share the stores'
capacity caps, so one can evict another's oldest entries.

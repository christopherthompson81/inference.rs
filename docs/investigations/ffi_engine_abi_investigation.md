# Engine C ABI investigation (#19)

Goal: grow `crates/inference-ffi` from layout detection into the engine surface, so C#, Python (ctypes/cffi, replacing
`inference-pyo3`) and other languages share one native library. Loading-path steps 1-4
(`codebase_size_investigation.md`, Runs 12-17) are done: every SDK builder, the server and reload load through
`ModelLoaderConfig`, so the ABI can load through the same path instead of growing its own option model.

## Run 1 - 2026-09-27 (night)

Question: what does the ABI have to build on, and what shape should the engine surface take?

Survey (read-only):
- `inference-ffi` today: 13 functions (layout load/detect/detect_batch/results, versions, `inference_last_error`,
  `inference_status_string`), ABI 0.1.0. Conventions worth keeping: opaque handles only; every fallible call returns
  `inference_status` and sets a thread-local `inference_last_error`; panics are caught at the boundary (`guard`);
  inputs are copied during the call; outputs are borrowed from the handle that made them; freeing NULL is a no-op;
  `tests/export_surface.py` pins the exported symbol set; `tests/layout_abi.rs` tests at the ABI level.
- Loading: `ModelSelected` is already serde (JSON). The server's `InferenceForServerBuilder::new().with_model(..)
  .build()` loads through `ModelLoaderConfig` (Run 15) and carries the runtime options (paged attention, ISQ, max
  seqs, prefix cache, chat template, MTP).
- Requests: `inference-server-core` owns the OpenAI types (`ChatCompletionRequest`, ...) and the non-HTTP halves of
  the handlers: `chat_completion::parse_request(req, ctx) -> (core Request, is_streaming)`, `completions::`,
  `embeddings::`, `image_generation::`, `speech_generation::parse_request`, plus response collection
  (`process_non_streaming_response`, the streamers). The HTTP layer (axum) sits on top of these.
- pyo3 surface to reach parity with (from the issue): chat/completion, streaming, image/audio generation, embeddings,
  LoRA adapters, sessions, calibration, tool/search callbacks.

Design proposal:
1. **Wire format is the server's.** Requests and responses are the OpenAI JSON the HTTP API already accepts and emits,
   parsed and produced by the same `parse_request` / response code. No second schema, and HTTP parity by construction.
2. **Load spec is JSON over the server builder.** `inference_engine_load(spec_json)` takes
   `{ "model": <ModelSelected>, "runtime": { dtype, device, isq, paged_attn, max_num_seqs, prefix_cache_n,
   chat_template, ... } }` and builds through `InferenceForServerBuilder`, so load and reload are the Run 15-17 path.
3. **Handles:** `inference_engine` (one loaded engine, possibly several models; owns the tokio runtime handle) and
   `inference_string` (an owned response buffer: `inference_string_data/len/free`).
4. **Blocking calls first:** `inference_chat(engine, request_json, len, &out_string)`, and the same for completions,
   embeddings, image and speech generation. Request validation failures get a new status,
   `INFERENCE_ERR_INVALID_REQUEST`, with the OpenAI error body as `inference_last_error`.
5. **Streaming by polling:** `inference_chat_stream_open(engine, json, &out_stream)`,
   `inference_stream_next(stream, timeout_ms, &out_chunk, &out_done)`, `inference_stream_cancel`, `_free`. Polling
   keeps engine threads out of the caller's runtime (no GIL, no C# callback thread rules) and maps directly onto
   Python iterators and C# `IAsyncEnumerable`.
6. **Media by pointer:** a request handle that owns attached buffers, referenced from the JSON by index
   (`"media://0"`), so images/audio skip base64. Base64 data URLs keep working because the server parser handles them.
7. **Callbacks last:** tool/search callbacks as C function pointers plus `void* user_data`, called on an engine worker
   thread (documented; Python takes the GIL in its trampoline).
8. **Versioning:** ABI 0.2.0; minor bumps add entry points, as today.

Open decision: where the OpenAI layer lives. Depending on `inference-server-core` from the FFI is the fastest start but
links axum/tower/hyper into the C library. The cleaner option extracts the non-HTTP request/response layer into its
own crate that the server and the FFI both depend on. That is an architecture change on its own, so measure first:
the cdylib size and cold build time of `inference-ffi` with and without `inference-server-core`.

Proposed PR sequence:
1. Measure the server-core dependency (size, build time); extract the OpenAI layer if it is heavy.
2. Engine handle, JSON load spec, blocking chat/completion, `inference_string`; ABI test on the tiny random-weight
   PaddleOCR-VL checkpoint (`crates/inference/tests/paddleocr_vl_tiny.rs`), C header test, export-surface update.
3. Streaming poll API with cancel.
4. Media attachments by pointer.
5. Embeddings, image/speech generation, model list/unload/reload, LoRA adapter management.
6. Tool/search callbacks; sessions; calibration.
7. C# coverage in InferenceRs-Bindings; the ctypes Python package with a parity checklist generated from the pyo3
   surface; then retire `inference-pyo3`.

Also to measure on the step-2 prototype (the issue's open question): JSON (de)serialization cost per request and per
streamed chunk, against token time.

## Run 2 - 2026-09-27 (night)

Direction (owner): "The ABI call surface should be the top-tier of what our project delivers because project
integration is the primary goal. The web server should be reliant on that surface for implementing much of its
operations. That design should help prove out the ABI as transferable to any external project."

That settles Run 1's open decision, and goes further than extracting a shared layer: the server becomes a client of
the ABI's operations.

Revised architecture:
- **`inference-api` (new crate, pure Rust, no HTTP):** the engine surface. An `Engine` handle; `load(spec JSON)`;
  every engine operation as JSON-in/JSON-out (or a stream handle for streaming ones); media attachments; model and
  adapter management. The OpenAI request/response types and the non-HTTP halves of today's handlers move here.
- **`inference-ffi`:** a 1:1 `extern "C"` shim over `inference-api`: handles, `inference_string`, status codes and
  `inference_last_error`, the panic guard. No logic of its own.
- **`inference-server-core`:** HTTP only (routing, auth, SSE framing, multipart, CORS, metrics) over `inference-api`.
  A server feature that needs something the API lacks adds it to the API first.
- The Rust SDK can move onto `inference-api` later; pyo3 is retired once the ctypes package reaches parity.

Operation inventory, from the server's routes (the API's first target list):
- Generation: chat completions, Anthropic messages, completions, Responses (create, cancel), embeddings, image
  generation, speech generation. Streaming for chat, completions, messages and responses.
- Models: list, status, unload, reload, tune, re-ISQ, calibration start/apply.
- LoRA adapters: list, load, unload.
- Files and containers: list, upload, get, content, delete; container files list/get.
- Agents and skills: approval resolution, skills list/upload.
- System: info, doctor. (Health, metrics and root stay HTTP-only.)

Revised PR sequence, each a vertical slice so the architecture is proven from the first PR:
1. `inference-api` with `Engine`, the JSON load spec and chat completions (blocking and streaming). The server's chat
   route calls it, `inference-ffi` exports it (ABI 0.2), and ABI-level tests run it on the tiny random-weight
   PaddleOCR-VL checkpoint, from Rust and from the C header test.
2. Media attachments by pointer (images/audio for chat).
3. Completions, embeddings, Anthropic messages, Responses.
4. Model and adapter management; image and speech generation.
5. Files, skills, agent approvals, tool/search callbacks, system info.
6. C# coverage in InferenceRs-Bindings; the ctypes Python package with a parity checklist; retire `inference-pyo3`.

Measure on PR 1: JSON cost per request and per streamed chunk against token time, and the cdylib size.

## Run 3 - 2026-09-27 (night)

Owner suggestion: disentangle chat first. Doing that inside `inference-server-core` before any crate move keeps each
step reviewable and makes the later move a `git mv`.

Survey (planning agent, read-only): only three things in the chat path are really HTTP: the SSE `Event` framing, the
HTTP status mapping on `ApiError` (`from_status`, `from_json_rejection`, `status()`), and `SkillStore`'s multipart
parsing. `openai.rs`, the loader builder, media/video/input-files and `util` have no axum at all.

Change:
- New `engine_chat.rs` (no axum imports): the agentic tool-call helpers, `parse_request` and its context, and the
  non-streaming collector move there verbatim. New: `ChatEngine::prepare` (LoRA alias resolution, agentic defaults
  and permission merge, the `ask` check, approval handler/notifier, parse, dispatch), `collect_chat` (returns the
  final core `Response`), and `ChatStream` yielding `ChatStreamEvent` (the old streamer's mapping and logging, a
  `ResponseTap` for the access-log/latency observer, the model override).
- `chat_completion.rs` is HTTP only: the route calls `prepare`; `ChatCompletionStreamer` wraps `ChatStream` and adds
  SSE framing, `[DONE]`, `on_chunk`/`on_done` and keep-alive. Existing public helpers (`create_streamer`,
  `process_non_streaming_response`, `match_responses`, the `parse_request` re-export) keep their signatures.
- Anthropic `/v1/messages` uses `prepare` instead of its copy of the policy block. It now resolves LoRA aliases like
  the OpenAI route (and reports the alias as the model), as does `count_tokens`.
- The `ask` messages became transport-neutral.

Tests: no test drove `/v1/chat/completions` end to end before. `tests/chat_route.rs` runs the real router on the
tiny random-weight PaddleOCR-VL (generator moved to `crates/inference/tests/support/`): JSON and SSE decode the same
text, `[DONE]` comes last with no error chunk, and non-streaming `ask` is a 400. A review found the SSE sequence,
logging order, collector and error mapping identical to master.

Left for the move: `ApiError`, `JsonError` and the dispatch helpers still live in the axum-importing `handler_core`.

## Run 4 - 2026-09-27 (night)

Change: the last HTTP tie in the chat path's dependencies. `handler_core.rs` splits into `api_error.rs` (the error
kinds, `ApiError` and its classification of core errors, `JsonError`, `ModelErrorMessage`, the stable messages, and a
single OpenAI envelope, `to_openai_body()`, that the JSON responses and the SSE error events now share instead of
each mapping the `type` themselves) and `dispatch.rs` (response channels, sending to a model, model override).
`handler_core` keeps the HTTP views: `ApiErrorHttp { from_status, from_json_rejection, status }` for `ApiError`, and
the response builders. `engine_chat.rs` now depends only on HTTP-free modules. Tests split with the code; a new test
pins every error kind's HTTP status.

Result: green, 2129 CPU / 2447 CUDA tests (+1). Next: move the HTTP-free modules into `inference-api`.

## Run 5 - 2026-09-27 (night)

Change: the crate move.
- In place first, so each HTTP-free half had its own file: `agentic.rs` (`AgenticDefaults` and the approval broker,
  whose `resolve` becomes public for the ABI), `skill_store.rs` (the store takes `SkillFiles`, `(name, bytes)` pairs
  checked against the size/count limits as they are added; the server's multipart reader fills it),
  `lora_routing.rs` (model-name to LoRA adapter resolution), `sampling.rs` (stop tokens, DRY params). Tests that
  mixed store internals with HTTP checks were split along the same line.
- Then `git mv` into the new `crates/inference-api`: agentic, api_error, dispatch, engine_chat, input_files,
  lora_routing, media_source, openai, sampling, skill_store, util, video, inference_for_server_builder, and types
  (minus the axum `State` alias). `inference-api` depends on no HTTP crate; it compiled on its own first time.
- The server depends on it and re-exports those modules at their old paths, so `inference_server_core::openai`,
  `::inference_for_server_builder` etc. (used by inference-cli) keep working. Features forward to both crates.
- Review cleanup: the server publicly re-exports only `inference_for_server_builder`, `openai`, `util` and `video`
  (old public paths); the other moved modules are private imports. Plumbing-only items in `inference-api` are
  `#[doc(hidden)]`. Internal re-export shims were replaced by imports from the new modules. Unused server-core
  dependencies (candle-core, chrono, data-url, indexmap, itertools, zip, reqwest) dropped.

Result: green, 2132 CPU / 2450 CUDA tests (+3 from the split tests). `docs/openapi.json` changed only in a doc
example path (`inference_api::openai`).

## Run 6 - 2026-09-27 (night)

Change: the engine C ABI, first slice (ABI 0.2.0).
- `inference-api`: `Engine` (load from an `EngineSpec` JSON over the server builder, so loading is the Run 15-17
  path; `chat` / `chat_stream` and their JSON forms, with the HTTP route's error mapping and logging), and a
  `blocking` module: one process-wide multi-thread runtime, and work runs on its workers because engine start-up
  calls `block_in_place`. `ChatStreamEvent::to_json` is the stream envelope `{"event", "data"}`.
- `inference-ffi` (shim only): `inference_engine_load/free`, `inference_chat`, `inference_chat_stream_open`,
  `inference_stream_next(timeout_ms)` / `free`, `inference_string_data/len/free`; new statuses
  `INFERENCE_ERR_INVALID_REQUEST` (7) and `INFERENCE_ERR_UNAVAILABLE` (8), with the OpenAI error JSON as
  `inference_last_error`. The spec rejects `agent_permission: "ask"` until approvals are exposed.
- `ModelSelected::MultimodalPlain` / `Run` gain serde defaults for dtype and device-map sizes (as `Plain` has), so a
  spec can name just the model.

Tests: `tests/engine_abi.rs` loads the tiny random-weight PaddleOCR-VL through the ABI: blocking and streamed chat
decode the same text, a finished stream stays finished, malformed JSON and an unknown model are
`INVALID_REQUEST` with an OpenAI error body, bad specs are `INVALID_ARGUMENT`, a missing model is `LOAD_FAILED`, NULL
arguments are rejected. First run: the unknown-model assertion expected `param: "model"`, but that error comes from
request validation with `param: null`, the same body HTTP returns today; the test now checks the message.
`tests/header.rs`: the header declares exactly the `#[no_mangle]` exports, and compiles as strict C99 (the same
checks `tests/run.sh` does against a release build, now in every test run).

Measurement (the open question from Run 1): `ChatStreamEvent::to_json` plus a client-side `serde_json` parse costs
5.6 us per chunk (288 bytes) in release (`inference-api/tests/json_cost.rs`, ignored benchmark). Against decode
steps of milliseconds per token this is noise, so JSON stays the wire format.

Result: green, 2137 CPU / 2455 CUDA tests.

Review fixes: a request (not only the spec) asking `agent_permission: "ask"` is rejected on this surface, since a
caller could not answer the approval and the request would wait out the broker's 300 s; `max_seqs` defaults to 32 as
`inference serve` does (the builder's own default is 16); a device the build or machine lacks is
`INFERENCE_ERR_NOT_AVAILABLE`; `inference_stream_next` requires `out_done`; a compile-time assert pins the header's
thread-safety claims; the header documents the error JSON on RUNTIME, the up-to-10 s wait when the last handle is
freed, and what the surface does not offer yet.

## Run 7 - 2026-09-27 (night)

Change: media attachments by pointer.
- `media_source::MediaAttachments`: a request's buffers, named from the JSON as `media://<index>` wherever an image,
  audio or video URL goes. `MediaAttachments::load` resolves those and hands everything else to `load_media_source`,
  so URL sources keep their parsers, policies and limits. Attachments skip the URL byte cap (`MAX_MEDIA_BYTES`): the
  caller already holds the buffer in process, and the decode-side limits (GIF size, video frame sampling) still apply. `ChatCompletionParseContext` and `ChatEngine::prepare` carry the
  table; HTTP passes an empty one.
- `Engine::chat` / `chat_stream` (and their JSON and blocking forms) take the attachments.
- ABI: `inference_media { data, len, mime_type }`, `inference_chat_with_media`,
  `inference_chat_stream_open_with_media`; the existing calls are the zero-attachment case.

Versioning (owner): "all of the ABI versions should be pre-0.x at this point. We haven't stabilized it." The ABI moves
to 0.0.3 (after 0.1.0 for layout and 0.2.0 for the first engine slice): while it is 0.0.x every change bumps the
patch number and may break callers, which should require an exact match; the add-only minor rule starts at 0.1.0.

Tests: `attached_media_decodes_like_the_same_image_as_a_data_url` sends a fixture page as a base64 data URL and as an
attachment and compares the decoded tokens (identical); a reference past the attachments is `INVALID_REQUEST`, and
`media == NULL` with a nonzero count is `INVALID_ARGUMENT`. First run: the missing-attachment error surfaces as the
parser's context ("Failed to parse image resource: media://1"), as every media error does over HTTP; the detail
("names no attachment") is in the cause chain, so the test checks for the source.

Review fixes: the header names the chat fields (`image_url` / `audio_url` / `video_url`) and requires non-NULL `data`;
the version tests (Rust and C) assert the exact version, since 0.0.x callers must match exactly. The packed version
goes down (0x000200 to 0x000003), so a binding that checked "minor >= 2" now rejects the library, as intended.

## Run 8 - 2026-09-27 (night)

Change: completions and embeddings on the engine surface, the same split as chat.
- `inference-api`: `engine_completion.rs` (`parse_request` moved; `prepare_completion` does LoRA routing, parse and
  dispatch; `collect_completion`; `CompletionStream` of `CompletionStreamEvent` with the old SSE streamer's mapping
  and logging) and `engine_embeddings.rs` (the embeddings handler minus HTTP: `embed(state, request)`).
  `ChatDispatchError` becomes `DispatchError`, shared by chat and completions. The generic non-streaming collector
  moves to `dispatch.rs`. `Engine` gains `completion` / `completion_stream` / `embeddings` and their JSON forms;
  the blocking stream becomes `BlockingStream` over JSON events, so one `inference_stream` serves chat and
  completions.
- Server: the completions and embeddings routes only frame HTTP. `BaseStreamer` had no users left and is removed.
- ABI 0.0.4: `inference_completion`, `inference_completion_stream_open`, `inference_embeddings`, sharing one
  `json_call` helper with `inference_chat_with_media` (whose bad-media path already left `out_response` NULL; the test
  now pins that).

Tests: blocking and streamed completions decode the same text on the tiny checkpoint. An embeddings request to the
(chat) tiny model is `INVALID_REQUEST`. First run returned `INFERENCE_ERR_RUNTIME` there: `embed` converted its
`anyhow` errors into `Box<dyn Error>`, which hid the `ApiError` the engine had classified, so the ABI reported the
engine's validation error as internal (HTTP, which kept the `anyhow` error, said 400). `embed` now returns
`EmbeddingError` over `anyhow::Error`, and both callers classify from it. A positive embeddings test needs a tiny
embedding checkpoint; not built yet.

Result: green, 2139 CPU / 2457 CUDA tests.

## Run 9 - 2026-09-27 (night)

Change: Anthropic Messages on the engine surface.
- `inference-api/src/anthropic.rs` (git-moved from the server): the Anthropic types, their conversion to and from
  chat, and the stream state machine, which were already HTTP-free. New there: `prepare_messages` (validation,
  thinking visibility, conversion, the shared `ChatEngine::prepare`), `collect_messages`, `AnthropicStream` (the old
  streamer's response mapping, with a `ResponseTap`), and `anthropic_error_type` / `anthropic_error_body`.
- Server `anthropic.rs`: HTTP only. The SSE wrapper adds the `ping` events while the engine is quiet (keep-alive is
  transport), the responders keep Anthropic's statuses (including 529 for overloaded), and `count_tokens` stays in
  the server for now. Tests split: the five that check HTTP statuses/bodies stay; the conversion and stream tests move.
- ABI 0.0.5: `inference_anthropic_messages` and `inference_anthropic_messages_stream_open`. Their failures carry the
  Anthropic error envelope, since a caller using that protocol expects it; statuses are the same as the OpenAI calls.

Test: blocking and streamed Messages decode the same text on the tiny checkpoint, the stream runs `message_start` to
`message_stop`, and a request without `max_tokens` is `INVALID_REQUEST` with an Anthropic error body. Passed first
run.

Result: green, 2140 CPU / 2458 CUDA tests (the moved Anthropic tests all still run).

Review fixes: the Anthropic engine calls skipped `Engine::prepare`'s per-request `agent_permission: "ask"` rejection,
so a streamed Messages request could emit approval events the C caller cannot answer and wait out the broker's
timeout; the check is now one `reject_ask` shared by the chat and Anthropic paths, with a test. The Anthropic error
envelope is built once (`anthropic_error_body`) for the stream, the HTTP responder and the ABI. `PreparedMessages`
wraps `PreparedChat`. The server re-exports only the Anthropic types that were public before. Dropping the
`AnthropicError` serde default would have made `type` required in `docs/openapi.json` (the snapshot test caught it), so
the default stays.

## Run 10 - 2026-09-27 15:18

Change: the Responses API on the engine surface.
- `responses.rs`, `responses_types/`, `background_tasks.rs` and `cached_responses.rs` git-move from the server to
  `inference-api`. New there: `prepare_response`, `collect_response` (shared by blocking and background requests, which
  each had their own copy of the collect-and-store loop), `spawn_background`, and `get_response` /
  `delete_response` / `cancel_response`. `OpenResponsesStreamer` yields `ResponsesStreamItem` (an OpenResponses event,
  agentic progress or a produced file) with a `ResponseTap`; the server wraps it for SSE and `[DONE]`.
- Dead code dropped on the way: the streamer's `on_done` callback and event log (always `None`), `IncludeConfig`
  (computed, never read), the server's `observe_response`.
- The streamer stores the conversation when its terminal event is produced, not after the SSE body ends, so a
  follow-up with `previous_response_id` can start as soon as `response.completed` arrives (the ABI test does).
- A background request is dispatched before the queued resource is returned, so a dispatch failure is an error
  rather than a queued response that fails. A cancelled task now stays cancelled when its abandoned request finishes
  (before, `mark_completed` overwrote `Cancelled`). The request itself still runs to the end.
- ABI 0.0.6: `inference_responses_create`, `inference_responses_stream_open`, `inference_responses_get` / `_delete` /
  `_cancel`, and `INFERENCE_ERR_NOT_FOUND` (9) for `ApiErrorKind::NotFound`. The stream openers share a `stream_call`
  helper. Stored responses are process-wide (the server's globals), which the header says.

Test, first run: the ABI test failed where a missing `previous_response_id` came back
`{"type": "invalid_request_error", "param": null, "code": null}` with `INVALID_REQUEST`, not the typed
`previous_response_not_found`. Cause: converting an `anyhow::Error` into `Box<dyn Error>` boxes anyhow's `ErrorImpl`,
whose `source()` is the wrapped error's source, so `ApiError::from_error`'s chain walk skips the typed root. HTTP had
the same bug on every path through `DispatchError::Validation(error.into())`: chat, completions, Anthropic and
Responses. `api_error::boxed_anyhow` wraps the error so the root is its first source. Effect: an unknown model is now
404 `model_not_found` over HTTP (it was 400 with a null code) and `INFERENCE_ERR_NOT_FOUND` over the ABI; the chat
ABI test is updated and a route test pins the HTTP status.

Tests: ABI, blocking and streamed Responses decode the same text; the stream runs `response.created` to
`response.completed`; a stored response is fetched, cancelled (unchanged), deleted, then NOT_FOUND; a background
response goes from `queued` to `completed` with the same text; background plus stream is rejected. HTTP: the Responses
SSE names each event by its `type` and ends with `[DONE]`; a missing response is 404.

Local CI, first run: clippy `large_enum_variant` on `ResponsesStreamItem`; allowed, since nearly every item is an
`Event` and boxing would allocate per event.

Review fixes:
- A background response cancelled or deleted while it ran was still written to the response cache when it finished
  (the delete case resurrected it for GET and `previous_response_id`). `run_to_end` now returns what to store, and
  `spawn_background` saves it only if its task is still current. Cancel is one locked `Queued | InProgress ->
  Cancelled` transition; `request_cancel` / `mark_cancelled` / `cancel_requested` and the unused `list_tasks` /
  `cleanup_old_tasks` are gone. The cancelled request itself still runs to the end.
- Streamed responses stored their conversation but never their resource, so GET on a streamed id was 404. `finish`
  stores the terminal resource too.
- The Responses route stopped logging dispatch failures; `DispatchError::into_api_error` now does the logging for the
  engine and the route alike (it was `engine::dispatch_error`).
- `StreamOutcomeHandle::tap` replaces six copies of the tap closure; `OpenResponsesStreamer::next_item` had no caller.
- Header: unknown models are NOT_FOUND, not INVALID_REQUEST; lines reflowed to 120.
- Tests: GET on a streamed id; the follow-up's `input_tokens` exceed the first request's; HTTP 404
  `previous_response_not_found`.

## Run 11 - 2026-09-27 15:55

Change: model and LoRA adapter management on the engine surface.
- `inference-api/src/models.rs`: `list_models`, `unload_model`, `reload_model`, `model_status` and the request and
  status types, from the server's handlers. `lora_adapters.rs` git-moves from the server: the filesystem guard
  (adapter root, handle re-verification, one load at a time) and `load_adapter` / `unload_adapter` / `list_adapters`.
  The server routes frame HTTP only.
- One LoRA error mapping. `LoraAdapterApiError` (its own status, code and envelope) is gone; LoRA failures are
  `ApiError`s, with the lifecycle routes' specific codes (`lora_adapter_already_loaded`, `lora_rank_limit_exceeded`,
  ...) now also on inference paths, which used to collapse them to `lora_state_conflict` (the docs list the specific
  ones). New `ApiErrorKind::Forbidden` (403, Anthropic `permission_error`) for paths outside the adapter root. The
  lifecycle routes now answer with the OpenAI envelope (adds `param`; a 429 is `rate_limit_error`), and every 429 carries
  `Retry-After: 1`. The skills route's copy of the Anthropic error responder is replaced by the shared one.
- Engine spec `adapters: {runtime_updates, root}`; loading and unloading are `lora_updates_disabled` (Forbidden) without
  it, as the server hides those routes unless its env var enables them.
- `BlockingEngine` builds every call on one `call` helper.
- ABI 0.0.7: `inference_models_list`, `inference_model_{unload,reload,status}`,
  `inference_lora_adapters_list`, `inference_lora_adapter_{load,unload}`.

Test, first run: after unload, reload, the second (idempotent) reload failed `model_not_found`. Cause, in core:
`reload_model` never returned `ModelAlreadyLoaded` (it looked only in the unloaded map), so the "already loaded is
success" branch the server had was dead; and it marked the model reloading before that lookup, so the early return
left it marked, and `model_status` said "reloading" from then on. Now it checks the loaded map, then the unloaded
state, then marks with one `insert` (which also closes the check-then-mark race).

Tests: ABI unload/reload are idempotent, chat works after a reload, unknown models are NOT_FOUND, adapter loads are
refused outside the root and for missing paths, and without `runtime_updates`; HTTP adapter errors use the OpenAI
envelope. `docs/openapi.json` loses the LoRA-specific error schemas, like the other routes' errors.

Local CI, first run: `unpredictable_function_pointer_comparisons` in the new ABI test; it pairs each call with its
request now.

Review fixes:
- The reloading mark could still strand: it was cleared after `do_reload_model(...).await`, so a dropped caller (an
  HTTP client that disconnects mid-reload) or a loader panic left the model "reloading" for good. `reload_model` now
  marks first and clears through a `ReloadingMark` drop guard, and checks loaded and unloaded state after marking, so
  two reloads can no longer both pass the checks and load the model twice. Core test: a failed reload leaves no mark,
  and a refused one leaves the running reload's mark.
- `get_sender`'s auto-reload treats `ModelAlreadyLoaded` (another request won the race) as success; before, the now
  real error would have failed an ordinary inference request with 409.
- The adapter path is checked against the root before its metadata, and the not-a-directory message no longer echoes
  the canonical path, so a caller cannot probe what exists outside the root.
- `ListLoraAdaptersQuery` stays lenient about extra query parameters, as before. `from_status` maps 403 to Forbidden.
  One `json_response` helper in `handler_core`. Dead Unavailable/Overloaded masking in `from_lora_error` removed.
- Header: which adapter refusals are INVALID_REQUEST. Tests: Forbidden in both status tables; LoRA codes survive on
  the inference path.

## Run 12 - 2026-09-27 16:34

Change: image and speech generation on the engine surface.
- `inference-api/src/generation.rs`: `generate_image` and `generate_speech`, sharing one request builder (core's
  `NormalRequest::new_simple`, replacing two copies of the 30-field literal) and one collect step. Speech encoding (WAV,
  or s16le PCM) moves with it and returns `SpeechAudio { bytes, content_type }`. The routes frame HTTP only.
- Differences from the old routes: an unexpected engine response is an internal error, not an `unreachable!` panic;
  an unknown model is 404 `model_not_found` (the old path boxed the anyhow error and lost the type, like Run 10's fix).
- ABI 0.0.8: `inference_image_generation` (JSON out) and `inference_speech_generation`, which returns a new
  `inference_audio` handle (`_data`, `_len`, `_mime_type`, `_free`) rather than base64 in JSON.

Test: on the tiny (chat) checkpoint both calls reach the engine and come back INVALID_REQUEST "incompatible for this
model's category", with out-params NULL; `mp3` is `invalid_response_format` before dispatch; the audio accessors are
NULL-safe. Unit test: PCM is little-endian i16 and WAV has its RIFF/WAVE header. Not covered: a successful image or
speech generation, which needs a tiny FLUX or Dia checkpoint (FLUX needs T5, CLIP, a VAE and the transformer; not
built).

Local CI, first run: clippy `chunks_exact_to_as_chunks` in the PCM test.

Review fixes: the format check now runs before request logging and model validation (an mp3 request is refused
without being logged, and mp3 plus an unknown model reports the format, not the model); kept, since nothing unusable
should reach the log. An unexpected engine response is logged, not only mapped to internal. The PCM test asserts
literal samples rather than the encoder's own conversion. `engine_call` is the one FFI helper for engine, request and
owned handle out (`json_call`, `stream_call` and speech are built on it). Tests: an unknown image model is NOT_FOUND
`model_not_found`; speech on the chat model is refused as incompatible, not earlier.

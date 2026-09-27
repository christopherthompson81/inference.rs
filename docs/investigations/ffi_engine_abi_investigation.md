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

## Run 6 - 2026-09-28 (just after midnight)

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


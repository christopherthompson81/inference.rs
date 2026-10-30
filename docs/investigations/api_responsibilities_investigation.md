# What inference-api is responsible for (#701)

inference-api is the engine surface with no HTTP. The C ABI exposes what the inference engine does; an integrator's
own server handles server-level work over it (protocol translations, job queues, fetching media by URL), as a C#
server over llama.cpp's bindings did: it implemented Anthropic Messages and OpenAI Responses as translators onto its
chat-completions shape and fetched remote images itself. #701 moves that work out of inference-api and the C ABI.

## Run 1 - 2026-10-29 16:37

**Question:** Anthropic Messages out of inference-api: what does it take, and what does it change?

**Change:** `anthropic.rs` moves to a new crate, `inference-chat-protocols`, above inference-api. Its plumbing was core
internals (`self.state()`, core's response channel, `ChatEngine::prepare`); it now runs on the public engine:
`Engine::chat`, `Engine::chat_stream` (whose `ChatStreamEvent`s carry chunks, agentic progress and approvals, files
and errors) and `Engine::tokenize_chat` for count-tokens. The translation itself (request conversion, chat responses
back, the stream state machine) is unchanged. inference-api re-exports `FINISH_REASON_LENGTH` / `_CANCELED`. Gone:
`Engine::anthropic_messages*`, `Engine::count_tokens*`, their `BlockingEngine` twins, the C ABI's
`inference_anthropic_messages`, `_stream_open` and `inference_anthropic_count_tokens`, and the C# and Python
wrappers and tests. server-core's routes call the new crate.

**Finding:**
- CPU suite 3005 passed, binding tests pass, `docs/openapi.json` unchanged (the HTTP routes and schemas stay).
- IR lines: inference-api 1,384,072 -> 1,314,942 (-69,130; measured without #700's image fix), the new crate 124,220,
  server-core +2.6k. The new crate's own code is ~44k (with its utoipa schemas and decoders); the rest is std generics
  (core, alloc, hashbrown) each crate compiles its own copies of, since opt-level 3 dev builds share no generics.
- Whole-crate compile: inference-api 13 -> 12 s.
- A core edit that changes its metadata (a new `pub const`), then `cargo build -p inference-server-core -p
  inference-ffi --lib`: 15 s on master and on this branch. Neutral for the serial tail.

**Implication:** the move is about where the responsibility sits; the build gains, if any, come from moving Responses
(the larger module, with its store and background runs) and the job queue, measured the same way.

## Run 2 - 2026-10-29 17:43

**Question:** OpenAI Responses out of inference-api: it dispatched core requests itself (its own LoRA routing, agent
policy, approvals, logits processors and host tools) with the Responses tool surface. What does the public engine need,
and what does the move change?

**Change:**
- Two public engine pieces. `Engine::chat_events(request, media, tool_surface)` runs `ChatEngine::prepare` (which
  already does everything Responses did by hand) with the caller's tool surface and the request's own `stream` flag;
  a run that does not stream ends with a new `ChatStreamEvent::Done(response)`. `Engine::tag_file` tags a stored file
  into a container. `ChatStream::model_override` reports the model id its responses name, and
  `api_error::MODEL_ERROR_CODE` names a model failure. The C ABI lists both methods with their reason (server-side
  translators), and every `ChatStreamEvent` consumer handles `Done`.
- `responses.rs`, its store (`cached_responses`) and the background-task table move to `inference-chat-protocols`.
  A run's description (`ResponseRun`) separates from its event source, so the streamer and the non-streaming collector
  read any `ChatStreamEvent` stream: an engine run's, or a test's. Request parsing becomes a pure function returning
  the chat request, history and echo context. Entry points: `responses::{create, create_stream, get, delete, cancel}`.
- Gone: `Engine::responses*`, `response*`, `delete_response*`, `cancel_response*` and their `BlockingEngine` twins;
  the C ABI's `inference_responses_create`, `_stream_open`, `_get`, `_delete`, `_cancel`; their C# and Python wrappers
  and tests; the header's Responses wording.
- The Responses stream tests move with it and run on synthetic events, without a tiny engine.
- Found on the way: chat's `shell_skill_references` was `serde(skip)`, so a JSON caller reached uploaded skills only
  through Responses' shell tool. Skills are an engine capability, so chat requests now accept
  `shell_skill_references` (with `enable_shell`); the Python shell-skills example and the skills guide use chat.
  `docs/openapi.json` and the Python types gain that field and `OpenAiShellSkillReference`, nothing else.

**Finding:**
- CPU suite 3003 passed, binding tests pass, the new crate's 38 tests pass.
- IR lines: inference-api 1,173,510 (master with #700) -> 976,554 after both moves (-196,956, -16.8%; from the
  1,916,996 before the FromJson work, -49%). inference-chat-protocols 246,068, server-core 984,573.
- Whole-crate compile (`--crate-time`): inference-api 12 -> 10 s; server-core 10 -> 11 s.
- A core metadata change, then `cargo build -p inference-server-core -p inference-ffi --lib`, three runs each: master
  6 / 19 / 5 s, branch 6 / 23 / 6 s. Noisy and no measurable change: the downstream re-check and links dominate.

**Implication:** inference-api is about half the size it was this morning and compiles in 10 s. The job queue is
next; the rebuild after a core edit is now bounded by other crates, not api.

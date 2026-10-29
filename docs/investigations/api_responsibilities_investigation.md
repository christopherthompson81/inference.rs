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

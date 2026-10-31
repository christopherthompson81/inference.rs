---
title: OpenAI compatibility
description: Which OpenAI API fields inference implements, which it extends, and which it does not support.
---

inference.rs targets field-level OpenAI API compatibility. Most OpenAI client libraries work against inference.rs unchanged. This page lists the exceptions. For setup and examples, see [OpenAI-compatible APIs](/guides/serve/openai-compatible-apis/).

## Chat Completions fields

### Implemented

- `model`
- `messages` (including multimodal content parts)
- `max_tokens`
- `max_completion_tokens` (OpenAI's newer alias for `max_tokens`)
- `temperature`
- `top_p`
- `stream`
- `stop`
- `tools`, `tool_choice`
- `response_format` (`text`, `json_schema`)
- `logit_bias`
- `logprobs`, `top_logprobs`
- `presence_penalty`, `frequency_penalty`
- `n` (multiple completions)

Loaded dynamic LoRA aliases receive stable qualified model-card IDs from `GET /v1/models` and may be sent as `model`. Use the exact returned ID rather than constructing one; reserved characters are escaped and collisions receive a suffix. A unique short alias is also accepted for vLLM-style routing. The card's `parent` identifies the base model, `root` uses the public adapter alias rather than a local filesystem path, and `adapter_generation` identifies the current immutable generation. The explicit `adapter` extension below is available when callers need separate base-model routing or an exact generation.

### Implemented with deviation

- `tool_choice`: `"auto"`, `"none"`, `"required"`, Chat Completions specific function objects (`{"type":"function","function":{"name":"..."}}`), Responses-style specific function objects (`{"type":"function","name":"..."}`), and `{"type":"allowed_tools","mode":"auto"|"required","tools":[{"type":"function","name":"..."}]}` work for function tools. `"required"` rejects requests with no available tools.
- `tools[*].function.strict`: accepted on function tools. When `true`, inference.rs constrains generated tool arguments to the tool's `parameters` JSON Schema. See [tool calling](/guides/agents/tool-calling-basics/).
- `tools[*].type="code_interpreter"`: accepted as the OpenAI-compatible opt-in for the built-in Python executor. The server must be started with code execution enabled. The only supported container form is `{"type":"auto"}`. Container ids, `container.file_ids`, `container.memory_limit`, and OpenAI container lifecycle endpoints are not supported.
- `messages[].content[]` file parts: `{"type":"file","file":{"file_id":"file-..."}}` and `{"type":"file","file":{"filename":"data.csv","file_data":"data:text/csv;base64,..."}}` are supported. Chat Completions file URLs are not supported; upload the file first or use Responses.
- `response_format` with `json_schema`: supported; output shape may differ from OpenAI's on ambiguous schemas. `json_object` is not accepted. See [structured output](/guides/serve/structured-output/).
- `seed`: initializes a deterministic request-scoped sampling stream. Requests with multiple choices derive a distinct stream for each choice.

### Silently ignored

`user`, `stream_options`, `metadata`, `service_tier`, `parallel_tool_calls`, `store`. The request body accepts these fields (unknown fields are not rejected) but no behavior is wired to them. Use inference.rs `session_id` for persistence.

### Additional request controls

Accepted alongside the base Chat Completions fields. Several are inference.rs extensions:

- `top_k`: hard candidate cap.
- `min_p`: min-p sampling threshold.
- `repetition_penalty`: simpler alternative to frequency/presence.
- `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers`: DRY sampling parameters.
- `grammar`: llguidance constraints beyond JSON schemas.
- `reasoning_effort`: `off`, `low`, `medium`, `high`, or `xhigh`. `none` is accepted as an alias for `off`. Values are trimmed and case-insensitive.
- `enable_thinking`: legacy boolean toggle. If both controls are omitted, inference.rs enables thinking and leaves the effort unspecified. An explicit positive effort enables thinking; `off` disables it. Contradictory pairs such as `reasoning_effort: "off"` with `enable_thinking: true` return a validation error.

The selected effort is passed to the chat template as both `reasoning_effort` and the compatibility name `reasoning_strength`. The template determines how each tier affects the model; it may treat positive tiers alike or ignore controls it does not use. The Python SDK uses the same values and defaults.
- `web_search_options`: search tool configuration (de facto OpenAI field, not yet universal).
- `session_id`: multi-turn session persistence.
- `files`: required output files for server-side code execution.
- `agent_permission`, `code_execution_permission`: per-request permission tightening for server-executed tools.
- `max_tool_rounds`: cap server-side tool loop rounds for one request.
- `truncate_sequence`: truncate long prompts at the model's context limit instead of erroring.
- `adapter`: select a loaded dynamic LoRA alias string or an exact immutable generation object, `{"generation":"<generation-id>"}`. Omit it or use `null` for the base model. Unknown aliases, nonresident generations, and models without a dynamic LoRA runtime return an error.

`GET /v1/lora_adapters` is always registered and returns detailed alias, generation, and capacity state for models started with dynamic LoRA. A model without that runtime returns 409 with `lora_runtime_unavailable`. Adapter source is redacted unless runtime mutation is enabled. With `inference serve`, `POST /v1/load_lora_adapter` and `POST /v1/unload_lora_adapter` require `INFERENCE_RS_ALLOW_RUNTIME_LORA_UPDATING`; embedded servers configure mutation through `LoraAdapterApiConfig`. Read-only discovery does not require mutation access.

Chat Completions and Completions responses expose the exact resolved generation as `adapter_generation`, including streaming chunks. Completed Responses resources expose the same field. The field is omitted for base-model requests.

## Responses API

inference.rs implements the OpenAI Responses API alongside Chat Completions:

- `POST /v1/responses`: create a response. Returns a response object with a unique id.
- `GET /v1/responses/{id}`: fetch the current state of a stored response.
- `DELETE /v1/responses/{id}`: delete a stored response.
- `POST /v1/responses/{id}/cancel`: cancel a background response that has not finished. Generation stops, and the response then holds what was generated, with usage.

Use Responses when the client:

- expects OpenAI's Responses object shape,
- needs response ids,
- needs polling or background processing,
- needs cancellation.

Chat Completions, by contrast, returns the full response on a single connection. Codex speaks the Responses API; see [coding agents](/guides/serve/coding-agents/).

### Implemented

- `input`: messages or a raw prompt string.
- `input_file` content parts with `file_id`, `file_data`, or `file_url`.
- `previous_response_id`: continues a stored conversation.
- `max_output_tokens`: with `max_tokens` and `max_completion_tokens` as aliases.
- `instructions`, `temperature`, `top_p`, `stop`, `stream`, `tools`, `tool_choice`, `response_format`, `logit_bias`, `logprobs`, `top_logprobs`, `presence_penalty`, `frequency_penalty`, `n`, `metadata`, `background`, `store`.

`store` defaults to `true`; `store: false` skips caching, which makes the response unavailable to `GET /v1/responses/{id}` and `previous_response_id`. Function tools support `strict: true` with the same JSON-Schema-constrained argument generation as Chat Completions.

Responses `tools` accepts:

- Function tools in the Responses flat form: `{"type":"function","name":"...","parameters":{...}}`.
- Function tools in the Chat Completions nested form: `{"type":"function","function":{...}}`.
- `{"type":"web_search", ...}` and `{"type":"web_search_preview"}` for server-side web search.
- `{"type":"code_interpreter","container":{"type":"auto"}}` for server-side Python code execution.
- `{"type":"shell","environment":{"type":"container_auto","skills":[{"type":"skill_reference","skill_id":"skill_...","version":"latest"}]}}` for server-side shell execution and OpenAI-compatible Skills. Skills require the shell executor, so start the server with at least `--enable-shell`; `--agent` is the recommended preset when you also want the full agentic runtime.

`tool_choice: "required"` is accepted and rejects requests that provide no tools. Specific function choices must reference a declared function tool. `tool_choice: {"type":"allowed_tools", ...}` is supported for function tool subsets only. Hosted tool forcing or filtering, such as `{"type":"web_search_preview"}`, `{"type":"code_interpreter"}`, `{"type":"shell"}`, or hosted tools inside `allowed_tools`, is not supported.

### Skills API

- `POST /v1/skills`: upload one OpenAI-compatible Skill as multipart form data. Use `files` fields containing either a zip archive or the files from one top-level skill directory. The top-level directory must contain `SKILL.md` with `name` and `description` frontmatter.
- `GET /v1/skills`: list uploaded skills for the current server process and skills directory.
- `POST /v1/skills/{skill_id}/versions`: upload a new version for an existing skill.

Uploaded skill versions remain available from the server's skills directory (`--skills-dir`, or a system temporary directory by default). Referenced skills are made available to the shell session.

### Rejected non-default values

- `parallel_tool_calls` must be `true` (default) or omitted; `false` returns an error.
- `max_tool_calls` returns an error for any value. To cap tool rounds, use the server-level `--max-tool-rounds` flag (applies to both Chat Completions and Responses).
- The server's `--agent-permission` applies to Responses as to chat: `deny` blocks agent tools, and under `ask` a request must stream, with approvals arriving as `agentic_tool_approval_required` events; a blocking or background request is rejected on `agent_permission`.
- `tools[*].type="web_search"` rejects image search (`search_content_types: ["image"]` or `image_settings`) and `external_web_access: false`. Domain filters are supported for up to 100 allowed or blocked domains and include subdomains.
- `tools[*].type="web_search_preview"` rejects `filters` and `return_token_budget`; `external_web_access` is ignored.
- `tools[*].type="code_interpreter"` rejects container ids, `container.file_ids`, and `container.memory_limit`.
- `tools[*].type="shell"` rejects local environments, container references, local skill paths, inline/container-created skills, and OpenAI container lifecycle APIs. Uploaded `skill_reference` skills are supported.

### inference extensions on Responses

`top_k`, `min_p`, `repetition_penalty`, `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers`, `grammar`, `adapter`. The `adapter` field selects a loaded dynamic LoRA alias string or exact generation object; omit it or use `null` for the base model. A loaded alias can instead be sent as `model`. `max_tool_rounds` and `files` are accepted as on chat; the other chat agentic fields (`session_id`, `agent_permission`, `web_search_options`) are not part of this endpoint's schema. Use the Responses `tools` array for web search, code interpreter, shell, and OpenAI-compatible Skills.

Thinking, reasoning effort, and truncation are not top-level extension fields here; they are controlled through the standard Responses objects. Use `reasoning.effort` with `off`, `low`, `medium`, `high`, or `xhigh`; `none` is accepted as an alias for `off`. Omission enables thinking without choosing an effort. `reasoning.summary` is accepted for compatibility but currently does not change the response. Use `truncation` for sequence truncation. Top-level `enable_thinking`, `reasoning_effort`, and `truncate_sequence` keys are silently ignored on this endpoint.

### Background runs

```bash
curl http://localhost:1234/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": "Summarize today in tech news.",
    "background": true
  }'
```

Poll with `curl http://localhost:1234/v1/responses/<id>`, cancel with `curl -X POST http://localhost:1234/v1/responses/<id>/cancel`. Streaming event names are in the [HTTP API semantics page](/reference/http-api/#streaming-responses); full schemas in the [generated reference](/reference/http-api-generated/).

## Completions (legacy)

`/v1/completions` (non-chat) is supported with a subset of Chat Completions extensions: `top_k`, `min_p`, `repetition_penalty`, `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers`, `grammar`, `truncate_sequence`, `adapter`. LoRA accepts the same alias-as-`model`, adapter alias, and exact-generation forms as Chat Completions. The agentic, session, file, web-search, thinking, and reasoning-effort fields are not part of this endpoint's schema and have no effect. `prompt` is a string or an array of token ids, which the model reads as they are; `echo` and a `best_of` above 1 need a string prompt. Arrays of strings and batches of token arrays are not supported.

## Embeddings

- `input` accepts a string or a list of strings.
- `encoding_format`: `"float"` (default) or `"base64"`.
- `dimensions`: passing any value returns an error. Custom dimensions are not supported.
- `user`: accepted but not used.

Extensions:

- `truncate_sequence`: truncate long prompts at the model's context limit instead of erroring.

## Image Generation

- `prompt`
- `n` (1 to 10)
- `response_format`: `"url"` (default; `url` is the path that serves the PNG from the file store, `/v1/files/{id}/content`) or `"b64_json"` (response carries the base64-encoded PNG in `b64_json`).

OpenAI's `size` string (e.g. `"1024x1024"`) is not supported. Use the `height` and `width` fields instead:

- `height` (default 720)
- `width` (default 1280)

HiDream-O1-Image snaps `height` and `width` to its closest trained resolution.

inference.rs extensions:

- `steps` (1 to 1000): denoising steps, when not the model's own; HiDream-O1-Image-Dev takes only its 28.
- `guidance_scale`: FLUX.1-dev's distilled guidance or HiDream-O1-Image's classifier-free guidance; FLUX.1-schnell refuses it, and HiDream-O1-Image-Dev refuses one above 1.
- `seed`: seeds the noise, so the same request repeats its image on the same device and dtype; choice `i` of `n` takes `seed + i`.

`quality` and `style` are ignored.

## Image Edits

`POST /v1/images/edits` edits one image, or places several (subject references) in a new scene, with a model that takes
reference images (HiDream-O1-Image). It takes OpenAI's multipart form, `image` (or `image[]`, repeated) as the files, or a
JSON body whose `images` are `{"image_url": ...}` entries (`http(s)` or `data:` URLs).

- `prompt`, `n`, `response_format`, as for generation.
- `size`, in the form only: `<width>x<height>`, or `auto`. The JSON body takes `width` and `height`.
- With no size, one image keeps its own aspect at the model's trained area, and several take the first image's aspect.
- `steps`, `guidance_scale` and `seed`, as for generation.

`mask` is refused: the model edits from the instruction alone. `quality`, `background`, `input_fidelity` and
`output_format` are ignored.

## Audio

### `/v1/audio/speech` (TTS)

- `model`, `input`: supported.
- `response_format`: only `wav` and `pcm` are accepted; `mp3`, `opus`, `aac`, `flac` return a validation error.
- `voice`, `speed`: read by Kokoro (a voice it ships, comma-separated to blend; the speaking rate); Dia ignores them. `instructions`: ignored.
- `phonemes`, `seed`, `timestamps` (inference.rs extensions, Kokoro): a phoneme string spoken in place of `input`; a seed for repeatable audio; a JSON answer with the audio base64-encoded and each word's start, end and IPA. See [speech models guide](/guides/models/use-speech-models/).

### `/v1/audio/transcriptions` (speech recognition)

- `file`, `model`: supported; the file may be any format the audio decoder reads (WAV, MP3, FLAC, OGG, ...).
- `response_format`: `json`, `text`, `srt`, `vtt` and `verbose_json` are supported.
- `timestamp_granularities[]`: `word` and `segment`; `verbose_json` always carries segments and adds words when asked.
- `language`: conditions Nemotron-3.5 and Qwen3-ASR, which refuse one outside their lists; Parakeet accepts and ignores it.
- `prompt`, `temperature`: accepted and ignored.
- `beam_size` (an inference.rs extension, 1 to 16): beam search for RNN-T and TDT models; unset decodes greedily.
- `verbose_json`'s `language` is the one Nemotron-3.5 was given or identified, otherwise the request's, echoed. Its segments carry `id`, `start`, `end` and `text` only.

Served by transcription models such as Parakeet, and by audio-in decoders that transcribe (Qwen3-ASR), which give whole-clip segments and no words. See [speech models guide](/guides/models/use-speech-models/).

### `/v1/audio/translations`

Not exposed. Voxtral and similar audio-understanding models also go through `/v1/chat/completions` with audio content parts.

## Realtime (transcription sessions)

`GET /v1/realtime?intent=transcription` is a WebSocket speaking the Realtime API's transcription sessions, on a cache-aware streaming recogniser such as Nemotron streaming ASR. The official SDKs' `client.realtime.connect(extra_query={"intent": "transcription"})` connect as they do to OpenAI.

- **Connecting:**
  - The server sends `session.created`, or `transcription_session.created` for a client that announces the beta (an `OpenAI-Beta: realtime=v1` header or the `openai-beta.realtime-v1` subprotocol).
  - A browser may pass its key as the `openai-insecure-api-key.<key>` subprotocol. The key must then be a valid subprotocol token, with no `=`, `/` or spaces. Like a header key, it reaches anything that logs the handshake.
  - A browser signed in by cookie may open a socket only from the server's own origin.
- **Configuring:** `session.update` (`session.type: "transcription"`, `audio.input.{format, transcription, turn_detection}`) or the beta's `transcription_session.update` (`input_audio_format: "pcm16"`, `input_audio_transcription`, `turn_detection`).
  - Each is answered by `session.updated` or `transcription_session.updated`. An update is checked whole: a refused one changes nothing.
  - Audio is 16-bit mono PCM. `audio/pcm` takes its `rate`, 8 to 192 kHz; OpenAI documents only 24 kHz. `pcm16` is 24 kHz, and G.711 is refused.
  - A `transcription.model` the server does not serve (such as `gpt-4o-transcribe`) uses the default model; the server logs the substitution. `language` conditions Nemotron-3.5, and `prompt` is accepted and ignored.
- **Turns:**
  - `turn_detection: null` leaves commits to the client.
  - `server_vad`, the default as on OpenAI, is an energy detector here.
  - Speech starts on a 20 ms frame louder than a level set by `threshold` (0 is -60 dBFS, 1 is -20 dBFS, default 0.5). An item opens then, starting `prefix_padding_ms` (default 300) before.
  - The turn ends, and commits, after `silence_duration_ms` (default 500) without a loud frame.
  - A turn ending mid-append keeps its own audio; the next turn starts its own item. `semantic_vad` is refused.
- **Client events:** `input_audio_buffer.append` (base64 PCM16, non-empty), `.commit` and `.clear`. An empty commit is `input_audio_buffer_commit_empty`; any other client event is an `unknown_event` error. Conversation and response events are not served.
- **Server events:**
  - `conversation.item.added` (beta: `conversation.item.created`) when an item opens.
  - `conversation.item.input_audio_transcription.delta` as audio arrives, a model chunk (about a second) at a time. Unlike OpenAI, deltas come before the item's `input_audio_buffer.committed`, not after.
  - On commit, `input_audio_buffer.committed`, then `conversation.item.input_audio_transcription.completed` with `usage: {"type": "duration", "seconds"}`, or `.failed`.
  - With server VAD, `input_audio_buffer.speech_started` and `.speech_stopped`, with times in milliseconds of the connection's audio.
  - Each item is a fresh engine transcription session, so its first words are decoded without the item before as context.
- **Limits:**
  - a message is at most 16 MiB;
  - a connection silent for 10 minutes, or not reading its events for 30 s, is closed;
  - a model holds 64 sessions.

## Moderation

Not supported. inference.rs has no built-in moderation model; run one as a separate service if needed.

## Files and Assistants APIs

`POST /v1/files` multipart uploads are supported for user-provided input files. Use `purpose="user_data"` for OpenAI-compatible request attachments. Uploaded files, inline request files, URL-fetched request files, agent-produced files, and generated images (`response_format: "url"`) are available by id through `GET /v1/files/{id}`, `GET /v1/files/{id}/content`, and `DELETE /v1/files/{id}`. `GET /v1/files` lists the store only on a server started with `--allow-file-listing`, since the store is shared by every client; otherwise it answers 403.

Text-like UTF-8 files are exposed to the model as bounded decoded previews, with additional text available during agentic runs when file access is active. Binary files are stored, downloadable, and mounted into shell/code workdirs when those tools are active, but inference.rs does not perform OpenAI's private PDF/image/spreadsheet extraction pipeline. The Assistants API is not supported; the inference.rs equivalent is the session-based agentic loop on the chat completions endpoint.

## Fine-tuning and Batch

Not supported. inference.rs is an inference engine, not a training platform.

## Tokenization

inference.rs does not expose `/v1/tokenize` or `/v1/detokenize` HTTP endpoints. Tokenizer access is available through the SDKs (`engine.tokenize` / `engine.detokenize` in Python, `inference_tokenize` / `inference_detokenize` in the C ABI; `tokenize_with_model` / `detokenize_with_model` in Rust).

## Authentication

OpenAI requires an `Authorization: Bearer ...` header. inference.rs does not validate it. Clients that require an API key for initialization can send any non-empty string. For real authentication, place an authenticating reverse proxy in front.

## Response headers

`Content-Type: application/json` for non-streaming responses; `text/event-stream` for streaming. The session id (when assigned or matched) is in the response body's `session_id` field.

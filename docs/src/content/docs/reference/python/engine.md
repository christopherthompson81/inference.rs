---
title: Engine
description: "Load a model and serve requests; streams, results, errors and host callbacks."
sidebar:
  order: 2
---
## `Blob`

Bytes the engine returned with their MIME type: generated speech, or a file's content.

| Field | Type |
| --- | --- |
| `data` | `bytes` |
| `mime_type` | `str` |


## `Engine`

A loaded model serving typed requests; each takes a class from `inference_rs.types` or its JSON string.

Responses come back as those classes, with anything the schema leaves open as parsed JSON. Failures raise
InferenceError with the protocol's error JSON. Calls block and release the GIL, so several threads may share an
engine. Close it, or use `with`. `json` serves the same operations as JSON strings.

### `Engine.__init__`

```text
__init__(
    spec: types.EngineSpec | dict | str,
    callbacks: HostCallbacks | None = None,
)
```

`spec` is what to load and how to run it: an EngineSpec, or its JSON as a dict or string.

### `Engine.close`

```text
close()
```

### `Engine.for_owner`

```text
for_owner(owner: str) -> 'Engine'
```

The same engine acting for `owner`; see `JsonEngine.for_owner`.

### `Engine.__enter__`

```text
__enter__()
```

### `Engine.__exit__`

```text
__exit__()
```

### `Engine.chat`

```text
chat(
    request: types.ChatCompletionRequest | str,
    media: Sequence[MediaAttachment] = (),
) -> types.ChatCompletionResponse
```

### `Engine.chat_stream`

```text
chat_stream(
    request: types.ChatCompletionRequest | str,
    media: Sequence[MediaAttachment] = (),
) -> Stream
```

Events named `chunk` carry a ChatCompletionChunkResponse.

### `Engine.completion`

```text
completion(
    request: types.CompletionRequest | str,
) -> types.CompletionResponse
```

### `Engine.completion_stream`

```text
completion_stream(request: types.CompletionRequest | str) -> Stream
```

Events named `chunk` carry a CompletionChunkResponse.

### `Engine.embeddings`

```text
embeddings(
    request: types.EmbeddingRequest | str,
) -> types.EmbeddingResponse
```

### `Engine.anthropic_messages`

```text
anthropic_messages(
    request: types.AnthropicMessagesRequest | str,
) -> types.AnthropicMessageResponse
```

### `Engine.anthropic_count_tokens`

```text
anthropic_count_tokens(
    request: types.AnthropicMessagesRequest | str,
) -> types.AnthropicCountTokensResponse
```

### `Engine.anthropic_messages_stream`

```text
anthropic_messages_stream(
    request: types.AnthropicMessagesRequest | str,
) -> Stream
```

Anthropic stream events, as parsed JSON.

### `Engine.create_response`

```text
create_response(
    request: types.OpenResponsesCreateRequest | str,
) -> types.ResponseResource
```

### `Engine.response_stream`

```text
response_stream(
    request: types.OpenResponsesCreateRequest | str,
) -> Stream
```

OpenResponses events, each read as its variant of OpenResponsesStreamEvent.

### `Engine.get_response`

```text
get_response(response_id: str) -> types.ResponseResource
```

### `Engine.delete_response`

```text
delete_response(response_id: str) -> types.ResponseDeleted
```

### `Engine.cancel_response`

```text
cancel_response(response_id: str) -> types.ResponseResource
```

### `Engine.re_isq`

```text
re_isq(
    ggml_type: str,
    model: str | None = None,
) -> types.ReIsqResponse
```

Requantizes a model that loaded with ISQ; answers once the engine has queued it.

### `Engine.calibration_start`

```text
calibration_start(model: str | None = None) -> types.CalibrationStatus
```

Starts collecting activation statistics from the requests the engine serves.

### `Engine.calibration_status`

```text
calibration_status(
    model: str | None = None,
) -> types.CalibrationStatus
```

### `Engine.cache_stats`

```text
cache_stats() -> types.CacheStats
```

### `Engine.calibration_apply`

```text
calibration_apply(
    save_cimatrix: str | None = None,
    model: str | None = None,
) -> types.CalibrationStatus
```

Requantizes from the collected statistics; returns the status as it stood before.

### `Engine.list_sessions`

```text
list_sessions() -> types.SessionList
```

### `Engine.get_session`

```text
get_session(session_id: str) -> types.SerializedSession
```

### `Engine.put_session`

```text
put_session(
    session_id: str,
    session: types.SerializedSession | str,
) -> types.SessionStored
```

Imports a session under `session_id`, replacing any session there.

### `Engine.fork_session`

```text
fork_session(session_id: str, num_turns: int) -> types.SessionStored
```

Branches a session into a new one, named by the engine, holding its first `num_turns` turns.

### `Engine.delete_session`

```text
delete_session(session_id: str) -> types.SessionDeleted
```

### `Engine.prompt_logits`

```text
prompt_logits(
    prompt: str | list[int],
    output: str = 'logprobs',
    model: str | None = None,
) -> tuple[types.PromptLogits, array.array | None]
```

Each prompt token's log-probability, and with ``output="logits"`` the row-major logits.

### `Engine.register_logits_processor`

```text
register_logits_processor(name: str, processor) -> HostRegistration
```

See `JsonEngine.register_logits_processor`.

### `Engine.register_tool`

```text
register_tool(tool: HostTool) -> HostRegistration
```

See `JsonEngine.register_tool`.

### `Engine.tokenize`

```text
tokenize(
    text: str,
    add_special_tokens: bool = True,
    model: str | None = None,
) -> list[int]
```

### `Engine.tokenize_chat`

```text
tokenize_chat(request: types.ChatCompletionRequest | str) -> list[int]
```

The prompt tokens a chat request renders to, with its tools, reasoning controls and the chat template.

### `Engine.detokenize`

```text
detokenize(
    tokens: Sequence[int],
    skip_special_tokens: bool = True,
    model: str | None = None,
) -> str
```

### `Engine.list_models`

```text
list_models() -> types.ModelObjects
```

### `Engine.add_model`

```text
add_model(spec: types.ModelSpec | str) -> types.ModelStatusResponse
```

Loads another model into the running engine with the runtime settings it was loaded with.

### `Engine.remove_model`

```text
remove_model(model_id: str) -> types.ModelRemoved
```

### `Engine.set_default_model`

```text
set_default_model(model_id: str) -> types.DefaultModel
```

### `Engine.add_model_alias`

```text
add_model_alias(alias: str, model_id: str) -> types.ModelAlias
```

### `Engine.model_served`

```text
model_served(model_id: str) -> bool
```

Whether a request naming `model_id` would be routed, adapter aliases included.

### `Engine.list_mcp_tools`

```text
list_mcp_tools() -> types.McpToolList
```

### `Engine.unload_model`

```text
unload_model(model_id: str) -> types.ModelStatusResponse
```

### `Engine.reload_model`

```text
reload_model(model_id: str) -> types.ModelStatusResponse
```

### `Engine.model_status`

```text
model_status(model_id: str) -> types.ModelStatusResponse
```

### `Engine.list_lora_adapters`

```text
list_lora_adapters(
    model: str | None = None,
) -> types.LoraAdapterListResponse
```

### `Engine.load_lora_adapter`

```text
load_lora_adapter(
    request: types.LoadLoraAdapterRequest | str,
) -> types.LoraAdapterObject
```

### `Engine.unload_lora_adapter`

```text
unload_lora_adapter(
    request: types.UnloadLoraAdapterRequest | str,
) -> types.LoraAdapterObject
```

### `Engine.image_generation`

```text
image_generation(
    request: types.ImageGenerationRequest | str,
) -> types.ImageGenerationResponse
```

### `Engine.speech_generation`

```text
speech_generation(
    request: types.SpeechGenerationRequest | str,
) -> Blob
```

The audio; its MIME type carries the sample rate and channel count.

### `Engine.resolve_approval`

```text
resolve_approval(
    approval_id: str,
    decision: types.ApprovalDecisionRequest | str,
) -> types.ApprovalDecisionResponse
```

### `Engine.upload_file`

```text
upload_file(
    data: bytes,
    filename: str,
    purpose: str,
    mime_type: str | None = None,
) -> types.FileMetadata
```

### `Engine.list_files`

```text
list_files() -> types.FileListObject
```

### `Engine.get_file`

```text
get_file(file_id: str) -> types.FileMetadata
```

### `Engine.delete_file`

```text
delete_file(file_id: str) -> types.FileDeleted
```

### `Engine.file_content`

```text
file_content(file_id: str) -> Blob
```

### `Engine.list_container_files`

```text
list_container_files(
    container_id: str,
) -> types.ContainerFileListObject
```

### `Engine.get_container_file`

```text
get_container_file(
    container_id: str,
    file_id: str,
) -> types.ContainerFileMetadata
```

### `Engine.container_file_content`

```text
container_file_content(container_id: str, file_id: str) -> Blob
```

### `Engine.upload_skill`

```text
upload_skill(files: Sequence[SkillFile]) -> types.SkillObject
```

### `Engine.upload_skill_version`

```text
upload_skill_version(
    skill_id: str,
    files: Sequence[SkillFile],
) -> types.SkillVersionObject
```

### `Engine.list_skills`

```text
list_skills() -> types.SkillListObject
```

### `Engine.list_skill_versions`

```text
list_skill_versions(
    skill_id: str,
) -> types.AnthropicSkillVersionListObject
```


## `HostCallbacks`


## `HostRegistration`

A logits processor or tool registered on a running engine, holding it open until closed (or its `with` ends).

Closing it unregisters the entry; requests still running that named it fail once it is closed.

### `HostRegistration.__init__`

```text
__init__(
    handle,
    engine,
    unregister_native,
    name: bytes,
    entry_id: int,
)
```

### `HostRegistration.close`

```text
close() -> None
```

### `HostRegistration.__enter__`

```text
__enter__()
```

### `HostRegistration.__exit__`

```text
__exit__()
```


## `HostTool`


## `HostToolCall`

| Field | Type |
| --- | --- |
| `name` | `str` |
| `arguments_json` | `str` |
| `session_id` | `str \| None` |
| `round` | `int \| None` |


## `InferenceError`

A native call returned a non-OK status; `detail` is, for engine calls, the protocol's error JSON.

### `InferenceError.__init__`

```text
__init__(status: int, detail: str, operation: str)
```

### `InferenceError.code`

```text
code()
```

The error JSON's `code` (OpenAI envelope) or `type` (Anthropic envelope), if it has one.


## `JsonEngine`

A loaded model serving JSON strings, as the HTTP server takes and returns them; see `Engine` for classes.

Calls block and release the GIL, so several threads may share an engine. Close it, or use `with`. A host
callback must not hold the last reference to its engine, which would then be freed on the engine's own thread.

### `JsonEngine.__init__`

```text
__init__(
    spec_json: str,
    callbacks: _callbacks.HostCallbacks | None = None,
)
```

### `JsonEngine.abi_version`

```text
abi_version() -> int
```

(major << 16) | (minor << 8) | patch of the ABI the library implements.

### `JsonEngine.build_version`

```text
build_version() -> str
```

### `JsonEngine.close`

```text
close()
```

### `JsonEngine.for_owner`

```text
for_owner(owner: str) -> 'JsonEngine'
```

The same engine acting for `owner`: what it stores is that owner's, and it reaches no one else's.

Close it like any engine; this one stays open, with its callbacks, until every engine made from it is closed.

### `JsonEngine.__enter__`

```text
__enter__()
```

### `JsonEngine.__exit__`

```text
__exit__()
```

### `JsonEngine._call`

```text
_call(name: str, value: str) -> str
```

### `JsonEngine._stream`

```text
_stream(name: str, value: str) -> Stream
```

### `JsonEngine._get`

```text
_get(name: str) -> str
```

### `JsonEngine._blob`

```text
_blob(name: str, value: str) -> Blob
```

### `JsonEngine._media`

```text
_media(media: Sequence[MediaAttachment], buffers: _Buffers)
```

### `JsonEngine._skill_files`

```text
_skill_files(files: Sequence[SkillFile], buffers: _Buffers)
```

### `JsonEngine.chat`

```text
chat(request_json: str, media: Sequence[MediaAttachment] = ()) -> str
```

### `JsonEngine.chat_stream`

```text
chat_stream(
    request_json: str,
    media: Sequence[MediaAttachment] = (),
) -> Stream
```

### `JsonEngine.completion`

```text
completion(request_json: str) -> str
```

### `JsonEngine.completion_stream`

```text
completion_stream(request_json: str) -> Stream
```

### `JsonEngine.embeddings`

```text
embeddings(request_json: str) -> str
```

### `JsonEngine.anthropic_messages`

```text
anthropic_messages(request_json: str) -> str
```

An Anthropic Messages request; failures carry the Anthropic error envelope.

### `JsonEngine.anthropic_count_tokens`

```text
anthropic_count_tokens(request_json: str) -> str
```

The prompt tokens an Anthropic Messages request would use: {"input_tokens"}.

### `JsonEngine.anthropic_messages_stream`

```text
anthropic_messages_stream(request_json: str) -> Stream
```

### `JsonEngine.create_response`

```text
create_response(request_json: str) -> str
```

### `JsonEngine.response_stream`

```text
response_stream(request_json: str) -> Stream
```

### `JsonEngine.get_response`

```text
get_response(response_id: str) -> str
```

### `JsonEngine.delete_response`

```text
delete_response(response_id: str) -> str
```

### `JsonEngine.cancel_response`

```text
cancel_response(response_id: str) -> str
```

### `JsonEngine.list_models`

```text
list_models() -> str
```

### `JsonEngine.add_model`

```text
add_model(request_json: str) -> str
```

Loads another model into the running engine; the request is one entry of the spec's "models".

### `JsonEngine.remove_model`

```text
remove_model(request_json: str) -> str
```

### `JsonEngine.set_default_model`

```text
set_default_model(request_json: str) -> str
```

### `JsonEngine.add_model_alias`

```text
add_model_alias(request_json: str) -> str
```

### `JsonEngine.model_served`

```text
model_served(request_json: str) -> str
```

Whether a request naming the model in {"model_id"} would be routed: {"model_id", "served"}.

### `JsonEngine.list_mcp_tools`

```text
list_mcp_tools() -> str
```

The tools the engine's MCP servers give the default model.

### `JsonEngine.unload_model`

```text
unload_model(request_json: str) -> str
```

### `JsonEngine.reload_model`

```text
reload_model(request_json: str) -> str
```

### `JsonEngine.model_status`

```text
model_status(request_json: str) -> str
```

### `JsonEngine.list_lora_adapters`

```text
list_lora_adapters(request_json: str = '{}') -> str
```

### `JsonEngine.load_lora_adapter`

```text
load_lora_adapter(request_json: str) -> str
```

### `JsonEngine.unload_lora_adapter`

```text
unload_lora_adapter(request_json: str) -> str
```

### `JsonEngine.image_generation`

```text
image_generation(request_json: str) -> str
```

### `JsonEngine.speech_generation`

```text
speech_generation(request_json: str) -> Blob
```

Speaks text; the blob's MIME type carries the sample rate and channel count.

### `JsonEngine._call2`

```text
_call2(name: str, first: str, second: str) -> str
```

### `JsonEngine._blob2`

```text
_blob2(name: str, first: str, second: str) -> Blob
```

### `JsonEngine._pair`

```text
_pair(name: str, first: str, second: str) -> ctypes.c_void_p
```

### `JsonEngine.resolve_approval`

```text
resolve_approval(approval_id: str, decision_json: str) -> str
```

Answers the approval an agentic_tool_approval_required stream event named.

### `JsonEngine.upload_file`

```text
upload_file(
    data: bytes,
    filename: str,
    purpose: str,
    mime_type: str | None = None,
) -> str
```

### `JsonEngine.re_isq`

```text
re_isq(request_json: str) -> str
```

### `JsonEngine.calibration_start`

```text
calibration_start(request_json: str = '{}') -> str
```

### `JsonEngine.calibration_status`

```text
calibration_status(request_json: str = '{}') -> str
```

### `JsonEngine.cache_stats`

```text
cache_stats() -> str
```

Each loaded model's cumulative prefix- and encoder-cache counters; diff two readings for a span.

### `JsonEngine.calibration_apply`

```text
calibration_apply(request_json: str = '{}') -> str
```

### `JsonEngine.list_sessions`

```text
list_sessions() -> str
```

### `JsonEngine.get_session`

```text
get_session(session_id: str) -> str
```

### `JsonEngine.put_session`

```text
put_session(session_id: str, session_json: str) -> str
```

### `JsonEngine.register_logits_processor`

```text
register_logits_processor(
    name: str,
    processor,
) -> _callbacks.HostRegistration
```

Makes `processor(logits, context)` selectable by name in a request's "logits_processors".

Each decoding step it edits `logits` (a ctypes float array of the vocabulary's size) in place, given `context`,
every token so far, the prompt's included; both are valid only during the call. It runs on engine worker
threads, and an exception fails the request. Every engine sharing this one sees it. The result keeps the
engine open until it is closed, which unregisters the processor.

### `JsonEngine.register_tool`

```text
register_tool(
    tool: _callbacks.HostTool,
) -> _callbacks.HostRegistration
```

Registers `tool` after load; a chat request offers it to the model by naming it in "host_tools".

Every engine sharing this one sees it. The result keeps the engine open until it is closed, which
unregisters the tool.

### `JsonEngine.fork_session`

```text
fork_session(session_id: str, request_json: str) -> str
```

Branches a session into a new one the engine names; the request is {"num_turns"}, the answer {"id"}.

### `JsonEngine.delete_session`

```text
delete_session(session_id: str) -> str
```

### `JsonEngine.prompt_logits`

```text
prompt_logits(request_json: str) -> tuple[str, array.array | None]
```

Scores a prompt: the response JSON, and with "output": "logits" its row-major f32 logits.

### `JsonEngine.tokenize`

```text
tokenize(request_json: str) -> str
```

### `JsonEngine.tokenize_chat`

```text
tokenize_chat(request_json: str) -> str
```

Tokenizes a chat completion request as the chat template renders it, to {"tokens"}.

### `JsonEngine.detokenize`

```text
detokenize(request_json: str) -> str
```

### `JsonEngine.list_files`

```text
list_files() -> str
```

### `JsonEngine.get_file`

```text
get_file(file_id: str) -> str
```

### `JsonEngine.delete_file`

```text
delete_file(file_id: str) -> str
```

### `JsonEngine.file_content`

```text
file_content(file_id: str) -> Blob
```

### `JsonEngine.list_container_files`

```text
list_container_files(container_id: str) -> str
```

The files a Responses container (a code-running session) produced.

### `JsonEngine.get_container_file`

```text
get_container_file(container_id: str, file_id: str) -> str
```

### `JsonEngine.container_file_content`

```text
container_file_content(container_id: str, file_id: str) -> Blob
```

### `JsonEngine.upload_skill`

```text
upload_skill(files: Sequence[SkillFile]) -> str
```

### `JsonEngine.upload_skill_version`

```text
upload_skill_version(skill_id: str, files: Sequence[SkillFile]) -> str
```

### `JsonEngine.list_skills`

```text
list_skills() -> str
```

### `JsonEngine.list_skill_versions`

```text
list_skill_versions(skill_id: str) -> str
```


## `MediaAttachment`

A buffer a request names by position: an image, audio or video URL of media://0 is the first.

| Field | Type | Default |
| --- | --- | --- |
| `data` | `bytes` | required |
| `mime_type` | `str \| None` | optional |


## `SkillFile`


## `Status`

Mirrors inference_status.


## `Stream`

A streaming request's events; close it, or use `with`, to abandon it. An open stream keeps its engine.

### `Stream.__init__`

```text
__init__(engine: Handle, pointer)
```

### `Stream.next`

```text
next(timeout: float | None = None) -> StreamEvent | None
```

The next event within `timeout` seconds; None on a timeout, or at the end (then `done` is True).

### `Stream.cancel`

```text
cancel()
```

Asks the request to stop; keep reading for its final event, which carries usage. Safe from any thread.

### `Stream.__iter__`

```text
__iter__() -> Iterator[StreamEvent]
```

### `Stream.close`

```text
close()
```

### `Stream.__enter__`

```text
__enter__()
```

### `Stream.__exit__`

```text
__exit__()
```


## `StreamEvent`

A stream event by protocol name; a failed request ends with an `error` event holding the error JSON.

| Field | Type |
| --- | --- |
| `name` | `str` |
| `data` | `object` |

---

<small>Generated from [`bindings/python/inference_rs`](https://github.com/christopherthompson81/inference.rs/blob/master/bindings/python/inference_rs).</small>

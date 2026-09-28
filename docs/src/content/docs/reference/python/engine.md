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

### `Engine.list_models`

```text
list_models() -> types.ModelObjects
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

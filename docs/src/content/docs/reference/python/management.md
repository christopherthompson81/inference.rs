---
title: Models, adapters, files and skills
description: "Model status and cache counters, LoRA adapters, files, skills, approvals, sessions, MCP tools, calibration, tokenization and the media generation calls."
sidebar:
  order: 7
---
## `ApprovalDecision`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `ApprovalDecision.APPROVE` | `'approve'` |
| `ApprovalDecision.DENY` | `'deny'` |


## `ApprovalDecisionRequest`

Decision payload for a pending agentic tool approval.

| Field | Type | Default |
| --- | --- | --- |
| `decision` | `ApprovalDecision` | required |
| `message` | `str \| None` | optional |
| `remember_for_session` | `bool \| None` | optional |


## `ApprovalDecisionResponse`

| Field | Type |
| --- | --- |
| `status` | `str` |


## `AudioResponseFormat`

Audio format options for speech generation responses.

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `AudioResponseFormat.MP3` | `'mp3'` |
| `AudioResponseFormat.OPUS` | `'opus'` |
| `AudioResponseFormat.AAC` | `'aac'` |
| `AudioResponseFormat.FLAC` | `'flac'` |
| `AudioResponseFormat.WAV` | `'wav'` |
| `AudioResponseFormat.PCM` | `'pcm'` |


## `CacheStats`

Cache counters for each loaded model, counted since it loaded (the prefix ones since its engine last started); a caller diffs two readings to see what the requests between them used.

| Field | Type |
| --- | --- |
| `data` | `list[ModelCacheStats]` |
| `object` | `str` |


## `CalibrationApplyRequest`

| Field | Type | Default |
| --- | --- | --- |
| `save_cimatrix` | `str \| None` | optional |


## `CalibrationStatus`

| Field | Type |
| --- | --- |
| `collecting` | `bool` |
| `layers` | `int` |
| `layers_tracking` | `int` |
| `max_rows` | `int` |
| `min_rows` | `int` |
| `total_rows` | `int` |


## `ContainerFileListObject`

| Field | Type |
| --- | --- |
| `data` | `list[ContainerFileMetadata]` |
| `object` | `str` |


## `ContainerFileMetadata`

OpenAI-compatible container file metadata backed by the same in-process file store.

| Field | Type | Default |
| --- | --- | --- |
| `bytes` | `int` | required |
| `container_id` | `str` | required |
| `created_at` | `int` | required |
| `filename` | `str` | required |
| `format` | `str \| None` | optional |
| `id` | `str` | required |
| `mime_type` | `str` | required |
| `object` | `str` | required |
| `source` | `SourceMeta` | required |


## `DefaultModel`

The model a request without `model`, or naming `default`, goes to.

| Field | Type |
| --- | --- |
| `model_id` | `str` |


## `DetokenizeRequest`

| Field | Type | Default |
| --- | --- | --- |
| `model` | `str \| None` | optional |
| `skip_special_tokens` | `bool \| None` | `True` |
| `tokens` | `list[int]` | required |


## `DetokenizeResponse`

| Field | Type |
| --- | --- |
| `text` | `str` |


## `EncoderCacheStats`

| Field | Type |
| --- | --- |
| `hits` | `int` |
| `misses` | `int` |


## `FileDeleted`

| Field | Type |
| --- | --- |
| `deleted` | `bool` |
| `id` | `str` |
| `object` | `str` |


## `FileListObject`

| Field | Type |
| --- | --- |
| `data` | `list[FileMetadata]` |
| `object` | `str` |


## `FileMetadata`

OpenAI file metadata + inference.rs extensions (`format`, `mime_type`, `source`, `truncated`).

| Field | Type | Default |
| --- | --- | --- |
| `bytes` | `int` | required |
| `created_at` | `int` | required |
| `filename` | `str` | required |
| `format` | `str \| None` | optional |
| `id` | `str` | required |
| `mime_type` | `str` | required |
| `object` | `str` | required |
| `purpose` | `str` | required |
| `source` | `SourceMeta` | required |
| `truncated` | `bool \| None` | optional |


## `GenerationDefaults`

| Field | Type | Default |
| --- | --- | --- |
| `do_sample` | `bool \| None` | optional |
| `max_length` | `int \| None` | optional |
| `max_new_tokens` | `int \| None` | optional |
| `min_p` | `float \| None` | optional |
| `repetition_penalty` | `float \| None` | optional |
| `suppress_tokens` | `list[int] \| None` | optional |
| `temperature` | `float \| None` | optional |
| `top_k` | `int \| None` | optional |
| `top_p` | `float \| None` | optional |


## `ImageChoice`

| Field | Type | Default |
| --- | --- | --- |
| `b64_json` | `str \| None` | optional |
| `url` | `str \| None` | optional |


## `ImageGenerationRequest`

Image generation request

| Field | Type | Default |
| --- | --- | --- |
| `height` | `int \| None` | optional |
| `model` | `str \| None` | optional |
| `n` | `int \| None` | optional |
| `prompt` | `str` | required |
| `response_format` | `ImageGenerationResponseFormat \| None` | optional |
| `width` | `int \| None` | optional |


## `ImageGenerationResponse`

| Field | Type |
| --- | --- |
| `created` | `int` |
| `data` | `list[ImageChoice]` |


## `ImageGenerationResponseFormat`

Image generation response format

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `ImageGenerationResponseFormat.URL` | `'url'` |
| `ImageGenerationResponseFormat.B64_JSON` | `'b64_json'` |


## `LoadLoraAdapterRequest`

| Field | Type | Default |
| --- | --- | --- |
| `expected_generation` | `str \| None` | optional |
| `load_inplace` | `bool \| None` | `False` |
| `lora_name` | `str` | required |
| `lora_path` | `str` | required |
| `model` | `str \| None` | optional |


## `LoraAdapterListResponse`

| Field | Type |
| --- | --- |
| `data` | `list[LoraAdapterObject]` |
| `generations` | `list[LoraResidentGenerationObject]` |
| `max_adapters` | `int` |
| `max_bytes` | `int` |
| `max_rank` | `int` |
| `object` | `str` |
| `resident_bytes` | `int` |
| `resident_generations` | `int` |
| `retired_generations` | `int` |


## `LoraAdapterObject`

| Field | Type | Default |
| --- | --- | --- |
| `bytes` | `int` | required |
| `generation` | `str` | required |
| `id` | `str` | required |
| `object` | `str` | required |
| `rank` | `int` | required |
| `revision` | `str \| None` | optional |
| `source` | `str \| None` | optional |


## `LoraResidentGenerationObject`

| Field | Type |
| --- | --- |
| `active_leases` | `int` |
| `aliases` | `list[str]` |
| `bytes` | `int` |
| `generation` | `str` |
| `rank` | `int` |
| `retired` | `bool` |


## `McpToolList`

The tools the engine's MCP servers provide to the default model; built-in tools aren't listed.

| Field | Type |
| --- | --- |
| `data` | `list[McpToolObject]` |
| `object` | `str` |


## `McpToolObject`

| Field | Type | Default |
| --- | --- | --- |
| `description` | `str \| None` | optional |
| `name` | `str` | required |


## `Modality`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `Modality.TEXT` | `'text'` |
| `Modality.AUDIO` | `'audio'` |
| `Modality.VISION` | `'vision'` |
| `Modality.VIDEO` | `'video'` |
| `Modality.EMBEDDING` | `'embedding'` |


## `ModelAlias`

Another id requests can name a served model by.

| Field | Type |
| --- | --- |
| `alias` | `str` |
| `model_id` | `str` |


## `ModelCacheStats`

| Field | Type | Default |
| --- | --- | --- |
| `encoder_cache` | `EncoderCacheStats \| None` | optional |
| `model_id` | `str` | required |
| `prefix_cache_hits` | `int` | required |
| `prefix_cache_sequences` | `int` | required |


## `ModelCategory`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `ModelCategory.TEXT` | `'text'` |
| `ModelCategory.MULTIMODAL` | `'multimodal'` |
| `ModelCategory.DIFFUSION` | `'diffusion'` |
| `ModelCategory.AUDIO` | `'audio'` |
| `ModelCategory.SPEECH` | `'speech'` |
| `ModelCategory.EMBEDDING` | `'embedding'` |


## `ModelModalities`

| Field | Type |
| --- | --- |
| `input` | `list[Modality]` |
| `output` | `list[Modality]` |


## `ModelObject`

Model information metadata about an available mode

| Field | Type | Default |
| --- | --- | --- |
| `adapter_generation` | `str \| None` | optional |
| `category` | `ModelCategory \| None` | optional |
| `created` | `int` | required |
| `default` | `bool \| None` | optional |
| `generation_defaults` | `GenerationDefaults \| None` | optional |
| `id` | `str` | required |
| `max_model_len` | `int \| None` | optional |
| `mcp_servers_connected` | `int \| None` | optional |
| `mcp_tools_count` | `int \| None` | optional |
| `modalities` | `ModelModalities \| None` | optional |
| `object` | `str` | required |
| `owned_by` | `str` | required |
| `parent` | `str \| None` | optional |
| `root` | `str \| None` | optional |
| `status` | `str \| None` | optional |
| `tools_available` | `bool \| None` | optional |


## `ModelObjects`

Collection of available models

| Field | Type |
| --- | --- |
| `data` | `list[ModelObject]` |
| `object` | `str` |


## `ModelOperationRequest`

The body of an unload, reload or status request.

| Field | Type |
| --- | --- |
| `model_id` | `str` |


## `ModelRemoved`

| Field | Type |
| --- | --- |
| `model_id` | `str` |


## `ModelServed`

Whether a request naming `model_id` would be routed: a served model, the `default` alias, or a LoRA adapter.

| Field | Type |
| --- | --- |
| `model_id` | `str` |
| `served` | `bool` |


## `ModelStatus`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `ModelStatus.LOADED` | `'loaded'` |
| `ModelStatus.UNLOADED` | `'unloaded'` |
| `ModelStatus.RELOADING` | `'reloading'` |


## `ModelStatusResponse`

| Field | Type |
| --- | --- |
| `model_id` | `str` |
| `status` | `ModelStatus` |


## `ReIsqRequest`

| Field | Type |
| --- | --- |
| `ggml_type` | `str` |


## `ReIsqResponse`

Answered once the requantization is queued behind the requests already running.

| Field | Type |
| --- | --- |
| `ggml_type` | `str` |


## `SerializedSession`

Wire format. Images and video frames are base64 PNGs.

| Field | Type | Default |
| --- | --- | --- |
| `files` | `list[Any] \| None` | optional |
| `images` | `list[str] \| None` | optional |
| `messages` | `list[Any]` | required |
| `videos` | `list[SerializedVideo] \| None` | optional |


## `SessionDeleted`

| Field | Type |
| --- | --- |
| `deleted` | `bool` |
| `id` | `str` |


## `SessionForkRequest`

Branches a session into a new one, named by the engine, with the source's first `num_turns` turns (0 copies all).

| Field | Type |
| --- | --- |
| `num_turns` | `int` |


## `SessionList`

| Field | Type |
| --- | --- |
| `data` | `list[str]` |


## `SessionStored`

| Field | Type |
| --- | --- |
| `id` | `str` |


## `SignInRequest`

`{"key"}` to sign a browser in.

| Field | Type |
| --- | --- |
| `key` | `str` |


## `SkillListObject`

| Field | Type |
| --- | --- |
| `data` | `list[SkillObject]` |
| `object` | `str` |


## `SkillListQuery`

| Field | Type | Default |
| --- | --- | --- |
| `limit` | `int \| None` | optional |
| `page` | `str \| None` | optional |
| `source` | `str \| None` | optional |


## `SkillObject`

| Field | Type |
| --- | --- |
| `created_at` | `int` |
| `description` | `str` |
| `id` | `str` |
| `latest_version` | `int` |
| `name` | `str` |
| `object` | `str` |


## `SkillVersionObject`

| Field | Type |
| --- | --- |
| `created_at` | `int` |
| `description` | `str` |
| `id` | `str` |
| `name` | `str` |
| `object` | `str` |
| `skill_id` | `str` |
| `version` | `int` |


## `SourceMeta`

Which agentic tool produced the file, and when in the session.

| Field | Type |
| --- | --- |
| `round` | `int` |
| `tool` | `str` |
| `turn` | `int` |


## `SpeechGenerationRequest`

Speech generation request

| Field | Type | Default |
| --- | --- | --- |
| `input` | `str` | required |
| `model` | `str \| None` | optional |
| `response_format` | `AudioResponseFormat` | required |


## `TokenizeRequest`

| Field | Type | Default |
| --- | --- | --- |
| `add_special_tokens` | `bool \| None` | `True` |
| `model` | `str \| None` | optional |
| `text` | `str` | required |


## `TokenizeResponse`

| Field | Type |
| --- | --- |
| `tokens` | `list[int]` |


## `TuneModelRequest`

| Field | Type | Default |
| --- | --- | --- |
| `cpu` | `bool \| None` | optional |
| `dtype` | `str \| None` | optional |
| `hf_revision` | `str \| None` | optional |
| `max_batch_size` | `int \| None` | optional |
| `max_image_length` | `int \| None` | optional |
| `max_num_images` | `int \| None` | optional |
| `max_seq_len` | `int \| None` | optional |
| `model_id` | `str` | required |
| `profile` | `TuneProfileRequest \| None` | optional |
| `requested_isq` | `str \| None` | optional |
| `token_source` | `str \| None` | optional |


## `TuneProfileRequest`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `TuneProfileRequest.QUALITY` | `'quality'` |
| `TuneProfileRequest.BALANCED` | `'balanced'` |
| `TuneProfileRequest.FAST` | `'fast'` |


## `UnloadLoraAdapterRequest`

| Field | Type | Default |
| --- | --- | --- |
| `expected_generation` | `str \| None` | optional |
| `lora_int_id` | `int \| None` | optional |
| `lora_name` | `str` | required |
| `model` | `str \| None` | optional |

---

<small>Generated from [`bindings/python/inference_rs`](https://github.com/christopherthompson81/inference.rs/blob/master/bindings/python/inference_rs).</small>

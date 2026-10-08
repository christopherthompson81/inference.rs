---
title: Engine spec
description: "What to load and how to run it: EngineSpec, the ModelSelected variants and their options."
sidebar:
  order: 3
---
## `AdapterSpec`

Runtime LoRA adapter management; listing adapters is always allowed.

| Field | Type | Default |
| --- | --- | --- |
| `root` | `str \| None` | optional |
| `runtime_updates` | `bool \| None` | optional |


## `AgentPermission`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `AgentPermission.AUTO` | `'auto'` |
| `AgentPermission.ASK` | `'ask'` |
| `AgentPermission.DENY` | `'deny'` |


## `AgenticSpec`

| Field | Type | Default |
| --- | --- | --- |
| `agent_permission` | `AgentPermission \| None` | optional |
| `code_execution` | `CodeExecutionConfig \| None` | optional |
| `max_tool_rounds` | `int \| None` | optional |
| `mcp` | `McpClientConfig \| None` | optional |
| `sandbox` | `SandboxMode \| None` | optional |
| `sandbox_limits` | `SandboxLimits \| None` | optional |
| `sandbox_profile` | `SandboxProfile \| None` | optional |
| `search` | `SearchSpec \| None` | optional |
| `shell` | `ShellConfig \| None` | optional |
| `tool_dispatch_url` | `str \| None` | optional |


## `AnyMoeConfig`

| Field | Type | Default |
| --- | --- | --- |
| `batch_size` | `int \| None` | optional |
| `epochs` | `int \| None` | optional |
| `expert_type` | `AnyMoeExpertType` | required |
| `gate_model_id` | `str \| None` | optional |
| `hidden_size` | `int` | required |
| `loss_csv_path` | `str \| None` | optional |
| `lr` | `float \| None` | optional |
| `training` | `bool \| None` | optional |


## `AnyMoeExpertType`

One of: `Union[Literal['fine_tuned'], AnyMoeExpertTypeLoraAdapter]`.


## `AnyMoeExpertTypeLoraAdapter`

| Field | Type |
| --- | --- |
| `alpha` | `float` |
| `rank` | `int` |
| `target_modules` | `list[str]` |


## `AnyMoeSpec`

The AnyMoE layer to build on top of the loaded model.

| Field | Type | Default |
| --- | --- | --- |
| `config` | `AnyMoeConfig` | required |
| `layers` | `list[int] \| None` | optional |
| `mlp` | `str` | required |
| `model_ids` | `list[str]` | required |
| `path` | `str` | required |
| `prefix` | `str` | required |


## `CodeExecutionConfig`

Python code execution config.

| Field | Type | Default |
| --- | --- | --- |
| `permission` | `CodeExecutionPermission \| None` | optional |
| `python_path` | `str \| None` | optional |
| `sandbox_policy` | `SandboxPolicy \| None` | optional |
| `timeout_secs` | `int \| None` | optional |
| `working_directory` | `str \| None` | optional |


## `CodeExecutionPermission`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `CodeExecutionPermission.AUTO` | `'auto'` |
| `CodeExecutionPermission.ASK` | `'ask'` |
| `CodeExecutionPermission.DENY` | `'deny'` |


## `DiffusionLoaderType`

The architecture to load the diffusion model as.

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `DiffusionLoaderType.FLUX` | `'flux'` |
| `DiffusionLoaderType.FLUX_OFFLOADED` | `'flux-offloaded'` |


## `EmbeddingLoaderType`

The architecture to load the embedding model as.

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `EmbeddingLoaderType.EMBEDDINGGEMMA` | `'embeddinggemma'` |
| `EmbeddingLoaderType.QWEN3EMBEDDING` | `'qwen3embedding'` |


## `EngineSpec`

What to load and how to run it: the JSON form of the options `inference serve` takes.

| Field | Type | Default |
| --- | --- | --- |
| `adapters` | `AdapterSpec \| None` | optional |
| `agentic` | `AgenticSpec \| None` | optional |
| `anymoe` | `AnyMoeSpec \| None` | optional |
| `default_model_id` | `str \| None` | optional |
| `model` | `ModelSelected \| None` | optional |
| `model_id` | `str \| None` | optional |
| `models` | `list[ModelSpec] \| None` | optional |
| `runtime` | `RuntimeSpec \| None` | optional |
| `skills` | `SkillsSpec \| None` | optional |


## `IsqOrganization`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `IsqOrganization.DEFAULT` | `'default'` |
| `IsqOrganization.MOQE` | `'moqe'` |


## `LoraAdapterSpec`

Alias and source used to preload a LoRA adapter.

| Field | Type | Default |
| --- | --- | --- |
| `alias` | `str` | required |
| `base_model_name` | `str \| None` | optional |
| `revision` | `str \| None` | optional |
| `source` | `str` | required |


## `LoraRuntimeConfig`

Admission limits for a dynamic LoRA runtime.

| Field | Type | Default |
| --- | --- | --- |
| `max_adapters` | `int \| None` | `16` |
| `max_bytes` | `int \| None` | `8589934592` |
| `max_rank` | `int \| None` | `256` |


## `McpClientConfig`

Configuration for MCP client integration

| Field | Type | Default |
| --- | --- | --- |
| `auto_register_tools` | `bool \| None` | `True` |
| `max_concurrent_calls` | `int \| None` | optional |
| `servers` | `list[McpServerConfig] \| None` | optional |
| `tool_timeout_secs` | `int \| None` | optional |


## `McpServerConfig`

Configuration for an individual MCP server

| Field | Type | Default |
| --- | --- | --- |
| `bearer_token` | `str \| None` | optional |
| `enabled` | `bool \| None` | optional |
| `id` | `str \| None` | optional |
| `name` | `str` | required |
| `resources` | `list[str] \| None` | optional |
| `source` | `McpServerSource` | required |
| `tool_prefix` | `str \| None` | optional |


## `McpServerSource`

One of: `Union[McpServerSourceHttp, McpServerSourceProcess, McpServerSourceWebSocket]`.


## `McpServerSourceHttp`

HTTP-based MCP server using JSON-RPC over HTTP

| Field | Type | Default |
| --- | --- | --- |
| `headers` | `dict[str, str] \| None` | optional |
| `timeout_secs` | `int \| None` | optional |
| `type` | `Literal['Http']` | `'Http'` |
| `url` | `str` | required |


## `McpServerSourceProcess`

Local process-based MCP server using stdin/stdout communication

| Field | Type | Default |
| --- | --- | --- |
| `args` | `list[str]` | required |
| `command` | `str` | required |
| `env` | `dict[str, str] \| None` | optional |
| `type` | `Literal['Process']` | `'Process'` |
| `work_dir` | `str \| None` | optional |


## `McpServerSourceWebSocket`

WebSocket-based MCP server for real-time bidirectional communication

| Field | Type | Default |
| --- | --- | --- |
| `headers` | `dict[str, str] \| None` | optional |
| `timeout_secs` | `int \| None` | optional |
| `type` | `Literal['WebSocket']` | `'WebSocket'` |
| `url` | `str` | required |


## `MmprojSelection`

How a GGUF spec without `mmproj_filename` gets its multimodal projector.

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `MmprojSelection.GIVEN` | `'given'` |
| `MmprojSelection.ARTIFACT_REPO` | `'artifact_repo'` |
| `MmprojSelection.ANY` | `'any'` |
| `MmprojSelection.REQUIRED` | `'required'` |


## `ModelDType`

DType for the model.

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `ModelDType.AUTO` | `'auto'` |
| `ModelDType.BF16` | `'bf16'` |
| `ModelDType.F16` | `'f16'` |
| `ModelDType.F32` | `'f32'` |


## `ModelSelected`

One of: `Union[ModelSelectedRun, ModelSelectedPlain, ModelSelectedLora, ModelSelectedGGUF, ModelSelectedGGML, ModelSelectedMultimodalPlain, ModelSelectedDiffusionPlain, ModelSelectedSpeech, ModelSelectedEmbedding]`.


## `ModelSelectedDiffusionPlain`

Select a diffusion model, without quantization or adapters

| Field | Type | Default |
| --- | --- | --- |
| `arch` | `DiffusionLoaderType` | required |
| `dtype` | `ModelDType \| None` | `ModelDType.AUTO` |
| `model_id` | `str` | required |


## `ModelSelectedEmbedding`

Select an embedding model, without quantization or adapters

| Field | Type | Default |
| --- | --- | --- |
| `arch` | `EmbeddingLoaderType \| None` | optional |
| `calibration_file` | `str \| None` | optional |
| `dtype` | `ModelDType \| None` | `ModelDType.AUTO` |
| `from_uqff` | `str \| None` | optional |
| `hf_cache_path` | `str \| None` | optional |
| `imatrix` | `str \| None` | optional |
| `model_id` | `str` | required |
| `quant` | `str \| None` | optional |
| `tokenizer_json` | `str \| None` | optional |
| `topology` | `str \| None` | optional |
| `write_uqff` | `UqffWriteSpec \| None` | optional |


## `ModelSelectedGGML`

Select a GGML model.

| Field | Type | Default |
| --- | --- | --- |
| `dtype` | `ModelDType \| None` | `ModelDType.AUTO` |
| `gqa` | `int` | required |
| `max_batch_size` | `int \| None` | `1` |
| `max_seq_len` | `int \| None` | `4096` |
| `quantized_filename` | `str` | required |
| `quantized_model_id` | `str` | required |
| `tok_model_id` | `str` | required |
| `tokenizer_json` | `str \| None` | optional |
| `topology` | `str \| None` | optional |


## `ModelSelectedGGUF`

Select a GGUF model.

| Field | Type | Default |
| --- | --- | --- |
| `calibration_file` | `str \| None` | optional |
| `dtype` | `ModelDType \| None` | `ModelDType.AUTO` |
| `hf_cache_path` | `str \| None` | optional |
| `imatrix` | `str \| None` | optional |
| `lora_adapters` | `list[LoraAdapterSpec] \| None` | optional |
| `lora_runtime_config` | `LoraRuntimeConfig \| None` | optional |
| `matformer_config_path` | `str \| None` | optional |
| `matformer_slice_name` | `str \| None` | optional |
| `max_batch_size` | `int \| None` | `1` |
| `max_edge` | `int \| None` | optional |
| `max_image_length` | `int \| None` | optional |
| `max_num_images` | `int \| None` | optional |
| `max_seq_len` | `int \| None` | `4096` |
| `mmproj_filename` | `str \| None` | optional |
| `mmproj_selection` | `MmprojSelection \| None` | optional |
| `organization` | `IsqOrganization \| None` | optional |
| `quant` | `str \| None` | optional |
| `quantized_filename` | `str \| None` | optional |
| `quantized_model_id` | `str` | required |
| `tok_model_id` | `str \| None` | optional |
| `tokenizer_json` | `str \| None` | optional |
| `topology` | `str \| None` | optional |
| `write_uqff` | `UqffWriteSpec \| None` | optional |


## `ModelSelectedLora`

Select a LoRA architecture

| Field | Type | Default |
| --- | --- | --- |
| `adapters` | `list[LoraAdapterSpec] \| None` | optional |
| `arch` | `NormalLoaderType \| None` | optional |
| `calibration_file` | `str \| None` | optional |
| `dtype` | `ModelDType \| None` | `ModelDType.AUTO` |
| `from_uqff` | `str \| None` | optional |
| `hf_cache_path` | `str \| None` | optional |
| `imatrix` | `str \| None` | optional |
| `matformer_config_path` | `str \| None` | optional |
| `matformer_slice_name` | `str \| None` | optional |
| `max_batch_size` | `int \| None` | `1` |
| `max_edge` | `int \| None` | optional |
| `max_image_length` | `int \| None` | optional |
| `max_num_images` | `int \| None` | optional |
| `max_seq_len` | `int \| None` | `4096` |
| `mmproj_selection` | `MmprojSelection \| None` | optional |
| `model_id` | `str` | required |
| `organization` | `IsqOrganization \| None` | optional |
| `quant` | `str \| None` | optional |
| `runtime_config` | `LoraRuntimeConfig \| None` | optional |
| `tokenizer_json` | `str \| None` | optional |
| `topology` | `str \| None` | optional |
| `write_uqff` | `UqffWriteSpec \| None` | optional |


## `ModelSelectedMultimodalPlain`

Select a multimodal plain model, without quantization or adapters

| Field | Type | Default |
| --- | --- | --- |
| `arch` | `MultimodalLoaderType \| None` | optional |
| `calibration_file` | `str \| None` | optional |
| `dtype` | `ModelDType \| None` | `ModelDType.AUTO` |
| `from_uqff` | `str \| None` | optional |
| `hf_cache_path` | `str \| None` | optional |
| `imatrix` | `str \| None` | optional |
| `matformer_config_path` | `str \| None` | optional |
| `matformer_slice_name` | `str \| None` | optional |
| `max_batch_size` | `int \| None` | `1` |
| `max_edge` | `int \| None` | optional |
| `max_image_length` | `int \| None` | `1024` |
| `max_num_images` | `int \| None` | `1` |
| `max_seq_len` | `int \| None` | `4096` |
| `model_id` | `str` | required |
| `organization` | `IsqOrganization \| None` | optional |
| `quant` | `str \| None` | optional |
| `tokenizer_json` | `str \| None` | optional |
| `topology` | `str \| None` | optional |
| `write_uqff` | `UqffWriteSpec \| None` | optional |


## `ModelSelectedPlain`

Select a plain model, without quantization or adapters

| Field | Type | Default |
| --- | --- | --- |
| `arch` | `NormalLoaderType \| None` | optional |
| `calibration_file` | `str \| None` | optional |
| `dtype` | `ModelDType \| None` | `ModelDType.AUTO` |
| `from_uqff` | `str \| None` | optional |
| `hf_cache_path` | `str \| None` | optional |
| `imatrix` | `str \| None` | optional |
| `matformer_config_path` | `str \| None` | optional |
| `matformer_slice_name` | `str \| None` | optional |
| `max_batch_size` | `int \| None` | `1` |
| `max_seq_len` | `int \| None` | `4096` |
| `model_id` | `str` | required |
| `organization` | `IsqOrganization \| None` | optional |
| `quant` | `str \| None` | optional |
| `tokenizer_json` | `str \| None` | optional |
| `topology` | `str \| None` | optional |
| `write_uqff` | `UqffWriteSpec \| None` | optional |


## `ModelSelectedRun`

Select a model for running via auto loader

| Field | Type | Default |
| --- | --- | --- |
| `calibration_file` | `str \| None` | optional |
| `dtype` | `ModelDType \| None` | `ModelDType.AUTO` |
| `from_uqff` | `str \| None` | optional |
| `hf_cache_path` | `str \| None` | optional |
| `imatrix` | `str \| None` | optional |
| `matformer_config_path` | `str \| None` | optional |
| `matformer_slice_name` | `str \| None` | optional |
| `max_batch_size` | `int \| None` | `1` |
| `max_edge` | `int \| None` | optional |
| `max_image_length` | `int \| None` | optional |
| `max_num_images` | `int \| None` | optional |
| `max_seq_len` | `int \| None` | `4096` |
| `model_id` | `str` | required |
| `organization` | `IsqOrganization \| None` | optional |
| `quant` | `str \| None` | optional |
| `tokenizer_json` | `str \| None` | optional |
| `topology` | `str \| None` | optional |
| `write_uqff` | `UqffWriteSpec \| None` | optional |


## `ModelSelectedSpeech`

| Field | Type | Default |
| --- | --- | --- |
| `arch` | `SpeechLoaderType` | required |
| `dac_model_id` | `str \| None` | optional |
| `dtype` | `ModelDType \| None` | `ModelDType.AUTO` |
| `generation` | `SpeechGenerationSpec \| None` | optional |
| `model_id` | `str` | required |


## `ModelSpec`

One of several models an engine serves; unset settings fall back to `runtime`'s.

| Field | Type | Default |
| --- | --- | --- |
| `chat_template` | `str \| None` | optional |
| `device_layers` | `list[str] \| None` | optional |
| `encoder_cache_memory_bytes` | `int \| None` | optional |
| `hf_config_overrides` | `dict[str, Any] \| None` | optional |
| `hf_revision` | `str \| None` | optional |
| `isq` | `str \| None` | optional |
| `jinja_explicit` | `str \| None` | optional |
| `max_model_len` | `int \| None` | optional |
| `model` | `ModelSelected` | required |
| `model_id` | `str \| None` | optional |


## `ModelSpeculativeStats`

| Field | Type |
| --- | --- |
| `accepted_per_position` | `list[int]` |
| `draft_tokens_accepted` | `int` |
| `draft_tokens_proposed` | `int` |
| `drafts` | `int` |
| `model_id` | `str` |


## `MtpDraftSampling`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `MtpDraftSampling.AUTO` | `'auto'` |
| `MtpDraftSampling.GREEDY` | `'greedy'` |
| `MtpDraftSampling.PROBABILISTIC` | `'probabilistic'` |


## `MtpSpec`

MTP speculative decoding, drafting with an assistant model or the head built into the checkpoint.

| Field | Type | Default |
| --- | --- | --- |
| `draft_sampling` | `MtpDraftSampling \| None` | optional |
| `model` | `str \| None` | optional |
| `n_predict` | `int \| None` | optional |


## `MultimodalLoaderType`

The architecture to load the multimodal model as.

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `MultimodalLoaderType.PHI3V` | `'phi3v'` |
| `MultimodalLoaderType.IDEFICS2` | `'idefics2'` |
| `MultimodalLoaderType.LLAVA_NEXT` | `'llava_next'` |
| `MultimodalLoaderType.LLAVA` | `'llava'` |
| `MultimodalLoaderType.LFM2VL` | `'lfm2vl'` |
| `MultimodalLoaderType.VLLAMA` | `'vllama'` |
| `MultimodalLoaderType.QWEN2VL` | `'qwen2vl'` |
| `MultimodalLoaderType.IDEFICS3` | `'idefics3'` |
| `MultimodalLoaderType.MINICPMO` | `'minicpmo'` |
| `MultimodalLoaderType.PHI4MM` | `'phi4mm'` |
| `MultimodalLoaderType.QWEN2_5VL` | `'qwen2_5vl'` |
| `MultimodalLoaderType.GEMMA3` | `'gemma3'` |
| `MultimodalLoaderType.MISTRAL3` | `'mistral3'` |
| `MultimodalLoaderType.LLAMA4` | `'llama4'` |
| `MultimodalLoaderType.GEMMA3N` | `'gemma3n'` |
| `MultimodalLoaderType.QWEN3VL` | `'qwen3vl'` |
| `MultimodalLoaderType.QWEN3VLMOE` | `'qwen3vlmoe'` |
| `MultimodalLoaderType.QWEN3_5` | `'qwen3_5'` |
| `MultimodalLoaderType.QWEN3_5MOE` | `'qwen3_5moe'` |
| `MultimodalLoaderType.VOXTRAL` | `'voxtral'` |
| `MultimodalLoaderType.GEMMA4` | `'gemma4'` |
| `MultimodalLoaderType.MUSE_GLIMMER` | `'muse_glimmer'` |
| `MultimodalLoaderType.DIFFUSIONGEMMA` | `'diffusiongemma'` |
| `MultimodalLoaderType.PADDLEOCR_VL` | `'paddleocr_vl'` |


## `NetworkMode`

Network access permitted to sandboxed processes.

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `NetworkMode.NONE` | `'none'` |
| `NetworkMode.LOOPBACK` | `'loopback'` |
| `NetworkMode.FULL` | `'full'` |


## `NormalLoaderType`

The architecture to load the normal model as.

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `NormalLoaderType.MISTRAL` | `'mistral'` |
| `NormalLoaderType.GEMMA` | `'gemma'` |
| `NormalLoaderType.MIXTRAL` | `'mixtral'` |
| `NormalLoaderType.LLAMA` | `'llama'` |
| `NormalLoaderType.PHI2` | `'phi2'` |
| `NormalLoaderType.PHI3` | `'phi3'` |
| `NormalLoaderType.QWEN2` | `'qwen2'` |
| `NormalLoaderType.GEMMA2` | `'gemma2'` |
| `NormalLoaderType.STARCODER2` | `'starcoder2'` |
| `NormalLoaderType.PHI3_5MOE` | `'phi3.5moe'` |
| `NormalLoaderType.DEEPSEEKV2` | `'deepseekv2'` |
| `NormalLoaderType.DEEPSEEKV3` | `'deepseekv3'` |
| `NormalLoaderType.QWEN3` | `'qwen3'` |
| `NormalLoaderType.GLM4` | `'glm4'` |
| `NormalLoaderType.GLM4MOELITE` | `'glm4moelite'` |
| `NormalLoaderType.GLM4MOE` | `'glm4moe'` |
| `NormalLoaderType.QWEN3MOE` | `'qwen3moe'` |
| `NormalLoaderType.SMOLLM3` | `'smollm3'` |
| `NormalLoaderType.GRANITEMOEHYBRID` | `'granitemoehybrid'` |
| `NormalLoaderType.GPT_OSS` | `'gpt_oss'` |
| `NormalLoaderType.HUNYUANV1DENSE` | `'hunyuanv1dense'` |
| `NormalLoaderType.HUNYUANV1MOE` | `'hunyuanv1moe'` |
| `NormalLoaderType.QWEN3NEXT` | `'qwen3next'` |
| `NormalLoaderType.QWEN3_5` | `'qwen3_5'` |
| `NormalLoaderType.QWEN3_5MOE` | `'qwen3_5moe'` |
| `NormalLoaderType.LFM2` | `'lfm2'` |
| `NormalLoaderType.LFM2_MOE` | `'lfm2_moe'` |


## `PagedCacheSpec`

How much the paged-attention KV cache holds; at most one of `context_len`, `memory_mb` and `memory_fraction`.

| Field | Type | Default |
| --- | --- | --- |
| `block_size` | `int \| None` | optional |
| `cache_type` | `PagedCacheType \| None` | optional |
| `context_len` | `int \| None` | optional |
| `memory_fraction` | `float \| None` | optional |
| `memory_mb` | `int \| None` | optional |


## `PagedCacheType`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `PagedCacheType.AUTO` | `'auto'` |
| `PagedCacheType.F8E4M3` | `'f8e4m3'` |


## `RuntimeSpec`

| Field | Type | Default |
| --- | --- | --- |
| `chat_template` | `str \| None` | optional |
| `device` | `str \| None` | optional |
| `device_layers` | `list[str] \| None` | optional |
| `disable_eos_stop` | `bool \| None` | optional |
| `encoder_cache_memory_bytes` | `int \| None` | optional |
| `hf_config_overrides` | `dict[str, Any] \| None` | optional |
| `hf_revision` | `str \| None` | optional |
| `isq` | `str \| None` | optional |
| `jinja_explicit` | `str \| None` | optional |
| `log` | `str \| None` | optional |
| `max_decode_steps_before_prefill` | `int \| None` | optional |
| `max_model_len` | `int \| None` | optional |
| `max_num_batched_tokens` | `int \| None` | optional |
| `max_prefill_chunk_tokens` | `int \| None` | optional |
| `max_seqs` | `int \| None` | optional |
| `mtp` | `MtpSpec \| None` | optional |
| `no_kv_cache` | `bool \| None` | optional |
| `paged_attn` | `bool \| None` | optional |
| `paged_cache` | `PagedCacheSpec \| None` | optional |
| `prefix_cache_n` | `int \| None` | optional |
| `seed` | `int \| None` | optional |
| `throughput_logging` | `bool \| None` | optional |
| `token_source` | `str \| None` | optional |


## `SandboxLimits`

Limits that replace a sandbox profile's; unset ones keep the profile's.

| Field | Type | Default |
| --- | --- | --- |
| `max_cpu_secs` | `int \| None` | optional |
| `max_memory_mb` | `int \| None` | optional |
| `max_procs` | `int \| None` | optional |
| `network` | `NetworkMode \| None` | optional |


## `SandboxMode`

Whether tools that run model-written code are sandboxed.

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `SandboxMode.AUTO` | `'auto'` |
| `SandboxMode.ON` | `'on'` |
| `SandboxMode.OFF` | `'off'` |


## `SandboxPolicy`

Policy applied to a sandboxed process.

| Field | Type | Default |
| --- | --- | --- |
| `extra_env` | `list[str] \| None` | optional |
| `extra_fs_read` | `list[str] \| None` | optional |
| `extra_fs_write` | `list[str] \| None` | optional |
| `max_cpu_secs` | `int \| None` | `600` |
| `max_file_sz_mb` | `int \| None` | `256` |
| `max_memory_mb` | `int \| None` | `2048` |
| `max_open_fds` | `int \| None` | `1024` |
| `max_procs` | `int \| None` | `64` |
| `network` | `NetworkMode \| None` | `NetworkMode.LOOPBACK` |
| `strict` | `bool \| None` | `False` |


## `SandboxProfile`

The starting policy for sandboxed tools: `restricted` or `developer` (toolchain paths, full network by default).

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `SandboxProfile.RESTRICTED` | `'restricted'` |
| `SandboxProfile.DEVELOPER` | `'developer'` |


## `SearchEmbeddingModel`

Embedding model used for ranking web search results internally.

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `SearchEmbeddingModel.EMBEDDING_GEMMA` | `'embedding_gemma'` |


## `SearchSpec`

| Field | Type | Default |
| --- | --- | --- |
| `embedding_model` | `SearchEmbeddingModel \| None` | optional |


## `ShellConfig`

Shell execution config.

| Field | Type | Default |
| --- | --- | --- |
| `permission` | `AgentPermission \| None` | optional |
| `sandbox_policy` | `SandboxPolicy \| None` | optional |
| `shell_path` | `str \| None` | optional |
| `timeout_secs` | `int \| None` | optional |
| `working_directory` | `str \| None` | optional |


## `SkillsSpec`

Where uploaded skills are kept; requests reference them from the shell tool.

| Field | Type | Default |
| --- | --- | --- |
| `root` | `str \| None` | optional |


## `SpeechLoaderType`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `SpeechLoaderType.DIA` | `'dia'` |


## `UqffWriteSpec`

One of: `Union[str, UqffWriteSpecConfig]`.


## `UqffWriteSpecConfig`

| Field | Type | Default |
| --- | --- | --- |
| `base_model` | `str \| None` | optional |
| `output` | `str` | required |
| `repo_id` | `str \| None` | optional |
| `types` | `list[str] \| None` | optional |

---

<small>Generated from [`bindings/python/inference_rs`](https://github.com/christopherthompson81/inference.rs/blob/master/bindings/python/inference_rs).</small>

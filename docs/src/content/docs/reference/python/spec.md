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
| `max_tool_rounds` | `int \| None` | optional |
| `tool_dispatch_url` | `str \| None` | optional |


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
| `model` | `ModelSelected` | required |
| `model_id` | `str \| None` | optional |
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

One of: `Union[ModelSelectedRun, ModelSelectedPlain, ModelSelectedXLora, ModelSelectedLora, ModelSelectedGGUF, ModelSelectedXLoraGGUF, ModelSelectedLoraGGUF, ModelSelectedGGML, ModelSelectedXLoraGGML, ModelSelectedLoraGGML, ModelSelectedMultimodalPlain, ModelSelectedDiffusionPlain, ModelSelectedSpeech, ModelSelectedMultiModel, ModelSelectedEmbedding]`.


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
| `organization` | `IsqOrganization \| None` | optional |
| `quantized_filename` | `str` | required |
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
| `model_id` | `str` | required |
| `organization` | `IsqOrganization \| None` | optional |
| `runtime_config` | `LoraRuntimeConfig \| None` | optional |
| `tokenizer_json` | `str \| None` | optional |
| `topology` | `str \| None` | optional |
| `write_uqff` | `UqffWriteSpec \| None` | optional |


## `ModelSelectedLoraGGML`

Select a GGML model with LoRA.

| Field | Type | Default |
| --- | --- | --- |
| `adapters_model_id` | `str` | required |
| `dtype` | `ModelDType \| None` | `ModelDType.AUTO` |
| `gqa` | `int` | required |
| `max_batch_size` | `int \| None` | `1` |
| `max_seq_len` | `int \| None` | `4096` |
| `order` | `str` | required |
| `quantized_filename` | `str` | required |
| `quantized_model_id` | `str` | required |
| `tok_model_id` | `str \| None` | optional |
| `tokenizer_json` | `str \| None` | optional |
| `topology` | `str \| None` | optional |


## `ModelSelectedLoraGGUF`

Select a GGUF model with LoRA.

| Field | Type | Default |
| --- | --- | --- |
| `adapters_model_id` | `str` | required |
| `calibration_file` | `str \| None` | optional |
| `dtype` | `ModelDType \| None` | `ModelDType.AUTO` |
| `hf_cache_path` | `str \| None` | optional |
| `imatrix` | `str \| None` | optional |
| `matformer_config_path` | `str \| None` | optional |
| `matformer_slice_name` | `str \| None` | optional |
| `max_batch_size` | `int \| None` | `1` |
| `max_seq_len` | `int \| None` | `4096` |
| `order` | `str` | required |
| `organization` | `IsqOrganization \| None` | optional |
| `quantized_filename` | `str` | required |
| `quantized_model_id` | `str` | required |
| `tok_model_id` | `str \| None` | optional |
| `tokenizer_json` | `str \| None` | optional |
| `topology` | `str \| None` | optional |
| `write_uqff` | `UqffWriteSpec \| None` | optional |


## `ModelSelectedMultiModel`

Select multi-model mode with configuration file

| Field | Type | Default |
| --- | --- | --- |
| `config` | `str` | required |
| `default_model_id` | `str \| None` | optional |


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
| `tokenizer_json` | `str \| None` | optional |
| `topology` | `str \| None` | optional |
| `write_uqff` | `UqffWriteSpec \| None` | optional |


## `ModelSelectedSpeech`

| Field | Type | Default |
| --- | --- | --- |
| `arch` | `SpeechLoaderType` | required |
| `dac_model_id` | `str \| None` | optional |
| `dtype` | `ModelDType \| None` | `ModelDType.AUTO` |
| `model_id` | `str` | required |


## `ModelSelectedXLora`

Select an X-LoRA architecture

| Field | Type | Default |
| --- | --- | --- |
| `arch` | `NormalLoaderType \| None` | optional |
| `dtype` | `ModelDType \| None` | `ModelDType.AUTO` |
| `from_uqff` | `str \| None` | optional |
| `hf_cache_path` | `str \| None` | optional |
| `max_batch_size` | `int \| None` | `1` |
| `max_seq_len` | `int \| None` | `4096` |
| `model_id` | `str \| None` | optional |
| `order` | `str` | required |
| `organization` | `IsqOrganization \| None` | optional |
| `tgt_non_granular_index` | `int \| None` | optional |
| `tokenizer_json` | `str \| None` | optional |
| `topology` | `str \| None` | optional |
| `write_uqff` | `UqffWriteSpec \| None` | optional |
| `xlora_model_id` | `str` | required |


## `ModelSelectedXLoraGGML`

Select a GGML model with X-LoRA.

| Field | Type | Default |
| --- | --- | --- |
| `dtype` | `ModelDType \| None` | `ModelDType.AUTO` |
| `gqa` | `int` | required |
| `max_batch_size` | `int \| None` | `1` |
| `max_seq_len` | `int \| None` | `4096` |
| `order` | `str` | required |
| `quantized_filename` | `str` | required |
| `quantized_model_id` | `str` | required |
| `tgt_non_granular_index` | `int \| None` | optional |
| `tok_model_id` | `str \| None` | optional |
| `tokenizer_json` | `str \| None` | optional |
| `topology` | `str \| None` | optional |
| `xlora_model_id` | `str` | required |


## `ModelSelectedXLoraGGUF`

Select a GGUF model with X-LoRA.

| Field | Type | Default |
| --- | --- | --- |
| `calibration_file` | `str \| None` | optional |
| `dtype` | `ModelDType \| None` | `ModelDType.AUTO` |
| `hf_cache_path` | `str \| None` | optional |
| `imatrix` | `str \| None` | optional |
| `matformer_config_path` | `str \| None` | optional |
| `matformer_slice_name` | `str \| None` | optional |
| `max_batch_size` | `int \| None` | `1` |
| `max_seq_len` | `int \| None` | `4096` |
| `order` | `str` | required |
| `organization` | `IsqOrganization \| None` | optional |
| `quantized_filename` | `str` | required |
| `quantized_model_id` | `str` | required |
| `tgt_non_granular_index` | `int \| None` | optional |
| `tok_model_id` | `str \| None` | optional |
| `tokenizer_json` | `str \| None` | optional |
| `topology` | `str \| None` | optional |
| `write_uqff` | `UqffWriteSpec \| None` | optional |
| `xlora_model_id` | `str` | required |


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
| `NormalLoaderType.LFM2` | `'lfm2'` |
| `NormalLoaderType.LFM2_MOE` | `'lfm2_moe'` |


## `RuntimeSpec`

| Field | Type | Default |
| --- | --- | --- |
| `chat_template` | `str \| None` | optional |
| `device` | `str \| None` | optional |
| `isq` | `str \| None` | optional |
| `jinja_explicit` | `str \| None` | optional |
| `max_model_len` | `int \| None` | optional |
| `max_seqs` | `int \| None` | optional |
| `no_kv_cache` | `bool \| None` | optional |
| `paged_attn` | `bool \| None` | optional |
| `prefix_cache_n` | `int \| None` | optional |
| `seed` | `int \| None` | optional |
| `token_source` | `str \| None` | optional |


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

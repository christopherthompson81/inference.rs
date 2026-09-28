---
title: Chat and completions
description: "Chat completion, completion and embedding requests and responses, tools and output formats."
sidebar:
  order: 4
---
## `AdapterGenerationSelection`

An exact immutable LoRA adapter generation.

| Field | Type |
| --- | --- |
| `generation` | `str` |


## `AdapterSelection`

One of: `Union[str, AdapterGenerationSelection]`.


## `AllowedToolChoice`

One of: `Union[AllowedToolChoiceFunction, AllowedToolChoiceWebSearchPreview, AllowedToolChoiceCodeInterpreter, AllowedToolChoiceShell]`.


## `AllowedToolChoiceCodeInterpreter`

| Field | Type | Default |
| --- | --- | --- |
| `type` | `Literal['code_interpreter']` | `'code_interpreter'` |


## `AllowedToolChoiceFunction`

| Field | Type | Default |
| --- | --- | --- |
| `name` | `str` | required |
| `type` | `Literal['function']` | `'function'` |


## `AllowedToolChoiceShell`

| Field | Type | Default |
| --- | --- | --- |
| `type` | `Literal['shell']` | `'shell'` |


## `AllowedToolChoiceWebSearchPreview`

| Field | Type | Default |
| --- | --- | --- |
| `type` | `Literal['web_search_preview']` | `'web_search_preview'` |


## `AllowedToolsMode`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `AllowedToolsMode.AUTO` | `'auto'` |
| `AllowedToolsMode.REQUIRED` | `'required'` |


## `AllowedToolsToolChoice`

| Field | Type | Default |
| --- | --- | --- |
| `mode` | `AllowedToolsMode` | required |
| `tools` | `list[AllowedToolChoice]` | required |
| `type` | `AllowedToolsToolChoiceType` | `'allowed_tools'` |


## `AllowedToolsToolChoiceType`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `AllowedToolsToolChoiceType.ALLOWED_TOOLS` | `'allowed_tools'` |


## `ApproximateUserLocation`

| Field | Type | Default |
| --- | --- | --- |
| `city` | `str \| None` | optional |
| `country` | `str \| None` | optional |
| `region` | `str \| None` | optional |
| `timezone` | `str \| None` | optional |


## `BuiltinToolChoice`

| Field | Type |
| --- | --- |
| `type` | `BuiltinToolChoiceType` |


## `BuiltinToolChoiceType`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `BuiltinToolChoiceType.WEB_SEARCH_PREVIEW` | `'web_search_preview'` |
| `BuiltinToolChoiceType.CODE_INTERPRETER` | `'code_interpreter'` |
| `BuiltinToolChoiceType.SHELL` | `'shell'` |


## `ChatCompletionChunkChoice`

| Field | Type | Default |
| --- | --- | --- |
| `delta` | `ChatCompletionChunkDelta` | required |
| `finish_reason` | `str \| None` | optional |
| `index` | `int` | required |
| `logprobs` | `Any` | optional |


## `ChatCompletionChunkDelta`

| Field | Type | Default |
| --- | --- | --- |
| `content` | `str \| None` | optional |
| `reasoning_content` | `str \| None` | optional |
| `role` | `str` | required |
| `tool_calls` | `list[Any] \| None` | optional |


## `ChatCompletionChunkResponse`

| Field | Type | Default |
| --- | --- | --- |
| `adapter_generation` | `str \| None` | optional |
| `choices` | `list[ChatCompletionChunkChoice]` | required |
| `created` | `int` | required |
| `id` | `str` | required |
| `model` | `str` | required |
| `object` | `str` | required |
| `session_id` | `str \| None` | optional |
| `system_fingerprint` | `str` | required |
| `usage` | `CompletionUsageResponse \| None` | optional |


## `ChatCompletionRequest`

Chat completion request following OpenAI's specification

| Field | Type | Default |
| --- | --- | --- |
| `adapter` | `AdapterSelection \| None` | optional |
| `agent_permission` | `str \| None` | optional |
| `chat_template_kwargs` | `dict[str, Any] \| None` | optional |
| `code_execution_permission` | `str \| None` | optional |
| `dry_allowed_length` | `int \| None` | optional |
| `dry_base` | `float \| None` | optional |
| `dry_multiplier` | `float \| None` | optional |
| `dry_sequence_breakers` | `list[str] \| None` | optional |
| `enable_shell` | `bool \| None` | optional |
| `enable_thinking` | `bool \| None` | optional |
| `files` | `list[Any] \| None` | optional |
| `frequency_penalty` | `float \| None` | optional |
| `grammar` | `Grammar \| None` | optional |
| `ignore_eos` | `bool \| None` | optional |
| `logit_bias` | `dict[str, float] \| None` | optional |
| `logprobs` | `bool \| None` | optional |
| `max_tokens` | `int \| None` | optional |
| `max_tool_rounds` | `int \| None` | optional |
| `messages` | `Union[list[Message], str]` | required |
| `min_p` | `float \| None` | optional |
| `model` | `str \| None` | optional |
| `n` | `int \| None` | optional |
| `presence_penalty` | `float \| None` | optional |
| `reasoning_effort` | `ReasoningEffort \| None` | optional |
| `repetition_penalty` | `float \| None` | optional |
| `response_format` | `ResponseFormat \| None` | optional |
| `seed` | `int \| None` | optional |
| `session_id` | `str \| None` | optional |
| `stop` | `StopTokens \| None` | optional |
| `stream` | `bool \| None` | optional |
| `temperature` | `float \| None` | optional |
| `tool_choice` | `ToolChoice \| None` | optional |
| `tools` | `list[OpenAiTool] \| None` | optional |
| `top_k` | `int \| None` | optional |
| `top_logprobs` | `int \| None` | optional |
| `top_p` | `float \| None` | optional |
| `truncate_sequence` | `bool \| None` | optional |
| `web_search_options` | `WebSearchOptions \| None` | optional |


## `ChatCompletionResponse`

| Field | Type | Default |
| --- | --- | --- |
| `adapter_generation` | `str \| None` | optional |
| `agentic_tool_calls` | `list[Any] \| None` | optional |
| `choices` | `list[ChatCompletionResponseChoice]` | required |
| `created` | `int` | required |
| `files` | `list[Any] \| None` | optional |
| `id` | `str` | required |
| `model` | `str` | required |
| `object` | `str` | required |
| `session_id` | `str \| None` | optional |
| `system_fingerprint` | `str` | required |
| `usage` | `CompletionUsageResponse` | required |


## `ChatCompletionResponseChoice`

| Field | Type | Default |
| --- | --- | --- |
| `finish_reason` | `str` | required |
| `index` | `int` | required |
| `logprobs` | `Any` | optional |
| `message` | `ChatCompletionResponseMessage` | required |


## `ChatCompletionResponseMessage`

| Field | Type | Default |
| --- | --- | --- |
| `content` | `str \| None` | optional |
| `reasoning_content` | `str \| None` | optional |
| `role` | `str` | required |
| `tool_calls` | `list[Any] \| None` | optional |


## `CompletionChunkChoice`

| Field | Type | Default |
| --- | --- | --- |
| `finish_reason` | `str \| None` | optional |
| `index` | `int` | required |
| `logprobs` | `Any` | optional |
| `text` | `str` | required |


## `CompletionChunkResponse`

| Field | Type | Default |
| --- | --- | --- |
| `adapter_generation` | `str \| None` | optional |
| `choices` | `list[CompletionChunkChoice]` | required |
| `created` | `int` | required |
| `id` | `str` | required |
| `model` | `str` | required |
| `object` | `str` | required |
| `system_fingerprint` | `str` | required |


## `CompletionRequest`

Legacy OpenAI compatible text completion request

| Field | Type | Default |
| --- | --- | --- |
| `adapter` | `AdapterSelection \| None` | optional |
| `best_of` | `int \| None` | optional |
| `dry_allowed_length` | `int \| None` | optional |
| `dry_base` | `float \| None` | optional |
| `dry_multiplier` | `float \| None` | optional |
| `dry_sequence_breakers` | `list[str] \| None` | optional |
| `echo` | `bool \| None` | optional |
| `frequency_penalty` | `float \| None` | optional |
| `grammar` | `Grammar \| None` | optional |
| `ignore_eos` | `bool \| None` | optional |
| `logit_bias` | `dict[str, float] \| None` | optional |
| `logprobs` | `int \| None` | optional |
| `max_tokens` | `int \| None` | optional |
| `min_p` | `float \| None` | optional |
| `model` | `str \| None` | optional |
| `n` | `int \| None` | optional |
| `presence_penalty` | `float \| None` | optional |
| `prompt` | `str` | required |
| `repetition_penalty` | `float \| None` | optional |
| `seed` | `int \| None` | optional |
| `stop` | `StopTokens \| None` | optional |
| `stream` | `bool \| None` | optional |
| `suffix` | `str \| None` | optional |
| `temperature` | `float \| None` | optional |
| `tool_choice` | `ToolChoice \| None` | optional |
| `tools` | `list[Tool] \| None` | optional |
| `top_k` | `int \| None` | optional |
| `top_p` | `float \| None` | optional |
| `truncate_sequence` | `bool \| None` | optional |
| `user` | `str \| None` | optional |


## `CompletionResponse`

| Field | Type | Default |
| --- | --- | --- |
| `adapter_generation` | `str \| None` | optional |
| `choices` | `list[CompletionResponseChoice]` | required |
| `created` | `int` | required |
| `id` | `str` | required |
| `model` | `str` | required |
| `object` | `str` | required |
| `system_fingerprint` | `str` | required |
| `usage` | `CompletionUsageResponse` | required |


## `CompletionResponseChoice`

| Field | Type | Default |
| --- | --- | --- |
| `finish_reason` | `str` | required |
| `index` | `int` | required |
| `logprobs` | `Any` | optional |
| `text` | `str` | required |


## `CompletionUsageResponse`

| Field | Type | Default |
| --- | --- | --- |
| `avg_compl_tok_per_sec` | `float` | required |
| `avg_prompt_tok_per_sec` | `float` | required |
| `avg_tok_per_sec` | `float` | required |
| `completion_tokens` | `int` | required |
| `prompt_tokens` | `int` | required |
| `prompt_tokens_details` | `PromptTokensDetailsResponse \| None` | optional |
| `total_completion_time_sec` | `float` | required |
| `total_prompt_time_sec` | `float` | required |
| `total_time_sec` | `float` | required |
| `total_tokens` | `int` | required |


## `EmbeddingData`

| Field | Type |
| --- | --- |
| `embedding` | `EmbeddingVector` |
| `index` | `int` |
| `object` | `str` |


## `EmbeddingEncodingFormat`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `EmbeddingEncodingFormat.FLOAT` | `'float'` |
| `EmbeddingEncodingFormat.BASE64` | `'base64'` |


## `EmbeddingInput`

One of: `Union[str, list[str], list[int], list[list[int]]]`.


## `EmbeddingRequest`

| Field | Type | Default |
| --- | --- | --- |
| `dimensions` | `int \| None` | optional |
| `encoding_format` | `EmbeddingEncodingFormat \| None` | optional |
| `input` | `EmbeddingInput` | required |
| `model` | `str \| None` | optional |
| `truncate_sequence` | `bool \| None` | optional |
| `user` | `str \| None` | optional |


## `EmbeddingResponse`

| Field | Type |
| --- | --- |
| `data` | `list[EmbeddingData]` |
| `model` | `str` |
| `object` | `str` |
| `usage` | `EmbeddingUsage` |


## `EmbeddingUsage`

| Field | Type |
| --- | --- |
| `prompt_tokens` | `int` |
| `total_tokens` | `int` |


## `EmbeddingVector`

One of: `Union[list[float], str]`.


## `Function`

Function definition for a tool

| Field | Type | Default |
| --- | --- | --- |
| `description` | `str \| None` | optional |
| `name` | `str` | required |
| `parameters` | `dict[str, Any] \| None` | optional |
| `strict` | `bool \| None` | optional |


## `FunctionCalled`

Represents a function call made by the assistant

| Field | Type |
| --- | --- |
| `arguments` | `str` |
| `name` | `str` |


## `Grammar`

One of: `Union[GrammarRegex, GrammarJsonSchema, GrammarLlguidance, GrammarLark]`.


## `GrammarJsonSchema`

| Field | Type | Default |
| --- | --- | --- |
| `type` | `Literal['json_schema']` | `'json_schema'` |
| `value` | `dict[str, Any]` | required |


## `GrammarLark`

| Field | Type | Default |
| --- | --- | --- |
| `type` | `Literal['lark']` | `'lark'` |
| `value` | `str` | required |


## `GrammarLlguidance`

| Field | Type | Default |
| --- | --- | --- |
| `type` | `Literal['llguidance']` | `'llguidance'` |
| `value` | `GrammarLlguidanceValue` | required |


## `GrammarLlguidanceValue`

Top-level grammar configuration for LLGuidance

| Field | Type | Default |
| --- | --- | --- |
| `grammars` | `list[GrammarLlguidanceValueGrammars]` | required |
| `max_tokens` | `int \| None` | optional |


## `GrammarLlguidanceValueGrammars`

Grammar configuration with lexer settings

| Field | Type | Default |
| --- | --- | --- |
| `json_schema` | `dict[str, Any] \| None` | optional |
| `lark_grammar` | `str \| None` | optional |
| `name` | `str \| None` | optional |


## `GrammarRegex`

| Field | Type | Default |
| --- | --- | --- |
| `type` | `Literal['regex']` | `'regex'` |
| `value` | `str` | required |


## `JsonSchemaResponseFormat`

JSON Schema for structured responses

| Field | Type |
| --- | --- |
| `name` | `str` |
| `schema` | `Any` |


## `Message`

Represents a single message in a conversation

| Field | Type | Default |
| --- | --- | --- |
| `content` | `MessageContent \| None` | optional |
| `name` | `str \| None` | optional |
| `reasoning_content` | `str \| None` | optional |
| `role` | `str` | required |
| `tool_call_id` | `str \| None` | optional |
| `tool_calls` | `list[ToolCall] \| None` | optional |


## `MessageContent`

One of: `Union[str, list[dict[str, MessageInnerContent]]]`.


## `MessageInnerContent`

One of: `Union[str, dict[str, str]]`.


## `NamedFunctionToolChoice`

| Field | Type | Default |
| --- | --- | --- |
| `name` | `str` | required |
| `type` | `ToolType` | `'function'` |


## `OpenAiCodeInterpreterAutoContainer`

| Field | Type | Default |
| --- | --- | --- |
| `file_ids` | `list[str] \| None` | optional |
| `memory_limit` | `str \| None` | optional |
| `type` | `OpenAiCodeInterpreterContainerType` | `'auto'` |


## `OpenAiCodeInterpreterContainer`

One of: `Union[str, OpenAiCodeInterpreterAutoContainer]`.


## `OpenAiCodeInterpreterContainerType`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `OpenAiCodeInterpreterContainerType.AUTO` | `'auto'` |


## `OpenAiCodeInterpreterTool`

| Field | Type | Default |
| --- | --- | --- |
| `container` | `OpenAiCodeInterpreterContainer` | required |
| `type` | `OpenAiCodeInterpreterToolType` | `'code_interpreter'` |


## `OpenAiCodeInterpreterToolType`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `OpenAiCodeInterpreterToolType.CODE_INTERPRETER` | `'code_interpreter'` |


## `OpenAiFunctionToolType`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `OpenAiFunctionToolType.FUNCTION` | `'function'` |


## `OpenAiNamespaceEntry`

One of: `Union[OpenAiResponsesFunctionTool, Any]`.


## `OpenAiNamespaceTool`

A grouping of client-executed tools; the model sees each inner tool as `<namespace>.<name>`.

| Field | Type | Default |
| --- | --- | --- |
| `description` | `str \| None` | optional |
| `name` | `str` | required |
| `tools` | `list[OpenAiNamespaceEntry] \| None` | optional |
| `type` | `OpenAiNamespaceToolType` | `'namespace'` |


## `OpenAiNamespaceToolType`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `OpenAiNamespaceToolType.NAMESPACE` | `'namespace'` |


## `OpenAiResponsesFunctionTool`

| Field | Type | Default |
| --- | --- | --- |
| `description` | `str \| None` | optional |
| `name` | `str` | required |
| `parameters` | `dict[str, Any] \| None` | optional |
| `strict` | `bool \| None` | optional |
| `type` | `OpenAiFunctionToolType` | `'function'` |


## `OpenAiShellEnvironment`

One of: `Union[OpenAiShellEnvironmentContainerAuto, OpenAiShellEnvironmentLocal, OpenAiShellEnvironmentContainerReference]`.


## `OpenAiShellEnvironmentContainerAuto`

| Field | Type | Default |
| --- | --- | --- |
| `skills` | `list[OpenAiShellSkill] \| None` | optional |
| `type` | `Literal['container_auto']` | `'container_auto'` |


## `OpenAiShellEnvironmentContainerReference`

| Field | Type | Default |
| --- | --- | --- |
| `container_id` | `str` | required |
| `type` | `Literal['container_reference']` | `'container_reference'` |


## `OpenAiShellEnvironmentLocal`

| Field | Type | Default |
| --- | --- | --- |
| `path` | `str` | required |
| `type` | `Literal['local']` | `'local'` |


## `OpenAiShellSkill`

One of: `Union[OpenAiShellSkillSkillReference, OpenAiShellSkillLocal]`.


## `OpenAiShellSkillLocal`

| Field | Type | Default |
| --- | --- | --- |
| `path` | `str` | required |
| `type` | `Literal['local']` | `'local'` |


## `OpenAiShellSkillSkillReference`

| Field | Type | Default |
| --- | --- | --- |
| `skill_id` | `str` | required |
| `type` | `Literal['skill_reference']` | `'skill_reference'` |
| `version` | `Any` | optional |


## `OpenAiShellTool`

| Field | Type | Default |
| --- | --- | --- |
| `environment` | `OpenAiShellEnvironment` | required |
| `type` | `OpenAiShellToolType` | `'shell'` |


## `OpenAiShellToolType`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `OpenAiShellToolType.SHELL` | `'shell'` |


## `OpenAiTool`

One of: `Union[Tool, OpenAiResponsesFunctionTool, OpenAiWebSearchTool, OpenAiCodeInterpreterTool, OpenAiShellTool, OpenAiNamespaceTool]`.


## `OpenAiWebSearchTool`

| Field | Type | Default |
| --- | --- | --- |
| `external_web_access` | `bool \| None` | optional |
| `filters` | `WebSearchFilters \| None` | optional |
| `image_settings` | `WebSearchImageSettings \| None` | optional |
| `return_token_budget` | `WebSearchReturnTokenBudget \| None` | optional |
| `search_content_types` | `list[WebSearchContentType] \| None` | optional |
| `search_context_size` | `SearchContextSize \| None` | optional |
| `type` | `OpenAiWebSearchToolType` | required |
| `user_location` | `OpenAiWebSearchUserLocation \| None` | optional |


## `OpenAiWebSearchToolType`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `OpenAiWebSearchToolType.WEB_SEARCH` | `'web_search'` |
| `OpenAiWebSearchToolType.WEB_SEARCH_PREVIEW` | `'web_search_preview'` |


## `OpenAiWebSearchUserLocationApproximate`

| Field | Type | Default |
| --- | --- | --- |
| `city` | `str \| None` | optional |
| `country` | `str \| None` | optional |
| `region` | `str \| None` | optional |
| `timezone` | `str \| None` | optional |
| `type` | `Literal['approximate']` | `'approximate'` |


## `PromptTokensDetailsResponse`

| Field | Type |
| --- | --- |
| `cached_tokens` | `int` |


## `ReasoningEffort`

Reasoning effort. `none` aliases `off`; `max` aliases `xhigh`.

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `ReasoningEffort.OFF` | `'off'` |
| `ReasoningEffort.NONE` | `'none'` |
| `ReasoningEffort.LOW` | `'low'` |
| `ReasoningEffort.MEDIUM` | `'medium'` |
| `ReasoningEffort.HIGH` | `'high'` |
| `ReasoningEffort.XHIGH` | `'xhigh'` |
| `ReasoningEffort.MAX` | `'max'` |


## `ResponseFormat`

One of: `Union[ResponseFormatText, ResponseFormatJsonObject, ResponseFormatJsonSchema]`.


## `ResponseFormatJsonObject`

Structured response as any JSON object

| Field | Type | Default |
| --- | --- | --- |
| `type` | `Literal['json_object']` | `'json_object'` |


## `ResponseFormatJsonSchema`

Structured response following a JSON schema

| Field | Type | Default |
| --- | --- | --- |
| `json_schema` | `JsonSchemaResponseFormat` | required |
| `type` | `Literal['json_schema']` | `'json_schema'` |


## `ResponseFormatText`

Free-form text response

| Field | Type | Default |
| --- | --- | --- |
| `type` | `Literal['text']` | `'text'` |


## `SearchContextSize`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `SearchContextSize.LOW` | `'low'` |
| `SearchContextSize.MEDIUM` | `'medium'` |
| `SearchContextSize.HIGH` | `'high'` |


## `SerializedVideo`

| Field | Type |
| --- | --- |
| `fps` | `float` |
| `frames` | `list[str]` |
| `sampled_indices` | `list[int]` |
| `total_num_frames` | `int` |


## `StopTokens`

One of: `Union[list[str], str]`.


## `Tool`

Tool definition

| Field | Type | Default |
| --- | --- | --- |
| `function` | `Function` | required |
| `type` | `ToolType` | `'function'` |


## `ToolCall`

Represents a tool call made by the assistant

| Field | Type | Default |
| --- | --- | --- |
| `function` | `FunctionCalled` | required |
| `id` | `str \| None` | optional |
| `type` | `ToolType` | `'function'` |


## `ToolChoice`

One of: `Union[Literal['none'], Literal['auto'], Literal['required'], AllowedToolsToolChoice, BuiltinToolChoice, Tool, NamedFunctionToolChoice]`.


## `ToolType`

Type of tool

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `ToolType.FUNCTION` | `'function'` |


## `WebSearchContentType`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `WebSearchContentType.TEXT` | `'text'` |
| `WebSearchContentType.IMAGE` | `'image'` |


## `WebSearchFilters`

| Field | Type | Default |
| --- | --- | --- |
| `allowed_domains` | `list[str] \| None` | optional |
| `blocked_domains` | `list[str] \| None` | optional |


## `WebSearchImageSettings`

| Field | Type | Default |
| --- | --- | --- |
| `caption` | `bool \| None` | optional |
| `max_results` | `int \| None` | optional |


## `WebSearchOptions`

| Field | Type | Default |
| --- | --- | --- |
| `external_web_access` | `bool \| None` | optional |
| `extract_description` | `str \| None` | optional |
| `filters` | `WebSearchFilters \| None` | optional |
| `image_settings` | `WebSearchImageSettings \| None` | optional |
| `return_token_budget` | `WebSearchReturnTokenBudget \| None` | optional |
| `search_content_types` | `list[WebSearchContentType] \| None` | optional |
| `search_context_size` | `SearchContextSize \| None` | optional |
| `search_description` | `str \| None` | optional |
| `user_location` | `WebSearchUserLocation \| None` | optional |


## `WebSearchReturnTokenBudget`

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `WebSearchReturnTokenBudget.DEFAULT` | `'default'` |
| `WebSearchReturnTokenBudget.UNLIMITED` | `'unlimited'` |


## `WebSearchUserLocationApproximate`

| Field | Type | Default |
| --- | --- | --- |
| `approximate` | `ApproximateUserLocation` | required |
| `type` | `Literal['approximate']` | `'approximate'` |

---

<small>Generated from [`bindings/python/inference_rs`](https://github.com/christopherthompson81/inference.rs/blob/master/bindings/python/inference_rs).</small>

---
title: Anthropic
description: "Anthropic Messages requests, responses and skill listings."
sidebar:
  order: 6
---
## `AnthropicContainer`

| Field | Type | Default |
| --- | --- | --- |
| `skills` | `list[AnthropicSkillReference] \| None` | optional |


## `AnthropicContentBlock`

| Field | Type | Default |
| --- | --- | --- |
| `cache_control` | `dict[str, Any] \| None` | optional |
| `citations` | `dict[str, Any] \| None` | optional |
| `content` | `dict[str, Any] \| None` | optional |
| `id` | `str \| None` | optional |
| `input` | `dict[str, Any] \| None` | optional |
| `is_error` | `bool \| None` | optional |
| `name` | `str \| None` | optional |
| `signature` | `str \| None` | optional |
| `source` | `AnthropicImageSource \| None` | optional |
| `text` | `str \| None` | optional |
| `thinking` | `str \| None` | optional |
| `tool_use_id` | `str \| None` | optional |
| `type` | `str` | required |


## `AnthropicCountTokensResponse`

| Field | Type |
| --- | --- |
| `input_tokens` | `int` |


## `AnthropicError`

| Field | Type | Default |
| --- | --- | --- |
| `error` | `AnthropicErrorBody` | required |
| `type` | `str \| None` | optional |


## `AnthropicErrorBody`

| Field | Type |
| --- | --- |
| `message` | `str` |
| `type` | `str` |


## `AnthropicImageSource`

| Field | Type | Default |
| --- | --- | --- |
| `data` | `str \| None` | optional |
| `media_type` | `str \| None` | optional |
| `type` | `str` | required |
| `url` | `str \| None` | optional |


## `AnthropicJsonOutputFormat`

| Field | Type |
| --- | --- |
| `schema` | `dict[str, Any]` |
| `type` | `str` |


## `AnthropicMessage`

| Field | Type |
| --- | --- |
| `content` | `AnthropicMessageContent` |
| `role` | `str` |


## `AnthropicMessageContent`

One of: `Union[str, list[AnthropicContentBlock]]`.


## `AnthropicMessageResponse`

| Field | Type | Default |
| --- | --- | --- |
| `content` | `list[AnthropicResponseContentBlock]` | required |
| `files` | `list[Any] \| None` | optional |
| `id` | `str` | required |
| `model` | `str` | required |
| `role` | `str \| None` | optional |
| `session_id` | `str \| None` | optional |
| `stop_reason` | `str` | required |
| `stop_sequence` | `str \| None` | optional |
| `type` | `str \| None` | optional |
| `usage` | `AnthropicUsage` | required |


## `AnthropicMessagesRequest`

| Field | Type | Default |
| --- | --- | --- |
| `agent_permission` | `str \| None` | optional |
| `code_execution_permission` | `str \| None` | optional |
| `container` | `AnthropicContainer \| None` | optional |
| `dry_allowed_length` | `int \| None` | optional |
| `dry_base` | `float \| None` | optional |
| `dry_multiplier` | `float \| None` | optional |
| `dry_sequence_breakers` | `list[str] \| None` | optional |
| `enable_code_execution` | `bool \| None` | optional |
| `enable_thinking` | `bool \| None` | optional |
| `files` | `list[Any] \| None` | optional |
| `frequency_penalty` | `float \| None` | optional |
| `grammar` | `Grammar \| None` | optional |
| `host_tools` | `list[str] \| None` | optional |
| `logit_bias` | `dict[str, float] \| None` | optional |
| `logits_processors` | `list[str] \| None` | optional |
| `logprobs` | `bool \| None` | optional |
| `max_tokens` | `int \| None` | optional |
| `max_tool_rounds` | `int \| None` | optional |
| `messages` | `list[AnthropicMessage]` | required |
| `metadata` | `dict[str, Any] \| None` | optional |
| `min_p` | `float \| None` | optional |
| `model` | `str \| None` | optional |
| `output_config` | `AnthropicOutputConfig \| None` | optional |
| `presence_penalty` | `float \| None` | optional |
| `reasoning_effort` | `ReasoningEffort \| None` | optional |
| `repetition_penalty` | `float \| None` | optional |
| `response_format` | `ResponseFormat \| None` | optional |
| `session_id` | `str \| None` | optional |
| `stop_sequences` | `list[str] \| None` | optional |
| `stream` | `bool \| None` | optional |
| `system` | `AnthropicSystem \| None` | optional |
| `temperature` | `float \| None` | optional |
| `thinking` | `AnthropicThinking \| None` | optional |
| `tool_choice` | `AnthropicToolChoice \| None` | optional |
| `tools` | `list[AnthropicTool] \| None` | optional |
| `top_k` | `int \| None` | optional |
| `top_logprobs` | `int \| None` | optional |
| `top_p` | `float \| None` | optional |
| `truncate_sequence` | `bool \| None` | optional |
| `web_search_options` | `WebSearchOptions \| None` | optional |


## `AnthropicOutputConfig`

| Field | Type | Default |
| --- | --- | --- |
| `effort` | `str \| None` | optional |
| `format` | `AnthropicJsonOutputFormat \| None` | optional |


## `AnthropicResponseContentBlock`

| Field | Type | Default |
| --- | --- | --- |
| `id` | `str \| None` | optional |
| `input` | `dict[str, Any] \| None` | optional |
| `name` | `str \| None` | optional |
| `signature` | `str \| None` | optional |
| `text` | `str \| None` | optional |
| `thinking` | `str \| None` | optional |
| `type` | `str` | required |


## `AnthropicSkillListObject`

| Field | Type | Default |
| --- | --- | --- |
| `data` | `list[AnthropicSkillObject]` | required |
| `has_more` | `bool` | required |
| `next_page` | `str \| None` | optional |


## `AnthropicSkillObject`

| Field | Type |
| --- | --- |
| `created_at` | `str` |
| `display_title` | `str` |
| `id` | `str` |
| `latest_version` | `str` |
| `source` | `str` |
| `type` | `str` |
| `updated_at` | `str` |


## `AnthropicSkillReference`

| Field | Type | Default |
| --- | --- | --- |
| `skill_id` | `str` | required |
| `type` | `str` | required |
| `version` | `Any` | optional |


## `AnthropicSkillVersionListObject`

| Field | Type | Default |
| --- | --- | --- |
| `data` | `list[AnthropicSkillVersionObject]` | required |
| `has_more` | `bool` | required |
| `next_page` | `str \| None` | optional |


## `AnthropicSkillVersionObject`

| Field | Type |
| --- | --- |
| `created_at` | `str` |
| `description` | `str` |
| `directory` | `str` |
| `id` | `str` |
| `name` | `str` |
| `skill_id` | `str` |
| `type` | `str` |
| `version` | `str` |


## `AnthropicSystem`

One of: `Union[str, list[AnthropicContentBlock]]`.


## `AnthropicThinking`

| Field | Type | Default |
| --- | --- | --- |
| `budget_tokens` | `int \| None` | optional |
| `display` | `str \| None` | optional |
| `type` | `str` | required |


## `AnthropicTool`

| Field | Type | Default |
| --- | --- | --- |
| `allowed_domains` | `list[str] \| None` | optional |
| `blocked_domains` | `list[str] \| None` | optional |
| `description` | `str \| None` | optional |
| `input_schema` | `dict[str, Any] \| None` | optional |
| `max_uses` | `int \| None` | optional |
| `name` | `str \| None` | optional |
| `type` | `str \| None` | optional |
| `user_location` | `AnthropicWebSearchUserLocation \| None` | optional |


## `AnthropicToolChoice`

| Field | Type | Default |
| --- | --- | --- |
| `name` | `str \| None` | optional |
| `type` | `str` | required |


## `AnthropicUsage`

| Field | Type |
| --- | --- |
| `cache_creation_input_tokens` | `int` |
| `cache_read_input_tokens` | `int` |
| `input_tokens` | `int` |
| `output_tokens` | `int` |


## `AnthropicWebSearchUserLocation`

| Field | Type | Default |
| --- | --- | --- |
| `city` | `str \| None` | optional |
| `country` | `str \| None` | optional |
| `region` | `str \| None` | optional |
| `timezone` | `str \| None` | optional |
| `type` | `str` | required |

---

<small>Generated from [`bindings/python/inference_rs`](https://github.com/christopherthompson81/inference.rs/blob/master/bindings/python/inference_rs).</small>

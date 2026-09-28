---
title: Responses
description: "OpenResponses requests, resources and stream events."
sidebar:
  order: 5
---
## `FileCitation`

File citation details

| Field | Type | Default |
| --- | --- | --- |
| `file_id` | `str` | required |
| `quote` | `str \| None` | optional |


## `FilePathInfo`

File path information

| Field | Type |
| --- | --- |
| `file_id` | `str` |


## `IncludeOption`

Include options for response content.

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `IncludeOption.FILE_SEARCH_CALL_RESULTS` | `'file_search_call.results'` |
| `IncludeOption.MESSAGE_INPUT_IMAGE_IMAGE_URL` | `'message.input_image.image_url'` |
| `IncludeOption.COMPUTER_CALL_OUTPUT_OUTPUT_IMAGE_URL` | `'computer_call_output.output.image_url'` |
| `IncludeOption.REASONING_ENCRYPTED_CONTENT` | `'reasoning.encrypted_content'` |


## `IncompleteDetails`

Details about incomplete responses

| Field | Type |
| --- | --- |
| `reason` | `IncompleteReason` |


## `IncompleteReason`

Reason for incomplete response

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `IncompleteReason.MAX_OUTPUT_TOKENS` | `'max_output_tokens'` |
| `IncompleteReason.CONTENT_FILTER` | `'content_filter'` |
| `IncompleteReason.INTERRUPTED` | `'interrupted'` |


## `InputTokensDetails`

Detailed input token breakdown

| Field | Type | Default |
| --- | --- | --- |
| `audio_tokens` | `int \| None` | optional |
| `cached_tokens` | `int \| None` | optional |
| `image_tokens` | `int \| None` | optional |
| `text_tokens` | `int \| None` | optional |


## `OpenResponsesCreateRequest`

OpenResponses API create request

| Field | Type | Default |
| --- | --- | --- |
| `adapter` | `AdapterSelection \| None` | optional |
| `background` | `bool \| None` | optional |
| `dry_allowed_length` | `int \| None` | optional |
| `dry_base` | `float \| None` | optional |
| `dry_multiplier` | `float \| None` | optional |
| `dry_sequence_breakers` | `list[str] \| None` | optional |
| `files` | `list[Any] \| None` | optional |
| `frequency_penalty` | `float \| None` | optional |
| `grammar` | `Grammar \| None` | optional |
| `ignore_eos` | `bool \| None` | optional |
| `include` | `list[IncludeOption] \| None` | optional |
| `input` | `OpenResponsesInput` | required |
| `instructions` | `str \| None` | optional |
| `logit_bias` | `dict[str, float] \| None` | optional |
| `logprobs` | `bool \| None` | optional |
| `max_output_tokens` | `int \| None` | optional |
| `max_tool_calls` | `int \| None` | optional |
| `max_tool_rounds` | `int \| None` | optional |
| `metadata` | `Any` | optional |
| `min_p` | `float \| None` | optional |
| `model` | `str \| None` | optional |
| `n` | `int \| None` | optional |
| `parallel_tool_calls` | `bool \| None` | optional |
| `presence_penalty` | `float \| None` | optional |
| `previous_response_id` | `str \| None` | optional |
| `reasoning` | `ReasoningConfig \| None` | optional |
| `repetition_penalty` | `float \| None` | optional |
| `response_format` | `ResponseFormat \| None` | optional |
| `seed` | `int \| None` | optional |
| `stop` | `StopTokens \| None` | optional |
| `store` | `bool \| None` | optional |
| `stream` | `bool \| None` | optional |
| `stream_options` | `StreamOptions \| None` | optional |
| `temperature` | `float \| None` | optional |
| `text` | `TextConfig \| None` | optional |
| `tool_choice` | `ToolChoice \| None` | optional |
| `tools` | `list[OpenAiTool] \| None` | optional |
| `top_k` | `int \| None` | optional |
| `top_logprobs` | `int \| None` | optional |
| `top_p` | `float \| None` | optional |
| `truncation` | `TruncationStrategy \| None` | optional |


## `OpenResponsesInput`

One of: `Union[str, list[Union[OpenResponsesInputMessage, OpenResponsesInputItemReference, OpenResponsesInputFunctionCall, OpenResponsesInputFunctionCallOutput, OpenResponsesInputReasoning]]]`.


## `OpenResponsesInputFunctionCall`

| Field | Type | Default |
| --- | --- | --- |
| `arguments` | `str` | required |
| `call_id` | `str` | required |
| `name` | `str` | required |
| `namespace` | `str \| None` | optional |
| `type` | `Literal['function_call']` | `'function_call'` |


## `OpenResponsesInputFunctionCallOutput`

| Field | Type | Default |
| --- | --- | --- |
| `call_id` | `str` | required |
| `output` | `str` | required |
| `type` | `Literal['function_call_output']` | `'function_call_output'` |


## `OpenResponsesInputItemReference`

| Field | Type | Default |
| --- | --- | --- |
| `id` | `str` | required |
| `type` | `Literal['item_reference']` | `'item_reference'` |


## `OpenResponsesInputMessage`

| Field | Type | Default |
| --- | --- | --- |
| `content` | `Union[str, list[Union[OpenResponsesInputMessageInputText, OpenResponsesInputMessageInputImage, OpenResponsesInputMessageInputAudio, OpenResponsesInputMessageInputFile]]]` | required |
| `role` | `str` | required |
| `type` | `Literal['message']` | `'message'` |


## `OpenResponsesInputMessageInputAudio`

| Field | Type | Default |
| --- | --- | --- |
| `data` | `str` | required |
| `format` | `str` | required |
| `type` | `Literal['input_audio']` | `'input_audio'` |


## `OpenResponsesInputMessageInputFile`

| Field | Type | Default |
| --- | --- | --- |
| `file_data` | `str \| None` | optional |
| `file_id` | `str \| None` | optional |
| `file_url` | `str \| None` | optional |
| `filename` | `str \| None` | optional |
| `type` | `Literal['input_file']` | `'input_file'` |


## `OpenResponsesInputMessageInputImage`

| Field | Type | Default |
| --- | --- | --- |
| `image_url` | `str \| None` | optional |
| `type` | `Literal['input_image']` | `'input_image'` |


## `OpenResponsesInputMessageInputText`

| Field | Type | Default |
| --- | --- | --- |
| `annotations` | `list[Union[OpenResponsesInputMessageInputTextFileCitation, OpenResponsesInputMessageInputTextUrlCitation, OpenResponsesInputMessageInputTextFilePath, OpenResponsesInputMessageInputTextContainerFileCitation]] \| None` | optional |
| `text` | `str` | required |
| `type` | `Literal['input_text']` | `'input_text'` |


## `OpenResponsesInputMessageInputTextContainerFileCitation`

Container file citation annotation

| Field | Type | Default |
| --- | --- | --- |
| `container_id` | `str` | required |
| `end_index` | `int` | required |
| `file_id` | `str` | required |
| `filename` | `str` | required |
| `index` | `int \| None` | optional |
| `start_index` | `int` | required |
| `type` | `Literal['container_file_citation']` | `'container_file_citation'` |


## `OpenResponsesInputMessageInputTextFileCitation`

File citation annotation

| Field | Type | Default |
| --- | --- | --- |
| `end_index` | `int` | required |
| `file_citation` | `FileCitation` | required |
| `start_index` | `int` | required |
| `text` | `str` | required |
| `type` | `Literal['file_citation']` | `'file_citation'` |


## `OpenResponsesInputMessageInputTextFilePath`

File path annotation

| Field | Type | Default |
| --- | --- | --- |
| `end_index` | `int` | required |
| `file_path` | `FilePathInfo` | required |
| `start_index` | `int` | required |
| `text` | `str` | required |
| `type` | `Literal['file_path']` | `'file_path'` |


## `OpenResponsesInputMessageInputTextUrlCitation`

URL citation annotation

| Field | Type | Default |
| --- | --- | --- |
| `end_index` | `int` | required |
| `start_index` | `int` | required |
| `text` | `str` | required |
| `type` | `Literal['url_citation']` | `'url_citation'` |
| `url_citation` | `UrlCitation` | required |


## `OpenResponsesInputReasoning`

| Field | Type | Default |
| --- | --- | --- |
| `content` | `list[OpenResponsesInputReasoningReasoningText] \| None` | optional |
| `id` | `str \| None` | optional |
| `summary` | `list[OpenResponsesInputReasoningSummaryText] \| None` | optional |
| `type` | `Literal['reasoning']` | `'reasoning'` |


## `OpenResponsesInputReasoningReasoningText`

| Field | Type | Default |
| --- | --- | --- |
| `text` | `str` | required |
| `type` | `Literal['reasoning_text']` | `'reasoning_text'` |


## `OpenResponsesInputReasoningSummaryText`

| Field | Type | Default |
| --- | --- | --- |
| `text` | `str` | required |
| `type` | `Literal['summary_text']` | `'summary_text'` |


## `OpenResponsesStreamEvent`

One of: `Union[OpenResponsesStreamEventResponseCreated, OpenResponsesStreamEventResponseInProgress, OpenResponsesStreamEventResponseOutputItemAdded, OpenResponsesStreamEventResponseContentPartAdded, OpenResponsesStreamEventResponseOutputTextDelta, OpenResponsesStreamEventResponseContentPartDone, OpenResponsesStreamEventResponseOutputItemDone, OpenResponsesStreamEventResponseFunctionCallArgumentsDelta, OpenResponsesStreamEventResponseFunctionCallArgumentsDone, OpenResponsesStreamEventResponseReasoningTextDelta, OpenResponsesStreamEventResponseReasoningTextDone, OpenResponsesStreamEventResponseCompleted, OpenResponsesStreamEventResponseFailed, OpenResponsesStreamEventResponseIncomplete, OpenResponsesStreamEventError]`.


## `OpenResponsesStreamEventError`

Error event

| Field | Type | Default |
| --- | --- | --- |
| `code` | `str` | required |
| `message` | `str` | required |
| `param` | `str \| None` | optional |
| `sequence_number` | `int` | required |
| `type` | `Literal['error']` | `'error'` |


## `OpenResponsesStreamEventResponseCompleted`

Response completed event

| Field | Type | Default |
| --- | --- | --- |
| `response` | `ResponseResource` | required |
| `sequence_number` | `int` | required |
| `type` | `Literal['response.completed']` | `'response.completed'` |


## `OpenResponsesStreamEventResponseContentPartAdded`

Content part added event

| Field | Type | Default |
| --- | --- | --- |
| `content_index` | `int` | required |
| `output_index` | `int` | required |
| `part` | `OutputContent` | required |
| `sequence_number` | `int` | required |
| `type` | `Literal['response.content_part.added']` | `'response.content_part.added'` |


## `OpenResponsesStreamEventResponseContentPartDone`

Content part done event

| Field | Type | Default |
| --- | --- | --- |
| `content_index` | `int` | required |
| `output_index` | `int` | required |
| `part` | `OutputContent` | required |
| `sequence_number` | `int` | required |
| `type` | `Literal['response.content_part.done']` | `'response.content_part.done'` |


## `OpenResponsesStreamEventResponseCreated`

Response created event

| Field | Type | Default |
| --- | --- | --- |
| `response` | `ResponseResource` | required |
| `sequence_number` | `int` | required |
| `type` | `Literal['response.created']` | `'response.created'` |


## `OpenResponsesStreamEventResponseFailed`

Response failed event

| Field | Type | Default |
| --- | --- | --- |
| `response` | `ResponseResource` | required |
| `sequence_number` | `int` | required |
| `type` | `Literal['response.failed']` | `'response.failed'` |


## `OpenResponsesStreamEventResponseFunctionCallArgumentsDelta`

Function call arguments delta

| Field | Type | Default |
| --- | --- | --- |
| `call_id` | `str` | required |
| `delta` | `str` | required |
| `output_index` | `int` | required |
| `sequence_number` | `int` | required |
| `type` | `Literal['response.function_call_arguments.delta']` | `'response.function_call_arguments.delta'` |


## `OpenResponsesStreamEventResponseFunctionCallArgumentsDone`

Function call arguments done

| Field | Type | Default |
| --- | --- | --- |
| `arguments` | `str` | required |
| `call_id` | `str` | required |
| `output_index` | `int` | required |
| `sequence_number` | `int` | required |
| `type` | `Literal['response.function_call_arguments.done']` | `'response.function_call_arguments.done'` |


## `OpenResponsesStreamEventResponseInProgress`

Response in progress event

| Field | Type | Default |
| --- | --- | --- |
| `response` | `ResponseResource` | required |
| `sequence_number` | `int` | required |
| `type` | `Literal['response.in_progress']` | `'response.in_progress'` |


## `OpenResponsesStreamEventResponseIncomplete`

Response incomplete event

| Field | Type | Default |
| --- | --- | --- |
| `response` | `ResponseResource` | required |
| `sequence_number` | `int` | required |
| `type` | `Literal['response.incomplete']` | `'response.incomplete'` |


## `OpenResponsesStreamEventResponseOutputItemAdded`

Output item added event

| Field | Type | Default |
| --- | --- | --- |
| `item` | `OutputItem` | required |
| `output_index` | `int` | required |
| `sequence_number` | `int` | required |
| `type` | `Literal['response.output_item.added']` | `'response.output_item.added'` |


## `OpenResponsesStreamEventResponseOutputItemDone`

Output item done event

| Field | Type | Default |
| --- | --- | --- |
| `item` | `OutputItem` | required |
| `output_index` | `int` | required |
| `sequence_number` | `int` | required |
| `type` | `Literal['response.output_item.done']` | `'response.output_item.done'` |


## `OpenResponsesStreamEventResponseOutputTextDelta`

Text delta event

| Field | Type | Default |
| --- | --- | --- |
| `content_index` | `int` | required |
| `delta` | `str` | required |
| `output_index` | `int` | required |
| `sequence_number` | `int` | required |
| `type` | `Literal['response.output_text.delta']` | `'response.output_text.delta'` |


## `OpenResponsesStreamEventResponseReasoningTextDelta`

Reasoning text delta

| Field | Type | Default |
| --- | --- | --- |
| `content_index` | `int` | required |
| `delta` | `str` | required |
| `item_id` | `str` | required |
| `output_index` | `int` | required |
| `sequence_number` | `int` | required |
| `type` | `Literal['response.reasoning_text.delta']` | `'response.reasoning_text.delta'` |


## `OpenResponsesStreamEventResponseReasoningTextDone`

Reasoning text done

| Field | Type | Default |
| --- | --- | --- |
| `content_index` | `int` | required |
| `item_id` | `str` | required |
| `output_index` | `int` | required |
| `sequence_number` | `int` | required |
| `text` | `str` | required |
| `type` | `Literal['response.reasoning_text.done']` | `'response.reasoning_text.done'` |


## `OutputContent`

One of: `Union[OutputContentOutputText, OutputContentRefusal]`.


## `OutputContentOutputText`

| Field | Type | Default |
| --- | --- | --- |
| `text` | `str` | required |
| `type` | `Literal['output_text']` | `'output_text'` |


## `OutputContentRefusal`

| Field | Type | Default |
| --- | --- | --- |
| `refusal` | `str` | required |
| `type` | `Literal['refusal']` | `'refusal'` |


## `OutputItem`

One of: `Union[OutputItemMessage, OutputItemFunctionCall, OutputItemShellCall, OutputItemShellCallOutput, OutputItemReasoning]`.


## `OutputItemFunctionCall`

| Field | Type | Default |
| --- | --- | --- |
| `arguments` | `str` | required |
| `call_id` | `str` | required |
| `id` | `str` | required |
| `name` | `str` | required |
| `namespace` | `str \| None` | optional |
| `status` | `str` | required |
| `type` | `Literal['function_call']` | `'function_call'` |


## `OutputItemMessage`

| Field | Type | Default |
| --- | --- | --- |
| `content` | `list[Union[OutputItemMessageOutputText, OutputItemMessageRefusal]]` | required |
| `id` | `str` | required |
| `role` | `str` | required |
| `status` | `str` | required |
| `type` | `Literal['message']` | `'message'` |


## `OutputItemMessageOutputText`

| Field | Type | Default |
| --- | --- | --- |
| `text` | `str` | required |
| `type` | `Literal['output_text']` | `'output_text'` |


## `OutputItemMessageRefusal`

| Field | Type | Default |
| --- | --- | --- |
| `refusal` | `str` | required |
| `type` | `Literal['refusal']` | `'refusal'` |


## `OutputItemReasoning`

| Field | Type | Default |
| --- | --- | --- |
| `content` | `list[OutputItemReasoningReasoningText]` | required |
| `id` | `str` | required |
| `status` | `str` | required |
| `summary` | `list[OutputItemReasoningSummaryText]` | required |
| `type` | `Literal['reasoning']` | `'reasoning'` |


## `OutputItemReasoningReasoningText`

| Field | Type | Default |
| --- | --- | --- |
| `text` | `str` | required |
| `type` | `Literal['reasoning_text']` | `'reasoning_text'` |


## `OutputItemReasoningSummaryText`

| Field | Type | Default |
| --- | --- | --- |
| `text` | `str` | required |
| `type` | `Literal['summary_text']` | `'summary_text'` |


## `OutputItemShellCall`

| Field | Type | Default |
| --- | --- | --- |
| `action` | `dict[str, Any]` | required |
| `call_id` | `str` | required |
| `id` | `str` | required |
| `status` | `str` | required |
| `type` | `Literal['shell_call']` | `'shell_call'` |


## `OutputItemShellCallOutput`

| Field | Type | Default |
| --- | --- | --- |
| `call_id` | `str` | required |
| `id` | `str` | required |
| `output` | `list[dict[str, Any]]` | required |
| `status` | `str` | required |
| `type` | `Literal['shell_call_output']` | `'shell_call_output'` |


## `OutputTokensDetails`

Detailed output token breakdown

| Field | Type | Default |
| --- | --- | --- |
| `audio_tokens` | `int \| None` | optional |
| `reasoning_tokens` | `int \| None` | optional |
| `text_tokens` | `int \| None` | optional |


## `ReasoningConfig`

Reasoning configuration for models that support extended thinking

| Field | Type | Default |
| --- | --- | --- |
| `effort` | `ReasoningEffort \| None` | optional |
| `summary` | `ReasoningSummary \| None` | optional |


## `ReasoningSummary`

Reasoning summary configuration

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `ReasoningSummary.CONCISE` | `'concise'` |
| `ReasoningSummary.DETAILED` | `'detailed'` |
| `ReasoningSummary.AUTO` | `'auto'` |


## `ResponseDeleted`

What deleting a response returns.

| Field | Type |
| --- | --- |
| `deleted` | `bool` |
| `id` | `str` |
| `object` | `str` |


## `ResponseError`

Error information for a response

| Field | Type |
| --- | --- |
| `code` | `str` |
| `message` | `str` |


## `ResponseResource`

The main response resource returned by the OpenResponses API

| Field | Type | Default |
| --- | --- | --- |
| `adapter_generation` | `str \| None` | optional |
| `background` | `bool \| None` | optional |
| `completed_at` | `int \| None` | optional |
| `created_at` | `int` | required |
| `error` | `ResponseError \| None` | optional |
| `frequency_penalty` | `float \| None` | optional |
| `id` | `str` | required |
| `incomplete_details` | `IncompleteDetails \| None` | optional |
| `instructions` | `str \| None` | optional |
| `max_output_tokens` | `int \| None` | optional |
| `max_tool_calls` | `int \| None` | optional |
| `metadata` | `Any` | optional |
| `model` | `str` | required |
| `object` | `str` | required |
| `output` | `list[OutputItem]` | required |
| `output_text` | `str \| None` | optional |
| `parallel_tool_calls` | `bool \| None` | optional |
| `presence_penalty` | `float \| None` | optional |
| `previous_response_id` | `str \| None` | optional |
| `reasoning` | `str \| None` | optional |
| `status` | `ResponseStatus` | required |
| `store` | `bool \| None` | optional |
| `temperature` | `float \| None` | optional |
| `text` | `TextConfig \| None` | optional |
| `tool_choice` | `ToolChoice \| None` | optional |
| `tools` | `list[OpenAiTool] \| None` | optional |
| `top_logprobs` | `int \| None` | optional |
| `top_p` | `float \| None` | optional |
| `truncation` | `TruncationStrategy \| None` | optional |
| `usage` | `ResponseUsage \| None` | optional |


## `ResponseStatus`

Status of a response in the OpenResponses API

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `ResponseStatus.QUEUED` | `'queued'` |
| `ResponseStatus.IN_PROGRESS` | `'in_progress'` |
| `ResponseStatus.COMPLETED` | `'completed'` |
| `ResponseStatus.FAILED` | `'failed'` |
| `ResponseStatus.INCOMPLETE` | `'incomplete'` |
| `ResponseStatus.CANCELLED` | `'cancelled'` |


## `ResponseUsage`

Usage information for a response

| Field | Type | Default |
| --- | --- | --- |
| `input_tokens` | `int` | required |
| `input_tokens_details` | `InputTokensDetails \| None` | optional |
| `output_tokens` | `int` | required |
| `output_tokens_details` | `OutputTokensDetails \| None` | optional |
| `total_tokens` | `int` | required |


## `ResponsesAnnotation`

Response annotation

| Field | Type |
| --- | --- |
| `end_index` | `int` |
| `start_index` | `int` |
| `text` | `str` |
| `type` | `str` |


## `ResponsesChunk`

Response streaming chunk

| Field | Type | Default |
| --- | --- | --- |
| `chunk_type` | `str` | required |
| `created_at` | `float` | required |
| `delta` | `ResponsesDelta \| None` | optional |
| `id` | `str` | required |
| `metadata` | `Any` | optional |
| `model` | `str` | required |
| `object` | `str` | required |
| `usage` | `ResponsesUsage \| None` | optional |


## `ResponsesContent`

Response content item

| Field | Type | Default |
| --- | --- | --- |
| `annotations` | `list[ResponsesAnnotation] \| None` | optional |
| `text` | `str \| None` | optional |
| `type` | `str` | required |


## `ResponsesCreateRequest`

Response creation request

| Field | Type | Default |
| --- | --- | --- |
| `adapter` | `AdapterSelection \| None` | optional |
| `dry_allowed_length` | `int \| None` | optional |
| `dry_base` | `float \| None` | optional |
| `dry_multiplier` | `float \| None` | optional |
| `dry_sequence_breakers` | `list[str] \| None` | optional |
| `enable_thinking` | `bool \| None` | optional |
| `frequency_penalty` | `float \| None` | optional |
| `grammar` | `Grammar \| None` | optional |
| `ignore_eos` | `bool \| None` | optional |
| `input` | `ResponsesMessages` | required |
| `instructions` | `str \| None` | optional |
| `logit_bias` | `dict[str, float] \| None` | optional |
| `logprobs` | `bool \| None` | optional |
| `max_tokens` | `int \| None` | optional |
| `max_tool_calls` | `int \| None` | optional |
| `metadata` | `Any` | optional |
| `min_p` | `float \| None` | optional |
| `modalities` | `list[str] \| None` | optional |
| `model` | `str \| None` | optional |
| `n` | `int \| None` | optional |
| `output_token_details` | `bool \| None` | optional |
| `parallel_tool_calls` | `bool \| None` | optional |
| `presence_penalty` | `float \| None` | optional |
| `previous_response_id` | `str \| None` | optional |
| `reasoning_effort` | `str \| None` | optional |
| `reasoning_enabled` | `bool \| None` | optional |
| `reasoning_max_tokens` | `int \| None` | optional |
| `reasoning_top_logprobs` | `int \| None` | optional |
| `repetition_penalty` | `float \| None` | optional |
| `response_format` | `ResponseFormat \| None` | optional |
| `seed` | `int \| None` | optional |
| `stop` | `StopTokens \| None` | optional |
| `store` | `bool \| None` | optional |
| `stream` | `bool \| None` | optional |
| `temperature` | `float \| None` | optional |
| `tool_choice` | `ToolChoice \| None` | optional |
| `tools` | `list[OpenAiTool] \| None` | optional |
| `top_k` | `int \| None` | optional |
| `top_logprobs` | `int \| None` | optional |
| `top_p` | `float \| None` | optional |
| `truncate_sequence` | `bool \| None` | optional |
| `truncation` | `dict[str, Any] \| None` | optional |


## `ResponsesDelta`

Response delta for streaming

| Field | Type | Default |
| --- | --- | --- |
| `output` | `list[ResponsesDeltaOutput] \| None` | optional |
| `status` | `str \| None` | optional |


## `ResponsesDeltaContent`

Response delta content item

| Field | Type | Default |
| --- | --- | --- |
| `text` | `str \| None` | optional |
| `type` | `str` | required |


## `ResponsesDeltaOutput`

Response delta output item

| Field | Type | Default |
| --- | --- | --- |
| `content` | `list[ResponsesDeltaContent] \| None` | optional |
| `id` | `str` | required |
| `type` | `str` | required |


## `ResponsesError`

Response error

| Field | Type |
| --- | --- |
| `message` | `str` |
| `type` | `str` |


## `ResponsesIncompleteDetails`

Incomplete details for incomplete responses

| Field | Type |
| --- | --- |
| `reason` | `str` |


## `ResponsesInputTokensDetails`

Input tokens details

| Field | Type | Default |
| --- | --- | --- |
| `audio_tokens` | `int \| None` | optional |
| `cached_tokens` | `int \| None` | optional |
| `image_tokens` | `int \| None` | optional |
| `text_tokens` | `int \| None` | optional |


## `ResponsesMessages`

One of: `Union[list[Message], str]`.


## `ResponsesObject`

Response object

| Field | Type | Default |
| --- | --- | --- |
| `created_at` | `float` | required |
| `error` | `ResponsesError \| None` | optional |
| `id` | `str` | required |
| `incomplete_details` | `ResponsesIncompleteDetails \| None` | optional |
| `instructions` | `str \| None` | optional |
| `metadata` | `Any` | optional |
| `model` | `str` | required |
| `object` | `str` | required |
| `output` | `list[ResponsesOutput]` | required |
| `output_text` | `str \| None` | optional |
| `status` | `str` | required |
| `usage` | `ResponsesUsage \| None` | optional |


## `ResponsesOutput`

Response output item

| Field | Type | Default |
| --- | --- | --- |
| `content` | `list[ResponsesContent]` | required |
| `id` | `str` | required |
| `role` | `str` | required |
| `status` | `str \| None` | optional |
| `type` | `str` | required |


## `ResponsesOutputTokensDetails`

Output tokens details

| Field | Type | Default |
| --- | --- | --- |
| `audio_tokens` | `int \| None` | optional |
| `reasoning_tokens` | `int \| None` | optional |
| `text_tokens` | `int \| None` | optional |


## `ResponsesUsage`

Response usage information

| Field | Type | Default |
| --- | --- | --- |
| `input_tokens` | `int` | required |
| `input_tokens_details` | `ResponsesInputTokensDetails \| None` | optional |
| `output_tokens` | `int` | required |
| `output_tokens_details` | `ResponsesOutputTokensDetails \| None` | optional |
| `total_tokens` | `int` | required |


## `StreamOptions`

Stream options configuration

| Field | Type | Default |
| --- | --- | --- |
| `include_usage` | `bool \| None` | optional |


## `TextConfig`

Text output configuration

| Field | Type | Default |
| --- | --- | --- |
| `format` | `TextFormat \| None` | optional |


## `TextFormat`

One of: `Union[TextFormatText, TextFormatJsonSchema, TextFormatJsonObject]`.


## `TextFormatJsonObject`

JSON object output

| Field | Type | Default |
| --- | --- | --- |
| `type` | `Literal['json_object']` | `'json_object'` |


## `TextFormatJsonSchema`

JSON output with optional schema

| Field | Type | Default |
| --- | --- | --- |
| `name` | `str` | required |
| `schema` | `Any` | optional |
| `strict` | `bool \| None` | optional |
| `type` | `Literal['json_schema']` | `'json_schema'` |


## `TextFormatText`

Plain text output

| Field | Type | Default |
| --- | --- | --- |
| `type` | `Literal['text']` | `'text'` |


## `TruncationStrategy`

Truncation strategy for input

Members and the names they are sent as; each member is a `str` enum whose `.value` is that name.

| Member | Wire/config name |
| --- | --- |
| `TruncationStrategy.AUTO` | `'auto'` |
| `TruncationStrategy.DISABLED` | `'disabled'` |


## `UrlCitation`

URL citation details

| Field | Type | Default |
| --- | --- | --- |
| `title` | `str \| None` | optional |
| `url` | `str` | required |

---

<small>Generated from [`bindings/python/inference_rs`](https://github.com/christopherthompson81/inference.rs/blob/master/bindings/python/inference_rs).</small>

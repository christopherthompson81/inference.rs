//! The OpenResponses API (<https://www.openresponses.org/>) as an engine operation, free of HTTP.

pub use inference_protocol::responses_types::text::{TextConfig, TextFormat};
use std::{
    collections::HashMap,
    pin::Pin,
    task::Poll,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use either::Either;
use futures::future::BoxFuture;
use inference_core::{
    AgentPermission, AgentToolApprovalHandler, AgenticToolCallData, AgenticToolCallPhase,
    ChatCompletionResponse, FINISH_REASON_CANCELED, FINISH_REASON_LENGTH, InferenceRs, Request,
    RequestCancellation, Response,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::{Receiver, Sender};
use utoipa::{
    PartialSchema, ToSchema,
    openapi::{ArrayBuilder, ObjectBuilder, OneOfBuilder, RefOr, Schema, Type, schema::SchemaType},
};
use uuid::Uuid;

use crate::{
    api_error::{ApiError, ApiErrorKind, ModelErrorMessage, boxed_anyhow},
    background_tasks::get_background_task_manager,
    cached_responses::{StoredConversation, get_response_cache},
    dispatch::{create_response_channel, response_model_id, send_request_with_model},
    engine_chat::{
        ASK_REQUIRES_STREAMING, ChatCompletionParseContext, ChatEngine, DispatchError, ResponseTap,
        parse_request as parse_chat_request, serialize_agentic_progress,
        serialize_approval_required,
    },
    lora_routing::{DEFAULT_MODEL_ID, resolve_lora_adapter_model},
    openai::{
        AdapterSelection, ChatCompletionRequest, Message, MessageContent, OpenAiNamespaceEntry,
        OpenAiTool, OpenAiToolSurface, ToolCall,
    },
    responses_types::{
        content::{Annotation, OutputContent},
        enums::{ItemStatus, ResponseStatus},
        events::StreamingState,
        items::{InputItem, MessageContentParam, OutputItem, ShellCallOutputPart},
        resource::{
            IncompleteDetails, InputTokensDetails, ResponseError, ResponseResource, ResponseUsage,
        },
    },
    types::SharedInferenceRsState,
};

/// Input type for OpenResponses API requests
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum OpenResponsesInput {
    /// Simple string input
    Text(String),
    /// Array of input items (OpenResponses format)
    Items(Vec<InputItem>),
}

impl PartialSchema for OpenResponsesInput {
    fn schema() -> RefOr<Schema> {
        RefOr::T(Schema::OneOf(
            OneOfBuilder::new()
                .item(Schema::Object(
                    ObjectBuilder::new()
                        .schema_type(SchemaType::Type(Type::String))
                        .description(Some("Simple text input"))
                        .build(),
                ))
                .item(Schema::Array(
                    ArrayBuilder::new()
                        .items(InputItem::schema())
                        .description(Some("Array of input items (OpenResponses format)"))
                        .build(),
                ))
                .build(),
        ))
    }
}

impl ToSchema for OpenResponsesInput {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((
            OpenResponsesInput::name().into(),
            OpenResponsesInput::schema(),
        ));
    }
}

impl OpenResponsesInput {
    /// Convert to Either for internal processing
    pub fn into_either(self) -> Either<Vec<Message>, String> {
        match self {
            OpenResponsesInput::Text(s) => Either::Right(s),
            OpenResponsesInput::Items(items) => {
                let messages = convert_input_items_to_messages(items);
                Either::Left(messages)
            }
        }
    }
}

const TEXT_PART_JOINER: &str = "\n\n";
const SYSTEM_ROLE: &str = "system";

/// Convert InputItem types to legacy Message format.
///
/// This function handles multimodal content including text, images, audio, and files.
fn convert_input_items_to_messages(items: Vec<InputItem>) -> Vec<Message> {
    use crate::responses_types::content::NormalizedInputContent;
    use crate::responses_types::items::TaggedInputItem;

    let mut messages = Vec::new();

    for item in items {
        // Normalize to TaggedInputItem for uniform processing
        match item.into_tagged() {
            TaggedInputItem::Message(msg_param) => {
                let content = match msg_param.content {
                    MessageContentParam::Text(text) => Some(MessageContent::from_text(text)),
                    MessageContentParam::Parts(parts) => {
                        // Handle multimodal content parts
                        let mut content_parts = Vec::new();
                        let mut has_non_text_content = false;

                        for part in parts {
                            // Normalize the content part to handle both OpenAI and OpenResponses formats
                            match part.into_normalized() {
                                NormalizedInputContent::Text { text } => {
                                    content_parts.push(MessageContent::text_part(text));
                                }
                                NormalizedInputContent::Image {
                                    image_url,
                                    image_data,
                                    detail,
                                } => {
                                    has_non_text_content = true;
                                    // Prefer image_url over image_data
                                    let url = if let Some(url) = image_url {
                                        url
                                    } else if let Some(data) = image_data {
                                        // Convert base64 data to data URL
                                        format!("data:image/png;base64,{}", data)
                                    } else {
                                        continue; // Skip if no image source
                                    };

                                    let image_part = if let Some(detail_level) = detail {
                                        let detail_str = match detail_level {
                                            crate::responses_types::enums::ImageDetail::Auto => {
                                                "auto"
                                            }
                                            crate::responses_types::enums::ImageDetail::Low => {
                                                "low"
                                            }
                                            crate::responses_types::enums::ImageDetail::High => {
                                                "high"
                                            }
                                        };
                                        MessageContent::image_url_part_with_detail(
                                            url,
                                            detail_str.to_string(),
                                        )
                                    } else {
                                        MessageContent::image_url_part(url)
                                    };
                                    content_parts.push(image_part);
                                }
                                NormalizedInputContent::Audio { data, format } => {
                                    has_non_text_content = true;
                                    // Convert audio to data URL format
                                    let mime_type = match format.as_str() {
                                        "wav" => "audio/wav",
                                        "mp3" => "audio/mpeg",
                                        "flac" => "audio/flac",
                                        "ogg" => "audio/ogg",
                                        _ => "audio/wav", // Default to wav
                                    };
                                    let audio_url = format!("data:{};base64,{}", mime_type, data);
                                    // Audio is represented as a special content part
                                    // Note: Not all models support audio input
                                    let mut audio_part = std::collections::HashMap::new();
                                    audio_part.insert(
                                        "type".to_string(),
                                        crate::openai::MessageInnerContent(Either::Left(
                                            "input_audio".to_string(),
                                        )),
                                    );
                                    let mut audio_obj = std::collections::HashMap::new();
                                    audio_obj.insert("data".to_string(), data);
                                    audio_obj.insert("format".to_string(), format);
                                    audio_part.insert(
                                        "input_audio".to_string(),
                                        crate::openai::MessageInnerContent(Either::Right(
                                            audio_obj,
                                        )),
                                    );
                                    content_parts.push(audio_part);
                                    // Also add as text reference for models that don't support audio
                                    content_parts.push(MessageContent::text_part(format!(
                                        "[Audio content: {}]",
                                        audio_url
                                    )));
                                }
                                NormalizedInputContent::File {
                                    file_id,
                                    file_data,
                                    file_url,
                                    filename,
                                } => {
                                    has_non_text_content = true;
                                    content_parts.push(MessageContent::file_part(
                                        file_id, file_data, file_url, filename,
                                    ));
                                }
                            }
                        }

                        if content_parts.is_empty() {
                            None
                        } else if !has_non_text_content {
                            // Text-only parts collapse to plain text so any role can carry them
                            let text = content_parts
                                .iter()
                                .filter_map(|part| match part.get("text").map(|v| &**v) {
                                    Some(Either::Left(text)) => Some(text.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join(TEXT_PART_JOINER);
                            Some(MessageContent::from_text(text))
                        } else {
                            Some(MessageContent::from_parts(content_parts))
                        }
                    }
                };

                messages.push(Message {
                    content,
                    role: msg_param.role,
                    name: msg_param.name,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            TaggedInputItem::ItemReference { id: _ } => {
                // Item references should be resolved before this point
                // Skip for now - they'll be handled in parse_responses_request
            }
            TaggedInputItem::FunctionCall {
                call_id,
                name,
                namespace,
                arguments,
            } => {
                // Rejoin so history matches the flattened name the model emitted
                let name = match namespace {
                    Some(ns) => format!("{ns}.{name}"),
                    None => name,
                };
                messages.push(Message {
                    content: None,
                    role: "assistant".to_string(),
                    name: None,
                    tool_calls: Some(vec![ToolCall {
                        id: Some(call_id),
                        tp: inference_core::ToolType::Function,
                        function: crate::openai::FunctionCalled { name, arguments },
                    }]),
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            TaggedInputItem::Reasoning { .. } => {}
            TaggedInputItem::FunctionCallOutput { call_id, output } => {
                // Convert to tool message
                messages.push(Message {
                    content: Some(MessageContent::from_text(output)),
                    role: "tool".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: Some(call_id),
                    reasoning_content: None,
                });
            }
        }
    }

    messages
}

/// Reasoning configuration for models that support extended thinking
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ReasoningConfig {
    /// Effort level for reasoning (off, low, medium, high, xhigh). "none" aliases "off".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<crate::responses_types::enums::ReasoningEffort>,
    /// Accepted for compatibility; currently does not change the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
}

/// Reasoning summary configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningSummary {
    /// Generate a concise summary
    Concise,
    /// Generate a detailed summary
    Detailed,
    /// Auto-select summary level
    Auto,
}

/// Stream options configuration
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct StreamOptions {
    /// Include usage statistics in stream
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

/// Request context carrying parameters to echo back in the response.
///
/// This struct captures relevant request parameters that should be
/// echoed back in the ResponseResource per the OpenResponses spec.
#[derive(Debug, Clone, Default)]
pub struct RequestContext {
    /// Tool definitions from the request
    pub tools: Option<Vec<OpenAiTool>>,
    /// Tool choice configuration from the request
    pub tool_choice: Option<inference_core::ToolChoice>,
    /// Whether parallel tool calls are enabled
    pub parallel_tool_calls: Option<bool>,
    /// Text configuration from the request
    pub text: Option<TextConfig>,
    /// Temperature from the request
    pub temperature: Option<f64>,
    /// Top-p from the request
    pub top_p: Option<f64>,
    /// Presence penalty from the request
    pub presence_penalty: Option<f32>,
    /// Frequency penalty from the request
    pub frequency_penalty: Option<f32>,
    /// Top logprobs from the request
    pub top_logprobs: Option<usize>,
    /// Max output tokens from the request
    pub max_output_tokens: Option<usize>,
    /// Max tool calls from the request (even if unsupported)
    pub max_tool_calls: Option<usize>,
    /// Whether to store the response
    pub store: Option<bool>,
    /// Whether request runs in background
    pub background: Option<bool>,
}

impl RequestContext {
    fn echo_into(&self, resource: &mut ResponseResource) {
        resource.tools = self.tools.clone();
        resource.tool_choice = self.tool_choice.clone();
        resource.parallel_tool_calls = self.parallel_tool_calls;
        resource.text = self.text.clone();
        resource.temperature = self.temperature;
        resource.top_p = self.top_p;
        resource.presence_penalty = self.presence_penalty;
        resource.frequency_penalty = self.frequency_penalty;
        resource.top_logprobs = self.top_logprobs;
        resource.max_output_tokens = self.max_output_tokens;
        resource.max_tool_calls = self.max_tool_calls;
        resource.store = self.store;
        resource.background = self.background;
    }

    /// Whether `called` names a tool the client defined, so the call is the client's to answer.
    fn defines_tool(&self, called: &str) -> bool {
        self.tools.iter().flatten().any(|tool| match tool {
            OpenAiTool::Function(f) => f.function.name == called,
            OpenAiTool::ResponsesFunction(f) => f.name == called,
            OpenAiTool::Namespace(namespace) => namespace.tools.iter().any(|entry| {
                matches!(entry, OpenAiNamespaceEntry::Function(f) if namespace.qualified_name(&f.name) == called)
            }),
            _ => false,
        })
    }

    /// Split a flattened `<namespace>.<name>` back into the fields Codex-style clients route on.
    fn split_tool_name(&self, called: &str) -> (String, Option<String>) {
        for tool in self.tools.iter().flatten() {
            let OpenAiTool::Namespace(namespace) = tool else {
                continue;
            };
            for entry in &namespace.tools {
                if let OpenAiNamespaceEntry::Function(f) = entry
                    && namespace.qualified_name(&f.name) == called
                {
                    return (f.name.clone(), Some(namespace.name.clone()));
                }
            }
        }
        (called.to_string(), None)
    }
}

/// Include options for response content.
///
/// This enum specifies additional content to include in the response.
/// By default, certain content may be omitted for efficiency.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum IncludeOption {
    /// Include file search results (not currently supported by inference.rs)
    #[serde(rename = "file_search_call.results")]
    FileSearchCallResults,
    /// Include message input image URLs in the response
    #[serde(rename = "message.input_image.image_url")]
    MessageInputImageUrl,
    /// Include computer call output image URLs (not currently supported by inference.rs)
    #[serde(rename = "computer_call_output.output.image_url")]
    ComputerCallOutputImageUrl,
    /// Include reasoning encrypted content
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
}

/// OpenResponses API create request
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct OpenResponsesCreateRequest {
    // ===== Core OpenResponses Fields =====
    /// The model to use for this request
    #[serde(default = "default_model")]
    pub model: String,

    /// Adapter alias or exact generation to activate for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter: Option<AdapterSelection>,

    /// The input for the response - can be a string or array of input items
    pub input: OpenResponsesInput,

    /// Additional instructions that guide the model's behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// ID of a previous response for multi-turn conversations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// Whether to stream the response using server-sent events
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Stream options for controlling streaming behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Whether to run the request in background (async)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,

    /// Whether to store the response for later retrieval
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    /// User-provided metadata (up to 16 key-value pairs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,

    /// Specifies additional content to include in the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<IncludeOption>>,

    // ===== Generation Parameters =====
    /// Maximum number of output tokens to generate
    #[serde(
        alias = "max_tokens",
        alias = "max_completion_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_output_tokens: Option<usize>,

    /// Temperature for sampling (0-2)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Top-p (nucleus) sampling parameter (0-1)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Continue generation past EOS until another stop condition is met
    #[serde(default)]
    pub ignore_eos: bool,

    /// Seed for deterministic request-scoped sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// Presence penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Frequency penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Number of top log probabilities to return
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<usize>,

    // ===== Tool Calling =====
    /// Tool definitions available for the model to call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAiTool>>,

    /// Controls how the model uses tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<inference_core::ToolChoice>,

    /// Whether to allow parallel tool calls.
    ///
    /// NOTE: Only `true` (default) or `None` is supported. Setting this to `false`
    /// will return an error as inference.rs does not support disabling parallel tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Maximum number of tool calls allowed.
    ///
    /// NOTE: This parameter is not supported. Setting any value will return an error
    /// as inference.rs does not support limiting the number of tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<usize>,

    /// Maximum number of agentic tool rounds (inference.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_rounds: Option<usize>,

    /// Required output files to surface from tool calls (inference.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<Vec<serde_json::Value>>)]
    pub files: Option<Vec<inference_core::RequestedFile>>,

    // ===== Reasoning =====
    /// Configuration for reasoning/thinking behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    // ===== Output Format =====
    /// Text output configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,

    /// Truncation strategy when input exceeds context window
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<crate::responses_types::enums::TruncationStrategy>,

    // ===== inference.rs Extensions (non-standard) =====
    /// Stop sequences to end generation
    #[serde(rename = "stop", skip_serializing_if = "Option::is_none")]
    pub stop_seqs: Option<crate::openai::StopTokens>,

    /// Response format (legacy, prefer `text` field)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<crate::openai::ResponseFormat>,

    /// Logit bias for token manipulation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<u32, f32>>,

    /// Logits processors the engine's host registered, applied by name in this order after the penalties.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits_processors: Option<Vec<String>>,

    /// Tools the engine's host registered after load, offered to the model by name and answered by the host.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_tools: Option<Vec<String>>,

    /// Whether to return log probabilities
    #[serde(default)]
    pub logprobs: bool,

    /// Number of completions to generate
    #[serde(rename = "n", default = "default_1usize")]
    pub n_choices: usize,

    /// Repetition penalty (inference.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,

    /// Top-k sampling (inference.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,

    /// Grammar for constrained generation (inference.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<crate::openai::Grammar>,

    /// Min-p sampling (inference.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,

    /// DRY multiplier (inference.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_multiplier: Option<f32>,

    /// DRY base (inference.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_base: Option<f32>,

    /// DRY allowed length (inference.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_allowed_length: Option<usize>,

    /// DRY sequence breakers (inference.rs extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_sequence_breakers: Option<Vec<String>>,
}

fn default_model() -> String {
    "default".to_string()
}

fn default_1usize() -> usize {
    1
}

/// OpenResponses streaming event format
#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(tag = "type")]
pub enum OpenResponsesStreamEvent {
    /// Response created event
    #[serde(rename = "response.created")]
    ResponseCreated {
        sequence_number: u64,
        response: ResponseResource,
    },
    /// Response in progress event
    #[serde(rename = "response.in_progress")]
    ResponseInProgress {
        sequence_number: u64,
        response: ResponseResource,
    },
    /// Output item added event
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        sequence_number: u64,
        output_index: usize,
        item: OutputItem,
    },
    /// Content part added event
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        sequence_number: u64,
        output_index: usize,
        content_index: usize,
        part: OutputContent,
    },
    /// Text delta event
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        sequence_number: u64,
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    /// Content part done event
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        sequence_number: u64,
        output_index: usize,
        content_index: usize,
        part: OutputContent,
    },
    /// Output item done event
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        sequence_number: u64,
        output_index: usize,
        item: OutputItem,
    },
    /// Function call arguments delta
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        sequence_number: u64,
        output_index: usize,
        call_id: String,
        delta: String,
    },
    /// Function call arguments done
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        sequence_number: u64,
        output_index: usize,
        call_id: String,
        arguments: String,
    },
    /// Reasoning text delta
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    /// Reasoning text done
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        text: String,
    },
    /// Response completed event
    #[serde(rename = "response.completed")]
    ResponseCompleted {
        sequence_number: u64,
        response: ResponseResource,
    },
    /// Response failed event
    #[serde(rename = "response.failed")]
    ResponseFailed {
        sequence_number: u64,
        response: ResponseResource,
    },
    /// Response incomplete event
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete {
        sequence_number: u64,
        response: ResponseResource,
    },
    /// The response stopped because its caller cancelled it; carries what was generated, with usage
    #[serde(rename = "response.cancelled")]
    ResponseCancelled {
        sequence_number: u64,
        response: ResponseResource,
    },
    /// Error event
    #[serde(rename = "error")]
    Error {
        sequence_number: u64,
        code: String,
        message: String,
        param: Option<String>,
    },
}

fn api_error_code(error: &ApiError) -> String {
    error.code.clone().unwrap_or_else(|| match error.kind {
        ApiErrorKind::InvalidRequest => "invalid_request".to_string(),
        ApiErrorKind::NotFound => "not_found".to_string(),
        ApiErrorKind::Gone => "gone".to_string(),
        ApiErrorKind::Unauthorized => "invalid_api_key".to_string(),
        ApiErrorKind::Forbidden => "forbidden".to_string(),
        ApiErrorKind::Conflict => "conflict".to_string(),
        ApiErrorKind::PayloadTooLarge => "request_body_too_large".to_string(),
        ApiErrorKind::UnsupportedMediaType => "invalid_content_type".to_string(),
        ApiErrorKind::RateLimited => "rate_limit_exceeded".to_string(),
        ApiErrorKind::Unavailable | ApiErrorKind::Overloaded => "service_unavailable".to_string(),
        ApiErrorKind::Internal => "internal_error".to_string(),
    })
}

fn response_error_from_api_error(error: ApiError) -> ResponseError {
    let code = api_error_code(&error);
    ResponseError::new(code, error.message)
}

fn unsupported_background_stream_error() -> ApiError {
    ApiError::new(
        ApiErrorKind::InvalidRequest,
        "`background: true` with `stream: true` is not supported by this server.",
        Some("unsupported_parameter_combination"),
        Some("background,stream"),
    )
}

fn stream_error_from_api_error(sequence_number: u64, error: ApiError) -> OpenResponsesStreamEvent {
    OpenResponsesStreamEvent::Error {
        sequence_number,
        code: api_error_code(&error),
        message: error.message,
        param: error.param,
    }
}

fn classify_api_error(
    state: &SharedInferenceRsState,
    error: &(dyn std::error::Error + 'static),
    fallback: ApiErrorKind,
) -> ApiError {
    let api_error = ApiError::from_error(error, fallback);
    if matches!(
        api_error.kind,
        ApiErrorKind::Internal | ApiErrorKind::Unavailable | ApiErrorKind::Overloaded
    ) {
        InferenceRs::maybe_log_error(state.clone(), error);
    }
    api_error
}

#[derive(Clone)]
struct PendingShellCall {
    call_id: String,
    commands: Vec<String>,
}

type PendingShellCalls = HashMap<String, PendingShellCall>;

#[derive(Clone)]
struct MessageOutputItemState {
    id: String,
}

impl MessageOutputItemState {
    fn new() -> Self {
        Self {
            id: format!("msg_{}", Uuid::new_v4()),
        }
    }

    fn added_item(&self) -> OutputItem {
        OutputItem::message(self.id.clone(), vec![], ItemStatus::InProgress)
    }

    fn item_with_text(
        &self,
        text: String,
        response_id: &str,
        files: &[inference_core::File],
        status: ItemStatus,
    ) -> OutputItem {
        OutputItem::message(
            self.id.clone(),
            vec![output_text_with_file_annotations(text, response_id, files)],
            status,
        )
    }
}

fn shell_output_parts(data: &AgenticToolCallData) -> Option<Vec<ShellCallOutputPart>> {
    let AgenticToolCallData::Shell {
        stdout,
        stderr,
        exit_code,
        status,
        timed_out,
        ..
    } = data
    else {
        return None;
    };

    let mut parts = Vec::new();
    if let Some(stdout) = stdout.as_ref().filter(|s| !s.is_empty()) {
        parts.push(ShellCallOutputPart::Stdout {
            text: stdout.clone(),
        });
    }
    if let Some(stderr) = stderr.as_ref().filter(|s| !s.is_empty()) {
        parts.push(ShellCallOutputPart::Stderr {
            text: stderr.clone(),
        });
    }
    parts.push(ShellCallOutputPart::Outcome {
        status: status.clone().unwrap_or_else(|| "completed".to_string()),
        exit_code: *exit_code,
        timed_out: *timed_out,
    });
    Some(parts)
}

fn record_shell_progress_items(
    pending: &mut PendingShellCalls,
    output_items: &mut Vec<OutputItem>,
    tool_call_id: &str,
    phase: &AgenticToolCallPhase,
) -> Option<Vec<OutputItem>> {
    match phase {
        AgenticToolCallPhase::Calling(AgenticToolCallData::Shell { commands, .. }) => {
            let call_id = tool_call_id.to_string();
            let item = OutputItem::shell_call(
                format!("sc_{}", Uuid::new_v4().simple()),
                call_id.clone(),
                commands.clone(),
                ItemStatus::Completed,
            );
            pending.insert(
                call_id.clone(),
                PendingShellCall {
                    call_id,
                    commands: commands.clone(),
                },
            );
            output_items.push(item.clone());
            Some(vec![item])
        }
        AgenticToolCallPhase::Complete(data @ AgenticToolCallData::Shell { commands, .. }) => {
            let pending_call = pending
                .remove(tool_call_id)
                .unwrap_or_else(|| PendingShellCall {
                    call_id: tool_call_id.to_string(),
                    commands: commands.clone(),
                });
            let mut items = Vec::new();
            if !output_items.iter().any(|item| match item {
                OutputItem::ShellCall { call_id, .. } => call_id == &pending_call.call_id,
                _ => false,
            }) {
                let call_item = OutputItem::shell_call(
                    format!("sc_{}", Uuid::new_v4().simple()),
                    pending_call.call_id.clone(),
                    pending_call.commands.clone(),
                    ItemStatus::Completed,
                );
                output_items.push(call_item.clone());
                items.push(call_item);
            }
            let output_item = OutputItem::shell_call_output(
                format!("sco_{}", Uuid::new_v4().simple()),
                pending_call.call_id,
                shell_output_parts(data).unwrap_or_default(),
                ItemStatus::Completed,
            );
            output_items.push(output_item.clone());
            items.push(output_item);
            Some(items)
        }
        _ => None,
    }
}

/// One item of a streaming Responses request: an OpenResponses event, or one of the engine's own progress events.
// Nearly every item is an `Event`, so boxing it would cost an allocation per event for nothing.
#[allow(clippy::large_enum_variant)]
pub enum ResponsesStreamItem {
    Event(OpenResponsesStreamEvent),
    AgenticToolCallProgress(Value),
    AgenticToolApprovalRequired(Value),
    FileProduced(inference_core::File),
}

impl ResponsesStreamItem {
    /// The SSE event name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Event(event) => event.event_type(),
            Self::AgenticToolCallProgress(_) => "agentic_tool_call_progress",
            Self::AgenticToolApprovalRequired(_) => "agentic_tool_approval_required",
            Self::FileProduced(_) => "file_produced",
        }
    }

    /// `{"event": <name>, "data": <payload>}`; errors arrive as the OpenResponses `error` event.
    pub fn to_json(&self) -> String {
        let data = match self {
            Self::Event(event) => serde_json::to_value(event),
            Self::AgenticToolCallProgress(value) | Self::AgenticToolApprovalRequired(value) => {
                Ok(value.clone())
            }
            Self::FileProduced(file) => serde_json::to_value(file),
        }
        .unwrap_or_else(|_| ApiError::internal().to_openai_body());
        serde_json::json!({ "event": self.name(), "data": data }).to_string()
    }
}

/// The events of a streaming Responses request. Dropping it abandons the request.
pub struct OpenResponsesStreamer {
    rx: Receiver<Response>,
    finished: bool,
    state: SharedInferenceRsState,
    streaming_state: StreamingState,
    metadata: Option<Value>,
    pending_events: Vec<OpenResponsesStreamEvent>,
    accumulated_text: String,
    /// Every round's reasoning, for the stored history and `reasoning`; each item holds only its own.
    accumulated_reasoning: String,
    reasoning_item_text: String,
    /// Reasoning items of earlier agentic rounds, closed before this one began, with their output indices.
    earlier_reasoning_items: Vec<(usize, OutputItem)>,
    /// Each item's `output_index` is fixed when it is added, in the order items are added.
    next_output_index: usize,
    reasoning_item_index: usize,
    message_index: Option<usize>,
    shell_output_indices: Vec<usize>,
    /// A tool ran since the last delta: reasoning that follows belongs to a new round.
    round_boundary: bool,
    content_part_added: bool,
    output_item_added: bool,
    reasoning_item_id: String,
    reasoning_item_added: bool,
    /// How the reasoning item ended, once it has.
    reasoning_item_done: Option<ItemStatus>,
    store: bool,
    /// Taken when the stream finishes and stored for `previous_response_id`.
    conversation_history: Option<Vec<Message>>,
    request_context: RequestContext,
    message_output_item: MessageOutputItemState,
    shell_output_items: Vec<OutputItem>,
    function_call_items: Vec<(usize, OutputItem)>,
    pending_shell_calls: PendingShellCalls,
    /// The calls handed back to the client, stored with the reply so a `function_call_output` can answer them.
    returned_tool_calls: Vec<ToolCall>,
    /// The agent session the run reported, stored so a follow-up continues it by id.
    session_id: Option<String>,
    owner: Option<String>,
    files: Vec<inference_core::File>,
    tap: Option<ResponseTap>,
    cancellation: RequestCancellation,
}

impl OpenResponsesStreamer {
    pub(crate) fn new(
        prepared: PreparedResponse,
        state: SharedInferenceRsState,
        tap: Option<ResponseTap>,
    ) -> Self {
        let PreparedResponse {
            rx,
            id,
            model,
            metadata,
            store,
            history,
            context,
            cancellation,
            session_id,
            owner,
            ..
        } = prepared;
        Self {
            rx,
            finished: false,
            state,
            streaming_state: StreamingState::new(id, model, unix_now()),
            metadata,
            pending_events: Vec::new(),
            accumulated_text: String::new(),
            accumulated_reasoning: String::new(),
            reasoning_item_text: String::new(),
            earlier_reasoning_items: Vec::new(),
            next_output_index: 0,
            reasoning_item_index: 0,
            message_index: None,
            shell_output_indices: Vec::new(),
            round_boundary: false,
            content_part_added: false,
            output_item_added: false,
            reasoning_item_id: format!("rs_{}", Uuid::new_v4()),
            reasoning_item_added: false,
            reasoning_item_done: None,
            store,
            conversation_history: Some(history),
            request_context: context,
            message_output_item: MessageOutputItemState::new(),
            shell_output_items: Vec::new(),
            function_call_items: Vec::new(),
            pending_shell_calls: HashMap::new(),
            returned_tool_calls: Vec::new(),
            session_id,
            owner,
            files: Vec::new(),
            tap,
            cancellation,
        }
    }

    /// Reports each engine response to `tap` as the stream reads it, e.g. for an access log.
    pub fn with_tap(mut self, tap: Option<ResponseTap>) -> Self {
        self.tap = tap;
        self
    }

    /// The request's cancellation, for a caller that cancels from elsewhere, e.g. a signal handler.
    pub fn cancellation(&self) -> RequestCancellation {
        self.cancellation.clone()
    }

    /// Ends the request on its next sampled token; the stream ends with `response.cancelled`, with usage.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    // Stores the final resource and the conversation before the terminal event goes out, so both are there for a
    // follow-up the moment the caller sees it.
    fn finish(&mut self, response: Option<&ResponseResource>) {
        self.finished = true;
        let Some(mut history) = self.conversation_history.take().filter(|_| self.store) else {
            return;
        };
        let cache = get_response_cache();
        let Some(response) = response else {
            return;
        };
        let owner = self.owner.as_deref();
        let _ = cache.store_response(
            self.streaming_state.response_id.clone(),
            response.clone(),
            owner,
        );
        // Its reply was cut short or failed; only a finished one is a conversation to continue.
        if matches!(
            response.status,
            ResponseStatus::Cancelled | ResponseStatus::Failed
        ) {
            return;
        }
        let tool_calls = std::mem::take(&mut self.returned_tool_calls);
        if !self.accumulated_text.is_empty() || !tool_calls.is_empty() {
            history.push(Message {
                content: (!self.accumulated_text.is_empty())
                    .then(|| MessageContent::from_text(self.accumulated_text.clone())),
                role: "assistant".to_string(),
                name: None,
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                tool_call_id: None,
                reasoning_content: (!self.accumulated_reasoning.is_empty())
                    .then(|| self.accumulated_reasoning.clone()),
            });
        }
        let conversation = StoredConversation {
            messages: history,
            session_id: self.session_id.take(),
        };
        let _ = cache.store_conversation(
            self.streaming_state.response_id.clone(),
            conversation,
            self.owner.as_deref(),
        );
    }

    fn claim_output_index(&mut self) -> usize {
        let index = self.next_output_index;
        self.next_output_index += 1;
        index
    }

    fn reasoning_output_index(&self) -> usize {
        self.reasoning_item_index
    }

    // A later agentic round reasons again after the earlier item closed: it gets an item of its own.
    fn start_next_reasoning_item(&mut self) {
        let Some(status) = self.reasoning_item_done.take() else {
            return;
        };
        let item = OutputItem::reasoning(
            std::mem::replace(
                &mut self.reasoning_item_id,
                format!("rs_{}", Uuid::new_v4()),
            ),
            std::mem::take(&mut self.reasoning_item_text),
            status,
        );
        self.earlier_reasoning_items
            .push((self.reasoning_item_index, item));
        self.reasoning_item_added = false;
    }

    // `status` is `completed` when the reply moved on from reasoning, else how the run ended.
    fn finish_reasoning_item(
        &mut self,
        events: &mut Vec<OpenResponsesStreamEvent>,
        status: ItemStatus,
    ) {
        if !self.reasoning_item_added || self.reasoning_item_done.is_some() {
            return;
        }
        self.reasoning_item_done = Some(status);
        let output_index = self.reasoning_output_index();
        let seq = self.streaming_state.next_sequence_number();
        events.push(OpenResponsesStreamEvent::ReasoningTextDone {
            sequence_number: seq,
            item_id: self.reasoning_item_id.clone(),
            output_index,
            content_index: 0,
            text: self.reasoning_item_text.clone(),
        });
        let seq = self.streaming_state.next_sequence_number();
        events.push(OpenResponsesStreamEvent::OutputItemDone {
            sequence_number: seq,
            output_index,
            item: OutputItem::reasoning(
                self.reasoning_item_id.clone(),
                self.reasoning_item_text.clone(),
                status,
            ),
        });
    }

    /// Build initial response resource
    fn build_response_resource(&self, status: ResponseStatus) -> ResponseResource {
        let mut resource = ResponseResource::new(
            self.streaming_state.response_id.clone(),
            self.streaming_state.model.clone(),
            self.streaming_state.created_at,
        );
        resource.status = status;
        resource.metadata = self.metadata.clone();

        self.request_context.echo_into(&mut resource);

        resource
    }

    /// Build current response resource with output
    fn build_current_response(&self, status: ResponseStatus) -> ResponseResource {
        let mut resource = self.build_response_resource(status);
        // In the order the stream added them, so each item sits at the `output_index` its events named.
        let mut output: Vec<(usize, OutputItem)> = self
            .shell_output_indices
            .iter()
            .copied()
            .zip(self.shell_output_items.iter().cloned())
            .chain(self.earlier_reasoning_items.iter().cloned())
            .chain(self.function_call_items.iter().cloned())
            .collect();
        if self.reasoning_item_added {
            output.push((
                self.reasoning_item_index,
                OutputItem::reasoning(
                    self.reasoning_item_id.clone(),
                    self.reasoning_item_text.clone(),
                    self.reasoning_item_done
                        .unwrap_or_else(|| unfinished_item_status(status)),
                ),
            ));
        }
        if let Some(index) = self
            .message_index
            .filter(|_| !self.accumulated_text.is_empty())
        {
            let item = self.message_output_item.item_with_text(
                self.accumulated_text.clone(),
                &self.streaming_state.response_id,
                &self.files,
                if status == ResponseStatus::Completed {
                    ItemStatus::Completed
                } else {
                    unfinished_item_status(status)
                },
            );
            output.push((index, item));
            resource.output_text = Some(self.accumulated_text.clone());
        }
        output.sort_by_key(|(index, _)| *index);
        resource.output = output.into_iter().map(|(_, item)| item).collect();

        // Include reasoning if available
        if !self.accumulated_reasoning.is_empty() {
            resource.reasoning = Some(self.accumulated_reasoning.clone());
        }

        resource
    }
}

impl futures::Stream for OpenResponsesStreamer {
    type Item = ResponsesStreamItem;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        // Check for pending events first
        if !self.pending_events.is_empty() {
            let event = self.pending_events.remove(0);
            return Poll::Ready(Some(ResponsesStreamItem::Event(event)));
        }

        if self.finished {
            return Poll::Ready(None);
        }

        // Emit response.created if not sent
        if !self.streaming_state.created_sent {
            self.streaming_state.created_sent = true;
            let seq = self.streaming_state.next_sequence_number();
            let response = self.build_response_resource(ResponseStatus::Queued);
            let event = OpenResponsesStreamEvent::ResponseCreated {
                sequence_number: seq,
                response,
            };
            return Poll::Ready(Some(ResponsesStreamItem::Event(event)));
        }

        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(resp)) => {
                if let Some(tap) = &self.tap {
                    tap(&resp);
                }
                match resp {
                    Response::ModelError(msg, _) => {
                        InferenceRs::maybe_log_error(
                            self.state.clone(),
                            &ModelErrorMessage(msg.to_string()),
                        );

                        let seq = self.streaming_state.next_sequence_number();
                        let mut response = self.build_current_response(ResponseStatus::Failed);
                        response.error =
                            Some(response_error_from_api_error(ApiError::model_error()));

                        self.finish(Some(&response));
                        let event = OpenResponsesStreamEvent::ResponseFailed {
                            sequence_number: seq,
                            response,
                        };
                        Poll::Ready(Some(ResponsesStreamItem::Event(event)))
                    }
                    Response::ValidationError(e) => {
                        let seq = self.streaming_state.next_sequence_number();
                        let error = classify_api_error(
                            &self.state,
                            e.as_ref(),
                            ApiErrorKind::InvalidRequest,
                        );
                        let event = stream_error_from_api_error(seq, error);
                        self.finish(None);
                        Poll::Ready(Some(ResponsesStreamItem::Event(event)))
                    }
                    Response::InternalError(e) => {
                        let seq = self.streaming_state.next_sequence_number();
                        let error =
                            classify_api_error(&self.state, e.as_ref(), ApiErrorKind::Internal);
                        let event = stream_error_from_api_error(seq, error);
                        self.finish(None);
                        Poll::Ready(Some(ResponsesStreamItem::Event(event)))
                    }
                    Response::Chunk(chat_chunk) => {
                        let mut events_to_emit = Vec::new();
                        if chat_chunk.session_id.is_some() {
                            self.session_id.clone_from(&chat_chunk.session_id);
                        }

                        // Emit response.in_progress if not sent
                        if !self.streaming_state.in_progress_sent {
                            self.streaming_state.in_progress_sent = true;
                            let seq = self.streaming_state.next_sequence_number();
                            let response = self.build_response_resource(ResponseStatus::InProgress);
                            events_to_emit.push(OpenResponsesStreamEvent::ResponseInProgress {
                                sequence_number: seq,
                                response,
                            });
                        }

                        // Check if all choices are finished
                        let all_finished =
                            chat_chunk.choices.iter().all(|c| c.finish_reason.is_some());

                        for choice in &chat_chunk.choices {
                            if let Some(reasoning) = &choice.delta.reasoning_content {
                                // a tool ran since: the round that reasoned before it is over
                                if std::mem::take(&mut self.round_boundary) {
                                    self.finish_reasoning_item(
                                        &mut events_to_emit,
                                        ItemStatus::Completed,
                                    );
                                }
                                self.start_next_reasoning_item();
                                if !self.reasoning_item_added {
                                    self.reasoning_item_index = self.claim_output_index();
                                }
                                let output_index = self.reasoning_output_index();
                                if !self.reasoning_item_added {
                                    self.reasoning_item_added = true;
                                    let seq = self.streaming_state.next_sequence_number();
                                    events_to_emit.push(
                                        OpenResponsesStreamEvent::OutputItemAdded {
                                            sequence_number: seq,
                                            output_index,
                                            item: OutputItem::reasoning(
                                                self.reasoning_item_id.clone(),
                                                String::new(),
                                                ItemStatus::InProgress,
                                            ),
                                        },
                                    );
                                }
                                self.accumulated_reasoning.push_str(reasoning);
                                self.reasoning_item_text.push_str(reasoning);
                                let seq = self.streaming_state.next_sequence_number();
                                events_to_emit.push(OpenResponsesStreamEvent::ReasoningTextDelta {
                                    sequence_number: seq,
                                    item_id: self.reasoning_item_id.clone(),
                                    output_index,
                                    content_index: 0,
                                    delta: reasoning.clone(),
                                });
                            }

                            if choice.delta.content.is_some() || choice.delta.tool_calls.is_some() {
                                self.finish_reasoning_item(
                                    &mut events_to_emit,
                                    ItemStatus::Completed,
                                );
                            }
                            // Handle text content
                            if let Some(content) = &choice.delta.content {
                                self.round_boundary = false;
                                let message_output_index = match self.message_index {
                                    Some(index) => index,
                                    None => {
                                        let index = self.claim_output_index();
                                        self.message_index = Some(index);
                                        index
                                    }
                                };
                                // Emit output_item.added if not done
                                if !self.output_item_added {
                                    self.output_item_added = true;
                                    let seq = self.streaming_state.next_sequence_number();
                                    let item = self.message_output_item.added_item();
                                    events_to_emit.push(
                                        OpenResponsesStreamEvent::OutputItemAdded {
                                            sequence_number: seq,
                                            output_index: message_output_index,
                                            item,
                                        },
                                    );
                                }

                                // Emit content_part.added if not done
                                if !self.content_part_added {
                                    self.content_part_added = true;
                                    let seq = self.streaming_state.next_sequence_number();
                                    let part = OutputContent::text(String::new());
                                    events_to_emit.push(
                                        OpenResponsesStreamEvent::ContentPartAdded {
                                            sequence_number: seq,
                                            output_index: message_output_index,
                                            content_index: 0,
                                            part,
                                        },
                                    );
                                }

                                // Accumulate text
                                self.accumulated_text.push_str(content);

                                // Emit text delta
                                let seq = self.streaming_state.next_sequence_number();
                                events_to_emit.push(OpenResponsesStreamEvent::OutputTextDelta {
                                    sequence_number: seq,
                                    output_index: message_output_index,
                                    content_index: 0,
                                    delta: content.clone(),
                                });
                            }

                            // Tool calls arrive fully parsed, so each one is a complete output item
                            if let Some(tool_calls) = &choice.delta.tool_calls {
                                for tool_call in tool_calls {
                                    let output_index = self.claim_output_index();
                                    let (name, namespace) = self
                                        .request_context
                                        .split_tool_name(&tool_call.function.name);
                                    let item = OutputItem::function_call(
                                        format!("fc_{}", Uuid::new_v4()),
                                        tool_call.id.clone(),
                                        name,
                                        namespace,
                                        tool_call.function.arguments.clone(),
                                        ItemStatus::Completed,
                                    );
                                    let seq = self.streaming_state.next_sequence_number();
                                    events_to_emit.push(
                                        OpenResponsesStreamEvent::OutputItemAdded {
                                            sequence_number: seq,
                                            output_index,
                                            item: item.clone(),
                                        },
                                    );
                                    let seq = self.streaming_state.next_sequence_number();
                                    events_to_emit.push(
                                        OpenResponsesStreamEvent::FunctionCallArgumentsDelta {
                                            sequence_number: seq,
                                            output_index,
                                            call_id: tool_call.id.clone(),
                                            delta: tool_call.function.arguments.clone(),
                                        },
                                    );
                                    let seq = self.streaming_state.next_sequence_number();
                                    events_to_emit.push(
                                        OpenResponsesStreamEvent::FunctionCallArgumentsDone {
                                            sequence_number: seq,
                                            output_index,
                                            call_id: tool_call.id.clone(),
                                            arguments: tool_call.function.arguments.clone(),
                                        },
                                    );
                                    let seq = self.streaming_state.next_sequence_number();
                                    events_to_emit.push(OpenResponsesStreamEvent::OutputItemDone {
                                        sequence_number: seq,
                                        output_index,
                                        item: item.clone(),
                                    });
                                    self.function_call_items.push((output_index, item));
                                    if self.request_context.defines_tool(&tool_call.function.name) {
                                        self.returned_tool_calls.push(history_tool_call(tool_call));
                                    }
                                }
                            }
                        }

                        // If all finished, emit completion events
                        if all_finished {
                            let status = finished_status(
                                chat_chunk
                                    .choices
                                    .iter()
                                    .map(|choice| choice.finish_reason.as_deref()),
                            );
                            // still reasoning when the run stopped: it ended with the run
                            self.finish_reasoning_item(
                                &mut events_to_emit,
                                finished_item_status(status),
                            );
                            let message_output_index = self.message_index.unwrap_or_default();
                            // Emit content_part.done
                            if self.content_part_added {
                                let seq = self.streaming_state.next_sequence_number();
                                let part = output_text_with_file_annotations(
                                    self.accumulated_text.clone(),
                                    &self.streaming_state.response_id,
                                    &self.files,
                                );
                                events_to_emit.push(OpenResponsesStreamEvent::ContentPartDone {
                                    sequence_number: seq,
                                    output_index: message_output_index,
                                    content_index: 0,
                                    part,
                                });
                            }

                            // Emit output_item.done
                            if self.output_item_added {
                                let seq = self.streaming_state.next_sequence_number();
                                let item = self.message_output_item.item_with_text(
                                    self.accumulated_text.clone(),
                                    &self.streaming_state.response_id,
                                    &self.files,
                                    finished_item_status(status),
                                );
                                events_to_emit.push(OpenResponsesStreamEvent::OutputItemDone {
                                    sequence_number: seq,
                                    output_index: message_output_index,
                                    item,
                                });
                            }

                            let seq = self.streaming_state.next_sequence_number();
                            let mut response = self.build_current_response(status);
                            apply_incomplete_details(&mut response);
                            response.adapter_generation = chat_chunk.adapter_generation.clone();
                            response.completed_at = Some(unix_now());

                            // Add usage from chunk if available
                            if let Some(usage) = &chat_chunk.usage {
                                let mut resp_usage = ResponseUsage::new(
                                    usage.prompt_tokens,
                                    usage.completion_tokens,
                                );
                                if let Some(details) = &usage.prompt_tokens_details {
                                    resp_usage.input_tokens_details = Some(InputTokensDetails {
                                        cached_tokens: Some(details.cached_tokens),
                                        ..Default::default()
                                    });
                                }
                                response.usage = Some(resp_usage);
                            }

                            self.finish(Some(&response));
                            events_to_emit.push(terminal_event(seq, response));
                        }

                        InferenceRs::maybe_log_response(self.state.clone(), &chat_chunk);

                        // Return first event, queue the rest
                        if !events_to_emit.is_empty() {
                            let first_event = events_to_emit.remove(0);
                            self.pending_events.extend(events_to_emit);
                            Poll::Ready(Some(ResponsesStreamItem::Event(first_event)))
                        } else {
                            // Chunk consumed without producing an event; re-poll immediately
                            cx.waker().wake_by_ref();
                            Poll::Pending
                        }
                    }
                    Response::Done(chat_resp) => {
                        // Handle non-streaming completion through chunk path
                        // This shouldn't normally happen in streaming mode
                        let seq = self.streaming_state.next_sequence_number();
                        let response = chat_response_to_response_resource(
                            &chat_resp,
                            self.streaming_state.response_id.clone(),
                            self.streaming_state.model.clone(),
                            self.metadata.clone(),
                            &self.request_context,
                            &self.shell_output_items,
                            &self.files,
                        );
                        for choice in &chat_resp.choices {
                            if let Some(content) = &choice.message.content {
                                self.accumulated_text.push_str(content);
                            }
                            if let Some(reasoning) = &choice.message.reasoning_content {
                                self.accumulated_reasoning.push_str(reasoning);
                            }
                            let calls = client_tool_calls(&self.request_context, &choice.message);
                            self.returned_tool_calls.extend(calls);
                        }
                        if chat_resp.session_id.is_some() {
                            self.session_id.clone_from(&chat_resp.session_id);
                        }
                        self.finish(Some(&response));
                        let event = terminal_event(seq, response);
                        Poll::Ready(Some(ResponsesStreamItem::Event(event)))
                    }
                    Response::AgenticToolCallProgress {
                        round,
                        tool_call_id,
                        tool_name,
                        phase,
                    } => {
                        let mut pending_shell_calls = std::mem::take(&mut self.pending_shell_calls);
                        let mut shell_output_items = std::mem::take(&mut self.shell_output_items);
                        let shell_items = record_shell_progress_items(
                            &mut pending_shell_calls,
                            &mut shell_output_items,
                            &tool_call_id,
                            &phase,
                        );
                        self.pending_shell_calls = pending_shell_calls;
                        self.shell_output_items = shell_output_items;
                        self.round_boundary = true;
                        if let Some(items) = shell_items {
                            let mut events = Vec::new();
                            for item in items {
                                let output_index = self.claim_output_index();
                                self.shell_output_indices.push(output_index);
                                let seq = self.streaming_state.next_sequence_number();
                                events.push(OpenResponsesStreamEvent::OutputItemAdded {
                                    sequence_number: seq,
                                    output_index,
                                    item: item.clone(),
                                });
                                let seq = self.streaming_state.next_sequence_number();
                                events.push(OpenResponsesStreamEvent::OutputItemDone {
                                    sequence_number: seq,
                                    output_index,
                                    item,
                                });
                            }
                            let first = events.remove(0);
                            self.pending_events.extend(events);
                            Poll::Ready(Some(ResponsesStreamItem::Event(first)))
                        } else {
                            Poll::Ready(Some(ResponsesStreamItem::AgenticToolCallProgress(
                                serialize_agentic_progress(
                                    round,
                                    &tool_call_id,
                                    &tool_name,
                                    &phase,
                                ),
                            )))
                        }
                    }
                    Response::AgenticToolApprovalRequired {
                        approval_id,
                        session_id,
                        round,
                        tool,
                        arguments,
                    } => Poll::Ready(Some(ResponsesStreamItem::AgenticToolApprovalRequired(
                        serialize_approval_required(
                            &approval_id,
                            &session_id,
                            round,
                            &tool,
                            &arguments,
                        ),
                    ))),
                    Response::File(file) => {
                        tag_with_container(
                            &self.state,
                            &file,
                            &self.streaming_state.response_id,
                            self.owner.as_deref(),
                        );
                        self.files.push(file.clone());
                        Poll::Ready(Some(ResponsesStreamItem::FileProduced(file)))
                    }
                    _ => {
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    }
                }
            }
            Poll::Ready(None) => {
                let channel_error = anyhow::anyhow!("Response channel closed before completion.");
                let api_error =
                    classify_api_error(&self.state, channel_error.as_ref(), ApiErrorKind::Internal);
                let seq = self.streaming_state.next_sequence_number();
                let event = stream_error_from_api_error(seq, api_error);
                self.finish(None);
                Poll::Ready(Some(ResponsesStreamItem::Event(event)))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl OpenResponsesStreamEvent {
    /// The event's `type`, which is also its SSE event name.
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::ResponseCreated { .. } => "response.created",
            Self::ResponseInProgress { .. } => "response.in_progress",
            Self::OutputItemAdded { .. } => "response.output_item.added",
            Self::ContentPartAdded { .. } => "response.content_part.added",
            Self::OutputTextDelta { .. } => "response.output_text.delta",
            Self::ContentPartDone { .. } => "response.content_part.done",
            Self::OutputItemDone { .. } => "response.output_item.done",
            Self::FunctionCallArgumentsDelta { .. } => "response.function_call_arguments.delta",
            Self::FunctionCallArgumentsDone { .. } => "response.function_call_arguments.done",
            Self::ReasoningTextDelta { .. } => "response.reasoning_text.delta",
            Self::ReasoningTextDone { .. } => "response.reasoning_text.done",
            Self::ResponseCompleted { .. } => "response.completed",
            Self::ResponseFailed { .. } => "response.failed",
            Self::ResponseIncomplete { .. } => "response.incomplete",
            Self::ResponseCancelled { .. } => "response.cancelled",
            Self::Error { .. } => "error",
        }
    }
}

// The response cites its files under its container id; listing that container returns them and nothing else.
fn tag_with_container(
    state: &SharedInferenceRsState,
    file: &inference_core::File,
    response_id: &str,
    owner: Option<&str>,
) {
    let _ = state.try_tag_file(&file.id, &response_container_id(response_id), owner);
}

fn response_container_id(response_id: &str) -> String {
    let sanitized: String = response_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("cntr_{sanitized}")
}

fn container_file_annotations(
    response_id: &str,
    files: &[inference_core::File],
) -> Vec<Annotation> {
    let container_id = response_container_id(response_id);
    files
        .iter()
        .filter(|file| !file.content.is_error())
        .map(|file| Annotation::ContainerFileCitation {
            file_id: file.id.clone(),
            index: None,
            container_id: container_id.clone(),
            end_index: 0,
            filename: file.name.clone(),
            start_index: 0,
        })
        .collect()
}

fn output_text_with_file_annotations(
    text: String,
    response_id: &str,
    files: &[inference_core::File],
) -> OutputContent {
    let annotations = container_file_annotations(response_id, files);
    if annotations.is_empty() {
        OutputContent::text(text)
    } else {
        OutputContent::text_with_annotations(text, annotations)
    }
}

/// Convert chat completion response to ResponseResource
fn chat_response_to_response_resource(
    chat_resp: &ChatCompletionResponse,
    request_id: String,
    model: String,
    metadata: Option<Value>,
    request_ctx: &RequestContext,
    shell_output_items: &[OutputItem],
    files: &[inference_core::File],
) -> ResponseResource {
    let created_at = chat_resp.created;
    let mut resource = ResponseResource::new(request_id.clone(), model, created_at);
    resource.adapter_generation = chat_resp.adapter_generation.clone();

    let status = finished_status(
        chat_resp
            .choices
            .iter()
            .map(|choice| Some(choice.finish_reason.as_str())),
    );
    let mut output_items = shell_output_items.to_vec();
    let mut output_text_parts = Vec::new();
    let mut reasoning_parts = Vec::new();

    for choice in &chat_resp.choices {
        let choice_status =
            finished_item_status(finished_status([Some(choice.finish_reason.as_str())]));
        let mut content_items = Vec::new();

        // Handle text content
        if let Some(text) = &choice.message.content {
            output_text_parts.push(text.clone());
            content_items.push(output_text_with_file_annotations(
                text.clone(),
                &request_id,
                files,
            ));
        }

        // Handle reasoning content
        if let Some(reasoning) = &choice.message.reasoning_content {
            reasoning_parts.push(reasoning.clone());
            // Reasoning followed by a reply or tool call finished; reasoning alone ended with its choice.
            // reasoning models report no reply as `Some("")`
            let replied = choice
                .message
                .content
                .as_deref()
                .is_some_and(|text| !text.is_empty());
            let moved_on = replied || choice.message.tool_calls.is_some();
            output_items.push(OutputItem::reasoning(
                format!("rs_{}", Uuid::new_v4()),
                reasoning.clone(),
                if moved_on {
                    ItemStatus::Completed
                } else {
                    choice_status
                },
            ));
        }

        // Handle tool calls - convert to function_call output items
        if let Some(tool_calls) = &choice.message.tool_calls {
            for tool_call in tool_calls {
                let (name, namespace) = request_ctx.split_tool_name(&tool_call.function.name);
                let item = OutputItem::function_call(
                    format!("fc_{}", Uuid::new_v4()),
                    tool_call.id.clone(),
                    name,
                    namespace,
                    tool_call.function.arguments.clone(),
                    ItemStatus::Completed,
                );
                output_items.push(item);
            }
        }

        // Create message output item if there's content
        if !content_items.is_empty() {
            let item = OutputItem::message(
                format!("msg_{}", Uuid::new_v4()),
                content_items,
                choice_status,
            );
            output_items.push(item);
        }
    }

    resource.status = status;
    apply_incomplete_details(&mut resource);
    resource.output = output_items;
    resource.output_text = if output_text_parts.is_empty() {
        None
    } else {
        Some(output_text_parts.join(""))
    };
    resource.reasoning = if reasoning_parts.is_empty() {
        None
    } else {
        Some(reasoning_parts.join(""))
    };
    let mut resp_usage = ResponseUsage::new(
        chat_resp.usage.prompt_tokens,
        chat_resp.usage.completion_tokens,
    );
    if let Some(details) = &chat_resp.usage.prompt_tokens_details {
        resp_usage.input_tokens_details = Some(InputTokensDetails {
            cached_tokens: Some(details.cached_tokens),
            ..Default::default()
        });
    }
    resource.usage = Some(resp_usage);
    resource.metadata = metadata;
    resource.completed_at = Some(unix_now());

    request_ctx.echo_into(&mut resource);

    resource
}

/// Parse OpenResponses request into internal format
async fn parse_openresponses_request(
    oairequest: OpenResponsesCreateRequest,
    chat: &ChatEngine,
    tx: Sender<Response>,
) -> Result<(Request, Vec<Message>, RequestContext)> {
    let state = chat.state.clone();
    // parallel_tool_calls=false is accepted best-effort; max_tool_calls has no engine support
    if oairequest.max_tool_calls.is_some() {
        anyhow::bail!(
            "max_tool_calls is not supported. \
             inference.rs does not currently support limiting the number of tool calls."
        );
    }

    if oairequest.max_output_tokens == Some(0) {
        anyhow::bail!("max_output_tokens must be at least 1.");
    }

    // Build request context to echo back request parameters
    // Must capture these before consuming oairequest
    let request_context = RequestContext {
        tools: oairequest.tools.clone(),
        tool_choice: oairequest.tool_choice.clone(),
        parallel_tool_calls: oairequest.parallel_tool_calls,
        text: oairequest.text.clone(),
        temperature: oairequest.temperature,
        top_p: oairequest.top_p,
        presence_penalty: oairequest.presence_penalty,
        frequency_penalty: oairequest.frequency_penalty,
        top_logprobs: oairequest.top_logprobs,
        max_output_tokens: oairequest.max_output_tokens,
        max_tool_calls: oairequest.max_tool_calls,
        store: oairequest.store,
        background: oairequest.background,
    };

    // If previous_response_id is provided, get the full conversation history from cache
    let previous_messages = if let Some(prev_id) = &oairequest.previous_response_id {
        let cache = get_response_cache();
        match cache.get_conversation(prev_id, chat.owner.as_deref()) {
            Ok(Some(mut conversation)) => {
                // a follow-up on an older reply branches; continuing the session would rewrite its newer turns
                let head = conversation
                    .session_id
                    .as_deref()
                    .map(|session| cache.session_head(session, chat.owner.as_deref()))
                    .transpose()
                    .ok()
                    .flatten()
                    .flatten();
                if head.as_deref() != Some(prev_id.as_str()) {
                    conversation.session_id = None;
                }
                Some(conversation)
            }
            Ok(None) => {
                return Err(ApiError::new(
                    ApiErrorKind::NotFound,
                    format!("Previous response with ID '{prev_id}' was not found."),
                    Some("previous_response_not_found"),
                    Some("previous_response_id"),
                )
                .into());
            }
            Err(error) => {
                InferenceRs::maybe_log_error(state.clone(), error.as_ref());
                return Err(ApiError::internal().into());
            }
        }
    } else {
        None
    };

    // Get messages from input field
    let messages = oairequest.input.into_either();

    let StoredConversation {
        messages: mut final_messages,
        session_id,
    } = previous_messages.unwrap_or_default();
    match messages {
        Either::Left(msgs) => {
            let msgs = without_stored_calls(&final_messages, msgs);
            final_messages.extend(msgs);
        }
        Either::Right(prompt) => {
            final_messages.push(Message {
                content: Some(MessageContent::from_text(prompt)),
                role: "user".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }
    }

    if let Some(instructions) = oairequest.instructions.clone() {
        final_messages.insert(
            0,
            Message {
                content: Some(MessageContent::from_text(instructions)),
                role: SYSTEM_ROLE.to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        );
    }

    let reasoning_effort = oairequest
        .reasoning
        .as_ref()
        .and_then(|reasoning| reasoning.effort);
    inference_core::resolve_reasoning_controls(None, reasoning_effort)?;
    let reasoning_effort = reasoning_effort.map(|effort| effort.as_str().to_string());
    let enable_thinking = None;

    // Convert truncation enum to truncate_sequence bool
    let truncate_sequence = oairequest
        .truncation
        .map(|t| matches!(t, crate::responses_types::enums::TruncationStrategy::Auto));

    // Convert OpenResponses `text` field to `response_format`, falling back to legacy field
    let response_format = if let Some(text_config) = oairequest.text {
        text_config.format.map(|fmt| match fmt {
            TextFormat::Text => crate::openai::ResponseFormat::Text,
            TextFormat::JsonSchema {
                name,
                schema,
                strict: _,
            } => crate::openai::ResponseFormat::JsonSchema {
                json_schema: crate::openai::JsonSchemaResponseFormat {
                    name,
                    schema: schema.unwrap_or(serde_json::Value::Object(Default::default())),
                },
            },
            TextFormat::JsonObject => crate::openai::ResponseFormat::JsonObject,
        })
    } else {
        oairequest.response_format
    };

    // Convert to ChatCompletionRequest
    let mut chat_request = ChatCompletionRequest {
        messages: Either::Left(final_messages.clone()),
        model: oairequest.model,
        adapter: oairequest.adapter,
        logit_bias: oairequest.logit_bias,
        logprobs: oairequest.logprobs,
        top_logprobs: oairequest.top_logprobs,
        max_tokens: oairequest.max_output_tokens,
        n_choices: oairequest.n_choices,
        presence_penalty: oairequest.presence_penalty,
        frequency_penalty: oairequest.frequency_penalty,
        repetition_penalty: oairequest.repetition_penalty,
        stop_seqs: oairequest.stop_seqs,
        ignore_eos: oairequest.ignore_eos,
        seed: oairequest.seed,
        temperature: oairequest.temperature,
        top_p: oairequest.top_p,
        stream: oairequest.stream,
        tools: oairequest.tools,
        tool_choice: oairequest.tool_choice,
        response_format,
        web_search_options: None,
        agent_permission: None,
        code_execution_permission: None,
        enable_shell: false,
        shell_skill_references: Vec::new(),
        session_id,
        max_tool_rounds: oairequest.max_tool_rounds,
        top_k: oairequest.top_k,
        grammar: oairequest.grammar,
        min_p: oairequest.min_p,
        dry_multiplier: oairequest.dry_multiplier,
        dry_base: oairequest.dry_base,
        dry_allowed_length: oairequest.dry_allowed_length,
        dry_sequence_breakers: oairequest.dry_sequence_breakers,
        enable_thinking,
        truncate_sequence,
        logits_processors: oairequest.logits_processors,
        host_tools: oairequest.host_tools,
        reasoning_effort,
        chat_template_kwargs: None,
        files: oairequest.files,
    };

    chat.apply_agent_policy(&mut chat_request);
    let asks = chat_request.agent_permission == Some(AgentPermission::Ask);
    if asks && chat_request.stream != Some(true) {
        return Err(ApiError::new(
            ApiErrorKind::InvalidRequest,
            ASK_REQUIRES_STREAMING,
            Some("unsupported_parameter"),
            Some("agent_permission"),
        )
        .into());
    }
    let agent_approval_handler = asks.then(|| {
        AgentToolApprovalHandler::from_async(
            chat.agentic.approval_broker.callback(chat.owner.clone()),
        )
    });
    let agent_approval_notifier = asks.then(|| {
        chat.agentic
            .approval_broker
            .notifier(tx.clone(), chat.owner.clone())
    });
    let (request, _) = parse_chat_request(
        chat_request,
        ChatCompletionParseContext {
            state,
            tx,
            tool_dispatch_url: chat.agentic.tool_dispatch_url.clone(),
            agent_approval_handler,
            agent_approval_notifier,
            tool_surface: OpenAiToolSurface::Responses,
            skill_store: chat.skill_store.clone(),
            media: Default::default(),
            owner: chat.owner.clone(),
        },
    )
    .await?;
    Ok((request, final_messages, request_context))
}

fn response_not_found_error(response_id: &str) -> ApiError {
    ApiError::new(
        ApiErrorKind::NotFound,
        format!("Response with ID '{response_id}' was not found."),
        Some("response_not_found"),
        Some("response_id"),
    )
}

/// A dispatched Responses request and what to echo back in its response resource.
pub struct PreparedResponse {
    pub rx: Receiver<Response>,
    pub id: String,
    /// The model name the request asked for.
    pub model: String,
    pub metadata: Option<Value>,
    pub store: bool,
    pub stream: bool,
    pub background: bool,
    /// The conversation through this request, stored with the reply for `previous_response_id`.
    pub history: Vec<Message>,
    pub context: RequestContext,
    pub cancellation: RequestCancellation,
    /// The agent session the request continues, kept for the reply when its run reports none.
    pub session_id: Option<String>,
    /// Who the request acts for; the stored reply is theirs alone.
    pub owner: Option<String>,
}

/// Validates a Responses request, resolves the conversation it continues and sends it to its model.
pub(crate) fn prepare_response<'a>(
    chat: &'a ChatEngine,
    request: OpenResponsesCreateRequest,
) -> BoxFuture<'a, Result<PreparedResponse, DispatchError>> {
    Box::pin(prepare_response_inner(chat, request))
}

async fn prepare_response_inner(
    chat: &ChatEngine,
    mut request: OpenResponsesCreateRequest,
) -> Result<PreparedResponse, DispatchError> {
    let state = &chat.state;
    let stream = request.stream == Some(true);
    let background = request.background == Some(true);
    if background && stream {
        return Err(DispatchError::Validation(Box::new(
            unsupported_background_stream_error(),
        )));
    }
    let (tx, rx) = create_response_channel(None);
    let requested_model = request.model.clone();
    resolve_lora_adapter_model(state, &mut request.model, &mut request.adapter)
        .map_err(|error| DispatchError::Validation(Box::new(error)))?;
    let model = response_model_id(state, requested_model.clone(), &request.model)
        .unwrap_or(requested_model);
    let model_id = (request.model != DEFAULT_MODEL_ID).then(|| request.model.clone());
    let metadata = request.metadata.clone();
    let store = request.store.unwrap_or(true);
    let logits_processors = chat
        .logits_processors
        .resolve(request.logits_processors.as_deref())
        .map_err(|error| DispatchError::Validation(Box::new(error)))?;
    let host_tools = chat
        .host_tools
        .resolve(request.host_tools.as_deref())
        .map_err(|error| DispatchError::Validation(Box::new(error)))?;
    let (mut core_request, history, context) = parse_openresponses_request(request, chat, tx)
        .await
        .map_err(|error| DispatchError::Validation(boxed_anyhow(error)))?;
    let cancellation = RequestCancellation::default();
    let mut session_id = None;
    if let Request::Normal(normal) = &mut core_request {
        normal.cancellation = Some(cancellation.clone());
        normal.logits_processors = logits_processors;
        normal.host_tools = host_tools.unwrap_or_default();
        session_id.clone_from(&normal.session_id);
    }
    send_request_with_model(state, core_request, model_id.as_deref())
        .await
        .map_err(|error| DispatchError::Internal(error.into()))?;
    Ok(PreparedResponse {
        rx,
        id: format!("resp_{}", Uuid::new_v4()),
        model,
        metadata,
        store,
        stream,
        background,
        history,
        context,
        cancellation,
        session_id,
        owner: chat.owner.clone(),
    })
}

/// What a finished request leaves in the response cache when it asked to be stored.
struct StoredResponse {
    id: String,
    owner: Option<String>,
    response: ResponseResource,
    /// `None` for a response that is not a conversation to continue with `previous_response_id` (failed, cancelled).
    history: Option<StoredConversation>,
}

impl StoredResponse {
    fn save(self) {
        let cache = get_response_cache();
        let owner = self.owner.as_deref();
        let _ = cache.store_response(self.id.clone(), self.response, owner);
        if let Some(history) = self.history {
            let _ = cache.store_conversation(self.id, history, owner);
        }
    }
}

/// Waits for a non-streaming request's reply, storing it and its conversation when the request asked to.
pub(crate) fn collect_response<'a>(
    prepared: PreparedResponse,
    state: &'a SharedInferenceRsState,
) -> BoxFuture<'a, Result<ResponseResource, ApiError>> {
    Box::pin(collect_response_inner(prepared, state))
}

async fn collect_response_inner(
    prepared: PreparedResponse,
    state: &SharedInferenceRsState,
) -> Result<ResponseResource, ApiError> {
    let (result, stored) = run_to_end(prepared, state).await;
    if let Some(stored) = stored {
        stored.save();
    }
    result
}

async fn run_to_end(
    prepared: PreparedResponse,
    state: &SharedInferenceRsState,
) -> (Result<ResponseResource, ApiError>, Option<StoredResponse>) {
    let PreparedResponse {
        mut rx,
        id,
        model,
        metadata,
        store,
        mut history,
        context,
        session_id,
        owner,
        ..
    } = prepared;
    // Files stay reachable through the file store; the response cites them.
    let mut shell_output_items = Vec::new();
    let mut pending_shell_calls = HashMap::new();
    let mut files = Vec::new();
    let response = loop {
        match rx.recv().await {
            Some(Response::AgenticToolCallProgress {
                tool_call_id,
                phase,
                ..
            }) => {
                record_shell_progress_items(
                    &mut pending_shell_calls,
                    &mut shell_output_items,
                    &tool_call_id,
                    &phase,
                );
            }
            Some(Response::BlockDenoisingProgress(_)) => {}
            Some(Response::File(file)) => {
                tag_with_container(state, &file, &id, owner.as_deref());
                files.push(file);
            }
            other => break other,
        }
    };
    let resource = |chat_resp: &ChatCompletionResponse, metadata| {
        chat_response_to_response_resource(
            chat_resp,
            id.clone(),
            model.clone(),
            metadata,
            &context,
            &shell_output_items,
            &files,
        )
    };

    match response {
        Some(Response::Done(chat_resp)) => {
            let response = resource(&chat_resp, metadata);
            let stored = store.then(|| {
                for choice in &chat_resp.choices {
                    let tool_calls = client_tool_calls(&context, &choice.message);
                    if choice.message.content.is_some() || !tool_calls.is_empty() {
                        history.push(Message {
                            content: choice
                                .message
                                .content
                                .clone()
                                .map(MessageContent::from_text),
                            role: choice.message.role.clone(),
                            name: None,
                            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                            tool_call_id: None,
                            reasoning_content: choice.message.reasoning_content.clone(),
                        });
                    }
                }
                StoredResponse {
                    id: id.clone(),
                    owner: owner.clone(),
                    response: response.clone(),
                    history: (response.status != ResponseStatus::Cancelled).then(|| {
                        StoredConversation {
                            messages: history,
                            session_id: chat_resp.session_id.clone().or(session_id),
                        }
                    }),
                }
            });
            (Ok(response), stored)
        }
        Some(Response::ModelError(msg, partial)) => {
            InferenceRs::maybe_log_error(state.clone(), &ModelErrorMessage(msg.to_string()));
            let stored = store.then(|| {
                let mut response = resource(&partial, metadata);
                response.error = Some(response_error_from_api_error(ApiError::model_error()));
                response.status = ResponseStatus::Failed;
                StoredResponse {
                    id: id.clone(),
                    owner: owner.clone(),
                    response,
                    history: None,
                }
            });
            (Err(ApiError::model_error()), stored)
        }
        Some(Response::ValidationError(e)) => (
            Err(classify_api_error(
                state,
                e.as_ref(),
                ApiErrorKind::InvalidRequest,
            )),
            None,
        ),
        Some(Response::InternalError(e)) => (
            Err(classify_api_error(
                state,
                e.as_ref(),
                ApiErrorKind::Internal,
            )),
            None,
        ),
        _ => {
            let error = anyhow::anyhow!("No response received from the model.");
            (
                Err(classify_api_error(
                    state,
                    error.as_ref(),
                    ApiErrorKind::Internal,
                )),
                None,
            )
        }
    }
}

/// Answers `prepared` off the caller's task and returns its queued resource; [`get_response`] follows it.
pub(crate) fn spawn_background(
    prepared: PreparedResponse,
    state: SharedInferenceRsState,
) -> ResponseResource {
    let task_manager = get_background_task_manager();
    let id = prepared.id.clone();
    task_manager.create_task(
        id.clone(),
        prepared.model.clone(),
        prepared.cancellation.clone(),
        prepared.owner.clone(),
    );
    let queued = ResponseResource::new(id.clone(), prepared.model.clone(), unix_now())
        .with_status(ResponseStatus::Queued)
        .with_metadata(prepared.metadata.clone().unwrap_or(Value::Null));
    tokio::spawn(async move {
        task_manager.mark_in_progress(&id);
        let (result, stored) = run_to_end(prepared, &state).await;
        // A task cancelled or deleted meanwhile is not stored, so nothing continues from it.
        let current = match result {
            Ok(response) => task_manager.mark_completed(&id, response),
            Err(error) => task_manager.mark_failed(&id, response_error_from_api_error(error)),
        };
        if let (true, Some(stored)) = (current, stored) {
            stored.save();
        }
    });
    queued
}

/// A background response in whatever state it has reached, or a stored one.
pub(crate) fn get_response(
    state: &SharedInferenceRsState,
    response_id: &str,
    owner: Option<&str>,
) -> Result<ResponseResource, ApiError> {
    if let Some(response) = get_background_task_manager().get_response(response_id, owner) {
        return Ok(response);
    }
    match get_response_cache().get_response(response_id, owner) {
        Ok(Some(response)) => Ok(response),
        Ok(None) => Err(response_not_found_error(response_id)),
        Err(error) => Err(classify_api_error(
            state,
            error.as_ref(),
            ApiErrorKind::Internal,
        )),
    }
}

/// What deleting a response returns.
#[derive(Debug, Serialize, ToSchema)]
pub struct ResponseDeleted {
    pub id: String,
    pub object: &'static str,
    pub deleted: bool,
}

/// Forgets a response: its background task, the stored resource and its conversation.
pub(crate) fn delete_response(
    state: &SharedInferenceRsState,
    response_id: &str,
    owner: Option<&str>,
) -> Result<ResponseDeleted, ApiError> {
    let task_deleted = get_background_task_manager().delete_task(response_id, owner);
    match get_response_cache().delete_response(response_id, owner) {
        Ok(cache_deleted) if task_deleted || cache_deleted => Ok(ResponseDeleted {
            id: response_id.to_string(),
            object: "response.deleted",
            deleted: true,
        }),
        Ok(_) => Err(response_not_found_error(response_id)),
        Err(error) => Err(classify_api_error(
            state,
            error.as_ref(),
            ApiErrorKind::Internal,
        )),
    }
}

/// Cancels a queued or running background response and returns it; finished responses come back unchanged.
pub(crate) fn cancel_response(
    state: &SharedInferenceRsState,
    response_id: &str,
    owner: Option<&str>,
) -> Result<ResponseResource, ApiError> {
    get_background_task_manager().cancel(response_id, owner);
    get_response(state, response_id, owner)
}

/// The input minus `function_call` items the stored history holds, which a client resends with their outputs.
fn without_stored_calls(stored: &[Message], input: Vec<Message>) -> Vec<Message> {
    let stored_ids: std::collections::HashSet<&str> = stored
        .iter()
        .flat_map(|message| message.tool_calls.iter().flatten())
        .filter_map(|call| call.id.as_deref())
        .collect();
    input
        .into_iter()
        .filter_map(|mut message| {
            let Some(calls) = message.tool_calls.take() else {
                return Some(message);
            };
            let calls: Vec<ToolCall> = calls
                .into_iter()
                .filter(|call| call.id.as_deref().is_none_or(|id| !stored_ids.contains(id)))
                .collect();
            message.tool_calls = (!calls.is_empty()).then_some(calls);
            (message.tool_calls.is_some() || message.content.is_some()).then_some(message)
        })
        .collect()
}

/// The calls in a reply that the client answers; a server tool's call stopped by the round limit has no answer coming.
fn client_tool_calls(
    context: &RequestContext,
    message: &inference_core::ResponseMessage,
) -> Vec<ToolCall> {
    message
        .tool_calls
        .iter()
        .flatten()
        .filter(|call| context.defines_tool(&call.function.name))
        .map(history_tool_call)
        .collect()
}

/// A tool call the run returned, as history keeps it: the name the model emitted, so a follow-up matches it.
fn history_tool_call(call: &inference_core::ToolCallResponse) -> ToolCall {
    ToolCall {
        id: Some(call.id.clone()),
        tp: inference_core::ToolType::Function,
        function: crate::openai::FunctionCalled {
            name: call.function.name.clone(),
            arguments: call.function.arguments.clone(),
        },
    }
}

/// How a run that stopped ended: cancelled by its caller, cut off by its token cap, or finished.
fn finished_status<'a>(
    finish_reasons: impl IntoIterator<Item = Option<&'a str>>,
) -> ResponseStatus {
    let (mut cancelled, mut capped) = (false, false);
    for reason in finish_reasons.into_iter().flatten() {
        cancelled |= reason == FINISH_REASON_CANCELED;
        capped |= reason == FINISH_REASON_LENGTH;
    }
    if cancelled {
        ResponseStatus::Cancelled
    } else if capped {
        ResponseStatus::Incomplete
    } else {
        ResponseStatus::Completed
    }
}

/// The event that ends a stream, named for how its response ended.
fn terminal_event(sequence_number: u64, response: ResponseResource) -> OpenResponsesStreamEvent {
    match response.status {
        ResponseStatus::Cancelled => OpenResponsesStreamEvent::ResponseCancelled {
            sequence_number,
            response,
        },
        ResponseStatus::Incomplete => OpenResponsesStreamEvent::ResponseIncomplete {
            sequence_number,
            response,
        },
        _ => OpenResponsesStreamEvent::ResponseCompleted {
            sequence_number,
            response,
        },
    }
}

/// A message item in a run that ended: completed only if the run did, else cut short.
fn finished_item_status(status: ResponseStatus) -> ItemStatus {
    match status {
        ResponseStatus::Completed => ItemStatus::Completed,
        _ => ItemStatus::Incomplete,
    }
}

fn apply_incomplete_details(resource: &mut ResponseResource) {
    if resource.status == ResponseStatus::Incomplete {
        resource.incomplete_details = Some(IncompleteDetails::max_output_tokens());
    }
}

/// What an item that never finished reports in a response that has stopped: cut short once the response is over.
fn unfinished_item_status(response: ResponseStatus) -> ItemStatus {
    match response {
        ResponseStatus::Cancelled | ResponseStatus::Incomplete | ResponseStatus::Failed => {
            ItemStatus::Incomplete
        }
        _ => ItemStatus::InProgress,
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stream_error_matches_openai_shape() {
        let event = OpenResponsesStreamEvent::Error {
            sequence_number: 7,
            code: "invalid_request".to_string(),
            message: "Invalid model.".to_string(),
            param: Some("model".to_string()),
        };

        assert_eq!(
            serde_json::to_value(event).unwrap(),
            json!({
                "type": "error",
                "sequence_number": 7,
                "code": "invalid_request",
                "message": "Invalid model.",
                "param": "model"
            })
        );
    }

    #[test]
    fn background_streams_are_a_typed_bad_request() {
        assert_eq!(
            unsupported_background_stream_error().to_openai_body(),
            json!({
                "error": {
                    "message": "`background: true` with `stream: true` is not supported by this server.",
                    "type": "invalid_request_error",
                    "param": "background,stream",
                    "code": "unsupported_parameter_combination"
                }
            })
        );
    }

    #[test]
    fn ignore_eos_defaults_false_and_accepts_true() {
        let default: OpenResponsesCreateRequest =
            serde_json::from_value(json!({"input": "hello"})).unwrap();
        let enabled: OpenResponsesCreateRequest =
            serde_json::from_value(json!({"input": "hello", "ignore_eos": true})).unwrap();

        assert!(!default.ignore_eos);
        assert!(enabled.ignore_eos);
    }

    #[test]
    fn request_accepts_sampling_seed() {
        let request: OpenResponsesCreateRequest =
            serde_json::from_value(json!({"input": "hello", "seed": 42})).unwrap();

        assert_eq!(request.seed, Some(42));
    }

    #[test]
    fn namespace_tools_round_trip_through_request_and_output() {
        let request: OpenResponsesCreateRequest = serde_json::from_value(json!({
            "input": [
                { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "hi" }] },
                { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "ok" }] },
                { "type": "reasoning", "id": "rs_1", "summary": [],
                  "content": [{ "type": "reasoning_text", "text": "thinking" }] },
                { "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "close_agent",
                  "namespace": "multi_agent_v1", "arguments": "{}" },
                { "type": "function_call_output", "id": "fco_1", "call_id": "call_1", "output": "done" }
            ],
            "tools": [
                { "type": "function", "name": "exec_command" },
                { "type": "namespace", "name": "multi_agent_v1", "tools": [
                    { "type": "function", "name": "close_agent" }
                ] },
                { "type": "web_search", "external_web_access": false }
            ]
        }))
        .unwrap();

        let OpenResponsesInput::Items(items) = request.input else {
            panic!("expected item input");
        };
        let messages = convert_input_items_to_messages(items);
        let call = messages[2].tool_calls.as_ref().unwrap();
        assert_eq!(call[0].function.name, "multi_agent_v1.close_agent");

        let ctx = RequestContext {
            tools: request.tools,
            ..Default::default()
        };
        assert_eq!(
            ctx.split_tool_name("multi_agent_v1.close_agent"),
            (
                "close_agent".to_string(),
                Some("multi_agent_v1".to_string())
            )
        );
        assert_eq!(
            ctx.split_tool_name("exec_command"),
            ("exec_command".to_string(), None)
        );

        let item = OutputItem::function_call(
            "fc_x".to_string(),
            "call_x".to_string(),
            "exec_command".to_string(),
            None,
            "{}".to_string(),
            ItemStatus::Completed,
        );
        assert!(!serde_json::to_string(&item).unwrap().contains("namespace"));
    }

    #[test]
    fn roles_pass_through_and_text_parts_collapse() {
        let items: Vec<InputItem> = serde_json::from_value(json!([
            { "type": "message", "role": "developer", "content": [
                { "type": "input_text", "text": "a" }, { "type": "input_text", "text": "b" } ] },
            { "type": "message", "role": "user", "content": [
                { "type": "input_text", "text": "hi" },
                { "type": "input_image", "image_url": "http://x/y.png" } ] }
        ]))
        .unwrap();
        let messages = convert_input_items_to_messages(items);
        assert_eq!(messages[0].role, "developer");
        assert_eq!(
            messages[0].content.as_ref().unwrap().as_text().as_deref(),
            Some("a\n\nb")
        );
        assert!(messages[1].content.as_ref().unwrap().as_text().is_none());
    }

    #[test]
    fn message_output_item_reuses_id_across_stream_lifecycle() {
        let message_item = MessageOutputItemState::new();
        let added = message_item.added_item();
        let done = message_item.item_with_text(
            "hello".to_string(),
            "resp_test",
            &[],
            ItemStatus::Completed,
        );
        let mut response = ResponseResource::new("resp_test".to_string(), "model".to_string(), 0);
        response.output.push(done.clone());

        assert_eq!(added.id(), done.id());
        assert_eq!(done.id(), response.output[0].id());
        assert_eq!(added.status(), ItemStatus::InProgress);
        assert_eq!(done.status(), ItemStatus::Completed);

        let OutputItem::Message { content, .. } = done else {
            panic!("expected message output item");
        };
        let [OutputContent::OutputText { text, .. }] = content.as_slice() else {
            panic!("expected one output text content part");
        };
        assert_eq!(text, "hello");
    }

    fn chat_response(choices: Vec<inference_core::Choice>) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: "chatcmpl_test".to_string(),
            choices,
            created: 1,
            model: "base-model".to_string(),
            system_fingerprint: "local".to_string(),
            object: "chat.completion".to_string(),
            usage: inference_core::Usage {
                completion_tokens: 0,
                prompt_tokens: 0,
                total_tokens: 0,
                prompt_tokens_details: None,
                avg_tok_per_sec: 0.0,
                avg_prompt_tok_per_sec: 0.0,
                avg_compl_tok_per_sec: 0.0,
                total_time_sec: 0.0,
                total_prompt_time_sec: 0.0,
                total_completion_time_sec: 0.0,
            },
            adapter_generation: None,
            agentic_tool_calls: None,
            files: None,
            session_id: None,
        }
    }

    fn choice(finish_reason: &str, content: Option<&str>) -> inference_core::Choice {
        inference_core::Choice {
            finish_reason: finish_reason.to_string(),
            stop_sequence: None,
            index: 0,
            message: inference_core::ResponseMessage {
                content: content.map(str::to_string),
                role: "assistant".to_string(),
                tool_calls: None,
                reasoning_content: Some("thinking".to_string()),
            },
            logprobs: None,
        }
    }

    #[test]
    fn reasoning_cut_off_by_the_cap_is_incomplete_but_reasoning_before_a_reply_is_not() {
        let convert = |choice| {
            chat_response_to_response_resource(
                &chat_response(vec![choice]),
                "resp_test".to_string(),
                "base-model".to_string(),
                None,
                &RequestContext::default(),
                &[],
                &[],
            )
        };
        let reasoning_status = |response: &ResponseResource| {
            response
                .output
                .iter()
                .find(|item| matches!(item, OutputItem::Reasoning { .. }))
                .map(OutputItem::status)
        };

        let capped = convert(choice(FINISH_REASON_LENGTH, None));
        assert_eq!(capped.status, ResponseStatus::Incomplete);
        assert_eq!(reasoning_status(&capped), Some(ItemStatus::Incomplete));
        // what a tag-based reasoning model reports when the cap lands before its reply
        let capped_empty = convert(choice(FINISH_REASON_LENGTH, Some("")));
        assert_eq!(
            reasoning_status(&capped_empty),
            Some(ItemStatus::Incomplete)
        );

        let capped_reply = convert(choice(FINISH_REASON_LENGTH, Some("ok")));
        assert_eq!(reasoning_status(&capped_reply), Some(ItemStatus::Completed));

        let finished = convert(choice("stop", None));
        assert_eq!(reasoning_status(&finished), Some(ItemStatus::Completed));
    }

    #[test]
    fn response_resource_preserves_the_request_facing_model() {
        let chat_response = ChatCompletionResponse {
            id: "chatcmpl_test".to_string(),
            choices: Vec::new(),
            created: 1,
            model: "base-model".to_string(),
            system_fingerprint: "local".to_string(),
            object: "chat.completion".to_string(),
            usage: inference_core::Usage {
                completion_tokens: 0,
                prompt_tokens: 0,
                total_tokens: 0,
                prompt_tokens_details: None,
                avg_tok_per_sec: 0.0,
                avg_prompt_tok_per_sec: 0.0,
                avg_compl_tok_per_sec: 0.0,
                total_time_sec: 0.0,
                total_prompt_time_sec: 0.0,
                total_completion_time_sec: 0.0,
            },
            adapter_generation: Some("generation".to_string()),
            agentic_tool_calls: None,
            files: None,
            session_id: None,
        };

        let response = chat_response_to_response_resource(
            &chat_response,
            "resp_test".to_string(),
            "base-model::production".to_string(),
            None,
            &RequestContext::default(),
            &[],
            &[],
        );

        assert_eq!(response.model, "base-model::production");
        assert_eq!(response.adapter_generation.as_deref(), Some("generation"));
    }
}

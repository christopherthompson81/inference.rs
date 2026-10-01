//! Chat completions as an engine operation, free of HTTP: the server routes and the C ABI drive these.

use std::{ops::Deref, pin::Pin, sync::Arc, task::Poll};

use anyhow::{Context, Result};
use either::Either;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use inference_core::{
    AgentPermission, AgentToolApprovalHandler, AgentToolApprovalNotifier,
    ChatCompletionChunkResponse, ChatResponseCollector, Constraint, InferenceRs, MessageContent,
    ModelCategory, NormalRequest, Request, RequestMessage, SamplingParams, TokenizationRequest,
    encode_agentic_tool_images, is_chat_template_request_error,
};
pub use inference_core::{ReasoningEffort, resolve_reasoning_controls};
use itertools::Itertools;
use serde_json::{Value, json};
use tokio::sync::mpsc::{Receiver, Sender};

use crate::{
    agentic::AgenticDefaults,
    api_error::{ApiError, ApiErrorKind, JsonError, ModelErrorMessage, boxed_anyhow},
    dispatch::{
        apply_model_override, create_response_channel, response_model_id, send_request_with_model,
    },
    input_files::{InputFileSpec, resolve_input_file},
    logits_processors::LogitsProcessors,
    lora_routing::{DEFAULT_MODEL_ID, resolve_lora_adapter_model},
    media_source::MediaAttachments,
    openai::{
        ChatCompletionRequest, Grammar, JsonSchemaResponseFormat, Message, MessageInnerContent,
        OpenAiToolSurface, ResponseFormat, normalize_chat_completion_tools,
        normalize_responses_tools, validate_openai_tool_choice,
    },
    registry::HostTools,
    sampling::{convert_stop_tokens, get_dry_sampling_params},
    skill_store::SkillStore,
    types::SharedInferenceRsState,
    util::{parse_audio_url_for_server, parse_image_url_for_server, validate_model_name},
    video::parse_video_url_for_server,
};

const ONLY_CHAT_IS_TOKENIZED: &str = "only chat messages can be tokenized as a chat";
pub(crate) const ASK_REQUIRES_STREAMING: &str = "agent_permission \"ask\" requires stream=true, so approval requests can be delivered and answered.";

pub fn serialize_agentic_progress(
    round: usize,
    tool_call_id: &str,
    tool_name: &str,
    phase: &AgenticToolCallPhase,
) -> Value {
    let (phase_str, data) = match phase {
        AgenticToolCallPhase::Calling(data) => ("calling", serialize_agentic_data(data)),
        AgenticToolCallPhase::Complete(data) => ("complete", serialize_agentic_data(data)),
    };
    json!({
        "type": "agentic_tool_call_progress",
        "round": round,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "phase": phase_str,
        "data": data,
    })
}

/// The payload of an `agentic_tool_approval_required` event; answer it with the engine's approval resolution.
pub fn serialize_approval_required(
    approval_id: &str,
    session_id: &str,
    round: usize,
    tool: &inference_core::AgentToolMetadata,
    arguments: &Value,
) -> Value {
    json!({
        "type": "agentic_tool_approval_required",
        "approval_id": approval_id,
        "session_id": session_id,
        "round": round,
        "tool": tool,
        "arguments": arguments,
    })
}

fn serialize_agentic_data(data: &AgenticToolCallData) -> Value {
    match data {
        AgenticToolCallData::CodeExecution {
            code,
            stdout,
            stderr,
            exception,
            images,
            video_frames,
            video_frame_count,
            working_directory,
            execution_time_ms,
        } => {
            let mut v = json!({"tool_type": "code_execution"});
            if let Some(c) = code {
                v["code"] = json!(c);
            }
            if let Some(s) = stdout {
                v["stdout"] = json!(s);
            }
            if let Some(s) = stderr {
                v["stderr"] = json!(s);
            }
            if let Some(e) = exception {
                v["exception"] = json!(e);
            }
            if !images.is_empty() {
                v["images_base64"] = json!(encode_agentic_tool_images(images));
            }
            if !video_frames.is_empty() {
                v["video_frames_base64"] = json!(encode_agentic_tool_images(video_frames));
            }
            if let Some(n) = video_frame_count {
                v["video_frame_count"] = json!(n);
            }
            if let Some(d) = working_directory {
                v["working_directory"] = json!(d);
            }
            if let Some(ms) = execution_time_ms {
                v["execution_time_ms"] = json!(ms);
            }
            v
        }
        AgenticToolCallData::WebSearch {
            query,
            results_count,
            sources,
        } => {
            let mut v = json!({"tool_type": "web_search"});
            if let Some(q) = query {
                v["query"] = json!(q);
            }
            if let Some(n) = results_count {
                v["results_count"] = json!(n);
            }
            if !sources.is_empty() {
                v["sources"] = json!(sources);
            }
            v
        }
        AgenticToolCallData::Shell {
            commands,
            stdout,
            stderr,
            exit_code,
            status,
            working_directory,
            timed_out,
        } => {
            let mut v = json!({"tool_type": "shell", "commands": commands});
            if let Some(s) = stdout {
                v["stdout"] = json!(s);
            }
            if let Some(s) = stderr {
                v["stderr"] = json!(s);
            }
            if let Some(code) = exit_code {
                v["exit_code"] = json!(code);
            }
            if let Some(s) = status {
                v["status"] = json!(s);
            }
            if let Some(d) = working_directory {
                v["working_directory"] = json!(d);
            }
            if let Some(t) = timed_out {
                v["timed_out"] = json!(t);
            }
            v
        }
        AgenticToolCallData::Custom { arguments, content } => {
            let mut v = json!({"tool_type": "custom"});
            if !arguments.is_empty() {
                v["arguments"] = json!(arguments);
            }
            if !content.is_empty() {
                v["content"] = json!(content);
            }
            v
        }
    }
}

fn parse_reasoning_controls(
    enable_thinking: Option<bool>,
    effort: Option<&str>,
) -> Result<(Option<bool>, Option<ReasoningEffort>)> {
    let effort = effort.map(str::parse).transpose()?;
    resolve_reasoning_controls(enable_thinking, effort)?;
    Ok((enable_thinking, effort))
}

fn insert_reasoning_content(output: &mut IndexMap<String, MessageContent>, message: &Message) {
    if let Some(reasoning_content) = &message.reasoning_content {
        output.insert(
            "reasoning_content".to_string(),
            Either::Left(reasoning_content.clone()),
        );
    }
}

pub(crate) struct ChatCompletionParseContext {
    pub state: SharedInferenceRsState,
    pub tx: Sender<Response>,
    pub tool_dispatch_url: Option<String>,
    pub agent_approval_handler: Option<AgentToolApprovalHandler>,
    pub agent_approval_notifier: Option<Arc<AgentToolApprovalNotifier>>,
    pub tool_surface: OpenAiToolSurface,
    pub skill_store: Option<Arc<SkillStore>>,
    /// Buffers the request's `media://N` sources name.
    pub media: MediaAttachments,
    pub owner: Option<String>,
}

/// Parses and validates a chat completion request.
///
/// This function transforms an OpenAI-compatible chat completion request into the
/// request format used by inference.rs.
pub(crate) fn parse_request(
    oairequest: ChatCompletionRequest,
    ctx: ChatCompletionParseContext,
) -> BoxFuture<'static, Result<(Request, bool)>> {
    Box::pin(parse_request_inner(oairequest, ctx))
}

async fn parse_request_inner(
    oairequest: ChatCompletionRequest,
    ctx: ChatCompletionParseContext,
) -> Result<(Request, bool)> {
    let ChatCompletionParseContext {
        state,
        tx,
        tool_dispatch_url,
        agent_approval_handler,
        agent_approval_notifier,
        tool_surface,
        skill_store,
        media,
        owner,
    } = ctx;
    let repr = serde_json::to_string(&oairequest)
        .context("Failed to serialize chat completion request for logging")?;
    InferenceRs::maybe_log_request(state.clone(), repr);

    // Validate that the requested model matches the loaded model
    validate_model_name(&oairequest.model, state.clone())?;
    // Before any media is fetched, so a malformed generation ID costs no downloads.
    let adapter = oairequest
        .adapter
        .clone()
        .map(crate::lora_routing::core_adapter_selection)
        .transpose()?;

    let mut enable_thinking = oairequest.enable_thinking;
    let mut reasoning_effort = oairequest.reasoning_effort.clone();
    if let Some(kwargs) = &oairequest.chat_template_kwargs {
        for (key, value) in kwargs {
            match (key.as_str(), value) {
                ("enable_thinking", Value::Bool(flag)) if enable_thinking.is_none() => {
                    enable_thinking = Some(*flag);
                }
                ("reasoning_effort", Value::String(effort)) if reasoning_effort.is_none() => {
                    reasoning_effort = Some(effort.clone());
                }
                ("enable_thinking" | "reasoning_effort", _) => {}
                _ => tracing::warn!("Ignoring unsupported chat_template_kwargs entry `{key}`"),
            }
        }
    }
    let (enable_thinking, reasoning_effort) =
        parse_reasoning_controls(enable_thinking, reasoning_effort.as_deref())?;

    let mut normalized_tools = match tool_surface {
        OpenAiToolSurface::ChatCompletions => {
            normalize_chat_completion_tools(oairequest.tools, oairequest.web_search_options)?
        }
        OpenAiToolSurface::Responses => normalize_responses_tools(oairequest.tools)?,
    };
    normalized_tools.enable_shell |= oairequest.enable_shell;
    normalized_tools
        .shell_skill_references
        .extend(oairequest.shell_skill_references);
    validate_openai_tool_choice(oairequest.tool_choice.as_ref(), &normalized_tools)?;
    let shell_options = if normalized_tools.shell_skill_references.is_empty() {
        None
    } else {
        let store = skill_store
            .as_ref()
            .context("tools[].type=\"shell\" skill references require a configured skill store.")?;
        Some(store.resolve_references(&normalized_tools.shell_skill_references, owner.as_deref())?)
    };

    let stop_toks = convert_stop_tokens(oairequest.stop_seqs, oairequest.stop_token_ids);
    let mut input_files = Vec::new();

    let messages = match oairequest.messages {
        Either::Left(req_messages) => {
            let mut messages = Vec::new();
            let mut image_urls = Vec::new();
            let mut audio_urls = Vec::new();
            let mut video_urls = Vec::new();
            for message in req_messages {
                let content = match message.content.as_deref() {
                    Some(content) => content.clone(),
                    None => {
                        // Templates render tool_calls themselves; HF treats a missing content as empty text
                        message.tool_calls.as_ref().context(
                            "No content was provided, expected tool calls to be provided.",
                        )?;
                        Either::Left(String::new())
                    }
                };

                match &content {
                    Either::Left(content) => {
                        let mut message_map: IndexMap<
                            String,
                            Either<String, Vec<IndexMap<String, Value>>>,
                        > = IndexMap::new();
                        message_map.insert("role".to_string(), Either::Left(message.role.clone()));
                        message_map.insert("content".to_string(), Either::Left(content.clone()));
                        insert_reasoning_content(&mut message_map, &message);

                        // Add tool_calls for assistant messages that have them
                        if let Some(ref tool_calls) = message.tool_calls {
                            // Convert tool_calls to Vec<IndexMap<String, Value>> for Jinja template
                            let tool_calls_vec: Vec<IndexMap<String, Value>> = tool_calls
                                .iter()
                                .map(|tc| {
                                    let mut tc_map = IndexMap::new();
                                    // Use provided ID or fallback to function name
                                    let id =
                                        tc.id.clone().unwrap_or_else(|| tc.function.name.clone());
                                    tc_map.insert("id".to_string(), Value::String(id));
                                    tc_map.insert(
                                        "type".to_string(),
                                        Value::String("function".to_string()),
                                    );
                                    let mut function_map = serde_json::Map::new();
                                    function_map.insert(
                                        "name".to_string(),
                                        Value::String(tc.function.name.clone()),
                                    );
                                    function_map.insert(
                                        "arguments".to_string(),
                                        Value::String(tc.function.arguments.clone()),
                                    );
                                    tc_map.insert(
                                        "function".to_string(),
                                        Value::Object(function_map),
                                    );
                                    tc_map
                                })
                                .collect();
                            message_map
                                .insert("tool_calls".to_string(), Either::Right(tool_calls_vec));
                        }

                        // Add tool_call_id for tool messages
                        if let Some(ref tool_call_id) = message.tool_call_id {
                            message_map.insert(
                                "tool_call_id".to_string(),
                                Either::Left(tool_call_id.clone()),
                            );
                        }

                        // Add name for tool messages
                        if let Some(ref name) = message.name {
                            message_map.insert("name".to_string(), Either::Left(name.clone()));
                        }

                        messages.push(message_map);
                    }
                    Either::Right(image_messages) => {
                        // If there is only one message, it is possible a text message
                        // found when rig is used as client. In this case, we need to check if
                        // the message is a text message or an image message.
                        if image_messages.len() == 1 && !image_messages[0].contains_key("type") {
                            if !image_messages[0].contains_key("text") {
                                anyhow::bail!("Expected `text` key in input message.");
                            }
                            let content = match image_messages[0]["text"].deref() {
                                Either::Left(left) => left.to_string(),
                                Either::Right(right) => format!("{right:?}"),
                            };
                            let mut message_map: IndexMap<
                                String,
                                Either<String, Vec<IndexMap<String, Value>>>,
                            > = IndexMap::new();
                            message_map
                                .insert("role".to_string(), Either::Left(message.role.clone()));
                            message_map.insert("content".to_string(), Either::Left(content));
                            insert_reasoning_content(&mut message_map, &message);
                            messages.push(message_map);
                            continue;
                        }
                        if message.role != "user" {
                            anyhow::bail!(
                                "Role for an image message must be `user`, but it is {}",
                                message.role
                            );
                        }

                        enum ContentPart {
                            Text { text: String },
                            Image { image_url: String },
                            Audio { audio_url: String },
                            Video { video_url: String },
                            File { spec: InputFileSpec },
                        }

                        let mut items = Vec::new();
                        for image_message in image_messages {
                            match image_message.get("type") {
                                Some(MessageInnerContent(Either::Left(x))) if x == "text" => {
                                    items.push(ContentPart::Text {
                                        text: image_message
                                            .get("text")
                                            .as_ref()
                                            .context("Text sub-content must have `text` key.")?
                                            .as_ref()
                                            .left()
                                            .context(
                                                "Text sub-content `text` key must be a string.",
                                            )?
                                            .clone(),
                                    });
                                }
                                Some(MessageInnerContent(Either::Left(x))) if x == "image_url" => {
                                    items.push(ContentPart::Image {
                                        image_url: image_message
                                            .get("image_url")
                                            .as_ref()
                                            .context("Image sub-content must have `image_url` key.")?
                                            .as_ref()
                                            .right()
                                            .context("Image sub-content `image_url` key must be an object.")?
                                            .get("url")
                                            .context("Image sub-content `image_url` object must have a `url` key.")?
                                            .clone(),
                                    });
                                }
                                Some(MessageInnerContent(Either::Left(x))) if x == "audio_url" => {
                                    items.push(ContentPart::Audio {
                                        audio_url: image_message
                                            .get("audio_url")
                                            .as_ref()
                                            .context("Audio sub-content must have `audio_url` key.")?
                                            .as_ref()
                                            .right()
                                            .context("Audio sub-content `audio_url` key must be an object.")?
                                            .get("url")
                                            .context("Audio sub-content `audio_url` object must have a `url` key.")?
                                            .clone(),
                                    });
                                }
                                Some(MessageInnerContent(Either::Left(x))) if x == "video_url" => {
                                    items.push(ContentPart::Video {
                                        video_url: image_message
                                            .get("video_url")
                                            .as_ref()
                                            .context("Video sub-content must have `video_url` key.")?
                                            .as_ref()
                                            .right()
                                            .context("Video sub-content `video_url` key must be an object.")?
                                            .get("url")
                                            .context("Video sub-content `video_url` object must have a `url` key.")?
                                            .clone(),
                                    });
                                }
                                Some(MessageInnerContent(Either::Left(x))) if x == "file" => {
                                    let file = image_message
                                        .get("file")
                                        .as_ref()
                                        .context("File sub-content must have `file` key.")?
                                        .as_ref()
                                        .right()
                                        .context(
                                            "File sub-content `file` key must be an object.",
                                        )?;
                                    let spec = InputFileSpec {
                                        file_id: file.get("file_id").cloned(),
                                        file_data: file.get("file_data").cloned(),
                                        file_url: file.get("file_url").cloned(),
                                        filename: file.get("filename").cloned(),
                                    };
                                    if spec.file_url.is_some()
                                        && matches!(
                                            tool_surface,
                                            OpenAiToolSurface::ChatCompletions
                                        )
                                    {
                                        anyhow::bail!(
                                            "Chat Completions file content does not support `file_url`; use Responses `input_file`."
                                        );
                                    }
                                    items.push(ContentPart::File { spec });
                                }
                                _ => anyhow::bail!(
                                    "Expected array content sub-content to be one of `text`, `image_url`, `audio_url`, `video_url`, or `file`."
                                ),
                            }
                        }

                        let text_content = items
                            .iter()
                            .filter_map(|item| match item {
                                ContentPart::Text { text } => Some(text),
                                _ => None,
                            })
                            .join(" ");
                        let image_urls_iter = items
                            .iter()
                            .filter_map(|item| match item {
                                ContentPart::Image { image_url } => Some(image_url.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>();

                        let audio_urls_iter = items
                            .iter()
                            .filter_map(|item| match item {
                                ContentPart::Audio { audio_url } => Some(audio_url.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>();

                        let video_urls_iter = items
                            .iter()
                            .filter_map(|item| match item {
                                ContentPart::Video { video_url } => Some(video_url.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        let file_specs_iter = items
                            .iter()
                            .filter_map(|item| match item {
                                ContentPart::File { spec } => Some(spec.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        for spec in file_specs_iter {
                            let file = resolve_input_file(
                                state.clone(),
                                spec,
                                "input_file",
                                owner.as_deref(),
                            )
                            .await?;
                            state.insert_file(None, file.clone(), None, owner.as_deref())?;
                            input_files.push(file);
                        }

                        // Apply prefixer to text content if this is a multimodal model with images/audio/video
                        // This matches the behavior of interactive mode which auto-inserts media tokens
                        let text_content = if !image_urls_iter.is_empty()
                            || !audio_urls_iter.is_empty()
                            || !video_urls_iter.is_empty()
                        {
                            match state.get_model_category(None) {
                                Ok(ModelCategory::Multimodal { prefixer, .. }) => {
                                    let mut prefixed = text_content;

                                    // Apply image prefixer
                                    if !image_urls_iter.is_empty() {
                                        let start_idx = image_urls.len();
                                        let image_indices: Vec<usize> = (start_idx
                                            ..start_idx + image_urls_iter.len())
                                            .collect();
                                        prefixed = prefixer.prefix_image(image_indices, &prefixed);
                                    }

                                    // Apply audio prefixer
                                    if !audio_urls_iter.is_empty() {
                                        let start_idx = audio_urls.len();
                                        let audio_indices: Vec<usize> = (start_idx
                                            ..start_idx + audio_urls_iter.len())
                                            .collect();
                                        prefixed = prefixer.prefix_audio(audio_indices, &prefixed);
                                    }

                                    // Apply video prefixer
                                    if !video_urls_iter.is_empty() {
                                        let start_idx = video_urls.len();
                                        let video_indices: Vec<usize> = (start_idx
                                            ..start_idx + video_urls_iter.len())
                                            .collect();
                                        prefixed = prefixer.prefix_video(video_indices, &prefixed);
                                    }

                                    prefixed
                                }
                                _ => text_content,
                            }
                        } else {
                            text_content
                        };

                        let mut message_map: IndexMap<
                            String,
                            Either<String, Vec<IndexMap<String, Value>>>,
                        > = IndexMap::new();
                        message_map.insert("role".to_string(), Either::Left(message.role));

                        let mut content_map: Vec<IndexMap<String, Value>> = Vec::new();
                        for _ in &image_urls_iter {
                            let mut content_image_map = IndexMap::new();
                            content_image_map
                                .insert("type".to_string(), Value::String("image".to_string()));
                            content_map.push(content_image_map);
                        }
                        for _ in &audio_urls_iter {
                            let mut content_audio_map = IndexMap::new();
                            content_audio_map
                                .insert("type".to_string(), Value::String("audio".to_string()));
                            content_map.push(content_audio_map);
                        }
                        for _ in &video_urls_iter {
                            let mut content_video_map = IndexMap::new();
                            content_video_map
                                .insert("type".to_string(), Value::String("video".to_string()));
                            content_map.push(content_video_map);
                        }
                        {
                            let mut content_text_map = IndexMap::new();
                            content_text_map
                                .insert("type".to_string(), Value::String("text".to_string()));
                            content_text_map
                                .insert("text".to_string(), Value::String(text_content));
                            content_map.push(content_text_map);
                        }

                        message_map.insert("content".to_string(), Either::Right(content_map));
                        messages.push(message_map);
                        image_urls.extend(image_urls_iter);
                        audio_urls.extend(audio_urls_iter);
                        video_urls.extend(video_urls_iter);
                    }
                }
            }
            if !image_urls.is_empty() || !audio_urls.is_empty() || !video_urls.is_empty() {
                // Parse images
                let mut images = Vec::new();
                for url_unparsed in image_urls {
                    let image = parse_image_url_for_server(&url_unparsed, &media)
                        .await
                        .context(format!("Failed to parse image resource: {url_unparsed}"))?;
                    images.push(image);
                }

                // Parse audios
                let mut audios = Vec::new();
                for url_unparsed in audio_urls {
                    let audio = parse_audio_url_for_server(&url_unparsed, &media)
                        .await
                        .context(format!("Failed to parse audio resource: {url_unparsed}"))?;
                    audios.push(audio);
                }

                // Parse videos
                let video_sampling = match state.get_model_category(None) {
                    Ok(ModelCategory::Multimodal { video_sampling, .. }) => Some(video_sampling),
                    _ => None,
                };
                let mut videos = Vec::new();
                for url_unparsed in video_urls {
                    let video = parse_video_url_for_server(&url_unparsed, video_sampling, &media)
                        .await
                        .context(format!("Failed to parse video resource: {url_unparsed}"))?;
                    videos.push(video);
                }

                RequestMessage::MultimodalChat {
                    messages,
                    images,
                    audios,
                    videos,
                    enable_thinking,
                    reasoning_effort,
                }
            } else {
                RequestMessage::Chat {
                    messages,
                    enable_thinking,
                    reasoning_effort,
                }
            }
        }
        Either::Right(prompt) => {
            let mut messages = Vec::new();
            let mut message_map: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
                IndexMap::new();
            message_map.insert("role".to_string(), Either::Left("user".to_string()));
            message_map.insert("content".to_string(), Either::Left(prompt));
            messages.push(message_map);
            RequestMessage::Chat {
                messages,
                enable_thinking,
                reasoning_effort,
            }
        }
    };

    let dry_params = get_dry_sampling_params(
        oairequest.dry_multiplier,
        oairequest.dry_sequence_breakers,
        oairequest.dry_base,
        oairequest.dry_allowed_length,
    )?;

    if oairequest.max_tokens == Some(0) {
        anyhow::bail!("max_tokens must be at least 1.");
    }

    let is_streaming = oairequest.stream.unwrap_or(false);

    if oairequest.grammar.is_some() && oairequest.response_format.is_some() {
        anyhow::bail!(
            "Request `grammar` and `response_format` were both provided but are mutually exclusive."
        )
    }

    let constraint = match oairequest.grammar {
        Some(Grammar::Regex(regex)) => Constraint::Regex(regex),
        Some(Grammar::Lark(lark)) => Constraint::Lark(lark),
        Some(Grammar::JsonSchema(schema)) => Constraint::JsonSchema(schema),
        Some(Grammar::Llguidance(llguidance)) => Constraint::Llguidance(llguidance),
        None => match oairequest.response_format {
            Some(ResponseFormat::JsonSchema {
                json_schema: JsonSchemaResponseFormat { name: _, schema },
            }) => Constraint::JsonSchema(schema),
            Some(ResponseFormat::JsonObject) => Constraint::JsonSchema(json!({"type": "object"})),
            Some(ResponseFormat::Text) => Constraint::None,
            None => Constraint::None,
        },
    };

    Ok((
        Request::Normal(Box::new(NormalRequest {
            id: state.next_request_id(),
            queued_at: None,
            messages,
            sampling_params: SamplingParams {
                temperature: oairequest.temperature,
                top_k: oairequest.top_k,
                top_p: oairequest.top_p,
                min_p: oairequest.min_p,
                top_n_logprobs: oairequest.top_logprobs.unwrap_or(1),
                frequency_penalty: oairequest.frequency_penalty,
                presence_penalty: oairequest.presence_penalty,
                repetition_penalty: oairequest.repetition_penalty,
                max_len: oairequest.max_tokens,
                stop_toks,
                ignore_eos: oairequest.ignore_eos,
                logits_bias: oairequest.logit_bias,
                n_choices: oairequest.n_choices,
                dry_params,
            },
            seed: oairequest.seed,
            response: tx,
            return_logprobs: oairequest.logprobs,
            is_streaming,
            suffix: None,
            constraint,
            tool_choice: oairequest.tool_choice,
            tools: normalized_tools.tools,
            logits_processors: None,
            host_tools: Vec::new(),
            sequential_tool_calls: oairequest.parallel_tool_calls == Some(false),
            return_raw_logits: false,
            web_search_options: normalized_tools.web_search_options,
            enable_code_execution: normalized_tools.enable_code_execution,
            enable_shell: normalized_tools.enable_shell,
            shell_options,
            code_execution_permission: oairequest.code_execution_permission,
            code_execution_approval_notifier: None,
            agent_permission: oairequest.agent_permission,
            agent_approval_handler,
            agent_approval_notifier,
            session_id: oairequest.session_id,
            owner,
            files: oairequest.files,
            input_files,
            cancellation: None,
            max_tool_rounds: oairequest.max_tool_rounds,
            tool_dispatch_url,
            model_id: if oairequest.model == DEFAULT_MODEL_ID {
                None
            } else {
                Some(oairequest.model.clone())
            },
            adapter,
            truncate_sequence: oairequest.truncate_sequence.unwrap_or(false),
        })),
        is_streaming,
    ))
}

/// The prompt tokens `oairequest` renders to, with its tools, reasoning controls and the model's chat template.
pub(crate) async fn tokenize_chat(
    state: &SharedInferenceRsState,
    mut oairequest: ChatCompletionRequest,
    owner: Option<&str>,
) -> Result<Vec<u32>, ApiError> {
    let invalid = |error: &(dyn std::error::Error + 'static)| {
        ApiError::from_error(error, ApiErrorKind::InvalidRequest)
    };
    let internal = |error: &(dyn std::error::Error + 'static)| {
        InferenceRs::maybe_log_error(state.clone(), error);
        ApiError::from_error(error, ApiErrorKind::Internal)
    };
    oairequest.stream = Some(false);
    resolve_lora_adapter_model(state, &mut oairequest.model, &mut oairequest.adapter)
        .map_err(|error| invalid(&error))?;
    let model_id = (oairequest.model != DEFAULT_MODEL_ID).then(|| oairequest.model.clone());

    // Parsed like a chat request, for its rendered messages and tools, but never sent as one.
    let (tx, _) = create_response_channel(Some(1));
    let (parsed, _) = parse_request(
        oairequest,
        ChatCompletionParseContext {
            state: state.clone(),
            tx,
            tool_dispatch_url: None,
            agent_approval_handler: None,
            agent_approval_notifier: None,
            tool_surface: OpenAiToolSurface::ChatCompletions,
            skill_store: None,
            media: Default::default(),
            owner: owner.map(str::to_string),
        },
    )
    .await
    .map_err(|error| invalid(error.as_ref()))?;
    let Request::Normal(parsed) = parsed else {
        return Err(ApiError::internal());
    };
    let (messages, enable_thinking, reasoning_effort) = match parsed.messages {
        RequestMessage::Chat {
            messages,
            enable_thinking,
            reasoning_effort,
        }
        | RequestMessage::MultimodalChat {
            messages,
            enable_thinking,
            reasoning_effort,
            ..
        } => (messages, enable_thinking, reasoning_effort),
        _ => return Err(ApiError::invalid_request(ONLY_CHAT_IS_TOKENIZED)),
    };

    let (response, mut rx) = tokio::sync::mpsc::channel(1);
    let tokenize = Request::Tokenize(TokenizationRequest {
        text: Either::Left(messages),
        tools: parsed.tools,
        add_generation_prompt: true,
        add_special_tokens: true,
        enable_thinking,
        reasoning_effort,
        response,
    });
    send_request_with_model(state, tokenize, model_id.as_deref())
        .await
        .map_err(|error| internal(&error))?;
    match rx.recv().await {
        Some(Ok(tokens)) => Ok(tokens),
        Some(Err(error)) if is_chat_template_request_error(&error) => Err(invalid(error.as_ref())),
        Some(Err(error)) => Err(internal(error.as_ref())),
        None => Err(ApiError::internal()),
    }
}

/// Server-level chat policy and the state a chat request runs against.
#[derive(Clone)]
pub struct ChatEngine {
    pub state: SharedInferenceRsState,
    pub agentic: AgenticDefaults,
    pub skill_store: Option<Arc<SkillStore>>,
    /// Who requests act for; their sessions and files are that owner's. `None` for an unscoped caller.
    pub owner: Option<String>,
    pub logits_processors: LogitsProcessors,
    pub host_tools: HostTools,
}

/// A dispatched chat request: its response channel and how to present what comes back.
pub struct PreparedChat {
    pub rx: Receiver<Response>,
    pub is_streaming: bool,
    /// The model name the caller asked for, when routing resolved it to another id.
    pub model_override: Option<String>,
    /// Cancels the dispatched request; its final response still arrives, marked `canceled`.
    pub cancellation: RequestCancellation,
}

/// Why a chat request was not dispatched. Validation errors are the caller's; internal ones are the engine's.
pub enum DispatchError {
    Validation(Box<dyn std::error::Error + Send + Sync>),
    Internal(Box<dyn std::error::Error + Send + Sync>),
}

impl DispatchError {
    /// The error to report, logging it when it is the engine's fault.
    pub(crate) fn into_api_error(self, state: SharedInferenceRsState) -> ApiError {
        match self {
            DispatchError::Validation(error) => {
                let api = ApiError::from_error(error.as_ref(), ApiErrorKind::InvalidRequest);
                if matches!(
                    api.kind,
                    ApiErrorKind::Internal | ApiErrorKind::Unavailable | ApiErrorKind::Overloaded
                ) {
                    InferenceRs::maybe_log_error(state, error.as_ref());
                }
                api
            }
            DispatchError::Internal(error) => {
                InferenceRs::maybe_log_error(state, error.as_ref());
                ApiError::from_error(error.as_ref(), ApiErrorKind::Internal)
            }
        }
    }
}

impl ChatEngine {
    /// Fills the server's `max_tool_rounds` default and merges its agent permission with the request's, strictest.
    pub fn apply_agent_policy(&self, request: &mut ChatCompletionRequest) {
        request.max_tool_rounds = request.max_tool_rounds.or(self.agentic.max_tool_rounds);
        let request_permission = request
            .agent_permission
            .or_else(|| request.code_execution_permission.map(Into::into));
        request.agent_permission = match (self.agentic.agent_permission, request_permission) {
            (Some(server_permission), Some(request_permission)) => {
                Some(server_permission.strictest(request_permission))
            }
            (Some(server_permission), None) => Some(server_permission),
            (None, permission) => permission,
        };
        request.code_execution_permission = None;
    }

    /// Applies the server's agentic policy to `oairequest`, parses it and sends it to its model.
    pub fn prepare<'a>(
        &'a self,
        oairequest: ChatCompletionRequest,
        tool_surface: OpenAiToolSurface,
        media: MediaAttachments,
    ) -> BoxFuture<'a, Result<PreparedChat, DispatchError>> {
        Box::pin(self.prepare_inner(oairequest, tool_surface, media))
    }

    async fn prepare_inner(
        &self,
        mut oairequest: ChatCompletionRequest,
        tool_surface: OpenAiToolSurface,
        media: MediaAttachments,
    ) -> Result<PreparedChat, DispatchError> {
        let (tx, rx) = create_response_channel(None);
        let requested_model = oairequest.model.clone();
        resolve_lora_adapter_model(&self.state, &mut oairequest.model, &mut oairequest.adapter)
            .map_err(|error| DispatchError::Validation(Box::new(error)))?;
        let model_override = response_model_id(&self.state, requested_model, &oairequest.model);

        self.apply_agent_policy(&mut oairequest);
        let asks = matches!(oairequest.agent_permission, Some(AgentPermission::Ask));
        let is_streaming = oairequest.stream.unwrap_or(false);
        if asks && !is_streaming {
            return Err(DispatchError::Validation(Box::new(ApiError::new(
                ApiErrorKind::InvalidRequest,
                ASK_REQUIRES_STREAMING,
                Some("unsupported_parameter"),
                Some("agent_permission"),
            ))));
        }
        let agent_approval_handler = asks.then(|| {
            AgentToolApprovalHandler::from_async(
                self.agentic.approval_broker.callback(self.owner.clone()),
            )
        });
        let agent_approval_notifier = asks.then(|| {
            self.agentic
                .approval_broker
                .notifier(tx.clone(), self.owner.clone())
        });

        let logits_processors = self
            .logits_processors
            .resolve(oairequest.logits_processors.as_deref())
            .map_err(|error| DispatchError::Validation(Box::new(error)))?;
        let host_tools = self
            .host_tools
            .resolve(oairequest.host_tools.as_deref())
            .map_err(|error| DispatchError::Validation(Box::new(error)))?;
        let model_id = (oairequest.model != DEFAULT_MODEL_ID).then(|| oairequest.model.clone());
        let (mut request, is_streaming) = parse_request(
            oairequest,
            ChatCompletionParseContext {
                state: self.state.clone(),
                tx,
                tool_dispatch_url: self.agentic.tool_dispatch_url.clone(),
                agent_approval_handler,
                agent_approval_notifier,
                tool_surface,
                skill_store: self.skill_store.clone(),
                media,
                owner: self.owner.clone(),
            },
        )
        .await
        .map_err(|error| DispatchError::Validation(boxed_anyhow(error)))?;
        let cancellation = RequestCancellation::default();
        if let Request::Normal(normal) = &mut request {
            normal.cancellation = Some(cancellation.clone());
            normal.logits_processors = logits_processors;
            normal.host_tools = host_tools.unwrap_or_default();
        }
        send_request_with_model(&self.state, request, model_id.as_deref())
            .await
            .map_err(|error| DispatchError::Internal(error.into()))?;
        Ok(PreparedChat {
            rx,
            is_streaming,
            model_override,
            cancellation,
        })
    }
}

/// Waits for a non-streaming chat request's final response, with its agentic tool calls and files attached.
pub fn collect_chat<'a>(
    rx: &'a mut Receiver<Response>,
    model_override: Option<&'a str>,
) -> BoxFuture<'a, Response> {
    Box::pin(collect_chat_inner(rx, model_override))
}

async fn collect_chat_inner(rx: &mut Receiver<Response>, model_override: Option<&str>) -> Response {
    let mut collector = ChatResponseCollector::default();
    let finish = |collector: ChatResponseCollector, response| {
        let mut response = collector.finish(response);
        apply_model_override(&mut response.model, model_override);
        response
    };
    loop {
        let Some(response) = rx.recv().await else {
            return Response::InternalError(
                anyhow::Error::msg("No response received from the model.").into(),
            );
        };
        match collector.absorb(response) {
            None | Some(Response::BlockDenoisingProgress(_)) => continue,
            Some(Response::AgenticToolApprovalRequired { .. }) => {
                return Response::ValidationError(Box::new(JsonError::new(
                    "code execution approval requires a streaming request.".to_string(),
                )));
            }
            Some(Response::Done(response)) => return Response::Done(finish(collector, response)),
            Some(Response::ModelError(msg, response)) => {
                return Response::ModelError(msg, finish(collector, response));
            }
            Some(response) => return response,
        }
    }
}

/// One event of a streaming chat request, in the order the engine produced it.
pub enum ChatStreamEvent {
    Chunk(ChatCompletionChunkResponse),
    AgenticToolCallProgress(AgenticToolProgress),
    AgenticToolApprovalRequired(AgenticToolApproval),
    FileProduced(inference_core::File),
    /// A block-diffusion model's denoising step; only streams that asked with `with_denoising_progress` get these.
    BlockDenoisingProgress(inference_core::BlockDenoisingProgress),
    /// Terminal: nothing follows an error.
    Error(ApiError),
}

// The types the agentic events carry, so API consumers can match on them without naming inference_core.
pub use inference_core::{
    AgentToolKind, AgentToolMetadata, AgentToolSource, AgenticToolCallData, AgenticToolCallPhase,
    BlockDenoisingProgress, RequestCancellation, Usage,
};

/// A tool call's progress in an agentic run.
#[derive(Debug, Clone)]
pub struct AgenticToolProgress {
    pub round: usize,
    pub tool_call_id: String,
    pub tool_name: String,
    pub phase: AgenticToolCallPhase,
}

impl AgenticToolProgress {
    /// The `agentic_tool_call_progress` payload the HTTP and C ABI streams carry.
    pub fn to_json(&self) -> Value {
        serialize_agentic_progress(self.round, &self.tool_call_id, &self.tool_name, &self.phase)
    }
}

/// An agent action waiting for approval; answer it with the engine's approval resolution.
#[derive(Debug, Clone)]
pub struct AgenticToolApproval {
    pub approval_id: String,
    pub session_id: String,
    pub round: usize,
    pub tool: inference_core::AgentToolMetadata,
    pub arguments: Value,
}

impl AgenticToolApproval {
    /// The `agentic_tool_approval_required` payload the HTTP and C ABI streams carry.
    pub fn to_json(&self) -> Value {
        serialize_approval_required(
            &self.approval_id,
            &self.session_id,
            self.round,
            &self.tool,
            &self.arguments,
        )
    }
}

/// The engine's raw responses, as a [`ResponseTap`] sees them.
pub use inference_core::Response;

/// Observes every engine response before it is mapped, e.g. for usage and latency accounting.
pub type ResponseTap = Box<dyn Fn(&Response) + Send + Sync>;

/// The events of a streaming chat request. Dropping it abandons the request.
pub struct ChatStream {
    rx: Receiver<Response>,
    state: SharedInferenceRsState,
    model_override: Option<String>,
    tap: Option<ResponseTap>,
    cancellation: Option<RequestCancellation>,
    denoising_progress: bool,
    finished: bool,
}

impl ChatStream {
    pub(crate) fn new(
        rx: Receiver<Response>,
        state: SharedInferenceRsState,
        model_override: Option<String>,
        tap: Option<ResponseTap>,
    ) -> Self {
        Self {
            rx,
            state,
            model_override,
            tap,
            cancellation: None,
            denoising_progress: false,
            finished: false,
        }
    }

    pub fn with_denoising_progress(mut self) -> Self {
        self.denoising_progress = true;
        self
    }

    /// Reports each engine response to `tap` as the stream reads it, e.g. for an access log.
    pub fn with_tap(mut self, tap: Option<ResponseTap>) -> Self {
        self.tap = tap;
        self
    }

    /// The request's cancellation, for a caller that cancels from elsewhere, e.g. a signal handler.
    pub fn cancellation(&self) -> Option<RequestCancellation> {
        self.cancellation.clone()
    }

    pub fn with_cancellation(mut self, cancellation: RequestCancellation) -> Self {
        self.cancellation = Some(cancellation);
        self
    }

    /// Ends the request on its next sampled token; the stream still yields its final chunk, with usage.
    pub fn cancel(&self) {
        if let Some(cancellation) = &self.cancellation {
            cancellation.cancel();
        }
    }

    fn map(&mut self, response: Response) -> Option<ChatStreamEvent> {
        Some(match response {
            Response::ModelError(msg, _) => {
                InferenceRs::maybe_log_error(
                    self.state.clone(),
                    &ModelErrorMessage(msg.to_string()),
                );
                self.finished = true;
                ChatStreamEvent::Error(ApiError::model_error())
            }
            Response::ValidationError(e) => {
                self.finished = true;
                ChatStreamEvent::Error(ApiError::from_error(
                    e.as_ref(),
                    ApiErrorKind::InvalidRequest,
                ))
            }
            Response::InternalError(e) => {
                InferenceRs::maybe_log_error(self.state.clone(), &*e);
                self.finished = true;
                ChatStreamEvent::Error(ApiError::from_error(e.as_ref(), ApiErrorKind::Internal))
            }
            Response::Chunk(mut response) => {
                if response.choices.iter().all(|x| x.finish_reason.is_some()) {
                    self.finished = true;
                }
                InferenceRs::maybe_log_response(self.state.clone(), &response);
                apply_model_override(&mut response.model, self.model_override.as_deref());
                ChatStreamEvent::Chunk(response)
            }
            Response::AgenticToolCallProgress {
                round,
                tool_call_id,
                tool_name,
                phase,
            } => ChatStreamEvent::AgenticToolCallProgress(AgenticToolProgress {
                round,
                tool_call_id,
                tool_name,
                phase,
            }),
            Response::AgenticToolApprovalRequired {
                approval_id,
                session_id,
                round,
                tool,
                arguments,
            } => ChatStreamEvent::AgenticToolApprovalRequired(AgenticToolApproval {
                approval_id,
                session_id,
                round,
                tool,
                arguments,
            }),
            Response::BlockDenoisingProgress(progress) => {
                if !self.denoising_progress {
                    return None;
                }
                ChatStreamEvent::BlockDenoisingProgress(progress)
            }
            Response::File(file) => ChatStreamEvent::FileProduced(file),
            Response::Done(_)
            | Response::CompletionDone(_)
            | Response::CompletionModelError(_, _)
            | Response::CompletionChunk(_)
            | Response::ImageGeneration(_)
            | Response::Speech { .. }
            | Response::Raw { .. }
            | Response::Embeddings { .. } => unreachable!("not a chat stream response"),
        })
    }
}

impl futures::Stream for ChatStream {
    type Item = ChatStreamEvent;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        loop {
            if self.finished {
                return Poll::Ready(None);
            }
            match self.rx.poll_recv(cx) {
                Poll::Ready(Some(response)) => {
                    if let Some(tap) = &self.tap {
                        tap(&response);
                    }
                    if let Some(event) = self.map(response) {
                        return Poll::Ready(Some(event));
                    }
                }
                Poll::Ready(None) => {
                    self.finished = true;
                    return Poll::Ready(Some(ChatStreamEvent::Error(ApiError::internal())));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The typed events' JSON is the payload the HTTP and C ABI streams carry.
    #[test]
    fn typed_agentic_events_serialize_to_the_wire_payloads() {
        let progress = AgenticToolProgress {
            round: 2,
            tool_call_id: "call_lookup".to_string(),
            tool_name: "lookup".to_string(),
            phase: AgenticToolCallPhase::Complete(AgenticToolCallData::Custom {
                arguments: "{}".to_string(),
                content: "found".to_string(),
            }),
        };
        let progress = progress.to_json();
        assert_eq!(progress["type"], "agentic_tool_call_progress");
        assert_eq!(progress["round"], 2);
        assert_eq!(progress["tool_call_id"], "call_lookup");
        assert_eq!(progress["phase"], "complete");
        assert_eq!(progress["data"]["tool_type"], "custom");
        assert_eq!(progress["data"]["content"], "found");

        let approval = AgenticToolApproval {
            approval_id: "appr_1".to_string(),
            session_id: "session".to_string(),
            round: 1,
            tool: inference_core::AgentToolMetadata {
                source: inference_core::AgentToolSource::BuiltIn,
                kind: inference_core::AgentToolKind::Shell,
                label: "Shell".to_string(),
            },
            arguments: json!({"command": "ls"}),
        }
        .to_json();
        assert_eq!(approval["type"], "agentic_tool_approval_required");
        assert_eq!(approval["approval_id"], "appr_1");
        assert_eq!(approval["arguments"]["command"], "ls");
        assert_eq!(approval["tool"]["label"], "Shell");
    }

    #[test]
    fn reasoning_controls_normalize_http_values() {
        assert_eq!(parse_reasoning_controls(None, None).unwrap(), (None, None));
        assert_eq!(
            parse_reasoning_controls(None, Some(" NONE ")).unwrap(),
            (None, Some(ReasoningEffort::Off))
        );
        assert_eq!(
            parse_reasoning_controls(None, Some("XHIGH")).unwrap(),
            (None, Some(ReasoningEffort::XHigh))
        );
    }

    #[test]
    fn reasoning_controls_validate_http_values() {
        assert!(parse_reasoning_controls(None, Some("extreme")).is_err());
        assert!(parse_reasoning_controls(Some(true), Some("off")).is_err());
        assert!(parse_reasoning_controls(Some(false), Some("high")).is_err());
    }

    #[test]
    fn assistant_reasoning_content_reaches_the_core_message() {
        let message: Message = serde_json::from_value(json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"}
            }],
            "reasoning_content": "Need weather"
        }))
        .unwrap();
        let mut output = IndexMap::new();

        insert_reasoning_content(&mut output, &message);

        assert_eq!(
            output.get("reasoning_content"),
            Some(&Either::Left("Need weather".to_string()))
        );
    }
}

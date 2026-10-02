//! Chat requests built in Rust: messages, media and options, sent as the engine's `ChatCompletionRequest`.

use std::{collections::HashMap, fmt::Display, sync::Arc};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use inference_api::{
    engine::{AgentToolApproval, AgentToolApprovalDecision, AgentToolApprovalHandler},
    logits_processors::CustomLogitsProcessor,
    media_source::{ATTACHMENT_PREFIX, Media, MediaAttachment, MediaAttachments},
    openai::{
        AdapterSelection, ChatCompletionRequest, Grammar, OpenAiShellSkillReference, OpenAiTool,
        StopTokens,
    },
    sdk::ToolChoice,
};
use serde_json::{Value, json};

const DEFAULT_FILE_MIME: &str = "application/octet-stream";
const GREEDY_TOP_K: usize = 1;

use crate::{
    AgentPermission, AudioInput, CodeExecutionPermission, DynamicImage, ReasoningEffort,
    RequestedFile, Tool, ToolCallResponse, VideoInput, WebSearchOptions,
};

/// A chat request ready to send: the engine request, its `media://N` media, and what the SDK registers around it.
pub struct ChatRequest {
    pub request: ChatCompletionRequest,
    pub media: MediaAttachments,
    pub(crate) logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
    pub(crate) approval: Option<AgentToolApprovalHandler>,
}

impl From<ChatCompletionRequest> for ChatRequest {
    fn from(request: ChatCompletionRequest) -> Self {
        Self {
            request,
            media: MediaAttachments::default(),
            logits_processors: Vec::new(),
            approval: None,
        }
    }
}

/// A chat message's sender.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextMessageRole {
    User,
    Assistant,
    System,
    Tool,
    Custom(String),
}

impl Display for TextMessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::System => write!(f, "system"),
            Self::Tool => write!(f, "tool"),
            Self::Custom(role) => write!(f, "{role}"),
        }
    }
}

/// A file the model can read with its file tools, sent inline with the request.
#[derive(Debug, Clone)]
pub struct InputFile {
    name: String,
    bytes: Vec<u8>,
    mime_type: Option<String>,
}

impl InputFile {
    pub fn from_bytes(name: impl Into<String>, bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            name: name.into(),
            bytes: bytes.into(),
            mime_type: None,
        }
    }

    pub fn from_bytes_with_mime(
        name: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
        mime_type: impl Into<String>,
    ) -> Self {
        Self {
            mime_type: Some(mime_type.into()),
            ..Self::from_bytes(name, bytes)
        }
    }

    pub fn from_text(name: impl Into<String>, text: impl Into<String>) -> Self {
        Self::from_bytes_with_mime(name, text.into().into_bytes(), "text/plain")
    }

    pub fn from_path(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned());
        Ok(Self::from_bytes(name, std::fs::read(path)?))
    }

    fn part(&self) -> Value {
        let mime = self.mime_type.as_deref().unwrap_or(DEFAULT_FILE_MIME);
        let data = format!("data:{mime};base64,{}", STANDARD.encode(&self.bytes));
        json!({"type": "file", "file": {"filename": self.name, "file_data": data}})
    }
}

/// Messages with media, as [`RequestBuilder`] takes them without its other options.
#[derive(Default, Clone, Debug)]
pub struct MultimodalMessages(RequestBuilder);

/// Text-only messages, as [`RequestBuilder`] takes them without its other options.
#[derive(Default, Clone, Debug)]
pub struct TextMessages(RequestBuilder);

impl TextMessages {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_message(self, role: TextMessageRole, text: impl ToString) -> Self {
        Self(self.0.add_message(role, text))
    }

    pub fn enable_thinking(self, enable_thinking: bool) -> Self {
        Self(self.0.enable_thinking(enable_thinking))
    }

    pub fn with_reasoning_effort(self, effort: ReasoningEffort) -> Self {
        Self(self.0.with_reasoning_effort(effort))
    }
}

impl MultimodalMessages {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_message(self, role: TextMessageRole, text: impl ToString) -> Self {
        Self(self.0.add_message(role, text))
    }

    pub fn add_image_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        images: Vec<DynamicImage>,
    ) -> Self {
        Self(self.0.add_image_message(role, text, images))
    }

    pub fn add_audio_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        audios: Vec<AudioInput>,
    ) -> Self {
        Self(self.0.add_audio_message(role, text, audios))
    }

    pub fn add_video_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        videos: Vec<VideoInput>,
    ) -> Self {
        Self(self.0.add_video_message(role, text, videos))
    }

    pub fn add_multimodal_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        media: MessageMedia,
    ) -> Self {
        Self(self.0.add_multimodal_message(role, text, media))
    }

    /// A message whose media the engine fetches from URLs (http, https, data or file).
    pub fn add_media_url_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        kind: EncodedKind,
        urls: Vec<String>,
    ) -> Self {
        Self(self.0.add_media_url_message(role, text, kind, urls))
    }

    pub fn enable_thinking(self, enable_thinking: bool) -> Self {
        Self(self.0.enable_thinking(enable_thinking))
    }

    pub fn with_reasoning_effort(self, effort: ReasoningEffort) -> Self {
        Self(self.0.with_reasoning_effort(effort))
    }
}

impl From<TextMessages> for RequestBuilder {
    fn from(messages: TextMessages) -> Self {
        messages.0
    }
}

impl From<MultimodalMessages> for RequestBuilder {
    fn from(messages: MultimodalMessages) -> Self {
        messages.0
    }
}

impl From<TextMessages> for ChatRequest {
    fn from(messages: TextMessages) -> Self {
        messages.0.into()
    }
}

impl From<MultimodalMessages> for ChatRequest {
    fn from(messages: MultimodalMessages) -> Self {
        messages.0.into()
    }
}

/// The media of one multimodal message, in the order the model sees them: images, then audio, then video.
#[derive(Default)]
pub struct MessageMedia {
    pub images: Vec<DynamicImage>,
    pub audios: Vec<AudioInput>,
    pub videos: Vec<VideoInput>,
}

/// A chat request: its messages and every request option.
#[derive(Clone)]
pub struct RequestBuilder {
    messages: Vec<Value>,
    media: Vec<Media>,
    request: ChatCompletionRequest,
    input_files: Vec<InputFile>,
    logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
    approval: Option<AgentToolApprovalHandler>,
}

impl std::fmt::Debug for RequestBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestBuilder")
            .field("messages", &self.messages)
            .field("media", &self.media.len())
            .field("logits_processors", &self.logits_processors.len())
            .finish_non_exhaustive()
    }
}

impl Default for RequestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A chat request with no messages and every option at the engine's default.
pub fn empty_chat_request() -> ChatCompletionRequest {
    serde_json::from_value(json!({"messages": []}))
        .expect("a chat request needs nothing but its messages")
}

impl RequestBuilder {
    /// A request that decodes greedily (top-k 1) until a sampling setter says otherwise.
    pub fn new() -> Self {
        let mut request = empty_chat_request();
        request.top_k = Some(GREEDY_TOP_K);
        Self {
            messages: Vec::new(),
            media: Vec::new(),
            request,
            input_files: Vec::new(),
            logits_processors: Vec::new(),
            approval: None,
        }
    }

    /// The request goes to `model` rather than the engine's default model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.request.model = model.into();
        self
    }

    pub fn add_message(mut self, role: TextMessageRole, text: impl ToString) -> Self {
        self.messages
            .push(json!({"role": role.to_string(), "content": text.to_string()}));
        self
    }

    pub fn add_tool_message(mut self, content: impl ToString, tool_call_id: impl ToString) -> Self {
        self.messages.push(json!({
            "role": TextMessageRole::Tool.to_string(),
            "content": content.to_string(),
            "tool_call_id": tool_call_id.to_string(),
        }));
        self
    }

    /// An assistant turn that called tools, to answer with [`Self::add_tool_message`].
    pub fn add_message_with_tool_call(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        tool_calls: Vec<ToolCallResponse>,
    ) -> Self {
        let calls: Vec<Value> = tool_calls
            .iter()
            .map(|call| {
                json!({
                    "id": call.id,
                    "type": call.tp.to_string(),
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                })
            })
            .collect();
        self.messages.push(
            json!({"role": role.to_string(), "content": text.to_string(), "tool_calls": calls}),
        );
        self
    }

    pub fn add_image_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        images: Vec<DynamicImage>,
    ) -> Self {
        let media = MessageMedia {
            images,
            ..MessageMedia::default()
        };
        self.add_multimodal_message(role, text, media)
    }

    pub fn add_audio_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        audios: Vec<AudioInput>,
    ) -> Self {
        let media = MessageMedia {
            audios,
            ..MessageMedia::default()
        };
        self.add_multimodal_message(role, text, media)
    }

    pub fn add_video_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        videos: Vec<VideoInput>,
    ) -> Self {
        let media = MessageMedia {
            videos,
            ..MessageMedia::default()
        };
        self.add_multimodal_message(role, text, media)
    }

    /// A message whose media go with the request already decoded, named by `media://N` sources.
    pub fn add_multimodal_message(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        media: MessageMedia,
    ) -> Self {
        let mut parts = Vec::new();
        let decoded = (media.images.into_iter().map(Media::Image))
            .chain(media.audios.into_iter().map(Media::Audio))
            .chain(media.videos.into_iter().map(Media::Video));
        for item in decoded {
            let source = format!("{ATTACHMENT_PREFIX}{}", self.media.len());
            parts.push(match &item {
                Media::Audio(_) => json!({"type": "audio_url", "audio_url": {"url": source}}),
                Media::Video(_) => json!({"type": "video_url", "video_url": {"url": source}}),
                _ => json!({"type": "image_url", "image_url": {"url": source}}),
            });
            self.media.push(item);
        }
        parts.push(json!({"type": "text", "text": text.to_string()}));
        self.messages
            .push(json!({"role": role.to_string(), "content": parts}));
        self
    }

    /// A message with encoded media (a PNG, a WAV), which the engine decodes as it would an upload.
    pub fn add_encoded_media_message(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        kind: EncodedKind,
        attachment: MediaAttachment,
    ) -> Self {
        let source = format!("{ATTACHMENT_PREFIX}{}", self.media.len());
        self.media.push(Media::Encoded(attachment));
        self.add_media_url_message(role, text, kind, vec![source])
    }

    /// A message whose media the engine fetches from URLs (http, https, data or file).
    pub fn add_media_url_message(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        kind: EncodedKind,
        urls: Vec<String>,
    ) -> Self {
        let (field, part) = kind.part_name();
        let mut parts: Vec<Value> = urls
            .into_iter()
            .map(|url| json!({"type": part, field: {"url": url}}))
            .collect();
        parts.push(json!({"type": "text", "text": text.to_string()}));
        self.messages
            .push(json!({"role": role.to_string(), "content": parts}));
        self
    }

    /// Runs `processor` on this request's logits each step, after the engine's penalties.
    pub fn add_logits_processor(mut self, processor: Arc<dyn CustomLogitsProcessor>) -> Self {
        self.logits_processors.push(processor);
        self
    }

    /// Names a processor registered on the engine with `register_logits_processor`.
    pub fn with_logits_processor_name(mut self, name: impl Into<String>) -> Self {
        self.request
            .logits_processors
            .get_or_insert_with(Vec::new)
            .push(name.into());
        self
    }

    /// Offers a tool registered on the engine with `register_tool`.
    pub fn with_host_tool(mut self, name: impl Into<String>) -> Self {
        self.request
            .host_tools
            .get_or_insert_with(Vec::new)
            .push(name.into());
        self
    }

    pub fn set_adapter(mut self, adapter: impl Into<AdapterSelection>) -> Self {
        self.request.adapter = Some(adapter.into());
        self
    }

    pub fn set_tools(mut self, tools: Vec<Tool>) -> Self {
        self.request.tools = Some(tools.into_iter().map(OpenAiTool::Function).collect());
        self
    }

    pub fn set_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.request.tool_choice = Some(tool_choice);
        self
    }

    pub fn with_web_search_options(mut self, options: WebSearchOptions) -> Self {
        self.request.web_search_options = Some(options);
        self
    }

    /// Offers the engine's Python code execution tool.
    pub fn with_code_execution(mut self) -> Self {
        let tool: OpenAiTool = serde_json::from_value(
            json!({"type": "code_interpreter", "container": {"type": "auto"}}),
        )
        .expect("the code interpreter tool has no required settings");
        self.request.tools.get_or_insert_with(Vec::new).push(tool);
        self
    }

    /// Offers the engine's shell tool.
    pub fn with_shell_execution(mut self) -> Self {
        self.request.enable_shell = true;
        self
    }

    /// Mounts a skill uploaded with the engine's skill store into the shell tool.
    pub fn with_shell_skill(mut self, skill_id: impl Into<String>) -> Self {
        let reference = OpenAiShellSkillReference {
            skill_id: skill_id.into(),
            version: None,
        };
        self.request.enable_shell = true;
        self.request.shell_skill_references.push(reference);
        self
    }

    pub fn with_code_execution_permission(mut self, permission: CodeExecutionPermission) -> Self {
        self.request.code_execution_permission = Some(permission);
        self
    }

    pub fn with_agent_permission(mut self, permission: AgentPermission) -> Self {
        self.request.agent_permission = Some(permission);
        self
    }

    /// Answers the request's tool approvals (with `AgentPermission::Ask`) in this process; no stream is needed.
    pub fn with_agent_approval_callback(
        mut self,
        callback: impl Fn(&AgentToolApproval) -> AgentToolApprovalDecision + Send + Sync + 'static,
    ) -> Self {
        self.approval = Some(AgentToolApprovalHandler::from_sync(Arc::new(callback)));
        self
    }

    /// As [`Self::with_agent_approval_callback`], for an answer that waits on async state (a UI, a queue).
    pub fn with_agent_approval_async_callback<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(AgentToolApproval) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = AgentToolApprovalDecision> + Send + 'static,
    {
        let callback = move |approval| Box::pin(callback(approval)) as _;
        self.approval = Some(AgentToolApprovalHandler::from_async(Arc::new(callback)));
        self
    }

    pub fn with_session_id(mut self, id: impl Into<String>) -> Self {
        self.request.session_id = Some(id.into());
        self
    }

    pub fn with_max_tool_rounds(mut self, rounds: usize) -> Self {
        self.request.max_tool_rounds = Some(rounds);
        self
    }

    /// Runs a round's tool calls one at a time, in the model's order.
    pub fn with_sequential_tool_calls(mut self) -> Self {
        self.request.parallel_tool_calls = Some(false);
        self
    }

    pub fn return_logprobs(mut self, return_logprobs: bool) -> Self {
        self.request.logprobs = return_logprobs;
        self
    }

    pub fn set_grammar(mut self, grammar: Grammar) -> Self {
        self.request.grammar = Some(grammar);
        self
    }

    /// Greedy decoding: temperature 0 and top-k 1.
    pub fn set_deterministic_sampler(mut self) -> Self {
        self.request.temperature = Some(0.0);
        self.request.top_k = Some(1);
        self
    }

    pub fn set_sampler_temperature(mut self, temperature: f64) -> Self {
        self.request.temperature = Some(temperature);
        self
    }

    pub fn set_sampler_topk(mut self, top_k: usize) -> Self {
        self.request.top_k = Some(top_k);
        self
    }

    pub fn set_sampler_topp(mut self, top_p: f64) -> Self {
        self.request.top_p = Some(top_p);
        self
    }

    pub fn set_sampler_minp(mut self, min_p: f64) -> Self {
        self.request.min_p = Some(min_p);
        self
    }

    pub fn set_sampler_topn_logprobs(mut self, top_n: usize) -> Self {
        self.request.top_logprobs = Some(top_n);
        self
    }

    pub fn set_sampler_frequency_penalty(mut self, penalty: f32) -> Self {
        self.request.frequency_penalty = Some(penalty);
        self
    }

    pub fn set_sampler_presence_penalty(mut self, penalty: f32) -> Self {
        self.request.presence_penalty = Some(penalty);
        self
    }

    pub fn set_sampler_repetition_penalty(mut self, penalty: f32) -> Self {
        self.request.repetition_penalty = Some(penalty);
        self
    }

    pub fn set_sampler_stop(mut self, stop: Vec<String>) -> Self {
        self.request.stop_seqs = Some(StopTokens::Multi(stop));
        self
    }

    pub fn set_sampler_stop_token_ids(mut self, ids: Vec<u32>) -> Self {
        self.request.stop_token_ids = Some(ids);
        self
    }

    pub fn set_sampler_max_len(mut self, max_tokens: usize) -> Self {
        self.request.max_tokens = Some(max_tokens);
        self
    }

    pub fn set_sampler_ignore_eos(mut self, ignore_eos: bool) -> Self {
        self.request.ignore_eos = ignore_eos;
        self
    }

    pub fn set_sampler_logits_bias(mut self, bias: HashMap<u32, f32>) -> Self {
        self.request.logit_bias = Some(bias);
        self
    }

    pub fn set_sampler_n_choices(mut self, n_choices: usize) -> Self {
        self.request.n_choices = n_choices;
        self
    }

    pub fn set_sampler_seed(mut self, seed: u64) -> Self {
        self.request.seed = Some(seed);
        self
    }

    /// DRY repetition penalty; unset parameters keep the engine's defaults.
    pub fn set_sampler_dry(mut self, dry: DrySampling) -> Self {
        self.request.dry_multiplier = Some(dry.multiplier);
        self.request.dry_base = dry.base;
        self.request.dry_allowed_length = dry.allowed_length;
        self.request.dry_sequence_breakers = dry.sequence_breakers;
        self
    }

    pub fn enable_thinking(mut self, enable_thinking: bool) -> Self {
        self.request.enable_thinking = Some(enable_thinking);
        self
    }

    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.request.reasoning_effort = Some(effort.as_str().to_string());
        self
    }

    pub fn with_truncate_sequence(mut self, truncate: bool) -> Self {
        self.request.truncate_sequence = Some(truncate);
        self
    }

    pub fn require_file(mut self, name: impl Into<String>) -> Self {
        self.request
            .files
            .get_or_insert_with(Vec::new)
            .push(RequestedFile::new(name));
        self
    }

    pub fn require_file_described(
        mut self,
        name: impl Into<String>,
        format: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        let file = RequestedFile::new(name)
            .with_format(format)
            .with_description(description);
        self.request.files.get_or_insert_with(Vec::new).push(file);
        self
    }

    pub fn with_files(mut self, files: Vec<RequestedFile>) -> Self {
        self.request.files = Some(files);
        self
    }

    /// Attaches `file` to the request's last user message, where the model's file tools can read it.
    pub fn with_input_file(mut self, file: InputFile) -> Self {
        self.input_files.push(file);
        self
    }

    pub fn with_input_files(mut self, files: Vec<InputFile>) -> Self {
        self.input_files = files;
        self
    }

    /// The engine request this builds, with every option set directly.
    pub fn request_mut(&mut self) -> &mut ChatCompletionRequest {
        &mut self.request
    }
}

/// DRY ("don't repeat yourself") sampling, which penalizes extending spans that already appeared.
#[derive(Debug, Clone)]
pub struct DrySampling {
    pub multiplier: f32,
    pub base: Option<f32>,
    pub allowed_length: Option<usize>,
    pub sequence_breakers: Option<Vec<String>>,
}

/// What an encoded attachment holds.
#[derive(Debug, Clone, Copy)]
pub enum EncodedKind {
    Image,
    Audio,
    Video,
}

impl EncodedKind {
    fn part_name(self) -> (&'static str, &'static str) {
        match self {
            Self::Image => ("image_url", "image_url"),
            Self::Audio => ("audio_url", "audio_url"),
            Self::Video => ("video_url", "video_url"),
        }
    }
}

impl From<RequestBuilder> for ChatRequest {
    fn from(builder: RequestBuilder) -> Self {
        let RequestBuilder {
            mut messages,
            media,
            mut request,
            input_files,
            logits_processors,
            approval,
        } = builder;
        if !input_files.is_empty() {
            attach_files(&mut messages, &input_files);
        }
        let messages = serde_json::from_value(Value::Array(messages))
            .expect("the builder only writes messages the protocol accepts");
        request.messages = either::Either::Left(messages);
        Self {
            request,
            media: MediaAttachments::from_media(media),
            logits_processors,
            approval,
        }
    }
}

// Input files ride on the last user message as `file` parts, after whatever it already says.
fn attach_files(messages: &mut Vec<Value>, files: &[InputFile]) {
    let user = messages
        .iter()
        .rposition(|message| message["role"] == TextMessageRole::User.to_string());
    let Some(user) = user else {
        let parts: Vec<Value> = files.iter().map(InputFile::part).collect();
        messages.push(json!({"role": TextMessageRole::User.to_string(), "content": parts}));
        return;
    };
    let message = &mut messages[user];
    let mut parts = match message["content"].take() {
        Value::String(text) => vec![json!({"type": "text", "text": text})],
        Value::Array(parts) => parts,
        _ => Vec::new(),
    };
    parts.extend(files.iter().map(InputFile::part));
    message["content"] = Value::Array(parts);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_request_parses_and_builders_fill_it() {
        let request: ChatRequest = RequestBuilder::new()
            .with_model("named")
            .add_message(TextMessageRole::System, "be brief")
            .add_message(TextMessageRole::User, "hi")
            .set_sampler_stop_token_ids(vec![7])
            .with_input_file(InputFile::from_text("notes.txt", "x"))
            .into();
        assert_eq!(request.request.model, "named");
        assert_eq!(request.request.stop_token_ids, Some(vec![7]));
        let messages = request.request.messages.left().unwrap();
        assert_eq!(messages.len(), 2);
        let user = serde_json::to_value(&messages[1]).unwrap();
        assert_eq!(user["content"][1]["type"], "file", "{user}");
    }

    #[test]
    fn decoded_media_are_named_in_order() {
        let image = DynamicImage::new_rgb8(2, 2);
        let request: ChatRequest = RequestBuilder::new()
            .add_image_message(TextMessageRole::User, "first", vec![image.clone()])
            .add_image_message(TextMessageRole::User, "second", vec![image])
            .into();
        let messages = serde_json::to_value(request.request.messages.left().unwrap()).unwrap();
        assert_eq!(messages[1]["content"][0]["image_url"]["url"], "media://1");
    }
}

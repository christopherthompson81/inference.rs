use either::Either;
use image::DynamicImage;
use indexmap::IndexMap;
use inference_audio::AudioInput;
use inference_quant::IsqType;
use serde::{Deserialize, Serialize};

use crate::VideoInput;

use crate::{
    AdapterSelection, AgentPermission, AgentToolApprovalHandler, CodeExecutionPermission,
    CustomLogitsProcessor, DiffusionGenerationParams, SpeechOptions, Tool, ToolCallbackWithTool,
    response::Response, sampler::SamplingParams, tools::ToolChoice,
};
use std::{
    fmt::Debug,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering as AtomicOrdering},
    },
    time::Instant,
};
use tokio::sync::mpsc::Sender;

pub use inference_protocol::request::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Message or messages for a [`Request`].
pub enum RequestMessage {
    Chat {
        messages: Vec<IndexMap<String, MessageContent>>,
        enable_thinking: Option<bool>,
        /// Reasoning effort level for templates with configurable reasoning
        reasoning_effort: Option<ReasoningEffort>,
    },
    Completion {
        text: String,
        echo_prompt: bool,
        best_of: Option<usize>,
    },
    CompletionTokens(Vec<u32>),
    MultimodalChat {
        #[serde(skip)]
        images: Vec<image::DynamicImage>,
        #[serde(skip)]
        audios: Vec<AudioInput>,
        #[serde(skip)]
        videos: Vec<VideoInput>,
        messages: Vec<IndexMap<String, MessageContent>>,
        enable_thinking: Option<bool>,
        /// Reasoning effort level for templates with configurable reasoning
        reasoning_effort: Option<ReasoningEffort>,
    },
    ImageGeneration {
        prompt: String,
        generation_params: DiffusionGenerationParams,
    },
    SpeechGeneration {
        prompt: String,
        options: SpeechOptions,
    },
    Embedding {
        prompt: String,
    },
    EmbeddingTokens {
        prompt: Vec<u32>,
    },
}

fn default_responder<T>() -> Sender<T> {
    let (sender, _) = tokio::sync::mpsc::channel(1);
    sender
}
#[derive(Clone, Serialize, Deserialize)]
/// A normal request request to the `InferenceRs`.
/// - `messages`: Messages for the request
/// - `sampling_params`: Sampling parameters for generation
/// - `response`: Object to send the result through
/// - `return_logprobs`: Whether to return logprobs
/// - `is_streaming`: Control whether the request is streaming, if so chunk responses will be sent
/// - `id`: Request ID
/// - `constraint`: Constraint to use during generation
/// - `suffix`: Suffix to add
/// - `tools`: Tools available in this request
/// - `tool_choice`: Choice of tools
/// - `logits_processors`: Custom logits processors. Order of application:
///     1) Apply penalties from `sampling_params`
///     2) Apply these custom logits processors sequentially
///     3) Apply temperature and softmax
///     4) Sample the next token (topk, topp, minp, etc)
/// - `return_raw_logits`: Return raw logits.
/// - `truncate_sequence`: Whether to truncate the prompt if it exceeds the model's maximum context length.
pub struct NormalRequest {
    pub messages: RequestMessage,
    pub sampling_params: SamplingParams,
    /// Initializes an independent sampling stream for every generated choice.
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default = "default_responder")]
    #[serde(skip)]
    pub response: Sender<Response>,
    pub return_logprobs: bool,
    pub is_streaming: bool,
    pub id: usize,
    #[doc(hidden)]
    #[serde(skip)]
    pub queued_at: Option<Instant>,
    pub constraint: Constraint,
    pub suffix: Option<String>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip)]
    pub logits_processors: Option<Vec<Arc<dyn CustomLogitsProcessor>>>,
    /// Host tools this request offers on top of the engine's, answered by their callbacks.
    #[serde(skip)]
    pub host_tools: Vec<ToolCallbackWithTool>,
    /// Runs a round's tool calls one at a time, in the model's order.
    #[serde(default)]
    pub sequential_tool_calls: bool,
    pub return_raw_logits: bool,
    pub web_search_options: Option<WebSearchOptions>,
    /// When true, registered code-execution tools are injected and the agentic loop runs.
    #[serde(default)]
    pub enable_code_execution: bool,
    /// When true, registered shell tools are injected and the agentic loop runs.
    #[serde(default)]
    pub enable_shell: bool,
    #[serde(default)]
    pub shell_options: Option<inference_mcp::ShellOptions>,
    #[serde(default)]
    pub code_execution_permission: Option<CodeExecutionPermission>,
    #[serde(skip)]
    pub code_execution_approval_notifier: Option<Arc<inference_mcp::CodeExecutionApprovalNotifier>>,
    #[serde(default)]
    pub agent_permission: Option<AgentPermission>,
    #[serde(skip)]
    pub agent_approval_handler: Option<AgentToolApprovalHandler>,
    #[serde(skip)]
    pub agent_approval_notifier: Option<Arc<inference_mcp::AgentToolApprovalNotifier>>,
    pub max_tool_rounds: Option<usize>,
    /// URL to POST `{"name": ..., "arguments": ...}` to when no server-side callback is registered. Expects `{"content": "..."}` back.
    pub tool_dispatch_url: Option<String>,
    pub model_id: Option<String>,
    #[serde(default)]
    pub adapter: Option<AdapterSelection>,
    #[serde(default)]
    pub truncate_sequence: bool,
    /// Persistent agentic state. If `None`, a new session is created and the ID is returned in the response.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Who the request acts for: its sessions and files are that owner's, and it sees no one else's.
    #[serde(default)]
    pub owner: Option<String>,
    /// Required output files. The runtime asks the model to produce them and surfaces a `File` (or error placeholder) for each.
    #[serde(default)]
    pub files: Option<Vec<crate::files::RequestedFile>>,
    /// User-provided input files attached to this request.
    #[serde(default)]
    pub input_files: Vec<crate::files::File>,
    /// Ends the request's sequences as `canceled` on their next sampled token, with their final response and usage.
    #[serde(skip)]
    // Like a closed response channel, it isn't sent to ring or NCCL workers, whose copies keep stepping.
    pub cancellation: Option<RequestCancellation>,
}

/// The `finish_reason` of a sequence its request canceled.
pub const FINISH_REASON_CANCELED: &str = "canceled";
/// The finish reason of a sequence stopped by its token cap (the request's or the model's).
pub const FINISH_REASON_LENGTH: &str = "length";

/// A flag the requester sets to cancel its request; cloning shares it.
#[derive(Clone, Debug, Default)]
pub struct RequestCancellation(Arc<AtomicBool>);

impl RequestCancellation {
    pub fn cancel(&self) {
        self.0.store(true, AtomicOrdering::Relaxed);
    }

    pub fn is_canceled(&self) -> bool {
        self.0.load(AtomicOrdering::Relaxed)
    }
}

impl NormalRequest {
    pub(crate) fn response_is_closed(&self) -> bool {
        self.response.is_closed()
    }

    pub(crate) fn mark_enqueued(&mut self) {
        self.mark_enqueued_at(Instant::now());
    }

    fn mark_enqueued_at(&mut self, now: Instant) {
        self.queued_at.get_or_insert(now);
    }

    pub(crate) fn take_queue_duration(&mut self) -> Option<std::time::Duration> {
        self.take_queue_duration_at(Instant::now())
    }

    fn take_queue_duration_at(&mut self, now: Instant) -> Option<std::time::Duration> {
        self.queued_at.take().map(|queued_at| now - queued_at)
    }

    /// The chat messages. Panics on a non-chat request, which the agentic path never sees.
    pub fn chat_messages(&self) -> &Vec<IndexMap<String, MessageContent>> {
        match &self.messages {
            RequestMessage::Chat { messages, .. }
            | RequestMessage::MultimodalChat { messages, .. } => messages,
            _ => unreachable!(),
        }
    }

    pub fn chat_messages_mut(&mut self) -> &mut Vec<IndexMap<String, MessageContent>> {
        match &mut self.messages {
            RequestMessage::Chat { messages, .. }
            | RequestMessage::MultimodalChat { messages, .. } => messages,
            _ => unreachable!(),
        }
    }

    /// Upgrade `Chat` to `MultimodalChat` in place. No-op if already multimodal.
    pub fn upgrade_to_multimodal(&mut self) {
        let dummy = RequestMessage::Chat {
            messages: vec![],
            enable_thinking: None,
            reasoning_effort: None,
        };
        let old = std::mem::replace(&mut self.messages, dummy);
        self.messages = match old {
            RequestMessage::Chat {
                messages,
                enable_thinking,
                reasoning_effort,
            } => RequestMessage::MultimodalChat {
                images: Vec::new(),
                audios: Vec::new(),
                videos: Vec::new(),
                messages,
                enable_thinking,
                reasoning_effort,
            },
            other @ RequestMessage::MultimodalChat { .. } => other,
            _ => unreachable!(),
        };
    }

    pub fn images_mut(&mut self) -> &mut Vec<DynamicImage> {
        match &mut self.messages {
            RequestMessage::MultimodalChat { images, .. } => images,
            _ => unreachable!("must call upgrade_to_multimodal first"),
        }
    }

    pub fn videos_mut(&mut self) -> &mut Vec<VideoInput> {
        match &mut self.messages {
            RequestMessage::MultimodalChat { videos, .. } => videos,
            _ => unreachable!("must call upgrade_to_multimodal first"),
        }
    }

    /// Prepends the system message that lists the request's input files, if it has any.
    pub fn inject_input_files_message(&mut self) {
        let Some(content) = crate::files::input_files_message(&self.input_files) else {
            return;
        };
        let mut message: IndexMap<String, MessageContent> = IndexMap::new();
        message.insert("role".to_string(), Either::Left("system".to_string()));
        message.insert("content".to_string(), Either::Left(content));
        self.chat_messages_mut().insert(0, message);
    }

    pub fn new_simple(
        messages: RequestMessage,
        sampling_params: SamplingParams,
        response: Sender<Response>,
        id: usize,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
    ) -> Self {
        Self {
            messages,
            sampling_params,
            seed: None,
            response,
            id,
            queued_at: None,
            tools,
            tool_choice,
            return_logprobs: false,
            is_streaming: false,
            constraint: Constraint::None,
            suffix: None,
            logits_processors: None,
            host_tools: Vec::new(),
            sequential_tool_calls: false,
            return_raw_logits: false,
            web_search_options: None,
            enable_code_execution: false,
            enable_shell: false,
            shell_options: None,
            code_execution_permission: None,
            code_execution_approval_notifier: None,
            agent_permission: None,
            agent_approval_handler: None,
            agent_approval_notifier: None,
            max_tool_rounds: None,
            tool_dispatch_url: None,
            model_id: None,
            adapter: None,
            truncate_sequence: false,
            session_id: None,
            owner: None,
            files: None,
            input_files: Vec::new(),
            cancellation: None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
/// Request to tokenize some messages or some text.
/// - `add_generation_prompt` is only applicable if chat messages are provided and not a raw string.
pub struct TokenizationRequest {
    pub text: Either<Vec<IndexMap<String, MessageContent>>, String>,
    pub tools: Option<Vec<Tool>>,
    pub add_generation_prompt: bool,
    pub add_special_tokens: bool,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default = "default_responder")]
    #[serde(skip)]
    pub response: Sender<anyhow::Result<Vec<u32>>>,
}

#[derive(Clone, Serialize, Deserialize)]
/// Request to detokenize some text.
pub struct DetokenizationRequest {
    pub tokens: Vec<u32>,
    pub skip_special_tokens: bool,
    #[serde(default = "default_responder")]
    #[serde(skip)]
    pub response: Sender<anyhow::Result<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Online calibration lifecycle action.
pub enum CalibrationAction {
    /// Begin collecting activation statistics from live traffic.
    Start,
    /// Report per-layer collection progress.
    Status,
    /// Requantize with the collected statistics and hot-swap the layers.
    Apply {
        save_cimatrix: Option<std::path::PathBuf>,
    },
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CalibrationRequest {
    pub action: CalibrationAction,
    #[serde(default = "default_responder")]
    #[serde(skip)]
    pub response: Sender<anyhow::Result<crate::CalibrationStatus>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RequantizeRequest {
    pub isq: IsqType,
    #[serde(default = "default_responder")]
    #[serde(skip)]
    pub response: Sender<anyhow::Result<()>>,
}

#[derive(Clone, Serialize, Deserialize)]
/// A request to the Engine, encapsulating the various parameters as well as
/// the `mpsc` response `Sender` used to return the [`Response`].
pub enum Request {
    Normal(Box<NormalRequest>),
    ReIsq(RequantizeRequest),
    Calibration(CalibrationRequest),
    Tokenize(TokenizationRequest),
    Detokenize(DetokenizationRequest),
    // Sending a terminate request causes the `run` function to return to the thread created in `InferenceRs::new`,
    // and then Engine will be dropped.
    Terminate,
    TerminateAllSeqsNextStep,
}

impl Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Request::Normal(boxed_req) => {
                let NormalRequest {
                    messages,
                    sampling_params,
                    is_streaming,
                    id,
                    ..
                } = &**boxed_req;
                write!(
                    f,
                    "Request {id} {{ messages: `{messages:?}`, sampling_params: {sampling_params:?}, is_streaming: {is_streaming}}}",
                )
            }
            Request::ReIsq(req) => {
                write!(f, "Re ISQ Request {:?}", req.isq)
            }
            Request::Calibration(req) => {
                write!(f, "Calibration Request {:?}", req.action)
            }
            Request::Tokenize(req) => {
                write!(f, "Tokenization Request {:?}", req.text)
            }
            Request::Detokenize(req) => {
                write!(f, "Tokenization Request {:?}", req.tokens)
            }
            Request::Terminate => write!(f, "Termination Request"),
            Request::TerminateAllSeqsNextStep => write!(f, "Terminate All Seqs Next Step"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_replication_keeps_an_exact_generation() {
        let (response, _) = tokio::sync::mpsc::channel(1);
        let mut request = NormalRequest::new_simple(
            RequestMessage::Completion {
                text: "hello".to_string(),
                echo_prompt: false,
                best_of: None,
            },
            SamplingParams::neutral(),
            response,
            0,
            None,
            None,
        );
        let generation = crate::AdapterGenerationId::from_bytes([7; 32]);
        request.adapter = Some(AdapterSelection::generation(generation));
        request.mark_enqueued();

        let serialized = serde_json::to_string(&Request::Normal(Box::new(request))).unwrap();
        let Request::Normal(request) = serde_json::from_str::<Request>(&serialized).unwrap() else {
            panic!("expected a normal request");
        };
        assert_eq!(
            request
                .adapter
                .as_ref()
                .and_then(AdapterSelection::resolved_generation),
            Some(generation)
        );
        assert!(request.queued_at.is_none());
    }

    #[test]
    fn queue_duration_preserves_ingress_time_and_is_consumed_once() {
        let (response, _) = tokio::sync::mpsc::channel(1);
        let mut request = NormalRequest::new_simple(
            RequestMessage::Completion {
                text: "hello".to_string(),
                echo_prompt: false,
                best_of: None,
            },
            SamplingParams::neutral(),
            response,
            0,
            None,
            None,
        );
        let ingress = Instant::now();
        request.mark_enqueued_at(ingress);
        request.mark_enqueued_at(ingress + std::time::Duration::from_secs(1));

        assert_eq!(
            request.take_queue_duration_at(ingress + std::time::Duration::from_secs(2)),
            Some(std::time::Duration::from_secs(2))
        );
        assert_eq!(
            request.take_queue_duration_at(ingress + std::time::Duration::from_secs(3)),
            None
        );
    }
}

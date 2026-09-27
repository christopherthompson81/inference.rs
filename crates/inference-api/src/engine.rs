//! A loaded engine and the operations it serves, as OpenAI-style requests and responses.

use candle_core::Device;
use futures::StreamExt;
use inference_core::{
    AgentPermission, ChatCompletionResponse, InferenceRs, ModelSelected, Response, TokenSource,
};
use serde::Deserialize;

use crate::{
    agentic::AgenticDefaults,
    api_error::{ApiError, ApiErrorKind, ModelErrorMessage},
    engine_chat::{collect_chat, ChatDispatchError, ChatEngine, ChatStream, ChatStreamEvent},
    inference_for_server_builder::InferenceRsForServerBuilder,
    openai::{ChatCompletionRequest, OpenAiToolSurface},
    types::SharedInferenceRsState,
};

const INVALID_REQUEST_BODY: &str = "invalid_request_body";
// Matches `inference serve`'s default, so an engine loaded from a spec batches like the server.
pub const DEFAULT_MAX_SEQS: usize = 32;
const ASK_UNAVAILABLE: &str =
    "agent_permission \"ask\" needs approval resolution, which this surface does not offer yet";

/// What to load and how to run it: the JSON form of the options `inference serve` takes. Skills are not served yet.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineSpec {
    pub model: ModelSelected,
    /// The id requests use for this model; defaults to the model's own id.
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub runtime: RuntimeSpec,
    #[serde(default)]
    pub agentic: AgenticSpec,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeSpec {
    /// `auto` (the best available), `cpu`, `cuda:N` or `metal:N`.
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub max_seqs: Option<usize>,
    /// Prefix cache capacity in sequences; 0 disables the prefix cache.
    #[serde(default)]
    pub prefix_cache_n: Option<usize>,
    #[serde(default)]
    pub no_kv_cache: bool,
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub jinja_explicit: Option<String>,
    #[serde(default)]
    pub max_model_len: Option<usize>,
    /// In-situ quantization, e.g. `q4k`.
    #[serde(default)]
    pub isq: Option<String>,
    /// Paged attention: unset picks the device default.
    #[serde(default)]
    pub paged_attn: Option<bool>,
    /// `cache`, `none`, `env:VAR`, `literal:TOKEN` or `path:FILE`.
    #[serde(default)]
    pub token_source: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgenticSpec {
    #[serde(default)]
    pub max_tool_rounds: Option<usize>,
    #[serde(default)]
    pub tool_dispatch_url: Option<String>,
    #[serde(default)]
    pub agent_permission: Option<AgentPermission>,
}

/// Why an engine did not load: the spec itself was unusable, or loading the model failed.
#[derive(Debug)]
pub enum EngineLoadError {
    InvalidSpec(String),
    /// The requested device is not compiled into this build or is not present.
    DeviceUnavailable(String),
    Load(anyhow::Error),
}

impl std::fmt::Display for EngineLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSpec(message) => write!(f, "invalid engine spec: {message}"),
            Self::DeviceUnavailable(message) => write!(f, "device unavailable: {message}"),
            Self::Load(error) => write!(f, "{error:#}"),
        }
    }
}

impl std::error::Error for EngineLoadError {}

impl EngineSpec {
    fn into_builder(self) -> Result<InferenceRsForServerBuilder, EngineLoadError> {
        let invalid = EngineLoadError::InvalidSpec;
        if matches!(self.agentic.agent_permission, Some(AgentPermission::Ask)) {
            return Err(invalid(ASK_UNAVAILABLE.into()));
        }
        let runtime = self.runtime;
        let mut builder = InferenceRsForServerBuilder::new()
            .with_model(self.model)
            .with_model_id_override_optional(self.model_id)
            .with_no_kv_cache(runtime.no_kv_cache)
            .with_chat_template_optional(runtime.chat_template)
            .with_jinja_explicit_optional(runtime.jinja_explicit)
            .with_max_model_len_optional(runtime.max_model_len)
            .with_in_situ_quant_optional(runtime.isq)
            .with_seed_optional(runtime.seed)
            .set_paged_attn(runtime.paged_attn)
            .with_max_seqs(runtime.max_seqs.unwrap_or(DEFAULT_MAX_SEQS));
        if let Some(prefix_cache_n) = runtime.prefix_cache_n {
            builder = builder.with_prefix_cache_n(prefix_cache_n);
        }
        if let Some(token_source) = runtime.token_source {
            let token_source: TokenSource = token_source.parse().map_err(invalid)?;
            builder = builder.with_token_source(token_source);
        }
        builder = match runtime.device.as_deref() {
            None | Some("auto") => builder,
            Some("cpu") => builder.with_cpu(true),
            Some(device) => builder.with_device(explicit_device(device, runtime.seed)?),
        };
        Ok(builder)
    }
}

fn explicit_device(device: &str, seed: Option<u64>) -> Result<Device, EngineLoadError> {
    let invalid = || {
        EngineLoadError::InvalidSpec(format!(
            "device `{device}` is not one of auto, cpu, cuda:N, metal:N"
        ))
    };
    let (kind, ordinal) = device.split_once(':').ok_or_else(invalid)?;
    let ordinal: usize = ordinal.parse().map_err(|_| invalid())?;
    let device = match kind {
        "cuda" => Device::new_cuda(ordinal),
        "metal" => Device::new_metal(ordinal),
        _ => return Err(invalid()),
    }
    .map_err(|error| EngineLoadError::DeviceUnavailable(format!("{device}: {error}")))?;
    // An explicit device skips the builder's own device setup, which is where the seed is applied.
    if let Some(seed) = seed {
        device
            .set_seed(seed)
            .map_err(|error| EngineLoadError::Load(error.into()))?;
    }
    Ok(device)
}

/// A loaded engine. Cloning shares it.
#[derive(Clone)]
pub struct Engine {
    chat: ChatEngine,
}

impl Engine {
    /// Wraps an engine the caller already built, with its server-level chat policy.
    pub fn new(chat: ChatEngine) -> Self {
        Self { chat }
    }

    pub async fn load(spec: EngineSpec) -> Result<Self, EngineLoadError> {
        let agentic = AgenticDefaults {
            max_tool_rounds: spec.agentic.max_tool_rounds,
            tool_dispatch_url: spec.agentic.tool_dispatch_url.clone(),
            agent_permission: spec.agentic.agent_permission,
            approval_broker: Default::default(),
        };
        let state = spec
            .into_builder()?
            .build()
            .await
            .map_err(EngineLoadError::Load)?;
        Ok(Self::new(ChatEngine {
            state,
            agentic,
            skill_store: None,
        }))
    }

    pub async fn load_json(spec: &[u8]) -> Result<Self, EngineLoadError> {
        let spec = serde_json::from_slice(spec)
            .map_err(|error| EngineLoadError::InvalidSpec(error.to_string()))?;
        Self::load(spec).await
    }

    pub fn state(&self) -> &SharedInferenceRsState {
        &self.chat.state
    }

    /// Runs a chat completion to its end and returns the full response.
    pub async fn chat(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ApiError> {
        request.stream = Some(false);
        let prepared = self.prepare(request).await?;
        let mut rx = prepared.rx;
        let response = collect_chat(&mut rx, prepared.model_override.as_deref()).await;
        let state = self.state().clone();
        match response {
            Response::Done(response) => {
                InferenceRs::maybe_log_response(state, &response);
                Ok(response)
            }
            Response::ModelError(msg, response) => {
                InferenceRs::maybe_log_error(state.clone(), &ModelErrorMessage(msg));
                InferenceRs::maybe_log_response(state, &response);
                Err(ApiError::model_error())
            }
            Response::ValidationError(error) => Err(ApiError::from_error(
                error.as_ref(),
                ApiErrorKind::InvalidRequest,
            )),
            Response::InternalError(error) => {
                InferenceRs::maybe_log_error(state, &*error);
                Err(ApiError::from_error(error.as_ref(), ApiErrorKind::Internal))
            }
            _ => Err(ApiError::internal()),
        }
    }

    /// Starts a streaming chat completion. Dropping the stream abandons the request.
    pub async fn chat_stream(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<ChatStream, ApiError> {
        request.stream = Some(true);
        let prepared = self.prepare(request).await?;
        Ok(ChatStream::new(
            prepared.rx,
            self.state().clone(),
            prepared.model_override,
            None,
        ))
    }

    /// [`Engine::chat`] over JSON: an OpenAI chat completion request in, the response out.
    pub async fn chat_json(&self, request: &[u8]) -> Result<String, ApiError> {
        let response = self.chat(parse_request(request)?).await?;
        serde_json::to_string(&response).map_err(|_| ApiError::internal())
    }

    /// [`Engine::chat_stream`] over JSON; each event serializes with [`ChatStreamEvent::to_json`].
    pub async fn chat_stream_json(&self, request: &[u8]) -> Result<ChatStream, ApiError> {
        self.chat_stream(parse_request(request)?).await
    }

    async fn prepare(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<crate::engine_chat::PreparedChat, ApiError> {
        let asks = request
            .agent_permission
            .or_else(|| request.code_execution_permission.map(Into::into))
            .is_some_and(|permission| permission == AgentPermission::Ask);
        if asks {
            return Err(ApiError::new(
                ApiErrorKind::InvalidRequest,
                ASK_UNAVAILABLE,
                Some("unsupported_parameter"),
                Some("agent_permission"),
            ));
        }
        let state = self.state().clone();
        self.chat
            .prepare(request, OpenAiToolSurface::ChatCompletions)
            .await
            .map_err(|error| match error {
                ChatDispatchError::Validation(error) => {
                    let api = ApiError::from_error(error.as_ref(), ApiErrorKind::InvalidRequest);
                    if matches!(
                        api.kind,
                        ApiErrorKind::Internal
                            | ApiErrorKind::Unavailable
                            | ApiErrorKind::Overloaded
                    ) {
                        InferenceRs::maybe_log_error(state, error.as_ref());
                    }
                    api
                }
                ChatDispatchError::Internal(error) => {
                    InferenceRs::maybe_log_error(state, error.as_ref());
                    ApiError::from_error(error.as_ref(), ApiErrorKind::Internal)
                }
            })
    }
}

fn parse_request(request: &[u8]) -> Result<ChatCompletionRequest, ApiError> {
    serde_json::from_slice(request).map_err(|error| {
        ApiError::new(
            ApiErrorKind::InvalidRequest,
            error.to_string(),
            Some(INVALID_REQUEST_BODY),
            None,
        )
    })
}

impl ChatStreamEvent {
    /// The event's name in the JSON envelope.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Chunk(_) => "chunk",
            Self::AgenticToolCallProgress(_) => "agentic_tool_call_progress",
            Self::AgenticToolApprovalRequired(_) => "agentic_tool_approval_required",
            Self::FileProduced(_) => "file_produced",
            Self::Error(_) => "error",
        }
    }

    /// `{"event": <name>, "data": <payload>}`; an error's payload is the OpenAI error envelope.
    pub fn to_json(&self) -> String {
        let data = match self {
            Self::Chunk(chunk) => serde_json::to_value(chunk),
            Self::AgenticToolCallProgress(value) | Self::AgenticToolApprovalRequired(value) => {
                Ok(value.clone())
            }
            Self::FileProduced(file) => serde_json::to_value(file),
            Self::Error(error) => Ok(error.to_openai_body()),
        }
        .unwrap_or_else(|_| ApiError::internal().to_openai_body());
        serde_json::json!({ "event": self.name(), "data": data }).to_string()
    }
}

impl ChatStream {
    /// The next event, or `None` once the stream has finished.
    pub async fn next_event(&mut self) -> Option<ChatStreamEvent> {
        self.next().await
    }
}

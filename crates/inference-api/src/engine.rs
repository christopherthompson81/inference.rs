//! A loaded engine and the operations it serves, as OpenAI-style requests and responses.

use candle_core::Device;
use futures::StreamExt;
use inference_core::{
    AgentPermission, ChatCompletionResponse, CompletionResponse, ImageGenerationResponse,
    InferenceRs, ModelSelected, Response, TokenSource,
};
use serde::Deserialize;

use crate::{
    agentic::AgenticDefaults,
    agentic::{resolve_approval, ApprovalDecisionRequest, ApprovalDecisionResponse},
    anthropic::{
        collect_messages, prepare_messages, AnthropicMessageResponse, AnthropicMessagesRequest,
        AnthropicStream, MessagesFailure,
    },
    api_error::{ApiError, ApiErrorKind, ModelErrorMessage},
    engine_chat::{collect_chat, ChatEngine, ChatStream, ChatStreamEvent},
    engine_completion::{collect_completion, prepare_completion, CompletionStream},
    engine_embeddings::{embed, EmbeddingError},
    files::{self, FileBody, FileMetadata, FileUpload},
    generation::{generate_image, generate_speech, SpeechAudio},
    inference_for_server_builder::InferenceRsForServerBuilder,
    lora_adapters::{
        list_adapters, load_adapter, unload_adapter, ListLoraAdaptersQuery, LoadLoraAdapterRequest,
        LoraAdapterApiConfig, LoraAdapterListResponse, LoraAdapterObject, UnloadLoraAdapterRequest,
    },
    media_source::MediaAttachments,
    models::{
        list_models, model_status, reload_model, unload_model, ModelOperationRequest,
        ModelStatusResponse,
    },
    openai::{
        ChatCompletionRequest, CompletionRequest, EmbeddingRequest, EmbeddingResponse,
        ImageGenerationRequest, ModelObjects, OpenAiToolSurface, SpeechGenerationRequest,
    },
    responses::{
        cancel_response, collect_response, delete_response, get_response, prepare_response,
        spawn_background, OpenResponsesCreateRequest, OpenResponsesStreamer, PreparedResponse,
        ResponseDeleted,
    },
    responses_types::ResponseResource,
    types::SharedInferenceRsState,
};

const INVALID_REQUEST_BODY: &str = "invalid_request_body";
// Matches `inference serve`'s default, so an engine loaded from a spec batches like the server.
pub const DEFAULT_MAX_SEQS: usize = 32;

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
    #[serde(default)]
    pub adapters: AdapterSpec,
}

/// Runtime LoRA adapter management; listing adapters is always allowed.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterSpec {
    /// Allow loading and unloading adapters while the engine runs.
    #[serde(default)]
    pub runtime_updates: bool,
    /// The directory adapters must load from; relative adapter paths resolve under it.
    #[serde(default)]
    pub root: Option<std::path::PathBuf>,
}

impl AdapterSpec {
    fn into_config(self) -> Result<LoraAdapterApiConfig, EngineLoadError> {
        let mut config = LoraAdapterApiConfig::default().with_enabled(self.runtime_updates);
        if let Some(root) = self.root {
            config = config.with_allowed_root(root);
        }
        config
            .prepare()
            .map_err(|error| EngineLoadError::InvalidSpec(format!("{error:#}")))
    }
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
    adapters: LoraAdapterApiConfig,
}

impl Engine {
    /// Wraps an engine the caller already built, with its server-level chat policy and adapter management policy.
    pub fn new(chat: ChatEngine, adapters: LoraAdapterApiConfig) -> Self {
        Self { chat, adapters }
    }

    pub async fn load(mut spec: EngineSpec) -> Result<Self, EngineLoadError> {
        let adapters = std::mem::take(&mut spec.adapters).into_config()?;
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
        Ok(Self::new(
            ChatEngine {
                state,
                agentic,
                skill_store: None,
            },
            adapters,
        ))
    }

    pub async fn load_json(spec: &[u8]) -> Result<Self, EngineLoadError> {
        let spec = serde_json::from_slice(spec)
            .map_err(|error| EngineLoadError::InvalidSpec(error.to_string()))?;
        Self::load(spec).await
    }

    pub fn state(&self) -> &SharedInferenceRsState {
        &self.chat.state
    }

    /// Runs a chat completion to its end; `media` holds the buffers its `media://N` sources name.
    pub async fn chat(
        &self,
        mut request: ChatCompletionRequest,
        media: MediaAttachments,
    ) -> Result<ChatCompletionResponse, ApiError> {
        request.stream = Some(false);
        let prepared = self.prepare(request, media).await?;
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
        media: MediaAttachments,
    ) -> Result<ChatStream, ApiError> {
        request.stream = Some(true);
        let prepared = self.prepare(request, media).await?;
        Ok(ChatStream::new(
            prepared.rx,
            self.state().clone(),
            prepared.model_override,
            None,
        ))
    }

    /// [`Engine::chat`] over JSON: an OpenAI chat completion request in, the response out.
    pub async fn chat_json(
        &self,
        request: &[u8],
        media: MediaAttachments,
    ) -> Result<String, ApiError> {
        let response = self.chat(parse_json(request)?, media).await?;
        to_json(&response)
    }

    /// [`Engine::chat_stream`] over JSON; each event serializes with [`ChatStreamEvent::to_json`].
    pub async fn chat_stream_json(
        &self,
        request: &[u8],
        media: MediaAttachments,
    ) -> Result<ChatStream, ApiError> {
        self.chat_stream(parse_json(request)?, media).await
    }

    async fn prepare(
        &self,
        request: ChatCompletionRequest,
        media: MediaAttachments,
    ) -> Result<crate::engine_chat::PreparedChat, ApiError> {
        let state = self.state().clone();
        self.chat
            .prepare(request, OpenAiToolSurface::ChatCompletions, media)
            .await
            .map_err(|error| error.into_api_error(state))
    }

    /// Runs a completion to its end.
    pub async fn completion(
        &self,
        mut request: CompletionRequest,
    ) -> Result<CompletionResponse, ApiError> {
        request.stream = Some(false);
        let state = self.state().clone();
        let prepared = prepare_completion(&state, request)
            .await
            .map_err(|error| error.into_api_error(state.clone()))?;
        let mut rx = prepared.rx;
        match collect_completion(&mut rx, prepared.model_override.as_deref()).await {
            Response::CompletionDone(response) => {
                InferenceRs::maybe_log_response(state, &response);
                Ok(response)
            }
            Response::CompletionModelError(msg, response) => {
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

    /// Starts a streaming completion. Dropping the stream abandons the request.
    pub async fn completion_stream(
        &self,
        mut request: CompletionRequest,
    ) -> Result<CompletionStream, ApiError> {
        request.stream = Some(true);
        let state = self.state().clone();
        let prepared = prepare_completion(&state, request)
            .await
            .map_err(|error| error.into_api_error(state.clone()))?;
        Ok(CompletionStream::new(
            prepared.rx,
            state,
            prepared.model_override,
            None,
        ))
    }

    pub async fn completion_json(&self, request: &[u8]) -> Result<String, ApiError> {
        let response = self.completion(parse_json(request)?).await?;
        to_json(&response)
    }

    pub async fn completion_stream_json(
        &self,
        request: &[u8],
    ) -> Result<CompletionStream, ApiError> {
        self.completion_stream(parse_json(request)?).await
    }

    /// Runs an Anthropic Messages request to its end.
    pub async fn anthropic_messages(
        &self,
        mut request: AnthropicMessagesRequest,
    ) -> Result<AnthropicMessageResponse, ApiError> {
        request.stream = Some(false);
        let state = self.state().clone();
        let prepared = prepare_messages(&self.chat, request)
            .await
            .map_err(|error| error.into_api_error(state.clone()))?;
        let (omit_thinking, model_override) =
            (prepared.omit_thinking, prepared.chat.model_override);
        let mut rx = prepared.chat.rx;
        collect_messages(&mut rx, state, model_override.as_deref(), omit_thinking)
            .await
            .map_err(|failure| match failure {
                MessagesFailure::Validation(error) => {
                    ApiError::from_error(error.as_ref(), ApiErrorKind::InvalidRequest)
                }
                MessagesFailure::Internal(error) => {
                    ApiError::from_error(error.as_ref(), ApiErrorKind::Internal)
                }
                MessagesFailure::Model(_) => ApiError::model_error(),
            })
    }

    /// Starts a streaming Anthropic Messages request. Dropping the stream abandons the request.
    pub async fn anthropic_messages_stream(
        &self,
        mut request: AnthropicMessagesRequest,
    ) -> Result<AnthropicStream, ApiError> {
        request.stream = Some(true);
        let state = self.state().clone();
        let prepared = prepare_messages(&self.chat, request)
            .await
            .map_err(|error| error.into_api_error(state.clone()))?;
        Ok(AnthropicStream::new(prepared, state, None))
    }

    pub async fn anthropic_messages_json(&self, request: &[u8]) -> Result<String, ApiError> {
        let response = self.anthropic_messages(parse_json(request)?).await?;
        to_json(&response)
    }

    pub async fn anthropic_messages_stream_json(
        &self,
        request: &[u8],
    ) -> Result<AnthropicStream, ApiError> {
        self.anthropic_messages_stream(parse_json(request)?).await
    }

    /// Runs a Responses request to its end, or queues it when it asks for `background` and returns it queued.
    pub async fn responses(
        &self,
        mut request: OpenResponsesCreateRequest,
    ) -> Result<ResponseResource, ApiError> {
        request.stream = Some(false);
        let prepared = self.prepare_response(request).await?;
        if prepared.background {
            return Ok(spawn_background(prepared, self.state().clone()));
        }
        collect_response(prepared, self.state()).await
    }

    /// Starts a streaming Responses request. Dropping the stream abandons the request.
    pub async fn responses_stream(
        &self,
        mut request: OpenResponsesCreateRequest,
    ) -> Result<OpenResponsesStreamer, ApiError> {
        request.stream = Some(true);
        let prepared = self.prepare_response(request).await?;
        Ok(OpenResponsesStreamer::new(
            prepared,
            self.state().clone(),
            None,
        ))
    }

    async fn prepare_response(
        &self,
        request: OpenResponsesCreateRequest,
    ) -> Result<PreparedResponse, ApiError> {
        let state = self.state().clone();
        prepare_response(&state, self.chat.skill_store.clone(), request)
            .await
            .map_err(|error| error.into_api_error(state))
    }

    /// A background response in its current state, or a stored one. The store is shared by the process's engines.
    pub fn response(&self, response_id: &str) -> Result<ResponseResource, ApiError> {
        get_response(self.state(), response_id)
    }

    pub fn delete_response(&self, response_id: &str) -> Result<ResponseDeleted, ApiError> {
        delete_response(self.state(), response_id)
    }

    /// Cancels a background response that has not finished, and returns it.
    pub fn cancel_response(&self, response_id: &str) -> Result<ResponseResource, ApiError> {
        cancel_response(self.state(), response_id)
    }

    pub async fn responses_json(&self, request: &[u8]) -> Result<String, ApiError> {
        to_json(&self.responses(parse_json(request)?).await?)
    }

    pub async fn responses_stream_json(
        &self,
        request: &[u8],
    ) -> Result<OpenResponsesStreamer, ApiError> {
        self.responses_stream(parse_json(request)?).await
    }

    pub fn response_json(&self, response_id: &str) -> Result<String, ApiError> {
        to_json(&self.response(response_id)?)
    }

    pub fn delete_response_json(&self, response_id: &str) -> Result<String, ApiError> {
        to_json(&self.delete_response(response_id)?)
    }

    pub fn cancel_response_json(&self, response_id: &str) -> Result<String, ApiError> {
        to_json(&self.cancel_response(response_id)?)
    }

    pub fn models(&self) -> Result<ModelObjects, ApiError> {
        list_models(self.state())
    }

    pub fn unload_model(
        &self,
        request: ModelOperationRequest,
    ) -> Result<ModelStatusResponse, ApiError> {
        unload_model(self.state(), request)
    }

    pub async fn reload_model(
        &self,
        request: ModelOperationRequest,
    ) -> Result<ModelStatusResponse, ApiError> {
        reload_model(self.state(), request).await
    }

    pub fn model_status(
        &self,
        request: ModelOperationRequest,
    ) -> Result<ModelStatusResponse, ApiError> {
        model_status(self.state(), request)
    }

    pub async fn lora_adapters(
        &self,
        query: ListLoraAdaptersQuery,
    ) -> Result<LoraAdapterListResponse, ApiError> {
        list_adapters(self.state(), &self.adapters, query).await
    }

    /// Loads a LoRA adapter; the spec's `adapters.runtime_updates` must allow it.
    pub async fn load_lora_adapter(
        &self,
        request: LoadLoraAdapterRequest,
    ) -> Result<LoraAdapterObject, ApiError> {
        load_adapter(self.state(), &self.adapters, request).await
    }

    pub async fn unload_lora_adapter(
        &self,
        request: UnloadLoraAdapterRequest,
    ) -> Result<LoraAdapterObject, ApiError> {
        unload_adapter(self.state(), &self.adapters, request).await
    }

    pub fn models_json(&self) -> Result<String, ApiError> {
        to_json(&self.models()?)
    }

    pub fn unload_model_json(&self, request: &[u8]) -> Result<String, ApiError> {
        to_json(&self.unload_model(parse_json(request)?)?)
    }

    pub async fn reload_model_json(&self, request: &[u8]) -> Result<String, ApiError> {
        to_json(&self.reload_model(parse_json(request)?).await?)
    }

    pub fn model_status_json(&self, request: &[u8]) -> Result<String, ApiError> {
        to_json(&self.model_status(parse_json(request)?)?)
    }

    pub async fn lora_adapters_json(&self, request: &[u8]) -> Result<String, ApiError> {
        to_json(&self.lora_adapters(parse_json(request)?).await?)
    }

    pub async fn load_lora_adapter_json(&self, request: &[u8]) -> Result<String, ApiError> {
        to_json(&self.load_lora_adapter(parse_json(request)?).await?)
    }

    pub async fn unload_lora_adapter_json(&self, request: &[u8]) -> Result<String, ApiError> {
        to_json(&self.unload_lora_adapter(parse_json(request)?).await?)
    }

    /// Generates images with a diffusion model.
    pub async fn image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, ApiError> {
        generate_image(self.state(), request).await
    }

    /// Speaks text with a speech model, as WAV or 16-bit PCM.
    pub async fn speech_generation(
        &self,
        request: SpeechGenerationRequest,
    ) -> Result<SpeechAudio, ApiError> {
        generate_speech(self.state(), request).await
    }

    pub async fn image_generation_json(&self, request: &[u8]) -> Result<String, ApiError> {
        to_json(&self.image_generation(parse_json(request)?).await?)
    }

    pub async fn speech_generation_json(&self, request: &[u8]) -> Result<SpeechAudio, ApiError> {
        self.speech_generation(parse_json(request)?).await
    }

    /// Answers the approval an `agentic_tool_approval_required` stream event named.
    pub fn resolve_approval(
        &self,
        approval_id: &str,
        request: ApprovalDecisionRequest,
    ) -> Result<ApprovalDecisionResponse, ApiError> {
        resolve_approval(&self.chat.agentic.approval_broker, approval_id, request)
    }

    pub fn resolve_approval_json(
        &self,
        approval_id: &str,
        request: &[u8],
    ) -> Result<String, ApiError> {
        to_json(&self.resolve_approval(approval_id, parse_json(request)?)?)
    }

    pub fn upload_file(&self, upload: FileUpload) -> Result<FileMetadata, ApiError> {
        files::upload_file(self.state(), upload)
    }

    pub fn upload_file_json(&self, upload: FileUpload) -> Result<String, ApiError> {
        to_json(&self.upload_file(upload)?)
    }

    pub fn files_json(&self) -> Result<String, ApiError> {
        to_json(&files::list_files(self.state())?)
    }

    pub fn file_json(&self, file_id: &str) -> Result<String, ApiError> {
        to_json(&files::get_file(self.state(), file_id)?)
    }

    pub fn delete_file_json(&self, file_id: &str) -> Result<String, ApiError> {
        to_json(&files::delete_file(self.state(), file_id)?)
    }

    pub fn file_content(&self, file_id: &str) -> Result<FileBody, ApiError> {
        files::file_content(self.state(), file_id)
    }

    /// Embeds every input of an embeddings request.
    pub async fn embeddings(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, ApiError> {
        let state = self.state().clone();
        embed(state.clone(), request)
            .await
            .map_err(|error| match error {
                EmbeddingError::Validation(error) => {
                    ApiError::from_error(error.as_ref(), ApiErrorKind::InvalidRequest)
                }
                EmbeddingError::Internal(error) => {
                    ApiError::from_error(error.as_ref(), ApiErrorKind::Internal)
                }
            })
    }

    pub async fn embeddings_json(&self, request: &[u8]) -> Result<String, ApiError> {
        let response = self.embeddings(parse_json(request)?).await?;
        to_json(&response)
    }
}

fn parse_json<T: serde::de::DeserializeOwned>(request: &[u8]) -> Result<T, ApiError> {
    serde_json::from_slice(request).map_err(|error| {
        ApiError::new(
            ApiErrorKind::InvalidRequest,
            error.to_string(),
            Some(INVALID_REQUEST_BODY),
            None,
        )
    })
}

fn to_json(response: &impl serde::Serialize) -> Result<String, ApiError> {
    serde_json::to_string(response).map_err(|_| ApiError::internal())
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

//! A loaded engine and the operations it serves, as OpenAI-style requests and responses.

use std::sync::Arc;

use candle_core::Device;
use futures::StreamExt;
use inference_core::{
    AgentPermission, AnyMoeSpec, CalibrationAction, CalibrationStatus, ChatCompletionResponse,
    CodeExecutionConfig, CompletionResponse, HfConfigOverrides, ImageGenerationResponse,
    InferenceRs, McpClientConfig, MtpConfig, MtpDraftSamplingMethod, NetworkMode, PagedCacheType,
    Response, SandboxMode, SandboxPolicy, SandboxProfile, SearchCallback, SearchEmbeddingModel,
    SerializedSession, ShellConfig, TokenSource, ToolCallbackWithTool,
};
use inference_selection::ModelSelected;
use inference_selection::quant;
use serde::Deserialize;

use crate::{
    agentic::AgenticDefaults,
    agentic::{ApprovalDecisionRequest, ApprovalDecisionResponse, resolve_approval},
    anthropic::{
        AnthropicMessageResponse, AnthropicMessagesRequest, AnthropicStream, MessagesFailure,
        collect_messages, prepare_messages,
    },
    api_error::{ApiError, ApiErrorKind, ModelErrorMessage},
    engine_chat::{ChatEngine, ChatStream, ChatStreamEvent, collect_chat},
    engine_completion::{CompletionStream, collect_completion, prepare_completion},
    engine_embeddings::{EmbeddingError, embed},
    files::{self, FileBody, FileMetadata, FileUpload},
    generation::{SpeechAudio, generate_image, generate_speech},
    inference_for_server_builder::{
        InferenceRsForServerBuilder, ModelConfig, defaults, parse_device_layers,
    },
    lora_adapters::{
        ListLoraAdaptersQuery, LoadLoraAdapterRequest, LoraAdapterApiConfig,
        LoraAdapterListResponse, LoraAdapterObject, UnloadLoraAdapterRequest, list_adapters,
        load_adapter, unload_adapter,
    },
    media_source::MediaAttachments,
    models::{
        ModelOperationRequest, ModelStatusResponse, list_models, model_status, reload_model,
        unload_model,
    },
    openai::{
        ChatCompletionRequest, CompletionRequest, EmbeddingRequest, EmbeddingResponse,
        ImageGenerationRequest, ModelObjects, OpenAiToolSurface, SpeechGenerationRequest,
    },
    operations::{
        self, CalibrationApplyRequest, DetokenizeRequest, DetokenizeResponse, ReIsqRequest,
        ReIsqResponse, SessionDeleted, SessionList, SessionStored, TokenizeRequest,
        TokenizeResponse,
    },
    request_body::JsonRequest,
    responses::{
        OpenResponsesCreateRequest, OpenResponsesStreamer, PreparedResponse, ResponseDeleted,
        cancel_response, collect_response, delete_response, get_response, prepare_response,
        spawn_background,
    },
    responses_types::ResponseResource,
    skill_store::{
        AnthropicSkillVersionListObject, AnthropicSkillVersionObject, SkillFiles, SkillListObject,
        SkillStore, skill_api_error,
    },
    types::SharedInferenceRsState,
};

const ONE_MODEL_SOURCE: &str = "give either `model` or a non-empty `models`, not both";
const DEFAULT_WITHOUT_MODELS: &str =
    "`default_model_id` picks one of `models`; with `model`, use `model_id`";
const ANYMOE_WITH_MODELS: &str = "`anymoe` wraps the single `model`; it cannot apply to `models`";
const MODEL_ID_WITH_MODELS: &str =
    "`model_id` names `model`; with `models`, give each its own `model_id`";
const PAGED_CACHE_ONE_SIZE: &str =
    "paged_cache takes at most one of context_len, memory_mb and memory_fraction";
const CODE_EXECUTION_UNAVAILABLE: &str =
    "code execution and the shell tool need a build with the `code-execution` feature";
const QUANT_WITH_ISQ: &str = "`quant` picks the quantization itself; drop `isq`";
// Matches `inference serve`'s default, so an engine loaded from a spec batches like the server.
pub const DEFAULT_MAX_SEQS: usize = 32;

/// What to load and how to run it: the JSON form of the options `inference serve` takes.
#[derive(Debug, Default, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct EngineSpec {
    /// The one model to serve; give `models` instead to serve several.
    #[serde(default)]
    pub model: Option<ModelSelected>,
    /// The id requests use for `model`; defaults to the model's own id.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Several models served by one engine, each with its own overrides of the runtime settings.
    #[serde(default)]
    pub models: Vec<ModelSpec>,
    /// Which of `models` a request without a `model` goes to; defaults to the first.
    #[serde(default)]
    pub default_model_id: Option<String>,
    #[serde(default)]
    pub runtime: RuntimeSpec,
    #[serde(default)]
    pub agentic: AgenticSpec,
    #[serde(default)]
    pub adapters: AdapterSpec,
    #[serde(default)]
    pub skills: SkillsSpec,
    /// Mixes the model's MLPs with expert models' through a trained gate.
    #[serde(default)]
    pub anymoe: Option<AnyMoeSpec>,
}

/// One of several models an engine serves; unset settings fall back to `runtime`'s.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub model: ModelSelected,
    /// The id requests use for this model; defaults to the model's own id.
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub jinja_explicit: Option<String>,
    #[serde(default)]
    pub max_model_len: Option<usize>,
    #[serde(default)]
    #[schema(value_type = Option<Object>)]
    pub hf_config_overrides: Option<HfConfigOverrides>,
    #[serde(default)]
    pub device_layers: Option<Vec<String>>,
    #[serde(default)]
    pub isq: Option<String>,
    #[serde(default)]
    pub encoder_cache_memory_bytes: Option<usize>,
}

impl ModelSpec {
    fn into_config(self, index: usize) -> Result<ModelConfig, EngineLoadError> {
        if let Some(device_layers) = &self.device_layers {
            parse_device_layers(device_layers).map_err(|error| {
                EngineLoadError::InvalidSpec(format!("models[{index}]: {error:#}"))
            })?;
        }
        let zero = |field: &str| {
            EngineLoadError::InvalidSpec(format!("models[{index}].{field} must be at least 1"))
        };
        if self.max_model_len == Some(0) {
            return Err(zero("max_model_len"));
        }
        if self.encoder_cache_memory_bytes == Some(0) {
            return Err(zero("encoder_cache_memory_bytes"));
        }
        // The builder names a model in its errors by this key; the alias is what requests use.
        let key = self
            .model_id
            .clone()
            .unwrap_or_else(|| format!("models[{index}]"));
        let mut config = ModelConfig::new(key, self.model);
        config.alias = self.model_id;
        config.chat_template = self.chat_template;
        config.jinja_explicit = self.jinja_explicit;
        config.max_model_len = self.max_model_len;
        config.hf_config_overrides = self.hf_config_overrides;
        config.num_device_layers = self.device_layers;
        config.in_situ_quant = self.isq;
        if let Some(bytes) = self.encoder_cache_memory_bytes {
            config = config.with_encoder_cache_memory_bytes(bytes);
        }
        Ok(config)
    }
}

/// Where uploaded skills are kept; requests reference them from the shell tool.
#[derive(Debug, Default, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct SkillsSpec {
    /// Kept across loads and shareable between engines, each reading it as it was at load; without a root the engine
    /// keeps its skills in a directory of its own that goes away with it.
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub root: Option<std::path::PathBuf>,
}

impl SkillsSpec {
    fn open(self) -> Result<(SkillStore, Option<tempfile::TempDir>), EngineLoadError> {
        let invalid =
            |error: anyhow::Error| EngineLoadError::InvalidSpec(format!("skills: {error:#}"));
        match self.root {
            Some(root) => Ok((SkillStore::new(root).map_err(invalid)?, None)),
            None => {
                let dir =
                    tempfile::tempdir().map_err(|error| EngineLoadError::Load(error.into()))?;
                let store =
                    SkillStore::new(dir.path().to_path_buf()).map_err(EngineLoadError::Load)?;
                Ok((store, Some(dir)))
            }
        }
    }
}

/// Host functions the agent loop calls: tools by name, and the backend behind `web_search_options`.
#[derive(Default)]
pub struct EngineCallbacks {
    /// Called by their definitions' names.
    pub tools: Vec<ToolCallbackWithTool>,
    pub search: Option<Arc<SearchCallback>>,
}

/// Runtime LoRA adapter management; listing adapters is always allowed.
#[derive(Debug, Default, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct AdapterSpec {
    /// Allow loading and unloading adapters while the engine runs.
    #[serde(default)]
    pub runtime_updates: bool,
    /// The directory adapters must load from; relative adapter paths resolve under it.
    #[serde(default)]
    #[schema(value_type = Option<String>)]
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

#[derive(Debug, Default, Deserialize, utoipa::ToSchema)]
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
    /// Layers per device, as `--device-layers` takes them: `ORD:NUM` entries, or one count for device 0.
    #[serde(default)]
    pub device_layers: Option<Vec<String>>,
    #[serde(default)]
    pub paged_cache: PagedCacheSpec,
    #[serde(default)]
    pub mtp: Option<MtpSpec>,
    /// Most tokens one scheduler step batches across sequences.
    #[serde(default)]
    #[schema(minimum = 1)]
    pub max_num_batched_tokens: Option<usize>,
    /// Longest prompt chunk one prefill step takes.
    #[serde(default)]
    #[schema(minimum = 1)]
    pub max_prefill_chunk_tokens: Option<usize>,
    /// Decode steps the scheduler runs before admitting a waiting prefill.
    #[serde(default)]
    #[schema(minimum = 1)]
    pub max_decode_steps_before_prefill: Option<usize>,
    /// Byte budget for cached multimodal encoder outputs.
    #[serde(default)]
    pub encoder_cache_memory_bytes: Option<usize>,
    /// Merged recursively into the model's `config.json` before it loads.
    #[serde(default)]
    #[schema(value_type = Option<Object>)]
    pub hf_config_overrides: Option<HfConfigOverrides>,
    /// Appends each request and response to this file.
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub log: Option<std::path::PathBuf>,
    /// Periodic throughput logging; on unless set to false.
    #[serde(default)]
    pub throughput_logging: Option<bool>,
    /// Generate to `max_tokens` regardless of end-of-sequence tokens, as a benchmark needs.
    #[serde(default)]
    pub disable_eos_stop: bool,
}

fn nonzero(field: &str, value: usize) -> Result<std::num::NonZeroUsize, EngineLoadError> {
    std::num::NonZeroUsize::new(value)
        .ok_or_else(|| EngineLoadError::InvalidSpec(format!("runtime.{field} must be at least 1")))
}

/// How much the paged-attention KV cache holds; at most one of `context_len`, `memory_mb` and `memory_fraction`.
#[derive(Debug, Default, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct PagedCacheSpec {
    /// Tokens of context to allocate for; without a size the cache takes 90% of free device memory.
    #[serde(default)]
    pub context_len: Option<usize>,
    #[serde(default)]
    pub memory_mb: Option<usize>,
    /// Fraction of device memory, 0 to 1.
    #[serde(default)]
    pub memory_fraction: Option<f32>,
    /// Tokens per block.
    #[serde(default)]
    pub block_size: Option<usize>,
    #[serde(default)]
    pub cache_type: PagedCacheType,
}

/// MTP speculative decoding, drafting with an assistant model or the head built into the checkpoint.
#[derive(Debug, Default, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct MtpSpec {
    /// Assistant model id or path; unset uses the checkpoint's own MTP head.
    #[serde(default)]
    pub model: Option<String>,
    /// Draft tokens proposed per target step.
    #[serde(default)]
    pub n_predict: Option<usize>,
    #[serde(default)]
    pub draft_sampling: MtpDraftSampling,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum MtpDraftSampling {
    /// Probabilistic drafting where the model supports it, else greedy.
    #[default]
    Auto,
    Greedy,
    Probabilistic,
}

impl From<MtpDraftSampling> for MtpDraftSamplingMethod {
    fn from(sampling: MtpDraftSampling) -> Self {
        match sampling {
            MtpDraftSampling::Auto => Self::Auto,
            MtpDraftSampling::Greedy => Self::Greedy,
            MtpDraftSampling::Probabilistic => Self::Probabilistic,
        }
    }
}

impl MtpSpec {
    fn into_config(self) -> MtpConfig {
        match self.model {
            Some(model) => MtpConfig::new(model, self.n_predict),
            None => MtpConfig::builtin(self.n_predict),
        }
        .with_draft_sampling_method(self.draft_sampling.into())
    }
}

#[derive(Debug, Default, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct AgenticSpec {
    #[serde(default)]
    pub max_tool_rounds: Option<usize>,
    #[serde(default)]
    pub tool_dispatch_url: Option<String>,
    #[serde(default)]
    pub agent_permission: Option<AgentPermission>,
    /// Loads an embedding model that reranks web search results.
    #[serde(default)]
    pub search: Option<SearchSpec>,
    /// MCP servers whose tools the model may call.
    #[serde(default)]
    pub mcp: Option<McpClientConfig>,
    /// The Python code execution tool; needs a build with the `code-execution` feature.
    #[serde(default)]
    pub code_execution: Option<CodeExecutionConfig>,
    /// The shell tool, which also runs uploaded skills; needs a build with the `code-execution` feature.
    #[serde(default)]
    pub shell: Option<ShellConfig>,
    /// Sandboxes code execution and the shell unless their config gives its own policy: `auto` and `on` apply
    /// `sandbox_profile` with `sandbox_limits`, `off` runs them unsandboxed.
    #[serde(default)]
    pub sandbox: SandboxMode,
    /// The profile the sandbox starts from; defaults to `developer`.
    #[serde(default)]
    pub sandbox_profile: Option<SandboxProfile>,
    /// Overrides for the profile's limits.
    #[serde(default)]
    pub sandbox_limits: SandboxLimits,
}

/// Limits that replace a sandbox profile's; unset ones keep the profile's.
#[derive(Debug, Default, Clone, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct SandboxLimits {
    /// Per-session memory cap in MiB.
    #[serde(default)]
    pub max_memory_mb: Option<u64>,
    /// Per-session CPU time cap in seconds; raised to the tool's timeout where rlimits apply.
    #[serde(default)]
    pub max_cpu_secs: Option<u64>,
    /// Per-session process and thread cap.
    #[serde(default)]
    pub max_procs: Option<u32>,
    #[serde(default)]
    pub network: Option<NetworkMode>,
}

fn default_policy(
    mode: SandboxMode,
    profile: Option<SandboxProfile>,
    limits: &SandboxLimits,
) -> Option<SandboxPolicy> {
    if mode == SandboxMode::Off {
        return None;
    }
    // not `unwrap_or_default`: the profile type defaults to `restricted`, the spec to `developer`
    let mut policy = profile
        .unwrap_or(SandboxProfile::Developer)
        .default_policy();
    if let Some(max_memory_mb) = limits.max_memory_mb {
        policy.max_memory_mb = max_memory_mb;
    }
    if let Some(max_cpu_secs) = limits.max_cpu_secs {
        policy.max_cpu_secs = max_cpu_secs;
    }
    if let Some(max_procs) = limits.max_procs {
        policy.max_procs = max_procs;
    }
    if let Some(network) = limits.network {
        policy.network = network;
    }
    policy.strict = mode == SandboxMode::On;
    Some(policy)
}

#[derive(Debug, Default, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchSpec {
    #[serde(default)]
    pub embedding_model: SearchEmbeddingModel,
}

/// Why an engine did not load: the spec itself was unusable, or loading the model failed.
#[derive(Debug)]
pub enum EngineLoadError {
    InvalidSpec(String),
    /// A device or feature the spec asks for is not compiled into this build or is not present.
    Unavailable(String),
    Load(anyhow::Error),
}

impl std::fmt::Display for EngineLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSpec(message) => write!(f, "invalid engine spec: {message}"),
            Self::Unavailable(message) => write!(f, "unavailable: {message}"),
            Self::Load(error) => write!(f, "{error:#}"),
        }
    }
}

impl std::error::Error for EngineLoadError {}

/// Resolves a spec's `quant` and GGUF projector; the model keeps the id it was given when resolution changes repository.
async fn resolve_model_source(
    model: ModelSelected,
    model_id: &mut Option<String>,
    isq: &mut Option<String>,
    token_source: &TokenSource,
    force_cpu: bool,
) -> Result<ModelSelected, EngineLoadError> {
    if !model.needs_source_resolution() {
        return Ok(model);
    }
    if model.quant().is_some() && isq.is_some() {
        return Err(EngineLoadError::InvalidSpec(QUANT_WITH_ISQ.to_string()));
    }
    let resolved =
        quant::resolve_model_source(model, token_source, force_cpu, quant::QuantPolicy::Weights)
            .await
            .map_err(EngineLoadError::Load)?;
    if resolved.isq.is_some() {
        *isq = resolved.isq;
    }
    if model_id.is_none() {
        *model_id = resolved.requested_model_id;
    }
    Ok(resolved.model)
}

impl EngineSpec {
    async fn resolve_sources(&mut self) -> Result<(), EngineLoadError> {
        let models = self
            .model
            .iter()
            .chain(self.models.iter().map(|spec| &spec.model));
        if !models
            .into_iter()
            .any(ModelSelected::needs_source_resolution)
        {
            return Ok(());
        }
        let token_source = match &self.runtime.token_source {
            Some(source) => source.parse().map_err(EngineLoadError::InvalidSpec)?,
            None => defaults::TOKEN_SOURCE,
        };
        // Listed models inherit `runtime.isq`, which would land on top of the resolved weights.
        let any_quant = self.models.iter().any(|spec| spec.model.quant().is_some());
        if any_quant && self.runtime.isq.is_some() {
            return Err(EngineLoadError::InvalidSpec(QUANT_WITH_ISQ.to_string()));
        }
        let force_cpu = self.runtime.device.as_deref() == Some("cpu");
        if let Some(model) = self.model.take() {
            let (model_id, isq) = (&mut self.model_id, &mut self.runtime.isq);
            self.model =
                Some(resolve_model_source(model, model_id, isq, &token_source, force_cpu).await?);
        }
        for mut spec in std::mem::take(&mut self.models) {
            let (model_id, isq) = (&mut spec.model_id, &mut spec.isq);
            spec.model =
                resolve_model_source(spec.model, model_id, isq, &token_source, force_cpu).await?;
            self.models.push(spec);
        }
        Ok(())
    }

    fn into_builder(self) -> Result<InferenceRsForServerBuilder, EngineLoadError> {
        let invalid = EngineLoadError::InvalidSpec;
        let runtime = self.runtime;
        let mut builder = InferenceRsForServerBuilder::new();
        builder = match (self.model, self.models.is_empty()) {
            (Some(model), true) => {
                if self.default_model_id.is_some() {
                    return Err(invalid(DEFAULT_WITHOUT_MODELS.to_string()));
                }
                builder
                    .with_model(model)
                    .with_model_id_override_optional(self.model_id)
            }
            (None, false) => {
                if self.model_id.is_some() {
                    return Err(invalid(MODEL_ID_WITH_MODELS.to_string()));
                }
                if self.anymoe.is_some() {
                    return Err(invalid(ANYMOE_WITH_MODELS.to_string()));
                }
                for (index, model) in self.models.into_iter().enumerate() {
                    builder = builder.add_model_config(model.into_config(index)?);
                }
                match self.default_model_id {
                    Some(id) => builder.with_default_model_id(id),
                    None => builder,
                }
            }
            _ => return Err(invalid(ONE_MODEL_SOURCE.to_string())),
        };
        let mut builder = builder
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
        if let Some(tokens) = runtime.max_num_batched_tokens {
            builder =
                builder.with_max_num_batched_tokens(nonzero("max_num_batched_tokens", tokens)?);
        }
        if let Some(tokens) = runtime.max_prefill_chunk_tokens {
            builder =
                builder.with_max_prefill_chunk_tokens(nonzero("max_prefill_chunk_tokens", tokens)?);
        }
        if let Some(steps) = runtime.max_decode_steps_before_prefill {
            builder = builder.with_max_decode_steps_before_prefill(nonzero(
                "max_decode_steps_before_prefill",
                steps,
            )?);
        }
        if let Some(bytes) = runtime.encoder_cache_memory_bytes {
            builder = builder.with_encoder_cache_memory_bytes(bytes);
        }
        builder = builder
            .with_hf_config_overrides_optional(runtime.hf_config_overrides)
            .with_log_optional(runtime.log.map(|path| path.to_string_lossy().into_owned()))
            .with_interactive_mode(!runtime.throughput_logging.unwrap_or(true))
            .with_disable_eos_stop(runtime.disable_eos_stop);
        let paged = runtime.paged_cache;
        let sizes = [
            paged.context_len.is_some(),
            paged.memory_mb.is_some(),
            paged.memory_fraction.is_some(),
        ];
        if sizes.into_iter().filter(|set| *set).count() > 1 {
            return Err(invalid(PAGED_CACHE_ONE_SIZE.to_string()));
        }
        if let Some(device_layers) = &runtime.device_layers {
            parse_device_layers(device_layers).map_err(|error| invalid(format!("{error:#}")))?;
        }
        builder = builder
            .with_num_device_layers_optional(runtime.device_layers)
            .with_paged_ctxt_len_optional(paged.context_len)
            .with_paged_attn_gpu_mem_optional(paged.memory_mb)
            .with_paged_attn_gpu_mem_usage_optional(paged.memory_fraction)
            .with_paged_attn_block_size_optional(paged.block_size)
            .with_paged_attn_cache_type(paged.cache_type)
            .with_mtp_config_optional(runtime.mtp.map(MtpSpec::into_config))
            .with_anymoe_optional(self.anymoe);
        let agentic = self.agentic;
        if let Some(search) = agentic.search {
            builder = builder
                .with_enable_search(true)
                .with_search_embedding_model(search.embedding_model);
        }
        if !cfg!(feature = "code-execution")
            && (agentic.code_execution.is_some() || agentic.shell.is_some())
        {
            return Err(EngineLoadError::Unavailable(
                CODE_EXECUTION_UNAVAILABLE.to_string(),
            ));
        }
        let sandbox = default_policy(
            agentic.sandbox.resolve(),
            agentic.sandbox_profile,
            &agentic.sandbox_limits,
        );
        let code_execution = agentic.code_execution.map(|mut config| {
            config.sandbox_policy = config.sandbox_policy.or_else(|| sandbox.clone());
            config
        });
        let shell = agentic.shell.map(|mut config| {
            config.sandbox_policy = config.sandbox_policy.or(sandbox);
            config
        });
        builder = builder
            .with_mcp_config_optional(agentic.mcp)
            .with_code_exec_config_optional(code_execution)
            .with_shell_config_optional(shell);
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
    .map_err(|error| EngineLoadError::Unavailable(format!("device {device}: {error}")))?;
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
    // Removed with the last clone, when the engine owns its skill directory.
    _skill_dir: Option<Arc<tempfile::TempDir>>,
}

impl Engine {
    /// Wraps an engine the caller already built, with its server-level chat policy and adapter management policy.
    pub fn new(chat: ChatEngine, adapters: LoraAdapterApiConfig) -> Self {
        Self {
            chat,
            adapters,
            _skill_dir: None,
        }
    }

    fn with_skill_dir(mut self, dir: Option<tempfile::TempDir>) -> Self {
        self._skill_dir = dir.map(Arc::new);
        self
    }

    pub async fn load(spec: EngineSpec) -> Result<Self, EngineLoadError> {
        Self::load_with_callbacks(spec, EngineCallbacks::default()).await
    }

    pub async fn load_with_callbacks(
        mut spec: EngineSpec,
        callbacks: EngineCallbacks,
    ) -> Result<Self, EngineLoadError> {
        let adapters = std::mem::take(&mut spec.adapters).into_config()?;
        let (skill_store, skill_dir) = std::mem::take(&mut spec.skills).open()?;
        let agentic = AgenticDefaults {
            max_tool_rounds: spec.agentic.max_tool_rounds,
            tool_dispatch_url: spec.agentic.tool_dispatch_url.clone(),
            agent_permission: spec.agentic.agent_permission,
            approval_broker: Default::default(),
        };
        spec.resolve_sources().await?;
        let mut builder = spec.into_builder()?;
        if let Some(search) = callbacks.search {
            builder = builder.with_search_callback(search);
        }
        for tool in callbacks.tools {
            builder = builder.with_tool_callback(tool.tool.function.name.clone(), tool);
        }
        let state = builder.build().await.map_err(EngineLoadError::Load)?;
        Ok(Self::new(
            ChatEngine {
                state,
                agentic,
                skill_store: Some(Arc::new(skill_store)),
            },
            adapters,
        )
        .with_skill_dir(skill_dir))
    }

    pub async fn load_json(
        spec: &[u8],
        callbacks: EngineCallbacks,
    ) -> Result<Self, EngineLoadError> {
        let spec = serde_json::from_slice(spec)
            .map_err(|error| EngineLoadError::InvalidSpec(error.to_string()))?;
        Self::load_with_callbacks(spec, callbacks).await
    }

    pub fn state(&self) -> &SharedInferenceRsState {
        &self.chat.state
    }

    /// Stops the engine threads and waits for them; fails while another clone of this engine is alive.
    pub async fn shutdown(self) -> Result<(), String> {
        let Self { chat, .. } = self;
        chat.state.shutdown().await
    }

    /// The chat policy and skill store an HTTP server over this engine shares.
    pub fn chat_engine(&self) -> &ChatEngine {
        &self.chat
    }

    pub fn adapter_config(&self) -> &LoraAdapterApiConfig {
        &self.adapters
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
        )
        .with_cancellation(prepared.cancellation))
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
        prepare_response(&self.chat, request)
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

    /// Requantizes the loaded model to another ISQ type.
    pub async fn re_isq(&self, request: ReIsqRequest) -> Result<ReIsqResponse, ApiError> {
        operations::re_isq(self.state(), request).await
    }

    /// Starts, reports or applies online calibration; applying requantizes from the collected statistics.
    pub async fn calibration(
        &self,
        action: CalibrationAction,
    ) -> Result<CalibrationStatus, ApiError> {
        operations::calibration(self.state(), action).await
    }

    pub fn sessions(&self) -> Result<SessionList, ApiError> {
        operations::list_sessions(self.state())
    }

    pub fn session(&self, session_id: &str) -> Result<SerializedSession, ApiError> {
        operations::export_session(self.state(), session_id)
    }

    pub fn put_session(
        &self,
        session_id: &str,
        session: SerializedSession,
    ) -> Result<SessionStored, ApiError> {
        operations::import_session(self.state(), session_id.to_string(), session)?;
        Ok(SessionStored {
            id: session_id.to_string(),
        })
    }

    pub fn delete_session(&self, session_id: &str) -> Result<SessionDeleted, ApiError> {
        operations::delete_session(self.state(), session_id)
    }

    pub async fn tokenize(&self, request: TokenizeRequest) -> Result<TokenizeResponse, ApiError> {
        operations::tokenize(self.state(), request).await
    }

    pub async fn detokenize(
        &self,
        request: DetokenizeRequest,
    ) -> Result<DetokenizeResponse, ApiError> {
        operations::detokenize(self.state(), request).await
    }

    pub async fn re_isq_json(&self, request: &[u8]) -> Result<String, ApiError> {
        to_json(&self.re_isq(parse_json(request)?).await?)
    }

    pub async fn calibration_start_json(&self) -> Result<String, ApiError> {
        to_json(&self.calibration(CalibrationAction::Start).await?)
    }

    pub async fn calibration_status_json(&self) -> Result<String, ApiError> {
        to_json(&self.calibration(CalibrationAction::Status).await?)
    }

    pub async fn calibration_apply_json(&self, request: &[u8]) -> Result<String, ApiError> {
        let request: CalibrationApplyRequest = parse_json(request)?;
        let action = CalibrationAction::Apply {
            save_cimatrix: request.save_cimatrix.map(Into::into),
        };
        to_json(&self.calibration(action).await?)
    }

    pub fn sessions_json(&self) -> Result<String, ApiError> {
        to_json(&self.sessions()?)
    }

    pub fn session_json(&self, session_id: &str) -> Result<String, ApiError> {
        to_json(&self.session(session_id)?)
    }

    pub fn put_session_json(&self, session_id: &str, session: &[u8]) -> Result<String, ApiError> {
        to_json(&self.put_session(session_id, parse_json(session)?)?)
    }

    pub fn delete_session_json(&self, session_id: &str) -> Result<String, ApiError> {
        to_json(&self.delete_session(session_id)?)
    }

    pub async fn tokenize_json(&self, request: &[u8]) -> Result<String, ApiError> {
        to_json(&self.tokenize(parse_json(request)?).await?)
    }

    pub async fn detokenize_json(&self, request: &[u8]) -> Result<String, ApiError> {
        to_json(&self.detokenize(parse_json(request)?).await?)
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

    fn skill_store(&self) -> Result<&SkillStore, ApiError> {
        self.chat.skill_store.as_deref().ok_or_else(|| {
            ApiError::new(
                ApiErrorKind::Unavailable,
                "this engine has no skill store",
                Some("skills_unavailable"),
                None,
            )
        })
    }

    pub fn skills_json(&self) -> Result<String, ApiError> {
        let data = self.skill_store()?.list().map_err(skill_api_error)?;
        to_json(&SkillListObject {
            object: "list",
            data,
        })
    }

    pub fn skill_versions_json(&self, skill_id: &str) -> Result<String, ApiError> {
        let data = self
            .skill_store()?
            .list_versions(skill_id)
            .map_err(skill_api_error)?;
        to_json(&AnthropicSkillVersionListObject {
            data: data.iter().map(AnthropicSkillVersionObject::from).collect(),
            has_more: false,
            next_page: None,
        })
    }

    /// Stores a new skill from its files (a `SKILL.md` and whatever it references).
    pub fn upload_skill_json(&self, files: SkillFiles) -> Result<String, ApiError> {
        to_json(
            &self
                .skill_store()?
                .create_skill(files)
                .map_err(skill_api_error)?,
        )
    }

    pub fn upload_skill_version_json(
        &self,
        skill_id: &str,
        files: SkillFiles,
    ) -> Result<String, ApiError> {
        to_json(
            &self
                .skill_store()?
                .create_version(skill_id, files)
                .map_err(skill_api_error)?,
        )
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

fn parse_json<T: JsonRequest>(request: &[u8]) -> Result<T, ApiError> {
    T::from_json(request)
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
            Self::BlockDenoisingProgress(_) => "block_denoising_progress",
            Self::Error(_) => "error",
        }
    }

    /// `{"event": <name>, "data": <payload>}`; an error's payload is the OpenAI error envelope.
    pub fn to_json(&self) -> String {
        let data = match self {
            Self::Chunk(chunk) => serde_json::to_value(chunk),
            Self::AgenticToolCallProgress(progress) => Ok(progress.to_json()),
            Self::AgenticToolApprovalRequired(approval) => Ok(approval.to_json()),
            Self::FileProduced(file) => serde_json::to_value(file),
            Self::BlockDenoisingProgress(progress) => Ok(serde_json::json!({
                "index": progress.index,
                "step": progress.step,
                "total_steps": progress.total_steps,
                "text": progress.text,
                "finished": progress.finished,
                "final_block": progress.final_block,
            })),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_serve_option_the_spec_carries_parses() {
        let spec: EngineSpec = serde_json::from_value(serde_json::json!({
            "model": {"Plain": {"model_id": "org/model"}},
            "runtime": {
                "device": "cpu",
                "device_layers": ["0:8", "1:8"],
                "paged_cache": {"context_len": 4096, "block_size": 32, "cache_type": "f8e4m3"},
                "mtp": {"n_predict": 2, "draft_sampling": "greedy"},
            },
            "agentic": {
                "search": {},
                "mcp": {"servers": [{"name": "fs", "source": {"type": "Process", "command": "mcp-fs", "args": []}}]},
                "code_execution": {"timeout_secs": 5, "sandbox_policy": {"network": "none"}},
                "shell": {"permission": "ask"},
            },
            "anymoe": {
                "config": {"hidden_size": 8, "expert_type": {"lora_adapter": {"rank": 4, "alpha": 8.0, "target_modules": ["up_proj"]}}},
                "path": "train.json",
                "prefix": "model.layers",
                "mlp": "mlp",
                "model_ids": ["org/expert"],
            },
        }))
        .unwrap();
        let mtp = spec.runtime.mtp.as_ref().unwrap();
        assert!(mtp.model.is_none());
        assert!(matches!(mtp.draft_sampling, MtpDraftSampling::Greedy));
        assert_eq!(spec.runtime.paged_cache.cache_type, PagedCacheType::F8E4M3);
        let policy = spec
            .agentic
            .code_execution
            .as_ref()
            .unwrap()
            .sandbox_policy
            .as_ref()
            .unwrap();
        assert_eq!(
            policy.max_procs,
            inference_core::SandboxPolicy::default().max_procs
        );
        assert!(spec.anymoe.as_ref().unwrap().layers.is_empty());
        let built = spec.into_builder();
        assert_eq!(built.is_ok(), cfg!(feature = "code-execution"));
    }

    #[test]
    fn tools_without_a_policy_are_sandboxed_unless_the_mode_is_off() {
        let none = SandboxLimits::default();
        let developer = SandboxProfile::Developer.default_policy();
        let auto = default_policy(SandboxMode::Auto, None, &none).unwrap();
        assert_eq!(auto.max_procs, developer.max_procs);
        assert!(!auto.strict);
        assert!(default_policy(SandboxMode::On, None, &none).unwrap().strict);
        assert!(default_policy(SandboxMode::Off, None, &none).is_none());
    }

    #[test]
    fn a_sandbox_profile_and_limits_shape_the_default_policy() {
        let spec: AgenticSpec = serde_json::from_value(serde_json::json!({
            "sandbox": "on",
            "sandbox_profile": "restricted",
            "sandbox_limits": {"max_memory_mb": 512, "max_procs": 7},
        }))
        .unwrap();
        let restricted = SandboxProfile::Restricted.default_policy();
        let policy =
            default_policy(spec.sandbox, spec.sandbox_profile, &spec.sandbox_limits).unwrap();
        assert_eq!((policy.max_memory_mb, policy.max_procs), (512, 7));
        assert_eq!(policy.max_cpu_secs, restricted.max_cpu_secs);
        assert_eq!(policy.network, NetworkMode::Loopback);
        let limits = SandboxLimits {
            network: Some(NetworkMode::Full),
            ..SandboxLimits::default()
        };
        let policy = default_policy(SandboxMode::Auto, Some(SandboxProfile::Restricted), &limits);
        assert_eq!(policy.unwrap().network, NetworkMode::Full);
    }

    #[test]
    fn an_engine_serves_one_model_or_a_list_of_them() {
        let plain = serde_json::json!({"Plain": {"model_id": "org/model"}});
        let spec = |value: serde_json::Value| serde_json::from_value::<EngineSpec>(value).unwrap();
        let refused = |value: serde_json::Value| {
            spec(value)
                .into_builder()
                .err()
                .map(|error| error.to_string())
                .unwrap_or_default()
        };
        assert!(refused(serde_json::json!({})).contains("either `model`"));
        assert!(
            refused(serde_json::json!({"model": plain, "models": [{"model": plain}]}))
                .contains("either `model`")
        );
        assert!(
            refused(serde_json::json!({"models": [{"model": plain}], "model_id": "a"}))
                .contains("give each its own")
        );
        assert!(
            refused(serde_json::json!({"models": [{"model": plain, "device_layers": ["x"]}]}))
                .contains("models[0]")
        );
        assert!(
            refused(serde_json::json!({"model": plain, "default_model_id": "a"}))
                .contains("picks one of")
        );
        assert!(
            refused(serde_json::json!({"models": [{"model": plain, "max_model_len": 0}]}))
                .contains("models[0].max_model_len")
        );
        let anymoe = serde_json::json!({"config": {"hidden_size": 8, "expert_type": "fine_tuned"},
            "path": "p", "prefix": "model.layers", "mlp": "mlp", "model_ids": ["e"]});
        assert!(
            refused(serde_json::json!({"models": [{"model": plain}], "anymoe": anymoe}))
                .contains("single `model`")
        );
        let two = spec(serde_json::json!({
            "models": [{"model": plain, "model_id": "a", "isq": "q4k"}, {"model": plain, "model_id": "b"}],
            "default_model_id": "b",
            "runtime": {"device": "cpu"},
        }));
        assert_eq!(two.models.len(), 2);
        assert!(two.into_builder().is_ok());
    }

    #[tokio::test]
    async fn a_quant_is_resolved_before_the_builder_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model-Q4_K_M.gguf"), []).unwrap();
        let run =
            serde_json::json!({"Run": {"model_id": dir.path().to_string_lossy(), "quant": "4"}});
        let spec = |value: serde_json::Value| serde_json::from_value::<EngineSpec>(value).unwrap();

        let mut single = spec(serde_json::json!({"model": run, "runtime": {"device": "cpu"}}));
        single.resolve_sources().await.unwrap();
        assert!(matches!(single.model, Some(ModelSelected::GGUF { .. })));
        let mut listed = spec(serde_json::json!({"models": [{"model": run, "model_id": "a"}]}));
        listed.resolve_sources().await.unwrap();
        assert!(matches!(listed.models[0].model, ModelSelected::GGUF { .. }));
        assert_eq!(listed.models[0].model_id.as_deref(), Some("a"));

        let gguf = serde_json::json!({"GGUF": {"quantized_model_id": dir.path().to_string_lossy(), "quant": "4"}});
        let mut explicit = spec(serde_json::json!({"model": gguf, "runtime": {"isq": "q8_0"}}));
        let error = explicit.resolve_sources().await.unwrap_err().to_string();
        assert!(error.contains("drop `isq`"), "{error}");
        let mut explicit = spec(serde_json::json!({"model": gguf}));
        explicit.resolve_sources().await.unwrap();
        assert!(matches!(
            explicit.model,
            Some(ModelSelected::GGUF { ref quantized_filename, quant: None, .. }) if quantized_filename == "model-Q4_K_M.gguf"
        ));

        for mut refused in [
            spec(serde_json::json!({"model": run, "runtime": {"isq": "q8_0"}})),
            spec(serde_json::json!({"models": [{"model": run, "isq": "q8_0"}]})),
            spec(serde_json::json!({"models": [{"model": run}], "runtime": {"isq": "q8_0"}})),
        ] {
            let error = refused.resolve_sources().await.unwrap_err().to_string();
            assert!(error.contains("drop `isq`"), "{error}");
        }
    }

    #[test]
    fn an_unknown_option_is_refused() {
        let spec = serde_json::json!({
            "model": {"Plain": {"model_id": "org/model"}},
            "runtime": {"paged_cache": {"no_such_option": 32}},
        });
        assert!(serde_json::from_value::<EngineSpec>(spec).is_err());
    }
}

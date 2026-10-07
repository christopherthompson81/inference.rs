use futures::future::BoxFuture;

use crate::*;

/// Configuration for creating an engine instance
#[derive(Clone)]
pub struct EngineConfig {
    pub no_kv_cache: bool,
    pub no_prefix_cache: bool,
    pub prefix_cache_n: usize,
    pub disable_eos_stop: bool,
    pub throughput_logging_enabled: bool,
    pub search_embedding_model: Option<SearchEmbeddingModel>,
    pub search_callback: Option<Arc<SearchCallback>>,
    pub tool_callbacks: tools::ToolCallbacksWithTools,
    /// Runs agentic requests (tools, web search); without one they are rejected.
    pub agent_runner: Option<Arc<dyn AgentRunner>>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            no_kv_cache: false,
            no_prefix_cache: false,
            prefix_cache_n: 16,
            disable_eos_stop: false,
            throughput_logging_enabled: true,
            search_embedding_model: None,
            search_callback: None,
            tool_callbacks: HashMap::new(),
            agent_runner: None,
        }
    }
}

/// Configuration for adding a model to InferenceRs
#[derive(Clone)]
pub struct AddModelConfig {
    pub engine_config: EngineConfig,
    pub mcp_client_config: Option<McpClientConfig>,
    /// Optional loader config for enabling model unload/reload support.
    /// Without this, models cannot be unloaded and reloaded.
    pub loader_config: Option<ModelLoaderConfig>,
    pub code_exec_config: Option<CodeExecutionConfig>,
    pub shell_config: Option<ShellConfig>,
}

impl AddModelConfig {
    pub fn new(engine_config: EngineConfig) -> Self {
        Self {
            engine_config,
            mcp_client_config: None,
            loader_config: None,
            code_exec_config: None,
            shell_config: None,
        }
    }

    pub fn with_mcp_config(mut self, mcp_config: McpClientConfig) -> Self {
        self.mcp_client_config = Some(mcp_config);
        self
    }

    pub fn with_code_execution(mut self, config: CodeExecutionConfig) -> Self {
        self.code_exec_config = Some(config);
        self
    }

    pub fn with_shell_execution(mut self, config: ShellConfig) -> Self {
        self.shell_config = Some(config);
        self
    }

    /// Set the loader config for enabling model unload/reload support.
    /// Without this, models cannot be unloaded and reloaded.
    pub fn with_loader_config(mut self, loader_config: ModelLoaderConfig) -> Self {
        self.loader_config = Some(loader_config);
        self
    }
}

#[derive(Clone)]
pub struct InferenceRsConfig {
    pub kind: ModelKind,
    pub device: Device,
    pub category: ModelCategory,
    pub modalities: Modalities,
    pub max_seq_len: Option<usize>,
    pub generation_defaults: Option<ModelGenerationDefaults>,
}

/// What a model was loaded from, able to build its loader again; `inference_selection::ModelSelected` is the usual one.
pub trait LoaderSource: Send + Sync {
    fn build_loader(
        &self,
        config: &ModelLoaderConfig,
        no_kv_cache: bool,
    ) -> anyhow::Result<Box<dyn Loader>>;
}

/// Configuration for recreating a model loader when reloading an unloaded model.
/// This captures the essential parameters needed to reconstruct a loader.
#[derive(Clone)]
pub struct ModelLoaderConfig {
    /// Builds the loader, from these options and its own model selection.
    pub source: Arc<dyn LoaderSource>,
    /// Source of the HF token
    pub token_source: TokenSource,
    /// Optional HF revision
    pub hf_revision: Option<String>,
    /// Model data type
    pub dtype: ModelDType,
    /// Device to load the model on
    pub device: Device,
    /// Device mapping setting
    pub device_map_setting: DeviceMapSetting,
    /// In-situ quantization type
    pub isq: Option<IsqType>,
    /// Paged attention configuration
    pub paged_attn_config: Option<PagedAttentionConfig>,
    /// Whether to suppress logging during loading
    pub silent: bool,
    /// Chat template override
    pub chat_template: Option<String>,
    /// Explicit Jinja template path
    pub jinja_explicit: Option<String>,
    /// Optional runtime context cap applied by loaders that support it.
    pub max_model_len: Option<usize>,
    /// Optional recursively merged Hugging Face config.json overrides.
    pub hf_config_overrides: Option<HfConfigOverrides>,
    /// Optional speculative decoding attachment to recreate after reload.
    pub mtp_config: Option<MtpConfig>,
    /// Optional logical tensor byte budget for multimodal encoder outputs.
    pub encoder_cache_memory_bytes: Option<usize>,
    /// Values given inline (not by path), which a selection's paths cannot carry.
    pub overrides: LoadOverrides,
}

/// Inline load options that take precedence over their path-based `ModelSelected` counterparts.
#[derive(Clone, Default)]
pub struct LoadOverrides {
    /// Used instead of loading the selection's topology path.
    pub topology: Option<Topology>,
    /// Generation config for speech models.
    pub speech_cfg: Option<SpeechGenerationConfig>,
    /// Wraps the loaded model in an AnyMoE pipeline.
    pub anymoe: Option<AnyMoeSpec>,
}

/// The AnyMoE layer to build on top of the loaded model.
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnyMoeSpec {
    pub config: AnyMoeConfig,
    /// Training data (or gating weights) path.
    pub path: String,
    /// Prefix of the model's layer modules, e.g. `model.layers`.
    pub prefix: String,
    /// Name of the MLP module within each layer, e.g. `mlp`.
    pub mlp: String,
    pub model_ids: Vec<String>,
    /// Layers to apply AnyMoE to; empty applies it to all of them.
    #[serde(default)]
    pub layers: Vec<usize>,
}

impl ModelLoaderConfig {
    /// The loader this config describes. `no_kv_cache` is an engine setting, so it is passed in.
    pub fn build_loader(&self, no_kv_cache: bool) -> anyhow::Result<Box<dyn Loader>> {
        let loader = self.source.build_loader(self, no_kv_cache)?;
        Ok(match self.overrides.anymoe.clone() {
            Some(spec) => Box::new(AnyMoeLoader {
                target: loader,
                config: spec.config,
                path: spec.path,
                prefix: spec.prefix,
                mlp: spec.mlp,
                model_ids: spec.model_ids,
                layers: spec.layers,
            }),
            None => loader,
        })
    }

    /// Load `loader` with this config, attaching MTP speculative decoding when configured.
    pub fn load<'a>(
        &'a self,
        loader: &'a dyn Loader,
        mtp_runtime: MtpRuntimeConfig,
    ) -> BoxFuture<'a, anyhow::Result<Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>>> {
        Box::pin(self.load_inner(loader, mtp_runtime))
    }

    async fn load_inner(
        &self,
        loader: &dyn Loader,
        mtp_runtime: MtpRuntimeConfig,
    ) -> anyhow::Result<Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>> {
        let pipeline = loader.load_model_from_hf(
            self.hf_revision.clone(),
            self.token_source.clone(),
            &self.dtype,
            &self.device,
            self.silent,
            self.device_map_setting.clone(),
            self.isq,
            self.paged_attn_config,
        )?;
        if let Some(mtp_config) = self.mtp_config.clone() {
            pipeline
                .lock()
                .await
                .attach_speculative_with_runtime(
                    SpeculativeConfig::Mtp(mtp_config.with_draft_lm_head_isq(self.isq)),
                    mtp_runtime,
                )
                .map_err(|e| anyhow::anyhow!("Failed to attach MTP speculative decoding: {e}"))?;
        }
        Ok(pipeline)
    }
}

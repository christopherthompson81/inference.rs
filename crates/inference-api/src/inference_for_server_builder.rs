//! ## inference.rs instance for server builder.

use std::{num::NonZeroUsize, sync::Arc};

use anyhow::{Context, Result};
use inference_core::{
    AutoDeviceMapParams, DeviceLayerMapMetadata, DeviceMapMetadata, DeviceMapSetting,
    HfConfigOverrides, InferenceRs, InferenceRsBuilder, Loader, McpClientConfig, MemoryGpuConfig,
    ModelLoaderConfig, MtpConfig, MtpRuntimeConfig, PagedAttentionConfig, PagedCacheType,
    SchedulerConfig, SchedulerLimits, SearchCallback, SearchEmbeddingModel, TokenSource,
    ToolCallbackWithTool, paged_attn_supported, parse_isq_value,
    reserve_external_mtp_memory_with_runtime,
};
use inference_selection::{
    ModelSelected, PagedKvModelRequest, get_auto_device_map_params, get_model_dtype,
    get_tgt_non_granular_index, plan_paged_kv,
};
use inference_tensor::Device;
use tracing::{debug, info, warn};

use crate::types::{LoadedPipeline, SharedInferenceRsState};
use std::collections::{HashMap, HashSet};

/// Configuration for a single model in a multi-model setup
#[derive(Clone, serde::Deserialize)]
pub struct ModelConfig {
    /// Configuration key for this model (human-friendly label)
    pub model_id: String,
    /// Optional alias used as the API model ID
    pub alias: Option<String>,
    /// Model selector
    pub model: ModelSelected,
    /// Model-specific chat template
    pub chat_template: Option<String>,
    /// Model-specific JINJA template
    pub jinja_explicit: Option<String>,
    #[serde(default)]
    pub max_model_len: Option<usize>,
    #[serde(default)]
    pub hf_config_overrides: Option<HfConfigOverrides>,
    /// Model-specific device layers
    pub num_device_layers: Option<Vec<String>>,
    /// Model-specific in-situ quantization
    pub in_situ_quant: Option<String>,
    #[serde(default)]
    pub encoder_cache_memory_bytes: Option<NonZeroUsize>,
    /// Hub revision (branch, tag or commit) to load; the default branch when unset.
    #[serde(default)]
    pub hf_revision: Option<String>,
}

impl ModelConfig {
    pub fn new(model_id: String, model: ModelSelected) -> Self {
        Self {
            model_id,
            alias: None,
            model,
            chat_template: None,
            jinja_explicit: None,
            max_model_len: None,
            hf_config_overrides: None,
            num_device_layers: None,
            in_situ_quant: None,
            encoder_cache_memory_bytes: None,
            hf_revision: None,
        }
    }

    pub fn with_encoder_cache_memory_bytes(mut self, max_bytes: usize) -> Self {
        self.encoder_cache_memory_bytes = Some(
            NonZeroUsize::new(max_bytes).expect("encoder cache memory capacity must be nonzero"),
        );
        self
    }
}

pub mod defaults {
    use super::SearchEmbeddingModel;
    // Provides the default values used for the inference.rs instance for server.
    // These defaults can be used for CLI argument fallbacks, config loading, or general initialization.

    use std::sync::Arc;

    use inference_core::{
        DEFAULT_MAX_DECODE_STEPS_BEFORE_PREFILL, DEFAULT_MAX_NUM_BATCHED_TOKENS,
        DEFAULT_MAX_PREFILL_CHUNK_TOKENS, PagedCacheType,
    };

    pub const DEVICE: Option<inference_tensor::Device> = None;
    pub const SEED: Option<u64> = None;
    pub const LOG: Option<String> = None;
    pub const MODEL: Option<inference_selection::ModelSelected> = None;
    pub const MAX_SEQS: usize = 16;
    pub const MAX_NUM_BATCHED_TOKENS: usize = DEFAULT_MAX_NUM_BATCHED_TOKENS;
    pub const MAX_PREFILL_CHUNK_TOKENS: usize = DEFAULT_MAX_PREFILL_CHUNK_TOKENS;
    pub const MAX_DECODE_STEPS_BEFORE_PREFILL: usize = DEFAULT_MAX_DECODE_STEPS_BEFORE_PREFILL;
    pub const NO_KV_CACHE: bool = false;
    pub const CHAT_TEMPLATE: Option<String> = None;
    pub const JINJA_EXPLICIT: Option<String> = None;
    pub const INTERACTIVE_MODE: bool = false;
    pub const PREFIX_CACHE_N: usize = 16;
    pub const NUM_DEVICE_LAYERS: Option<Vec<String>> = None;
    pub const IN_SITU_QUANT: Option<String> = None;
    pub const PAGED_ATTN_GPU_MEM: Option<usize> = None;
    pub const PAGED_ATTN_GPU_MEM_USAGE: Option<f32> = None;
    pub const PAGED_CTXT_LEN: Option<usize> = None;
    pub const PAGED_ATTN_BLOCK_SIZE: Option<usize> = None;
    pub const PAGED_ATTN: Option<bool> = None;
    pub const PAGED_ATTN_CPU: bool = false;
    pub const PAGED_ATTN_CUDA: bool = true;
    pub const PAGED_ATTN_METAL: bool = false;
    pub const CPU: bool = false;
    pub const ENABLE_SEARCH: bool = false;
    pub const SEARCH_EMBEDDING_MODEL: Option<SearchEmbeddingModel> = None;
    pub const TOKEN_SOURCE: inference_core::TokenSource = inference_core::TokenSource::CacheToken;
    pub const SEARCH_CALLBACK: Option<Arc<inference_core::SearchCallback>> = None;
    pub const PAGED_CACHE_TYPE: PagedCacheType = PagedCacheType::Auto;
    pub const MTP_CONFIG: Option<inference_core::MtpConfig> = None;
}

/// A builder for creating a inference.rs instance with configured options for the inference.rs server.
///
/// ### Examples
///
/// Basic usage:
/// ```ignore
/// use inference_api::inference_for_server_builder::InferenceRsForServerBuilder;
///
/// let args = Args::parse();
///
/// let inference = InferenceRsForServerBuilder::new()
///        .with_model(args.model)
///        .with_max_seqs(args.max_seqs)
///        .with_no_kv_cache(args.no_kv_cache)
///        .with_token_source(args.token_source)
///        .with_interactive_mode(args.interactive_mode)
///        .with_prefix_cache_n(args.prefix_cache_n)
///        .with_paged_attn(args.paged_attn)
///        .with_cpu(args.cpu)
///        .with_enable_search(args.enable_search)
///        .with_seed_optional(args.seed)
///        .with_log_optional(args.log)
///        .with_chat_template_optional(args.chat_template)
///        .with_jinja_explicit_optional(args.jinja_explicit)
///        .with_num_device_layers_optional(args.num_device_layers)
///        .with_in_situ_quant_optional(args.in_situ_quant)
///        .with_paged_attn_gpu_mem_optional(args.paged_attn_gpu_mem)
///        .with_paged_attn_gpu_mem_usage_optional(args.paged_attn_gpu_mem_usage)
///        .with_paged_ctxt_len_optional(args.paged_ctxt_len)
///        .with_paged_attn_block_size_optional(args.paged_attn_block_size)
///        .build()
///        .await?;
/// ```
pub struct InferenceRsForServerBuilder {
    /// The Candle device to use for model execution (CPU, CUDA, Metal, etc.).
    device: Option<Device>,

    /// Integer seed to ensure reproducible random number generation.
    seed: Option<u64>,

    /// Log all responses and requests to this file
    log: Option<String>,

    /// Model selector (for single-model mode, deprecated in favor of models)
    model: Option<ModelSelected>,

    /// Optional API id override for single-model mode.
    model_id_override: Option<String>,

    /// Multiple model configurations (for multi-model mode)
    models: Vec<ModelConfig>,

    /// Default model ID to use when none is specified in requests
    default_model_id: Option<String>,

    /// Maximum running sequences at any time. If the `tgt_non_granular_index` flag is set for X-LoRA models, this will be set to 1.
    max_seqs: usize,

    /// Maximum tokens processed by one paged-attention scheduler step.
    max_num_batched_tokens: NonZeroUsize,

    /// Maximum chunkable CUDA text-prompt tokens in one paged-attention scheduler step.
    max_prefill_chunk_tokens: NonZeroUsize,

    /// Maximum decode steps before a waiting prefill batch is admitted.
    max_decode_steps_before_prefill: NonZeroUsize,

    /// Use no KV cache.
    no_kv_cache: bool,

    /// Chat template file with a JINJA file with `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs.
    /// Used if the automatic deserialization fails. If this ends with `.json` (ie., it is a file) then that template is loaded.
    chat_template: Option<String>,

    /// Explicit JINJA chat template file (.jinja) to be used. If specified, this overrides all other chat templates.
    jinja_explicit: Option<String>,

    /// Optional runtime context length applied by the selected model loader.
    max_model_len: Option<usize>,

    /// Optional recursively merged Hugging Face config.json overrides.
    hf_config_overrides: Option<HfConfigOverrides>,

    /// Source of the token for authentication.
    /// Can be in the formats: `literal:<value>`, `env:<value>`, `path:<value>`, `cache` to use a cached token, or `none` to use no token.
    /// Defaults to `cache`.
    token_source: TokenSource,

    /// Enter interactive mode instead of serving a chat server.
    interactive_mode: bool,

    /// Number of prefix caches to hold on the device. Other caches are evicted to the CPU based on a LRU strategy.
    prefix_cache_n: usize,

    /// NOTE: This can be omitted to use automatic device mapping!
    /// Number of device layers to load and run on GPU(s). All others will be on the CPU.
    /// If one GPU is used, then this value should be an integer. Otherwise, it follows the following pattern:
    /// ORD:NUM;... Where ORD is a unique device ordinal and NUM is the number of layers for that device.
    num_device_layers: Option<Vec<String>>,

    /// In-situ quantization to apply.
    in_situ_quant: Option<String>,

    /// Hub revision of the single model; listed models carry their own.
    hf_revision: Option<String>,

    /// GPU memory to allocate for KV cache with PagedAttention in MBs.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    paged_attn_gpu_mem: Option<usize>,

    /// Percentage of GPU memory to utilize after allocation of KV cache with PagedAttention, from 0 to 1.
    /// If this is not set and the device is CUDA, it will default to `0.9`.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    paged_attn_gpu_mem_usage: Option<f32>,

    /// Total context length to allocate the KV cache for (total number of tokens which the KV cache can hold).
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    /// This is the default setting, and it defaults to the `max-seq-len` specified in after the model type.
    paged_ctxt_len: Option<usize>,

    /// Block size (number of tokens per block) for PagedAttention. If this is not set and the device is CUDA, it will default to 32.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    paged_attn_block_size: Option<usize>,

    /// Enables or disables PagedAttention. By default, PagedAttention is enabled on CUDA and disabled on Metal (and not supported on CPU). Use this to override the default behavior.
    paged_attn: Option<bool>,

    /// Use CPU only
    cpu: bool,

    /// Enable searching compatible with the OpenAI `web_search_options` setting. This loads the selected search embedding reranker (EmbeddingGemma by default).
    enable_search: bool,

    /// Specify which built-in search embedding model to load.
    search_embedding_model: Option<SearchEmbeddingModel>,

    /// Optional override search callback
    search_callback: Option<Arc<SearchCallback>>,

    /// Host tools the agent loop calls by name
    tool_callbacks: HashMap<String, ToolCallbackWithTool>,

    /// Optional MCP client configuration
    mcp_client_config: Option<McpClientConfig>,

    /// PagedAttention KV cache type
    paged_cache_type: PagedCacheType,

    /// Optional MTP assistant configuration.
    mtp_config: Option<MtpConfig>,
    encoder_cache_memory_bytes: Option<usize>,

    /// Disable EOS token stopping (generate until max_len regardless of EOS)
    disable_eos_stop: bool,

    /// Python code execution configuration
    code_exec_config: Option<inference_core::CodeExecutionConfig>,
    /// Shell execution configuration
    shell_config: Option<inference_core::ShellConfig>,
    /// AnyMoE layer built over the single model
    anymoe: Option<inference_core::AnyMoeSpec>,
}

impl Default for InferenceRsForServerBuilder {
    /// Creates a new builder with default configuration.
    fn default() -> Self {
        Self {
            device: defaults::DEVICE,
            seed: defaults::SEED,
            log: defaults::LOG,
            model: defaults::MODEL,
            model_id_override: None,
            models: Vec::new(),
            default_model_id: None,
            max_seqs: defaults::MAX_SEQS,
            max_num_batched_tokens: NonZeroUsize::new(defaults::MAX_NUM_BATCHED_TOKENS).unwrap(),
            max_prefill_chunk_tokens: NonZeroUsize::new(defaults::MAX_PREFILL_CHUNK_TOKENS)
                .unwrap(),
            max_decode_steps_before_prefill: NonZeroUsize::new(
                defaults::MAX_DECODE_STEPS_BEFORE_PREFILL,
            )
            .unwrap(),
            no_kv_cache: defaults::NO_KV_CACHE,
            chat_template: defaults::CHAT_TEMPLATE,
            jinja_explicit: defaults::JINJA_EXPLICIT,
            max_model_len: None,
            hf_config_overrides: None,
            token_source: defaults::TOKEN_SOURCE,
            interactive_mode: defaults::INTERACTIVE_MODE,
            prefix_cache_n: defaults::PREFIX_CACHE_N,
            num_device_layers: defaults::NUM_DEVICE_LAYERS,
            in_situ_quant: defaults::IN_SITU_QUANT,
            hf_revision: None,
            paged_attn_gpu_mem: defaults::PAGED_ATTN_GPU_MEM,
            paged_attn_gpu_mem_usage: defaults::PAGED_ATTN_GPU_MEM_USAGE,
            paged_ctxt_len: defaults::PAGED_CTXT_LEN,
            paged_attn_block_size: defaults::PAGED_ATTN_BLOCK_SIZE,
            paged_attn: defaults::PAGED_ATTN,
            cpu: defaults::CPU,
            enable_search: defaults::ENABLE_SEARCH,
            search_embedding_model: defaults::SEARCH_EMBEDDING_MODEL,
            search_callback: defaults::SEARCH_CALLBACK,
            tool_callbacks: HashMap::new(),
            mcp_client_config: None,
            paged_cache_type: defaults::PAGED_CACHE_TYPE,
            mtp_config: defaults::MTP_CONFIG,
            encoder_cache_memory_bytes: None,
            disable_eos_stop: false,
            code_exec_config: None,
            shell_config: None,
            anymoe: None,
        }
    }
}

impl InferenceRsForServerBuilder {
    /// Creates a new `InferenceRsForServerBuilder` with default settings.
    ///
    /// This is equivalent to calling `Default::default()`.
    ///
    /// ### Examples
    ///
    /// ```ignore
    /// use inference_api::inference_for_server_builder::InferenceRsForServerBuilder;
    ///
    /// let builder = inference_api::inference_for_server_builder::InferenceRsForServerBuilder::new();
    /// ```
    pub fn new() -> Self {
        Default::default()
    }

    /// Sets the Candle device to use for model execution.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Sets the random seed for deterministic model behavior.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the random seed if provided.
    pub fn with_seed_optional(mut self, seed: Option<u64>) -> Self {
        if let Some(seed) = seed {
            self = self.with_seed(seed);
        }
        self
    }

    /// Sets the logging configuration.
    pub fn with_log(mut self, log: String) -> Self {
        self.log = Some(log);
        self
    }

    /// Sets the logging configuration if provided.
    pub fn with_log_optional(mut self, log: Option<String>) -> Self {
        if let Some(log) = log {
            self = self.with_log(log);
        }
        self
    }

    /// Sets the model to be used.
    pub fn with_model(mut self, model: ModelSelected) -> Self {
        self.model = Some(model);
        self
    }

    /// The id requests use for the single model, when the spec gives one.
    pub fn with_model_id_override_optional(mut self, id: Option<String>) -> Self {
        if let Some(id) = id {
            self.model_id_override = Some(id);
        }
        self
    }

    /// Set the default model ID to use when none is specified in requests.
    pub fn with_default_model_id(mut self, default_model_id: String) -> Self {
        self.default_model_id = Some(default_model_id);
        self
    }

    /// Add a model configuration.
    pub fn add_model_config(mut self, config: ModelConfig) -> Self {
        self.models.push(config);
        self
    }

    /// Sets the maximum number of concurrent sequences.
    pub fn with_max_seqs(mut self, max_seqs: usize) -> Self {
        self.max_seqs = max_seqs;
        self
    }

    pub fn with_encoder_cache_memory_bytes(mut self, max_bytes: usize) -> Self {
        assert!(
            max_bytes > 0,
            "encoder cache memory capacity must be nonzero"
        );
        self.encoder_cache_memory_bytes = Some(max_bytes);
        self
    }

    /// Sets the maximum number of tokens processed by one paged-attention scheduler step.
    pub fn with_max_num_batched_tokens(mut self, max_num_batched_tokens: NonZeroUsize) -> Self {
        self.max_num_batched_tokens = max_num_batched_tokens;
        self
    }

    /// Sets the maximum chunkable CUDA text-prompt tokens in one scheduler step.
    pub fn with_max_prefill_chunk_tokens(mut self, max_prefill_chunk_tokens: NonZeroUsize) -> Self {
        self.max_prefill_chunk_tokens = max_prefill_chunk_tokens;
        self
    }

    /// Sets the maximum decode steps before a waiting prefill batch is admitted.
    pub fn with_max_decode_steps_before_prefill(
        mut self,
        max_decode_steps_before_prefill: NonZeroUsize,
    ) -> Self {
        self.max_decode_steps_before_prefill = max_decode_steps_before_prefill;
        self
    }

    /// Sets whether to disable the key-value cache.
    pub fn with_no_kv_cache(mut self, no_kv_cache: bool) -> Self {
        self.no_kv_cache = no_kv_cache;
        self
    }

    /// Sets the chat template configuration.
    pub fn with_chat_template(mut self, chat_template: String) -> Self {
        self.chat_template = Some(chat_template);
        self
    }

    /// Sets the chat template configuration if provided.
    pub fn with_chat_template_optional(mut self, chat_template: Option<String>) -> Self {
        if let Some(chat_template) = chat_template {
            self = self.with_chat_template(chat_template);
        }
        self
    }

    /// Sets an explicit JINJA chat template file.
    pub fn with_jinja_explicit(mut self, jinja_explicit: String) -> Self {
        self.jinja_explicit = Some(jinja_explicit);
        self
    }

    /// Sets an explicit JINJA chat template file if provided.
    pub fn with_jinja_explicit_optional(mut self, jinja_explicit: Option<String>) -> Self {
        if let Some(jinja_explicit) = jinja_explicit {
            self = self.with_jinja_explicit(jinja_explicit);
        }
        self
    }

    /// Sets the runtime model context length.
    pub fn with_max_model_len(mut self, max_model_len: usize) -> Self {
        assert!(max_model_len > 0, "maximum model length must be nonzero");
        self.max_model_len = Some(max_model_len);
        self
    }

    /// Sets the runtime model context length if provided.
    pub fn with_max_model_len_optional(mut self, max_model_len: Option<usize>) -> Self {
        if let Some(max_model_len) = max_model_len {
            self = self.with_max_model_len(max_model_len);
        }
        self
    }

    /// Sets recursively merged Hugging Face config.json overrides.
    pub fn with_hf_config_overrides(mut self, overrides: HfConfigOverrides) -> Self {
        self.hf_config_overrides = Some(overrides);
        self
    }

    /// Sets Hugging Face config.json overrides if provided.
    pub fn with_hf_config_overrides_optional(
        mut self,
        overrides: Option<HfConfigOverrides>,
    ) -> Self {
        if let Some(overrides) = overrides {
            self = self.with_hf_config_overrides(overrides);
        }
        self
    }

    /// Sets the token source for authentication.
    pub fn with_token_source(mut self, token_source: TokenSource) -> Self {
        self.token_source = token_source;
        self
    }

    /// Sets whether to run in interactive mode.
    pub fn with_interactive_mode(mut self, interactive_mode: bool) -> Self {
        self.interactive_mode = interactive_mode;
        self
    }

    /// Sets the number of prefix caches to hold on the device.
    pub fn with_prefix_cache_n(mut self, prefix_cache_n: usize) -> Self {
        self.prefix_cache_n = prefix_cache_n;
        self
    }

    /// Sets the device layer mapping
    pub fn with_num_device_layers(mut self, num_device_layers: Vec<String>) -> Self {
        self.num_device_layers = Some(num_device_layers);
        self
    }

    /// Sets the device layer mapping if provided.
    pub fn with_num_device_layers_optional(
        mut self,
        num_device_layers: Option<Vec<String>>,
    ) -> Self {
        if let Some(num_device_layers) = num_device_layers {
            self = self.with_num_device_layers(num_device_layers);
        }
        self
    }

    /// Sets the in-situ quantization method.
    pub fn with_in_situ_quant(mut self, in_situ_quant: String) -> Self {
        self.in_situ_quant = Some(in_situ_quant);
        self
    }

    /// Sets the in-situ quantization method if provided.
    pub fn with_in_situ_quant_optional(mut self, in_situ_quant: Option<String>) -> Self {
        if let Some(in_situ_quant) = in_situ_quant {
            self = self.with_in_situ_quant(in_situ_quant);
        }
        self
    }

    /// Loads the single model at this hub revision (branch, tag or commit).
    pub fn with_hf_revision_optional(mut self, revision: Option<String>) -> Self {
        self.hf_revision = revision;
        self
    }

    /// Sets PagedAttention.
    ///
    /// Unlike other `with_PROP` or `with_PROP_optional` methods, this method
    /// sets the value to whatever `Option<bool>` is passed in as `None`, `Some(true)`
    /// and `Some(false)` have different implications.
    ///
    /// `None`: default behavior for target device (e.g. enable for CUDA, disable for Metal)
    /// `Some(true)`: enable (if supported by target device)
    /// `Some(false)`: disable
    pub fn set_paged_attn(mut self, paged_attn: Option<bool>) -> Self {
        self.paged_attn = paged_attn;
        self
    }

    /// Sets the GPU memory allocation for PagedAttention KV cache.
    pub fn with_paged_attn_gpu_mem(mut self, paged_attn_gpu_mem: usize) -> Self {
        self.paged_attn_gpu_mem = Some(paged_attn_gpu_mem);
        self
    }

    /// Sets the GPU memory allocation for PagedAttention KV cache if provided.
    pub fn with_paged_attn_gpu_mem_optional(mut self, paged_attn_gpu_mem: Option<usize>) -> Self {
        if let Some(paged_attn_gpu_mem) = paged_attn_gpu_mem {
            self = self.with_paged_attn_gpu_mem(paged_attn_gpu_mem);
        }
        self
    }

    /// Sets the percentage of GPU memory to utilize for PagedAttention.
    pub fn with_paged_attn_gpu_mem_usage(mut self, paged_attn_gpu_mem_usage: f32) -> Self {
        self.paged_attn_gpu_mem_usage = Some(paged_attn_gpu_mem_usage);
        self
    }

    /// Sets the percentage of GPU memory to utilize for PagedAttention if provided.
    pub fn with_paged_attn_gpu_mem_usage_optional(
        mut self,
        paged_attn_gpu_mem_usage: Option<f32>,
    ) -> Self {
        if let Some(paged_attn_gpu_mem_usage) = paged_attn_gpu_mem_usage {
            self = self.with_paged_attn_gpu_mem_usage(paged_attn_gpu_mem_usage);
        }
        self
    }

    /// Sets the total context length for KV cache allocation.
    pub fn with_paged_ctxt_len(mut self, paged_ctxt_len: usize) -> Self {
        self.paged_ctxt_len = Some(paged_ctxt_len);
        self
    }

    /// Sets the total context length for KV cache allocation if provided.
    pub fn with_paged_ctxt_len_optional(mut self, paged_ctxt_len: Option<usize>) -> Self {
        if let Some(paged_ctxt_len) = paged_ctxt_len {
            self = self.with_paged_ctxt_len(paged_ctxt_len);
        }
        self
    }

    /// Sets the block size for PagedAttention.
    pub fn with_paged_attn_block_size(mut self, paged_attn_block_size: usize) -> Self {
        self.paged_attn_block_size = Some(paged_attn_block_size);
        self
    }

    /// Sets the block size for PagedAttention.
    pub fn with_paged_attn_cache_type(mut self, cache_type: PagedCacheType) -> Self {
        self.paged_cache_type = cache_type;
        self
    }

    /// Attach an MTP assistant after the target model loads.
    pub fn with_mtp_config(mut self, config: MtpConfig) -> Self {
        self.mtp_config = Some(config);
        self
    }

    /// Attach an MTP assistant if provided.
    pub fn with_mtp_config_optional(mut self, config: Option<MtpConfig>) -> Self {
        if let Some(config) = config {
            self = self.with_mtp_config(config);
        }
        self
    }

    /// Disable EOS token stopping (generate until max_len regardless of EOS).
    pub fn with_disable_eos_stop(mut self, disable: bool) -> Self {
        self.disable_eos_stop = disable;
        self
    }

    /// Sets the block size for PagedAttention if provided.
    pub fn with_paged_attn_block_size_optional(
        mut self,
        paged_attn_block_size: Option<usize>,
    ) -> Self {
        if let Some(paged_attn_block_size) = paged_attn_block_size {
            self = self.with_paged_attn_block_size(paged_attn_block_size);
        }
        self
    }

    /// Sets whether to force CPU-only execution.
    pub fn with_cpu(mut self, cpu: bool) -> Self {
        self.cpu = cpu;
        self
    }

    /// Sets whether to enable web search functionality.
    pub fn with_enable_search(mut self, enable_search: bool) -> Self {
        self.enable_search = enable_search;
        self
    }

    /// Sets the embedding model used for web search assistance.
    pub fn with_search_embedding_model(
        mut self,
        search_embedding_model: SearchEmbeddingModel,
    ) -> Self {
        self.search_embedding_model = Some(search_embedding_model);
        self
    }

    /// Override the search function used when `web_search_options` is enabled.
    pub fn with_search_callback(mut self, callback: Arc<SearchCallback>) -> Self {
        self.search_callback = Some(callback);
        self
    }

    /// Registers a host tool: requests may call it by name and the agent loop runs `callback`.
    pub fn with_tool_callback(
        mut self,
        name: impl Into<String>,
        callback: ToolCallbackWithTool,
    ) -> Self {
        self.tool_callbacks.insert(name.into(), callback);
        self
    }

    /// Sets the MCP client configuration.
    pub fn with_mcp_config(mut self, mcp_config: McpClientConfig) -> Self {
        self.mcp_client_config = Some(mcp_config);
        self
    }

    /// Sets the MCP client configuration if provided.
    pub fn with_mcp_config_optional(mut self, mcp_config: Option<McpClientConfig>) -> Self {
        if let Some(mcp_config) = mcp_config {
            self = self.with_mcp_config(mcp_config);
        }
        self
    }

    /// Sets the Python code execution configuration if present.
    pub fn with_code_exec_config_optional(
        mut self,
        config: Option<inference_core::CodeExecutionConfig>,
    ) -> Self {
        self.code_exec_config = config;
        self
    }

    pub fn with_shell_config_optional(
        mut self,
        config: Option<inference_core::ShellConfig>,
    ) -> Self {
        self.shell_config = config;
        self
    }

    /// Wraps the single model in an AnyMoE layer.
    pub fn with_anymoe_optional(mut self, anymoe: Option<inference_core::AnyMoeSpec>) -> Self {
        self.anymoe = anymoe;
        self
    }

    /// The settings this builder loads each model with, keeping the device it resolves for the build that follows.
    pub fn model_load_settings(&mut self) -> Result<ModelLoadSettings> {
        let device = match &self.device {
            Some(device) => device.clone(),
            None => {
                let device = init_device(self.cpu, self.seed)?;
                self.device = Some(device.clone());
                device
            }
        };
        let paged_attn = configure_paged_attn(&device, self.paged_attn);
        let requested_cache = init_cache_config(
            self.paged_attn_block_size,
            self.paged_attn_gpu_mem,
            self.paged_attn_gpu_mem_usage,
            self.paged_ctxt_len,
            self.paged_cache_type,
            !paged_attn,
        )?
        .map(|config| config.with_serving_capacity(self.max_seqs))
        .transpose()?
        .map(|config| config.with_recurrent_prefix_capacity(self.prefix_cache_n));
        // A non-granular X-LoRA first model runs every model one sequence at a time.
        let first = self
            .models
            .first()
            .map(|config| &config.model)
            .or(self.model.as_ref());
        let max_seqs = match first.and_then(get_tgt_non_granular_index) {
            Some(_) => 1,
            None => self.max_seqs,
        };
        Ok(ModelLoadSettings {
            device,
            token_source: self.token_source.clone(),
            num_device_layers: self.num_device_layers.clone(),
            in_situ_quant: self.in_situ_quant.clone(),
            chat_template: self.chat_template.clone(),
            jinja_explicit: self.jinja_explicit.clone(),
            max_model_len: self.max_model_len,
            hf_config_overrides: self.hf_config_overrides.clone(),
            encoder_cache_memory_bytes: self.encoder_cache_memory_bytes,
            requested_cache,
            max_seqs,
            limits: self.scheduler_limits(),
            no_kv_cache: self.no_kv_cache,
            add_model_config: self.shared_model_config(),
            mtp_runtime: MtpRuntimeConfig::new(self.prefix_cache_n),
        })
    }

    /// What every model this builder loads shares, short of its loader config.
    fn shared_model_config(&self) -> inference_core::AddModelConfig {
        let engine_config = inference_core::EngineConfig {
            no_kv_cache: self.no_kv_cache,
            no_prefix_cache: false,
            prefix_cache_n: self.prefix_cache_n,
            disable_eos_stop: self.disable_eos_stop,
            throughput_logging_enabled: !self.interactive_mode,
            search_embedding_model: get_search_embedding_model(
                self.enable_search,
                self.search_embedding_model,
            ),
            search_callback: self.search_callback.clone(),
            tool_callbacks: self.tool_callbacks.clone(),
            agent_runner: Some(inference_agent::runner()),
        };
        inference_core::AddModelConfig {
            engine_config,
            mcp_client_config: self.mcp_client_config.clone(),
            loader_config: None,
            code_exec_config: self.code_exec_config.clone(),
            shell_config: self.shell_config.clone(),
        }
    }

    fn scheduler_limits(&self) -> SchedulerLimits {
        SchedulerLimits {
            max_num_batched_tokens: self.max_num_batched_tokens.get(),
            max_prefill_chunk_tokens: self.max_prefill_chunk_tokens.get(),
            max_decode_steps_before_prefill: self.max_decode_steps_before_prefill.get(),
        }
    }

    /// Builds the configured inference.rs instance.
    ///
    /// ### Examples
    ///
    /// ```ignore
    /// use inference_api::inference_for_server_builder::InferenceRsForServerBuilder;
    ///
    /// let shared_inference = InferenceRsForServerBuilder::new()
    ///     .with_model(model)
    ///     .with_in_situ_quant("8".to_string())
    ///     .set_paged_attn(Some(true))
    ///     .build()
    ///     .await?;
    /// ```
    pub async fn build(self) -> Result<SharedInferenceRsState> {
        inference_tensor::utils::init_global_threadpool();
        // Determine if we're in single-model or multi-model mode
        if !self.models.is_empty() {
            self.build_multi_model().await
        } else {
            self.build_single_model().await
        }
    }

    /// Build a single-model instance (legacy mode)
    async fn build_single_model(mut self) -> Result<SharedInferenceRsState> {
        let mtp_runtime = MtpRuntimeConfig::new(self.prefix_cache_n);
        let add_model_config = self.shared_model_config();
        let limits = self.scheduler_limits();
        let model = self.model.context("Model was None")?;

        let tgt_non_granular_index = get_tgt_non_granular_index(&model);
        let dtype = get_model_dtype(&model)?;
        let auto_device_map_params = get_auto_device_map_params(&model)?;

        if tgt_non_granular_index.is_some() {
            self.max_seqs = 1;
        }

        let device = if let Some(device) = self.device {
            device
        } else {
            init_device(self.cpu, self.seed)?
        };

        let mapper = init_mapper(&self.num_device_layers, &auto_device_map_params)?;
        let paged_attn = configure_paged_attn(&device, self.paged_attn);

        let cache_config = reserve_external_mtp_memory_with_runtime(
            init_cache_config(
                self.paged_attn_block_size,
                self.paged_attn_gpu_mem,
                self.paged_attn_gpu_mem_usage,
                self.paged_ctxt_len,
                self.paged_cache_type,
                !paged_attn,
            )?
            .map(|config| config.with_serving_capacity(self.max_seqs))
            .transpose()?
            .map(|config| config.with_recurrent_prefix_capacity(self.prefix_cache_n)),
            self.mtp_config.as_ref(),
            mtp_runtime,
            &dtype,
            &device,
        )?;

        let isq = self
            .in_situ_quant
            .as_ref()
            .map(|isq| parse_isq_value(isq, Some(&device)).map_err(|e| anyhow::anyhow!("{e}")))
            .transpose()?;

        let loader_config = ModelLoaderConfig {
            source: Arc::new(model),
            token_source: self.token_source,
            hf_revision: self.hf_revision,
            dtype,
            device: device.clone(),
            device_map_setting: mapper,
            isq,
            paged_attn_config: cache_config,
            silent: false,
            chat_template: self.chat_template,
            jinja_explicit: self.jinja_explicit,
            max_model_len: self.max_model_len,
            hf_config_overrides: self.hf_config_overrides,
            mtp_config: self.mtp_config.clone(),
            encoder_cache_memory_bytes: self.encoder_cache_memory_bytes,
            overrides: inference_core::LoadOverrides {
                anymoe: self.anymoe,
                ..Default::default()
            },
        };
        let loader = loader_config.build_loader(self.no_kv_cache)?;
        inference_instance_info(&*loader);
        let pipeline: LoadedPipeline = loader_config.load(&*loader, mtp_runtime).await?;
        info!("Model loaded.");

        let scheduler_config =
            SchedulerConfig::for_pipeline(&pipeline, cache_config.is_some(), self.max_seqs, limits)
                .await?;
        let mut builder = InferenceRsBuilder::from_config(
            pipeline,
            scheduler_config,
            add_model_config.with_loader_config(loader_config),
        )
        .with_opt_log(self.log);
        if let Some(id) = self.model_id_override {
            builder = builder.with_model_id(id);
        }
        let inference = builder.build().await;

        Ok(inference)
    }

    /// Build a multi-model instance
    pub async fn build_multi_model(mut self) -> Result<SharedInferenceRsState> {
        if self.models.is_empty() {
            anyhow::bail!("No models configured for multi-model mode");
        }

        inference_core::distributed::begin_tensor_parallel_session(self.models.len())?;

        let first = &self.models[0].model;
        let tgt_non_granular_index = get_tgt_non_granular_index(first);
        let first_dtype = get_model_dtype(first)?;
        if tgt_non_granular_index.is_some() {
            self.max_seqs = 1;
        }
        let settings = self.model_load_settings()?;

        let mut paged_kv_plan = plan_paged_kv(
            &self
                .models
                .iter()
                .map(|_| PagedKvModelRequest {
                    paged_attn: settings.requested_cache,
                    max_num_seqs: settings.max_seqs,
                })
                .collect::<Vec<_>>(),
            Default::default(),
        )?;
        if let Some(first) = paged_kv_plan.paged_attn.first_mut() {
            *first = reserve_external_mtp_memory_with_runtime(
                *first,
                self.mtp_config.as_ref(),
                settings.mtp_runtime,
                &first_dtype,
                &settings.device,
            )?;
        }

        let mut loaded_model_ids = Vec::new();
        let mut registered_ids = HashSet::new();
        let mut inference: Option<SharedInferenceRsState> = None;
        for (model_index, model_config) in self.models.iter().enumerate() {
            if model_index > 0 {
                info!(
                    "Loading additional model from config key: {}",
                    model_config.model_id
                );
            }
            // the global MTP setting and its memory reservation belong to the first model only
            let mtp_config = self.mtp_config.clone().filter(|_| model_index == 0);
            let loaded = settings
                .load(
                    model_config,
                    paged_kv_plan.paged_attn[model_index],
                    mtp_config,
                    settings.max_seqs,
                    model_index == 0,
                )
                .await?;
            if !registered_ids.insert(loaded.model_id.clone()) {
                anyhow::bail!(
                    "Model ID conflict: '{}' is already registered (config key: {}).",
                    loaded.model_id,
                    model_config.model_id
                );
            }
            let model_id = match &inference {
                Some(inference) => loaded.add_to(inference, &model_config.model_id).await?,
                None => {
                    let LoadedModel {
                        model_id,
                        pipeline_name,
                        pipeline,
                        scheduler_config,
                        config,
                    } = loaded;
                    let mut builder =
                        InferenceRsBuilder::from_config(pipeline, scheduler_config, config)
                            .with_opt_log(self.log.clone())
                            .with_deferred_daemon_start(true);
                    if model_id != pipeline_name {
                        builder = builder.with_model_id(model_id.clone());
                    }
                    let built = inference.insert(builder.build().await);
                    register_alias(built, &model_id, &pipeline_name, &model_config.model_id)?;
                    model_id
                }
            };
            loaded_model_ids.push(model_id);
        }
        let inference = inference.expect("at least one model is configured");

        // Set the default model if specified
        if let Some(ref default_model_id) = self.default_model_id {
            inference
                .set_default_model_id(default_model_id)
                .map_err(|e| anyhow::anyhow!("Failed to set default model: {}", e))?;
        }

        // Log all models loaded
        info!("All models loaded: `{}`", loaded_model_ids.join("`, `"));

        // Log default model
        if let Some(ref default_id) = self.default_model_id {
            info!("Default model: {}", default_id);
        } else {
            info!(
                "Default model: {} (first model, from config key: {})",
                loaded_model_ids[0], self.models[0].model_id
            );
        }

        if inference_core::distributed::is_daemon() {
            inference.run_daemon_replicator_forever();
        }

        Ok(inference)
    }
}

// TODO: replace with best device?
/// Initializes the device to be used for computation, optionally forcing CPU usage and setting a seed.
fn init_device(force_cpu: bool, seed: Option<u64>) -> Result<inference_tensor::Device> {
    #[cfg(feature = "metal")]
    let device = if force_cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };
    #[cfg(not(feature = "metal"))]
    #[allow(clippy::if_same_then_else)]
    let device = if force_cpu {
        Device::Cpu
    } else if inference_core::distributed::use_nccl() {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    if let Some(seed) = seed {
        device.set_seed(seed)?;
    }

    Ok(device)
}

/// Initializes the device mapping configuration for distributing model layers.
/// Parses `--device-layers` entries: one layer count for device 0, or `ORD:NUM` per device.
pub fn parse_device_layers(device_layers: &[String]) -> Result<Vec<DeviceLayerMapMetadata>> {
    if let [layers] = device_layers
        && let Ok(layers) = layers.parse::<usize>()
    {
        return Ok(vec![DeviceLayerMapMetadata { ordinal: 0, layers }]);
    }
    let mut mapping: Vec<DeviceLayerMapMetadata> = Vec::new();
    for entry in device_layers {
        let parsed = entry
            .split_once(':')
            .and_then(|(ord, num)| Some((ord.parse::<usize>().ok()?, num.parse::<usize>().ok()?)));
        let Some((ordinal, layers)) = parsed else {
            anyhow::bail!("device layers entry `{entry}` is not ORD:NUM");
        };
        if mapping.iter().any(|m| m.ordinal == ordinal) {
            anyhow::bail!("device layers name ordinal {ordinal} twice");
        }
        mapping.push(DeviceLayerMapMetadata { ordinal, layers });
    }
    Ok(mapping)
}

fn init_mapper(
    num_device_layers: &Option<Vec<String>>,
    auto_device_map_params: &AutoDeviceMapParams,
) -> Result<DeviceMapSetting> {
    Ok(match num_device_layers {
        Some(device_layers) => DeviceMapSetting::Map(DeviceMapMetadata::from_num_device_layers(
            parse_device_layers(device_layers)?,
        )),
        None => DeviceMapSetting::Auto(auto_device_map_params.clone()),
    })
}

/// Logs hardware feature information and the model's sampling strategy and kind.
fn inference_instance_info(loader: &dyn Loader) {
    debug!(
        "avx: {}, neon: {}, f16c: {}",
        inference_tensor::utils::with_avx(),
        inference_tensor::utils::with_neon(),
        inference_tensor::utils::with_f16c()
    );

    debug!("Sampling method: penalties -> temperature -> topk -> topp -> minp -> multinomial");
    debug!("Model kind is: {}", loader.get_kind().to_string());
}

/// Determines whether paged attention should be enabled based on device type and preferences.
fn configure_paged_attn(device: &Device, paged_attn: Option<bool>) -> bool {
    if inference_core::distributed::use_nccl() {
        paged_attn.unwrap_or(defaults::PAGED_ATTN_CUDA)
    } else if device.is_cpu() {
        if paged_attn == Some(true) {
            warn!("Paged attention is not supported on CPU.");
        }

        defaults::PAGED_ATTN_CPU
    } else if device.is_cuda() {
        paged_attn.unwrap_or(defaults::PAGED_ATTN_CUDA)
    } else if device.is_metal() {
        paged_attn.unwrap_or(defaults::PAGED_ATTN_METAL)
    } else {
        false
    }
}

/// Initializes the cache configuration for paged attention based on provided parameters.
fn init_cache_config(
    paged_attn_block_size: Option<usize>,
    paged_attn_gpu_mem: Option<usize>,
    paged_attn_gpu_mem_usage: Option<f32>,
    paged_ctxt_len: Option<usize>,
    cache_type: PagedCacheType,
    no_paged_attn: bool,
) -> Result<Option<PagedAttentionConfig>> {
    match (
        paged_attn_block_size,
        paged_attn_gpu_mem,
        paged_attn_gpu_mem_usage,
        paged_ctxt_len,
        paged_attn_supported(),
        no_paged_attn,
    ) {
        (block_size, None, None, None, true, false) => Ok(Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::Utilization(0.9),
            cache_type,
        )?)),
        (block_size, None, None, Some(ctxt), true, false) => Ok(Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::ContextSize(ctxt),
            cache_type,
        )?)),
        (block_size, None, Some(f), None, true, false) => Ok(Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::Utilization(f),
            cache_type,
        )?)),
        (block_size, Some(m), None, None, true, false) => Ok(Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::MbAmount(m),
            cache_type,
        )?)),
        (block_size, Some(_m), Some(f), None, true, false) => {
            warn!("Both memory size and usage were specified, defaulting to the usage value.");
            Ok(Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::Utilization(f),
                cache_type,
            )?))
        }
        (block_size, Some(_m), None, Some(ctxt), true, false) => {
            warn!(
                "Both memory size and context length were specified, defaulting to context length."
            );
            Ok(Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::ContextSize(ctxt),
                cache_type,
            )?))
        }
        (block_size, None, Some(f), Some(_ctxt), true, false) => {
            warn!("Both context length and usage were specified, defaulting to the usage value.");
            Ok(Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::Utilization(f),
                cache_type,
            )?))
        }
        (_, _, _, _, _, _) => Ok(None),
    }
}

/// Creates a search embedding model configuration for agentic search reranking.
pub fn get_search_embedding_model(
    enable_search: bool,
    search_embedding_model: Option<SearchEmbeddingModel>,
) -> Option<SearchEmbeddingModel> {
    if enable_search {
        Some(search_embedding_model.unwrap_or_default())
    } else {
        None
    }
}

/// The runtime settings an engine loads its models with, kept so [`ModelLoadSettings::load_additional`] can add one.
#[derive(Clone)]
pub struct ModelLoadSettings {
    device: Device,
    token_source: TokenSource,
    num_device_layers: Option<Vec<String>>,
    in_situ_quant: Option<String>,
    chat_template: Option<String>,
    jinja_explicit: Option<String>,
    max_model_len: Option<usize>,
    hf_config_overrides: Option<HfConfigOverrides>,
    encoder_cache_memory_bytes: Option<usize>,
    requested_cache: Option<PagedAttentionConfig>,
    max_seqs: usize,
    limits: SchedulerLimits,
    no_kv_cache: bool,
    add_model_config: inference_core::AddModelConfig,
    mtp_runtime: MtpRuntimeConfig,
}

/// A loaded model not yet served: its pipeline, the scheduler it runs under and the id requests will use.
pub struct LoadedModel {
    pub model_id: String,
    pipeline_name: String,
    pipeline: LoadedPipeline,
    scheduler_config: SchedulerConfig,
    config: inference_core::AddModelConfig,
}

impl ModelLoadSettings {
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn token_source(&self) -> &TokenSource {
        &self.token_source
    }

    /// Whether listed models inherit an ISQ setting, which a `quant` choice would land on top of.
    pub fn has_isq(&self) -> bool {
        self.in_situ_quant.is_some()
    }

    async fn load(
        &self,
        model_config: &ModelConfig,
        paged_attn_config: Option<PagedAttentionConfig>,
        mtp_config: Option<MtpConfig>,
        max_seqs: usize,
        announce: bool,
    ) -> Result<LoadedModel> {
        let model = model_config.model.clone();
        let dtype = get_model_dtype(&model)?;
        let mapper = init_mapper(
            &model_config
                .num_device_layers
                .clone()
                .or(self.num_device_layers.clone()),
            &get_auto_device_map_params(&model)?,
        )?;
        let isq = model_config
            .in_situ_quant
            .as_ref()
            .or(self.in_situ_quant.as_ref())
            .map(|isq| parse_isq_value(isq, Some(&self.device)).map_err(|e| anyhow::anyhow!("{e}")))
            .transpose()?;
        let loader_config = ModelLoaderConfig {
            source: Arc::new(model),
            token_source: self.token_source.clone(),
            hf_revision: model_config.hf_revision.clone(),
            dtype,
            device: self.device.clone(),
            device_map_setting: mapper,
            isq,
            paged_attn_config,
            silent: false,
            chat_template: model_config
                .chat_template
                .clone()
                .or(self.chat_template.clone()),
            jinja_explicit: model_config
                .jinja_explicit
                .clone()
                .or(self.jinja_explicit.clone()),
            max_model_len: model_config.max_model_len.or(self.max_model_len),
            hf_config_overrides: model_config
                .hf_config_overrides
                .clone()
                .or(self.hf_config_overrides.clone()),
            mtp_config,
            encoder_cache_memory_bytes: model_config
                .encoder_cache_memory_bytes
                .map(NonZeroUsize::get)
                .or(self.encoder_cache_memory_bytes),
            overrides: Default::default(),
        };
        let loader = loader_config.build_loader(self.no_kv_cache)?;
        if announce {
            inference_instance_info(&*loader);
        }
        let pipeline: LoadedPipeline = loader_config.load(&*loader, self.mtp_runtime).await?;
        let scheduler_config = SchedulerConfig::for_pipeline(
            &pipeline,
            paged_attn_config.is_some(),
            max_seqs,
            self.limits,
        )
        .await?;
        // Use the pipeline's name() as the canonical ID, but allow an alias.
        let pipeline_name = pipeline.lock().await.name();
        Ok(LoadedModel {
            model_id: model_config
                .alias
                .clone()
                .unwrap_or_else(|| pipeline_name.clone()),
            pipeline_name,
            pipeline,
            scheduler_config,
            config: self
                .add_model_config
                .clone()
                .with_loader_config(loader_config),
        })
    }

    /// Loads one more model for an engine already serving others, sizing its paged cache from the memory left now.
    pub async fn load_additional(&self, model_config: &ModelConfig) -> Result<LoadedModel> {
        let max_seqs = match get_tgt_non_granular_index(&model_config.model) {
            Some(_) => 1,
            None => self.max_seqs,
        };
        let plan = plan_paged_kv(
            &[PagedKvModelRequest {
                paged_attn: self.requested_cache,
                max_num_seqs: max_seqs,
            }],
            Default::default(),
        )?;
        self.load(model_config, plan.paged_attn[0], None, max_seqs, false)
            .await
    }
}

impl LoadedModel {
    /// Serves this model from `inference` beside the models it has; returns the id requests use.
    pub(crate) async fn add_to(self, inference: &InferenceRs, config_key: &str) -> Result<String> {
        let Self {
            model_id,
            pipeline_name,
            pipeline,
            scheduler_config,
            config,
        } = self;
        inference
            .add_model(model_id.clone(), pipeline, scheduler_config, config)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to add model {model_id}: {e}"))?;
        register_alias(inference, &model_id, &pipeline_name, config_key)?;
        Ok(model_id)
    }
}

fn register_alias(
    inference: &InferenceRs,
    model_id: &str,
    pipeline_name: &str,
    config_key: &str,
) -> Result<()> {
    if model_id != pipeline_name {
        // The pipeline's own name is a convenience alias; another copy of the same checkpoint may already hold it.
        if inference.model_exists(pipeline_name)? {
            info!(
                "Model `{model_id}` loaded (config key: {config_key}; `{pipeline_name}` names another model)"
            );
            return Ok(());
        }
        inference
            .register_model_alias(pipeline_name.to_string(), model_id)
            .map_err(|e| anyhow::anyhow!(e))?;
        info!("Model `{model_id}` loaded (pipeline: `{pipeline_name}`; config key: {config_key})");
    } else {
        info!("Model `{model_id}` loaded (from config key: {config_key})");
    }
    Ok(())
}

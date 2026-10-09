#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
// Shared helpers go unused when a family is off; the all-families lint still catches real dead code.
#![cfg_attr(
    not(all(
        feature = "models-gemma",
        feature = "models-llama",
        feature = "models-other",
        feature = "models-phi",
        feature = "models-qwen"
    )),
    allow(dead_code, unused_imports, unused_macros)
)]
pub use engine::{
    AgentRunner, DEFAULT_MAX_TOOL_ROUNDS, ENGINE_INSTRUCTIONS, Engine, EngineInstruction,
    IntervalLogger, SearchEmbeddingModel, SpeculativeStats, TERMINATE_ALL_NEXT_STEP, agent,
    agentic_session,
    agentic_session::{AgenticSessionStore, SerializedSession, SerializedVideo},
};
use hf_hub::Cache;
use inference_nn::matformer;
use inference_tensor::Device;
pub use pipeline::CalibrationStatus;
pub use pipeline::ModelCategory;
pub use pipeline::Pipeline;
use speculative::SpeculativeConfig;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use std::{
    cell::RefCell,
    error::Error,
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex, RwLock},
    thread::{self, JoinHandle},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc::{Sender, channel};
use tracing::{debug, info, warn};

fn build_engine_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(inference_tensor::utils::get_num_threads())
        .on_thread_start(inference_tensor::utils::set_thread_affinity)
        .build()
        .unwrap()
}

pub const INFERENCE_RS_GIT_REVISION: &str = match option_env!("INFERENCE_RS_GIT_REVISION") {
    Some(value) => value,
    None => "unknown",
};
pub const INFERENCE_RS_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const DEFAULT_ENGINE_REQUEST_QUEUE_CAPACITY: usize = 10_000;
// Bounded so a wedged engine cannot hang drop forever; normal termination takes milliseconds.
const ENGINE_DROP_JOIN_TIMEOUT: Duration = Duration::from_secs(10);
const ENGINE_DROP_POLL_INTERVAL: Duration = Duration::from_millis(5);
pub const REQUEST_QUEUE_DURATION_METRIC: &str = "inference_request_queue_duration_seconds";

// GPU tests share one device and process-global CUDA state (memory pools, graph scopes), so they run one at a time.
#[cfg(all(test, feature = "cuda"))]
static CUDA_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// GPU tests run by default under `--features cuda`, skip without a device, and hold the lock for the test's lifetime.
#[cfg(all(test, feature = "cuda"))]
macro_rules! skip_without_cuda {
    () => {
        let _cuda_test_guard = crate::CUDA_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if inference_tensor::Device::new_cuda(0).is_err() {
            eprintln!("SKIP {}: no CUDA device", module_path!());
            return Ok(());
        }
    };
}

use inference_nn::get_mut_arcmutex;
pub use inference_nn::metal::warmup_metal_kernels;
use inference_nn::{
    amoe, attention, cuda, device_map, flashinfer, gdn, kv_cache, lora, model, moe,
    paged_attention, sampler, topology, utils,
};

mod adapter;
mod agent_approval;
mod chat_collector;
mod engine;
use inference_nn::media_inputs::video as video_input;
pub use video_input::{VideoFrameSampling, VideoInput, sample_frame_indices};
mod embedding_models;
pub mod search;

mod block_diffusion;
pub mod distributed;
use inference_gguf as gguf;
pub use inference_protocol::chat_template::{ChatTemplate, is_chat_template_request_error};
pub use inference_protocol::files;
pub use inference_protocol::images;
mod models;
mod pipeline;
mod prefix_cacher;
pub use inference_protocol::reasoning_parsers;
pub mod remote_fetch;
mod request;
mod response;
mod scheduler;
mod sequence;
pub(crate) mod sequence_macros;
pub(crate) mod speculative;
use inference_protocol::tools;
mod vision_models;

pub(crate) use adapter::AdapterLease;
#[doc(hidden)]
pub use adapter::DynamicLoraRuntime;
pub use adapter::{
    AdapterGenerationId, AdapterGenerationParseError, AdapterSelection, DEFAULT_LORA_MAX_ADAPTERS,
    DEFAULT_LORA_MAX_BYTES, DEFAULT_LORA_MAX_RANK, LoraAdapterError, LoraAdapterFiles,
    LoraAdapterInfo, LoraAdapterLoadPolicy, LoraAdapterRoute, LoraAdapterSpec,
    LoraAdapterSpecParseError, LoraResidentGenerationInfo, LoraRuntimeConfig, LoraRuntimeStatus,
    MAX_LORA_ALIAS_BYTES,
};
pub use agent_approval::{
    AgentToolApproval, AgentToolApprovalAsyncCallback, AgentToolApprovalCallback,
    AgentToolApprovalDecision, AgentToolApprovalHandler,
};
pub use amoe::{AnyMoeConfig, AnyMoeExpertType};
pub use chat_collector::{ChatResponseCollector, encode_agentic_tool_images};
pub use device_map::{DeviceLayerMapMetadata, DeviceMapMetadata, DeviceMapSetting};
pub use files::{
    FILE_PURPOSE_GENERATED_IMAGE, FILE_PURPOSE_USER_DATA, File, FileContent, FileSource, FileStore,
    RequestedFile,
};
pub use gguf::{GGUF_MULTI_FILE_DELIMITER, GGUFArchitecture};
pub use inference_audio::AudioInput;
pub use inference_code_exec::{
    CodeExecutionConfig, DEFAULT_CODE_EXEC_TIMEOUT_SECS, DEFAULT_SHELL_TIMEOUT_SECS, ShellConfig,
};
pub use inference_mcp::{
    AgentPermission, AgentToolApprovalNotifier, AgentToolApprovalRequest, AgentToolKind,
    AgentToolMetadata, AgentToolSource, CalledFunction, CodeExecutionApprovalNotifier,
    CodeExecutionPermission, Function, MultimodalToolCallback, ShellOptions, ShellSkillMount, Tool,
    ToolCallContext, ToolCallback, ToolCallbackKind, ToolCallbackWithTool, ToolOutput, ToolType,
    sandbox_key,
};
pub use inference_mcp::{McpClient, McpClientConfig, McpServerConfig, McpServerSource};
pub use inference_models_speech::{SpeechGenerationConfig, SpeechOptions, utils as speech_utils};
pub use inference_quant::parse_isq_value;
pub use inference_quant::{IsqBits, IsqType};
pub use inference_sandbox::{NetworkMode, SandboxMode, SandboxPolicy, SandboxProfile};
pub use paged_attention::{MemoryGpuConfig, PagedAttentionConfig, PagedCacheType};
#[cfg(feature = "models-gemma")]
pub use pipeline::GemmaLoader;
#[cfg(feature = "models-qwen")]
pub use pipeline::Qwen2Loader;
#[cfg(feature = "models-other")]
pub use pipeline::Starcoder2Loader;
pub use pipeline::get_device_layers_for_loader;
pub use pipeline::hf::build_api_with_cache;
pub use pipeline::hf::{
    HF_HUB_OFFLINE_ENV, hf_home_dir, hf_hub_cache_dir, hf_token_path, is_hf_hub_offline,
    list_model_files, probe_hf_repo_files, read_model_file_range, try_get_model_file,
};
// Named only by the ModelSelected schema attributes.
#[cfg(feature = "utoipa")]
pub use pipeline::UqffWriteSpec;
pub use pipeline::{
    AdapterPaths, AnyMoeLoader, AnyMoePipeline, AutoDeviceMapParams, AutoEmbeddingLoader,
    AutoLoader, AutoLoaderBuilder, AutoMultimodalLoader, AutoNormalLoader,
    DiffusionGenerationParams, DiffusionLoader, DiffusionLoaderBuilder, DiffusionLoaderType,
    EmbeddingLoader, EmbeddingLoaderBuilder, EmbeddingLoaderType, EmbeddingModelPaths,
    EmbeddingSpecificConfig, GGMLLoader, GGMLLoaderBuilder, GGMLSpecificConfig, GGUFLoader,
    GGUFLoaderBuilder, GGUFSpecificConfig, HfConfigOverrides, IsqOrganization, LoadOptions, Loader,
    LocalModelPaths, Modalities, ModelKind, ModelPaths, MultimodalLoader, MultimodalLoaderBuilder,
    MultimodalLoaderType, MultimodalPromptPrefixer, MultimodalSpecificConfig, NormalLoader,
    NormalLoaderBuilder, NormalLoaderType, NormalSpecificConfig, ResolvedLoraAdapter, SpeechLoader,
    SpeechLoaderType, SpeechPipeline, SupportedModality, TokenSource, UQFF_MULTI_FILE_DELIMITER,
    UqffWriteConfig, expand_isq_value, expand_uqff_shards, parse_uqff_shard,
    resolve_uqff_report_output, resolve_uqff_shorthand,
};
#[cfg(feature = "models-llama")]
pub use pipeline::{
    Idefics2Loader, LLaVALoader, LLaVANextLoader, LlamaLoader, MistralLoader, MixtralLoader,
};
#[cfg(feature = "models-phi")]
pub use pipeline::{Phi2Loader, Phi3Loader, Phi3VLoader};
pub use request::{
    ApproximateUserLocation, CalibrationAction, CalibrationRequest, Constraint,
    DetokenizationRequest, FINISH_REASON_CANCELED, FINISH_REASON_LENGTH,
    ImageGenerationResponseFormat, LlguidanceGrammar, MessageContent, NormalRequest,
    ReasoningEffort, RequantizeRequest, Request, RequestCancellation, RequestMessage,
    SearchContextSize, TokenizationRequest, WebSearchContentType, WebSearchFilters,
    WebSearchImageSettings, WebSearchOptions, WebSearchReturnTokenBudget, WebSearchUserLocation,
    resolve_reasoning_controls,
};
pub use response::*;
pub use sampler::{
    CustomLogitsProcessor, DrySamplingParams, ModelGenerationDefaults, SamplingParams, StopTokens,
};
pub use scheduler::{
    DEFAULT_MAX_DECODE_STEPS_BEFORE_PREFILL, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_PREFILL_CHUNK_TOKENS, SchedulerConfig, SchedulerLimits,
};
pub use search::{SearchCallback, SearchEmbedder, SearchFunctionParameters, SearchResult};
use serde::Serialize;
pub use speculative::{
    MtpConfig, MtpDraftSamplingMethod, MtpRuntimeConfig, reserve_external_mtp_memory,
    reserve_external_mtp_memory_with_runtime,
};
use tokio::runtime::Runtime;
pub use tools::{
    AllowedToolChoice, AllowedToolsMode, AllowedToolsToolChoice, AllowedToolsToolChoiceType,
    NamedFunctionToolChoice, ToolCallResponse, ToolCallType, ToolChoice,
};
pub use topology::Topology;
pub use utils::debug::{LogVerbosity, initialize_inference_logging, initialize_logging};
pub use utils::memory_usage::MemoryUsage;
pub use utils::normal::{ModelDType, TryIntoDType};
pub use utils::{paged_attn_supported, using_flash_attn};

// re-export llguidance for easier LlguidanceGrammar construction
pub use llguidance;

pub static GLOBAL_HF_CACHE: OnceLock<Cache> = OnceLock::new();

/// Set the process-wide Hugging Face cache path before model discovery.
pub fn set_hf_cache_path(path: impl Into<PathBuf>) {
    GLOBAL_HF_CACHE.get_or_init(|| Cache::new(path.into()));
}

mod inference_rs;
pub use inference_rs::*;

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
use candle_core::Device;
use engine::Engine;
pub use engine::{
    agentic_session::{AgenticSessionStore, SerializedSession, SerializedVideo},
    get_engine_terminate_flag, reset_engine_terminate_flag, should_terminate_engine_sequences,
    EngineInstruction, IntervalLogger, SearchEmbeddingModel, DEFAULT_MAX_TOOL_ROUNDS,
    ENGINE_INSTRUCTIONS, TERMINATE_ALL_NEXT_STEP,
};
use hf_hub::Cache;
pub use lora::Ordering;
pub use pipeline::CalibrationStatus;
pub use pipeline::ModelCategory;
pub use pipeline::Pipeline;
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
use tokio::sync::mpsc::{channel, Sender};
use tracing::{debug, info, warn};

fn build_engine_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(candle_core::utils::get_num_threads())
        .on_thread_start(candle_core::utils::set_thread_affinity)
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
        if candle_core::Device::new_cuda(0).is_err() {
            eprintln!("SKIP {}: no CUDA device", module_path!());
            return Ok(());
        }
    };
}

use inference_nn::get_mut_arcmutex;
pub use inference_nn::metal::warmup_metal_kernels;
use inference_nn::{
    amoe, attention, cuda, device_map, flashinfer, gdn, kv_cache, lora, model, moe, ops,
    paged_attention, sampler, topology, utils,
};
pub use inference_nn::{layers, matformer};

mod adapter;
mod agent_approval;
mod engine;
mod video_input;
pub use selection::model_loader::{
    get_auto_device_map_params, get_model_dtype, get_tgt_non_granular_index, LoaderBuilder,
};
pub use video_input::{
    sample_frame_indices, VideoFrameSampling, VideoInput, DEFAULT_VIDEO_FRAME_LIMIT,
};
mod embedding_models;
mod search;

pub use selection::model_selected::ModelSelected;

mod block_diffusion;
mod diagnostics;
pub mod distributed;
pub mod files;
mod gguf;
mod models;
mod pipeline;
mod prefix_cacher;
pub mod reasoning_parsers;
pub mod remote_fetch;
mod request;
pub mod resource_plan;
mod response;
mod scheduler;
pub mod selection;
mod sequence;
pub(crate) mod sequence_macros;
pub mod speculative;
mod tools;
mod vision_models;
mod xlora_models;

pub use diagnostics::{
    check_hf_gated_access, collect_system_info, run_doctor, BuildInfo, CpuInfo, DeviceInfo,
    DoctorCheck, DoctorReport, DoctorStatus, HfConnectivityInfo, MemoryInfo, SystemInfo,
};
mod tuning;
pub use tuning::{
    auto_tune, AutoTuneRequest, AutoTuneResult, FitStatus, QualityTier, TuneCandidate, TuneProfile,
};

pub(crate) use adapter::AdapterLease;
#[doc(hidden)]
pub use adapter::DynamicLoraRuntime;
pub use adapter::{
    AdapterGenerationId, AdapterGenerationParseError, AdapterSelection, LoraAdapterError,
    LoraAdapterFiles, LoraAdapterInfo, LoraAdapterLoadPolicy, LoraAdapterRoute, LoraAdapterSpec,
    LoraAdapterSpecParseError, LoraResidentGenerationInfo, LoraRuntimeConfig, LoraRuntimeStatus,
    DEFAULT_LORA_MAX_ADAPTERS, DEFAULT_LORA_MAX_BYTES, DEFAULT_LORA_MAX_RANK, MAX_LORA_ALIAS_BYTES,
};
pub use agent_approval::{
    AgentToolApproval, AgentToolApprovalAsyncCallback, AgentToolApprovalCallback,
    AgentToolApprovalDecision, AgentToolApprovalFuture, AgentToolApprovalHandler,
};
pub use amoe::{AnyMoeConfig, AnyMoeExpertType};
pub use device_map::{
    DeviceLayerMapMetadata, DeviceMapMetadata, DeviceMapSetting, LayerDeviceMapper,
};
pub use files::{
    format_from_name, is_text_mime, mime_for_format, File, FileContent, FileSource, FileStore,
    RequestedFile, FILE_PURPOSE_AGENT_OUTPUT, FILE_PURPOSE_USER_DATA, MODEL_INLINE_BYTES,
    WIRE_EMBED_LIMIT_BYTES,
};
pub use gguf::{GGUFArchitecture, GGUF_MULTI_FILE_DELIMITER};
pub use inference_audio::AudioInput;
pub use inference_code_exec::{
    CodeExecutionApproval, CodeExecutionApprovalCallback, CodeExecutionConfig, ShellConfig,
    DEFAULT_CODE_EXEC_TIMEOUT_SECS, DEFAULT_SHELL_TIMEOUT_SECS,
};
pub use inference_mcp::{
    AgentPermission, AgentToolApprovalNotifier, AgentToolApprovalRequest, AgentToolKind,
    AgentToolMetadata, AgentToolSource, CalledFunction, CodeExecutionApprovalNotifier,
    CodeExecutionApprovalRequest, CodeExecutionPermission, Function, MultimodalToolCallback,
    ShellOptions, ShellSkillMount, Tool, ToolCallContext, ToolCallback, ToolCallbackKind,
    ToolCallbackWithTool, ToolOutput, ToolType,
};
pub use inference_mcp::{
    McpClient, McpClientConfig, McpServerConfig, McpServerSource, McpToolInfo,
};
pub use inference_models_speech::{utils as speech_utils, SpeechGenerationConfig};
pub use inference_quant::parse_isq_value;
pub use inference_quant::{IsqBits, IsqType};
pub use inference_sandbox::{NetworkMode, SandboxMode, SandboxPolicy, SandboxProfile};
pub use paged_attention::{MemoryGpuConfig, PagedAttentionConfig, PagedCacheType};
pub use pipeline::hf::{
    get_model_file, hf_home_dir, hf_hub_cache_dir, hf_token_path, is_hf_hub_offline,
    list_model_files, probe_hf_repo_files, read_model_file_range, try_get_model_file,
    HF_HUB_OFFLINE_ENV,
};
#[cfg(feature = "models-gemma")]
pub use pipeline::GemmaLoader;
#[cfg(feature = "models-qwen")]
pub use pipeline::Qwen2Loader;
#[cfg(feature = "models-other")]
pub use pipeline::Starcoder2Loader;
pub use pipeline::{
    chat_template::{is_chat_template_request_error, ChatTemplate},
    expand_isq_value, expand_uqff_shards, parse_uqff_shard, resolve_uqff_report_output,
    resolve_uqff_shorthand, AdapterPaths, AnyMoeLoader, AnyMoePipeline, AutoDeviceMapParams,
    AutoLoader, AutoLoaderBuilder, DiffusionGenerationParams, DiffusionLoader,
    DiffusionLoaderBuilder, DiffusionLoaderType, EmbeddingLoader, EmbeddingLoaderBuilder,
    EmbeddingLoaderType, EmbeddingModelPaths, EmbeddingSpecificConfig, GGMLLoader,
    GGMLLoaderBuilder, GGMLSpecificConfig, GGUFLoader, GGUFLoaderBuilder, GGUFSpecificConfig,
    HfConfigOverrides, IsqOrganization, Loader, LocalModelPaths, Modalities, ModelKind, ModelPaths,
    MultimodalLoader, MultimodalLoaderBuilder, MultimodalLoaderType, MultimodalPromptPrefixer,
    MultimodalSpecificConfig, NormalLoader, NormalLoaderBuilder, NormalLoaderType,
    NormalSpecificConfig, ResolvedLoraAdapter, SpeechLoader, SpeechLoaderType, SpeechPipeline,
    SupportedModality, TokenSource, UqffWriteConfig, UQFF_MULTI_FILE_DELIMITER,
};
#[cfg(feature = "models-llama")]
pub use pipeline::{
    Idefics2Loader, LLaVALoader, LLaVANextLoader, LlamaLoader, MistralLoader, MixtralLoader,
};
#[cfg(feature = "models-phi")]
pub use pipeline::{Phi2Loader, Phi3Loader, Phi3VLoader};
pub use request::{
    resolve_reasoning_controls, ApproximateUserLocation, CalibrationAction, CalibrationRequest,
    Constraint, DetokenizationRequest, ImageGenerationResponseFormat, LlguidanceGrammar,
    MessageContent, NormalRequest, ReasoningControlError, ReasoningEffort,
    ReasoningEffortParseError, Request, RequestMessage, ResolvedReasoningControls,
    SearchContextSize, TokenizationRequest, WebSearchContentType, WebSearchFilters,
    WebSearchImageSettings, WebSearchOptions, WebSearchReturnTokenBudget, WebSearchUserLocation,
    DEFAULT_ENABLE_THINKING,
};
pub use resource_plan::{
    plan_paged_kv, PagedKvModelRequest, PagedKvPlan, PagedKvPolicy, RuntimeResourcePlanOptions,
};
pub use response::*;
pub use sampler::{
    CustomLogitsProcessor, DrySamplingParams, ModelGenerationDefaults, SamplingParams, StopTokens,
    TopLogprob,
};
pub use scheduler::{
    DefaultSchedulerMethod, SchedulerConfig, SchedulerLimits,
    DEFAULT_MAX_DECODE_STEPS_BEFORE_PREFILL, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_PREFILL_CHUNK_TOKENS,
};
pub use search::{SearchCallback, SearchFunctionParameters, SearchResult};
use serde::Serialize;
pub use speculative::{
    reserve_external_mtp_memory, reserve_external_mtp_memory_with_runtime, MtpConfig,
    MtpDraftSamplingMethod, MtpRuntimeConfig, SpeculativeConfig,
};
use tokio::runtime::Runtime;
pub use tools::{
    AllowedToolChoice, AllowedToolsMode, AllowedToolsToolChoice, AllowedToolsToolChoiceType,
    BuiltinToolChoice, BuiltinToolChoiceType, NamedFunctionToolChoice, ToolCallResponse,
    ToolCallType, ToolCallbacks, ToolChoice,
};
pub use topology::{LayerTopology, Topology};
pub use utils::debug::{
    default_inference_filter, initialize_inference_logging, initialize_logging,
    initialize_logging_with_filter, LogVerbosity,
};
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

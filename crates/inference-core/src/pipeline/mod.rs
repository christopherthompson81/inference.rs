mod amoe;
mod auto;
pub(crate) mod cache_manager;
pub(crate) use crate::model::{
    ModelForwardContext, RecurrentMetadata, recurrent_batch_kind_for_input,
};
pub use cache_manager::CacheManager;
pub mod chat_template;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_graph;
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use cuda_graph::CudaDecodeGraphLaunch;
mod diffusion;
mod diffusion_processor;
mod embedding;
pub(crate) mod execution;
mod ggml;
mod gguf;
pub(crate) mod hf;
mod hf_config;
mod inputs_processor;
mod isq;
mod isq_flow;
pub use isq_flow::CalibrationStatus;
pub(crate) mod llg;
mod loaders;
mod loading;
pub(crate) mod model_config;
mod multimodal;
mod normal;
mod paths;
pub(crate) mod processing;
pub(crate) mod prompt_chunks;
mod response;
pub(crate) mod sampling;
mod speech;
mod step;
mod tiktoken;
pub(crate) mod tokenizer;
mod tokens;

use crate::IntervalLogger;
use crate::PagedAttentionConfig;
use crate::amoe::{AnyMoeConfig, AnyMoeExpertType, AnyMoeTrainingInputs, AnyMoeTrainingResult};
use crate::attention::FlashParams;
use crate::device_map::DeviceMapper;
use crate::kv_cache::PagedAuxiliaryPrefixState;
pub use crate::model::DiffusionGenerationParams;
use crate::paged_attention::PagedAttentionInputMetadata;
use crate::paged_attention::{
    AttentionBackendKind, CacheConfig, CacheEngine, CacheMemoryReservations, MemoryGpuConfig,
    ModelConfigLike,
};
use crate::prefix_cacher::PrefixCacheManagerV2;
pub use amoe::{AnyMoeLoader, AnyMoePipeline};
pub use auto::{AutoLoader, AutoLoaderBuilder};
pub use diffusion::{DiffusionLoader, DiffusionLoaderBuilder};
pub(crate) use embedding::EmbeddingLoadContext;
pub use embedding::{EmbeddingLoader, EmbeddingLoaderBuilder, EmbeddingSpecificConfig};
#[doc(hidden)]
pub use execution::{StepLookahead, StepSubmission};
use futures::future::BoxFuture;
pub use ggml::{GGMLLoader, GGMLLoaderBuilder, GGMLSpecificConfig};
pub use gguf::{GGUFLoader, GGUFLoaderBuilder, GGUFSpecificConfig};
pub use hf_config::HfConfigOverrides;
use image::DynamicImage;
pub use inference_nn::loaders::{Modalities, MultimodalPromptPrefixer, SupportedModality};
use inference_protocol::chat_template::ChatTemplate;
pub use inputs_processor::InputProcessorOutput;
pub(crate) use isq::IsqModelLoader;
#[cfg(feature = "utoipa")]
pub use isq::UqffWriteSpec;
pub use isq::{
    IsqOrganization, UQFF_MULTI_FILE_DELIMITER, UqffWriteConfig, expand_isq_value,
    expand_uqff_shards, parse_uqff_shard, resolve_uqff_report_output, resolve_uqff_shorthand,
};
use llguidance::toktrie::TokEnv;
#[cfg(feature = "models-gemma")]
pub use loaders::GemmaLoader;
pub(crate) use loaders::MultimodalProcessorFactory;
#[cfg(feature = "models-qwen")]
pub use loaders::Qwen2Loader;
#[cfg(feature = "models-other")]
pub use loaders::Starcoder2Loader;
pub(crate) use loaders::checkpoint_runtime_size;
pub use loaders::{
    AdapterKind, AutoDeviceMapParams, AutoEmbeddingLoader, AutoMultimodalLoader, AutoNormalLoader,
    DeviceMappedModelLoader, DiffusionLoaderType, DiffusionModel, DiffusionModelLoader,
    EmbeddingLoaderType, EmbeddingModel, EmbeddingModelLoader, EmbeddingModelPaths,
    EmbeddingModule, EmbeddingModulePaths, EmbeddingModuleType, FluxLoader, Loader,
    LocalModelPaths, ModelKind, ModelPaths, MultimodalLoaderType, MultimodalModel,
    MultimodalModelLoader, NormalLoaderType, NormalLoadingMetadata, NormalModel, NormalModelLoader,
    PrettyName, QuantizationKind, TokenSource,
};
#[cfg(feature = "models-llama")]
pub use loaders::{
    Idefics2Loader, LLaVALoader, LLaVANextLoader, LlamaLoader, MistralLoader, MixtralLoader,
};
#[cfg(feature = "models-phi")]
pub use loaders::{Phi2Loader, Phi3Loader, Phi3VLoader};
pub(crate) use step::start_decoding_prompt_rows;
#[allow(clippy::too_many_arguments)]
pub fn get_device_layers_for_loader(
    loader: &dyn loaders::DeviceMappedModelLoader,
    config: &str,
    num_layers: usize,
    layer_sizes_in_bytes: Vec<usize>,
    non_mapped_size_in_bytes: usize,
    total_model_size_in_bytes: usize,
    devices: &[Device],
    dtype: DType,
    params: &loaders::AutoDeviceMapParams,
    paged_attn_config: Option<&mut PagedAttentionConfig>,
) -> Result<crate::device_map::DeviceMapMetadata> {
    loaders::auto_device_map::get_device_layers(
        loader,
        config,
        num_layers,
        layer_sizes_in_bytes,
        non_mapped_size_in_bytes,
        total_model_size_in_bytes,
        devices,
        dtype,
        params,
        paged_attn_config,
    )
}

fn finish_dynamic_lora_runtime(
    paths: &dyn ModelPaths,
    layers: Arc<inference_quant::LoraLayerRegistry>,
    runtime_config: crate::LoraRuntimeConfig,
    live_updates: bool,
) -> Result<Arc<crate::DynamicLoraRuntime>> {
    let AdapterPaths::Lora(adapter_paths) = paths.get_adapter_paths() else {
        unreachable!("LoRA loaders require resolved LoRA adapter paths")
    };

    layers.finalize()?;
    let runtime = Arc::new(crate::DynamicLoraRuntime::new(
        layers,
        runtime_config,
        live_updates,
    )?);
    for adapter in adapter_paths {
        let info = runtime.load_from_safetensors(
            adapter.alias.clone(),
            adapter.source.clone(),
            adapter.revision.clone(),
            &adapter.config_path,
            &adapter.weights_path,
        )?;
        tracing::info!(
            alias = %info.alias,
            generation = %info.generation,
            rank = info.rank,
            bytes = info.bytes,
            "LoRA adapter preloaded"
        );
    }
    Ok(runtime)
}

use inference_quant::IsqType;
pub use multimodal::{MultimodalLoader, MultimodalLoaderBuilder, MultimodalSpecificConfig};
pub use normal::{NormalLoader, NormalLoaderBuilder, NormalSpecificConfig};
pub(crate) use paths::{AdapterPathOptions, XLoraPreload, get_chat_template};
pub use paths::{AdapterPaths, ResolvedLoraAdapter};
#[cfg(feature = "models-llama")]
pub(crate) use processing::apply_chat_template;
pub(crate) use processing::{BasicProcessor, MessagesAction, Processor};
use rand_isaac::Isaac64Rng;
pub use speech::{SpeechLoader, SpeechLoaderType, SpeechPipeline};
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;

use tokenizers::Tokenizer;

use anyhow::Result;
use candle_core::{DType, Device, DeviceLocation, IndexOp, Tensor, Var};

use crate::paged_attention::block_hash::{
    MultimodalAttentionPolicy, adapter_generation_key, compute_block_hashes,
};
use crate::sequence::Sequence;

use prompt_chunks::{PromptChunkPlan, next_prompt_chunk_group};

pub(crate) use self::inputs_processor::is_inputs_processor_validation_error;
pub use self::inputs_processor::{
    InputsProcessor, InputsProcessorType, text_models_inputs_processor,
};
use crate::paged_attention::PagedAttentionMeta;

#[cfg(feature = "cuda")]
pub(crate) fn synchronize_cuda_contexts(primary: &Device, mapper: &dyn DeviceMapper) -> Result<()> {
    let mut devices = mapper.get_unique_devices();
    if !devices.iter().any(|device| device.same_device(primary)) {
        devices.push(primary.clone());
    }
    for device in devices {
        if let Device::Cuda(cuda) = device {
            cuda.cuda_stream().context().synchronize()?;
        }
    }
    Ok(())
}

pub(crate) fn resolve_lora_execution(
    runtime: Option<&crate::DynamicLoraRuntime>,
    input_ids: &Tensor,
    paged_attn_meta: Option<&PagedAttentionInputMetadata>,
    flash_meta: &FlashParams,
    adapter_leases: &[Option<crate::AdapterLease>],
) -> candle_core::Result<Option<Arc<inference_quant::LoraExecution>>> {
    let (batch, sequence_length) = input_ids.dims2()?;
    let query_lens = if flash_meta.packed {
        if batch != 1 {
            candle_core::bail!("packed adapter routing requires a flat physical batch");
        }
        let query_lens = paged_attn_meta
            .and_then(|metadata| metadata.query_lens.as_deref())
            .ok_or_else(|| {
                candle_core::Error::msg("packed adapter routing requires logical query lengths")
            })?;
        if adapter_leases.len() != query_lens.len() {
            candle_core::bail!(
                "adapter lease count {} does not match packed logical sequence count {}",
                adapter_leases.len(),
                query_lens.len()
            );
        }
        let logical_tokens = query_lens.iter().sum::<usize>();
        if logical_tokens != sequence_length {
            candle_core::bail!(
                "packed logical query lengths total {logical_tokens} does not match physical sequence length {sequence_length}"
            );
        }
        Some(query_lens)
    } else {
        if adapter_leases.len() != batch {
            candle_core::bail!(
                "adapter lease count {} does not match model batch size {batch}",
                adapter_leases.len()
            );
        }
        None
    };
    if adapter_leases.iter().all(Option::is_none) {
        return Ok(None);
    }
    let runtime = runtime.ok_or_else(|| {
        candle_core::Error::msg("request selected an adapter on a pipeline without dynamic LoRA")
    })?;
    match query_lens {
        Some(query_lens) => runtime
            .ragged_execution(adapter_leases, query_lens)
            .map(Some),
        None => runtime.execution(adapter_leases, sequence_length).map(Some),
    }
}

pub(crate) fn validate_lora_loader_config(
    adapters: Option<&[crate::LoraAdapterSpec]>,
    runtime_config: Option<crate::LoraRuntimeConfig>,
) -> anyhow::Result<()> {
    if let Some(runtime_config) = runtime_config {
        runtime_config.validate()?;
    }
    let Some(adapters) = adapters else {
        return Ok(());
    };
    let mut aliases = std::collections::HashSet::new();
    for adapter in adapters {
        let alias = adapter.alias.trim();
        if alias.is_empty() {
            anyhow::bail!("LoRA adapter alias must not be empty");
        }
        if alias.len() > crate::MAX_LORA_ALIAS_BYTES {
            anyhow::bail!(
                "LoRA adapter alias must not exceed {} bytes",
                crate::MAX_LORA_ALIAS_BYTES
            );
        }
        if adapter.source.trim().is_empty() {
            anyhow::bail!(
                "LoRA adapter source for alias `{}` must not be empty",
                adapter.alias
            );
        }
        if adapter.revision().is_empty() {
            anyhow::bail!(
                "LoRA adapter revision for alias `{}` must not be empty",
                adapter.alias
            );
        }
        if adapter
            .base_model_name
            .as_deref()
            .is_some_and(|model| model.trim().is_empty())
        {
            anyhow::bail!(
                "LoRA adapter `{}` has an empty base_model_name",
                adapter.alias
            );
        }
        if !aliases.insert(alias) {
            anyhow::bail!(
                "LoRA adapter alias `{}` is specified more than once",
                adapter.alias
            );
        }
    }
    if let Some(config) = runtime_config.filter(|config| aliases.len() > config.max_adapters) {
        anyhow::bail!(
            "LoRA adapter preload count {} exceeds the configured maximum {}",
            aliases.len(),
            config.max_adapters
        );
    }
    Ok(())
}

pub use crate::kv_cache::{Cache, EitherCache, KvCache, LayerCaches, NormalCache};

pub(crate) use crate::kv_cache::RECURRENT_GRAPH_PAD_SLOTS;
const AUTO_RECURRENT_KV_FLOOR_FRACTION: f64 = 0.05;
const BYTES_PER_MIB: usize = 1024 * 1024;

#[derive(Clone, Copy)]
struct RecurrentCheckpointBudget {
    requested_lanes: usize,
    capacity: usize,
    snapshot_bytes: usize,
    current_capacity: usize,
    current_lanes: usize,
    memory_total: usize,
    memory_available: usize,
    allocation_available: usize,
    future_reserved_bytes: usize,
    kv_floor_bytes: usize,
    memory_utilization: Option<f32>,
}

fn effective_recurrent_checkpoint_lanes(requested: usize, supported: bool) -> usize {
    if supported { requested } else { 1 }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn automatic_recurrent_checkpoint_lane_budget(budget: RecurrentCheckpointBudget) -> Result<usize> {
    let current_layout_bytes = budget
        .snapshot_bytes
        .checked_mul(budget.current_capacity)
        .and_then(|bytes| bytes.checked_mul(budget.current_lanes))
        .ok_or_else(|| anyhow::anyhow!("current recurrent layout size overflow"))?;
    let capacity = budget.capacity.max(budget.current_capacity);
    let bytes_per_lane = budget
        .snapshot_bytes
        .checked_mul(capacity)
        .ok_or_else(|| anyhow::anyhow!("recurrent checkpoint lane size overflow"))?;
    if bytes_per_lane == 0 {
        anyhow::bail!("recurrent checkpoint lane size must be nonzero");
    }
    let steady_layout_budget = match budget.memory_utilization {
        Some(fraction) => {
            let target_used = (budget.memory_total as f64 * f64::from(fraction)) as usize;
            let current_used_without_layout = budget
                .memory_total
                .saturating_sub(budget.memory_available)
                .saturating_sub(current_layout_bytes);
            target_used
                .saturating_sub(current_used_without_layout)
                .saturating_sub(budget.future_reserved_bytes)
                .saturating_sub(budget.kv_floor_bytes)
        }
        None => budget
            .allocation_available
            .saturating_add(current_layout_bytes)
            .saturating_sub(budget.future_reserved_bytes)
            .saturating_sub(budget.kv_floor_bytes),
    };
    let steady_lanes = steady_layout_budget / bytes_per_lane;
    let allocation_lanes = budget.allocation_available / bytes_per_lane;
    let checkpoint_lanes = budget
        .requested_lanes
        .min(steady_lanes)
        .min(allocation_lanes);
    Ok(if checkpoint_lanes >= 2 {
        checkpoint_lanes
    } else {
        1
    })
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn recurrent_kv_floor_bytes(
    memory_config: MemoryGpuConfig,
    memory_total: usize,
    kv_bytes_per_token: usize,
) -> Result<usize> {
    let bytes = match memory_config {
        MemoryGpuConfig::MbAmount(mb) => mb
            .checked_mul(BYTES_PER_MIB)
            .ok_or_else(|| anyhow::anyhow!("PagedAttention memory size overflow"))?,
        MemoryGpuConfig::BestEffortMbAmount { target_mb, min_mb } => {
            let mb = min_mb.unwrap_or_else(|| {
                ((memory_total as f64 * AUTO_RECURRENT_KV_FLOOR_FRACTION) as usize / BYTES_PER_MIB)
                    .min(target_mb)
            });
            mb.checked_mul(BYTES_PER_MIB)
                .ok_or_else(|| anyhow::anyhow!("PagedAttention memory size overflow"))?
        }
        MemoryGpuConfig::Utilization(fraction) => {
            let floor_fraction = f64::from(fraction).min(AUTO_RECURRENT_KV_FLOOR_FRACTION);
            (memory_total as f64 * floor_fraction) as usize
        }
        MemoryGpuConfig::ContextSize(tokens) => {
            let bytes = tokens
                .checked_mul(kv_bytes_per_token)
                .ok_or_else(|| anyhow::anyhow!("PagedAttention context memory size overflow"))?;
            bytes
                .div_ceil(BYTES_PER_MIB)
                .checked_mul(BYTES_PER_MIB)
                .ok_or_else(|| anyhow::anyhow!("PagedAttention context memory size overflow"))?
        }
    };
    Ok(bytes)
}

fn paged_kv_bytes_per_token(
    paged_attn_config: PagedAttentionConfig,
    dtype: DType,
    model_config: &dyn ModelConfigLike,
) -> Result<usize> {
    paged_attn_config
        .cache_type
        .to_dtype(dtype)
        .size_in_bytes()
        .checked_mul(model_config.total_kv_cache_elements_per_token())
        .ok_or_else(|| anyhow::anyhow!("PagedAttention token memory size overflow"))
}

fn automatic_recurrent_checkpoint_lanes(
    cache: &EitherCache,
    paged_attn_config: PagedAttentionConfig,
    primary_device: &Device,
    capacity: usize,
    kv_bytes_per_token: usize,
) -> Result<usize> {
    let hybrid = cache.hybrid();
    let snapshot_bytes_by_device = hybrid.recurrent_snapshot_bytes_by_device()?;
    let recurrent_devices = hybrid.recurrent_devices();
    let current_capacity = hybrid.recurrent_capacity();
    let current_lanes = hybrid.checkpoint_lanes();
    drop(hybrid);
    let reservations =
        paged_attention_memory_reservations(cache, paged_attn_config, primary_device)?;
    let mut selected_lanes = paged_attn_config.recurrent_checkpoint_lanes;
    for device in recurrent_devices {
        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            crate::utils::memory_usage::MemoryUsage.trim_cuda_memory_pool(&device, 0)?;
        }
        let memory = crate::utils::memory_usage::MemoryUsage.query(&device)?;
        let snapshot_bytes = snapshot_bytes_by_device
            .get(&device.location())
            .copied()
            .ok_or_else(|| {
                candle_core::Error::msg(
                    "recurrent device is missing from the snapshot memory inventory",
                )
            })?;
        let future_reserved_bytes = if device.same_device(primary_device) {
            reservations.primary_device_bytes
        } else {
            reservations.secondary_device_bytes
        };
        let kv_floor_bytes = recurrent_kv_floor_bytes(
            paged_attn_config.mem_gpu,
            memory.total(),
            kv_bytes_per_token,
        )?;
        selected_lanes = selected_lanes.min(automatic_recurrent_checkpoint_lane_budget(
            RecurrentCheckpointBudget {
                requested_lanes: paged_attn_config.recurrent_checkpoint_lanes,
                capacity,
                snapshot_bytes,
                current_capacity,
                current_lanes,
                memory_total: memory.total(),
                memory_available: memory.available(),
                allocation_available: crate::paged_attention::device_memory_cap(
                    memory.available(),
                    &device,
                ),
                future_reserved_bytes,
                kv_floor_bytes,
                memory_utilization: match paged_attn_config.mem_gpu {
                    MemoryGpuConfig::Utilization(fraction) => Some(fraction),
                    _ => None,
                },
            },
        )?);
    }
    Ok(selected_lanes)
}

/// A model whose recurrent state pools are reserved before any KV blocks are sized.
pub(crate) struct RecurrentReservation<'a> {
    pub target: &'a dyn inference_nn::speculative::SpeculativeTargetMixin,
    pub cache: &'a EitherCache,
    pub paged_attn_config: Option<PagedAttentionConfig>,
    pub dtype: DType,
    pub model_config: &'a dyn ModelConfigLike,
    pub device: &'a Device,
}

impl RecurrentReservation<'_> {
    pub fn reserve(self, mapper: &dyn DeviceMapper) -> Result<()> {
        let checkpoints = self.target.supports_recurrent_speculative_checkpoints();
        let transitions = self.target.supports_recurrent_speculative_transitions();
        let mut grew = self
            .paged_attn_config
            .map(|config| {
                let kv_bytes_per_token =
                    paged_kv_bytes_per_token(config, self.dtype, self.model_config)?;
                reserve_recurrent_serving_capacity(
                    self.cache,
                    config,
                    checkpoints,
                    transitions,
                    self.device,
                    kv_bytes_per_token,
                )
            })
            .transpose()?
            .unwrap_or(false);
        if transitions && uses_recurrent_transition_log(self.cache) {
            grew = self
                .target
                .reserve_recurrent_speculative_transition_storage()?
                || grew;
        }
        grew = self.target.reserve_recurrent_decode_deferred_storage()? || grew;
        #[cfg(feature = "cuda")]
        if grew {
            synchronize_cuda_contexts(self.device, mapper)?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = (grew, mapper);
        Ok(())
    }
}

/// The number of layers a model's cache holds.
pub(crate) fn cache_layer_count(cache: &EitherCache) -> usize {
    match cache {
        EitherCache::Full(full) => full.lock().len(),
        EitherCache::Normal(normal) => normal.lock().unwrap().0.len(),
        EitherCache::Hybrid(hybrid) => hybrid.lock().unwrap().num_layers(),
    }
}

fn reserve_recurrent_serving_capacity(
    cache: &EitherCache,
    paged_attn_config: PagedAttentionConfig,
    recurrent_checkpoints_supported: bool,
    recurrent_transitions_supported: bool,
    primary_device: &Device,
    kv_bytes_per_token: usize,
) -> Result<bool> {
    if !cache.is_hybrid() {
        return Ok(false);
    }
    let requested_lanes = effective_recurrent_checkpoint_lanes(
        paged_attn_config.recurrent_checkpoint_lanes,
        recurrent_checkpoints_supported || recurrent_transitions_supported,
    );
    let transition_log = recurrent_transitions_supported
        && requested_lanes > 1
        && requested_lanes <= crate::cuda::gdn::GDN_SPEC_FUSED_MAX_TOKENS;
    let Some(serving_capacity) = paged_attn_config.serving_capacity else {
        return Ok(if transition_log {
            cache.hybrid().configure_transition_lanes(requested_lanes)?
        } else {
            cache.hybrid().configure_checkpoint_lanes(requested_lanes)?
        });
    };
    let capacity = serving_capacity
        .checked_add(RECURRENT_GRAPH_PAD_SLOTS)
        .ok_or_else(|| candle_core::Error::msg("recurrent serving capacity overflow"))?;
    let checkpoint_lanes = if !transition_log
        && paged_attn_config.recurrent_checkpoint_lanes_auto
        && requested_lanes > 1
    {
        automatic_recurrent_checkpoint_lanes(
            cache,
            paged_attn_config,
            primary_device,
            capacity,
            kv_bytes_per_token,
        )?
    } else {
        requested_lanes
    };
    if checkpoint_lanes != paged_attn_config.recurrent_checkpoint_lanes {
        tracing::info!(
            requested_lanes = paged_attn_config.recurrent_checkpoint_lanes,
            checkpoint_lanes,
            serving_capacity,
            "Adjusted recurrent speculative checkpoint depth for the serving memory budget"
        );
    }
    Ok(if transition_log {
        cache
            .hybrid()
            .reserve_recurrent_transition_layout(capacity, checkpoint_lanes)?
    } else {
        cache
            .hybrid()
            .reserve_recurrent_layout(capacity, checkpoint_lanes)?
    })
}

fn uses_recurrent_transition_log(cache: &EitherCache) -> bool {
    cache.is_hybrid() && cache.hybrid().uses_recurrent_transition_log()
}

fn add_recurrent_prefix_memory_reservations(
    mut reservations: CacheMemoryReservations,
    bytes_by_device: HashMap<DeviceLocation, usize>,
    primary_device: DeviceLocation,
    prefix_capacity: usize,
) -> Result<CacheMemoryReservations> {
    if prefix_capacity == 0 {
        return Ok(reservations);
    }
    let peak_snapshots = prefix_capacity
        .checked_add(1)
        .ok_or_else(|| candle_core::Error::msg("recurrent prefix capacity overflow"))?;
    let mut secondary_prefix_bytes = 0usize;
    for (device, bytes_per_snapshot) in bytes_by_device {
        let bytes = bytes_per_snapshot
            .checked_mul(peak_snapshots)
            .ok_or_else(|| candle_core::Error::msg("recurrent prefix reservation overflow"))?;
        if device == primary_device {
            reservations.primary_device_bytes = reservations
                .primary_device_bytes
                .checked_add(bytes)
                .ok_or_else(|| candle_core::Error::msg("recurrent prefix reservation overflow"))?;
        } else {
            secondary_prefix_bytes = secondary_prefix_bytes.max(bytes);
        }
    }
    reservations.secondary_device_bytes = reservations
        .secondary_device_bytes
        .checked_add(secondary_prefix_bytes)
        .ok_or_else(|| candle_core::Error::msg("recurrent prefix reservation overflow"))?;
    Ok(reservations)
}

fn paged_attention_memory_reservations(
    cache: &EitherCache,
    paged_attn_config: PagedAttentionConfig,
    primary_device: &Device,
) -> Result<CacheMemoryReservations> {
    let reservations = paged_attn_config
        .memory_reservations()
        .map_err(candle_core::Error::msg)?;
    if paged_attn_config.recurrent_prefix_capacity == 0 || !cache.is_hybrid() {
        return Ok(reservations);
    }
    let bytes_by_device = cache.hybrid().recurrent_snapshot_bytes_by_device()?;
    add_recurrent_prefix_memory_reservations(
        reservations,
        bytes_by_device,
        primary_device.location(),
        paged_attn_config.recurrent_prefix_capacity,
    )
}

pub struct GeneralMetadata {
    pub max_seq_len: usize,
    /// Only None if it doesn't make sense for the model
    pub llg_factory: Option<Arc<llguidance::ParserFactory>>,
    pub no_kv_cache: bool,
    pub no_prefix_cache: bool,
    pub num_hidden_layers: usize,
    pub eos_tok: Vec<u32>,
    pub kind: ModelKind,
    // TODO: Replace is_xlora queries to check via kind instead:
    pub is_xlora: bool,
    pub activation_dtype: DType,
    pub sliding_window: Option<usize>,
    // PagedAttention stuff
    pub cache_config: Option<CacheConfig>,
    pub cache_engine: Option<CacheEngine>,
    pub model_metadata: Option<Arc<dyn ModelConfigLike + Send + Sync>>,
    pub modalities: Modalities,
    // UQFF writes force the whole model onto CPU, so the pipeline is not servable afterwards.
    pub loaded_for_uqff_write: bool,
}

impl GeneralMetadata {
    pub fn tok_env(&self) -> Option<TokEnv> {
        self.llg_factory.as_ref().map(|f| f.tok_env().clone())
    }
}

#[derive(Clone, Copy)]
pub enum CacheInstruction {
    In,
    Out,
    /// load_preallocated_cache means to load the preallocated cache, if applicable.
    Reset {
        load_preallocated_cache: bool,
        reset_non_granular: bool,
    },
    Nothing,
}

pub trait PreProcessingMixin: MetadataMixin {
    fn get_processor(&self) -> Arc<dyn Processor> {
        Arc::new(BasicProcessor)
    }
    /// Only None if it doesnt make sense for the model
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>>;
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>>;
}

pub trait IsqPipelineMixin {
    fn re_isq_model(&mut self, dtype: IsqType) -> Result<()>;

    /// Start collecting activation statistics from live traffic on every ISQ-tracked layer.
    fn begin_calibration(&mut self) -> Result<()> {
        anyhow::bail!("This pipeline does not support online calibration.")
    }

    fn calibration_status(&self) -> Result<isq_flow::CalibrationStatus> {
        anyhow::bail!("This pipeline does not support online calibration.")
    }

    /// Requantize with the collected statistics and swap the layers into the live model.
    fn apply_calibration(
        &mut self,
        _save_cimatrix: Option<std::path::PathBuf>,
    ) -> Result<isq_flow::CalibrationStatus> {
        anyhow::bail!("This pipeline does not support online calibration.")
    }
}

pub trait CacheManagerMixin {
    /// Clone the cache FROM the sequences' cache TO the model cache. Only called for completion seqs.
    /// It is not a guarantee that this will be called for each completion step.
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) -> candle_core::Result<()>;
    /// Clone the cache FROM the model cache TO the sequences. Called for prompt and completion seqs.
    /// It is not a guarantee that this will be called for each step.
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]);
    /// Set the model cache to all None. Only called for prompt seqs.
    /// It is not a guarantee that this will be called for each prompt step.
    /// This may also reset the non granular state if applicable.
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) -> candle_core::Result<()>;
    fn cache(&self) -> &EitherCache;
}

pub trait MetadataMixin {
    fn device(&self) -> Device;
    /// Only None if it doesnt make sense for the model
    fn tokenizer(&self) -> Option<Arc<Tokenizer>>;
    fn name(&self) -> String;
    fn reset_non_granular_state(&self);
    /// Destroy decode graphs at teardown, while the engine thread's cuTile modules are still loaded.
    fn cleanup_cuda_graphs(&self) {}
    /// Evict least-recently-used decode graphs without disturbing recurrent state.
    fn reclaim_cuda_graph_memory(&self, _max_entries: usize) -> usize {
        0
    }
    /// Capture the decode graphs for the common batch sizes up front, so live requests never pay for
    /// an eager step plus a capture when the batch composition changes.
    fn precapture_cuda_decode_graphs(&self, _ctx: &DecodeGraphPrecaptureCtx) {}
    fn get_metadata(&self) -> Arc<GeneralMetadata>;
    fn generation_defaults(&self) -> Option<crate::ModelGenerationDefaults> {
        None
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper>;
    fn execution_devices(&self) -> Vec<Device> {
        let primary = self.device();
        let mut devices = self
            .device_mapper()
            .map(DeviceMapper::get_unique_devices)
            .unwrap_or_default();
        if !devices.iter().any(|device| device.same_device(&primary)) {
            devices.push(primary);
        }
        devices
    }
}

/// Implemented by the base model of an AnyMoe.
pub trait AnyMoePipelineMixin {
    /// Get vars for each gating layer
    fn amoe_layer_vars(&self) -> Vec<Vec<Var>> {
        unreachable!()
    }
    fn amoe_finish_training(&mut self, _gate_model_id: Option<String>) -> candle_core::Result<()> {
        unreachable!()
    }
    fn amoe_base_model_trainable_params(&self) -> usize {
        unreachable!()
    }
    fn amoe_supported(&self) -> bool {
        false
    }
    /// Per-layer cached outputs.
    fn amoe_take_cached_gating_outputs(&mut self) -> Vec<Tensor> {
        unreachable!()
    }
    /// Inject the MoE layers
    #[allow(clippy::too_many_arguments)]
    fn amoe_create_layers(
        &mut self,
        _model_ids: Vec<String>,
        _token: &TokenSource,
        _revision: Option<String>,
        _match_regex: &str,
        _config: AnyMoeConfig,
        _dtype: DType,
        _dev: &Device,
        (_prefix, _mlp): (String, String),
        _layers: Vec<usize>,
        _expert_type: AnyMoeExpertType,
        _silent: bool,
        _gate_model_id: Option<String>,
    ) -> candle_core::Result<()> {
        unreachable!()
    }
    /// Pre-train the gating layers
    #[allow(clippy::too_many_arguments)]
    fn amoe_pre_train(
        &self,
        _inputs: AnyMoeTrainingInputs,
        (_prefix, _mlp): (String, String),
        _model_ids: Vec<String>,
        _token: TokenSource,
        _revision: Option<String>,
        _layers: Vec<usize>,
        _silent: bool,
    ) -> Result<Option<AnyMoeTrainingResult>, candle_core::Error> {
        unreachable!()
    }
}

/// Category of the model. This can also be used to extract model-category specific tools,
/// such as the multimodal model prompt prefixer.
#[derive(Clone)]
pub enum ModelCategory {
    Text,
    Multimodal {
        prefixer: Arc<dyn MultimodalPromptPrefixer>,
        video_sampling: crate::VideoFrameSampling,
    },
    Diffusion,
    Audio,
    Speech,
    Embedding,
}

impl std::fmt::Debug for ModelCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelCategory::Text => write!(f, "ModelCategory::Text"),
            ModelCategory::Multimodal { .. } => {
                write!(f, "ModelCategory::Multimodal {{ prefixer: .. }}")
            }
            ModelCategory::Diffusion => write!(f, "ModelCategory::Diffusion"),
            ModelCategory::Audio => write!(f, "ModelCategory::Audio"),
            ModelCategory::Speech => write!(f, "ModelCategory::Speech"),
            ModelCategory::Embedding => write!(f, "ModelCategory::Embedding"),
        }
    }
}

impl PartialEq for ModelCategory {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Text, Self::Text) => true,
            (Self::Multimodal { .. }, Self::Multimodal { .. }) => true,
            (Self::Audio, Self::Audio) => true,
            (Self::Speech, Self::Speech) => true,
            (Self::Diffusion, Self::Diffusion) => true,
            (Self::Embedding, Self::Embedding) => true,
            (
                Self::Text
                | Self::Multimodal { .. }
                | Self::Diffusion
                | Self::Audio
                | Self::Speech
                | Self::Embedding,
                _,
            ) => false,
        }
    }
}

/// Paged-attention facts the engine knows once the KV cache exists, needed to fabricate a decode step.
#[derive(Clone, Debug)]
pub struct DecodeGraphPrecaptureCtx {
    pub block_size: usize,
    pub max_batch_size: usize,
    pub max_paged_context_len: usize,
    pub attention_backend: AttentionBackendKind,
    pub sliding_window: Option<usize>,
    pub num_kv_heads: usize,
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub enum CacheBackendMetadata {
    DefaultInstructions {
        pre_op: CacheInstruction,
        post_op: CacheInstruction,
    },
    PagedAttention {
        metadata: PagedAttentionMeta,
    },
}

#[derive(Clone, Debug)]
pub enum ForwardInputsResult {
    RawLogits {
        logits: Tensor,
    },
    Embeddings {
        embeddings: Tensor,
    },
    CausalGeneration {
        logits: Tensor,
    },
    Image {
        images: Vec<DynamicImage>,
    },
    Speech {
        pcms: Vec<Arc<Vec<f32>>>,
        rates: Vec<usize>,
        channels: Vec<usize>,
    },
    BlockGeneration {
        token_blocks: Vec<Vec<u32>>,
        denoise_time: std::time::Duration,
    },
}

impl ForwardInputsResult {
    fn index_bs(&self, bs_idx: usize) -> candle_core::Result<Self> {
        match self {
            Self::CausalGeneration { logits } => Ok(Self::CausalGeneration {
                logits: logits.i(bs_idx)?,
            }),
            Self::Embeddings { embeddings } => Ok(Self::Embeddings {
                embeddings: embeddings.i(bs_idx)?,
            }),
            Self::RawLogits { logits } => Ok(Self::RawLogits {
                logits: logits.i(bs_idx)?,
            }),
            Self::Image { images } => Ok(Self::Image {
                images: vec![images[bs_idx].clone()],
            }),
            Self::Speech {
                pcms,
                rates,
                channels,
            } => Ok(Self::Speech {
                pcms: vec![pcms[bs_idx].clone()],
                rates: vec![rates[bs_idx]],
                channels: vec![channels[bs_idx]],
            }),
            Self::BlockGeneration {
                token_blocks,
                denoise_time,
            } => Ok(Self::BlockGeneration {
                token_blocks: vec![token_blocks[bs_idx].clone()],
                denoise_time: *denoise_time,
            }),
        }
    }

    fn to_device(&self, device: &Device) -> candle_core::Result<Self> {
        match self {
            Self::CausalGeneration { logits } => Ok(Self::CausalGeneration {
                logits: logits.to_device(device)?,
            }),
            Self::RawLogits { logits } => Ok(Self::RawLogits {
                logits: logits.to_device(device)?,
            }),
            Self::Embeddings { embeddings } => Ok(Self::Embeddings {
                embeddings: embeddings.to_device(device)?,
            }),
            Self::Image { .. } => Ok(self.clone()),
            Self::Speech { .. } => Ok(self.clone()),
            Self::BlockGeneration { .. } => Ok(self.clone()),
        }
    }

    fn into_cpu_for_batch(
        self,
        batch_size: usize,
        preserve_causal_generation: bool,
    ) -> candle_core::Result<Self> {
        if batch_size <= 1
            || preserve_causal_generation && matches!(&self, Self::CausalGeneration { .. })
        {
            return Ok(self);
        }
        self.to_device(&Device::Cpu)
    }
}

#[doc(hidden)]
pub struct ForwardStepResult {
    pub output: ForwardInputsResult,
    #[cfg(feature = "cuda")]
    pub(crate) cuda_decode: Option<cuda_graph::CudaDecodeGraphLaunch>,
}

impl ForwardStepResult {
    pub fn eager(output: ForwardInputsResult) -> Self {
        Self {
            output,
            #[cfg(feature = "cuda")]
            cuda_decode: None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_decode(
        output: ForwardInputsResult,
        launch: Option<cuda_graph::CudaDecodeGraphLaunch>,
    ) -> Self {
        Self {
            output,
            cuda_decode: launch,
        }
    }
}

/// One sequence's slice of a prompt chunk the target just processed, for proposers that keep their
/// own KV cache; `tokens` is the full prompt so the shifted next token is available past `range`.
pub struct SpeculativePromptRow {
    pub seq_idx: usize,
    pub range: (usize, usize),
    pub tokens: Vec<u32>,
}

pub struct SpeculativePromptChunk {
    /// In forward-batch order (row `i` of the batch is `rows[i].seq_idx`)
    pub rows: Vec<SpeculativePromptRow>,
    pub is_final_prompt_chunk: bool,
}

fn should_sample_step(
    is_prompt: bool,
    scheduler_visible_prompt_step: bool,
    is_final_prompt_chunk: bool,
) -> bool {
    !is_prompt || !scheduler_visible_prompt_step || is_final_prompt_chunk
}

fn should_try_speculative_sampling(
    is_prompt: bool,
    scheduler_visible_prompt_step: bool,
    is_final_prompt_chunk: bool,
    return_raw_logits: bool,
    supports_prompt_bootstrap: bool,
) -> bool {
    !return_raw_logits
        && (!is_prompt || supports_prompt_bootstrap)
        && should_sample_step(
            is_prompt,
            scheduler_visible_prompt_step,
            is_final_prompt_chunk,
        )
}

fn prompt_chunk_is_final(
    scheduler_visible_prompt_step: bool,
    scheduler_visible_prompt_is_final: bool,
    planned_final_prompt_chunk: bool,
) -> bool {
    if scheduler_visible_prompt_step {
        scheduler_visible_prompt_is_final
    } else {
        planned_final_prompt_chunk
    }
}

fn next_pipeline_prompt_chunk_group(
    plan_indices: &[usize],
    chunk_plans: &[Vec<PromptChunkPlan>],
    requires_uniform_prompt_batch: bool,
    supports_packed_prefill: bool,
    hybrid_recurrent: bool,
) -> Option<(Vec<usize>, MultimodalAttentionPolicy, bool)> {
    let planned_final_prompt_chunk = plan_indices
        .iter()
        .zip(chunk_plans)
        .find_map(|(&plan_idx, plan)| plan.get(plan_idx).map(|_| plan_idx + 1 == plan.len()))?;
    let require_uniform_query_len = requires_uniform_prompt_batch
        || !supports_packed_prefill
        || (hybrid_recurrent && !planned_final_prompt_chunk);
    next_prompt_chunk_group(plan_indices, chunk_plans, require_uniform_query_len)
}

pub trait Pipeline:
    Send
    + Sync
    + PreProcessingMixin
    + IsqPipelineMixin
    + CacheManagerMixin
    + MetadataMixin
    + AnyMoePipelineMixin
{
    fn requires_uniform_prompt_batch(&self) -> bool {
        true
    }

    fn requires_uniform_completion_batch(&self) -> bool {
        true
    }

    fn requires_uniform_media_batch(&self) -> bool {
        false
    }

    fn supports_batched_cuda_sampling(&self) -> bool {
        false
    }

    fn supports_packed_prefill(&self) -> bool {
        false
    }

    fn adapter_runtime(&self) -> Option<Arc<crate::DynamicLoraRuntime>> {
        None
    }

    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error>;

    #[doc(hidden)]
    fn forward_step(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardStepResult, candle_core::Error> {
        self.forward_inputs(inputs, return_raw_logits)
            .map(ForwardStepResult::eager)
    }

    #[cfg(feature = "cuda")]
    #[doc(hidden)]
    fn replay_cuda_decode_one_token(
        &mut self,
        _launch: CudaDecodeGraphLaunch,
    ) -> Result<Option<ForwardStepResult>, candle_core::Error> {
        Ok(None)
    }

    fn attach_speculative(
        &mut self,
        _config: crate::speculative::SpeculativeConfig,
    ) -> Result<(), candle_core::Error> {
        candle_core::bail!("This pipeline does not support speculative decoding attachment.")
    }

    #[doc(hidden)]
    fn attach_speculative_with_runtime(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
        _runtime: crate::speculative::MtpRuntimeConfig,
    ) -> Result<(), candle_core::Error> {
        self.attach_speculative(config)
    }

    fn release_speculative_sequences(&mut self, _seq_ids: &[usize]) -> candle_core::Result<()> {
        Ok(())
    }

    fn flush_recurrent_speculative_transitions(
        &self,
        _seq_ids: &[usize],
    ) -> candle_core::Result<()> {
        Ok(())
    }

    fn supports_speculative_prompt_bootstrap(&self) -> bool {
        false
    }

    fn speculative_prefix_replay(&self) -> crate::speculative::SpeculativePrefixReplay {
        crate::speculative::SpeculativePrefixReplay::NotRequired
    }

    fn supports_paged_auxiliary_prefix_state(&self) -> bool {
        false
    }

    fn speculative_prefix_checkpoint_policy(
        &self,
    ) -> crate::speculative::SpeculativePrefixCheckpointPolicy {
        crate::speculative::SpeculativePrefixCheckpointPolicy::new(
            self.speculative_prefix_replay(),
            self.supports_paged_auxiliary_prefix_state(),
        )
    }

    fn capture_paged_auxiliary_prefix_state(
        &mut self,
        _sequence_id: usize,
        _cached_tokens: usize,
    ) -> Result<Option<Arc<dyn PagedAuxiliaryPrefixState>>, candle_core::Error> {
        Ok(None)
    }

    fn restore_paged_auxiliary_prefix_state(
        &mut self,
        _sequence_id: usize,
        _cached_tokens: usize,
        _state: &dyn PagedAuxiliaryPrefixState,
    ) -> Result<(), candle_core::Error> {
        candle_core::bail!("This pipeline does not support auxiliary paged prefix state.")
    }

    /// Called after a prompt chunk forward so a speculative proposer with its own KV cache can
    /// process the chunk. Default: nothing to do.
    fn speculative_prompt_chunk(
        &mut self,
        _seqs: &[&mut Sequence],
        _chunk: &SpeculativePromptChunk,
        _metadata: &PagedAttentionMeta,
    ) -> Result<(), candle_core::Error> {
        Ok(())
    }

    /// Append pre-sampled token blocks (block-diffusion canvases) to the sequences via the
    /// standard per-token finalize path. Overridden by pipelines whose models emit
    /// `ForwardInputsResult::BlockGeneration`.
    fn sample_block_gen<'a>(
        &'a self,
        _input_seqs: &'a mut [&mut Sequence],
        _token_blocks: Vec<Vec<u32>>,
        _denoise_times: Vec<std::time::Duration>,
        _prefix_cacher: &'a mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
    ) -> BoxFuture<'a, Result<(), candle_core::Error>> {
        Box::pin(std::future::ready(Err(candle_core::Error::Msg(
            "This pipeline does not support block generation.".to_string(),
        )
        .bt())))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_sample_speculative_causal_gen<'a>(
        &'a mut self,
        _input_seqs: &'a mut [&mut Sequence],
        _logits: &'a [Tensor],
        _batched_logits: Option<&'a Tensor>,
        _prefix_cacher: &'a mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        _metadata: Option<PagedAttentionMeta>,
        _logger: &'a IntervalLogger,
    ) -> BoxFuture<'a, Result<bool, candle_core::Error>> {
        Box::pin(std::future::ready(Ok(false)))
    }

    fn snapshot_paged_recurrent_prefix(
        &mut self,
        seq: &Sequence,
        prefix_cacher: &mut PrefixCacheManagerV2,
        block_size: usize,
        cached_tokens: usize,
    ) -> Result<(), candle_core::Error> {
        if cached_tokens == 0
            || !cached_tokens.is_multiple_of(block_size)
            || !self.cache().is_hybrid()
            || !prefix_cacher.accepts_paged_recurrent_prefix()
        {
            return Ok(());
        }
        let Some(slot_idx) = seq.recurrent_state_idx() else {
            return Ok(());
        };

        self.flush_recurrent_speculative_transitions(&[*seq.id()])?;

        let snapshots = self
            .cache()
            .hybrid()
            .snapshot_recurrent_state(*seq.id(), slot_idx)?;
        if snapshots.is_empty() {
            return Ok(());
        }
        let auxiliary = if self
            .speculative_prefix_checkpoint_policy()
            .uses_auxiliary_state(crate::scheduler::modality_signature(seq))
        {
            self.capture_paged_auxiliary_prefix_state(*seq.id(), cached_tokens)?
        } else {
            None
        };
        let adapter_key = adapter_generation_key(seq.adapter_generation());
        let block_hashes = compute_block_hashes(
            seq.get_toks(),
            block_size,
            seq.mm_features(),
            adapter_key.as_slice(),
        );
        let n_blocks = cached_tokens / block_size;
        if block_hashes.len() >= n_blocks {
            let owner = *block_hashes
                .last()
                .expect("recurrent prefix owner requires a full block");
            prefix_cacher.add_paged_recurrent_prefix(
                owner,
                block_hashes[..n_blocks].to_vec(),
                snapshots,
                auxiliary,
            );
        }
        Ok(())
    }

    fn try_sample_causal_gen_batched<'a>(
        &'a self,
        _seqs: &'a mut [&mut Sequence],
        _logits: &'a Tensor,
        _prefix_cacher: &'a mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> BoxFuture<'a, Result<bool, candle_core::Error>> {
        Box::pin(std::future::ready(Ok(false)))
    }

    fn sample_causal_gen<'a>(
        &'a self,
        seqs: &'a mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &'a mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> BoxFuture<'a, Result<(), candle_core::Error>>;

    fn category(&self) -> ModelCategory;

    /// Return encoder cache hit/miss counters (hits, misses) if this pipeline has an encoder cache.
    fn encoder_cache_counters(&self) -> Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)> {
        None
    }
}

// Not trait defaults: those compile once per pipeline type.
impl dyn Pipeline {
    /// Returns the total of model execution time.
    #[allow(clippy::too_many_arguments)]
    pub async fn step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        return_raw_logits: bool,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        backend_metadata: CacheBackendMetadata,
        logger: &IntervalLogger,
    ) -> Result<Duration, candle_core::Error> {
        let completion = self
            .submit_step(
                input_seqs,
                is_prompt,
                return_raw_logits,
                prefix_cacher,
                disable_eos_stop,
                rng,
                backend_metadata,
                logger,
                StepLookahead::Disabled,
            )
            .await?
            .into_ready()
            .expect("lookahead-disabled pipeline step must complete eagerly");
        Ok(completion.duration())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
    };

    use crate::model::decode_positions_tensor;

    use super::{
        CacheMemoryReservations, ModelForwardContext, RecurrentCheckpointBudget,
        add_recurrent_prefix_memory_reservations, automatic_recurrent_checkpoint_lane_budget,
        effective_recurrent_checkpoint_lanes, next_pipeline_prompt_chunk_group,
        paged_attention_memory_reservations, prompt_chunk_is_final, recurrent_batch_kind_for_input,
        recurrent_kv_floor_bytes, reserve_recurrent_serving_capacity, resolve_lora_execution,
        should_sample_step, should_try_speculative_sampling,
    };
    use crate::gdn::RecurrentBatchKind;
    use crate::model::{ForwardCache, LogitsSelection};
    use crate::{
        MemoryGpuConfig, MessageContent, PagedAttentionConfig, PagedCacheType,
        attention::FlashParams,
        kv_cache::{
            EitherCache, HybridCache, HybridCacheConfig, HybridLayerType, RecurrentLayerConfig,
            RecurrentStateSpec,
        },
        paged_attention::PagedAttentionInputMetadata,
        paged_attention::block_hash::MultimodalAttentionPolicy,
        pipeline::prompt_chunks::PromptChunkPlan,
    };
    use candle_core::{Device, DeviceLocation, Tensor};
    use either::Either;
    use indexmap::IndexMap;

    fn prompt_chunk(start: usize, end: usize) -> PromptChunkPlan {
        PromptChunkPlan {
            start,
            end,
            attention_policy: MultimodalAttentionPolicy::Causal,
        }
    }

    #[test]
    fn scheduler_visible_prefill_samples_only_the_final_chunk() {
        assert!(!should_sample_step(true, true, false));
        assert!(should_sample_step(true, true, true));
        assert!(should_sample_step(true, false, false));
        assert!(should_sample_step(false, true, false));
    }

    #[test]
    fn speculative_sampling_bootstraps_on_the_final_prompt_chunk() {
        assert!(!should_try_speculative_sampling(
            true, true, false, false, true
        ));
        assert!(should_try_speculative_sampling(
            true, true, true, false, true
        ));
        assert!(should_try_speculative_sampling(
            true, false, false, false, true
        ));
        assert!(!should_try_speculative_sampling(
            true, true, true, false, false
        ));
        assert!(should_try_speculative_sampling(
            false, true, false, false, false
        ));
        assert!(!should_try_speculative_sampling(
            true, true, true, true, true
        ));
    }

    #[test]
    fn scheduler_visible_prefill_preserves_nonfinal_proposer_state() {
        assert!(!prompt_chunk_is_final(true, false, true));
        assert!(prompt_chunk_is_final(true, true, false));
        assert!(prompt_chunk_is_final(false, false, true));
    }

    #[test]
    fn packed_hybrid_final_prompt_chunks_remain_ragged() {
        let plans = vec![
            vec![prompt_chunk(0, 128), prompt_chunk(128, 129)],
            vec![prompt_chunk(0, 128), prompt_chunk(128, 134)],
        ];

        assert_eq!(
            next_pipeline_prompt_chunk_group(&[1, 1], &plans, false, true, true),
            Some((vec![0, 1], MultimodalAttentionPolicy::Causal, true))
        );
    }

    #[test]
    fn hybrid_nonfinal_checkpoint_chunks_remain_uniform() {
        let plans = vec![
            vec![prompt_chunk(0, 64), prompt_chunk(64, 129)],
            vec![prompt_chunk(0, 128), prompt_chunk(128, 134)],
        ];

        assert_eq!(
            next_pipeline_prompt_chunk_group(&[0, 0], &plans, false, true, true),
            Some((vec![0], MultimodalAttentionPolicy::Causal, false))
        );
    }

    #[test]
    fn unsupported_packed_prefill_keeps_final_chunks_uniform() {
        let plans = vec![vec![prompt_chunk(0, 129)], vec![prompt_chunk(0, 134)]];

        assert_eq!(
            next_pipeline_prompt_chunk_group(&[0, 0], &plans, false, false, true),
            Some((vec![0], MultimodalAttentionPolicy::Causal, true))
        );
    }

    #[test]
    fn recurrent_batch_kind_tracks_staged_speculative_completions() {
        assert_eq!(
            recurrent_batch_kind_for_input(false, true),
            RecurrentBatchKind::SpeculativeDecode
        );
        assert_eq!(
            recurrent_batch_kind_for_input(false, false),
            RecurrentBatchKind::Decode
        );
        assert_eq!(
            recurrent_batch_kind_for_input(true, true),
            RecurrentBatchKind::Prefill
        );
    }

    #[test]
    fn unsupported_recurrent_checkpoint_placement_uses_one_lane() {
        assert_eq!(effective_recurrent_checkpoint_lanes(8, false), 1);
        assert_eq!(effective_recurrent_checkpoint_lanes(8, true), 8);
        assert_eq!(effective_recurrent_checkpoint_lanes(1, false), 1);
    }

    #[test]
    fn automatic_recurrent_checkpoint_depth_preserves_kv_budget() {
        const GIB: usize = 1024 * 1024 * 1024;
        const SNAPSHOT_BYTES: usize = 147 * 1024 * 1024 + 3 * 256 * 1024;

        let lanes = automatic_recurrent_checkpoint_lane_budget(RecurrentCheckpointBudget {
            requested_lanes: 8,
            capacity: 65,
            snapshot_bytes: SNAPSHOT_BYTES,
            current_capacity: 9,
            current_lanes: 1,
            memory_total: 100 * GIB,
            memory_available: 68 * GIB,
            allocation_available: 66 * GIB,
            future_reserved_bytes: 7 * GIB,
            kv_floor_bytes: 5 * GIB,
            memory_utilization: Some(0.85),
        })
        .unwrap();

        assert_eq!(lanes, 4);
    }

    #[test]
    fn automatic_recurrent_checkpoint_depth_keeps_requested_small_capacity() {
        const GIB: usize = 1024 * 1024 * 1024;
        const SNAPSHOT_BYTES: usize = 147 * 1024 * 1024 + 3 * 256 * 1024;

        let lanes = automatic_recurrent_checkpoint_lane_budget(RecurrentCheckpointBudget {
            requested_lanes: 8,
            capacity: 33,
            snapshot_bytes: SNAPSHOT_BYTES,
            current_capacity: 9,
            current_lanes: 1,
            memory_total: 100 * GIB,
            memory_available: 68 * GIB,
            allocation_available: 66 * GIB,
            future_reserved_bytes: 7 * GIB,
            kv_floor_bytes: 5 * GIB,
            memory_utilization: Some(0.85),
        })
        .unwrap();

        assert_eq!(lanes, 8);
    }

    #[test]
    fn automatic_recurrent_checkpoint_depth_uses_replay_when_none_fit() {
        const GIB: usize = 1024 * 1024 * 1024;
        const SNAPSHOT_BYTES: usize = 147 * 1024 * 1024 + 3 * 256 * 1024;

        let lanes = automatic_recurrent_checkpoint_lane_budget(RecurrentCheckpointBudget {
            requested_lanes: 8,
            capacity: 512,
            snapshot_bytes: SNAPSHOT_BYTES,
            current_capacity: 9,
            current_lanes: 1,
            memory_total: 100 * GIB,
            memory_available: 68 * GIB,
            allocation_available: 66 * GIB,
            future_reserved_bytes: 7 * GIB,
            kv_floor_bytes: 5 * GIB,
            memory_utilization: Some(0.85),
        })
        .unwrap();

        assert_eq!(lanes, 1);
    }

    #[test]
    fn automatic_recurrent_checkpoint_depth_uses_existing_minimum_capacity() {
        const GIB: usize = 1024 * 1024 * 1024;

        let lanes = automatic_recurrent_checkpoint_lane_budget(RecurrentCheckpointBudget {
            requested_lanes: 8,
            capacity: 2,
            snapshot_bytes: GIB,
            current_capacity: 9,
            current_lanes: 1,
            memory_total: 100 * GIB,
            memory_available: 15 * GIB,
            allocation_available: 15 * GIB,
            future_reserved_bytes: 0,
            kv_floor_bytes: 0,
            memory_utilization: None,
        })
        .unwrap();

        assert_eq!(lanes, 1);
    }

    #[test]
    fn recurrent_context_floor_preserves_requested_kv_bytes() {
        const MIB: usize = 1024 * 1024;

        let bytes = recurrent_kv_floor_bytes(
            MemoryGpuConfig::ContextSize(1_000_000),
            100 * 1024 * MIB,
            1024,
        )
        .unwrap();

        assert_eq!(bytes, 977 * MIB);
    }

    #[test]
    fn cpu_recurrent_reservation_does_not_allocate_checkpoint_lanes() {
        let hybrid = HybridCache::new(
            HybridCacheConfig {
                layer_types: vec![HybridLayerType::Recurrent],
                max_seq_len: 32,
                recurrent: RecurrentLayerConfig {
                    conv_dim: 8,
                    conv_width: 4,
                    state: RecurrentStateSpec::Gdn {
                        heads: 2,
                        key_dim: 4,
                        value_dim: 4,
                    },
                    recurrent_dtype: Some(candle_core::DType::F32),
                },
            },
            candle_core::DType::BF16,
            &[Device::Cpu],
        )
        .unwrap();
        let cache = EitherCache::Hybrid(Arc::new(Mutex::new(hybrid)));
        let config =
            PagedAttentionConfig::new(None, MemoryGpuConfig::MbAmount(1), PagedCacheType::Auto)
                .unwrap()
                .with_serving_capacity(16)
                .unwrap()
                .with_recurrent_checkpoint_lanes(8)
                .unwrap();

        reserve_recurrent_serving_capacity(&cache, config, false, false, &Device::Cpu, 1).unwrap();

        let cache = cache.hybrid();
        assert_eq!(cache.checkpoint_lanes(), 1);
        let pool = cache.get(0).unwrap().as_recurrent_pool().unwrap();
        assert_eq!(pool.capacity(), 17);
        assert_eq!(pool.physical_capacity(), 17);
    }

    #[test]
    fn recurrent_transition_reservation_keeps_one_physical_lane() {
        let hybrid = HybridCache::new(
            HybridCacheConfig {
                layer_types: vec![HybridLayerType::Recurrent],
                max_seq_len: 32,
                recurrent: RecurrentLayerConfig {
                    conv_dim: 8,
                    conv_width: 4,
                    state: RecurrentStateSpec::Gdn {
                        heads: 2,
                        key_dim: 4,
                        value_dim: 4,
                    },
                    recurrent_dtype: Some(candle_core::DType::F32),
                },
            },
            candle_core::DType::BF16,
            &[Device::Cpu],
        )
        .unwrap();
        let cache = EitherCache::Hybrid(Arc::new(Mutex::new(hybrid)));
        let config =
            PagedAttentionConfig::new(None, MemoryGpuConfig::MbAmount(1), PagedCacheType::Auto)
                .unwrap()
                .with_serving_capacity(64)
                .unwrap()
                .with_recurrent_checkpoint_lanes(8)
                .unwrap();

        reserve_recurrent_serving_capacity(&cache, config, true, true, &Device::Cpu, 1).unwrap();

        let cache = cache.hybrid();
        assert_eq!(cache.recurrent_capacity(), 65);
        assert_eq!(cache.checkpoint_lanes(), 8);
        assert_eq!(cache.physical_checkpoint_lanes(), 1);
        assert!(cache.uses_recurrent_transition_log());
        let pool = cache.get(0).unwrap().as_recurrent_pool().unwrap();
        assert_eq!(pool.capacity(), 65);
        assert_eq!(pool.physical_capacity(), 65);
    }

    #[test]
    fn one_lane_recurrent_reservation_omits_transition_log() {
        let hybrid = HybridCache::new(
            HybridCacheConfig {
                layer_types: vec![HybridLayerType::Recurrent],
                max_seq_len: 32,
                recurrent: RecurrentLayerConfig {
                    conv_dim: 8,
                    conv_width: 4,
                    state: RecurrentStateSpec::Gdn {
                        heads: 2,
                        key_dim: 4,
                        value_dim: 4,
                    },
                    recurrent_dtype: Some(candle_core::DType::F32),
                },
            },
            candle_core::DType::BF16,
            &[Device::Cpu],
        )
        .unwrap();
        let cache = EitherCache::Hybrid(Arc::new(Mutex::new(hybrid)));
        let config =
            PagedAttentionConfig::new(None, MemoryGpuConfig::MbAmount(1), PagedCacheType::Auto)
                .unwrap()
                .with_serving_capacity(64)
                .unwrap()
                .with_recurrent_checkpoint_lanes(1)
                .unwrap();

        reserve_recurrent_serving_capacity(&cache, config, true, true, &Device::Cpu, 1).unwrap();

        let cache = cache.hybrid();
        assert_eq!(cache.recurrent_capacity(), 65);
        assert_eq!(cache.checkpoint_lanes(), 1);
        assert!(!cache.uses_recurrent_transition_log());
        let pool = cache.get(0).unwrap().as_recurrent_pool().unwrap();
        assert_eq!(pool.capacity(), 65);
        assert_eq!(pool.physical_capacity(), 65);
    }

    #[test]
    fn recurrent_prefix_reservation_includes_staging_and_device_baselines() {
        let reservations = add_recurrent_prefix_memory_reservations(
            CacheMemoryReservations {
                primary_device_bytes: 100,
                secondary_device_bytes: 50,
            },
            HashMap::from([
                (DeviceLocation::Cpu, 10),
                (DeviceLocation::Cuda { gpu_id: 0 }, 20),
                (DeviceLocation::Cuda { gpu_id: 1 }, 30),
            ]),
            DeviceLocation::Cpu,
            2,
        )
        .unwrap();

        assert_eq!(reservations.primary_device_bytes, 130);
        assert_eq!(reservations.secondary_device_bytes, 140);
    }

    #[test]
    fn loaded_hybrid_cache_drives_recurrent_prefix_reservation() {
        let hybrid = HybridCache::new(
            HybridCacheConfig {
                layer_types: vec![HybridLayerType::Recurrent],
                max_seq_len: 32,
                recurrent: RecurrentLayerConfig {
                    conv_dim: 8,
                    conv_width: 4,
                    state: RecurrentStateSpec::Gdn {
                        heads: 2,
                        key_dim: 4,
                        value_dim: 4,
                    },
                    recurrent_dtype: Some(candle_core::DType::F32),
                },
            },
            candle_core::DType::BF16,
            &[Device::Cpu],
        )
        .unwrap();
        let cache = EitherCache::Hybrid(Arc::new(Mutex::new(hybrid)));
        let config =
            PagedAttentionConfig::new(None, MemoryGpuConfig::MbAmount(1), PagedCacheType::Auto)
                .unwrap()
                .with_base_device_memory_reservation(100)
                .unwrap()
                .with_recurrent_prefix_capacity(2);

        let reservations =
            paged_attention_memory_reservations(&cache, config, &Device::Cpu).unwrap();

        assert_eq!(reservations.primary_device_bytes, 676);
        assert_eq!(reservations.secondary_device_bytes, 0);
    }
    use serde_json::Value;

    #[test]
    fn base_lora_routes_still_validate_dense_batch_cardinality() -> candle_core::Result<()> {
        let input_ids = Tensor::zeros((2, 3), candle_core::DType::U32, &Device::Cpu)?;
        let flash_meta = FlashParams::empty(true);

        assert!(
            resolve_lora_execution(None, &input_ids, None, &flash_meta, &[None, None])?.is_none()
        );
        let error =
            resolve_lora_execution(None, &input_ids, None, &flash_meta, &[None]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("adapter lease count 1 does not match model batch size 2")
        );
        Ok(())
    }

    #[test]
    fn base_lora_routes_validate_packed_logical_shape() -> candle_core::Result<()> {
        let input_ids = Tensor::zeros((1, 5), candle_core::DType::U32, &Device::Cpu)?;
        let mut flash_meta = FlashParams::empty(true);
        flash_meta.packed = true;
        let mut paged_meta = PagedAttentionInputMetadata::dummy(&Device::Cpu)?;
        paged_meta.query_lens = Some(vec![2, 3]);

        assert!(
            resolve_lora_execution(
                None,
                &input_ids,
                Some(&paged_meta),
                &flash_meta,
                &[None, None],
            )?
            .is_none()
        );

        let error =
            resolve_lora_execution(None, &input_ids, Some(&paged_meta), &flash_meta, &[None])
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("adapter lease count 1 does not match packed logical sequence count 2")
        );

        paged_meta.query_lens = Some(vec![2, 2]);
        let error = resolve_lora_execution(
            None,
            &input_ids,
            Some(&paged_meta),
            &flash_meta,
            &[None, None],
        )
        .unwrap_err();
        assert!(error.to_string().contains(
            "packed logical query lengths total 4 does not match physical sequence length 5"
        ));
        Ok(())
    }

    #[test]
    fn packed_logits_select_each_logical_sequence() {
        let source = Tensor::from_vec(
            (0u8..8).map(f32::from).collect::<Vec<_>>(),
            (1, 8, 1),
            &Device::Cpu,
        )
        .unwrap();
        let selection = LogitsSelection::from_packed_context_lens(
            &source,
            &[(2, 1), (0, 1), (3, 1)],
            &[3, 1, 4],
            &[Device::Cpu],
        )
        .unwrap();
        let selected = selection.select(&source).unwrap();

        assert_eq!(selected.dims(), &[3, 1, 1]);
        assert_eq!(
            selected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![2.0, 3.0, 7.0]
        );
    }

    #[test]
    fn packed_logits_select_multi_token_spans() {
        let source = Tensor::from_vec(
            (0u8..5).map(f32::from).collect::<Vec<_>>(),
            (1, 5, 1),
            &Device::Cpu,
        )
        .unwrap();
        let selection = LogitsSelection::from_packed_context_lens(
            &source,
            &[(1, 2), (0, 2)],
            &[3, 2],
            &[Device::Cpu],
        )
        .unwrap();
        let selected = selection.select(&source).unwrap();

        assert_eq!(selected.dims(), &[2, 2, 1]);
        assert_eq!(
            selected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn packed_positions_require_explicit_metadata() {
        let mut flash_params = FlashParams::empty(true);
        flash_params.packed = true;
        let seqlen_offsets = [0];
        let context_lens = [(0, 1)];
        let position_ids = [1];
        let mut context = ModelForwardContext::with_cache(
            ForwardCache::None,
            &seqlen_offsets,
            &context_lens,
            &position_ids,
            &flash_params,
        );

        let error = context.text_positions(&Device::Cpu, 1).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("packed prefill is missing RoPE positions")
        );
    }

    #[test]
    fn decode_positions_expand_exclusive_row_ends() {
        let positions = decode_positions_tensor(&[4, 10], 3, &Device::Cpu).unwrap();

        assert_eq!(positions.to_vec1::<u32>().unwrap(), vec![1, 2, 3, 7, 8, 9]);
        assert!(decode_positions_tensor(&[2], 3, &Device::Cpu).is_err());
    }

    macro_rules! hashmap {
        (@single $($x:tt)*) => (());
        (@count $($rest:expr),*) => (<[()]>::len(&[$(hashmap!(@single $rest)),*]));

        ($($key:expr => $value:expr,)+) => { hashmap!($($key => $value),+) };
        ($($key:expr => $value:expr),*) => {
            {
                let _cap = hashmap!(@count $($key),*);
                let mut _map = ::indexmap::IndexMap::with_capacity(_cap);
                $(
                    let _ = _map.insert($key, Value::String($value));
                )*
                _map
            }
        };
    }

    #[cfg(test)]
    #[track_caller]
    fn test_with_inputs(
        templates: &[(bool, &str, &str, &str, &str)],
        expected_outputs: &[&str],
        inputs: Vec<IndexMap<String, MessageContent>>,
    ) {
        use inference_protocol::chat_template::ChatTemplateValue;

        use inference_protocol::chat_template::apply_chat_template_to;
        let mut failed = Vec::new();
        let n_templates = templates.len();
        for ((has_system, bos, eos, unk, template), expected) in
            templates.iter().zip(expected_outputs)
        {
            let output = match apply_chat_template_to(
                if !has_system {
                    inputs[1..].to_vec()
                } else {
                    inputs.clone()
                },
                true,
                None,
                None, // reasoning_effort
                &ChatTemplateValue(Either::Left(template.to_string())),
                Some(bos.to_string()),
                Some(eos.to_string()),
                Some(unk.to_string()),
                Vec::new(),
            ) {
                Ok(v) => v,
                Err(e) => {
                    failed.push(format!("Failed with {e}."));
                    continue;
                }
            };
            if output != *expected {
                failed.push(format!(
                    "Expected: `{}` \n\nGot:      `{}`",
                    expected.replace('\n', "\\n"),
                    output.replace('\n', "\\n")
                ));
            }
        }
        if !failed.is_empty() {
            for (i, line) in failed.iter().enumerate() {
                println!("------------ Template {i} ------------");
                println!("{line}");
            }
            println!("------------------------");
            panic!("{}/{n_templates} chat templates failed.", failed.len());
        }
    }

    #[test]
    /// Generating these cases:
    /// ```py
    /// >>> t=transformers.AutoTokenizer.from_pretrained(...)
    /// # If non-system prompt model
    /// >>> t.apply_chat_template([{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi there"},{"role":"user","content":"Who are you"},{"role":"assistant","content":"   I am an assistant   "},{"role":"user","content":"Another question"}], add_generation_prompt=True, tokenize=False)
    /// # If system prompt model
    /// >>> t.apply_chat_template([{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi there"},{"role":"user","content":"Who are you"},{"role":"assistant","content":"   I am an assistant   "},{"role":"user","content":"Another question"}], add_generation_prompt=True, tokenize=False)
    /// ```
    fn test_chat_templates() {
        let templates = [
            // ChatML: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
            (
                true,
                "<s>",
                "</s>",
                "<unk>",
                "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
            ),
            // mistralai/Mistral-7B-Instruct-v0.1
            (
                false,
                "<s>",
                "</s>",
                "<unk>",
                "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
            ),
            // meta-llama/Llama-2-13b-chat-hf
            (
                true,
                "<s>",
                "</s>",
                "<unk>",
                "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}",
            ),
            // mistralai/Mixtral-8x7B-Instruct-v0.1
            (
                false,
                "<s>",
                "</s>",
                "<unk>",
                "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
            ),
            // google/gemma-7b-it
            (
                false,
                "<bos>",
                "<eos>",
                "<unk>",
                "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}",
            ),
            // HuggingFaceM4/idefics2-8b-chatty
            (
                true,
                "<s>",
                "</s>",
                "<unk>",
                "{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}",
            ),
        ];
        let expected_outputs = [
            // ChatML: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
            "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nWho are you<|im_end|>\n<|im_start|>assistant\n   I am an assistant   <|im_end|>\n<|im_start|>user\nAnother question<|im_end|>\n<|im_start|>assistant\n",
            // mistralai/Mistral-7B-Instruct-v0.1
            "<s>[INST] Hello [/INST]Hi there</s> [INST] Who are you [/INST]   I am an assistant   </s> [INST] Another question [/INST]",
            // meta-llama/Llama-2-13b-chat-hf
            "<s>[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST] Hi there </s><s>[INST] Who are you [/INST] I am an assistant </s><s>[INST] Another question [/INST]",
            // mistralai/Mixtral-8x7B-Instruct-v0.1
            "<s>[INST] Hello [/INST]Hi there</s>[INST] Who are you [/INST]   I am an assistant   </s>[INST] Another question [/INST]",
            // google/gemma-7b-it
            "<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nHi there<end_of_turn>\n<start_of_turn>user\nWho are you<end_of_turn>\n<start_of_turn>model\nI am an assistant<end_of_turn>\n<start_of_turn>user\nAnother question<end_of_turn>\n<start_of_turn>model\n",
        ];
        let messages = [
            ["system", "You are a helpful assistant"],
            ["user", "Hello"],
            ["assistant", "Hi there"],
            ["user", "Who are you"],
            ["assistant", "   I am an assistant   "],
            ["user", "Another question"],
        ];
        let mut inputs = Vec::new();
        for [role, content] in messages {
            let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
                IndexMap::new();
            message.insert("role".to_string(), Either::Left(role.to_string()));
            message.insert("content".to_string(), Either::Left(content.to_string()));
            inputs.push(message);
        }
        test_with_inputs(&templates, &expected_outputs, inputs);
    }

    #[test]
    /// Generating these cases:
    /// ```py
    /// >>> processor=transformers.AutoProcessor.from_pretrained(...)
    /// >>> processor.apply_chat_template([
    ///         {"role":"system","content":[{"type":"text", "text": "You are a helpful assistant"}]},
    ///         {"role":"user","content":[{"type":"image"}, {"type":"text", "text": "Hello, please describe the above."}]},
    ///         {"role":"assistant","content":[{"type":"text", "text": "Hi there"}]},
    ///         {"role":"user","content":[{"type":"text", "text": "Who are you"}]},
    ///         {"role":"assistant","content":[{"type":"text", "text": "   I am an assistant   "}]},
    ///         {"role":"user","content":[{"type":"text", "text": "Another question"}]}
    ///     ], add_generation_prompt=True, tokenize=False)
    /// ```
    fn test_image_chat_templates() {
        let templates = [
            // HuggingFaceM4/idefics2-8b-chatty
            (
                true,
                "<s>",
                "</s>",
                "<unk>",
                "{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}",
            ),
        ];
        let expected_outputs = [
            // HuggingFaceM4/idefics2-8b-chatty
            "System: You are a helpful assistant<end_of_utterance>\nUser:<image>Hello, please describe the above.<end_of_utterance>\nAssistant: Hi there<end_of_utterance>\nUser:<image>This is me, who are you<end_of_utterance>\nAssistant:    I am an assistant   <end_of_utterance>\nUser:<image>Another question, what is this?<end_of_utterance>\nAssistant:",
        ];

        let mut inputs = Vec::new();

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("system".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![hashmap! {
                "type".to_string() => "text".to_string(),
                "text".to_string() => "You are a helpful assistant".to_string()
            }]),
        );
        inputs.push(message);

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("user".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![
                hashmap! {
                    "type".to_string() => "image".to_string()
                },
                hashmap! {
                    "type".to_string() => "text".to_string(),
                    "text".to_string() => "Hello, please describe the above.".to_string()
                },
            ]),
        );
        inputs.push(message);

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("assistant".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![hashmap! {
                "type".to_string() => "text".to_string(),
                "text".to_string() => "Hi there".to_string()
            }]),
        );
        inputs.push(message);

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("user".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![
                hashmap! {
                    "type".to_string() => "image".to_string()
                },
                hashmap! {
                    "type".to_string() => "text".to_string(),
                    "text".to_string() => "This is me, who are you".to_string()
                },
            ]),
        );
        inputs.push(message);

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("assistant".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![hashmap! {
                "type".to_string() => "text".to_string(),
                "text".to_string() => "   I am an assistant   ".to_string()
            }]),
        );
        inputs.push(message);

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("user".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![
                hashmap! {
                    "type".to_string() => "image".to_string()
                },
                hashmap! {
                    "type".to_string() => "text".to_string(),
                    "text".to_string() => "Another question, what is this?".to_string()
                },
            ]),
        );
        inputs.push(message);

        test_with_inputs(&templates, &expected_outputs, inputs);
    }
}

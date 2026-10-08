//! The state the text and multimodal pipelines share, and the construction both loaders end with.

use std::path::PathBuf;
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::sync::Mutex as StdMutex;

use anyhow::Result;
use inference_nn::speculative::SpeculativeTargetMixin;
use inference_tensor::{DType, Device};
use tokenizers::Tokenizer;

use super::chat_template::{GenerationConfig, calculate_eos_tokens};
#[cfg(feature = "cuda")]
use super::cuda_graph::CudaDecodeGraphState;
use super::llg::build_llg_factory;
use super::{
    ChatTemplate, EitherCache, GeneralMetadata, Modalities, ModelKind,
    paged_attention_memory_reservations,
};
use crate::device_map::DeviceMapper;
use crate::paged_attention::{CacheEngine, ModelConfigLike, calculate_cache_config};
use crate::{DynamicLoraRuntime, PagedAttentionConfig};

pub(crate) struct DecoderCore {
    pub tokenizer: Arc<Tokenizer>,
    pub chat_template: Arc<ChatTemplate>,
    pub model_id: String,
    pub metadata: Arc<GeneralMetadata>,
    pub mapper: Box<dyn DeviceMapper + Send + Sync>,
    #[cfg(feature = "cuda")]
    pub cuda_decode_graph: StdMutex<CudaDecodeGraphState>,
    #[cfg(feature = "cuda")]
    pub cuda_sparse_rejection: StdMutex<Option<crate::speculative::CudaSparseRejectionWorkspace>>,
    pub generation_defaults: Option<crate::ModelGenerationDefaults>,
    pub tracked_modules: Vec<inference_quant::TrackedModule>,
    pub source_weight_files: Vec<PathBuf>,
    pub source_weight_source: Option<Arc<dyn inference_quant::QuantizedWeightSource>>,
    pub dynamic_lora: Option<Arc<DynamicLoraRuntime>>,
}

/// What the constructor reads from a loaded model.
pub(crate) struct LoadedModelView<'a> {
    pub target: &'a dyn SpeculativeTargetMixin,
    pub cache: &'a EitherCache,
    pub config: Arc<dyn ModelConfigLike + Send + Sync>,
    pub max_seq_len: usize,
    pub sliding_window: Option<usize>,
    pub block_diffusion: bool,
}

pub(crate) struct DecoderCoreArgs<'a> {
    pub model: LoadedModelView<'a>,
    pub tokenizer: Tokenizer,
    pub chat_template: ChatTemplate,
    pub generation_config: Option<GenerationConfig>,
    pub paged_attn_config: Option<PagedAttentionConfig>,
    pub dtype: DType,
    pub device: Device,
    pub mapper: Box<dyn DeviceMapper + Send + Sync>,
    /// The mapped layers' devices from the load session; None for the non-mapped device.
    pub layer_devices: Vec<Option<Device>>,
    pub silent: bool,
    pub max_kv_tokens: Option<usize>,
    pub no_kv_cache: bool,
    pub no_prefix_cache: bool,
    pub kind: ModelKind,
    pub model_id: String,
    pub modalities: Modalities,
    pub loaded_for_uqff_write: bool,
    pub tracked_modules: Vec<inference_quant::TrackedModule>,
    pub source_weight_files: Vec<PathBuf>,
    pub source_weight_source: Option<Arc<dyn inference_quant::QuantizedWeightSource>>,
    pub dynamic_lora: Option<Arc<DynamicLoraRuntime>>,
}

impl DecoderCore {
    /// Reserves recurrent state, sizes and allocates the paged KV cache, and gathers the pipeline metadata.
    pub(crate) fn new(args: DecoderCoreArgs<'_>) -> Result<Self> {
        // parallel loading leaves stream-ordered frees pending; drain so KV sizing sees the real free VRAM
        #[cfg(feature = "cuda")]
        super::synchronize_cuda_contexts(&args.device, args.mapper.as_ref())?;

        // layers past the mapped stack (e.g. an MTP head) live on the non-mapped device
        let mut layer_devices = args.layer_devices;
        layer_devices.resize(
            layer_devices.len().max(args.model.config.num_layers()),
            Some(args.device.clone()),
        );

        super::RecurrentReservation {
            target: args.model.target,
            cache: args.model.cache,
            paged_attn_config: args.paged_attn_config,
            dtype: args.dtype,
            model_config: args.model.config.as_ref(),
            device: &args.device,
        }
        .reserve(args.mapper.as_ref())?;

        let (cache_config, cache_engine) = if let Some(paged_attn_config) = args.paged_attn_config {
            let cache_config = calculate_cache_config(
                paged_attn_config.mem_gpu,
                paged_attention_memory_reservations(
                    args.model.cache,
                    paged_attn_config,
                    &args.device,
                )?,
                paged_attn_config.block_size,
                args.dtype,
                paged_attn_config.cache_type,
                args.model.config.as_ref(),
                &args.device,
                &layer_devices,
                args.silent,
                None,
                args.max_kv_tokens,
            )?;
            let cache_engine = CacheEngine::new(
                args.model.config.as_ref(),
                &cache_config,
                args.dtype,
                &args.device,
                layer_devices,
            )?;
            (Some(cache_config), Some(cache_engine))
        } else {
            (None, None)
        };

        let mut generation_defaults = args
            .generation_config
            .as_ref()
            .and_then(GenerationConfig::generation_defaults);
        // a block-diffusion checkpoint's max_new_tokens is one canvas, not a session cap
        if args.model.block_diffusion
            && let Some(defaults) = generation_defaults.as_mut()
        {
            defaults.max_new_tokens = None;
            defaults.max_length = None;
        }
        let eos_tok = calculate_eos_tokens(
            &args.chat_template,
            args.generation_config.as_ref(),
            &args.tokenizer,
        );

        Ok(Self {
            metadata: Arc::new(GeneralMetadata {
                max_seq_len: args.model.max_seq_len,
                llg_factory: Some(build_llg_factory(args.tokenizer.clone())?),
                no_kv_cache: args.no_kv_cache,
                no_prefix_cache: args.no_prefix_cache,
                num_hidden_layers: super::cache_layer_count(args.model.cache),
                eos_tok,
                kind: args.kind,
                activation_dtype: args.dtype,
                sliding_window: args.model.sliding_window,
                cache_config,
                cache_engine,
                model_metadata: Some(args.model.config),
                modalities: args.modalities,
                loaded_for_uqff_write: args.loaded_for_uqff_write,
            }),
            tokenizer: args.tokenizer.into(),
            chat_template: Arc::new(args.chat_template),
            model_id: args.model_id,
            mapper: args.mapper,
            #[cfg(feature = "cuda")]
            cuda_decode_graph: StdMutex::new(CudaDecodeGraphState::default()),
            #[cfg(feature = "cuda")]
            cuda_sparse_rejection: StdMutex::new(None),
            generation_defaults,
            tracked_modules: args.tracked_modules,
            source_weight_files: args.source_weight_files,
            source_weight_source: args.source_weight_source,
            dynamic_lora: args.dynamic_lora,
        })
    }
}

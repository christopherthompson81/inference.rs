use std::{
    any::Any,
    sync::{atomic::AtomicUsize, Arc, Mutex},
};

use candle_core::{Device, Tensor};
use indicatif::MultiProgress;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::FlashParams,
    device_map::DeviceMapper,
    kv_cache::EitherCache,
    matformer::MatformerSliceConfig,
    model::ModelForwardContext,
    paged_attention::{encoder_cache::EncoderCacheManager, ModelConfigLike, ModelConfigMetadata},
    speculative::SpeculativeTargetMixin,
};

pub trait IsqModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)>;

    fn residual_tensors_moe_experts_only(&self) -> Option<Vec<(String, Tensor)>> {
        None
    }
}

pub trait NormalModel: IsqModel + AnyMoeBaseModelMixin + SpeculativeTargetMixin {
    fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor>;
    #[allow(clippy::too_many_arguments)]
    fn xlora_forward(
        &self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        no_kv_cache: bool,
        non_granular_state: &Option<NonGranularState>,
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        flash_params: &FlashParams,
        flash_params_full: &FlashParams,
    ) -> candle_core::Result<Tensor>;
    fn is_xlora(&self) -> bool;
    fn device(&self) -> &Device;
    fn cache(&self) -> &EitherCache;
    fn max_seq_len(&self) -> usize;
    fn config(&self) -> &ModelConfigMetadata;
    /// True only when the full forward handles packed prompts and never treats physical rows as logical requests.
    fn supports_packed_prefill(&self) -> bool {
        false
    }
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        false
    }
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        Arc::new(self.config().clone())
    }
}

/// Metadata for loading a model with ISQ or device mapping.
pub struct NormalLoadingMetadata {
    // Device mapping metadata which can be used to construct a concrete device mapper
    pub mapper: Box<dyn DeviceMapper + Send + Sync>,
    // Flag to check if loading in ISQ
    pub loading_isq: bool,
    // Device mapping target device (the one that is not the cpu)
    pub real_device: Device,
    // MultiProgress support for parallelized loading
    pub multi_progress: Arc<MultiProgress>,
    // Optional Matryoshka Transformer slicing configuration
    pub matformer_slicing_config: Option<MatformerSliceConfig>,
    pub rope_pairing: Option<RopePairing>,
}

pub trait MultimodalModel:
    IsqModel + AnyMoeBaseModelMixin + SpeculativeTargetMixin + BlockDiffusionMixin
{
    // pixel_values and pixel_attention_mask only specified for prompt seqs
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>, // pixel attention mask, or image sizes, or anything else
        ctx: &mut ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor>;
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        false
    }
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs_for_args(&self, _model_specific_args: &dyn Any) -> bool {
        self.supports_cuda_decode_graphs()
    }
    fn requires_uniform_completion_batch(&self) -> bool {
        self.is_block_diffusion()
    }
    fn supports_packed_prefill(&self) -> bool {
        false
    }
    fn supports_mixed_media_batches(&self) -> bool {
        false
    }
    fn device(&self) -> &Device;
    fn cache(&self) -> &EitherCache;
    fn max_seq_len(&self) -> usize;
    fn config(&self) -> &ModelConfigMetadata;
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        Arc::new(self.config().clone())
    }
    /// For a prompt without images. Requires batch size of 1!
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any>;
    fn encoder_cache(&self) -> Option<&Mutex<EncoderCacheManager>> {
        None
    }
    fn configure_encoder_cache_memory_bytes(&self, max_bytes: usize) -> bool {
        let Some(cache) = self.encoder_cache() else {
            return false;
        };
        cache
            .lock()
            .expect("encoder cache poisoned")
            .set_max_logical_bytes(max_bytes);
        true
    }
    fn encoder_cache_counters(&self) -> Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)> {
        self.encoder_cache()
            .map(|cache| cache.lock().expect("encoder cache poisoned").counters())
    }
    fn reset_model_specific_state(&self) {}
    fn reset_model_specific_state_for_sequences(&self, _sequence_ids: &[usize]) {
        self.reset_model_specific_state();
    }
}

pub trait EmbeddingModel: IsqModel + AnyMoeBaseModelMixin {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input_ids: &Tensor,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor>;
    fn device(&self) -> &Device;
}

pub trait DiffusionModel {
    /// This returns a tensor of shape (bs, c, h, w), with values in [0, 255].
    fn forward(
        &mut self,
        prompts: Vec<String>,
        params: DiffusionGenerationParams,
    ) -> candle_core::Result<Tensor>;
    fn device(&self) -> &Device;
    fn max_seq_len(&self) -> usize;
}

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffusionGenerationParams {
    pub height: usize,
    pub width: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopePairing {
    Adjacent,
    HalfSplit,
}

pub struct NonGranularState {
    pub non_granular_index: Arc<tokio::sync::Mutex<usize>>,
    pub tgt_non_granular_index: usize,
}

/// Mixin for block-diffusion models. Defaults describe an ordinary autoregressive model;
/// diffusion models override all three.
pub trait BlockDiffusionMixin {
    /// When true, `forward` returns committed block token ids as a u32 tensor
    /// [bs, block_len] rather than logits.
    fn is_block_diffusion(&self) -> bool {
        false
    }

    /// Hand the model the checkpoint's raw `generation_config.json` (the source of truth
    /// for denoising parameters). No-op for other models.
    fn configure_block_diffusion(&self, _generation_config_json: &str) {}

    /// Time the last forward spent in the denoising loop (vs encoding); lets the engine
    /// book that share as completion time rather than prompt time.
    fn take_block_denoise_time(&self) -> Option<std::time::Duration> {
        None
    }
}

impl Default for DiffusionGenerationParams {
    /// Image dimensions will be 720x1280.
    fn default() -> Self {
        Self {
            height: 720,
            width: 1280,
        }
    }
}

#[cfg(feature = "pyo3_macros")]
#[pyo3::pymethods]
impl DiffusionGenerationParams {
    fn __repr__(&self) -> String {
        format!("{self:#?}")
    }
}

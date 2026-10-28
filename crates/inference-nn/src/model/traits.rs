use std::{
    any::Any,
    sync::{Arc, Mutex, atomic::AtomicUsize},
};

use indicatif::MultiProgress;
use inference_tensor::{Device, Tensor};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::FlashParams,
    device_map::DeviceMapper,
    kv_cache::EitherCache,
    matformer::MatformerSliceConfig,
    model::ModelForwardContext,
    paged_attention::{ModelConfigLike, ModelConfigMetadata, encoder_cache::EncoderCacheManager},
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
    ) -> inference_tensor::Result<Tensor>;
    fn device(&self) -> &Device;
    fn cache(&self) -> &EitherCache;
    fn max_seq_len(&self) -> usize;
    fn config(&self) -> &ModelConfigMetadata;
    /// How many tokens before each chunk the forward reads from `ModelForwardContext::token_window`; `None` when it
    /// reads no window.
    fn token_window(&self) -> Option<usize> {
        None
    }
    /// True only when the full forward handles packed prompts and never treats physical rows as logical requests.
    fn supports_packed_prefill(&self) -> bool {
        false
    }
    /// The prompt is read bidirectionally, so it runs in one pass and no cached prefix stands for part of it.
    fn bidirectional_prompt(&self) -> bool {
        false
    }
    /// A joint decision head's option scores over a prompt's normed hidden states (clef).
    fn joint_decision(
        &self,
        _hidden: &Tensor,
        _layout: &crate::decision::JointLayout,
    ) -> inference_tensor::Result<Vec<f32>> {
        inference_tensor::bail!("this model has no joint decision head")
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
    ) -> inference_tensor::Result<Tensor>;
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
    /// The prompt is read bidirectionally, so it runs in one pass and no cached prefix stands for part of it.
    fn bidirectional_prompt(&self) -> bool {
        false
    }
    fn supports_mixed_media_batches(&self) -> bool {
        false
    }
    /// As `NormalModel::token_window`.
    fn token_window(&self) -> Option<usize> {
        None
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
    fn forward(
        &self,
        input_ids: &Tensor,
        extras: &crate::embedding::EmbeddingExtras,
        flash_params: &FlashParams,
    ) -> inference_tensor::Result<Tensor>;
    fn device(&self) -> &Device;
}

pub trait DiffusionModel {
    /// This returns a tensor of shape (bs, c, h, w), with values in [0, 255].
    fn forward(
        &mut self,
        prompts: Vec<String>,
        params: DiffusionGenerationParams,
    ) -> inference_tensor::Result<Tensor>;
    fn device(&self) -> &Device;
    fn max_seq_len(&self) -> usize;
    /// Why this model refuses `params`, checked before the request is queued.
    fn validate(&self, _params: &DiffusionGenerationParams) -> std::result::Result<(), String> {
        Ok(())
    }
}

/// The most reference images one request may carry, which the API checks before fetching any.
pub const MAX_REFERENCE_IMAGES: usize = 16;

/// The images an edit or subject-reference request conditions on, shared by each of its choices.
#[derive(Clone, Default)]
pub struct ReferenceImages(pub std::sync::Arc<Vec<image::DynamicImage>>);

// the choices of one request share the images, which batching compares for every queued request
impl PartialEq for ReferenceImages {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

impl std::fmt::Debug for ReferenceImages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} reference images", self.0.len())
    }
}

impl ReferenceImages {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DiffusionGenerationParams {
    pub height: usize,
    pub width: usize,
    #[serde(skip)]
    pub references: ReferenceImages,
    /// The output takes the only reference image's own size, in place of `width` and `height`.
    #[serde(default)]
    pub match_reference: bool,
    /// Denoising steps, when not the model's own.
    pub steps: Option<usize>,
    /// Classifier-free guidance scale, when not the model's own.
    pub guidance_scale: Option<f64>,
    /// Seeds the noise, for a reproducible image.
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopePairing {
    Adjacent,
    HalfSplit,
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
            references: ReferenceImages::default(),
            match_reference: false,
            steps: None,
            guidance_scale: None,
            seed: None,
        }
    }
}

/// Receives a block-diffusion model's per-step progress for one sequence.
pub trait BlockDenoisingProgressSink: Send + Sync {
    fn emit(
        &self,
        step: usize,
        total_steps: usize,
        tokens: &[u32],
        finished: bool,
        final_block: bool,
    );
}

/// Routes one batch row's denoising progress to the engine while the model runs its loop.
#[derive(Clone)]
pub struct BlockDenoisingProgressEmitter {
    batch_index: usize,
    sink: Arc<dyn BlockDenoisingProgressSink>,
}

impl BlockDenoisingProgressEmitter {
    pub fn new(batch_index: usize, sink: Arc<dyn BlockDenoisingProgressSink>) -> Self {
        Self { batch_index, sink }
    }

    pub fn emit(
        &self,
        step: usize,
        total_steps: usize,
        tokens: &[u32],
        finished: bool,
        final_block: bool,
    ) {
        self.sink
            .emit(step, total_steps, tokens, finished, final_block);
    }

    pub fn batch_index(&self) -> usize {
        self.batch_index
    }
}

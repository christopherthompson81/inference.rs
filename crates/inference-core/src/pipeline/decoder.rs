//! The one pipeline the text and multimodal decoders run in: shared state, decode graphs, forward and sampling.

use super::decoder_core::DecoderCore;
use super::{
    AnyMoePipelineMixin, CacheManagerMixin, DecodeGraphPrecaptureCtx, EitherCache,
    ForwardInputsResult, ForwardStepResult, GeneralMetadata, IsqPipelineMixin, MetadataMixin,
    ModelCategory, MultimodalModel, MultimodalPromptPrefixer, PreProcessingMixin, Processor,
    TokenSource,
};
#[cfg(feature = "cuda")]
use crate::cuda::gdn::GDN_PAD_SLOT;
use crate::device_map::DeviceMapper;

use crate::attention::FlashParams;
use crate::gdn::RecurrentBatchKind;
use crate::paged_attention::PagedAttentionInputMetadata;
#[cfg(feature = "cuda")]
use crate::pipeline::cuda_graph::{
    CudaDecodeGraphCaptureCtx, CudaDecodeGraphKey, CudaDecodeGraphLaunch, CudaDecodeGraphReplay,
    CudaDecodeGraphReplayInput, CudaDecodeGraphState, CudaGraphComponent, CudaGraphDecodeStep,
    CudaGraphDecodeStepInputs, CudaGraphDispatchMode, CudaGraphDispatchReason, CudaGraphEvent,
    CudaGraphEventGuard, CudaGraphPrecaptureInputs, capture_cuda_decode_graph,
    cuda_decode_graph_batch_kind_supported, cuda_decode_graph_supported_for_model,
    cuda_decode_graphs_enabled, cuda_graph_batch_bucket, cuda_graph_precapture_batches,
    cuda_graph_precapture_max_batch, cuda_graph_startup_capture_allowed, disable_cuda_decode_graph,
    finish_cuda_graph_capture_attempt, hybrid_graph_slots, install_hybrid_graph_state_indices,
    record_cuda_graph_dispatch, restore_hybrid_state_indices,
    snapshot_hybrid_recurrent_checkpoints, snapshot_hybrid_state_indices,
    speculative_decode_logs_transitions, target_cuda_graph_cache_capacity,
};
use crate::pipeline::sampling::{sample_and_add_toks, sample_and_add_toks_batched};
use crate::pipeline::{ChatTemplate, ModelForwardContext};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::vision_models::ModelInputs;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::{AnyMoeExpertType, DynamicLoraRuntime, Pipeline};
use anyhow::Result;
use futures::{FutureExt, future::BoxFuture};
use inference_quant::IsqType;
use inference_tensor::{DType, Device, Tensor, Var};
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::AtomicUsize;
use tokenizers::Tokenizer;
#[cfg(feature = "cuda")]
use tracing::{info, warn};

use crate::paged_attention::ModelConfigMetadata;
use inference_nn::amoe::AnyMoeBaseModelMixin;
use inference_nn::speculative::SpeculativeTargetMixin;

use super::NormalModel;
use super::text_models_inputs_processor::ModelInputs as TextModelInputs;

#[cfg(feature = "cuda")]
struct CudaDecodeGraphCaptureInputs<'a> {
    kv_cache: &'a [(Tensor, Tensor)],
    flash_meta: &'a FlashParams,
    recurrent_batch_kind: RecurrentBatchKind,
    speculative: bool,
}
#[cfg(feature = "cuda")]
struct CudaDecodeGraphForwardInput<'a> {
    input_ids: &'a Tensor,
    seqlen_offsets: &'a [usize],
    context_lens: &'a [(usize, usize)],
    position_ids: &'a [usize],
    paged_attn_meta: Option<(Vec<(Tensor, Tensor)>, &'a PagedAttentionInputMetadata)>,
    flash_meta: &'a FlashParams,
    model_specific_args: Option<&'a dyn Any>,
    recurrent_batch_kind: RecurrentBatchKind,
}
#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Debug, PartialEq, Eq)]
struct SpeculativeGraphTensorMetadata {
    shape: Vec<usize>,
    strides: Vec<usize>,
    contiguous: bool,
    dtype: DType,
    device: inference_tensor::DeviceLocation,
}

#[cfg(feature = "cuda")]
fn speculative_graph_tensor_metadata(
    state: &dyn crate::speculative::SpeculativeGraphState,
) -> Vec<SpeculativeGraphTensorMetadata> {
    state
        .tensors()
        .into_iter()
        .map(|tensor| {
            let (_storage, layout) = tensor.storage_and_layout();
            SpeculativeGraphTensorMetadata {
                shape: tensor.dims().to_vec(),
                strides: layout.stride().to_vec(),
                contiguous: tensor.is_contiguous(),
                dtype: tensor.dtype(),
                device: tensor.device().location(),
            }
        })
        .collect()
}

#[cfg(any(feature = "cuda", test))]
fn validate_speculative_graph_tensor_metadata(
    expected: &[SpeculativeGraphTensorMetadata],
    actual: &[SpeculativeGraphTensorMetadata],
) -> inference_tensor::Result<()> {
    if actual.len() != expected.len() {
        inference_tensor::bail!(
            "speculative graph state changed tensor count between warmup and capture"
        );
    }
    if actual.iter().zip(expected).any(|(actual, expected)| {
        actual.shape != expected.shape
            || actual.strides != expected.strides
            || actual.contiguous != expected.contiguous
            || actual.dtype != expected.dtype
            || actual.device != expected.device
    }) {
        inference_tensor::bail!(
            "speculative graph state changed tensor metadata between warmup and capture"
        );
    }
    Ok(())
}

/// A decoder the pipeline runs: a text model, or a multimodal one with its media inputs.
pub(crate) enum DecoderModel {
    Text(Box<dyn NormalModel + Send + Sync>),
    Multimodal(Box<dyn MultimodalModel + Send + Sync>),
}

impl std::ops::Deref for DecoderModel {
    type Target = dyn SpeculativeTargetMixin;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Text(model) => &**model,
            Self::Multimodal(model) => &**model,
        }
    }
}

impl std::ops::DerefMut for DecoderModel {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Text(model) => &mut **model,
            Self::Multimodal(model) => &mut **model,
        }
    }
}

impl DecoderModel {
    fn cache(&self) -> &EitherCache {
        match self {
            Self::Text(model) => model.cache(),
            Self::Multimodal(model) => model.cache(),
        }
    }

    fn device(&self) -> &Device {
        match self {
            Self::Text(model) => model.device(),
            Self::Multimodal(model) => model.device(),
        }
    }

    fn config(&self) -> &ModelConfigMetadata {
        match self {
            Self::Text(model) => model.config(),
            Self::Multimodal(model) => model.config(),
        }
    }

    /// A text model takes no media; a multimodal one gets its text-only arguments when `args` is None.
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        args: Option<Box<dyn Any>>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> inference_tensor::Result<Tensor> {
        match self {
            Self::Text(model) => model.forward(input_ids, ctx),
            Self::Multimodal(model) => {
                let args = args.unwrap_or_else(|| model.default_model_specific_args(input_ids));
                model.forward(input_ids, pixel_values, args, ctx)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self, args: Option<&dyn Any>) -> bool {
        match self {
            Self::Text(model) => model.supports_cuda_decode_graphs(),
            Self::Multimodal(model) => match args {
                Some(args) => model.supports_cuda_decode_graphs_for_args(args),
                None => model.supports_cuda_decode_graphs(),
            },
        }
    }

    fn supports_packed_prefill(&self) -> bool {
        match self {
            Self::Text(model) => model.supports_packed_prefill(),
            Self::Multimodal(model) => model.supports_packed_prefill(),
        }
    }

    fn multimodal(&self) -> Option<&dyn MultimodalModel> {
        match self {
            Self::Text(_) => None,
            Self::Multimodal(model) => Some(&**model),
        }
    }

    fn amoe(&self) -> &dyn AnyMoeBaseModelMixin {
        match self {
            Self::Text(model) => &**model,
            Self::Multimodal(model) => &**model,
        }
    }

    fn amoe_mut(&mut self) -> &mut dyn AnyMoeBaseModelMixin {
        match self {
            Self::Text(model) => &mut **model,
            Self::Multimodal(model) => &mut **model,
        }
    }
}

/// What only a multimodal pipeline carries.
pub(crate) struct MediaState {
    pub processor: Arc<dyn Processor + Send + Sync>,
    pub preprocessor_config: Arc<PreProcessorConfig>,
    pub prefixer: Arc<dyn MultimodalPromptPrefixer>,
    pub video_sampling: crate::VideoFrameSampling,
}

pub struct DecoderPipeline {
    model: DecoderModel,
    core: DecoderCore,
    media: Option<MediaState>,
    // Attention inputs of the last prompt-chunk forward, so a built-in drafter can prefill with them
    last_prompt_attention: StdMutex<Option<(PagedAttentionInputMetadata, FlashParams)>>,
}

impl DecoderPipeline {
    pub(crate) fn new(model: DecoderModel, core: DecoderCore, media: Option<MediaState>) -> Self {
        Self {
            model,
            core,
            media,
            last_prompt_attention: StdMutex::new(None),
        }
    }
}

fn text_requires_uniform_prompt_batch(
    is_hybrid: bool,
    packed_prefill_available: bool,
    has_speculative_proposer: bool,
) -> bool {
    (is_hybrid && !packed_prefill_available)
        || (has_speculative_proposer && !packed_prefill_available)
}

/// One forward step's inputs, from either input processor.
struct StepInputs {
    input_ids: Tensor,
    seqlen_offsets: Vec<usize>,
    context_lens: Vec<(usize, usize)>,
    position_ids: Vec<usize>,
    pixel_values: Option<Tensor>,
    model_specific_args: Option<Box<dyn Any>>,
    paged_attn_meta: Option<PagedAttentionInputMetadata>,
    flash_meta: FlashParams,
    recurrent_batch_kind: RecurrentBatchKind,
    adapter_leases: Arc<[Option<crate::AdapterLease>]>,
}

impl StepInputs {
    fn from_processor_output(model: &DecoderModel, inputs: Box<dyn Any>) -> Self {
        match model {
            DecoderModel::Text(_) => {
                let inputs = *inputs
                    .downcast::<TextModelInputs>()
                    .expect("Downcast failed.");
                Self {
                    input_ids: inputs.input_ids,
                    seqlen_offsets: inputs.seqlen_offsets,
                    context_lens: inputs.context_lens,
                    position_ids: inputs.position_ids,
                    pixel_values: None,
                    model_specific_args: None,
                    paged_attn_meta: inputs.paged_attn_meta,
                    flash_meta: inputs.flash_meta,
                    recurrent_batch_kind: inputs.recurrent_batch_kind,
                    adapter_leases: inputs.adapter_leases,
                }
            }
            DecoderModel::Multimodal(_) => {
                let inputs = *inputs.downcast::<ModelInputs>().expect("Downcast failed.");
                Self {
                    input_ids: inputs.input_ids,
                    seqlen_offsets: inputs.seqlen_offsets,
                    context_lens: inputs.context_lens,
                    position_ids: inputs.position_ids,
                    pixel_values: inputs.pixel_values,
                    model_specific_args: Some(inputs.model_specific_args),
                    paged_attn_meta: inputs.paged_attn_meta,
                    flash_meta: inputs.flash_meta,
                    recurrent_batch_kind: inputs.recurrent_batch_kind,
                    adapter_leases: inputs.adapter_leases,
                }
            }
        }
    }
}

impl PreProcessingMixin for DecoderPipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.core.chat_template.clone())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        self.media
            .as_ref()
            .map(|media| media.preprocessor_config.clone() as Arc<dyn Any>)
    }
    fn get_processor(&self) -> Arc<dyn super::Processor> {
        match &self.media {
            Some(media) => media.processor.clone(),
            None => Arc::new(super::BasicProcessor),
        }
    }
}

impl IsqPipelineMixin for DecoderPipeline {
    fn re_isq_model(&mut self, dtype: IsqType) -> Result<()> {
        if !self.core.tracked_modules.is_empty() {
            self.cleanup_cuda_graphs();
        }
        super::isq_flow::requantize_tracked_modules(&self.core.tracked_modules, dtype)
    }

    fn begin_calibration(&mut self) -> Result<()> {
        super::isq_flow::begin_calibration(&self.core.tracked_modules)?;
        #[cfg(feature = "cuda")]
        self.core
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned")
            .suspend();
        Ok(())
    }

    fn calibration_status(&self) -> Result<super::isq_flow::CalibrationStatus> {
        Ok(super::isq_flow::calibration_status(
            &self.core.tracked_modules,
        ))
    }

    fn apply_calibration(
        &mut self,
        save_cimatrix: Option<std::path::PathBuf>,
    ) -> Result<super::isq_flow::CalibrationStatus> {
        self.cleanup_cuda_graphs();
        let result = super::isq_flow::apply_calibration(
            &self.core.tracked_modules,
            &self.core.source_weight_files,
            self.core.source_weight_source.as_deref(),
            save_cimatrix.as_deref(),
        );
        #[cfg(feature = "cuda")]
        if result.is_ok()
            || !super::isq_flow::calibration_status(&self.core.tracked_modules).collecting
        {
            self.core
                .cuda_decode_graph
                .lock()
                .expect("CUDA graph mutex poisoned")
                .resume();
        }
        result
    }
}

impl CacheManagerMixin for DecoderPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) -> inference_tensor::Result<()> {
        super::cache_manager::clone_in_cache_by_kind(self, seqs)
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        super::cache_manager::clone_out_cache_by_kind(self, seqs)
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) -> inference_tensor::Result<()> {
        super::cache_manager::set_none_cache_by_kind(
            self,
            seqs,
            modify_draft_cache,
            load_preallocated_cache,
        )?;
        if let Some(model) = self.model.multimodal() {
            let sequence_ids = seqs.iter().map(|seq| *seq.id()).collect::<Vec<_>>();
            model.reset_model_specific_state_for_sequences(&sequence_ids);
        }

        Ok(())
    }
    fn cache(&self) -> &EitherCache {
        self.model.cache()
    }
}

impl MetadataMixin for DecoderPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.core.metadata.clone()
    }
    fn name(&self) -> String {
        self.core.model_id.clone()
    }
    fn release_sequence_state(&self, sequence_id: usize) {
        if let Some(model) = self.model.multimodal() {
            model.reset_model_specific_state_for_sequences(&[sequence_id]);
        }
    }
    fn cleanup_cuda_graphs(&self) {
        #[cfg(feature = "cuda")]
        super::cuda_graph::clear_decode_graphs(&self.core.cuda_decode_graph, self.model.cache());
    }
    fn reclaim_cuda_graph_memory(&self, max_entries: usize) -> usize {
        #[cfg(feature = "cuda")]
        {
            super::cuda_graph::reclaim_decode_graphs(
                &self.core.cuda_decode_graph,
                &*self.model,
                max_entries,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = max_entries;
            0
        }
    }
    fn precapture_cuda_decode_graphs(&self, ctx: &DecodeGraphPrecaptureCtx) {
        #[cfg(feature = "cuda")]
        {
            if let Err(err) = self.precapture_cuda_decode_graphs_impl(ctx) {
                self.core
                    .cuda_decode_graph
                    .lock()
                    .expect("CUDA graph mutex poisoned")
                    .clear();
                warn!("CUDA decode graph precapture failed, graphs will be captured lazily: {err}");
            }
            if let Err(err) = self.model.precapture_speculative_cuda_graphs() {
                warn!(
                    "Speculative CUDA graph precapture failed, graphs will be captured lazily: {err}"
                );
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = ctx;
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.core.tokenizer.clone())
    }
    fn generation_defaults(&self) -> Option<crate::ModelGenerationDefaults> {
        self.core.generation_defaults.clone()
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.core.mapper)
    }
}

impl crate::speculative::driver::SpeculativePipelineExt for DecoderPipeline {
    fn speculative_target(&self) -> &dyn inference_nn::speculative::SpeculativeTargetMixin {
        &*self.model
    }

    fn speculative_target_mut(
        &mut self,
    ) -> &mut dyn inference_nn::speculative::SpeculativeTargetMixin {
        &mut *self.model
    }

    #[cfg(feature = "cuda")]
    fn cuda_sparse_rejection_workspace(
        &self,
    ) -> &StdMutex<Option<crate::speculative::CudaSparseRejectionWorkspace>> {
        &self.core.cuda_sparse_rejection
    }
}

impl DecoderPipeline {
    fn recurrent_batch_kind(
        &self,
        input_ids: &Tensor,
        paged_attn_meta: Option<&PagedAttentionInputMetadata>,
        recurrent_batch_kind: RecurrentBatchKind,
    ) -> inference_tensor::Result<RecurrentBatchKind> {
        if recurrent_batch_kind != RecurrentBatchKind::Decode || self.media.is_none() {
            return Ok(recurrent_batch_kind);
        }
        let seq_len = input_ids.dim(1)?;
        if let Some(metadata) = paged_attn_meta {
            return Ok(
                if !metadata.is_first_prompt_chunk
                    && metadata.num_cached_tokens.is_none()
                    && seq_len == 1
                {
                    RecurrentBatchKind::Decode
                } else {
                    RecurrentBatchKind::Prefill
                },
            );
        }
        Ok(recurrent_batch_kind)
    }
}

#[cfg(feature = "cuda")]
impl DecoderPipeline {
    fn try_cuda_decode_graph_forward(
        &self,
        input: CudaDecodeGraphForwardInput<'_>,
    ) -> inference_tensor::Result<Option<CudaDecodeGraphReplay>> {
        let CudaDecodeGraphForwardInput {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
            model_specific_args,
            recurrent_batch_kind,
        } = input;
        if !cuda_decode_graphs_enabled() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::Disabled,
            );
            return Ok(None);
        }
        if !cuda_decode_graph_batch_kind_supported(recurrent_batch_kind) {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::Prefill,
            );
            return Ok(None);
        }
        if !self.model.supports_cuda_decode_graphs(model_specific_args)
            || !cuda_decode_graph_supported_for_model(self.core.metadata.model_metadata.as_deref())
        {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::ModelUnsupported,
            );
            return Ok(None);
        }
        let speculative = self.model.has_speculative_proposer();
        let Some((kv_cache, metadata)) = paged_attn_meta else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::PagedAttentionUnavailable,
            );
            return Ok(None);
        };
        if metadata.is_first_prompt_chunk || metadata.num_cached_tokens.is_some() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::Prefill,
            );
            return Ok(None);
        }
        if metadata.decode_rows.is_none() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::IncompatibleShape,
            );
            return Ok(None);
        }
        let (batch, q_len) = input_ids.dims2()?;
        if (q_len != 1 && !speculative)
            || seqlen_offsets.len() != batch
            || context_lens.len() != batch
            || position_ids.len() != batch
            || !input_ids.device().is_cuda()
        {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::IncompatibleShape,
            );
            return Ok(None);
        }
        // With a proposer attached every step is a fixed-width verify; the model must expose the
        // outputs the proposer reads so a replay can refresh them.
        if speculative && self.model.take_speculative_graph_state().is_none() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::SpeculativeConflict,
            );
            return Ok(None);
        }
        let Some(bucket) = cuda_graph_batch_bucket(CudaGraphComponent::Target, q_len, batch) else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::BatchUnsupported,
            );
            return Ok(None);
        };
        let Some(_) = self.core.metadata.cache_config.as_ref() else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::CacheConfigUnavailable,
            );
            return Ok(None);
        };
        // Captured kernels require canonical strides, but an already contiguous input needs no copy.
        let input_ids = &input_ids.contiguous()?;

        let mut state = self
            .core
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned");
        if state.disabled() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::RuntimeDisabled,
            );
            return Ok(None);
        }
        let hybrid_slots = if self.model.cache().is_hybrid() {
            let slots = hybrid_graph_slots(&mut self.model.cache().hybrid())?;
            if let Some(slots) = &slots {
                state.observe_recurrent_storage_generation(slots.storage_generation);
            }
            slots
        } else {
            None
        };
        let graph_pad_slot = hybrid_slots.as_ref().map(|_| GDN_PAD_SLOT);
        let Some(step) = CudaGraphDecodeStep::padded(
            CudaGraphDecodeStepInputs {
                input_ids,
                seqlen_offsets,
                context_lens,
                position_ids,
                metadata,
                state_indices: hybrid_slots.as_ref().map(|slots| slots.real.as_slice()),
                pad_slot: graph_pad_slot,
            },
            bucket,
        )?
        else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::PaddingUnavailable,
            );
            return Ok(None);
        };
        let key = CudaDecodeGraphKey::new(&step.input_ids, &step.metadata, recurrent_batch_kind)?;
        if let Some(replay) = state.replay(&key, &step, CudaDecodeGraphReplayInput::Host)? {
            if let Some(spec_state) = replay.spec_state.as_deref()
                && let Err(err) = self.model.install_speculative_graph_state(spec_state)
            {
                state.block_eager_retry();
                return Err(err);
            }
            return Ok(Some(replay));
        }

        let replay_key = key.clone();
        let _ = self.capture_cuda_decode_graph_step(
            &mut state,
            key,
            &step,
            CudaDecodeGraphCaptureInputs {
                kv_cache: kv_cache.as_slice(),
                flash_meta,
                recurrent_batch_kind,
                speculative,
            },
            true,
        )?;
        super::synchronize_cuda_contexts(step.input_ids.device(), self.core.mapper.as_ref())
            .map_err(|err| {
                inference_tensor::Error::msg(format!(
                    "CUDA graph rollback synchronization failed: {err}"
                ))
            })?;
        let replay = state
            .replay(&replay_key, &step, CudaDecodeGraphReplayInput::Host)?
            .ok_or_else(|| {
                inference_tensor::Error::msg("newly captured CUDA decode graph was not replayable")
            })?;
        if let Some(spec_state) = replay.spec_state.as_deref()
            && let Err(err) = self.model.install_speculative_graph_state(spec_state)
        {
            state.block_eager_retry();
            return Err(err);
        }
        record_cuda_graph_dispatch(
            CudaGraphComponent::Target,
            CudaGraphDispatchMode::Eager,
            CudaGraphDispatchReason::CachePopulation,
        );
        Ok(Some(replay))
    }

    fn precapture_cuda_decode_graphs_impl(
        &self,
        ctx: &DecodeGraphPrecaptureCtx,
    ) -> inference_tensor::Result<()> {
        let device = self.device();
        if !cuda_decode_graphs_enabled()
            || !device.is_cuda()
            || !self.model.supports_cuda_decode_graphs(None)
            || !cuda_decode_graph_supported_for_model(self.core.metadata.model_metadata.as_deref())
        {
            return Ok(());
        }
        let (Some(_), Some(cache_engine)) = (
            &self.core.metadata.cache_config,
            &self.core.metadata.cache_engine,
        ) else {
            return Ok(());
        };
        let speculative = self.model.has_speculative_proposer();
        let graph_plans = self
            .model
            .speculative_graph_plans()
            .into_iter()
            .filter(|plan| cuda_graph_startup_capture_allowed(1 + plan.proposal_len))
            .collect::<Vec<_>>();
        let mut widths = vec![(1usize, usize::MAX)];
        for plan in graph_plans {
            if plan.proposal_len > 0 {
                widths.push((
                    1 + plan.proposal_len,
                    plan.max_batch_size.unwrap_or(usize::MAX),
                ));
            }
        }
        widths.sort_unstable();
        widths.dedup();
        let kv_cache = cache_engine.get_kv_cache().clone();
        let hybrid_slots = if self.model.cache().is_hybrid() {
            let mut cache = self.model.cache().hybrid();
            let Some(pad_slot) = cache.graph_pad_slot()? else {
                return Ok(());
            };
            let pad_slot = cache.active_physical_slot(pad_slot)?;
            let pad_slot = u32::try_from(pad_slot).map_err(|_| {
                inference_tensor::Error::msg(format!(
                    "recurrent graph pad slot {pad_slot} exceeds u32"
                ))
            })?;
            Some(pad_slot)
        } else {
            None
        };
        let mut state = self
            .core
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned");
        if state.disabled() {
            return Ok(());
        }
        let start = std::time::Instant::now();
        let mut captured = 0usize;
        let runtime_max_bucket =
            cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 1, ctx.max_batch_size);
        let runtime_batch_shapes = cuda_graph_precapture_batches(CudaGraphComponent::Target, 1)
            .filter(|bucket| *bucket <= runtime_max_bucket)
            .count();
        let startup_max_bucket = widths
            .iter()
            .map(|(q_len, max_bucket)| {
                (*max_bucket).min(cuda_graph_precapture_max_batch(
                    CudaGraphComponent::Target,
                    *q_len,
                    ctx.max_batch_size,
                ))
            })
            .max()
            .unwrap_or(runtime_max_bucket);
        let startup_shapes = widths.iter().fold(0usize, |total, (q_len, max_bucket)| {
            let width_max_bucket = cuda_graph_precapture_max_batch(
                CudaGraphComponent::Target,
                *q_len,
                ctx.max_batch_size,
            );
            total.saturating_add(
                cuda_graph_precapture_batches(CudaGraphComponent::Target, *q_len)
                    .filter(|bucket| *bucket <= (*max_bucket).min(width_max_bucket))
                    .count(),
            )
        });
        state.ensure_capacity(target_cuda_graph_cache_capacity(
            startup_shapes,
            runtime_batch_shapes,
        ));
        for (q_len, max_bucket) in widths {
            let inputs = CudaGraphPrecaptureInputs::new(ctx, q_len, &device, self.device_mapper())?;
            let live = hybrid_slots.map(|pad_slot| vec![pad_slot]);
            let recurrent_batch_kind = if q_len == 1 {
                RecurrentBatchKind::Decode
            } else if speculative {
                RecurrentBatchKind::SpeculativeDecode
            } else {
                RecurrentBatchKind::Prefill
            };
            let graph_pad_slot = hybrid_slots.map(|_| GDN_PAD_SLOT);
            let width_max_bucket = cuda_graph_precapture_max_batch(
                CudaGraphComponent::Target,
                q_len,
                ctx.max_batch_size,
            );
            for bucket in cuda_graph_precapture_batches(CudaGraphComponent::Target, q_len)
                .filter(|bucket| *bucket <= max_bucket.min(width_max_bucket))
            {
                let Some(step) = CudaGraphDecodeStep::padded(
                    inputs.step_inputs(live.as_deref(), graph_pad_slot),
                    bucket,
                )?
                else {
                    continue;
                };
                let key =
                    CudaDecodeGraphKey::new(&step.input_ids, &step.metadata, recurrent_batch_kind)?;
                if state.contains(&key) {
                    continue;
                }
                // The live step's slot table is whatever the pad slot is; the model only sees it
                // through the installed graph buffers
                self.capture_cuda_decode_graph_step(
                    &mut state,
                    key,
                    &step,
                    CudaDecodeGraphCaptureInputs {
                        kv_cache: kv_cache.as_slice(),
                        flash_meta: &inputs.flash_meta,
                        recurrent_batch_kind,
                        speculative,
                    },
                    false,
                )?;
                captured += 1;
            }
        }
        if speculative {
            let _ = self.model.take_speculative_graph_state();
        }
        if captured > 0 {
            info!(
                "Captured {captured} CUDA decode graphs through batch bucket {} in {:.2?}",
                startup_max_bucket,
                start.elapsed()
            );
        }
        Ok(())
    }

    /// Captures after one eager warmup; live calls roll it back so the first replay is canonical.
    fn capture_cuda_decode_graph_step(
        &self,
        state: &mut CudaDecodeGraphState,
        key: CudaDecodeGraphKey,
        step: &CudaGraphDecodeStep,
        inputs: CudaDecodeGraphCaptureInputs<'_>,
        rollback_live_state: bool,
    ) -> inference_tensor::Result<Tensor> {
        let graph_event =
            CudaGraphEventGuard::new(CudaGraphComponent::Target, CudaGraphEvent::Capture);
        let CudaDecodeGraphCaptureInputs {
            kv_cache,
            flash_meta,
            recurrent_batch_kind,
            speculative,
        } = inputs;
        if speculative {
            state.prepare_spec_state_admission_for_key(&key);
        }
        let Device::Cuda(cuda_device) = step.input_ids.device() else {
            inference_tensor::bail!("CUDA graph decode expected CUDA input ids");
        };
        let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();
        let metadata = step
            .metadata
            .materialize_decode_tensors()
            .map_err(inference_tensor::Error::msg)?;

        let uses_recurrent_transition_log = self.model.cache().is_hybrid()
            && self.model.cache().hybrid().uses_recurrent_transition_log();
        if rollback_live_state
            && recurrent_batch_kind == RecurrentBatchKind::Decode
            && self.model.supports_recurrent_speculative_transitions()
            && uses_recurrent_transition_log
            && !self
                .model
                .apply_recurrent_speculative_transitions_for_current_batch()?
        {
            inference_tensor::bail!(
                "CUDA graph capture could not materialize pending recurrent transitions"
            );
        }

        let nonmutating_transition_capture = speculative_decode_logs_transitions(
            self.model.cache(),
            &*self.model,
            recurrent_batch_kind,
        );
        let recurrent_snapshots = snapshot_hybrid_recurrent_checkpoints(
            self.model.cache(),
            &*self.model,
            recurrent_batch_kind,
        )?;
        let live_state_indices = snapshot_hybrid_state_indices(self.model.cache());
        let mut warm_spec_state = None;
        let mut warm_spec_metadata = None;
        let mut warm_live_spec_state = None;
        let mut graph_spec_state = None;
        let capture_attempt: inference_tensor::Result<_> = (|| {
            let state_index_buffers = match &step.state_indices {
                Some(host) => Some(install_hybrid_graph_state_indices(
                    &mut self.model.cache().hybrid(),
                    host,
                )?),
                None => None,
            };
            let mut ctx = ModelForwardContext::new(
                &step.seqlen_offsets,
                &step.context_lens,
                &step.position_ids,
                Some((kv_cache, &metadata)),
                flash_meta,
            )
            .with_recurrent_cache(self.model.cache(), recurrent_batch_kind);
            let warmup_logits = self.model.forward(&step.input_ids, None, None, &mut ctx)?;
            warm_spec_state = speculative
                .then(|| self.model.take_speculative_graph_state())
                .flatten();
            warm_spec_metadata = warm_spec_state
                .as_deref()
                .map(speculative_graph_tensor_metadata);
            if !nonmutating_transition_capture {
                warm_live_spec_state = warm_spec_state
                    .as_deref()
                    .map(|state| state.for_real_batch(step.real_batch))
                    .transpose()?;
            }
            step.input_ids.device().synchronize()?;
            let live_logits = step.narrow_rows(&warmup_logits)?;

            let spec_state_usage = warm_spec_state
                .as_deref()
                .map(|warm| state.prepare_spec_state_admission(warm))
                .transpose()?;
            if nonmutating_transition_capture {
                warm_spec_state = None;
            }
            // CUDA stream capture records recurrent writes without executing them.
            let entry = capture_cuda_decode_graph(
                CudaDecodeGraphCaptureCtx {
                    key,
                    input_ids: &step.input_ids,
                    seqlen_offsets: &step.seqlen_offsets,
                    position_ids: &step.position_ids,
                    kv_cache,
                    metadata: &metadata,
                    model_metadata: self.core.metadata.model_metadata.as_deref(),
                    activation_dtype: self.core.metadata.activation_dtype,
                    warmup_logits: &warmup_logits,
                    state_indices: state_index_buffers,
                    real_batch: step.real_batch,
                },
                |graph_input_ids, graph_metadata| {
                    let mut ctx = ModelForwardContext::new(
                        &step.seqlen_offsets,
                        &step.context_lens,
                        &step.position_ids,
                        Some((kv_cache, graph_metadata)),
                        flash_meta,
                    )
                    .with_recurrent_cache(self.model.cache(), recurrent_batch_kind);
                    let logits = self.model.forward(graph_input_ids, None, None, &mut ctx)?;
                    if let Some(expected) = warm_spec_metadata.as_deref() {
                        let captured =
                            self.model.take_speculative_graph_state().ok_or_else(|| {
                                inference_tensor::Error::msg(
                                    "captured forward left no speculative state",
                                )
                            })?;
                        let actual = speculative_graph_tensor_metadata(&*captured);
                        validate_speculative_graph_tensor_metadata(expected, &actual)?;
                        graph_spec_state = Some(captured);
                    }
                    Ok(logits)
                },
            )?;
            Ok((
                live_logits,
                entry.with_spec_state(graph_spec_state.take(), spec_state_usage),
            ))
        })();
        let capture_attempt = if let Some(warm) = warm_live_spec_state.as_deref() {
            restore_hybrid_state_indices(self.model.cache(), live_state_indices.as_ref());
            match self.model.install_speculative_graph_state(warm) {
                Ok(()) => capture_attempt,
                Err(install_err) => {
                    state.block_eager_retry();
                    match capture_attempt {
                        Ok(_) => Err(install_err),
                        Err(capture_err) => Err(inference_tensor::Error::msg(format!(
                            "CUDA graph capture failed: {capture_err}; warm speculative state restoration failed: {install_err}"
                        ))),
                    }
                }
            }
        } else {
            capture_attempt
        };
        let (logits, entry) = finish_cuda_graph_capture_attempt(
            self.model.cache(),
            state,
            capture_attempt,
            recurrent_snapshots.as_deref(),
            live_state_indices.as_ref(),
            rollback_live_state,
        )?;
        state.insert(entry);
        graph_event.success();
        Ok(logits)
    }
}

impl Pipeline for DecoderPipeline {
    fn requires_uniform_prompt_batch(&self) -> bool {
        match &self.model {
            DecoderModel::Text(_) => text_requires_uniform_prompt_batch(
                self.model.cache().is_hybrid(),
                self.supports_packed_prefill(),
                self.model.has_speculative_proposer(),
            ),
            DecoderModel::Multimodal(_) => !self.supports_packed_prefill(),
        }
    }

    fn requires_uniform_completion_batch(&self) -> bool {
        self.model
            .multimodal()
            .is_some_and(|model| model.requires_uniform_completion_batch())
    }

    fn requires_uniform_media_batch(&self) -> bool {
        self.model
            .multimodal()
            .is_some_and(|model| !model.supports_mixed_media_batches())
    }

    fn supports_packed_prefill(&self) -> bool {
        self.model.supports_packed_prefill()
            && self.core.metadata.cache_engine.is_some()
            && (!self.model.has_speculative_proposer()
                || self.model.supports_speculative_packed_prefill())
            && self.model.device().is_cuda()
            && self
                .core
                .mapper
                .get_unique_devices()
                .iter()
                .all(Device::is_cuda)
            && crate::using_flash_attn()
            && crate::attention::flash_backend_supports_sdpa(
                self.model.config().k_head_dim,
                false,
                self.core.metadata.sliding_window.is_some(),
            )
            && matches!(
                self.core.metadata.activation_dtype,
                DType::F16 | DType::BF16
            )
    }

    fn supports_batched_cuda_sampling(&self) -> bool {
        !self.model.has_speculative_proposer()
    }

    fn supports_speculative_prompt_bootstrap(&self) -> bool {
        self.model.supports_speculative_prompt_bootstrap()
    }

    fn speculative_prefix_replay(&self) -> crate::speculative::SpeculativePrefixReplay {
        self.model.speculative_prefix_replay()
    }

    fn supports_paged_auxiliary_prefix_state(&self) -> bool {
        self.model.supports_paged_auxiliary_prefix_state()
    }

    fn capture_paged_auxiliary_prefix_state(
        &mut self,
        sequence_id: usize,
        cached_tokens: usize,
    ) -> inference_tensor::Result<Option<Arc<dyn crate::kv_cache::PagedAuxiliaryPrefixState>>> {
        self.model
            .capture_paged_auxiliary_prefix_state(sequence_id, cached_tokens)
    }

    fn restore_paged_auxiliary_prefix_state(
        &mut self,
        sequence_id: usize,
        cached_tokens: usize,
        state: &dyn crate::kv_cache::PagedAuxiliaryPrefixState,
    ) -> inference_tensor::Result<()> {
        self.model
            .restore_paged_auxiliary_prefix_state(sequence_id, cached_tokens, state)
    }

    fn adapter_runtime(&self) -> Option<Arc<DynamicLoraRuntime>> {
        self.core.dynamic_lora.clone()
    }

    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> inference_tensor::Result<ForwardInputsResult> {
        Ok(self.forward_step(inputs, return_raw_logits)?.output)
    }

    fn forward_step(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> inference_tensor::Result<ForwardStepResult> {
        let StepInputs {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args,
            paged_attn_meta,
            flash_meta,
            recurrent_batch_kind,
            adapter_leases,
        } = StepInputs::from_processor_output(&self.model, inputs);
        let lora_execution = super::resolve_lora_execution(
            self.core.dynamic_lora.as_deref(),
            &input_ids,
            paged_attn_meta.as_ref(),
            &flash_meta,
            &adapter_leases,
        )?;
        let metadata = self.get_metadata();
        let paged_attn_meta = match (&metadata.cache_engine, &paged_attn_meta) {
            (Some(engine), Some(meta)) => Some((engine.get_kv_cache().clone(), meta)),
            (Some(_), None) => {
                // This can happen if Rust-side user code is wrong
                inference_tensor::bail!(
                    "Forward step expected a PagedAttention input metadata. This was not provided, please ensure that the scheduler config is correctly configured for PagedAttention."
                )
            }
            (None, Some(_)) => {
                // This should never happen but we handle it anyway
                inference_tensor::bail!(
                    "Forward step got a PagedAttention input metadata but there is no cache engine. Please raise an issue."
                )
            }
            (None, None) => None,
        };
        let recurrent_batch_kind = self.recurrent_batch_kind(
            &input_ids,
            paged_attn_meta.as_ref().map(|(_, meta)| *meta),
            recurrent_batch_kind,
        )?;
        if self.model.has_speculative_proposer() {
            *self
                .last_prompt_attention
                .lock()
                .expect("prompt attention mutex poisoned") = paged_attn_meta
                .as_ref()
                .filter(|(_, meta)| meta.is_first_prompt_chunk || meta.num_cached_tokens.is_some())
                .map(|(_, meta)| ((*meta).clone(), flash_meta.clone()));
        }
        #[cfg(feature = "cuda")]
        let mut cuda_graph_eager_fallback = None;
        #[cfg(feature = "cuda")]
        if lora_execution.is_none() && !return_raw_logits && pixel_values.is_none() {
            match self.try_cuda_decode_graph_forward(CudaDecodeGraphForwardInput {
                input_ids: &input_ids,
                seqlen_offsets: &seqlen_offsets,
                context_lens: &context_lens,
                position_ids: &position_ids,
                paged_attn_meta: paged_attn_meta.as_ref().map(|(a, b)| (a.clone(), *b)),
                flash_meta: &flash_meta,
                model_specific_args: model_specific_args.as_deref(),
                recurrent_batch_kind,
            }) {
                Ok(Some(replay)) => {
                    return Ok(ForwardStepResult::cuda_decode(
                        ForwardInputsResult::CausalGeneration {
                            logits: replay.logits,
                        },
                        replay.launch,
                    ));
                }
                Ok(None) => {}
                Err(err) => {
                    if !disable_cuda_decode_graph(
                        &self.core.cuda_decode_graph,
                        self.model.cache(),
                        &err,
                    ) {
                        return Err(err);
                    }
                    cuda_graph_eager_fallback = Some(CudaGraphEventGuard::new(
                        CudaGraphComponent::Target,
                        CudaGraphEvent::EagerFallback,
                    ));
                }
            }
        }
        let paged_attn_meta = paged_attn_meta
            .map(|(kv_cache, metadata)| {
                metadata
                    .materialize_decode_tensors()
                    .map(|metadata| (kv_cache, metadata))
            })
            .transpose()
            .map_err(inference_tensor::Error::msg)?;
        let mut ctx = ModelForwardContext::new(
            &seqlen_offsets,
            &context_lens,
            &position_ids,
            paged_attn_meta
                .as_ref()
                .map(|(kv_cache, meta)| (kv_cache.as_slice(), meta)),
            &flash_meta,
        )
        .with_recurrent_cache(self.model.cache(), recurrent_batch_kind);
        let eager_result = inference_quant::with_lora_execution(lora_execution, || {
            self.model
                .forward(&input_ids, pixel_values, model_specific_args, &mut ctx)
        });
        #[cfg(feature = "cuda")]
        if eager_result.is_ok()
            && let Some(graph_event) = cuda_graph_eager_fallback.take()
        {
            graph_event.success();
        }
        let logits = eager_result?;
        if let Some(model) = self.model.multimodal()
            && model.is_block_diffusion()
            && !return_raw_logits
        {
            return Ok(ForwardStepResult::eager(
                ForwardInputsResult::BlockGeneration {
                    token_blocks: logits
                        .to_dtype(inference_tensor::DType::U32)?
                        .to_vec2::<u32>()?,
                    denoise_time: model.take_block_denoise_time().unwrap_or_default(),
                },
            ));
        }
        let output = if return_raw_logits {
            ForwardInputsResult::RawLogits { logits }
        } else {
            ForwardInputsResult::CausalGeneration { logits }
        };
        Ok(ForwardStepResult::eager(output))
    }

    #[cfg(feature = "cuda")]
    fn replay_cuda_decode_one_token(
        &mut self,
        launch: CudaDecodeGraphLaunch,
    ) -> inference_tensor::Result<Option<ForwardStepResult>> {
        let replay = {
            let mut state = self
                .core
                .cuda_decode_graph
                .lock()
                .expect("CUDA graph mutex poisoned");
            if state.disabled() {
                return Ok(None);
            }
            state.replay_one_token(launch)
        };
        match replay {
            Ok(Some(replay)) => {
                if let Some(spec_state) = replay.spec_state.as_deref()
                    && let Err(err) = self.model.install_speculative_graph_state(spec_state)
                {
                    let _ = disable_cuda_decode_graph(
                        &self.core.cuda_decode_graph,
                        self.model.cache(),
                        &err,
                    );
                    return Err(err);
                }
                Ok(Some(ForwardStepResult::cuda_decode(
                    ForwardInputsResult::CausalGeneration {
                        logits: replay.logits,
                    },
                    replay.launch,
                )))
            }
            Ok(None) => Ok(None),
            Err(err) => {
                let _ = disable_cuda_decode_graph(
                    &self.core.cuda_decode_graph,
                    self.model.cache(),
                    &err,
                );
                Err(err)
            }
        }
    }

    fn attach_speculative(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
    ) -> inference_tensor::Result<()> {
        self.attach_speculative_with_runtime(
            config,
            crate::speculative::MtpRuntimeConfig::default(),
        )
    }

    fn attach_speculative_with_runtime(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
        runtime: crate::speculative::MtpRuntimeConfig,
    ) -> inference_tensor::Result<()> {
        if self.core.dynamic_lora.is_some() {
            inference_tensor::bail!("dynamic LoRA does not support speculative decoding");
        }
        if matches!(config, crate::speculative::SpeculativeConfig::Mtp(_))
            && self.get_metadata().cache_engine.is_none()
        {
            inference_tensor::bail!(
                "MTP speculative decoding currently requires PagedAttention for this pipeline."
            );
        }
        if matches!(config, crate::speculative::SpeculativeConfig::Mtp(_)) {
            self.cleanup_cuda_graphs();
            self.model.disable_recurrent_decode_deferred_storage()?;
        }
        let config = crate::speculative::resolve_speculative_model(config)?;
        if let Some(info) = self
            .model
            .attach_speculative_with_runtime(config, runtime)?
        {
            self.model.log_speculative_attach(&info);
        }
        Ok(())
    }

    fn release_speculative_sequences(&mut self, seq_ids: &[usize]) -> inference_tensor::Result<()> {
        self.model.release_speculative_sequences(seq_ids)
    }

    fn flush_recurrent_speculative_transitions(
        &self,
        seq_ids: &[usize],
    ) -> inference_tensor::Result<()> {
        self.model.flush_recurrent_speculative_transitions(seq_ids)
    }

    fn speculative_prompt_chunk(
        &mut self,
        seqs: &[&mut Sequence],
        chunk: &crate::pipeline::SpeculativePromptChunk,
        metadata: &crate::paged_attention::PagedAttentionMeta,
    ) -> inference_tensor::Result<()> {
        if !self.model.has_speculative_proposer() {
            return Ok(());
        }
        let general_metadata = self.get_metadata();
        let Some(cache_engine) = general_metadata.cache_engine.as_ref() else {
            return Ok(());
        };
        let kv_cache = cache_engine.get_kv_cache().clone();
        let seq_ids = chunk
            .rows
            .iter()
            .map(|row| *seqs[row.seq_idx].id())
            .collect::<Vec<_>>();
        let batch_indices = (0..chunk.rows.len()).collect::<Vec<_>>();
        let tokens = chunk
            .rows
            .iter()
            .map(|row| row.tokens.as_slice())
            .collect::<Vec<_>>();
        let chunk_ranges = chunk.rows.iter().map(|row| row.range).collect::<Vec<_>>();
        let target_attention = self
            .last_prompt_attention
            .lock()
            .expect("prompt attention mutex poisoned")
            .take();
        self.model
            .speculative_prefill(crate::speculative::SpeculativePrefillCtx {
                seq_ids: &seq_ids,
                batch_indices: &batch_indices,
                tokens: &tokens,
                chunk_ranges: &chunk_ranges,
                is_final_prompt_chunk: chunk.is_final_prompt_chunk,
                cache: crate::speculative::SpeculativeKvCache::Paged {
                    metadata,
                    kv_cache: &kv_cache,
                },
                target_attention: target_attention.as_ref().map(|(metadata, flash_params)| {
                    crate::speculative::TargetAttentionInputs {
                        metadata,
                        flash_params,
                    }
                }),
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn try_sample_speculative_causal_gen<'a>(
        &'a mut self,
        seqs: &'a mut [&mut Sequence],
        logits: &'a [Tensor],
        batched_logits: Option<&'a Tensor>,
        prefix_cacher: &'a mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        metadata: Option<crate::paged_attention::PagedAttentionMeta>,
        logger: &'a crate::IntervalLogger,
    ) -> BoxFuture<'a, inference_tensor::Result<bool>> {
        Box::pin(async move {
            if !self.model.has_speculative_proposer() {
                crate::speculative::driver::clear_staged_speculative_tokens(seqs);
                return Ok(false);
            }

            let general_metadata = self.get_metadata();
            if let Some(cache_engine) = general_metadata.cache_engine.as_ref() {
                let Some(metadata) = metadata else {
                    crate::speculative::driver::clear_staged_speculative_tokens(seqs);
                    return Ok(false);
                };
                let cache = crate::speculative::cache::PagedSpeculativeCacheAccess::new(
                    &metadata,
                    cache_engine,
                );
                return crate::speculative::driver::try_sample_speculative_causal_gen(
                    self,
                    seqs,
                    logits,
                    batched_logits,
                    prefix_cacher,
                    disable_eos_stop,
                    rng,
                    &cache,
                    logger,
                )
                .await;
            }

            crate::speculative::driver::clear_staged_speculative_tokens(seqs);
            Ok(false)
        })
    }

    fn try_sample_causal_gen_batched<'a>(
        &'a self,
        seqs: &'a mut [&mut Sequence],
        logits: &'a Tensor,
        prefix_cacher: &'a mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> BoxFuture<'a, Result<bool, inference_tensor::Error>> {
        if self.model.has_speculative_proposer() {
            return Box::pin(std::future::ready(Ok(false)));
        }
        crate::speculative::driver::clear_staged_speculative_tokens(seqs);
        Box::pin(
            sample_and_add_toks_batched(
                self,
                seqs,
                logits.clone(),
                prefix_cacher,
                disable_eos_stop,
                rng,
            )
            .map(|result| result.map(|()| true)),
        )
    }

    fn sample_causal_gen<'a>(
        &'a self,
        seqs: &'a mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &'a mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> BoxFuture<'a, Result<(), inference_tensor::Error>> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng)
    }

    fn sample_block_gen<'a>(
        &'a self,
        input_seqs: &'a mut [&mut Sequence],
        token_blocks: Vec<Vec<u32>>,
        denoise_times: Vec<std::time::Duration>,
        prefix_cacher: &'a mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
    ) -> BoxFuture<'a, Result<(), inference_tensor::Error>> {
        crate::pipeline::sampling::finalize_block_gen(
            self,
            input_seqs,
            token_blocks,
            denoise_times,
            prefix_cacher,
            disable_eos_stop,
        )
    }
    fn category(&self) -> ModelCategory {
        let text_only = matches!(
            self.core.metadata.modalities.input.as_slice(),
            [crate::SupportedModality::Text]
        );
        match &self.media {
            Some(media) if !text_only => ModelCategory::Multimodal {
                prefixer: media.prefixer.clone(),
                video_sampling: media.video_sampling,
            },
            _ => ModelCategory::Text,
        }
    }

    fn encoder_cache_counters(&self) -> Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)> {
        self.model.multimodal()?.encoder_cache_counters()
    }
}

impl AnyMoePipelineMixin for DecoderPipeline {
    fn amoe_finish_training(
        &mut self,
        gate_model_id: Option<String>,
    ) -> inference_tensor::Result<()> {
        self.model.amoe_mut().finish_training(gate_model_id)
    }
    fn amoe_layer_vars(&self) -> Vec<Vec<Var>> {
        self.model.amoe().get_vars()
    }
    fn amoe_base_model_trainable_params(&self) -> usize {
        self.model.amoe().trainable_params()
    }
    fn amoe_take_cached_gating_outputs(&mut self) -> Vec<Tensor> {
        self.model.amoe_mut().take_cached_gating_outputs()
    }
    fn amoe_create_layers(
        &mut self,
        model_ids: Vec<String>,
        token: &TokenSource,
        revision: Option<String>,
        match_regex: &str,
        config: crate::amoe::AnyMoeConfig,
        dtype: inference_tensor::DType,
        dev: &Device,
        (prefix, mlp): (String, String),
        layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        silent: bool,
        gate_model_id: Option<String>,
    ) -> inference_tensor::Result<()> {
        let (vbs, gate_vb) = super::amoe::load_anymoe_weights(super::amoe::AnyMoeWeightSources {
            model_ids,
            token,
            revision,
            match_regex,
            dtype,
            dev,
            layers: &layers,
            silent,
            gate_model_id,
        })?;
        self.model.amoe_mut().create_anymoe_layers(
            vbs,
            config,
            (prefix, mlp),
            layers,
            expert_type,
            gate_vb,
        )
    }
    fn amoe_supported(&self) -> bool {
        self.model.amoe().amoe_supported()
    }
}

#[cfg(test)]
mod speculative_graph_tensor_metadata_tests {
    use super::{SpeculativeGraphTensorMetadata, validate_speculative_graph_tensor_metadata};
    use inference_tensor::{DType, DeviceLocation};

    fn metadata(
        shape: &[usize],
        strides: &[usize],
        contiguous: bool,
        dtype: DType,
    ) -> SpeculativeGraphTensorMetadata {
        SpeculativeGraphTensorMetadata {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            contiguous,
            dtype,
            device: DeviceLocation::Cpu,
        }
    }

    #[test]
    fn validates_tensor_count_and_metadata() {
        let expected = vec![metadata(&[4, 8, 16], &[128, 16, 1], true, DType::BF16)];
        assert!(validate_speculative_graph_tensor_metadata(&expected, &expected).is_ok());
        assert!(validate_speculative_graph_tensor_metadata(&expected, &[]).is_err());
        assert!(
            validate_speculative_graph_tensor_metadata(
                &expected,
                &[metadata(&[4, 8, 17], &[136, 17, 1], true, DType::BF16)]
            )
            .is_err()
        );
        assert!(
            validate_speculative_graph_tensor_metadata(
                &expected,
                &[metadata(&[4, 8, 16], &[128, 16, 1], true, DType::F16)]
            )
            .is_err()
        );
        assert!(
            validate_speculative_graph_tensor_metadata(
                &expected,
                &[metadata(&[4, 8, 16], &[1, 64, 4], false, DType::BF16)]
            )
            .is_err()
        );
        assert!(
            validate_speculative_graph_tensor_metadata(
                &expected,
                &[metadata(&[4, 8, 16], &[128, 16, 1], false, DType::BF16)]
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::text_requires_uniform_prompt_batch;

    #[test]
    fn hybrid_models_require_uniform_prompts_until_packed_prefill_is_proven() {
        assert!(text_requires_uniform_prompt_batch(true, false, false));
        assert!(!text_requires_uniform_prompt_batch(true, true, false));
        assert!(!text_requires_uniform_prompt_batch(false, false, false));
    }

    #[test]
    fn unsupported_speculative_models_remain_uniform() {
        assert!(text_requires_uniform_prompt_batch(true, false, true));
        assert!(!text_requires_uniform_prompt_batch(true, true, true));
    }
}

//! The interface between the engine and a multimodal input processor, and the types that cross it.

use std::{any::Any, ops::Range, sync::Arc};

use anyhow::Result;
use image::DynamicImage;
use inference_audio::AudioInput;
use inference_tensor::{Device, Tensor, WithDType};
use tokenizers::Tokenizer;

use super::{media::MultimodalData, video::VideoInput};
use crate::{
    attention::FlashParams,
    device_map::DeviceMapper,
    gdn::RecurrentBatchKind,
    model::BlockDenoisingProgressEmitter,
    paged_attention::{
        PagedAttentionInputMetadata, PagedAttentionMeta,
        block_hash::{MultiModalFeature, MultimodalKind, noncausal_mm_ranges},
    },
};

pub struct InputMetadata {
    pub input: Tensor,
    pub positions: Vec<usize>,
    pub context_lens: Vec<(usize, usize)>, // (start index, len)
    pub position_ids: Vec<usize>,
    pub paged_attn_meta: Option<PagedAttentionInputMetadata>, // For paged attention
    pub flash_meta: FlashParams,
}

pub struct InnerInputProcessorOutput {
    pub inputs: InputMetadata,
    pub seq_indices: Vec<usize>,
}

/// Prompt token slices of either id type; Phi-3V uses negative placeholder ids, so it needs i64.
pub enum PromptTokens<'a> {
    U32(Vec<&'a [u32]>),
    I64(Vec<&'a [i64]>),
}

impl PromptTokens<'_> {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        match self {
            Self::U32(toks) => toks.len(),
            Self::I64(toks) => toks.len(),
        }
    }

    pub fn seq_len(&self, i: usize) -> usize {
        match self {
            Self::U32(toks) => toks[i].len(),
            Self::I64(toks) => toks[i].len(),
        }
    }

    /// Sequence `i` from `start`, zero-padded to `len` tokens, as a 1-D tensor.
    pub fn padded_tensor(
        &self,
        i: usize,
        start: usize,
        len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        fn pad<T: WithDType>(toks: &[T], len: usize, device: &Device) -> Result<Tensor> {
            let mut ids = toks.to_vec();
            ids.resize(len, T::zero());
            Ok(Tensor::new(ids, device)?)
        }
        match self {
            Self::U32(toks) => pad(&toks[i][start..], len, device),
            Self::I64(toks) => pad(&toks[i][start..], len, device),
        }
    }
}

impl<'a> From<Vec<&'a [u32]>> for PromptTokens<'a> {
    fn from(toks: Vec<&'a [u32]>) -> Self {
        Self::U32(toks)
    }
}

impl<'a> From<Vec<&'a [i64]>> for PromptTokens<'a> {
    fn from(toks: Vec<&'a [i64]>) -> Self {
        Self::I64(toks)
    }
}

/// A request the processor rejects as malformed rather than failing on internally.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct InputsProcessorValidationError(pub String);

pub struct InputProcessorOutput {
    pub inputs: Box<dyn Any>,
    pub seq_indices: Vec<usize>,
}

/// What a multimodal processor builds for its model; the engine adds the adapter leases.
pub struct ModelInputs {
    pub input_ids: Tensor,
    pub seqlen_offsets: Vec<usize>,
    pub context_lens: Vec<(usize, usize)>,
    pub position_ids: Vec<usize>,
    pub pixel_values: Option<Tensor>,
    pub model_specific_args: Box<dyn Any>,
    pub paged_attn_meta: Option<PagedAttentionInputMetadata>,
    pub flash_meta: FlashParams,
    pub recurrent_batch_kind: RecurrentBatchKind,
}

/// The engine sequence as an input processor sees it.
pub trait MediaSequence {
    fn id(&self) -> &usize;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get_toks(&self) -> &[u32];
    fn is_chunked_prefill_view(&self) -> bool;
    fn prompt_position_source_toks(&self) -> &[u32];
    fn active_prompt_query_range(&self) -> Option<Range<usize>>;
    fn active_prompt_local_query_range(&self) -> Option<Range<usize>>;
    fn active_multimodal_item_range(&self, kind: MultimodalKind) -> Option<Range<usize>>;
    fn active_local_multimodal_item_range(
        &self,
        kind: MultimodalKind,
        available_items: usize,
    ) -> Option<Range<usize>>;
    fn prefix_cache_len(&self) -> usize;
    fn set_toks_and_reallocate(
        &mut self,
        toks: Vec<u32>,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    );
    fn set_initial_prompt(&mut self, new: String);
    fn get_initial_prompt(&self) -> &str;
    fn set_prefill_toks(&mut self, toks: Vec<u32>);
    fn has_prefill_toks(&self) -> bool;
    fn set_max_len(&mut self, max_len: usize);
    fn mm_features(&self) -> &[MultiModalFeature];
    fn set_mm_features(&mut self, features: Vec<MultiModalFeature>);
    fn count_prefix_cached_mm_items(&self) -> usize;
    fn count_prefix_cached_mm_items_by_kind(&self, kind: MultimodalKind) -> usize;
    fn images(&self) -> Option<&[DynamicImage]>;
    fn clone_images(&self) -> Option<Vec<DynamicImage>>;
    fn take_images(&mut self) -> Option<Vec<DynamicImage>>;
    fn image_hashes(&self) -> Option<&[u64]>;
    fn has_images(&self) -> bool;
    fn keep_num_images(&mut self, images_to_keep: usize);
    fn audios(&self) -> Option<&[AudioInput]>;
    fn clone_audios(&self) -> Option<Vec<AudioInput>>;
    fn take_audios(&mut self) -> Option<Vec<AudioInput>>;
    fn audio_hashes(&self) -> Option<&[u64]>;
    fn has_audios(&self) -> bool;
    fn videos(&self) -> Option<&[VideoInput]>;
    fn clone_videos(&self) -> Option<Vec<VideoInput>>;
    fn take_videos(&mut self) -> Option<Vec<VideoInput>>;
    fn video_hashes(&self) -> Option<&[u64]>;
    fn has_videos(&self) -> bool;
    fn multimodal(&self) -> &MultimodalData;
    fn multimodal_mut(&mut self) -> &mut MultimodalData;
    /// The engine's own sequence type, for the host's callbacks.
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Everything a text input builder needs besides the tokens and sequences.
pub struct TextInputs<'a> {
    pub device: &'a Device,
    pub last_n_context_len: Option<(usize, usize)>,
    pub return_raw_logits: bool,
    pub paged_attn_metadata: Option<&'a mut PagedAttentionMeta>,
    pub mapper: Option<&'a dyn DeviceMapper>,
    pub sliding_window: Option<usize>,
}

/// The engine's text-only inputs for a batch without media, for a processor to wrap in its `ModelInputs`.
pub struct TextOnlyInputs {
    pub input_ids: Tensor,
    pub seqlen_offsets: Vec<usize>,
    pub context_lens: Vec<(usize, usize)>,
    pub position_ids: Vec<usize>,
    pub paged_attn_meta: Option<PagedAttentionInputMetadata>,
    pub flash_meta: FlashParams,
    pub recurrent_batch_kind: RecurrentBatchKind,
    pub seq_indices: Vec<usize>,
}

/// Arguments of a whole-batch `process_inputs` call, for the processors that hand text-only batches back.
pub struct ProcessInputsCall<'a> {
    pub tokenizer: Option<Arc<Tokenizer>>,
    pub is_prompt: bool,
    pub device: &'a Device,
    pub no_kv_cache: bool,
    pub last_n_context_len: Option<(usize, usize)>,
    pub return_raw_logits: bool,
    pub sliding_window: Option<usize>,
    pub other_config: Option<Arc<dyn Any>>,
    pub paged_attn_metadata: Option<PagedAttentionMeta>,
    pub mapper: Option<&'a dyn DeviceMapper>,
}

/// The engine services an input processor calls back into.
pub trait InputsHost {
    /// Prompt-step token, position and attention metadata for `seqs`.
    fn prompt_inputs(
        &self,
        toks: PromptTokens<'_>,
        seqs: &[&mut dyn MediaSequence],
        text: TextInputs<'_>,
    ) -> Result<InnerInputProcessorOutput>;
    /// Decode-step metadata; `decode_window` limits each sequence to its last tokens.
    fn completion_inputs(
        &self,
        toks: PromptTokens<'_>,
        seqs: &[&mut dyn MediaSequence],
        text: TextInputs<'_>,
        no_kv_cache: bool,
        decode_window: Option<usize>,
    ) -> Result<InnerInputProcessorOutput>;
    /// The engine's text-only processing of the whole batch.
    fn text_only_inputs(
        &self,
        seqs: &mut [&mut dyn MediaSequence],
        call: ProcessInputsCall<'_>,
    ) -> Result<TextOnlyInputs>;
    /// Streaming progress sinks for block-diffusion denoising, one per streaming sequence in `seq_indices`.
    fn block_denoising_progress(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        seqs: &[&mut dyn MediaSequence],
        seq_indices: &[usize],
        return_raw_logits: bool,
    ) -> Option<Vec<BlockDenoisingProgressEmitter>>;
    /// The draft width shared by every sequence's staged speculative tokens, when they agree.
    fn staged_batch_width(&self, seqs: &[&mut dyn MediaSequence]) -> Option<usize>;
}

/// Records each sequence's non-causal multimodal spans on the attention metadata.
pub trait NoncausalMmContext {
    fn set_noncausal_mm_context<S>(&mut self, input_seqs: &[S])
    where
        S: std::ops::Deref,
        S::Target: MediaSequence;
    fn set_noncausal_mm_context_views<S>(&mut self, input_seqs: &[S], include_full_attention: bool)
    where
        S: std::ops::Deref,
        S::Target: MediaSequence;
}

impl NoncausalMmContext for PagedAttentionMeta {
    fn set_noncausal_mm_context<S>(&mut self, input_seqs: &[S])
    where
        S: std::ops::Deref,
        S::Target: MediaSequence,
    {
        self.set_noncausal_mm_context_views(input_seqs, true);
    }

    fn set_noncausal_mm_context_views<S>(&mut self, input_seqs: &[S], include_full_attention: bool)
    where
        S: std::ops::Deref,
        S::Target: MediaSequence,
    {
        self.mm_prefix_ranges_by_seq_id.clear();
        self.full_mm_prefix_ranges_by_seq_id.clear();
        for seq in input_seqs {
            let full_ranges = noncausal_mm_ranges(seq.mm_features(), None);
            if !full_ranges.is_empty() {
                if include_full_attention {
                    self.full_mm_prefix_ranges_by_seq_id
                        .insert(*seq.id(), full_ranges.clone());
                }
                self.mm_prefix_ranges_by_seq_id
                    .insert(*seq.id(), full_ranges);
            }
        }
        self.has_noncausal_mm_context = !self.mm_prefix_ranges_by_seq_id.is_empty()
            || !self.full_mm_prefix_ranges_by_seq_id.is_empty();
    }
}

/// A model's multimodal input preprocessing: media into tensors, and the prompt rewritten around it.
pub trait MultimodalInputsProcessor: Send + Sync {
    fn prepare_for_paged_prompt_planning(
        &self,
        _tokenizer: Option<Arc<Tokenizer>>,
        _input_seqs: &mut [&mut dyn MediaSequence],
        _device: &Device,
        _other_config: Option<Arc<dyn Any>>,
        _paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<()> {
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn process_inputs(
        &self,
        host: &dyn InputsHost,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut dyn MediaSequence],
        is_prompt: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        sliding_window: Option<usize>,
        other_config: Option<Arc<dyn Any>>,
        paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput>;
}

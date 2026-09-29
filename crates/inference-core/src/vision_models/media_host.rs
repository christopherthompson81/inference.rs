//! The engine side of the multimodal input-processor interface: `Sequence`, the host, and the processor adapter.

use std::{any::Any, ops::Range, sync::Arc};

use anyhow::Result;
use candle_core::Device;
use image::DynamicImage;
use inference_nn::media_inputs::{
    media::MultimodalData,
    processor::{
        InnerInputProcessorOutput, InputsHost, MediaSequence, ModelInputs as MediaModelInputs,
        MultimodalInputsProcessor, ProcessInputsCall, PromptTokens, TextInputs, TextOnlyInputs,
    },
    video::VideoInput,
};
use inference_nn::model::BlockDenoisingProgressEmitter;
use tokenizers::Tokenizer;

use crate::{
    AudioInput,
    device_map::DeviceMapper,
    paged_attention::{
        PagedAttentionMeta,
        block_hash::{MultiModalFeature, MultimodalKind},
    },
    pipeline::{
        InputProcessorOutput, InputsProcessor, InputsProcessorType,
        text_models_inputs_processor::{
            ModelInputs as TextModelInputs, TextInputsProcessor, get_completion_input,
            get_completion_input_windowed, get_prompt_input,
        },
    },
    sequence::Sequence,
};

use super::ModelInputs;

impl MediaSequence for Sequence {
    fn id(&self) -> &usize {
        Sequence::id(self)
    }
    fn len(&self) -> usize {
        Sequence::len(self)
    }
    fn get_toks(&self) -> &[u32] {
        Sequence::get_toks(self)
    }
    fn is_chunked_prefill_view(&self) -> bool {
        Sequence::is_chunked_prefill_view(self)
    }
    fn prompt_position_source_toks(&self) -> &[u32] {
        Sequence::prompt_position_source_toks(self)
    }
    fn active_prompt_query_range(&self) -> Option<Range<usize>> {
        Sequence::active_prompt_query_range(self)
    }
    fn active_prompt_local_query_range(&self) -> Option<Range<usize>> {
        Sequence::active_prompt_local_query_range(self)
    }
    fn active_multimodal_item_range(&self, kind: MultimodalKind) -> Option<Range<usize>> {
        Sequence::active_multimodal_item_range(self, kind)
    }
    fn active_local_multimodal_item_range(
        &self,
        kind: MultimodalKind,
        available_items: usize,
    ) -> Option<Range<usize>> {
        Sequence::active_local_multimodal_item_range(self, kind, available_items)
    }
    fn prefix_cache_len(&self) -> usize {
        Sequence::prefix_cache_len(self)
    }
    fn set_toks_and_reallocate(
        &mut self,
        toks: Vec<u32>,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) {
        Sequence::set_toks_and_reallocate(self, toks, paged_attn_metadata)
    }
    fn set_initial_prompt(&mut self, new: String) {
        Sequence::set_initial_prompt(self, new)
    }
    fn get_initial_prompt(&self) -> &str {
        Sequence::get_initial_prompt(self)
    }
    fn set_prefill_toks(&mut self, toks: Vec<u32>) {
        Sequence::set_prefill_toks(self, toks)
    }
    fn has_prefill_toks(&self) -> bool {
        Sequence::has_prefill_toks(self)
    }
    fn set_max_len(&mut self, max_len: usize) {
        Sequence::set_max_len(self, max_len)
    }
    fn mm_features(&self) -> &[MultiModalFeature] {
        Sequence::mm_features(self)
    }
    fn set_mm_features(&mut self, features: Vec<MultiModalFeature>) {
        Sequence::set_mm_features(self, features)
    }
    fn count_prefix_cached_mm_items(&self) -> usize {
        Sequence::count_prefix_cached_mm_items(self)
    }
    fn count_prefix_cached_mm_items_by_kind(&self, kind: MultimodalKind) -> usize {
        Sequence::count_prefix_cached_mm_items_by_kind(self, kind)
    }
    fn images(&self) -> Option<&[DynamicImage]> {
        Sequence::images(self)
    }
    fn clone_images(&self) -> Option<Vec<DynamicImage>> {
        Sequence::clone_images(self)
    }
    fn take_images(&mut self) -> Option<Vec<DynamicImage>> {
        Sequence::take_images(self)
    }
    fn image_hashes(&self) -> Option<&[u64]> {
        Sequence::image_hashes(self)
    }
    fn has_images(&self) -> bool {
        Sequence::has_images(self)
    }
    fn keep_num_images(&mut self, images_to_keep: usize) {
        Sequence::keep_num_images(self, images_to_keep)
    }
    fn audios(&self) -> Option<&[AudioInput]> {
        Sequence::audios(self)
    }
    fn clone_audios(&self) -> Option<Vec<AudioInput>> {
        Sequence::clone_audios(self)
    }
    fn take_audios(&mut self) -> Option<Vec<AudioInput>> {
        Sequence::take_audios(self)
    }
    fn audio_hashes(&self) -> Option<&[u64]> {
        Sequence::audio_hashes(self)
    }
    fn has_audios(&self) -> bool {
        Sequence::has_audios(self)
    }
    fn videos(&self) -> Option<&[VideoInput]> {
        Sequence::videos(self)
    }
    fn clone_videos(&self) -> Option<Vec<VideoInput>> {
        Sequence::clone_videos(self)
    }
    fn take_videos(&mut self) -> Option<Vec<VideoInput>> {
        Sequence::take_videos(self)
    }
    fn video_hashes(&self) -> Option<&[u64]> {
        Sequence::video_hashes(self)
    }
    fn has_videos(&self) -> bool {
        Sequence::has_videos(self)
    }
    fn multimodal(&self) -> &MultimodalData {
        &self.multimodal
    }
    fn multimodal_mut(&mut self) -> &mut MultimodalData {
        &mut self.multimodal
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// The engine sequences behind a processor's views; every view the engine hands out is a `Sequence`.
fn engine_seqs<'a>(seqs: &'a mut [&mut dyn MediaSequence]) -> Vec<&'a mut Sequence> {
    seqs.iter_mut()
        .map(|seq| {
            seq.as_any_mut()
                .downcast_mut::<Sequence>()
                .expect("media sequences are engine sequences")
        })
        .collect()
}

fn engine_seq_refs<'a>(seqs: &'a [&mut dyn MediaSequence]) -> Vec<&'a Sequence> {
    seqs.iter()
        .map(|seq| {
            seq.as_any()
                .downcast_ref::<Sequence>()
                .expect("media sequences are engine sequences")
        })
        .collect()
}

fn media_views<'a>(seqs: &'a mut [&mut Sequence]) -> Vec<&'a mut dyn MediaSequence> {
    seqs.iter_mut()
        .map(|seq| &mut **seq as &mut dyn MediaSequence)
        .collect()
}

struct EngineInputsHost;

impl InputsHost for EngineInputsHost {
    fn prompt_inputs(
        &self,
        toks: PromptTokens<'_>,
        seqs: &[&mut dyn MediaSequence],
        text: TextInputs<'_>,
    ) -> Result<InnerInputProcessorOutput> {
        let seqs = engine_seq_refs(seqs);
        let TextInputs {
            device,
            last_n_context_len,
            return_raw_logits,
            paged_attn_metadata,
            mapper,
            sliding_window,
        } = text;
        match toks {
            PromptTokens::U32(toks) => get_prompt_input(
                toks,
                &seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata,
                mapper,
                sliding_window,
            ),
            PromptTokens::I64(toks) => get_prompt_input(
                toks,
                &seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata,
                mapper,
                sliding_window,
            ),
        }
    }

    fn completion_inputs(
        &self,
        toks: PromptTokens<'_>,
        seqs: &[&mut dyn MediaSequence],
        text: TextInputs<'_>,
        no_kv_cache: bool,
        decode_window: Option<usize>,
    ) -> Result<InnerInputProcessorOutput> {
        let seqs = engine_seq_refs(seqs);
        let TextInputs {
            device,
            last_n_context_len,
            return_raw_logits,
            paged_attn_metadata,
            mapper,
            sliding_window,
        } = text;
        macro_rules! completion {
            ($toks:expr) => {
                match decode_window {
                    Some(window) => get_completion_input_windowed(
                        $toks,
                        &seqs,
                        device,
                        no_kv_cache,
                        last_n_context_len,
                        return_raw_logits,
                        paged_attn_metadata,
                        mapper,
                        sliding_window,
                        window,
                    ),
                    None => get_completion_input(
                        $toks,
                        &seqs,
                        device,
                        no_kv_cache,
                        last_n_context_len,
                        return_raw_logits,
                        paged_attn_metadata,
                        mapper,
                        sliding_window,
                    ),
                }
            };
        }
        match toks {
            PromptTokens::U32(toks) => completion!(toks),
            PromptTokens::I64(toks) => completion!(toks),
        }
    }

    fn text_only_inputs(
        &self,
        seqs: &mut [&mut dyn MediaSequence],
        call: ProcessInputsCall<'_>,
    ) -> Result<TextOnlyInputs> {
        let mut seqs = engine_seqs(seqs);
        let ProcessInputsCall {
            tokenizer,
            is_prompt,
            is_xlora,
            device,
            no_kv_cache,
            last_n_context_len,
            return_raw_logits,
            sliding_window,
            other_config,
            paged_attn_metadata,
            mapper,
        } = call;
        let InputProcessorOutput {
            inputs,
            seq_indices,
        } = TextInputsProcessor.process_inputs(
            tokenizer,
            &mut seqs,
            is_prompt,
            is_xlora,
            device,
            no_kv_cache,
            last_n_context_len,
            return_raw_logits,
            sliding_window,
            other_config,
            paged_attn_metadata,
            mapper,
        )?;
        // The media adapter adds the leases again, from the same sequences and indices.
        let TextModelInputs {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
            recurrent_batch_kind,
            ..
        } = *inputs
            .downcast::<TextModelInputs>()
            .expect("the text processor returns its ModelInputs");
        Ok(TextOnlyInputs {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
            recurrent_batch_kind,
            seq_indices,
        })
    }

    fn block_denoising_progress(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        seqs: &[&mut dyn MediaSequence],
        seq_indices: &[usize],
        return_raw_logits: bool,
    ) -> Option<Vec<BlockDenoisingProgressEmitter>> {
        crate::block_diffusion::block_denoising_progress_emitters(
            tokenizer,
            &engine_seq_refs(seqs),
            seq_indices,
            return_raw_logits,
        )
    }

    fn staged_batch_width(&self, seqs: &[&mut dyn MediaSequence]) -> Option<usize> {
        crate::speculative::staging::staged_batch_width(&engine_seq_refs(seqs))
    }
}

/// Runs a model's `MultimodalInputsProcessor` as the engine's `InputsProcessor`, adding the adapter leases.
pub(crate) struct MediaInputsProcessor(pub(crate) Arc<dyn MultimodalInputsProcessor>);

impl InputsProcessor for MediaInputsProcessor {
    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        device: &Device,
        other_config: Option<Arc<dyn Any>>,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<()> {
        self.0.prepare_for_paged_prompt_planning(
            tokenizer,
            &mut media_views(input_seqs),
            device,
            other_config,
            paged_attn_metadata,
        )
    }

    fn process_inputs(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        sliding_window: Option<usize>,
        other_config: Option<Arc<dyn Any>>,
        paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        let InputProcessorOutput {
            inputs,
            seq_indices,
        } = self.0.process_inputs(
            &EngineInputsHost,
            tokenizer,
            &mut media_views(input_seqs),
            is_prompt,
            is_xlora,
            device,
            no_kv_cache,
            last_n_context_len,
            return_raw_logits,
            sliding_window,
            other_config,
            paged_attn_metadata,
            mapper,
        )?;
        let media = inputs
            .downcast::<MediaModelInputs>()
            .expect("multimodal processors return the media ModelInputs");
        let inputs = {
            let MediaModelInputs {
                input_ids,
                seqlen_offsets,
                context_lens,
                position_ids,
                pixel_values,
                model_specific_args,
                paged_attn_meta,
                flash_meta,
                recurrent_batch_kind,
            } = *media;
            Box::new(ModelInputs {
                input_ids,
                seqlen_offsets,
                context_lens,
                position_ids,
                pixel_values,
                model_specific_args,
                paged_attn_meta,
                flash_meta,
                recurrent_batch_kind,
                adapter_leases: super::adapter_leases(input_seqs, &seq_indices),
            }) as Box<dyn Any>
        };
        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }

    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }
}

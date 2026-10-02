//! Input processing shared by Qwen2-VL/Qwen2.5-VL and Qwen3-VL/Qwen3.5; per-generation behavior is [`QwenVlSpec`].

use std::{any::Any, ops::Range, sync::Arc};

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use image::DynamicImage;
use tokenizers::Tokenizer;

use crate::device_map::DeviceMapper;
use crate::gdn::RecurrentBatchKind;
use crate::media_inputs::{
    image_processor::PreprocessedImages,
    media::find_placeholder_delimited_ranges,
    preprocessor_config::PreProcessorConfig,
    processor::{
        InnerInputProcessorOutput, InputMetadata, InputProcessorOutput, InputsHost,
        InputsProcessorValidationError, MediaSequence, ModelInputs, TextInputs,
    },
    video::VideoInput,
};
use crate::model::recurrent_batch_kind_for_input;
use crate::paged_attention::{
    PagedAttentionMeta,
    block_hash::{MultimodalAttentionPolicy, MultimodalKind},
};
use crate::qwen2vl::Qwen2VLVisionSpecificArgs;
use crate::qwen2vl::inputs_processor::{
    IMAGE_PAD, PLACEHOLDER, VIDEO_PAD, VISION_END, VISION_START, apply_mrope_position_deltas,
    expand_media_placeholders, find_sequences, media_data_cached_offset, qwen2_decode_args,
    select_media_batch, select_media_view, shift_media_spans, split_media_pixels,
    validate_qwen_media_dimensions, validated_mm_features, video_hashes,
};
use crate::vision::multimodal_layout::{
    MropePositionSource, MultimodalEmbeddingMap, MultimodalEncoderKey, MultimodalItemLayout,
    PackedMultimodalLayout, RequestMultimodalLayout, gather_packed_mrope_positions,
};

type Spans = Vec<(usize, usize)>;

pub(crate) struct QwenMropeConfig {
    pub(crate) spatial_merge_size: usize,
    pub(crate) image_token_id: u32,
    pub(crate) video_token_id: u32,
    pub(crate) vision_start_token_id: u32,
    pub(crate) vision_end_token_id: u32,
}

pub(crate) trait QwenVlSpec: Sync {
    fn name(&self) -> &'static str;
    fn preprocess_media(
        &self,
        images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
    ) -> candle_core::Result<PreprocessedImages>;
    // Minimum image and video edges the media validation enforces.
    fn media_resize_factors(&self, config: &PreProcessorConfig) -> (Option<usize>, Option<usize>);
    fn spatial_merge_size(&self, config: &PreProcessorConfig) -> Result<usize>;
    fn expand_video_placeholders(
        &self,
        text: &mut String,
        grid: Option<&Tensor>,
        videos: &[VideoInput],
        config: &PreProcessorConfig,
    ) -> Result<()>;
    // How many consecutive video pad runs make up each video.
    fn video_runs_per_item(&self, grid: Option<&Tensor>, run_count: usize) -> Result<Vec<usize>>;
    // Whether a packed text-only prefill still takes its MRoPE positions from the processor.
    fn packed_text_needs_prompt_mrope(&self) -> bool;
    fn mrope_position_source(
        &self,
        seq: &dyn MediaSequence,
        config: &QwenMropeConfig,
        device: &Device,
    ) -> Result<MropePositionSource>;
}

fn seq_videos_view(seq: &dyn MediaSequence) -> &[VideoInput] {
    let videos = seq.videos().unwrap_or_default();
    if !seq.is_chunked_prefill_view() {
        return videos;
    }
    seq.active_local_multimodal_item_range(MultimodalKind::Video, videos.len())
        .and_then(|range| videos.get(range))
        .unwrap_or_default()
}

// Delimited (start, len) ranges collapse to one covering range per item.
pub(crate) fn group_item_ranges(
    ranges: &[(usize, usize)],
    runs_per_item: &[usize],
) -> Result<Spans> {
    let expected: usize = runs_per_item.iter().sum();
    if ranges.len() != expected {
        anyhow::bail!(
            "Qwen video has {} placeholder ranges but its items expect {expected}",
            ranges.len()
        );
    }
    let mut grouped = Vec::with_capacity(runs_per_item.len());
    let mut offset = 0usize;
    for &runs in runs_per_item {
        if runs == 0 {
            anyhow::bail!("Qwen video item has no placeholder runs");
        }
        let (start, _) = ranges[offset];
        let (last_start, last_len) = ranges[offset + runs - 1];
        grouped.push((start, last_start + last_len - start));
        offset += runs;
    }
    Ok(grouped)
}

// Like shift_media_spans, but an item may span several runs and caches only whole; returns (cached, current) items.
pub(crate) fn shift_item_runs(
    runs: &mut Spans,
    runs_per_item: &[usize],
    prefix_len: usize,
) -> Result<(usize, usize)> {
    let expected: usize = runs_per_item.iter().sum();
    if runs.len() != expected {
        anyhow::bail!(
            "Qwen video has {} pad runs but its items expect {expected}",
            runs.len()
        );
    }
    let mut cached = 0usize;
    let mut current = 0usize;
    let mut kept = Vec::with_capacity(runs.len());
    let mut offset = 0usize;
    for &item_runs in runs_per_item {
        if item_runs == 0 {
            anyhow::bail!("Qwen video item has no pad runs");
        }
        let group = &runs[offset..offset + item_runs];
        offset += item_runs;
        let start = group.first().map_or(0, |run| run.0);
        let end = group.last().map_or(0, |run| run.1);
        if end <= prefix_len {
            cached += 1;
        } else if start < prefix_len {
            anyhow::bail!("Qwen prefix cache splits a multimodal item");
        } else {
            current += 1;
            kept.extend(
                group
                    .iter()
                    .map(|&(start, end)| (start - prefix_len, end - prefix_len)),
            );
        }
    }
    *runs = kept;
    Ok((cached, current))
}

fn token_id(tokenizer: &Tokenizer, token: &str, what: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| anyhow::anyhow!("Qwen tokenizer is missing {what} token"))
}

fn decode_prompts(
    tokenizer: &Tokenizer,
    input_seqs: &[&mut dyn MediaSequence],
) -> Result<Vec<String>> {
    tokenizer
        .decode_batch(
            &input_seqs
                .iter()
                .map(|seq| seq.get_toks())
                .collect::<Vec<_>>(),
            false,
        )
        .map_err(anyhow::Error::msg)
}

fn concat_rows(rows: &[Tensor]) -> Result<Option<Tensor>> {
    Ok((!rows.is_empty())
        .then(|| Tensor::cat(rows, 0))
        .transpose()?)
}

fn padded_ids<'a>(
    rows: impl Iterator<Item = &'a [u32]> + Clone,
    device: &Device,
) -> Result<Tensor> {
    let max_len = rows.clone().map(<[u32]>::len).max().unwrap_or(0);
    let rows = rows
        .map(|ids| {
            let mut ids = ids.to_vec();
            ids.resize(max_len, 0);
            Tensor::new(ids, device)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok(Tensor::stack(&rows, 0)?)
}

struct SeqPlaceholders<'a> {
    image_grid: Option<&'a Tensor>,
    video_grid: Option<&'a Tensor>,
    image_rows: usize,
    video_rows: usize,
    videos: &'a [VideoInput],
    // Error text names the rows, e.g. "selected " for a chunk's media window.
    scope: &'static str,
}

#[derive(Default)]
struct PromptMedia {
    input_ids_full: Option<Tensor>,
    pixel_values: Option<Tensor>,
    pixel_values_videos: Option<Tensor>,
    image_grid_thw: Option<Tensor>,
    video_grid_thw: Option<Tensor>,
    continuous_img_pad: Vec<Spans>,
    continuous_vid_pad: Vec<Spans>,
    image_item_counts: Vec<usize>,
    video_item_counts: Vec<usize>,
}

pub(crate) struct QwenVlInputs<'a> {
    pub(crate) spec: &'a dyn QwenVlSpec,
    pub(crate) tokenizer: Option<Arc<Tokenizer>>,
    pub(crate) device: &'a Device,
    pub(crate) other_config: Option<Arc<dyn Any>>,
}

pub(crate) struct QwenVlStep<'a> {
    pub(crate) host: &'a dyn InputsHost,
    pub(crate) is_prompt: bool,
    pub(crate) is_xlora: bool,
    pub(crate) no_kv_cache: bool,
    pub(crate) last_n_context_len: Option<(usize, usize)>,
    pub(crate) return_raw_logits: bool,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) paged_attn_metadata: Option<PagedAttentionMeta>,
    pub(crate) mapper: Option<&'a dyn DeviceMapper>,
}

impl QwenVlInputs<'_> {
    fn tokenizer(&self) -> Result<Arc<Tokenizer>> {
        self.tokenizer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("{} requires a specified tokenizer.", self.spec.name()))
    }

    fn config(&self) -> Result<&PreProcessorConfig> {
        self.other_config
            .as_ref()
            .and_then(|config| config.downcast_ref())
            .ok_or_else(|| anyhow::anyhow!("{} needs its PreProcessorConfig", self.spec.name()))
    }

    pub(crate) fn prepare(
        self,
        input_seqs: &mut [&mut dyn MediaSequence],
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<()> {
        let tokenizer = self.tokenizer()?;
        let config = self.config()?;
        if !input_seqs
            .iter()
            .any(|seq| seq.has_images() || seq.has_videos())
        {
            return Ok(());
        }
        self.validate_media(input_seqs, config)?;

        let mut detok_seqs = decode_prompts(&tokenizer, input_seqs)?;
        for (text, seq) in detok_seqs.iter_mut().zip(input_seqs.iter_mut()) {
            let (image_grid, video_grid) = if seq.has_images() || seq.has_videos() {
                let (_, image_grid, video_grid) = self.load_media(&mut **seq, config)?;
                (image_grid, video_grid)
            } else {
                (None, None)
            };
            if seq.multimodal().has_changed_prompt {
                continue;
            }
            let videos = seq.videos().unwrap_or_default();
            let placeholders = SeqPlaceholders {
                image_grid: image_grid.as_ref(),
                video_grid: video_grid.as_ref(),
                image_rows: seq.images().map_or(0, <[_]>::len),
                video_rows: videos.len(),
                videos,
                scope: "",
            };
            self.expand_placeholders(text, &**seq, placeholders, config)?;
        }

        for (detok, seq) in detok_seqs.into_iter().zip(input_seqs.iter_mut()) {
            if !seq.multimodal().has_changed_prompt {
                self.commit_prompt(
                    &tokenizer,
                    &mut **seq,
                    detok,
                    paged_attn_metadata.as_deref_mut(),
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn process(
        self,
        input_seqs: &mut [&mut dyn MediaSequence],
        step: QwenVlStep<'_>,
    ) -> Result<InputProcessorOutput> {
        let QwenVlStep {
            host,
            is_prompt,
            is_xlora,
            no_kv_cache,
            last_n_context_len,
            return_raw_logits,
            sliding_window,
            mut paged_attn_metadata,
            mapper,
        } = step;
        if is_xlora {
            return Err(anyhow::Error::msg(
                "Cannot make inputs for X-LoRA vision model.",
            ));
        }
        if no_kv_cache {
            return Err(anyhow::Error::msg("Vision model must have kv cache."));
        }
        let device = self.device;
        if !is_prompt {
            let InnerInputProcessorOutput {
                inputs:
                    InputMetadata {
                        input,
                        positions,
                        context_lens,
                        position_ids,
                        paged_attn_meta,
                        flash_meta,
                    },
                seq_indices,
            } = host.completion_inputs(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>()
                    .into(),
                input_seqs,
                TextInputs {
                    device,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata: paged_attn_metadata.as_mut(),
                    mapper,
                    sliding_window,
                },
                no_kv_cache,
                None,
            )?;
            let position_ids = apply_mrope_position_deltas(position_ids, input_seqs)?;
            let args = qwen2_decode_args(&input, input_seqs.iter().map(|seq| seq.len()).collect());
            let inputs: Box<dyn Any> = Box::new(ModelInputs {
                input_ids: input,
                seqlen_offsets: positions,
                context_lens,
                position_ids,
                pixel_values: None,
                model_specific_args: Box::new(args),
                paged_attn_meta,
                flash_meta,
                recurrent_batch_kind: recurrent_batch_kind_for_input(
                    false,
                    host.staged_batch_width(input_seqs).is_some(),
                ),
            });
            return Ok(InputProcessorOutput {
                inputs,
                seq_indices,
            });
        }

        let tokenizer = self.tokenizer()?;
        let config = self.config()?;
        for seq in input_seqs.iter_mut() {
            if seq.multimodal().rope_img_grid_thw.is_none()
                && seq.multimodal().rope_vid_grid_thw.is_none()
            {
                seq.multimodal_mut().mrope_position_delta = None;
            }
        }
        let PromptMedia {
            input_ids_full,
            mut pixel_values,
            mut pixel_values_videos,
            mut image_grid_thw,
            mut video_grid_thw,
            mut continuous_img_pad,
            mut continuous_vid_pad,
            image_item_counts,
            video_item_counts,
        } = if input_seqs
            .iter()
            .any(|seq| seq.has_images() || seq.has_videos())
        {
            self.prompt_media(&tokenizer, input_seqs, config, paged_attn_metadata.as_mut())?
        } else {
            PromptMedia {
                continuous_img_pad: vec![Vec::new(); input_seqs.len()],
                continuous_vid_pad: vec![Vec::new(); input_seqs.len()],
                image_item_counts: vec![0; input_seqs.len()],
                video_item_counts: vec![0; input_seqs.len()],
                ..Default::default()
            }
        };

        // Built after prompt_media so the text inputs carry the expanded placeholders.
        let InnerInputProcessorOutput {
            inputs:
                InputMetadata {
                    input,
                    positions,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                },
            seq_indices,
        } = host.prompt_inputs(
            input_seqs
                .iter()
                .map(|seq| seq.get_toks())
                .collect::<Vec<_>>()
                .into(),
            input_seqs,
            TextInputs {
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata: paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
            },
        )?;

        let has_rope_grids = input_seqs.iter().any(|seq| {
            seq.multimodal().rope_img_grid_thw.is_some()
                || seq.multimodal().rope_vid_grid_thw.is_some()
        });
        let needs_prompt_mrope =
            has_rope_grids || (flash_meta.packed && self.spec.packed_text_needs_prompt_mrope());
        let input_ids_full = match input_ids_full {
            Some(ids) => ids,
            None if flash_meta.packed || needs_prompt_mrope => {
                padded_ids(input_seqs.iter().map(|seq| seq.get_toks()), device)?
            }
            None => input.clone(),
        };

        let seq_count = input_seqs.len();
        let mut cached_images = vec![0; seq_count];
        let mut current_images = vec![0; seq_count];
        let mut cached_videos = vec![0; seq_count];
        let mut current_videos = vec![0; seq_count];
        for (seq_idx, (seq, (img_pads, vid_pads))) in input_seqs
            .iter()
            .zip(
                continuous_img_pad
                    .iter_mut()
                    .zip(continuous_vid_pad.iter_mut()),
            )
            .enumerate()
        {
            let local_prefix = seq
                .active_prompt_local_query_range()
                .map_or(seq.prefix_cache_len(), |query| query.start);
            let cached = shift_media_spans(img_pads, local_prefix)?;
            let runs_per_item = self
                .spec
                .video_runs_per_item(seq.multimodal().rope_vid_grid_thw.as_ref(), vid_pads.len())?;
            let (cached_vids, current_vids) =
                shift_item_runs(vid_pads, &runs_per_item, local_prefix)?;
            cached_images[seq_idx] = media_data_cached_offset(&**seq, cached);
            cached_videos[seq_idx] = media_data_cached_offset(&**seq, cached_vids);
            current_images[seq_idx] = img_pads.len();
            current_videos[seq_idx] = current_vids;
        }
        (pixel_values, image_grid_thw) = select_media_batch(
            pixel_values,
            image_grid_thw,
            &image_item_counts,
            &cached_images,
            &current_images,
        )?;
        (pixel_values_videos, video_grid_thw) = select_media_batch(
            pixel_values_videos,
            video_grid_thw,
            &video_item_counts,
            &cached_videos,
            &current_videos,
        )?;

        let seqlens = input_seqs.iter().map(|seq| seq.len()).collect::<Vec<_>>();
        let rope_img_grid_thw = concat_rows(
            &input_seqs
                .iter()
                .filter_map(|seq| seq.multimodal().rope_img_grid_thw.clone())
                .collect::<Vec<_>>(),
        )?;
        let rope_vid_grid_thw = concat_rows(
            &input_seqs
                .iter()
                .filter_map(|seq| seq.multimodal().rope_vid_grid_thw.clone())
                .collect::<Vec<_>>(),
        )?;

        let mut image_hashes = Vec::new();
        let mut selected_video_hashes = Vec::new();
        for (seq_idx, seq) in input_seqs.iter().enumerate() {
            let hashes = seq.image_hashes().unwrap_or_default();
            let (cached, current) = (cached_images[seq_idx], current_images[seq_idx]);
            let selected = hashes.get(cached..cached + current).ok_or_else(|| {
                anyhow::Error::msg("Qwen image hashes do not cover the selected media window")
            })?;
            image_hashes.extend_from_slice(selected);

            let hashes = video_hashes(&**seq);
            let (cached, current) = (cached_videos[seq_idx], current_videos[seq_idx]);
            let selected = hashes.get(cached..cached + current).ok_or_else(|| {
                anyhow::Error::msg("Qwen video hashes do not cover the selected media window")
            })?;
            selected_video_hashes.extend_from_slice(selected);
        }
        let packed_layout = if flash_meta.packed {
            let query_lens = paged_attn_meta
                .as_ref()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .ok_or_else(|| anyhow::Error::msg("packed Qwen prefill requires query lengths"))?;
            let layout = self.packed_layout(
                input_seqs,
                query_lens,
                &continuous_img_pad,
                &continuous_vid_pad,
            )?;
            if layout.token_count() != input.dim(1)? {
                anyhow::bail!(
                    "Qwen packed layout has {} tokens but input has {}",
                    layout.token_count(),
                    input.dim(1)?
                );
            }
            Some(layout)
        } else {
            None
        };
        let prompt_position_ids = if needs_prompt_mrope {
            let mrope = QwenMropeConfig {
                spatial_merge_size: self.spec.spatial_merge_size(config)?,
                image_token_id: token_id(&tokenizer, IMAGE_PAD, "image pad")?,
                video_token_id: token_id(&tokenizer, VIDEO_PAD, "video pad")?,
                vision_start_token_id: token_id(&tokenizer, VISION_START, "vision start")?,
                vision_end_token_id: token_id(&tokenizer, VISION_END, "vision end")?,
            };
            Some(self.prompt_mrope(input_seqs, flash_meta.packed, input.dim(1)?, &mrope)?)
        } else {
            None
        };

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args: Box::new(Qwen2VLVisionSpecificArgs {
                input_ids_full,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
                rope_img_grid_thw,
                rope_vid_grid_thw,
                seqlens,
                continuous_img_pad,
                continuous_vid_pad,
                image_hashes,
                video_hashes: selected_video_hashes,
                packed_layout,
                prompt_position_ids,
            }),
            paged_attn_meta,
            flash_meta,
            recurrent_batch_kind: RecurrentBatchKind::Prefill,
        });
        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }

    fn validate_media(
        &self,
        input_seqs: &[&mut dyn MediaSequence],
        config: &PreProcessorConfig,
    ) -> Result<()> {
        let (image_factor, video_factor) = self.spec.media_resize_factors(config);
        for seq in input_seqs.iter() {
            if let Some(images) = seq.images() {
                validate_qwen_media_dimensions(images, image_factor)?;
            }
            if let Some(videos) = seq.videos() {
                if videos.iter().any(|video| video.frames.is_empty()) {
                    return Err(InputsProcessorValidationError(
                        "Qwen video inputs must contain at least one frame".to_string(),
                    )
                    .into());
                }
                for video in videos {
                    validate_qwen_media_dimensions(&video.frames, video_factor)?;
                }
            }
        }
        Ok(())
    }

    // Pixels and image/video grids, preprocessed once per sequence; also seeds the sequence's MRoPE grids.
    fn load_media(
        &self,
        seq: &mut dyn MediaSequence,
        config: &PreProcessorConfig,
    ) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
        let (pixel_values, image_grid_thw, video_grid_thw) =
            if let Some(cached_pixel_values) = &seq.multimodal().cached_pixel_values {
                (
                    cached_pixel_values.clone(),
                    seq.multimodal().cached_img_thw.clone(),
                    seq.multimodal().cached_vid_thw.clone(),
                )
            } else {
                let image = seq
                    .has_images()
                    .then(|| {
                        self.spec.preprocess_media(
                            seq.clone_images().unwrap_or_default(),
                            Vec::new(),
                            config,
                            self.device,
                        )
                    })
                    .transpose()?;
                let video = seq
                    .has_videos()
                    .then(|| {
                        self.spec.preprocess_media(
                            Vec::new(),
                            seq.videos()
                                .unwrap_or_default()
                                .iter()
                                .map(|video| video.frames.clone())
                                .collect(),
                            config,
                            self.device,
                        )
                    })
                    .transpose()?;
                let image_grid_thw = image
                    .as_ref()
                    .and_then(|processed| processed.image_grid_thw.clone());
                let video_grid_thw = video
                    .as_ref()
                    .and_then(|processed| processed.video_grid_thw.clone());
                let pixels = image
                    .into_iter()
                    .chain(video)
                    .map(|processed| processed.pixel_values)
                    .collect::<Vec<_>>();
                let pixel_values = Tensor::cat(&pixels, 0)?;
                let multimodal = seq.multimodal_mut();
                multimodal.cached_pixel_values = Some(pixel_values.clone());
                multimodal.cached_img_thw = image_grid_thw.clone();
                multimodal.cached_vid_thw = video_grid_thw.clone();
                (pixel_values, image_grid_thw, video_grid_thw)
            };
        let multimodal = seq.multimodal_mut();
        if multimodal.rope_img_grid_thw.is_none() {
            multimodal.rope_img_grid_thw = image_grid_thw.clone();
        }
        if multimodal.rope_vid_grid_thw.is_none() {
            multimodal.rope_vid_grid_thw = video_grid_thw.clone();
        }
        Ok((pixel_values, image_grid_thw, video_grid_thw))
    }

    fn expand_placeholders(
        &self,
        text: &mut String,
        seq: &dyn MediaSequence,
        placeholders: SeqPlaceholders<'_>,
        config: &PreProcessorConfig,
    ) -> Result<()> {
        let SeqPlaceholders {
            image_grid,
            video_grid,
            image_rows,
            video_rows,
            videos,
            scope,
        } = placeholders;
        let image_hashes = seq.image_hashes().unwrap_or_default().len();
        if image_hashes != image_rows {
            anyhow::bail!(
                "Qwen has {image_rows} {scope}image rows but {image_hashes} image hashes"
            );
        }
        let video_hashes = video_hashes(seq).len();
        if video_hashes != video_rows {
            anyhow::bail!(
                "Qwen has {video_rows} {scope}video rows but {video_hashes} video hashes"
            );
        }
        expand_media_placeholders(
            text,
            IMAGE_PAD,
            PLACEHOLDER,
            image_grid,
            image_rows,
            self.spec.spatial_merge_size(config)?.pow(2),
            MultimodalKind::Image,
        )?;
        self.spec
            .expand_video_placeholders(text, video_grid, videos, config)
    }

    // Tokenizes an expanded prompt; a sequence's first pass also records its media features and new tokens.
    fn commit_prompt(
        &self,
        tokenizer: &Tokenizer,
        seq: &mut dyn MediaSequence,
        detok: String,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<Vec<u32>> {
        let ids = tokenizer
            .encode_fast(detok.as_str(), false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        if seq.multimodal().has_changed_prompt {
            return Ok(ids);
        }
        seq.set_initial_prompt(detok);
        if seq.mm_features().is_empty() {
            let start_id = token_id(tokenizer, VISION_START, "vision start")?;
            let end_id = token_id(tokenizer, VISION_END, "vision end")?;
            let img_pad_id = token_id(tokenizer, IMAGE_PAD, "image pad")?;
            let image_ranges =
                find_placeholder_delimited_ranges(&ids, img_pad_id, start_id, end_id);
            let mut features = validated_mm_features(
                &image_ranges,
                seq.image_hashes().unwrap_or_default(),
                MultimodalKind::Image,
            )?;
            let vid_pad_id = token_id(tokenizer, VIDEO_PAD, "video pad")?;
            let video_ranges =
                find_placeholder_delimited_ranges(&ids, vid_pad_id, start_id, end_id);
            let runs_per_item = self.spec.video_runs_per_item(
                seq.multimodal().rope_vid_grid_thw.as_ref(),
                video_ranges.len(),
            )?;
            let video_ranges = group_item_ranges(&video_ranges, &runs_per_item)?;
            features.extend(validated_mm_features(
                &video_ranges,
                &video_hashes(seq),
                MultimodalKind::Video,
            )?);
            if !features.is_empty() {
                seq.set_mm_features(features);
            }
        }
        seq.set_toks_and_reallocate(ids.clone(), paged_attn_metadata);
        seq.multimodal_mut().has_changed_prompt = true;
        Ok(ids)
    }

    fn prompt_media(
        &self,
        tokenizer: &Tokenizer,
        input_seqs: &mut [&mut dyn MediaSequence],
        config: &PreProcessorConfig,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<PromptMedia> {
        let seq_count = input_seqs.len();
        let mut image_item_counts = vec![0usize; seq_count];
        let mut video_item_counts = vec![0usize; seq_count];
        let mut image_pixels = Vec::new();
        let mut video_pixels = Vec::new();
        let mut image_grids = Vec::with_capacity(seq_count);
        let mut video_grids = Vec::with_capacity(seq_count);
        let mut detok_seqs = decode_prompts(tokenizer, input_seqs)?;

        for (seq_idx, seq) in input_seqs.iter_mut().enumerate() {
            if !seq.has_images() && !seq.has_videos() {
                image_grids.push(None);
                video_grids.push(None);
                continue;
            }
            let (pixel_values, image_grid_thw, video_grid_thw) =
                self.load_media(&mut **seq, config)?;
            let (images, videos) = split_media_pixels(
                &pixel_values,
                image_grid_thw.as_ref(),
                video_grid_thw.as_ref(),
            )?;
            let (images, image_grid_thw, image_count) =
                select_media_view(&**seq, MultimodalKind::Image, images, image_grid_thw)?;
            let (videos, video_grid_thw, video_count) =
                select_media_view(&**seq, MultimodalKind::Video, videos, video_grid_thw)?;
            image_item_counts[seq_idx] = image_count;
            video_item_counts[seq_idx] = video_count;
            image_pixels.extend(images);
            video_pixels.extend(videos);
            image_grids.push(image_grid_thw);
            video_grids.push(video_grid_thw);
        }

        for (seq_idx, (text, seq)) in detok_seqs.iter_mut().zip(input_seqs.iter()).enumerate() {
            if seq.multimodal().has_changed_prompt {
                continue;
            }
            let placeholders = SeqPlaceholders {
                image_grid: image_grids[seq_idx].as_ref(),
                video_grid: video_grids[seq_idx].as_ref(),
                image_rows: image_item_counts[seq_idx],
                video_rows: video_item_counts[seq_idx],
                videos: seq_videos_view(&**seq),
                scope: "selected ",
            };
            self.expand_placeholders(text, &**seq, placeholders, config)?;
        }

        let img_pad = token_id(tokenizer, IMAGE_PAD, "image pad")?;
        let vid_pad = token_id(tokenizer, VIDEO_PAD, "video pad")?;
        let mut all_ids = Vec::with_capacity(seq_count);
        let mut continuous_img_pad = Vec::with_capacity(seq_count);
        let mut continuous_vid_pad = Vec::with_capacity(seq_count);
        for (detok, seq) in detok_seqs.into_iter().zip(input_seqs.iter_mut()) {
            let ids = self.commit_prompt(
                tokenizer,
                &mut **seq,
                detok,
                paged_attn_metadata.as_deref_mut(),
            )?;
            continuous_img_pad.push(find_sequences(&ids, img_pad));
            continuous_vid_pad.push(find_sequences(&ids, vid_pad));
            all_ids.push(ids);
        }

        Ok(PromptMedia {
            input_ids_full: Some(padded_ids(all_ids.iter().map(Vec::as_slice), self.device)?),
            pixel_values: concat_rows(&image_pixels)?,
            pixel_values_videos: concat_rows(&video_pixels)?,
            image_grid_thw: concat_rows(&image_grids.into_iter().flatten().collect::<Vec<_>>())?,
            video_grid_thw: concat_rows(&video_grids.into_iter().flatten().collect::<Vec<_>>())?,
            continuous_img_pad,
            continuous_vid_pad,
            image_item_counts,
            video_item_counts,
        })
    }

    fn prompt_mrope(
        &self,
        input_seqs: &mut [&mut dyn MediaSequence],
        packed: bool,
        padded_len: usize,
        mrope: &QwenMropeConfig,
    ) -> Result<Tensor> {
        let device = self.device;
        let query_ranges = input_seqs
            .iter()
            .map(|seq| {
                seq.active_prompt_query_range().unwrap_or_else(|| {
                    // Paged prefix-cache hits trim the input to the tail without a prefill view.
                    let len = seq.prompt_position_source_toks().len();
                    seq.prefix_cache_len().min(len)..len
                })
            })
            .collect::<Vec<_>>();
        let mut sources = Vec::with_capacity(input_seqs.len());
        for seq in input_seqs.iter_mut() {
            let source = self.spec.mrope_position_source(&**seq, mrope, device)?;
            seq.multimodal_mut().mrope_position_delta = Some(source.delta);
            sources.push(source);
        }
        if packed {
            return Ok(gather_packed_mrope_positions(
                &sources,
                &query_ranges,
                device,
            )?);
        }

        let mut rows = Vec::with_capacity(sources.len());
        for (source, query) in sources.iter().zip(query_ranges) {
            if query.end > source.position_ids.dim(2)? {
                anyhow::bail!("Qwen MRoPE query range exceeds the sequence position source");
            }
            let positions = source
                .position_ids
                .i((.., 0, query))?
                .to_dtype(DType::I64)?;
            if positions.dim(1)? > padded_len {
                anyhow::bail!("Qwen MRoPE query is longer than the padded input");
            }
            let padding = padded_len - positions.dim(1)?;
            let positions = if padding == 0 {
                positions
            } else {
                Tensor::cat(
                    &[positions, Tensor::ones((3, padding), DType::I64, device)?],
                    1,
                )?
            };
            rows.push(positions);
        }
        Ok(Tensor::stack(&rows, 1)?)
    }

    fn packed_layout(
        &self,
        input_seqs: &[&mut dyn MediaSequence],
        query_lens: &[usize],
        continuous_img_pad: &[Spans],
        continuous_vid_pad: &[Spans],
    ) -> Result<PackedMultimodalLayout> {
        if input_seqs.len() != query_lens.len()
            || input_seqs.len() != continuous_img_pad.len()
            || input_seqs.len() != continuous_vid_pad.len()
        {
            anyhow::bail!("Qwen packed multimodal metadata length mismatch");
        }
        let mut requests = Vec::with_capacity(input_seqs.len());
        for (((seq, &query_len), image_spans), video_spans) in input_seqs
            .iter()
            .zip(query_lens)
            .zip(continuous_img_pad)
            .zip(continuous_vid_pad)
        {
            if query_len != seq.get_toks().len() {
                anyhow::bail!("Qwen packed multimodal prefill requires the complete prompt");
            }
            let image_hashes = seq.image_hashes().unwrap_or_default();
            if image_hashes.len() != image_spans.len() {
                anyhow::bail!(
                    "Qwen sequence has {} image hashes but {} image spans",
                    image_hashes.len(),
                    image_spans.len()
                );
            }
            let video_hashes = video_hashes(&**seq);
            let runs_per_item = self.spec.video_runs_per_item(
                seq.multimodal().rope_vid_grid_thw.as_ref(),
                video_spans.len(),
            )?;
            if video_hashes.len() != runs_per_item.len()
                || video_spans.len() != runs_per_item.iter().sum::<usize>()
            {
                anyhow::bail!(
                    "Qwen sequence has {} video hashes, {} video spans, and {:?} spans per video",
                    video_hashes.len(),
                    video_spans.len(),
                    runs_per_item
                );
            }
            let mut items = Vec::with_capacity(image_spans.len() + video_hashes.len());
            for (item_index, (&hash, &(start, end))) in
                image_hashes.iter().zip(image_spans).enumerate()
            {
                items.push(MultimodalItemLayout::new(
                    MultimodalEncoderKey {
                        kind: MultimodalKind::Image,
                        hash,
                    },
                    item_index,
                    start..end,
                    MultimodalAttentionPolicy::Causal,
                    vec![MultimodalEmbeddingMap::contiguous(start..end, 0, 0)?],
                )?);
            }
            let mut span_offset = 0usize;
            for (item_index, (&hash, &runs)) in video_hashes.iter().zip(&runs_per_item).enumerate()
            {
                let group = &video_spans[span_offset..span_offset + runs];
                span_offset += runs;
                let item_start = group.first().map_or(0, |span| span.0);
                let item_end = group.last().map_or(0, |span| span.1);
                let mut embedding_maps = Vec::with_capacity(runs);
                let mut embed_offset = 0usize;
                for &(start, end) in group {
                    embedding_maps.push(MultimodalEmbeddingMap::contiguous(
                        start..end,
                        embed_offset,
                        0,
                    )?);
                    embed_offset += end - start;
                }
                items.push(MultimodalItemLayout::new(
                    MultimodalEncoderKey {
                        kind: MultimodalKind::Video,
                        hash,
                    },
                    item_index,
                    item_start..item_end,
                    MultimodalAttentionPolicy::Causal,
                    embedding_maps,
                )?);
            }
            requests.push(RequestMultimodalLayout {
                sequence_id: *seq.id(),
                query: Range {
                    start: 0,
                    end: query_len,
                },
                items,
            });
        }
        Ok(PackedMultimodalLayout::new(&requests)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_run_per_item_grouping_keeps_each_range() -> Result<()> {
        let ranges = vec![(2, 4), (10, 6)];
        assert_eq!(group_item_ranges(&ranges, &[1, 1])?, ranges);
        Ok(())
    }

    #[test]
    fn grouping_covers_each_item_runs_and_rejects_empty_items() -> Result<()> {
        let ranges = [(2, 3), (6, 3), (12, 4)];
        assert_eq!(group_item_ranges(&ranges, &[2, 1])?, vec![(2, 7), (12, 4)]);
        assert!(group_item_ranges(&ranges, &[3, 0]).is_err());
        assert!(group_item_ranges(&ranges, &[1, 1]).is_err());
        Ok(())
    }

    // With one run per video the shift is master Qwen2-VL's shift_media_spans, at every prefix length.
    #[test]
    fn one_run_per_item_shift_matches_the_per_span_shift() {
        let spans = vec![(2, 5), (8, 12), (12, 15)];
        for prefix in 0..=16 {
            let mut per_span = spans.clone();
            let mut per_item = spans.clone();
            let expected =
                crate::qwen2vl::inputs_processor::shift_media_spans(&mut per_span, prefix);
            let shifted = shift_item_runs(&mut per_item, &[1, 1, 1], prefix);
            match (expected, shifted) {
                (Ok(cached), Ok((shift_cached, current))) => {
                    assert_eq!(
                        (shift_cached, per_item.clone()),
                        (cached, per_span),
                        "prefix {prefix}"
                    );
                    assert_eq!(current, per_item.len(), "prefix {prefix}");
                }
                (Err(_), Err(_)) => {}
                (expected, shifted) => panic!("prefix {prefix}: {expected:?} vs {shifted:?}"),
            }
        }
    }

    #[test]
    fn multi_run_shift_caches_or_keeps_whole_items() -> Result<()> {
        let mut runs = vec![(2, 5), (6, 9), (12, 16)];
        assert_eq!(shift_item_runs(&mut runs, &[2, 1], 10)?, (1, 1));
        assert_eq!(runs, vec![(2, 6)]);
        let mut split = vec![(2, 5), (6, 9), (12, 16)];
        assert!(shift_item_runs(&mut split, &[2, 1], 4).is_err());
        Ok(())
    }
}

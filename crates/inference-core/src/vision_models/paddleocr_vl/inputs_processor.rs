//! PaddleOCR-VL inputs processor; the template emits one placeholder per image, expanded to t*h*w/merge^2.

use std::{any::Any, sync::Arc};

use anyhow::Result;
use candle_core::{Device, Tensor};
use either::Either;
use image::DynamicImage;
use indexmap::IndexMap;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::paged_attention::PagedAttentionMeta;
use crate::{
    device_map::DeviceMapper,
    paged_attention::block_hash::MultimodalKind,
    pipeline::{
        processing::default_process, InputProcessorOutput, InputsProcessor, MessagesAction,
        Processor,
    },
    request::ReasoningEffort,
    sequence::{build_mm_features_from_ranges, find_placeholder_delimited_ranges},
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::PreProcessorConfig,
    },
    MessageContent, Tool,
};

use super::preprocess::{preprocess_decoded, MERGE};
use super::PaddleOcrVlVisionSpecificArgs;
use crate::vision_models::media_host::MediaInputsProcessor;
use inference_nn::media_inputs::processor::{
    InputsHost, MediaSequence, ModelInputs, MultimodalInputsProcessor, TextInputs,
};

// One image grid row as (t, h, w) in patches.
type ImageGrid = (usize, usize, usize);

pub struct PaddleOcrVlProcessor;

impl PaddleOcrVlProcessor {
    pub const IMAGE_START: &'static str = "<|IMAGE_START|>";
    pub const IMAGE_PLACEHOLDER: &'static str = "<|IMAGE_PLACEHOLDER|>";
    pub const IMAGE_END: &'static str = "<|IMAGE_END|>";
    // Stops the expand loop from re-matching the copies it just inserted.
    const EXPAND_MARKER: &'static str = "<|IMAGE_EXPAND_TMP|>";
}

impl Processor for PaddleOcrVlProcessor {
    // The template iterates content as typed parts; a bare string would be iterated per char and dropped.
    fn process(
        &self,
        pipeline: &dyn crate::pipeline::Pipeline,
        messages: Vec<IndexMap<String, MessageContent>>,
        add_generation_prompt: bool,
        add_special_tokens: bool,
        enable_thinking: Option<bool>,
        reasoning_effort: Option<ReasoningEffort>,
        tools: Vec<Tool>,
    ) -> anyhow::Result<(Vec<u32>, String)> {
        let messages = messages
            .into_iter()
            .map(|message| {
                message
                    .into_iter()
                    .map(|(key, value)| match (key.as_str(), value) {
                        ("content", Either::Left(text)) => (
                            key,
                            Either::Right(vec![IndexMap::from([
                                ("type".to_string(), Value::String("text".to_string())),
                                ("text".to_string(), Value::String(text)),
                            ])]),
                        ),
                        (_, value) => (key, value),
                    })
                    .collect()
            })
            .collect();
        default_process(
            pipeline,
            messages,
            add_generation_prompt,
            add_special_tokens,
            enable_thinking,
            reasoning_effort,
            self.template_action(),
            tools,
        )
    }
    // The model takes every image's patches and skips the cached ones itself, so a prefix hit must not drop them.
    fn retain_prefix_cached_images(&self) -> bool {
        true
    }

    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(Arc::new(PaddleOcrVlImageProcessor)))
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[Self::IMAGE_START, Self::IMAGE_PLACEHOLDER, Self::IMAGE_END]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

struct PaddleOcrVlImageProcessor;

fn replace_first_occurrence(text: &str, to_replace: &str, replacement: &str) -> String {
    if let Some(pos) = text.find(to_replace) {
        let mut result = text.to_string();
        result.replace_range(pos..pos + to_replace.len(), replacement);
        result
    } else {
        text.to_string()
    }
}

// The i-th placeholder takes the i-th grid; a prompt can carry a literal placeholder, so counts must match.
fn expand_placeholders(text: &str, grids: &[ImageGrid], merge: usize) -> anyhow::Result<String> {
    let placeholders = text
        .matches(PaddleOcrVlProcessor::IMAGE_PLACEHOLDER)
        .count();
    if placeholders != grids.len() {
        anyhow::bail!(
            "prompt has {placeholders} image placeholders for {} images",
            grids.len()
        );
    }
    let merge_length = merge * merge;
    let mut out = text.to_string();
    let mut index = 0;
    while out.contains(PaddleOcrVlProcessor::IMAGE_PLACEHOLDER) {
        let (t, h, w) = grids[index];
        let n = t * h * w / merge_length;
        out = replace_first_occurrence(
            &out,
            PaddleOcrVlProcessor::IMAGE_PLACEHOLDER,
            &PaddleOcrVlProcessor::EXPAND_MARKER.repeat(n),
        );
        index += 1;
    }
    Ok(out.replace(
        PaddleOcrVlProcessor::EXPAND_MARKER,
        PaddleOcrVlProcessor::IMAGE_PLACEHOLDER,
    ))
}

// Placeholders share one id; the span's image hash keeps same-shape images apart and prefix hits off mid-span.
fn register_image_span(seq: &mut dyn MediaSequence, ids: &[u32], tokenizer: &Tokenizer) {
    if !seq.mm_features().is_empty() {
        return;
    }
    let (Some(hashes), Some(pad_id), Some(start_id), Some(end_id)) = (
        seq.image_hashes().map(<[u64]>::to_vec),
        tokenizer.token_to_id(PaddleOcrVlProcessor::IMAGE_PLACEHOLDER),
        tokenizer.token_to_id(PaddleOcrVlProcessor::IMAGE_START),
        tokenizer.token_to_id(PaddleOcrVlProcessor::IMAGE_END),
    ) else {
        return;
    };
    let ranges = find_placeholder_delimited_ranges(ids, pad_id, start_id, end_id);
    let features = build_mm_features_from_ranges(&ranges, &hashes, MultimodalKind::Image);
    if !features.is_empty() {
        seq.set_mm_features(features);
    }
}

fn placeholder_runs(ids: &[u32], placeholder: u32) -> usize {
    ids.iter()
        .zip(std::iter::once(&u32::MAX).chain(ids))
        .filter(|&(&id, &prev)| id == placeholder && prev != placeholder)
        .count()
}

fn grid_rows(grid: &Tensor) -> Vec<ImageGrid> {
    grid.to_vec2::<u32>()
        .unwrap()
        .into_iter()
        .map(|g| (g[0] as usize, g[1] as usize, g[2] as usize))
        .collect()
}

impl PaddleOcrVlImageProcessor {
    /// Preprocesses the sequence's images once and expands its placeholders to the grid, registering the image span.
    fn expand_image_prompt(
        &self,
        seq: &mut dyn MediaSequence,
        tokenizer: &Tokenizer,
        config: &PreProcessorConfig,
        device: &Device,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<(Tensor, Vec<ImageGrid>)> {
        let (pixel_values, row_grids) = match &seq.multimodal().cached_pixel_values {
            Some(cached) => (
                cached.clone(),
                grid_rows(seq.multimodal().cached_img_thw.as_ref().unwrap()),
            ),
            None => {
                let PreprocessedImages {
                    pixel_values,
                    image_grid_thw,
                    ..
                } = self.preprocess(
                    seq.clone_images().expect("Need images by this point."),
                    vec![],
                    config,
                    device,
                    (usize::MAX, usize::MAX),
                )?;
                seq.multimodal_mut().cached_pixel_values = Some(pixel_values.clone());
                seq.multimodal_mut().cached_img_thw = image_grid_thw.clone();
                (pixel_values, grid_rows(image_grid_thw.as_ref().unwrap()))
            }
        };

        if !seq.multimodal().has_changed_prompt {
            let detok = tokenizer
                .decode(seq.get_toks(), false)
                .expect("Detokenization failed!");
            let detok = expand_placeholders(&detok, &row_grids, MERGE)?;
            let ids = tokenizer
                .encode_fast(detok.clone(), false)
                .expect("Tokenization failed!")
                .get_ids()
                .to_vec();
            seq.set_initial_prompt(detok);
            // Before set_toks_and_reallocate: the block hashes it triggers must see the span.
            register_image_span(seq, &ids, tokenizer);
            seq.set_toks_and_reallocate(ids, paged_attn_metadata);
            seq.multimodal_mut().has_changed_prompt = true;
        }
        Ok((pixel_values, row_grids))
    }
}

impl MultimodalInputsProcessor for PaddleOcrVlImageProcessor {
    // The scheduler looks up prefix-cache blocks before process_inputs runs, so it must see the expanded prompt.
    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut dyn MediaSequence],
        device: &Device,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<()> {
        if !input_seqs.iter().any(|seq| seq.has_images()) {
            return Ok(());
        }
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("PaddleOcrVlImageProcessor requires a specified tokenizer.");
        };
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");
        for seq in input_seqs.iter_mut().filter(|seq| seq.has_images()) {
            self.expand_image_prompt(
                &mut **seq,
                &tokenizer,
                config,
                device,
                paged_attn_metadata.as_deref_mut(),
            )?;
        }
        Ok(())
    }

    fn process_inputs(
        &self,
        host: &dyn InputsHost,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut dyn MediaSequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        sliding_window: Option<usize>,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        if is_xlora {
            anyhow::bail!("Cannot make inputs for X-LoRA vision model.");
        }
        if no_kv_cache {
            anyhow::bail!("Vision model must have kv cache.");
        }
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("PaddleOcrVlImageProcessor requires a specified tokenizer.");
        };
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        // Per row, as rows sit at different chunks; a grid attaches once its image tokens are in, else phantom rope.
        let image_pad_id = tokenizer.token_to_id(PaddleOcrVlProcessor::IMAGE_PLACEHOLDER);
        let mut grids: Vec<Vec<ImageGrid>> = Vec::with_capacity(input_seqs.len());
        let mut hashes: Vec<Vec<u64>> = Vec::with_capacity(input_seqs.len());
        let mut pixel_values_accum = Vec::new();
        let mut vision_rows: Vec<usize> = Vec::new();

        for (row, seq) in input_seqs.iter_mut().enumerate() {
            let pixel_values = if seq.has_images() {
                let (pixel_values, _) = self.expand_image_prompt(
                    &mut **seq,
                    &tokenizer,
                    config,
                    device,
                    paged_attn_metadata.as_mut(),
                )?;
                Some(pixel_values)
            } else {
                None
            };
            // One grid per image whose placeholders the prompt so far holds: a prefill chunk can stop between images.
            // After a prefix hit get_toks is only the suffix, so count over the whole prompt.
            let images_seen = image_pad_id.map_or(0, |id| {
                placeholder_runs(seq.prompt_position_source_toks(), id)
            });
            let mut row_grids = seq
                .multimodal()
                .cached_img_thw
                .as_ref()
                .map(grid_rows)
                .unwrap_or_default();
            row_grids.truncate(images_seen);
            // Full list, not the window-scoped one: the model pairs hash i with grid i.
            hashes.push(if row_grids.is_empty() {
                Vec::new()
            } else {
                seq.multimodal()
                    .image_hashes()
                    .map(<[u64]>::to_vec)
                    .unwrap_or_default()
            });
            // The model walks rows by their grids, so a row contributes only the seen images' patches.
            let seen_patches = row_grids.iter().map(|&(t, h, w)| t * h * w).sum::<usize>();
            grids.push(row_grids);
            let Some(pixel_values) = pixel_values else {
                continue;
            };
            if is_prompt && seen_patches > 0 {
                pixel_values_accum.push(pixel_values.narrow(0, 0, seen_patches)?);
                vision_rows.push(row);
            }
        }

        let pixel_values =
            (!pixel_values_accum.is_empty()).then(|| Tensor::cat(&pixel_values_accum, 0).unwrap());
        let image_grid_thw = if grids.iter().all(Vec::is_empty) {
            Vec::new()
        } else {
            grids
        };

        let inference_nn::media_inputs::processor::InnerInputProcessorOutput {
            inputs:
                inference_nn::media_inputs::processor::InputMetadata {
                    input,
                    positions,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                },
            seq_indices,
        } = if is_prompt {
            host.prompt_inputs(
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
            )
            .unwrap()
        } else {
            host.completion_inputs(
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
            )
            .unwrap()
        };

        // mrope positions are recomputed from the full history; after a prefix hit get_toks is only the suffix.
        let max_len = input_seqs
            .iter()
            .map(|seq| seq.prompt_position_source_toks().len())
            .max()
            .unwrap_or(0);
        let mut rows = Vec::with_capacity(input_seqs.len());
        for seq in input_seqs.iter() {
            let mut ids = seq.prompt_position_source_toks().to_vec();
            ids.resize(max_len, 0);
            rows.push(Tensor::new(ids, device).unwrap());
        }
        let input_ids_full = Tensor::stack(&rows, 0).unwrap();

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args: Box::new(PaddleOcrVlVisionSpecificArgs {
                input_ids_full,
                image_grid_thw,
                image_hashes: hashes,
                vision_rows,
            }),
            paged_attn_meta,
            flash_meta,
            recurrent_batch_kind: if is_prompt {
                crate::gdn::RecurrentBatchKind::Prefill
            } else {
                crate::gdn::RecurrentBatchKind::Decode
            },
        });
        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }
}

impl ImagePreProcessor for PaddleOcrVlImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
    const DEFAULT_STD: [f64; 3] = [0.5, 0.5, 0.5];

    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        _videos: Vec<Vec<DynamicImage>>,
        _config: &PreProcessorConfig,
        device: &Device,
        (_, _): (usize, usize),
    ) -> candle_core::Result<PreprocessedImages> {
        if images.is_empty() {
            candle_core::bail!("PaddleOCR-VL needs at least one image.");
        }
        let mut patches = Vec::with_capacity(images.len());
        let mut grid = Vec::with_capacity(images.len() * 3);
        for img in &images {
            let (px, (t, h, w)) = preprocess_decoded(img, device)?;
            patches.push(px);
            grid.extend([t as u32, h as u32, w as u32]);
        }
        let grid = Tensor::from_vec(grid, (images.len(), 3), device)?;
        Ok(PreprocessedImages {
            pixel_values: Tensor::cat(&patches, 0)?,
            pixel_attention_mask: None,
            image_sizes: None,
            num_img_tokens: None,
            aspect_ratio_ids: None,
            aspect_ratio_mask: None,
            num_tiles: None,
            image_grid_thw: Some(grid),
            video_grid_thw: None,
            rows: None,
            cols: None,
            pixel_values_list: None,
            tgt_sizes: None,
            image_sizes_all: None,
            num_crops: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::block_hash::compute_block_hashes;
    use crate::sequence::clamp_prefix_cache_len_for_mm_features;

    // Real tokenizer ids; only their distinctness matters.
    const IMAGE_START_ID: u32 = 101305;
    const IMAGE_PAD_ID: u32 = 101304;
    const IMAGE_END_ID: u32 = 101306;
    const BLOCK: usize = 16;

    fn expanded_ids(n_image_toks: usize) -> Vec<u32> {
        let mut ids = vec![7u32; 20];
        ids.push(IMAGE_START_ID);
        ids.extend(std::iter::repeat_n(IMAGE_PAD_ID, n_image_toks));
        ids.push(IMAGE_END_ID);
        ids.extend([8u32, 9]);
        ids
    }

    #[test]
    fn image_span_separates_prefix_cache_blocks() {
        let ids = expanded_ids(161);
        let ranges =
            find_placeholder_delimited_ranges(&ids, IMAGE_PAD_ID, IMAGE_START_ID, IMAGE_END_ID);
        assert_eq!(
            ranges,
            vec![(20, 163)],
            "span must cover START..END inclusive"
        );

        let feats =
            |hash: u64| build_mm_features_from_ranges(&ranges, &[hash], MultimodalKind::Image);
        let a = compute_block_hashes(&ids, BLOCK, &feats(0xAAAA_AAAA), &[]);
        let b = compute_block_hashes(&ids, BLOCK, &feats(0xBBBB_BBBB), &[]);
        assert!(!a.is_empty(), "prompt must span at least one full block");
        assert_ne!(a, b, "different images hashed to the same blocks");
        // the span is what separates them: without it both images hash like the bare token stream
        assert_ne!(a, compute_block_hashes(&ids, BLOCK, &[], &[]));
    }

    // A hit inside the span leaves fewer image slots than connector rows; `Merger::forward` counts from the start.
    #[test]
    fn prefix_cache_hit_cannot_land_inside_image_span() {
        let ids = expanded_ids(161);
        let ranges =
            find_placeholder_delimited_ranges(&ids, IMAGE_PAD_ID, IMAGE_START_ID, IMAGE_END_ID);
        let features =
            build_mm_features_from_ranges(&ranges, &[0xAAAA_AAAA], MultimodalKind::Image);
        for hit in [21usize, 100, 182] {
            let clamped = clamp_prefix_cache_len_for_mm_features(hit, BLOCK, &features);
            assert!(
                clamped <= 20,
                "hit {hit} clamped to {clamped}, inside the span"
            );
        }
        // Past the span the whole image is cached and `input_ids` has no image slots left: legal.
        assert_eq!(
            clamp_prefix_cache_len_for_mm_features(183, BLOCK, &features),
            183
        );
    }

    #[test]
    fn every_image_gets_its_own_grid_row() {
        let call = |images: Vec<DynamicImage>| {
            PaddleOcrVlImageProcessor.preprocess(
                images,
                vec![],
                &PreProcessorConfig::default(),
                &Device::Cpu,
                (usize::MAX, usize::MAX),
            )
        };
        let out = call(vec![
            DynamicImage::new_rgb8(64, 64),
            DynamicImage::new_rgb8(128, 64),
        ])
        .expect("two images must be accepted");
        let grid = out.image_grid_thw.expect("grid");
        assert_eq!(grid.dims(), &[2, 3], "one grid row per image");
        let rows = grid_rows(&grid);
        assert_ne!(
            rows[0], rows[1],
            "differently sized images need different grids"
        );
        let patches: usize = rows.iter().map(|&(t, h, w)| t * h * w).sum();
        assert_eq!(
            out.pixel_values.dim(0).unwrap(),
            patches,
            "patches must be both images concatenated"
        );
        assert!(call(vec![]).is_err(), "zero images must not panic");
    }

    #[test]
    fn expand_placeholder_count_matches_grid() {
        // 1*14*46 / 2^2 = 161
        let text = format!(
            "User: {}{}{}OCR:",
            PaddleOcrVlProcessor::IMAGE_START,
            PaddleOcrVlProcessor::IMAGE_PLACEHOLDER,
            PaddleOcrVlProcessor::IMAGE_END,
        );
        let expanded = expand_placeholders(&text, &[(1, 14, 46)], MERGE).unwrap();
        let count = expanded
            .matches(PaddleOcrVlProcessor::IMAGE_PLACEHOLDER)
            .count();
        assert_eq!(count, 161);
        assert!(!expanded.contains(PaddleOcrVlProcessor::EXPAND_MARKER));
        assert!(expanded.contains(PaddleOcrVlProcessor::IMAGE_START));
        assert!(expanded.contains("OCR:"));
    }

    #[test]
    fn literal_placeholder_in_user_text_is_an_error() {
        let text = format!(
            "User: {p}{p}OCR:",
            p = PaddleOcrVlProcessor::IMAGE_PLACEHOLDER
        );
        assert!(expand_placeholders(&text, &[(1, 14, 46)], MERGE).is_err());
    }
}

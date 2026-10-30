#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;
use std::{any::Any, ops::Range};

use image::{DynamicImage, GenericImageView, Rgb};
use inference_tensor::Result;
use inference_tensor::{DType, Device, Tensor};
use itertools::Itertools;
use regex_automata::meta::Regex;
use tokenizers::Tokenizer;

use super::config::Config as LLaVAConfig;
use super::model::LLaVAVisionSpecificArgs;
use super::utils::{
    LLaVAImageProcessor, calculate_unpad, divide_to_samples, expand2square,
    get_anyres_image_grid_shape, get_num_samples, resize_and_pad_image, select_best_resolution,
};
use crate::device_map::DeviceMapper;
use crate::media_inputs::{
    image_processor::{self, ImagePreProcessor, PreprocessedImages},
    media::build_mm_features_from_ranges,
    preprocessor_config::{self, PreProcessorConfig, ToFilter},
    processor::{
        InputProcessorOutput, InputsHost, InputsProcessorValidationError, MediaSequence,
        ModelInputs, MultimodalInputsProcessor, ProcessInputsCall, TextInputs,
    },
};
use crate::paged_attention::{
    PagedAttentionMeta,
    block_hash::{MultiModalFeature, MultimodalKind},
};
use crate::vision::multimodal_layout::{
    MultimodalEmbeddingMap, MultimodalEncoderKey, MultimodalItemLayout, PackedMultimodalLayout,
    RequestMultimodalLayout,
};

type LLaVAPromptTokens = (Vec<i64>, Vec<(usize, usize)>);

/// LLaVA 1.5 pads each image to a square, one encoder sample; LLaVA-Next (`anyres`) adds tiles over the best
/// `image_grid_pinpoints` resolution after a resized base view.
pub struct LLaVAInputProcessor {
    image_tag_splitter: Regex,
    model_config: LLaVAConfig,
    anyres: bool,
}

impl LLaVAInputProcessor {
    pub fn new(model_config: LLaVAConfig, anyres: bool) -> Self {
        let image_tag_splitter = Regex::new(r"<image>").expect("Failed to compile split regex.");
        Self {
            image_tag_splitter,
            model_config,
            anyres,
        }
    }

    /// Placeholder tokens of an image: the base view's patches, and for `anyres` its unpadded tile rows, each with
    /// a newline.
    pub fn get_num_image_tokens(cfg: &LLaVAConfig, anyres: bool, image_size: (u32, u32)) -> usize {
        let patch_size = cfg.vision_config.patch_size;
        let patch_per_side = cfg.vision_config.image_size / patch_size;
        if !anyres {
            return patch_per_side * patch_per_side;
        }
        let image_grid_pinpoints = cfg.image_grid_pinpoints.clone().unwrap();
        let anyres_grid_shape =
            get_anyres_image_grid_shape(image_size, &image_grid_pinpoints, patch_size as u32);
        let unpad_shape = calculate_unpad(anyres_grid_shape, image_size);
        patch_per_side * patch_per_side + (unpad_shape.0 as usize + 1) * (unpad_shape.1 as usize)
    }

    fn num_image_samples(&self, image: &DynamicImage, config: &PreProcessorConfig) -> usize {
        if !self.anyres {
            return 1;
        }
        let crop_size = (
            *config.crop_size.as_ref().unwrap().get("width").unwrap(),
            *config.crop_size.as_ref().unwrap().get("height").unwrap(),
        );
        get_num_samples(
            image.dimensions(),
            self.model_config
                .image_grid_pinpoints
                .as_ref()
                .expect("LLaVA-Next requires image_grid_pinpoints"),
            crop_size,
        ) as usize
    }
}

fn llava_prompt_tokens(
    tokenizer: &Tokenizer,
    image_tag_splitter: &Regex,
    detokenized: &str,
    image_token_counts: &[usize],
) -> anyhow::Result<LLaVAPromptTokens> {
    let splits = image_tag_splitter
        .split(detokenized)
        .map(|span| &detokenized[span.range()])
        .collect::<Vec<_>>();
    let tag_count = splits.len().saturating_sub(1);
    if tag_count != image_token_counts.len() {
        return Err(InputsProcessorValidationError(format!(
            "LLaVA prompt has {tag_count} image tags but {} images",
            image_token_counts.len()
        ))
        .into());
    }
    // one encode per chunk: encode_batch pads short chunks, which would shift the placeholders
    let prompt_chunks = splits
        .into_iter()
        .map(|chunk| {
            tokenizer
                .encode_fast(chunk, false)
                .map_err(|error| anyhow::Error::msg(error.to_string()))
                .map(|encoding| {
                    encoding
                        .get_ids()
                        .iter()
                        .map(|token| i64::from(*token))
                        .collect::<Vec<_>>()
                })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let image_pads = image_token_counts
        .iter()
        .enumerate()
        .map(|(index, &token_count)| {
            if token_count == 0 {
                return Err(InputsProcessorValidationError(
                    "LLaVA image placeholder cannot be empty".to_string(),
                )
                .into());
            }
            let mut pad = vec![0; token_count];
            pad[0] = -(index as i64 + 1);
            Ok(pad)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut ranges = Vec::with_capacity(image_pads.len());
    let mut offset = 0usize;
    for (chunk, pad) in prompt_chunks.iter().zip(&image_pads) {
        offset += chunk.len();
        ranges.push((offset, pad.len()));
        offset += pad.len();
    }
    let mut tokens = Vec::new();
    for part in prompt_chunks.into_iter().interleave(image_pads) {
        tokens.extend(part);
    }
    Ok((tokens, ranges))
}

fn restore_llava_image_markers(
    tokens: &[u32],
    features: &[MultiModalFeature],
    query: Range<usize>,
    local_query: Range<usize>,
    expected_images: usize,
) -> anyhow::Result<Vec<i64>> {
    if query.len() != local_query.len() {
        anyhow::bail!("LLaVA prompt coordinate ranges have different lengths");
    }
    let active_features = features
        .iter()
        .filter(|feature| {
            feature.kind == MultimodalKind::Image && feature.overlaps(query.start, query.end)
        })
        .collect::<Vec<_>>();
    if active_features.len() != expected_images {
        anyhow::bail!(
            "LLaVA prompt query has {} image placeholders but {expected_images} images",
            active_features.len()
        );
    }
    let mut tokens = tokens
        .iter()
        .map(|token| i64::from(*token))
        .collect::<Vec<_>>();
    for (image_index, feature) in active_features.into_iter().enumerate() {
        if feature.item_range.len() != 1 {
            anyhow::bail!("LLaVA image feature must describe exactly one image");
        }
        if feature.offset < query.start || feature.end() > query.end {
            anyhow::bail!("LLaVA prompt query partially overlaps an image placeholder");
        }
        let local = local_query.start + feature.offset - query.start;
        let token = tokens
            .get_mut(local)
            .ok_or_else(|| anyhow::Error::msg("LLaVA image marker is outside the query"))?;
        *token = -(image_index as i64 + 1);
    }
    Ok(tokens)
}

fn llava_layout_items(features: &[MultiModalFeature]) -> Result<Vec<MultimodalItemLayout>> {
    features
        .iter()
        .filter(|feature| feature.kind == MultimodalKind::Image)
        .map(|feature| {
            if feature.item_range.len() != 1 || feature.hashes.len() != 1 {
                inference_tensor::bail!("LLaVA image feature must describe exactly one image");
            }
            let placeholder = feature.offset..feature.end();
            MultimodalItemLayout::new(
                MultimodalEncoderKey {
                    kind: MultimodalKind::Image,
                    hash: feature.hashes[0],
                },
                feature.item_range.start,
                placeholder.clone(),
                feature.attention_policy,
                vec![MultimodalEmbeddingMap::contiguous(placeholder, 0, 0)?],
            )
        })
        .collect()
}

fn llava_packed_layout(
    input_seqs: &[&mut dyn MediaSequence],
    query_lens: &[usize],
) -> Result<PackedMultimodalLayout> {
    if input_seqs.len() != query_lens.len() {
        inference_tensor::bail!("LLaVA packed multimodal metadata length mismatch");
    }
    let requests = input_seqs
        .iter()
        .zip(query_lens)
        .map(|(seq, &query_len)| {
            if query_len != seq.get_toks().len() {
                inference_tensor::bail!(
                    "LLaVA packed multimodal prefill requires the complete uncached prompt"
                );
            }
            Ok(RequestMultimodalLayout {
                sequence_id: *seq.id(),
                query: 0..query_len,
                items: llava_layout_items(seq.mm_features())?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    PackedMultimodalLayout::new(&requests)
}

impl LLaVAInputProcessor {
    // the prompt with each `<image>` expanded to its placeholder, set on `seq` with its media features
    fn expand_prompt(
        &self,
        tokenizer: &Tokenizer,
        seq: &mut dyn MediaSequence,
        detokenized: &str,
        image_token_counts: &[usize],
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<Vec<i64>> {
        let (tokens, image_ranges) = llava_prompt_tokens(
            tokenizer,
            &self.image_tag_splitter,
            detokenized,
            image_token_counts,
        )?;
        let hashes = seq.image_hashes().unwrap_or_default().to_vec();
        if hashes.len() != image_ranges.len() {
            anyhow::bail!(
                "LLaVA has {} image hashes but {} image placeholders",
                hashes.len(),
                image_ranges.len()
            );
        }
        let new_ids = tokens
            .iter()
            .map(|x| if *x < 0 { 0u32 } else { *x as u32 })
            .collect::<Vec<_>>();
        let new_prompt = tokenizer
            .decode(&new_ids, false)
            .map_err(|error| anyhow::Error::msg(error.to_string()))?;
        seq.set_initial_prompt(new_prompt);
        if seq.mm_features().is_empty() {
            seq.set_mm_features(build_mm_features_from_ranges(
                &image_ranges,
                &hashes,
                MultimodalKind::Image,
            ));
        }
        seq.set_toks_and_reallocate(new_ids, paged_attn_metadata);
        seq.multimodal_mut().has_changed_prompt = true;
        Ok(tokens)
    }
}

impl MultimodalInputsProcessor for LLaVAInputProcessor {
    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut dyn MediaSequence],
        _device: &Device,
        _other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let Some(tokenizer) = tokenizer else {
            return Err(anyhow::Error::msg(
                "LLaVAInputProcessor requires a specified tokenizer.",
            ));
        };
        if !input_seqs.iter().any(|seq| seq.has_images()) {
            return Ok(());
        }

        for seq in input_seqs.iter_mut() {
            if seq.multimodal().has_changed_prompt {
                continue;
            }
            let Some(images) = seq.images() else {
                continue;
            };
            if images.is_empty() {
                continue;
            }
            let image_token_counts = images
                .iter()
                .map(|image| {
                    Self::get_num_image_tokens(&self.model_config, self.anyres, image.dimensions())
                })
                .collect::<Vec<_>>();
            let detokenized = tokenizer
                .decode(seq.get_toks(), false)
                .map_err(|error| anyhow::Error::msg(error.to_string()))?;
            self.expand_prompt(
                &tokenizer,
                &mut **seq,
                &detokenized,
                &image_token_counts,
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
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        sliding_window: Option<usize>,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> anyhow::Result<InputProcessorOutput> {
        if no_kv_cache {
            return Err(anyhow::Error::msg("Vision model must have kv cache."));
        }
        let Some(tokenizer) = tokenizer else {
            return Err(anyhow::Error::msg(
                "LLaVAInputProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config
            .clone()
            .expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        // a prefix-cache hit leaves the images on the sequence, so decode steps must not splice them again
        let has_images = is_prompt && input_seqs.iter().any(|seq| seq.has_images());
        if !has_images {
            return host
                .text_only_inputs(
                    input_seqs,
                    ProcessInputsCall {
                        tokenizer: Some(tokenizer),
                        is_prompt,
                        device,
                        no_kv_cache,
                        last_n_context_len,
                        return_raw_logits,
                        sliding_window,
                        other_config,
                        paged_attn_metadata,
                        mapper,
                    },
                )
                .map(|text| text.into_output(Box::new(LLaVAVisionSpecificArgs::default())));
        }

        let mut pixel_values_accum = Vec::new();
        let mut image_sizes = Vec::new();
        let mut image_token_counts = vec![Vec::new(); input_seqs.len()];
        let mut num_image_samples = Vec::new();
        for (seq_idx, seq) in input_seqs.iter_mut().enumerate() {
            if !seq.has_images() {
                continue;
            }
            let cached = seq.count_prefix_cached_mm_items();
            let images = seq
                .take_images()
                .expect("Need to have images by this point.");
            for image in images.into_iter().skip(cached) {
                let expected_samples = self.num_image_samples(&image, config);
                let PreprocessedImages {
                    pixel_values,
                    image_sizes: Some(image_size),
                    num_img_tokens: Some(num_img_tokens),
                    ..
                } = self.preprocess(
                    vec![image],
                    vec![],
                    config,
                    device,
                    (usize::MAX, usize::MAX),
                )?
                else {
                    anyhow::bail!("LLaVA preprocessing omitted required image metadata");
                };
                if pixel_values.dim(0)? != expected_samples || num_img_tokens.len() != 1 {
                    anyhow::bail!("LLaVA preprocessing returned inconsistent image metadata");
                }
                pixel_values_accum.push(pixel_values);
                image_sizes.push(image_size);
                image_token_counts[seq_idx].push(num_img_tokens[0]);
                num_image_samples.push(expected_samples);
            }
        }
        let pixel_values = if pixel_values_accum.is_empty() {
            None
        } else {
            Some(Tensor::cat(&pixel_values_accum, 0)?)
        };
        let image_sizes = (!image_sizes.is_empty()).then_some(image_sizes);
        let num_image_tokens_flat = image_token_counts
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();

        let mut toks = Vec::new();
        let detokenized = tokenizer
            .decode_batch(
                &input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                false,
            )
            .map_err(|error| anyhow::Error::msg(error.to_string()))?;

        for ((detokenized, seq), image_token_counts) in detokenized
            .into_iter()
            .zip(input_seqs.iter_mut())
            .zip(image_token_counts)
        {
            if image_token_counts.is_empty() {
                toks.push(
                    seq.get_toks()
                        .iter()
                        .map(|token| i64::from(*token))
                        .collect(),
                );
                continue;
            }
            if seq.multimodal().has_changed_prompt {
                let query = seq
                    .active_prompt_query_range()
                    .unwrap_or(0..seq.get_toks().len());
                let local_query = seq
                    .active_prompt_local_query_range()
                    .unwrap_or(0..seq.get_toks().len());
                toks.push(restore_llava_image_markers(
                    seq.get_toks(),
                    seq.mm_features(),
                    query,
                    local_query,
                    image_token_counts.len(),
                )?);
                continue;
            }
            toks.push(self.expand_prompt(
                &tokenizer,
                &mut **seq,
                &detokenized,
                &image_token_counts,
                paged_attn_metadata.as_mut(),
            )?);
        }

        let metadata = host.step_inputs(
            toks.iter().map(Vec::as_slice).collect::<Vec<_>>().into(),
            input_seqs,
            TextInputs {
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata: paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
            },
            is_prompt,
            no_kv_cache,
        );

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
        } = metadata?;
        let image_hashes = input_seqs
            .iter()
            .flat_map(|seq| {
                let cached = seq.count_prefix_cached_mm_items();
                seq.image_hashes()
                    .unwrap_or_default()
                    .iter()
                    .copied()
                    .skip(cached)
            })
            .collect::<Vec<_>>();
        let selected_image_count = image_sizes.as_ref().map_or(0, Vec::len);
        if image_hashes.len() != selected_image_count
            || num_image_tokens_flat.len() != selected_image_count
            || num_image_samples.len() != selected_image_count
        {
            anyhow::bail!("LLaVA image metadata does not match the selected media");
        }
        if let Some(pixels) = &pixel_values
            && pixels.dim(0)? != num_image_samples.iter().sum::<usize>()
        {
            anyhow::bail!("LLaVA pixel samples do not match image metadata");
        }
        let packed_layout = if is_prompt && flash_meta.packed {
            let query_lens = paged_attn_meta
                .as_ref()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .ok_or_else(|| {
                    anyhow::Error::msg("packed LLaVA prefill requires logical query lengths")
                })?;
            let layout = llava_packed_layout(input_seqs, query_lens)?;
            if layout.token_count() != input.dim(1)? {
                anyhow::bail!(
                    "LLaVA packed layout has {} tokens but input has {}",
                    layout.token_count(),
                    input.dim(1)?
                );
            }
            Some(layout)
        } else {
            None
        };
        Ok(InputProcessorOutput {
            inputs: Box::new(ModelInputs {
                input_ids: input,
                seqlen_offsets: positions,
                context_lens,
                position_ids,
                pixel_values,
                model_specific_args: Box::new(LLaVAVisionSpecificArgs {
                    image_sizes,
                    num_image_tokens: Some(num_image_tokens_flat),
                    num_image_samples: Some(num_image_samples),
                    image_hashes,
                    packed_layout,
                }),
                paged_attn_meta,
                flash_meta,
                recurrent_batch_kind: if is_prompt {
                    crate::gdn::RecurrentBatchKind::Prefill
                } else {
                    crate::gdn::RecurrentBatchKind::Decode
                },
            }),
            seq_indices,
        })
    }
}

impl ImagePreProcessor for LLaVAInputProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];
    fn preprocess(
        &self,
        images: Vec<image::DynamicImage>,
        videos: Vec<Vec<image::DynamicImage>>,
        config: &preprocessor_config::PreProcessorConfig,
        device: &inference_tensor::Device,
        (_, _): (usize, usize),
    ) -> inference_tensor::Result<image_processor::PreprocessedImages> {
        if images.len() > 1 {
            inference_tensor::bail!("Can only process one image per batch"); // This is no different from phi3_input_processor
        };
        assert!(videos.is_empty());

        let resized_size = *config.size.as_ref().unwrap().get("shortest_edge").unwrap() as usize;
        let image = &images[0];
        let original_size = image.dimensions();
        let filter = config.resampling.to_filter()?;
        let image_mean = config
            .image_mean
            .unwrap_or(Self::DEFAULT_MEAN)
            .map(|x| x as f32);
        let image_std = config
            .image_std
            .unwrap_or(Self::DEFAULT_STD)
            .map(|x| x as f32);
        let samples = if self.anyres {
            let image_grid_pinpoints = self.model_config.image_grid_pinpoints.clone().unwrap();
            let best_resolution = select_best_resolution(original_size, &image_grid_pinpoints);
            let image_padded = resize_and_pad_image(image, best_resolution);
            let mut samples = vec![inference_vision::resize_exact(
                image,
                resized_size as u32,
                resized_size as u32,
                filter,
            )];
            samples.extend(divide_to_samples(
                &image_padded,
                (
                    *config.crop_size.as_ref().unwrap().get("width").unwrap(),
                    *config.crop_size.as_ref().unwrap().get("height").unwrap(),
                ),
            ));
            samples
        } else {
            let mean_color = image_mean.map(|x| (x * 255.0) as u8);
            vec![expand2square(image, Rgb::from(mean_color))]
        };
        let pixel_values = samples
            .iter()
            .map(|x| {
                LLaVAImageProcessor::process_one_image(
                    x,
                    config,
                    resized_size as u32,
                    filter,
                    DType::BF16,
                    device,
                    &image_mean,
                    &image_std,
                )
            })
            .collect::<Result<Vec<Tensor>>>()?;
        let pixel_values = Tensor::stack(&pixel_values, 0)?;

        Ok(image_processor::PreprocessedImages {
            pixel_values,
            pixel_attention_mask: None,
            image_sizes: Some((original_size.0 as usize, original_size.1 as usize)),
            num_img_tokens: Some(vec![Self::get_num_image_tokens(
                &self.model_config,
                self.anyres,
                original_size,
            )]),
            aspect_ratio_ids: None,
            aspect_ratio_mask: None,
            num_tiles: None,
            image_grid_thw: None,
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
    use std::collections::HashMap;

    use inference_tensor::{Device, Tensor};

    use super::*;
    use crate::paged_attention::block_hash::MultimodalAttentionPolicy;
    use crate::vision::multimodal_layout::MultimodalEncoderOutputs;

    fn feature(item_range: Range<usize>) -> MultiModalFeature {
        MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range,
            hashes: vec![23],
            offset: 1,
            length: 2,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        }
    }

    #[test]
    fn packed_layout_splices_media_and_preserves_text_request() {
        let layout = PackedMultimodalLayout::new(&[
            RequestMultimodalLayout {
                sequence_id: 1,
                query: 0..4,
                items: llava_layout_items(&[feature(0..1)]).unwrap(),
            },
            RequestMultimodalLayout {
                sequence_id: 2,
                query: 0..2,
                items: vec![],
            },
        ])
        .unwrap();
        let text = Tensor::from_vec(
            vec![0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            (1, 6, 2),
            &Device::Cpu,
        )
        .unwrap();
        let outputs: MultimodalEncoderOutputs = HashMap::from([(
            MultimodalEncoderKey {
                kind: MultimodalKind::Image,
                hash: 23,
            },
            vec![Tensor::from_vec(vec![20f32, 21., 30., 31.], (2, 2), &Device::Cpu).unwrap()],
        )]);

        assert_eq!(
            layout
                .splice_embeddings(&text, &outputs)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0., 1., 20., 21., 30., 31., 6., 7., 8., 9., 10., 11.]
        );
    }

    #[test]
    fn suffix_only_marker_uses_local_coordinates() {
        let mut feature = feature(0..1);
        feature.offset = 4;
        feature.length = 1;

        assert_eq!(
            restore_llava_image_markers(&[7, 0, 8], &[feature], 3..6, 0..3, 1).unwrap(),
            vec![7, -1, 8]
        );
    }

    #[test]
    fn packed_layout_rejects_grouped_image_feature() {
        assert!(llava_layout_items(&[feature(0..2)]).is_err());
    }

    #[test]
    fn image_markers_use_suffix_local_coordinates() {
        let features =
            build_mm_features_from_ranges(&[(1, 2), (5, 2)], &[11, 13], MultimodalKind::Image);
        let tokens = restore_llava_image_markers(&[7, 0, 0, 8], &features, 4..8, 0..4, 1).unwrap();
        assert_eq!(tokens, vec![7, -1, 0, 8]);
    }

    #[test]
    fn image_markers_reject_partial_or_missing_placeholders() {
        let features = build_mm_features_from_ranges(&[(5, 2)], &[13], MultimodalKind::Image);
        assert!(restore_llava_image_markers(&[0, 8], &features, 6..8, 0..2, 1).is_err());
        assert!(restore_llava_image_markers(&[0, 0, 8], &features, 5..8, 0..3, 2).is_err());
    }
}

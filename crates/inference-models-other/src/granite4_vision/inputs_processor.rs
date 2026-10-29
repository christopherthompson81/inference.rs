#![allow(clippy::cast_possible_truncation)]

use std::{any::Any, ops::Range, sync::Arc};

use image::{DynamicImage, GenericImageView};
use inference_nn::vision::llava_tiling::{
    LLaVAImageProcessor, divide_to_samples, resize_and_pad_image, select_best_resolution,
};
use inference_tensor::{DType, Device, Result, Tensor};
use tokenizers::Tokenizer;

use crate::device_map::DeviceMapper;
use crate::media_inputs::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    media::build_mm_features_from_ranges,
    preprocessor_config::{PreProcessorConfig, ToFilter},
    processor::{
        InputProcessorOutput, InputsHost, InputsProcessorValidationError, MediaSequence,
        ModelInputs, MultimodalInputsProcessor, TextInputs,
    },
};
use crate::paged_attention::{PagedAttentionMeta, block_hash::MultimodalKind};

use super::{Config, Granite4VisionSpecificArgs, image_token_count};

pub const IMAGE_TOKEN: &str = "<image>";

pub struct Granite4VisionInputsProcessor {
    config: Config,
}

struct Preprocessed {
    pixel_values: Tensor,
    /// `(height, width)` per image.
    image_sizes: Vec<(usize, usize)>,
    num_crops: Vec<usize>,
}

/// Splits the image token's runs into one range per image, in order; adjacent images share a run.
fn image_ranges(
    ids: &[u32],
    image_token: u32,
    counts: &[usize],
) -> anyhow::Result<Vec<(usize, usize)>> {
    let mut ranges = Vec::with_capacity(counts.len());
    let mut counts = counts.iter();
    let mut position = 0;
    while position < ids.len() {
        if ids[position] != image_token {
            position += 1;
            continue;
        }
        let Some(&count) = counts.next() else {
            anyhow::bail!("Granite 4 Vision prompt has more image tokens than its images");
        };
        if ids[position..]
            .iter()
            .take(count)
            .any(|&id| id != image_token)
            || ids.len() < position + count
        {
            anyhow::bail!("Granite 4 Vision prompt cuts an image's {count} tokens short");
        }
        ranges.push((position, count));
        position += count;
    }
    if counts.next().is_some() {
        anyhow::bail!("Granite 4 Vision prompt has fewer image tokens than its images");
    }
    Ok(ranges)
}

impl Granite4VisionInputsProcessor {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// LLaVA-Next's crops: the image resized to one tile, then the tiles of its padded best-fit resolution.
    fn crops(
        &self,
        image: &DynamicImage,
        config: &PreProcessorConfig,
    ) -> Result<Vec<DynamicImage>> {
        let filter = config.resampling.to_filter()?;
        let tile = self.config.vision_config.image_size as u32;
        let image = DynamicImage::ImageRgb8(image.to_rgb8());
        let (width, height) = image.dimensions();
        let (best_height, best_width) =
            select_best_resolution((height, width), &self.config.image_grid_pinpoints);
        let padded = resize_and_pad_image(&image, (best_width, best_height));
        let mut crops = vec![inference_vision::resize_exact(&image, tile, tile, filter)];
        crops.extend(divide_to_samples(&padded, (tile, tile)));
        Ok(crops)
    }

    fn cached_or_preprocess(
        &self,
        seq: &mut dyn MediaSequence,
        config: &PreProcessorConfig,
        device: &Device,
    ) -> Result<Preprocessed> {
        let mm = seq.multimodal();
        if let (Some(pixel_values), Some(sizes), Some(num_crops)) = (
            &mm.cached_pixel_values,
            &mm.cached_spatial_shapes,
            &mm.cached_num_crops,
        ) {
            let image_sizes = sizes
                .to_vec2::<u32>()?
                .into_iter()
                .map(|size| (size[0] as usize, size[1] as usize))
                .collect();
            return Ok(Preprocessed {
                pixel_values: pixel_values.clone(),
                image_sizes,
                num_crops: num_crops.clone(),
            });
        }
        let images = seq
            .take_images()
            .expect("Need to have images by this point.");
        let processed =
            self.preprocess(images, vec![], config, device, (usize::MAX, usize::MAX))?;
        let image_sizes: Vec<(usize, usize)> = processed
            .image_sizes_all
            .expect("Granite 4 Vision returns image sizes")
            .into_iter()
            .map(|(height, width)| (height as usize, width as usize))
            .collect();
        let num_crops = processed
            .num_crops
            .expect("Granite 4 Vision returns crop counts");
        let sizes = image_sizes
            .iter()
            .flat_map(|&(height, width)| [height as u32, width as u32])
            .collect::<Vec<_>>();
        let mm = seq.multimodal_mut();
        mm.cached_pixel_values = Some(processed.pixel_values.clone());
        mm.cached_spatial_shapes = Some(Tensor::from_vec(
            sizes,
            (image_sizes.len(), 2),
            &Device::Cpu,
        )?);
        mm.cached_num_crops = Some(num_crops.clone());
        Ok(Preprocessed {
            pixel_values: processed.pixel_values,
            image_sizes,
            num_crops,
        })
    }

    fn maybe_expand_prompt(
        &self,
        tokenizer: &Tokenizer,
        seq: &mut dyn MediaSequence,
        processed: &Preprocessed,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        if seq.multimodal().has_changed_prompt {
            return Ok(());
        }
        let prompt = seq.get_initial_prompt();
        let placeholders = prompt.matches(IMAGE_TOKEN).count();
        if placeholders != processed.image_sizes.len() {
            return Err(InputsProcessorValidationError(format!(
                "The number of `{IMAGE_TOKEN}` tokens ({placeholders}) must match the number of images ({})",
                processed.image_sizes.len()
            ))
            .into());
        }
        let counts = processed
            .image_sizes
            .iter()
            .map(|&size| image_token_count(&self.config, size))
            .collect::<Result<Vec<_>>>()?;
        let mut parts = prompt.split(IMAGE_TOKEN);
        let mut expanded = parts.next().unwrap_or_default().to_string();
        for (count, part) in counts.iter().zip(parts) {
            expanded.push_str(&IMAGE_TOKEN.repeat(*count));
            expanded.push_str(part);
        }
        seq.set_initial_prompt(expanded.clone());
        let ids = tokenizer
            .encode_fast(expanded, false)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?
            .get_ids()
            .to_vec();
        if seq.mm_features().is_empty() {
            let hashes = seq
                .image_hashes()
                .ok_or_else(|| {
                    anyhow::Error::msg("Granite 4 Vision images are missing content hashes")
                })?
                .to_vec();
            let ranges = image_ranges(&ids, self.config.image_token_index, &counts)?;
            seq.set_mm_features(build_mm_features_from_ranges(
                &ranges,
                &hashes,
                MultimodalKind::Image,
            ));
        }
        seq.set_toks_and_reallocate(ids, paged_attn_metadata);
        seq.multimodal_mut().has_changed_prompt = true;
        Ok(())
    }

    /// The crops and metadata of the images in `range`.
    fn select(
        processed: &Preprocessed,
        range: Range<usize>,
    ) -> Result<Option<(Tensor, Granite4VisionSpecificArgs)>> {
        if range.start > range.end || range.end > processed.num_crops.len() {
            inference_tensor::bail!(
                "Granite 4 Vision image range {range:?} exceeds {} preprocessed images",
                processed.num_crops.len()
            );
        }
        let crop_start = processed.num_crops[..range.start].iter().sum();
        let crop_len = processed.num_crops[range.clone()].iter().sum();
        if crop_len == 0 {
            return Ok(None);
        }
        Ok(Some((
            processed.pixel_values.narrow(0, crop_start, crop_len)?,
            Granite4VisionSpecificArgs {
                image_sizes: processed.image_sizes[range.clone()].to_vec(),
                num_crops: processed.num_crops[range].to_vec(),
            },
        )))
    }
}

impl MultimodalInputsProcessor for Granite4VisionInputsProcessor {
    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut dyn MediaSequence],
        device: &Device,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("Granite 4 Vision image processor requires a tokenizer");
        };
        if !input_seqs.iter().any(|seq| seq.has_images()) {
            return Ok(());
        }
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");
        for seq in input_seqs {
            if !seq.has_images() || seq.multimodal().has_changed_prompt {
                continue;
            }
            let processed = self.cached_or_preprocess(&mut **seq, config, device)?;
            self.maybe_expand_prompt(
                &tokenizer,
                &mut **seq,
                &processed,
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
            anyhow::bail!("Vision model must have kv cache.");
        }
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("Granite 4 Vision image processor requires a tokenizer");
        };
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let mut pixel_values = Vec::new();
        let mut args = Granite4VisionSpecificArgs {
            image_sizes: Vec::new(),
            num_crops: Vec::new(),
        };
        if is_prompt {
            for seq in input_seqs.iter_mut().filter(|seq| seq.has_images()) {
                let processed = self.cached_or_preprocess(&mut **seq, config, device)?;
                self.maybe_expand_prompt(
                    &tokenizer,
                    &mut **seq,
                    &processed,
                    paged_attn_metadata.as_mut(),
                )?;
                let image_count = processed.num_crops.len();
                let range = if seq.is_chunked_prefill_view() {
                    seq.active_local_multimodal_item_range(MultimodalKind::Image, image_count)
                        .ok_or_else(|| {
                            inference_tensor::Error::msg(
                                "Granite 4 Vision image chunk is missing its active image range",
                            )
                        })?
                } else {
                    seq.count_prefix_cached_mm_items_by_kind(MultimodalKind::Image)..image_count
                };
                if let Some((crops, selected)) = Self::select(&processed, range)? {
                    pixel_values.push(crops);
                    args.image_sizes.extend(selected.image_sizes);
                    args.num_crops.extend(selected.num_crops);
                }
            }
        }

        let toks = input_seqs
            .iter()
            .map(|seq| seq.get_toks())
            .collect::<Vec<_>>()
            .into();
        let text = TextInputs {
            device,
            last_n_context_len,
            return_raw_logits,
            paged_attn_metadata: paged_attn_metadata.as_mut(),
            mapper,
            sliding_window,
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
            host.prompt_inputs(toks, input_seqs, text)?
        } else {
            host.completion_inputs(toks, input_seqs, text, no_kv_cache, None)?
        };

        Ok(InputProcessorOutput {
            inputs: Box::new(ModelInputs {
                input_ids: input,
                seqlen_offsets: positions,
                context_lens,
                position_ids,
                pixel_values: if pixel_values.is_empty() {
                    None
                } else {
                    Some(Tensor::cat(&pixel_values, 0)?)
                },
                model_specific_args: Box::new(args),
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

impl ImagePreProcessor for Granite4VisionInputsProcessor {
    const DEFAULT_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
    const DEFAULT_STD: [f64; 3] = [0.5, 0.5, 0.5];

    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        _batch_info: (usize, usize),
    ) -> Result<PreprocessedImages> {
        assert!(videos.is_empty());
        let filter = config.resampling.to_filter()?;
        let tile = self.config.vision_config.image_size as u32;
        let mean = config
            .image_mean
            .unwrap_or(Self::DEFAULT_MEAN)
            .map(|x| x as f32);
        let std = config
            .image_std
            .unwrap_or(Self::DEFAULT_STD)
            .map(|x| x as f32);
        let mut pixel_values = Vec::new();
        let mut image_sizes = Vec::with_capacity(images.len());
        let mut num_crops = Vec::with_capacity(images.len());
        for image in images {
            let (width, height) = image.dimensions();
            image_sizes.push((height, width));
            let crops = self.crops(&image, config)?;
            num_crops.push(crops.len());
            for crop in crops {
                pixel_values.push(LLaVAImageProcessor::process_one_image(
                    &crop,
                    config,
                    tile,
                    filter,
                    DType::F32,
                    device,
                    &mean,
                    &std,
                )?);
            }
        }
        if pixel_values.is_empty() {
            inference_tensor::bail!("Granite 4 Vision preprocessing received no images");
        }
        Ok(PreprocessedImages {
            pixel_values: Tensor::stack(&pixel_values, 0)?,
            pixel_attention_mask: None,
            image_sizes: None,
            num_img_tokens: None,
            aspect_ratio_ids: None,
            aspect_ratio_mask: None,
            num_tiles: None,
            image_grid_thw: None,
            video_grid_thw: None,
            rows: None,
            cols: None,
            pixel_values_list: None,
            tgt_sizes: None,
            image_sizes_all: Some(image_sizes),
            num_crops: Some(num_crops),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::image_ranges;

    const IMAGE: u32 = 7;

    #[test]
    fn adjacent_images_split_their_shared_run() {
        let ids = [1, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, 2, IMAGE, IMAGE];
        assert_eq!(
            image_ranges(&ids, IMAGE, &[2, 3, 2]).unwrap(),
            vec![(1, 2), (3, 3), (7, 2)]
        );
        assert!(image_ranges(&ids, IMAGE, &[2, 3]).is_err());
        assert!(image_ranges(&ids, IMAGE, &[2, 4, 2]).is_err());
    }
}

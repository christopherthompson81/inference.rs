use std::{any::Any, sync::Arc};

use anyhow::Result;
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use inference_tensor::{Device, IndexOp, Tensor};
use inference_vision::{
    ApplyTensorTransforms, ApplyTransforms, Normalize, TensorTransforms, ToTensor, Transforms,
};
use tokenizers::Tokenizer;

use crate::attention::AttentionMask;
use crate::device_map::DeviceMapper;
use crate::media_inputs::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    preprocessor_config::{PreProcessorConfig, ToFilter},
    processor::{
        InputProcessorOutput, InputsHost, InputsProcessorValidationError, MediaSequence,
        MultimodalInputsProcessor,
    },
    video::VideoInput,
};
use crate::paged_attention::PagedAttentionMeta;
use crate::qwen_vl_inputs::{QwenMropeConfig, QwenVlInputs, QwenVlSpec, QwenVlStep};
use crate::qwen2vl::inputs_processor::{
    PLACEHOLDER, VIDEO_PAD, VISION_END, VISION_START, replace_first_occurrence,
};
use crate::vision::multimodal_layout::MropePositionSource;

pub struct Qwen3VLImageProcessor {
    max_edge: Option<u32>,
}

struct VideoSizing {
    num_frames: usize,
    min_pixels: usize,
    max_pixels: usize,
}

impl Qwen3VLImageProcessor {
    const DEFAULT_PATCH_SIZE: usize = 14;
    const DEFAULT_MERGE_SIZE: usize = 2;
    const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;
    const DEFAULT_MIN_PIXELS: usize = 256 * 256;
    const DEFAULT_MAX_PIXELS: usize = 1536 * 1536;
    // HF Qwen3VLVideoProcessor class defaults; the budget covers t*h*w across the whole video.
    const DEFAULT_VIDEO_MIN_PIXELS: usize = 128 * 32 * 32;
    const DEFAULT_VIDEO_MAX_PIXELS: usize = 32 * 32 * 768;

    pub fn new(max_edge: Option<u32>) -> Self {
        Self { max_edge }
    }

    fn patch_size(config: &PreProcessorConfig) -> usize {
        config.patch_size.unwrap_or(Self::DEFAULT_PATCH_SIZE)
    }

    fn merge_size(config: &PreProcessorConfig) -> usize {
        config.merge_size.unwrap_or(Self::DEFAULT_MERGE_SIZE)
    }

    fn temporal_patch_size(config: &PreProcessorConfig) -> usize {
        config
            .temporal_patch_size
            .unwrap_or(Self::DEFAULT_TEMPORAL_PATCH_SIZE)
    }

    fn min_pixels(config: &PreProcessorConfig) -> usize {
        config.min_pixels.unwrap_or_else(|| {
            config
                .size
                .as_ref()
                .and_then(|s| s.get("shortest_edge").copied())
                .map(|v| v as usize)
                .unwrap_or(Self::DEFAULT_MIN_PIXELS)
        })
    }

    fn max_pixels(config: &PreProcessorConfig) -> usize {
        config.max_pixels.unwrap_or_else(|| {
            config
                .size
                .as_ref()
                .and_then(|s| s.get("longest_edge").copied())
                .map(|v| v as usize)
                .unwrap_or(Self::DEFAULT_MAX_PIXELS)
        })
    }
}
fn video_grid_temporal_patches(grid: Option<&Tensor>) -> Result<Vec<usize>> {
    Ok(grid
        .map(Tensor::to_vec2::<u32>)
        .transpose()?
        .unwrap_or_default()
        .iter()
        .map(|row| row.first().copied().unwrap_or(0) as usize)
        .collect())
}

// HF averages the first/last frame timestamp within each temporal patch.
fn grouped_video_timestamps(
    video: &VideoInput,
    grid_t: usize,
    temporal_patch_size: usize,
) -> Result<Vec<f64>> {
    let timestamps = video.timestamps_secs();
    let mut grouped = Vec::with_capacity(grid_t);
    for group in 0..grid_t {
        let first = group * temporal_patch_size;
        if first >= timestamps.len() {
            anyhow::bail!(
                "Qwen video grid_t {grid_t} exceeds {} sampled frames",
                timestamps.len()
            );
        }
        let last = (first + temporal_patch_size - 1).min(timestamps.len() - 1);
        grouped.push((timestamps[first] + timestamps[last]) / 2.0);
    }
    Ok(grouped)
}

// HF Qwen3VLProcessor.replace_video_token: each temporal patch becomes `<T.T seconds><|vision_start|>pads<|vision_end|>`,
// nested inside the chat template's outer vision markers.
fn expand_video_placeholders(
    text: &mut String,
    grid: Option<&Tensor>,
    videos: &[VideoInput],
    merge_length: usize,
    temporal_patch_size: usize,
) -> Result<()> {
    if merge_length == 0 || temporal_patch_size == 0 {
        anyhow::bail!("Qwen merge length and temporal patch size must be nonzero");
    }
    let placeholder_count = text.match_indices(VIDEO_PAD).count();
    let grid_rows = grid.map(|grid| grid.dim(0)).transpose()?.unwrap_or(0);
    if placeholder_count != videos.len() {
        return Err(InputsProcessorValidationError(format!(
            "Qwen video has {placeholder_count} placeholders but {} video inputs",
            videos.len()
        ))
        .into());
    }
    if grid_rows != videos.len() {
        anyhow::bail!(
            "Qwen video has {grid_rows} grid rows but {} video inputs",
            videos.len()
        );
    }
    let Some(grid) = grid else {
        return Ok(());
    };
    for (index, video) in videos.iter().enumerate() {
        let row = grid.i(index)?.to_vec1::<u32>()?;
        if row.len() != 3 {
            anyhow::bail!("Qwen video grid row must contain t, h, and w");
        }
        let grid_t = row[0] as usize;
        let frame_patches = row[1] as usize * row[2] as usize;
        if grid_t == 0 || frame_patches == 0 || !frame_patches.is_multiple_of(merge_length) {
            anyhow::bail!(
                "Qwen video grid produces {frame_patches} patches per frame for merge length {merge_length}"
            );
        }
        let frame_seqlen = frame_patches / merge_length;
        let timestamps = grouped_video_timestamps(video, grid_t, temporal_patch_size)?;
        let mut replacement = String::with_capacity(grid_t * (frame_seqlen + 2) * VIDEO_PAD.len());
        for timestamp in timestamps {
            replacement.push_str(&format!("<{timestamp:.1} seconds>"));
            replacement.push_str(VISION_START);
            for _ in 0..frame_seqlen {
                replacement.push_str(PLACEHOLDER);
            }
            replacement.push_str(VISION_END);
        }
        *text = replace_first_occurrence(text, VIDEO_PAD, &replacement);
    }
    *text = text.replace(PLACEHOLDER, VIDEO_PAD);
    Ok(())
}

fn qwen3_mrope_position_source(
    toks: &[u32],
    image_grid_thw: Option<&Tensor>,
    video_grid_thw: Option<&Tensor>,
    config: &QwenMropeConfig,
    device: &Device,
) -> Result<MropePositionSource> {
    let full_ids = Tensor::new(toks, device)?.unsqueeze(0)?;
    let (position_ids, deltas) = super::get_rope_index(
        &full_ids,
        image_grid_thw,
        video_grid_thw,
        &AttentionMask::None,
        config.spatial_merge_size,
        config.image_token_id,
        config.video_token_id,
        config.vision_start_token_id,
        config.vision_end_token_id,
    )?;
    Ok(MropePositionSource {
        position_ids,
        delta: deltas.flatten_all()?.to_vec1::<i64>()?[0],
    })
}

impl QwenVlSpec for Qwen3VLImageProcessor {
    fn name(&self) -> &'static str {
        "Qwen3VLImageProcessor"
    }

    fn preprocess_media(
        &self,
        images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
    ) -> inference_tensor::Result<PreprocessedImages> {
        self.preprocess(images, videos, config, device, (usize::MAX, usize::MAX))
    }

    fn media_resize_factors(&self, config: &PreProcessorConfig) -> (Option<usize>, Option<usize>) {
        let image_factor = if config.do_resize.is_none_or(|resize| resize) {
            Self::patch_size(config)
                .checked_mul(Self::merge_size(config))
                .filter(|factor| *factor > 0)
        } else {
            None
        };
        // The video processor upscales frames below the factor, so videos only need nonzero edges.
        let video_config = config.video.as_deref().unwrap_or(config);
        let video_factor = video_config
            .do_resize
            .is_none_or(|resize| resize)
            .then_some(1);
        (
            image_factor.filter(|_| self.max_edge.is_none()),
            video_factor,
        )
    }

    fn spatial_merge_size(&self, config: &PreProcessorConfig) -> Result<usize> {
        Ok(Self::merge_size(config))
    }

    fn expand_video_placeholders(
        &self,
        text: &mut String,
        grid: Option<&Tensor>,
        videos: &[VideoInput],
        config: &PreProcessorConfig,
    ) -> Result<()> {
        let video_config = config.video.as_deref().unwrap_or(config);
        expand_video_placeholders(
            text,
            grid,
            videos,
            Self::merge_size(video_config).pow(2),
            Self::temporal_patch_size(video_config),
        )
    }

    fn video_runs_per_item(&self, grid: Option<&Tensor>) -> Result<Vec<usize>> {
        video_grid_temporal_patches(grid)
    }

    fn packed_text_needs_prompt_mrope(&self) -> bool {
        true
    }

    fn mrope_position_source(
        &self,
        seq: &dyn MediaSequence,
        config: &QwenMropeConfig,
        device: &Device,
    ) -> Result<MropePositionSource> {
        qwen3_mrope_position_source(
            seq.prompt_position_source_toks(),
            seq.multimodal().rope_img_grid_thw.as_ref(),
            seq.multimodal().rope_vid_grid_thw.as_ref(),
            config,
            device,
        )
    }
}

impl MultimodalInputsProcessor for Qwen3VLImageProcessor {
    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut dyn MediaSequence],
        device: &Device,
        other_config: Option<Arc<dyn Any>>,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<()> {
        QwenVlInputs {
            spec: self,
            tokenizer,
            device,
            other_config,
        }
        .prepare(input_seqs, paged_attn_metadata)
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
        paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        let step = QwenVlStep {
            host,
            is_prompt,
            no_kv_cache,
            last_n_context_len,
            return_raw_logits,
            sliding_window,
            paged_attn_metadata,
            mapper,
        };
        QwenVlInputs {
            spec: self,
            tokenizer,
            device,
            other_config,
        }
        .process(input_seqs, step)
    }
}

impl Qwen3VLImageProcessor {
    fn smart_resize(
        &self,
        height: usize,
        width: usize,
        factor: usize,
        min_pixels: usize,
        max_pixels: usize,
    ) -> inference_tensor::Result<(usize, usize)> {
        if height < factor || width < factor {
            inference_tensor::bail!(
                "height:{} or width:{} must be larger than factor:{}",
                height,
                width,
                factor
            );
        } else if (height.max(width) as f64 / height.min(width) as f64) > 200.0 {
            inference_tensor::bail!(
                "absolute aspect ratio must be smaller than 200, got {:.2}",
                height.max(width) as f64 / height.min(width) as f64
            );
        }

        let mut h_bar = (height as f64 / factor as f64).round() as usize * factor;
        let mut w_bar = (width as f64 / factor as f64).round() as usize * factor;

        if h_bar * w_bar > max_pixels {
            let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
            h_bar = ((height as f64 / beta / factor as f64).floor() as usize) * factor;
            w_bar = ((width as f64 / beta / factor as f64).floor() as usize) * factor;
        } else if h_bar * w_bar < min_pixels {
            let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
            h_bar = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
            w_bar = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
        }

        Ok((h_bar, w_bar))
    }

    // HF Qwen3VLVideoProcessor.smart_resize: the pixel budget spans t_bar*h_bar*w_bar across all frames.
    fn smart_resize_video(
        &self,
        sizing: &VideoSizing,
        height: usize,
        width: usize,
        factor: usize,
        temporal_factor: usize,
    ) -> inference_tensor::Result<(usize, usize)> {
        let VideoSizing {
            num_frames,
            min_pixels,
            max_pixels,
        } = *sizing;
        let (mut height, mut width) = (height, width);
        if height < factor || width < factor {
            let scale = (factor as f64 / height as f64).max(factor as f64 / width as f64);
            height = (height as f64 * scale) as usize;
            width = (width as f64 * scale) as usize;
        }
        if (height.max(width) as f64 / height.min(width) as f64) > 200.0 {
            inference_tensor::bail!(
                "absolute aspect ratio must be smaller than 200, got {:.2}",
                height.max(width) as f64 / height.min(width) as f64
            );
        }

        let mut h_bar = (height as f64 / factor as f64).round() as usize * factor;
        let mut w_bar = (width as f64 / factor as f64).round() as usize * factor;
        let t_bar = (num_frames as f64 / temporal_factor as f64)
            .round()
            .max(1.0) as usize
            * temporal_factor;

        let volume = num_frames * height * width;
        if t_bar * h_bar * w_bar > max_pixels {
            let beta = (volume as f64 / max_pixels as f64).sqrt();
            h_bar =
                (((height as f64 / beta / factor as f64).floor() as usize) * factor).max(factor);
            w_bar = (((width as f64 / beta / factor as f64).floor() as usize) * factor).max(factor);
        } else if t_bar * h_bar * w_bar < min_pixels {
            let beta = (min_pixels as f64 / volume as f64).sqrt();
            h_bar = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
            w_bar = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
        }

        Ok((h_bar, w_bar))
    }

    // patches and t,h,w
    fn preprocess_inner(
        &self,
        images: Vec<DynamicImage>,
        config: &PreProcessorConfig,
        device: &Device,
        (mut height, mut width): (u32, u32),
        video: Option<&VideoSizing>,
    ) -> inference_tensor::Result<(Tensor, (u32, u32, u32))> {
        let mut processed_images = Vec::new();

        for mut image in images {
            image = image.resize_exact(
                width,
                height,
                config
                    .resampling
                    .map(|resample| Some(resample).to_filter())
                    .unwrap_or(Ok(FilterType::CatmullRom))?,
            );
            image = DynamicImage::ImageRgb8(image.to_rgb8());
            if config.do_resize.is_none() || config.do_resize.is_some_and(|x| x) {
                let (resized_height, resized_width) = match video {
                    Some(sizing) => self.smart_resize_video(
                        sizing,
                        height as usize,
                        width as usize,
                        Self::patch_size(config) * Self::merge_size(config),
                        Self::temporal_patch_size(config),
                    )?,
                    None => self.smart_resize(
                        height as usize,
                        width as usize,
                        Self::patch_size(config) * Self::merge_size(config),
                        Self::min_pixels(config),
                        Self::max_pixels(config),
                    )?,
                };
                height = resized_height as u32;
                width = resized_width as u32;
                image = image.resize_exact(
                    resized_width as u32,
                    resized_height as u32,
                    config
                        .resampling
                        .map(|resample| Some(resample).to_filter())
                        .unwrap_or(Ok(FilterType::CatmullRom))?,
                );
            }

            let to_tensor_rescale = Transforms {
                input: &ToTensor,
                inner_transforms: &[],
            };
            let image = image.apply(to_tensor_rescale, device)?;

            let transforms = TensorTransforms {
                inner_transforms: &[&Normalize {
                    mean: config.image_mean.unwrap_or(Self::DEFAULT_MEAN).to_vec(),
                    std: config.image_std.unwrap_or(Self::DEFAULT_STD).to_vec(),
                }],
            };
            let image = <Tensor as ApplyTensorTransforms>::apply(&image, transforms, device)?;

            processed_images.push(image);
        }

        let temporal_patch_size = Self::temporal_patch_size(config);
        let patch_size = Self::patch_size(config);
        let merge_size = Self::merge_size(config);

        // Validate divisors to prevent division by zero
        if temporal_patch_size == 0 {
            inference_tensor::bail!("temporal_patch_size cannot be zero");
        }
        if patch_size == 0 {
            inference_tensor::bail!("patch_size cannot be zero");
        }
        if merge_size == 0 {
            inference_tensor::bail!("merge_size cannot be zero");
        }
        let remainder = processed_images.len() % temporal_patch_size;
        if remainder != 0 {
            let pad = temporal_patch_size - remainder;
            let last = processed_images.last().unwrap().clone();
            for _ in 0..pad {
                processed_images.push(last.clone());
            }
        }

        let mut patches = Tensor::stack(&processed_images, 0)?;
        // Image
        if patches.dim(0)? == 1 {
            patches = patches.repeat((temporal_patch_size, 1, 1, 1))?;
        }
        let channel = patches.dim(1)?;
        let grid_t = patches.dim(0)? / temporal_patch_size;
        let grid_h = height as usize / patch_size;
        let grid_w = width as usize / patch_size;
        patches = patches.reshape(&[
            grid_t,
            temporal_patch_size,
            channel,
            grid_h / merge_size,
            merge_size,
            patch_size,
            grid_w / merge_size,
            merge_size,
            patch_size,
        ])?;
        patches = patches.permute([0, 3, 6, 4, 7, 2, 1, 5, 8])?;
        let flattened_patches = patches.reshape((
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        ))?;

        Ok((
            flattened_patches,
            (grid_t as u32, grid_h as u32, grid_w as u32),
        ))
    }
}

impl ImagePreProcessor for Qwen3VLImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

    fn preprocess(
        &self,
        mut images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_, _): (usize, usize),
    ) -> inference_tensor::Result<PreprocessedImages> {
        let mut pixel_values = Vec::new();
        let mut vision_grid_thw = Vec::new();

        if !images.is_empty() {
            if let Some(max_edge) = self.max_edge {
                images = inference_vision::pad_to_max_edge(&images, max_edge);
            }

            for image in images {
                let (w, h) = image.dimensions();
                let (patches, (t, gh, gw)) =
                    self.preprocess_inner(vec![image], config, device, (h, w), None)?;
                pixel_values.push(patches);
                vision_grid_thw.push(Tensor::new(&[t, gh, gw], &Device::Cpu)?);
            }
            let pixel_values = Tensor::cat(&pixel_values, 0)?;
            let vision_grid_thw = Tensor::stack(&vision_grid_thw, 0)?;
            return Ok(PreprocessedImages {
                pixel_values,
                pixel_attention_mask: None,
                image_sizes: None,
                num_img_tokens: None,
                aspect_ratio_ids: None,
                aspect_ratio_mask: None,
                num_tiles: None,
                image_grid_thw: Some(vision_grid_thw),
                video_grid_thw: None,
                rows: None,
                cols: None,
                pixel_values_list: None,
                tgt_sizes: None,
                image_sizes_all: None,
                num_crops: None,
            });
        }

        if !videos.is_empty() {
            let video_config = config.video.as_deref();
            let (min_pixels, max_pixels) = match video_config {
                Some(video_config) => (
                    Self::min_pixels(video_config),
                    Self::max_pixels(video_config),
                ),
                None => (
                    Self::DEFAULT_VIDEO_MIN_PIXELS,
                    Self::DEFAULT_VIDEO_MAX_PIXELS,
                ),
            };
            let effective_config = video_config.unwrap_or(config);
            for images in videos {
                let (w, h) = images[0].dimensions();
                let sizing = VideoSizing {
                    num_frames: images.len(),
                    min_pixels,
                    max_pixels,
                };
                let (patches, (t, gh, gw)) =
                    self.preprocess_inner(images, effective_config, device, (h, w), Some(&sizing))?;
                pixel_values.push(patches);
                vision_grid_thw.push(Tensor::new(&[t, gh, gw], &Device::Cpu)?);
            }
            let pixel_values = Tensor::cat(&pixel_values, 0)?;
            let vision_grid_thw = Tensor::stack(&vision_grid_thw, 0)?;
            return Ok(PreprocessedImages {
                pixel_values,
                pixel_attention_mask: None,
                image_sizes: None,
                num_img_tokens: None,
                aspect_ratio_ids: None,
                aspect_ratio_mask: None,
                num_tiles: None,
                image_grid_thw: None,
                video_grid_thw: Some(vision_grid_thw),
                rows: None,
                cols: None,
                pixel_values_list: None,
                tgt_sizes: None,
                image_sizes_all: None,
                num_crops: None,
            });
        }
        unreachable!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen_vl_inputs::{group_item_ranges, shift_item_runs};
    use crate::qwen2vl::inputs_processor::apply_mrope_position_delta;
    use crate::vision::multimodal_layout::gather_packed_mrope_positions;

    fn group_video_feature_ranges(
        ranges: &[(usize, usize)],
        grid: Option<&Tensor>,
    ) -> Result<Vec<(usize, usize)>> {
        group_item_ranges(ranges, &video_grid_temporal_patches(grid)?)
    }

    fn shift_video_pad_runs(
        runs: &mut Vec<(usize, usize)>,
        grid: Option<&Tensor>,
        prefix_len: usize,
    ) -> Result<(usize, usize)> {
        shift_item_runs(runs, &video_grid_temporal_patches(grid)?, prefix_len)
    }

    #[test]
    fn packed_text_only_mrope_restarts_each_logical_sequence() -> Result<()> {
        let config = QwenMropeConfig {
            spatial_merge_size: 2,
            image_token_id: 100,
            video_token_id: 101,
            vision_start_token_id: 102,
            vision_end_token_id: 103,
        };
        let sources = [
            qwen3_mrope_position_source(&[1, 2], None, None, &config, &Device::Cpu)?,
            qwen3_mrope_position_source(&[3, 4, 5], None, None, &config, &Device::Cpu)?,
        ];
        let positions = gather_packed_mrope_positions(&sources, &[0..2, 0..3], &Device::Cpu)?;

        assert_eq!(positions.dims(), &[3, 1, 5]);
        assert_eq!(
            positions.flatten_all()?.to_vec1::<i64>()?,
            vec![0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2]
        );
        Ok(())
    }

    #[test]
    fn decode_position_ends_apply_per_sequence_mrope_deltas() -> Result<()> {
        assert_eq!(apply_mrope_position_delta(20, -7)?, 13);
        assert_eq!(apply_mrope_position_delta(30, 2)?, 32);
        assert_eq!(apply_mrope_position_delta(100, 0)?, 100);
        assert_eq!(apply_mrope_position_delta(100, -48)?, 52);
        assert!(apply_mrope_position_delta(3, -4).is_err());
        Ok(())
    }

    #[test]
    fn cached_decode_positions_match_full_history_mrope() -> Result<()> {
        let config = QwenMropeConfig {
            spatial_merge_size: 2,
            image_token_id: 100,
            video_token_id: 101,
            vision_start_token_id: 102,
            vision_end_token_id: 103,
        };
        let image_grid = Tensor::new(&[[1u32, 4, 4]], &Device::Cpu)?;
        let cases = [
            (
                vec![10, 102, 100, 100, 100, 100, 103, 11],
                Some(&image_grid),
            ),
            (vec![20, 21, 22, 23, 24], None),
        ];
        let deltas = cases
            .iter()
            .map(|(prompt, grid)| {
                qwen3_mrope_position_source(prompt, *grid, None, &config, &Device::Cpu)
                    .map(|source| source.delta)
            })
            .collect::<Result<Vec<_>>>()?;

        assert_ne!(deltas[0], deltas[1]);
        for query_len in [1, 8] {
            for ((prompt, grid), delta) in cases.iter().zip(&deltas) {
                let mut full_history = prompt.clone();
                full_history.extend((0..query_len).map(|token| 30 + token as u32));
                let legacy =
                    qwen3_mrope_position_source(&full_history, *grid, None, &config, &Device::Cpu)?;
                let actual = legacy
                    .position_ids
                    .i((.., 0, prompt.len()..full_history.len()))?
                    .to_vec2::<i64>()?;
                let adjusted_end = apply_mrope_position_delta(full_history.len(), *delta)?;
                let expected_row = (adjusted_end - query_len..adjusted_end)
                    .map(|position| position as i64)
                    .collect::<Vec<_>>();

                assert_eq!(actual, vec![expected_row; 3]);
            }
        }
        Ok(())
    }

    fn test_video(frames: usize, fps: f64) -> VideoInput {
        VideoInput::from_frames(
            vec![DynamicImage::new_rgb8(1, 1); frames],
            fps,
            Some((0..frames).collect()),
        )
    }

    #[test]
    fn video_expansion_emits_timestamped_per_frame_spans() -> Result<()> {
        let mut text = format!("hi {}{}{} bye", VISION_START, VIDEO_PAD, VISION_END);
        let grid = Tensor::new(&[[2u32, 4, 4]], &Device::Cpu)?;
        expand_video_placeholders(&mut text, Some(&grid), &[test_video(4, 1.0)], 4, 2)?;

        let frame = format!("{}{}{}", VISION_START, VIDEO_PAD.repeat(4), VISION_END);
        let expected = format!(
            "hi {}<0.5 seconds>{frame}<2.5 seconds>{frame}{} bye",
            VISION_START, VISION_END
        );
        assert_eq!(text, expected);
        Ok(())
    }

    #[test]
    fn video_expansion_marks_placeholder_count_as_validation() -> Result<()> {
        let mut text = VIDEO_PAD.repeat(2);
        let grid = Tensor::new(&[[1u32, 2, 2]], &Device::Cpu)?;
        let error = expand_video_placeholders(&mut text, Some(&grid), &[test_video(2, 1.0)], 4, 2)
            .unwrap_err();

        assert!(error.is::<InputsProcessorValidationError>());
        Ok(())
    }

    #[test]
    fn video_feature_ranges_group_per_video() -> Result<()> {
        let grid = Tensor::new(&[[2u32, 2, 2], [1, 2, 2]], &Device::Cpu)?;
        let grouped = group_video_feature_ranges(&[(2, 5), (9, 5), (20, 5)], Some(&grid))?;
        assert_eq!(grouped, vec![(2, 12), (20, 5)]);
        Ok(())
    }

    #[test]
    fn video_pad_run_shift_caches_whole_videos() -> Result<()> {
        let grid = Tensor::new(&[[2u32, 2, 2], [1, 2, 2]], &Device::Cpu)?;
        let mut runs = vec![(0, 2), (4, 6), (10, 12)];
        assert_eq!(shift_video_pad_runs(&mut runs, Some(&grid), 6)?, (1, 1));
        assert_eq!(runs, vec![(4, 6)]);

        let mut split = vec![(0, 2), (4, 6), (10, 12)];
        assert!(shift_video_pad_runs(&mut split, Some(&grid), 5).is_err());
        Ok(())
    }

    #[test]
    fn video_smart_resize_budgets_whole_video() -> inference_tensor::Result<()> {
        let processor = Qwen3VLImageProcessor { max_edge: None };
        let sizing = |num_frames| VideoSizing {
            num_frames,
            min_pixels: 4096,
            max_pixels: 25165824,
        };
        // Within budget: dimensions snap to the factor without scaling.
        assert_eq!(
            processor.smart_resize_video(&sizing(16), 640, 480, 32, 2)?,
            (640, 480)
        );
        // Over budget: t*h*w drives the downscale even though each frame fits the image budget.
        assert_eq!(
            processor.smart_resize_video(&sizing(64), 704, 1280, 32, 2)?,
            (448, 832)
        );
        Ok(())
    }

    #[test]
    fn rope_index_splits_video_grids_per_frame() -> Result<()> {
        // One video row [2,2,2] must satisfy two per-frame vision spans.
        let toks: Vec<u32> = vec![10, 102, 55, 102, 101, 103, 56, 102, 101, 103, 103, 11];
        let input_ids = Tensor::new(toks, &Device::Cpu)?.unsqueeze(0)?;
        let video_grid = Tensor::new(&[[2u32, 2, 2]], &Device::Cpu)?;
        let (positions, deltas) = super::super::get_rope_index(
            &input_ids,
            None,
            Some(&video_grid),
            &AttentionMask::None,
            2,
            100,
            101,
            102,
            103,
        )?;
        assert_eq!(positions.dims(), &[3, 1, 12]);
        let expected: Vec<i64> = (0..12).collect();
        assert_eq!(positions.i((0, 0, ..))?.to_vec1::<i64>()?, expected);
        assert_eq!(deltas.flatten_all()?.to_vec1::<i64>()?, vec![0]);
        Ok(())
    }
}

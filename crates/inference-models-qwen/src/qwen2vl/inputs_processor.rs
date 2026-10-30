use std::{
    any::Any,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::Arc,
};

use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use inference_tensor::{Context, Device, IndexOp, Tensor};
use tokenizers::Tokenizer;

use crate::attention::AttentionMask;
use crate::device_map::DeviceMapper;
use crate::media_inputs::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    media::build_mm_features_from_ranges,
    preprocessor_config::PreProcessorConfig,
    processor::{
        InputProcessorOutput, InputsHost, InputsProcessorValidationError, MediaSequence,
        MultimodalInputsProcessor,
    },
    video::VideoInput,
};
use crate::paged_attention::{
    PagedAttentionMeta,
    block_hash::{MultiModalFeature, MultimodalKind},
};
use crate::qwen_vl_inputs::{
    PatchLayout, QwenMropeConfig, QwenVlSpec, frames_to_patches, preprocessed_media,
    qwen_vl_inputs_processor, smart_resize,
};
use crate::vision::multimodal_layout::MropePositionSource;

use super::Qwen2VLVisionSpecificArgs;

const MAX_MEDIA_ASPECT_RATIO: f64 = 200.0;

pub const VISION_START: &str = "<|vision_start|>";
pub const VISION_END: &str = "<|vision_end|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
pub const VIDEO_PAD: &str = "<|video_pad|>";
pub const PLACEHOLDER: &str = "<|placeholder|>";

pub struct Qwen2VLImageProcessor {
    max_edge: Option<u32>,
}

impl Qwen2VLImageProcessor {
    pub fn new(max_edge: Option<u32>) -> Self {
        Self { max_edge }
    }
}

pub(crate) fn replace_first_occurrence(text: &str, to_replace: &str, replacement: &str) -> String {
    if let Some(pos) = text.find(to_replace) {
        let mut result = text.to_string();
        result.replace_range(pos..pos + to_replace.len(), replacement);
        result
    } else {
        text.to_string()
    }
}

pub(crate) fn expand_media_placeholders(
    text: &mut String,
    pad: &str,
    placeholder: &str,
    grid: Option<&Tensor>,
    media_rows: usize,
    merge_length: usize,
    kind: MultimodalKind,
) -> Result<()> {
    if merge_length == 0 {
        anyhow::bail!("Qwen merge length must be nonzero");
    }
    let placeholder_count = text.match_indices(pad).count();
    let grid_rows = grid.map(|grid| grid.dim(0)).transpose()?.unwrap_or(0);
    if placeholder_count != media_rows {
        return Err(InputsProcessorValidationError(format!(
            "Qwen {kind:?} has {placeholder_count} placeholders but {media_rows} media inputs"
        ))
        .into());
    }
    if grid_rows != media_rows {
        anyhow::bail!("Qwen {kind:?} has {grid_rows} grid rows but {media_rows} media inputs");
    }
    let mut repetitions = Vec::with_capacity(grid_rows);
    if let Some(grid) = grid {
        for index in 0..grid_rows {
            let row = grid.i(index)?.to_vec1::<u32>()?;
            if row.len() != 3 {
                anyhow::bail!("Qwen {kind:?} grid row must contain t, h, and w");
            }
            let patch_count = row.iter().try_fold(1usize, |product, &value| {
                product
                    .checked_mul(value as usize)
                    .ok_or_else(|| anyhow::Error::msg("Qwen media grid size overflow"))
            })?;
            if patch_count == 0 || patch_count % merge_length != 0 {
                anyhow::bail!(
                    "Qwen {kind:?} grid produces {patch_count} patches for merge length {merge_length}"
                );
            }
            repetitions.push(patch_count / merge_length);
        }
    }
    for repetition in repetitions {
        *text = replace_first_occurrence(text, pad, &placeholder.repeat(repetition));
    }
    *text = text.replace(placeholder, pad);
    Ok(())
}

pub(crate) fn validate_qwen_media_dimensions(
    images: &[DynamicImage],
    resize_factor: Option<usize>,
) -> Result<()> {
    for image in images {
        let (width, height) = image.dimensions();
        if width == 0 || height == 0 {
            return Err(InputsProcessorValidationError(
                "Qwen media dimensions must be nonzero".to_string(),
            )
            .into());
        }
        let Some(factor) = resize_factor else {
            continue;
        };
        if (height as usize) < factor || (width as usize) < factor {
            return Err(InputsProcessorValidationError(format!(
                "Qwen media height {height} or width {width} must be at least {factor}"
            ))
            .into());
        }
        let aspect_ratio = f64::from(height.max(width)) / f64::from(height.min(width));
        if aspect_ratio > MAX_MEDIA_ASPECT_RATIO {
            return Err(InputsProcessorValidationError(format!(
                "Qwen media absolute aspect ratio must be smaller than {MAX_MEDIA_ASPECT_RATIO}, got {aspect_ratio:.2}"
            ))
            .into());
        }
    }
    Ok(())
}

pub(crate) fn validated_mm_features(
    ranges: &[(usize, usize)],
    hashes: &[u64],
    kind: MultimodalKind,
) -> Result<Vec<MultiModalFeature>> {
    if ranges.len() != hashes.len() {
        anyhow::bail!(
            "Qwen {kind:?} has {} placeholder ranges but {} hashes",
            ranges.len(),
            hashes.len()
        );
    }
    Ok(build_mm_features_from_ranges(ranges, hashes, kind))
}

pub(crate) fn find_sequences(nums: &[u32], needle: u32) -> Vec<(usize, usize)> {
    let mut sequences = Vec::new();
    let mut start = None;

    for (i, &num) in nums.iter().enumerate() {
        if num == needle {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            sequences.push((s, i));
            start = None;
        }
    }

    if let Some(s) = start {
        sequences.push((s, nums.len()));
    }

    sequences
}

pub(crate) fn video_hashes(seq: &dyn MediaSequence) -> Vec<u64> {
    let hashes = seq
        .clone_videos()
        .unwrap_or_default()
        .iter()
        .map(|video| {
            let mut hasher = DefaultHasher::new();
            video.frame_hashes().hash(&mut hasher);
            hasher.finish()
        })
        .collect::<Vec<_>>();
    if !seq.is_chunked_prefill_view() {
        return hashes;
    }
    seq.active_local_multimodal_item_range(MultimodalKind::Video, hashes.len())
        .and_then(|range| hashes.get(range))
        .unwrap_or_default()
        .to_vec()
}

fn grid_patch_count(grid: Option<&Tensor>) -> Result<usize> {
    Ok(grid
        .map(Tensor::to_vec2::<u32>)
        .transpose()?
        .unwrap_or_default()
        .iter()
        .map(|row| row.iter().map(|value| *value as usize).product::<usize>())
        .sum())
}

pub(crate) fn split_media_pixels(
    pixel_values: &Tensor,
    image_grid_thw: Option<&Tensor>,
    video_grid_thw: Option<&Tensor>,
) -> Result<(Option<Tensor>, Option<Tensor>)> {
    let image_patches = grid_patch_count(image_grid_thw)?;
    let video_patches = grid_patch_count(video_grid_thw)?;
    if pixel_values.dim(0)? != image_patches + video_patches {
        anyhow::bail!(
            "Qwen media pixel rows {} do not match image/video grids {}",
            pixel_values.dim(0)?,
            image_patches + video_patches
        );
    }
    let images = (image_patches != 0)
        .then(|| pixel_values.narrow(0, 0, image_patches))
        .transpose()?;
    let videos = (video_patches != 0)
        .then(|| pixel_values.narrow(0, image_patches, video_patches))
        .transpose()?;
    Ok((images, videos))
}

pub(crate) fn select_media_view(
    seq: &dyn MediaSequence,
    kind: MultimodalKind,
    pixel_values: Option<Tensor>,
    grid_thw: Option<Tensor>,
) -> Result<(Option<Tensor>, Option<Tensor>, usize)> {
    let Some(grid_thw) = grid_thw else {
        if pixel_values.is_some() {
            anyhow::bail!("Qwen media pixels are missing grid metadata");
        }
        return Ok((None, None, 0));
    };
    let item_count = grid_thw.dim(0)?;
    let range = if seq.is_chunked_prefill_view() {
        seq.active_local_multimodal_item_range(kind, item_count)
            .unwrap_or(0..0)
    } else {
        0..item_count
    };
    if range.end > item_count {
        anyhow::bail!(
            "Qwen {:?} media window {:?} exceeds {} grid rows",
            kind,
            range,
            item_count
        );
    }
    let grid_data = grid_thw.to_vec2::<u32>()?;
    let patch_start = grid_data[..range.start]
        .iter()
        .map(|row| row.iter().map(|value| *value as usize).product::<usize>())
        .sum::<usize>();
    let patch_count = grid_data[range.clone()]
        .iter()
        .map(|row| row.iter().map(|value| *value as usize).product::<usize>())
        .sum::<usize>();
    let pixel_values = match (pixel_values, patch_count) {
        (Some(pixel_values), 0) => {
            if pixel_values.dim(0)? < patch_start {
                anyhow::bail!("Qwen media pixel rows do not cover the selected window");
            }
            None
        }
        (Some(pixel_values), patch_count) => {
            Some(pixel_values.narrow(0, patch_start, patch_count)?)
        }
        (None, 0) => None,
        (None, _) => anyhow::bail!("Qwen media grid is missing pixel rows"),
    };
    let selected_count = range.len();
    let grid_thw = (selected_count != 0)
        .then(|| grid_thw.narrow(0, range.start, selected_count))
        .transpose()?;
    Ok((pixel_values, grid_thw, selected_count))
}

pub(crate) fn shift_media_spans(
    spans: &mut Vec<(usize, usize)>,
    prefix_len: usize,
) -> Result<usize> {
    if prefix_len == 0 {
        return Ok(0);
    }
    let mut cached = 0usize;
    for &(start, end) in spans.iter() {
        if end <= prefix_len {
            cached += 1;
        } else if start < prefix_len {
            anyhow::bail!("Qwen prefix cache splits a multimodal item");
        }
    }
    spans.retain(|(_, end)| *end > prefix_len);
    for (start, end) in spans {
        *start -= prefix_len;
        *end -= prefix_len;
    }
    Ok(cached)
}

pub(crate) fn media_data_cached_offset(seq: &dyn MediaSequence, cached_items: usize) -> usize {
    if seq.is_chunked_prefill_view() {
        0
    } else {
        cached_items
    }
}

pub(crate) fn select_media_batch(
    mut pixel_values: Option<Tensor>,
    mut grid_thw: Option<Tensor>,
    item_counts: &[usize],
    cached_items: &[usize],
    current_items: &[usize],
) -> Result<(Option<Tensor>, Option<Tensor>)> {
    if item_counts.len() != cached_items.len() || item_counts.len() != current_items.len() {
        anyhow::bail!("Qwen per-sequence media metadata length mismatch");
    }
    let Some(grid) = grid_thw.as_ref() else {
        if item_counts.iter().sum::<usize>() != 0 || pixel_values.is_some() {
            anyhow::bail!("Qwen media selection is missing grid metadata");
        }
        return Ok((None, None));
    };
    if grid.dim(0)? != item_counts.iter().sum::<usize>() {
        anyhow::bail!("Qwen media grid rows do not match per-sequence item counts");
    }
    let grid_data = grid.to_vec2::<u32>()?;
    let mut selected_grids = Vec::new();
    let mut selected_pixels = Vec::new();
    let mut grid_offset = 0usize;
    let mut pixel_offset = 0usize;
    for ((&total, &cached), &current) in item_counts.iter().zip(cached_items).zip(current_items) {
        let end = cached
            .checked_add(current)
            .ok_or_else(|| anyhow::Error::msg("Qwen media item range overflow"))?;
        if end > total {
            anyhow::bail!(
                "Qwen media view requests items {cached}..{end} from a sequence with {total}"
            );
        }
        let start_row = grid_offset + cached;
        let patch_start = pixel_offset
            + grid_data[grid_offset..start_row]
                .iter()
                .map(|row| row.iter().map(|value| *value as usize).product::<usize>())
                .sum::<usize>();
        let patch_count = grid_data[start_row..start_row + current]
            .iter()
            .map(|row| row.iter().map(|value| *value as usize).product::<usize>())
            .sum::<usize>();
        if current != 0 {
            selected_grids.push(grid.narrow(0, start_row, current)?);
        }
        if patch_count != 0 {
            let pixels = pixel_values
                .as_ref()
                .ok_or_else(|| anyhow::Error::msg("Qwen media grid is missing pixel rows"))?;
            selected_pixels.push(pixels.narrow(0, patch_start, patch_count)?);
        }
        pixel_offset += grid_data[grid_offset..grid_offset + total]
            .iter()
            .map(|row| row.iter().map(|value| *value as usize).product::<usize>())
            .sum::<usize>();
        grid_offset += total;
    }
    grid_thw = (!selected_grids.is_empty())
        .then(|| Tensor::cat(&selected_grids, 0))
        .transpose()?;
    pixel_values = (!selected_pixels.is_empty())
        .then(|| Tensor::cat(&selected_pixels, 0))
        .transpose()?;
    Ok((pixel_values, grid_thw))
}

fn completed_media_grid(
    seq: &dyn MediaSequence,
    kind: MultimodalKind,
    grid: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    let Some(grid) = grid else {
        return Ok(None);
    };
    if !seq.is_chunked_prefill_view() {
        return Ok(Some(grid.clone()));
    }
    let token_count = seq.prompt_position_source_toks().len();
    let item_count = seq
        .mm_features()
        .iter()
        .filter(|feature| feature.kind == kind && feature.end() <= token_count)
        .map(|feature| feature.item_range.len())
        .sum::<usize>();
    if item_count > grid.dim(0)? {
        anyhow::bail!("Qwen completed media items exceed the stored MRoPE grid");
    }
    (item_count != 0)
        .then(|| grid.narrow(0, 0, item_count))
        .transpose()
        .map_err(Into::into)
}

pub(crate) fn apply_mrope_position_deltas(
    position_ids: Vec<usize>,
    input_seqs: &[&mut dyn MediaSequence],
) -> Result<Vec<usize>> {
    if position_ids.len() != input_seqs.len() {
        anyhow::bail!(
            "Qwen MRoPE position count {} does not match sequence count {}",
            position_ids.len(),
            input_seqs.len()
        );
    }
    position_ids
        .into_iter()
        .zip(input_seqs)
        .map(|(position, seq)| {
            apply_mrope_position_delta(position, seq.multimodal().mrope_position_delta.unwrap_or(0))
        })
        .collect()
}

pub(crate) fn apply_mrope_position_delta(position: usize, delta: i64) -> Result<usize> {
    let position = i64::try_from(position)?;
    let position = position
        .checked_add(delta)
        .ok_or_else(|| anyhow::anyhow!("Qwen MRoPE position overflow"))?;
    usize::try_from(position).map_err(anyhow::Error::from)
}

pub(crate) fn qwen2_decode_args(
    input_ids: &Tensor,
    seqlens: Vec<usize>,
) -> Qwen2VLVisionSpecificArgs {
    Qwen2VLVisionSpecificArgs {
        input_ids_full: input_ids.clone(),
        pixel_values_videos: None,
        image_grid_thw: None,
        video_grid_thw: None,
        rope_img_grid_thw: None,
        rope_vid_grid_thw: None,
        seqlens,
        continuous_img_pad: Vec::new(),
        continuous_vid_pad: Vec::new(),
        image_hashes: Vec::new(),
        video_hashes: Vec::new(),
        packed_layout: None,
        prompt_position_ids: None,
    }
}

impl QwenVlSpec for Qwen2VLImageProcessor {
    fn name(&self) -> &'static str {
        "Qwen2VLImageProcessor"
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
        let factor = if config.do_resize.is_none_or(|resize| resize) {
            config
                .patch_size
                .zip(config.merge_size)
                .and_then(|(patch_size, merge_size)| patch_size.checked_mul(merge_size))
                .filter(|factor| *factor > 0)
        } else {
            None
        };
        (factor.filter(|_| self.max_edge.is_none()), factor)
    }

    fn spatial_merge_size(&self, config: &PreProcessorConfig) -> Result<usize> {
        Ok(config.merge_size.context("Qwen requires merge_size")?)
    }

    fn expand_video_placeholders(
        &self,
        text: &mut String,
        grid: Option<&Tensor>,
        videos: &[VideoInput],
        config: &PreProcessorConfig,
    ) -> Result<()> {
        expand_media_placeholders(
            text,
            VIDEO_PAD,
            PLACEHOLDER,
            grid,
            videos.len(),
            self.spatial_merge_size(config)?.pow(2),
            MultimodalKind::Video,
        )
    }

    fn video_runs_per_item(&self, grid: Option<&Tensor>) -> Result<Vec<usize>> {
        Ok(vec![1; grid.map_or(Ok(0), |grid| grid.dim(0))?])
    }

    fn packed_text_needs_prompt_mrope(&self) -> bool {
        false
    }

    fn mrope_position_source(
        &self,
        seq: &dyn MediaSequence,
        config: &QwenMropeConfig,
        device: &Device,
    ) -> Result<MropePositionSource> {
        let image_grid = completed_media_grid(
            seq,
            MultimodalKind::Image,
            seq.multimodal().rope_img_grid_thw.as_ref(),
        )?;
        let video_grid = completed_media_grid(
            seq,
            MultimodalKind::Video,
            seq.multimodal().rope_vid_grid_thw.as_ref(),
        )?;
        let full_ids = Tensor::new(seq.prompt_position_source_toks(), device)?.unsqueeze(0)?;
        let (position_ids, deltas) = super::compute_rope_index(
            &full_ids,
            image_grid.as_ref(),
            video_grid.as_ref(),
            &AttentionMask::None,
            config.spatial_merge_size,
            config.image_token_id,
            config.video_token_id,
        )?;
        Ok(MropePositionSource {
            position_ids,
            delta: deltas.flatten_all()?.to_vec1::<i64>()?[0],
        })
    }
}

qwen_vl_inputs_processor!(Qwen2VLImageProcessor);

impl Qwen2VLImageProcessor {
    // patches and t,h,w
    fn preprocess_inner(
        &self,
        images: Vec<DynamicImage>,
        config: &PreProcessorConfig,
        device: &Device,
        size: (u32, u32),
    ) -> inference_tensor::Result<(Tensor, (u32, u32, u32))> {
        let patch_size = config.patch_size.context("Require `patch_size`.")?;
        let merge_size = config.merge_size.context("Require `merge_size`")?;
        let (min_pixels, max_pixels) = (
            config.min_pixels.context("Require `min_pixels`")?,
            config.max_pixels.context("Require `max_pixels`")?,
        );
        let layout = PatchLayout {
            patch_size,
            merge_size,
            temporal_patch_size: config
                .temporal_patch_size
                .context("Require `temporal_patch_size")?,
            mean: Self::DEFAULT_MEAN,
            std: Self::DEFAULT_STD,
        };
        let resize = |height, width| {
            smart_resize(
                height,
                width,
                patch_size * merge_size,
                min_pixels,
                max_pixels,
            )
        };
        frames_to_patches(images, config, &layout, device, size, &resize)
    }
}

impl ImagePreProcessor for Qwen2VLImageProcessor {
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
        if !images.is_empty() {
            if let Some(max_edge) = self.max_edge {
                images = inference_vision::pad_to_max_edge(&images, max_edge);
            }
            let mut patches = Vec::with_capacity(images.len());
            for image in images {
                let (w, h) = image.dimensions();
                patches.push(self.preprocess_inner(vec![image], config, device, (h, w))?);
            }
            return preprocessed_media(patches, false);
        }

        if !videos.is_empty() {
            let mut patches = Vec::with_capacity(videos.len());
            for images in videos {
                let (w, h) = images[0].dimensions();
                patches.push(self.preprocess_inner(images, config, device, (h, w))?);
            }
            return preprocessed_media(patches, true);
        }
        unreachable!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_tensor::DType;

    #[test]
    fn decode_position_ends_apply_mrope_delta() -> Result<()> {
        assert_eq!(apply_mrope_position_delta(20, -7)?, 13);
        assert_eq!(apply_mrope_position_delta(30, 2)?, 32);
        assert_eq!(apply_mrope_position_delta(100, 0)?, 100);
        assert!(apply_mrope_position_delta(3, -4).is_err());
        Ok(())
    }

    #[test]
    fn decode_args_only_retain_current_tokens() -> Result<()> {
        let input_ids = Tensor::new(&[[7u32], [9]], &Device::Cpu)?;
        let args = qwen2_decode_args(&input_ids, vec![100, 200]);

        assert_eq!(
            args.input_ids_full.to_vec2::<u32>()?,
            vec![vec![7], vec![9]]
        );
        assert_eq!(args.seqlens, vec![100, 200]);
        assert!(args.pixel_values_videos.is_none());
        assert!(args.image_grid_thw.is_none());
        assert!(args.video_grid_thw.is_none());
        assert!(args.rope_img_grid_thw.is_none());
        assert!(args.rope_vid_grid_thw.is_none());
        assert!(args.continuous_img_pad.is_empty());
        assert!(args.continuous_vid_pad.is_empty());
        assert!(args.image_hashes.is_empty());
        assert!(args.video_hashes.is_empty());
        assert!(args.packed_layout.is_none());
        assert!(args.prompt_position_ids.is_none());
        Ok(())
    }

    #[test]
    fn media_expansion_requires_exact_placeholder_grid_and_media_counts() -> Result<()> {
        let grid = Tensor::new(&[[1u32, 2, 2], [1, 2, 4]], &Device::Cpu)?;
        let mut text = format!("a{}b{}c", IMAGE_PAD, IMAGE_PAD);
        expand_media_placeholders(
            &mut text,
            IMAGE_PAD,
            PLACEHOLDER,
            Some(&grid),
            2,
            4,
            MultimodalKind::Image,
        )?;
        assert_eq!(text.match_indices(IMAGE_PAD).count(), 3);

        let one_grid = Tensor::new(&[[1u32, 2, 2]], &Device::Cpu)?;
        let mut excess = format!("{}{}", IMAGE_PAD, IMAGE_PAD);
        let original = excess.clone();
        assert!(
            expand_media_placeholders(
                &mut excess,
                IMAGE_PAD,
                PLACEHOLDER,
                Some(&one_grid),
                1,
                4,
                MultimodalKind::Image,
            )
            .unwrap_err()
            .is::<InputsProcessorValidationError>()
        );
        assert_eq!(excess, original);

        let mut missing = IMAGE_PAD.to_string();
        assert!(
            expand_media_placeholders(
                &mut missing,
                IMAGE_PAD,
                PLACEHOLDER,
                Some(&grid),
                2,
                4,
                MultimodalKind::Image,
            )
            .unwrap_err()
            .is::<InputsProcessorValidationError>()
        );

        let mut grid_mismatch = format!("{}{}", IMAGE_PAD, IMAGE_PAD);
        assert!(
            !expand_media_placeholders(
                &mut grid_mismatch,
                IMAGE_PAD,
                PLACEHOLDER,
                Some(&one_grid),
                2,
                4,
                MultimodalKind::Image,
            )
            .unwrap_err()
            .is::<InputsProcessorValidationError>()
        );
        Ok(())
    }

    #[test]
    fn malformed_grid_and_range_hash_cardinality_fail_closed() -> Result<()> {
        let malformed_grid = Tensor::new(&[[1u32, 2]], &Device::Cpu)?;
        let mut text = VIDEO_PAD.to_string();
        assert!(
            expand_media_placeholders(
                &mut text,
                VIDEO_PAD,
                PLACEHOLDER,
                Some(&malformed_grid),
                1,
                4,
                MultimodalKind::Video,
            )
            .is_err()
        );
        assert!(validated_mm_features(&[(1, 2), (5, 2)], &[11], MultimodalKind::Image).is_err());
        assert!(validated_mm_features(&[(1, 2)], &[11, 12], MultimodalKind::Image).is_err());
        Ok(())
    }

    #[test]
    fn invalid_media_dimensions_are_validation_errors() {
        let too_small = DynamicImage::new_rgb8(14, 28);
        let error = validate_qwen_media_dimensions(&[too_small], Some(28)).unwrap_err();
        assert!(error.is::<InputsProcessorValidationError>());

        let extreme = DynamicImage::new_rgb8(201, 1);
        let error = validate_qwen_media_dimensions(&[extreme], Some(1)).unwrap_err();
        assert!(error.is::<InputsProcessorValidationError>());

        let frames = [
            DynamicImage::new_rgb8(28, 28),
            DynamicImage::new_rgb8(14, 28),
        ];
        let error = validate_qwen_media_dimensions(&frames, Some(28)).unwrap_err();
        assert!(error.is::<InputsProcessorValidationError>());
    }

    #[test]
    fn split_pixels_keeps_image_video_order() -> Result<()> {
        let pixels = Tensor::arange(0f32, 10f32, &Device::Cpu)?.reshape((5, 2))?;
        let image_grid = Tensor::new(&[[1u32, 1, 2]], &Device::Cpu)?;
        let video_grid = Tensor::new(&[[1u32, 1, 3]], &Device::Cpu)?;
        let (images, videos) = split_media_pixels(&pixels, Some(&image_grid), Some(&video_grid))?;

        assert_eq!(
            images.unwrap().to_vec2::<f32>()?,
            vec![vec![0., 1.], vec![2., 3.]]
        );
        assert_eq!(
            videos.unwrap().to_vec2::<f32>()?,
            vec![vec![4., 5.], vec![6., 7.], vec![8., 9.]]
        );
        Ok(())
    }

    #[test]
    fn media_batch_selection_handles_heterogeneous_sequence_offsets() -> Result<()> {
        let pixels = Tensor::arange(0f32, 15f32, &Device::Cpu)?.reshape((15, 1))?;
        let grid = Tensor::new(
            &[[1u32, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4], [1, 1, 5]],
            &Device::Cpu,
        )?;
        let (pixels, grid) =
            select_media_batch(Some(pixels), Some(grid), &[3, 2], &[1, 1], &[1, 1])?;

        assert_eq!(
            grid.unwrap().to_vec2::<u32>()?,
            vec![vec![1, 1, 2], vec![1, 1, 5]]
        );
        assert_eq!(
            pixels.unwrap().flatten_all()?.to_vec1::<f32>()?,
            vec![1., 2., 10., 11., 12., 13., 14.]
        );
        Ok(())
    }

    #[test]
    fn media_span_shift_rejects_split_items() -> Result<()> {
        let mut split = vec![(2, 5)];
        assert!(shift_media_spans(&mut split, 3).is_err());

        let mut spans = vec![(0, 2), (4, 7)];
        assert_eq!(shift_media_spans(&mut spans, 2)?, 1);
        assert_eq!(spans, vec![(2, 5)]);
        Ok(())
    }

    #[test]
    fn split_pixels_rejects_grid_mismatch() -> Result<()> {
        let pixels = Tensor::zeros((4, 2), DType::F32, &Device::Cpu)?;
        let image_grid = Tensor::new(&[[1u32, 1, 3]], &Device::Cpu)?;

        assert!(split_media_pixels(&pixels, Some(&image_grid), None).is_err());
        Ok(())
    }
}

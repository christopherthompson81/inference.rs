//! The media a sequence carries and the per-sequence multimodal state the input processors keep on it.

use std::{
    hash::{DefaultHasher, Hash, Hasher},
    ops::Range,
};

use candle_core::Tensor;
use inference_audio::AudioInput;

use super::video::VideoInput;
use crate::paged_attention::block_hash::{
    MultiModalFeature, MultimodalAttentionPolicy, MultimodalKind,
};

pub struct SequenceImages {
    images: Vec<image::DynamicImage>,
    hashes: Vec<u64>,
}

#[derive(Clone)]
pub struct SequenceAudios {
    audios: Vec<AudioInput>,
    hashes: Vec<u64>,
}

impl SequenceAudios {
    pub fn new(input_audios: Vec<AudioInput>) -> Self {
        let hashes = input_audios.iter().map(|a| {
            let mut hasher = DefaultHasher::new();
            for s in &a.samples {
                s.to_bits().hash(&mut hasher);
            }
            a.sample_rate.hash(&mut hasher);
            hasher.finish()
        });
        Self {
            hashes: hashes.collect(),
            audios: input_audios,
        }
    }

    fn clone_audios(&self) -> Vec<AudioInput> {
        self.audios.clone()
    }

    fn clone_audios_range(&self, range: Range<usize>) -> Vec<AudioInput> {
        self.audios[range].to_vec()
    }

    fn audios(&self) -> &[AudioInput] {
        &self.audios
    }

    fn audios_mut(&mut self) -> &mut Vec<AudioInput> {
        &mut self.audios
    }

    pub fn hashes(&self) -> &[u64] {
        &self.hashes
    }

    fn keep_num_audios(&mut self, audios_to_keep: usize) {
        if self.audios.len() > audios_to_keep {
            let start = self.audios.len() - audios_to_keep;
            self.audios = self.audios[start..].to_vec();
            // Do not do this because we need all the hashes later in the prefix cacher.
            // self.hashes = self.hashes[start..].to_vec();
        }
    }
}

impl SequenceImages {
    pub fn new(input_images: Vec<image::DynamicImage>) -> Self {
        let hashes = input_images.iter().map(|image| {
            let mut hasher = DefaultHasher::new();
            image.width().hash(&mut hasher);
            image.height().hash(&mut hasher);
            image.color().hash(&mut hasher);
            image.as_bytes().hash(&mut hasher);
            hasher.finish()
        });
        Self {
            hashes: hashes.collect(),
            images: input_images,
        }
    }

    fn clone_images(&self) -> Vec<image::DynamicImage> {
        self.images.clone()
    }

    fn clone_images_range(&self, range: Range<usize>) -> Vec<image::DynamicImage> {
        self.images[range].to_vec()
    }

    fn images(&self) -> &[image::DynamicImage] {
        &self.images
    }

    fn images_mut(&mut self) -> &mut Vec<image::DynamicImage> {
        &mut self.images
    }

    pub fn hashes(&self) -> &[u64] {
        &self.hashes
    }

    fn keep_num_images(&mut self, images_to_keep: usize) {
        if self.images.len() > images_to_keep {
            let start = self.images.len() - images_to_keep;
            self.images = self.images[start..].to_vec();
            // Do not do this because we need all the hashes later in the prefix cacher.
            // self.hashes = self.hashes[start..].to_vec();
        }
    }
}

pub struct SequenceVideos {
    videos: Vec<VideoInput>,
    hashes: Vec<u64>,
}

impl SequenceVideos {
    pub fn new(input_videos: Vec<VideoInput>) -> Self {
        // Store per-frame hashes (not per-video) so they align 1:1 with
        // per-frame token ranges from `find_image_placeholder_ranges`.
        let hashes = input_videos.iter().flat_map(|v| v.frame_hashes()).collect();
        Self {
            videos: input_videos,
            hashes,
        }
    }

    fn clone_videos(&self) -> Vec<VideoInput> {
        self.videos.clone()
    }

    fn clone_frames_range(&self, range: Range<usize>) -> Vec<VideoInput> {
        let mut videos = Vec::new();
        let mut cursor = 0usize;
        for video in &self.videos {
            let next = cursor + video.frames.len();
            if range.start < next && range.end > cursor {
                let start = range.start.saturating_sub(cursor).min(video.frames.len());
                let end = range.end.saturating_sub(cursor).min(video.frames.len());
                if start < end {
                    videos.push(VideoInput {
                        frames: video.frames[start..end].to_vec(),
                        fps: video.fps,
                        total_num_frames: video.total_num_frames,
                        sampled_indices: video.sampled_indices[start..end].to_vec(),
                    });
                }
            }
            cursor = next;
            if cursor >= range.end {
                break;
            }
        }
        videos
    }

    fn videos(&self) -> &[VideoInput] {
        &self.videos
    }

    fn videos_mut(&mut self) -> &mut Vec<VideoInput> {
        &mut self.videos
    }

    pub fn hashes(&self) -> &[u64] {
        &self.hashes
    }

    fn keep_num_videos(&mut self, videos_to_keep: usize) {
        if self.videos.len() > videos_to_keep {
            let start = self.videos.len() - videos_to_keep;
            self.videos = self.videos[start..].to_vec();
        }
    }

    fn keep_num_video_frames(&mut self, video_frames_to_keep: usize) {
        let frame_count = self.videos.iter().map(|video| video.frames.len()).sum();
        if frame_count > video_frames_to_keep {
            self.videos = self.clone_frames_range(frame_count - video_frames_to_keep..frame_count);
        }
    }
}

// Holds all multimodal (vision/diffusion) data for a Sequence.
pub struct MultimodalData {
    pub input_images: Option<SequenceImages>,
    pub input_audios: Option<SequenceAudios>,
    pub input_videos: Option<SequenceVideos>,
    pub cached_pixel_values: Option<Tensor>,
    pub cached_pixel_attention_mask: Option<Tensor>,
    pub cached_spatial_shapes: Option<Tensor>,
    pub cached_num_crops: Option<Vec<usize>>,
    pub cached_img_thw: Option<Tensor>,
    pub cached_vid_thw: Option<Tensor>,
    /// Complete image grid metadata, including prefix-cached images.
    pub rope_img_grid_thw: Option<Tensor>,
    /// Complete video grid metadata, including prefix-cached videos.
    pub rope_vid_grid_thw: Option<Tensor>,
    /// Fixed offset between token indices and post-media MRoPE positions.
    pub mrope_position_delta: Option<i64>,
    pub has_changed_prompt: bool,
    /// Per-item multimodal feature positions for prefix caching block hashing.
    /// Each entry records which token range a multimodal item (image/audio) occupies,
    /// so that only blocks overlapping with that item include its content hash.
    /// Set once during the first `process_inputs()` call and never modified thereafter.
    mm_features: Vec<MultiModalFeature>,
}

impl MultimodalData {
    pub fn new(
        input_images: Option<Vec<image::DynamicImage>>,
        input_audios: Option<Vec<AudioInput>>,
        input_videos: Option<Vec<VideoInput>>,
    ) -> Self {
        MultimodalData {
            input_images: input_images.map(SequenceImages::new),
            input_audios: input_audios.map(SequenceAudios::new),
            input_videos: input_videos.map(SequenceVideos::new),
            cached_pixel_values: None,
            cached_pixel_attention_mask: None,
            cached_spatial_shapes: None,
            cached_num_crops: None,
            cached_img_thw: None,
            cached_vid_thw: None,
            rope_img_grid_thw: None,
            rope_vid_grid_thw: None,
            mrope_position_delta: None,
            has_changed_prompt: false,
            mm_features: Vec::new(),
        }
    }

    pub fn take_images(&mut self) -> Option<Vec<image::DynamicImage>> {
        if self.has_changed_prompt {
            if let Some(input_images) = self.input_images.as_mut() {
                let mut images = Vec::new();
                std::mem::swap(&mut images, input_images.images_mut());
                Some(images)
            } else {
                None
            }
        } else {
            self.input_images.as_ref().map(|imgs| imgs.clone_images())
        }
    }

    pub fn clone_images(&self) -> Option<Vec<image::DynamicImage>> {
        self.input_images.as_ref().map(|imgs| imgs.clone_images())
    }

    pub fn clone_images_range(&self, range: Range<usize>) -> Option<Vec<image::DynamicImage>> {
        self.input_images
            .as_ref()
            .map(|imgs| imgs.clone_images_range(range))
    }

    pub fn images(&self) -> Option<&[image::DynamicImage]> {
        self.input_images.as_ref().map(|imgs| imgs.images())
    }

    pub fn image_hashes(&self) -> Option<&[u64]> {
        self.input_images.as_ref().map(|imgs| imgs.hashes())
    }

    pub fn has_images(&self) -> bool {
        self.input_images
            .as_ref()
            .is_some_and(|imgs| !imgs.images().is_empty())
    }

    pub fn take_audios(&mut self) -> Option<Vec<AudioInput>> {
        if self.has_changed_prompt {
            if let Some(input_audios) = self.input_audios.as_mut() {
                let mut audios = Vec::new();
                std::mem::swap(&mut audios, input_audios.audios_mut());
                Some(audios)
            } else {
                None
            }
        } else {
            self.input_audios.as_ref().map(|imgs| imgs.clone_audios())
        }
    }

    pub fn clone_audios(&self) -> Option<Vec<AudioInput>> {
        self.input_audios.as_ref().map(|a| a.clone_audios())
    }

    pub fn clone_audios_range(&self, range: Range<usize>) -> Option<Vec<AudioInput>> {
        self.input_audios
            .as_ref()
            .map(|a| a.clone_audios_range(range))
    }

    pub fn audios(&self) -> Option<&[AudioInput]> {
        self.input_audios.as_ref().map(|a| a.audios())
    }

    pub fn audio_hashes(&self) -> Option<&[u64]> {
        self.input_audios.as_ref().map(|a| a.hashes())
    }

    pub fn has_audios(&self) -> bool {
        self.input_audios
            .as_ref()
            .is_some_and(|a| !a.audios().is_empty())
    }

    pub fn keep_num_audios(&mut self, audios_to_keep: usize) {
        if let Some(auds) = self.input_audios.as_mut() {
            auds.keep_num_audios(audios_to_keep)
        }
    }

    pub fn take_videos(&mut self) -> Option<Vec<VideoInput>> {
        if self.has_changed_prompt {
            if let Some(input_videos) = self.input_videos.as_mut() {
                let mut videos = Vec::new();
                std::mem::swap(&mut videos, input_videos.videos_mut());
                Some(videos)
            } else {
                None
            }
        } else {
            self.input_videos.as_ref().map(|v| v.clone_videos())
        }
    }

    pub fn clone_videos(&self) -> Option<Vec<VideoInput>> {
        self.input_videos.as_ref().map(|v| v.clone_videos())
    }

    pub fn clone_frames_range(&self, range: Range<usize>) -> Option<Vec<VideoInput>> {
        self.input_videos
            .as_ref()
            .map(|v| v.clone_frames_range(range))
    }

    pub fn videos(&self) -> Option<&[VideoInput]> {
        self.input_videos.as_ref().map(|v| v.videos())
    }

    pub fn video_hashes(&self) -> Option<&[u64]> {
        self.input_videos.as_ref().map(|v| v.hashes())
    }

    pub fn has_videos(&self) -> bool {
        self.input_videos
            .as_ref()
            .is_some_and(|v| !v.videos().is_empty())
    }

    pub fn keep_num_videos(&mut self, videos_to_keep: usize) {
        if let Some(vids) = self.input_videos.as_mut() {
            vids.keep_num_videos(videos_to_keep)
        }
    }

    pub fn keep_num_video_frames(&mut self, video_frames_to_keep: usize) {
        if let Some(vids) = self.input_videos.as_mut() {
            vids.keep_num_video_frames(video_frames_to_keep)
        }
    }

    pub fn keep_num_images(&mut self, images_to_keep: usize) {
        if let Some(imgs) = self.input_images.as_mut() {
            imgs.keep_num_images(images_to_keep);
        }
        // Invalidate preprocessed pixel value cache, the trimmed image set
        // no longer matches the cached tensor dimensions (used by Qwen VL models).
        self.cached_pixel_values = None;
        self.cached_pixel_attention_mask = None;
        self.cached_spatial_shapes = None;
        self.cached_num_crops = None;
        self.cached_img_thw = None;
        self.cached_vid_thw = None;
    }

    /// Per-item multimodal feature positions for prefix caching block hashing.
    pub fn mm_features(&self) -> &[MultiModalFeature] {
        &self.mm_features
    }

    /// Set per-item multimodal feature positions. Should be called once during the
    /// first `process_inputs()` call when all images/audios are available.
    pub fn set_mm_features(&mut self, features: Vec<MultiModalFeature>) {
        self.mm_features = features;
    }
}

/// Scan a token sequence for contiguous runs of a placeholder token ID.
/// Returns `(offset, length)` pairs for each run, in order of appearance.
///
/// Used by multimodal model input processors to find where each image's placeholder
/// tokens are in the expanded token sequence, so that `MultiModalFeature` entries
/// can be built for position-aware prefix cache block hashing.
pub fn find_image_placeholder_ranges(tokens: &[u32], placeholder_id: u32) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] == placeholder_id {
            let start = i;
            while i < tokens.len() && tokens[i] == placeholder_id {
                i += 1;
            }
            ranges.push((start, i - start));
        } else {
            i += 1;
        }
    }
    ranges
}

/// Scan a token sequence for ranges delimited by start and end token IDs (inclusive).
/// Returns `(offset, length)` pairs for each range found.
///
/// Useful for models like Llama4 that wrap each image in `<|image_start|>...<|image_end|>`.
pub fn find_image_delimited_ranges(
    tokens: &[u32],
    start_id: u32,
    end_id: u32,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] == start_id {
            let start = i;
            // Find matching end token
            while i < tokens.len() && tokens[i] != end_id {
                i += 1;
            }
            if i < tokens.len() {
                // Include the end token
                ranges.push((start, i - start + 1));
            }
        }
        i += 1;
    }
    ranges
}

pub fn find_placeholder_delimited_ranges(
    tokens: &[u32],
    placeholder_id: u32,
    start_id: u32,
    end_id: u32,
) -> Vec<(usize, usize)> {
    find_image_placeholder_ranges(tokens, placeholder_id)
        .into_iter()
        .map(|(offset, length)| {
            let placeholder_end = offset + length;
            let start = tokens[..=offset].iter().rposition(|&tok| tok == start_id);
            let end = tokens[placeholder_end..]
                .iter()
                .position(|&tok| tok == end_id)
                .map(|pos| placeholder_end + pos);
            match (start, end) {
                (Some(start), Some(end)) if start < offset && placeholder_end <= end => {
                    (start, end - start + 1)
                }
                _ => (offset, length),
            }
        })
        .collect()
}

pub fn clamp_prefix_cache_len_for_mm_features(
    prefix_len: usize,
    block_size: usize,
    features: &[MultiModalFeature],
) -> usize {
    if prefix_len == 0 || block_size == 0 {
        return prefix_len;
    }

    let mut prefix_len = prefix_len;
    loop {
        let next = features
            .iter()
            .filter(|feature| feature.offset < prefix_len && prefix_len < feature.end())
            .map(|feature| (feature.offset / block_size) * block_size)
            .min()
            .unwrap_or(prefix_len);
        if next == prefix_len {
            return prefix_len;
        }
        prefix_len = next;
    }
}

#[derive(Default)]
pub struct MultimodalPromptLayout {
    features: Vec<MultiModalFeature>,
}

impl MultimodalPromptLayout {
    pub fn extend_ranges(
        &mut self,
        ranges: &[(usize, usize)],
        hashes: &[u64],
        kind: MultimodalKind,
        attention_policy: MultimodalAttentionPolicy,
    ) {
        for (item_idx, (&(offset, length), hash)) in
            (self.next_item_index(kind)..).zip(ranges.iter().zip(hashes.iter()))
        {
            self.features.push(MultiModalFeature {
                kind,
                item_range: item_idx..item_idx + 1,
                hashes: vec![*hash],
                offset,
                length,
                attention_policy,
                splittable: false,
            });
        }
    }

    pub fn into_features(mut self) -> Vec<MultiModalFeature> {
        self.features.sort_by_key(|feature| feature.offset);
        self.features
    }

    pub fn next_item_index(&self, kind: MultimodalKind) -> usize {
        self.features
            .iter()
            .filter(|feature| feature.kind == kind)
            .map(|feature| feature.item_range.end)
            .max()
            .unwrap_or(0)
    }
}

pub fn build_mm_features_from_ranges(
    ranges: &[(usize, usize)],
    hashes: &[u64],
    kind: MultimodalKind,
) -> Vec<MultiModalFeature> {
    build_mm_features_from_ranges_with_policy(
        ranges,
        hashes,
        kind,
        MultimodalAttentionPolicy::Causal,
    )
}

pub fn build_mm_features_from_ranges_with_policy(
    ranges: &[(usize, usize)],
    hashes: &[u64],
    kind: MultimodalKind,
    attention_policy: MultimodalAttentionPolicy,
) -> Vec<MultiModalFeature> {
    let mut layout = MultimodalPromptLayout::default();
    layout.extend_ranges(ranges, hashes, kind, attention_policy);
    layout.into_features()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_hash_distinguishes_geometry() {
        let bytes = vec![1, 2, 3, 4, 5, 6];
        let wide =
            image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(2, 1, bytes.clone()).unwrap());
        let tall = image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(1, 2, bytes).unwrap());

        assert_eq!(wide.as_bytes(), tall.as_bytes());
        let images = SequenceImages::new(vec![wide, tall]);
        assert_ne!(images.hashes()[0], images.hashes()[1]);
    }

    #[test]
    fn image_hash_distinguishes_color_type() {
        let rgba = image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(1, 1, vec![1, 2, 3, 4]).unwrap(),
        );
        let luma_alpha = image::DynamicImage::ImageLumaA16(
            image::ImageBuffer::<image::LumaA<u16>, Vec<u16>>::from_raw(
                1,
                1,
                vec![u16::from_ne_bytes([1, 2]), u16::from_ne_bytes([3, 4])],
            )
            .unwrap(),
        );

        assert_eq!(rgba.as_bytes(), luma_alpha.as_bytes());
        assert_ne!(rgba.color(), luma_alpha.color());
        let images = SequenceImages::new(vec![rgba, luma_alpha]);
        assert_ne!(images.hashes()[0], images.hashes()[1]);
    }

    #[test]
    fn multimodal_prefix_placeholder_delimited_ranges_include_wrappers() {
        let tokens = vec![1, 10, 20, 20, 11, 2, 10, 30, 30, 30, 11, 3];
        let img = find_placeholder_delimited_ranges(&tokens, 20, 10, 11);
        let video = find_placeholder_delimited_ranges(&tokens, 30, 10, 11);
        let fallback = find_placeholder_delimited_ranges(&tokens, 2, 99, 100);

        assert_eq!(img, vec![(1, 4)]);
        assert_eq!(video, vec![(6, 5)]);
        assert_eq!(fallback, vec![(5, 1)]);
    }
}

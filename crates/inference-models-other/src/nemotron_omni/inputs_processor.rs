//! Nemotron-H Omni's inputs: one tile per image on a patch grid sized to a patch budget (aspect kept, sides even),
//! antialiased bicubic in float, `<image>` expanded to `<img>`, one `<image>` per token, `</img>`; a video's frames on
//! one grid, paired into tubelets, each a timestamped `<img>` block; each clip's normalized log-mel, `<so_embedding>`
//! expanded to `<so_start>`, one `<so_embedding>` per encoder frame, `<so_end>`.

use std::hash::{Hash, Hasher};
use std::{any::Any, sync::Arc};

use image::DynamicImage;
use inference_audio::nemo::NemoMel;
use inference_tensor::{Device, Result, Tensor};
use tokenizers::Tokenizer;

use crate::device_map::DeviceMapper;
use crate::media_inputs::{
    audio_prompt::{active_clip_range, mono_clips},
    media::{MultimodalPromptLayout, find_placeholder_delimited_ranges},
    processor::{
        InputProcessorOutput, InputsHost, InputsProcessorValidationError, MediaSequence,
        ModelInputs, MultimodalInputsProcessor, TextInputs,
    },
    video::VideoInput,
};
use crate::paged_attention::{
    PagedAttentionMeta,
    block_hash::{MultimodalAttentionPolicy, MultimodalKind},
};

use super::{
    Config, ImageLayout, ImagePatches, MODEL, NemotronOmniSpecificArgs, VideoLayout,
    sound::SoundConfig,
};

pub const IMAGE_TOKEN: &str = "<image>";
pub const IMAGE_START: &str = "<img>";
pub const IMAGE_END: &str = "</img>";
pub const SOUND_TOKEN: &str = "<so_embedding>";
pub const SOUND_START: &str = "<so_start>";
pub const SOUND_END: &str = "<so_end>";
pub const VIDEO_TOKEN: &str = "<video>";
const CHANNELS: usize = 3;
const PIXEL_SCALE: f32 = 255.0;
const MS_PER_SECOND: f64 = 1000.0;

pub struct NemotronOmniInputsProcessor {
    patch_size: usize,
    merge: usize,
    layout: ImageLayout,
    mean: Vec<f32>,
    std: Vec<f32>,
    image_token: u32,
    video: Option<VideoLayout>,
    sound: Option<SoundInputs>,
}

struct SoundInputs {
    config: SoundConfig,
    mel: NemoMel,
    token: u32,
}

impl SoundInputs {
    // the clip's `floor(samples / hop)` log-mel frames through the encoder's subsampling
    fn tokens(&self, samples: usize) -> usize {
        self.config.frames(samples / super::sound::HOP_LENGTH)
    }

    fn features(&self, pcm: &[f32]) -> anyhow::Result<Tensor> {
        let (features, frames) = self.mel.features(pcm)?;
        Ok(Tensor::from_vec(
            features,
            (frames, self.config.num_mel_bins),
            &Device::Cpu,
        )?)
    }
}

// the sequence's images as tiles, then its videos' tubelets, patches concatenated in that order (a tubelet's patch as
// its frames' rows one after another): each tile's grid and each image's tiles, each video's tubelets
struct Preprocessed {
    patches: Tensor,
    tiles: Vec<(usize, usize)>,
    images: Vec<std::ops::Range<usize>>,
    videos: Vec<VideoTubelets>,
}

// one video's tubelets: how many, their grid, and their first row in `Preprocessed::patches`
#[derive(Clone, Copy)]
struct VideoTubelets {
    count: usize,
    grid: (usize, usize),
    start: usize,
}

/// `(rows, cols)` a `width x height` video frame is resized to: about `target` patches, aspect kept, each side a
/// multiple of `merge` (rounded up while the product still fits), as NVIDIA's `_compute_target_patches_video`.
pub fn video_grid(
    (width, height): (usize, usize),
    target: usize,
    merge: usize,
    keep_aspect: bool,
) -> (usize, usize) {
    if !keep_aspect {
        let side = merge.max((target as f64).sqrt() as usize / merge * merge);
        return (side, side);
    }
    let aspect = width as f64 / height.max(1) as f64;
    // Python's `round`, ties to even
    let mut rows = ((target as f64 / aspect).sqrt().round_ties_even() as usize).max(1);
    let mut cols = ((target as f64 * aspect).sqrt().round_ties_even() as usize).max(1);
    if merge > 1 {
        let up = |side: usize| side + (merge - side % merge) % merge;
        if up(rows) * up(cols) <= target {
            (rows, cols) = (up(rows), up(cols));
        } else {
            rows = merge.max(rows - rows % merge);
            cols = merge.max(cols - cols % merge);
        }
    }
    (rows, cols)
}

/// Each tubelet's label: `"Frame i sampled at t seconds and frame i+1 sampled at t' seconds: "`, a frame's time its
/// source index at `int(1000 / fps)` ms a frame (vLLM's `get_video_repl`), or `"Frame i"` with no frame rate.
pub fn tubelet_labels(video: &VideoInput, temporal: usize) -> Vec<String> {
    let frames = video.frames.len();
    let frame_ms = (video.fps > 0.0).then(|| (MS_PER_SECOND / video.fps).trunc());
    (0..frames.div_ceil(temporal))
        .map(|tubelet| {
            let parts = (0..temporal)
                .map(|j| tubelet * temporal + j)
                .take_while(|&frame| frame < frames)
                .enumerate()
                .map(|(j, frame)| {
                    let prefix = if j == 0 { "Frame" } else { "frame" };
                    match frame_ms {
                        Some(ms) => {
                            let index = video.sampled_indices.get(frame).copied().unwrap_or(frame);
                            let seconds = index as f64 * ms / MS_PER_SECOND;
                            format!("{prefix} {} sampled at {seconds:.2} seconds", frame + 1)
                        }
                        None => format!("{prefix} {}", frame + 1),
                    }
                })
                .collect::<Vec<_>>();
            format!("{}: ", parts.join(" and "))
        })
        .collect()
}

// an image's `CHANNELS` planes of `height x width` pixels, as floats in 0..=255
fn planes(image: &DynamicImage) -> (Vec<f32>, (usize, usize)) {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    let mut planes = vec![0f32; CHANNELS * height * width];
    for (i, pixel) in rgb.pixels().enumerate() {
        for c in 0..CHANNELS {
            planes[c * height * width + i] = f32::from(pixel[c]);
        }
    }
    (planes, (height, width))
}

fn resized(planes: &[f32], size: (usize, usize), out: (usize, usize)) -> Vec<f32> {
    if out == size {
        planes.to_vec()
    } else {
        inference_vision::aa_bicubic_resize_planar(planes, CHANNELS, size, out)
    }
}

// a video tubelet's content hash: its video's and its place in it
fn tubelet_hash(video: u64, tubelet: usize) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (video, tubelet).hash(&mut hasher);
    hasher.finish()
}

/// InternVL's tiling of a `width x height` image: `(cols, rows)` of the closest aspect ratio among the grids of at
/// most `max_tiles`, a tie going to the larger grid when the image covers more than half of it.
pub fn tile_grid((width, height): (usize, usize), size: usize, max_tiles: usize) -> (usize, usize) {
    let mut ratios = (1..=max_tiles)
        .flat_map(|i| (1..=max_tiles).map(move |j| (i, j)))
        .filter(|&(i, j)| i * j <= max_tiles)
        .collect::<Vec<_>>();
    ratios.sort_by_key(|&(i, j)| i * j);
    let aspect = width as f64 / height as f64;
    let area = (width * height) as f64;
    let mut best = (1, 1);
    let mut best_diff = f64::INFINITY;
    for (i, j) in ratios {
        let diff = (aspect - i as f64 / j as f64).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (i, j);
        } else if diff == best_diff && area > 0.5 * (size * size * i * j) as f64 {
            best = (i, j);
        }
    }
    best
}

/// `(rows, cols)` of the tile an `width x height` image is resized to, as NVIDIA's processor picks it.
pub fn target_grid(
    (width, height): (usize, usize),
    patch_size: usize,
    merge: usize,
    (min_patches, max_patches): (usize, usize),
) -> (usize, usize) {
    let budget = max_patches;
    // Python's `round(x + 0.5)`, ties to even
    let closest = |side: usize| (side as f64 / patch_size as f64 + 0.5).round_ties_even();
    let (closest_h, closest_w) = (closest(height), closest(width));
    let factor = (budget as f64 / (closest_h * closest_w)).sqrt().min(1.0);
    let mut rows = (factor * closest_h).floor() as usize;
    let mut cols = (factor * closest_w).floor() as usize;
    if budget > min_patches && rows * cols < min_patches {
        let up = (min_patches as f64 / (rows * cols) as f64).sqrt();
        rows = (up * rows as f64).ceil() as usize;
        cols = (up * cols as f64).ceil() as usize;
    }
    if !rows.is_multiple_of(merge) {
        if (rows + merge - rows % merge) * cols <= budget {
            rows += merge - rows % merge;
        } else {
            rows = merge.max(rows - rows % merge);
        }
    }
    if !cols.is_multiple_of(merge) {
        if rows * (cols + merge - cols % merge) <= budget {
            cols += merge - cols % merge;
        } else {
            cols = merge.max(cols - cols % merge);
        }
    }
    (rows, cols)
}

impl NemotronOmniInputsProcessor {
    pub fn new(cfg: &Config, max_tiles: Option<usize>) -> Result<Self> {
        let layout = cfg.image_layout(max_tiles)?;
        if let Some(rate) = cfg.video_pruning_rate.filter(|&rate| rate > 0.0)
            && cfg.vision_config.video_layout().is_some()
        {
            tracing::info!(
                rate,
                "{MODEL}: videos keep every tubelet's tokens; the reference's EVS pruning is not applied"
            );
        }
        Ok(Self {
            patch_size: cfg.patch_size,
            merge: cfg.merge(),
            layout,
            mean: cfg.norm_mean.clone(),
            std: cfg.norm_std.clone(),
            image_token: cfg.img_context_token_id,
            video: cfg.vision_config.video_layout(),
            sound: cfg
                .sound_config
                .as_ref()
                .zip(cfg.sound_context_token_id)
                .map(|(config, token)| SoundInputs {
                    mel: NemoMel::new(config.mel()),
                    config: config.clone(),
                    token,
                }),
        })
    }

    fn clips(&self, seq: &dyn MediaSequence) -> anyhow::Result<Vec<Vec<f32>>> {
        if !seq.has_audios() {
            return Ok(Vec::new());
        }
        let Some(sound) = &self.sound else {
            anyhow::bail!("{MODEL}: this checkpoint has no sound encoder");
        };
        mono_clips(seq, sound.config.sampling_rate)
    }

    // `rows x cols` patches from the window at (`y`, `x`) of `planes` (`CHANNELS` planes of `height x width`), normalized
    fn window(
        &self,
        planes: &[f32],
        (height, width): (usize, usize),
        (y, x): (usize, usize),
        (rows, cols): (usize, usize),
        out: &mut Vec<f32>,
    ) {
        let p = self.patch_size;
        for row in 0..rows {
            for col in 0..cols {
                for c in 0..CHANNELS {
                    let plane = &planes[c * height * width..][..height * width];
                    for py in 0..p {
                        let line = &plane[(y + row * p + py) * width + x + col * p..][..p];
                        out.extend(
                            line.iter()
                                .map(|&v| (v / PIXEL_SCALE - self.mean[c]) / self.std[c]),
                        );
                    }
                }
            }
        }
    }

    /// One image's tiles, in order: each tile's normalized `(rows * cols, 3 * patch * patch)` patches and its grid.
    pub fn patches(&self, image: &DynamicImage) -> Vec<(Vec<f32>, (usize, usize))> {
        let (planes, (height, width)) = planes(image);
        let resized = |out: (usize, usize)| resized(&planes, (height, width), out);
        let p = self.patch_size;
        match self.layout {
            ImageLayout::Budget { min, max } => {
                let (rows, cols) = target_grid((width, height), p, self.merge, (min, max));
                let out = (rows * p, cols * p);
                let mut patches = Vec::new();
                self.window(&resized(out), out, (0, 0), (rows, cols), &mut patches);
                vec![(patches, (rows, cols))]
            }
            ImageLayout::Tiles {
                size,
                max_tiles,
                thumbnail,
            } => {
                let (across, down) = tile_grid((width, height), size, max_tiles);
                let out = (down * size, across * size);
                let whole = resized(out);
                let grid = (size / p, size / p);
                let mut tiles = Vec::with_capacity(across * down + 1);
                for ty in 0..down {
                    for tx in 0..across {
                        let mut patches = Vec::new();
                        self.window(&whole, out, (ty * size, tx * size), grid, &mut patches);
                        tiles.push((patches, grid));
                    }
                }
                if thumbnail && tiles.len() > 1 {
                    let mut patches = Vec::new();
                    self.window(
                        &resized((size, size)),
                        (size, size),
                        (0, 0),
                        grid,
                        &mut patches,
                    );
                    tiles.push((patches, grid));
                }
                tiles
            }
        }
    }

    /// One video's tubelets on one grid: each tubelet's patches, its frames' `(3 * patch * patch)` rows one after
    /// another a patch (the last frame repeated to fill the final tubelet), and the grid.
    pub fn video_patches(
        &self,
        video: &VideoInput,
    ) -> anyhow::Result<(Vec<f32>, usize, (usize, usize))> {
        let Some(layout) = self.video else {
            anyhow::bail!("{MODEL}: this checkpoint's vision tower takes no video");
        };
        let Some(first) = video.frames.first() else {
            anyhow::bail!("{MODEL}: a video with no frames");
        };
        let grid = video_grid(
            (first.width() as usize, first.height() as usize),
            layout.target,
            self.merge,
            layout.keep_aspect,
        );
        let p = self.patch_size;
        let out = (grid.0 * p, grid.1 * p);
        let frames = video
            .frames
            .iter()
            .map(|frame| {
                let (planes, size) = planes(frame);
                let mut patches = Vec::new();
                self.window(
                    &resized(&planes, size, out),
                    out,
                    (0, 0),
                    grid,
                    &mut patches,
                );
                patches
            })
            .collect::<Vec<_>>();
        let tubelets = frames.len().div_ceil(layout.temporal);
        let width = CHANNELS * p * p;
        let mut all = Vec::with_capacity(tubelets * layout.temporal * frames[0].len());
        for tubelet in 0..tubelets {
            for patch in 0..grid.0 * grid.1 {
                for j in 0..layout.temporal {
                    let frame = &frames[(tubelet * layout.temporal + j).min(frames.len() - 1)];
                    all.extend_from_slice(&frame[patch * width..][..width]);
                }
            }
        }
        Ok((all, tubelets, grid))
    }

    fn tokens(&self, (rows, cols): (usize, usize)) -> usize {
        rows * cols / (self.merge * self.merge)
    }

    // an image's tokens: every tile's, shuffled
    fn image_tokens(&self, processed: &Preprocessed, image: usize) -> usize {
        processed.tiles[processed.images[image].clone()]
            .iter()
            .map(|&grid| self.tokens(grid))
            .sum()
    }

    fn cached_or_preprocess(&self, seq: &mut dyn MediaSequence) -> anyhow::Result<Preprocessed> {
        let mm = seq.multimodal();
        let width = CHANNELS * self.patch_size * self.patch_size;
        let temporal = self.video.map_or(1, |video| video.temporal);
        // one row per tile (the image it belongs to and its grid), and per video (its tubelets and their grid)
        if let (Some(patches), Some(rows)) = (&mm.cached_pixel_values, &mm.cached_img_thw) {
            let rows = rows.to_vec2::<u32>()?;
            let tiles = rows
                .iter()
                .map(|g| (g[1] as usize, g[2] as usize))
                .collect::<Vec<(usize, usize)>>();
            let mut images: Vec<std::ops::Range<usize>> = Vec::new();
            for (i, g) in rows.iter().enumerate() {
                match images.get_mut(g[0] as usize) {
                    Some(range) => range.end = i + 1,
                    None => images.push(i..i + 1),
                }
            }
            let mut start = tiles.iter().map(|(r, c)| r * c).sum::<usize>();
            let videos = match &mm.cached_vid_thw {
                Some(rows) => rows
                    .to_vec2::<u32>()?
                    .iter()
                    .map(|g| {
                        let (count, grid) = (g[0] as usize, (g[1] as usize, g[2] as usize));
                        let video = VideoTubelets { count, grid, start };
                        start += count * grid.0 * grid.1 * temporal;
                        video
                    })
                    .collect(),
                None => Vec::new(),
            };
            return Ok(Preprocessed {
                patches: patches.clone(),
                tiles,
                images,
                videos,
            });
        }
        let mut all = Vec::new();
        let mut tiles = Vec::new();
        let mut ranges = Vec::new();
        let mut rows = Vec::new();
        for (index, image) in seq.clone_images().unwrap_or_default().iter().enumerate() {
            let start = tiles.len();
            for (patches, grid) in self.patches(image) {
                all.extend(patches);
                rows.extend([index as u32, grid.0 as u32, grid.1 as u32]);
                tiles.push(grid);
            }
            ranges.push(start..tiles.len());
        }
        let mut videos = Vec::new();
        let mut video_rows = Vec::new();
        for video in seq.videos().unwrap_or_default() {
            let (patches, count, grid) = self.video_patches(video)?;
            videos.push(VideoTubelets {
                count,
                grid,
                start: all.len() / width,
            });
            all.extend(patches);
            video_rows.extend([count as u32, grid.0 as u32, grid.1 as u32]);
        }
        let patches = Tensor::from_vec(all, ((), width), &Device::Cpu)?;
        let mm = seq.multimodal_mut();
        mm.cached_pixel_values = Some(patches.clone());
        mm.cached_img_thw = Some(Tensor::from_vec(rows, (tiles.len(), 3), &Device::Cpu)?);
        mm.cached_vid_thw = (!videos.is_empty())
            .then(|| Tensor::from_vec(video_rows, (videos.len(), 3), &Device::Cpu))
            .transpose()?;
        Ok(Preprocessed {
            patches,
            tiles,
            images: ranges,
            videos,
        })
    }

    fn maybe_expand_prompt(
        &self,
        tokenizer: &Tokenizer,
        seq: &mut dyn MediaSequence,
        images: Option<&Preprocessed>,
        clips: &[Vec<f32>],
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        if seq.multimodal().has_changed_prompt {
            return Ok(());
        }
        let image_counts = images.map_or_else(Vec::new, |images| {
            (0..images.images.len())
                .map(|image| self.image_tokens(images, image))
                .collect::<Vec<_>>()
        });
        let temporal = self.video.map_or(1, |video| video.temporal);
        let videos = seq
            .videos()
            .unwrap_or_default()
            .iter()
            .zip(images.map_or(&[][..], |images| &images.videos))
            .map(|(video, tubelets)| (tubelet_labels(video, temporal), self.tokens(tubelets.grid)))
            .collect::<Vec<_>>();
        let (mut expanded, blocks) =
            expand_visual(seq.get_initial_prompt(), &image_counts, &videos)?;
        if let Some(sound) = &self.sound {
            let counts = clips
                .iter()
                .map(|clip| sound.tokens(clip.len()))
                .collect::<Vec<_>>();
            expanded = expand(
                &expanded,
                (SOUND_TOKEN, SOUND_START, SOUND_END),
                &counts,
                "audio clips",
            )?;
        }
        seq.set_initial_prompt(expanded.clone());
        let ids = tokenizer
            .encode_fast(expanded, false)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?
            .get_ids()
            .to_vec();
        if seq.mm_features().is_empty() {
            let mut layout = MultimodalPromptLayout::default();
            let delimited =
                |token: u32, start: &str, end: &str| -> anyhow::Result<Vec<(usize, usize)>> {
                    let (Some(start_id), Some(end_id)) =
                        (tokenizer.token_to_id(start), tokenizer.token_to_id(end))
                    else {
                        anyhow::bail!("{MODEL} tokenizer has no `{start}` / `{end}`");
                    };
                    Ok(find_placeholder_delimited_ranges(
                        &ids, token, start_id, end_id,
                    ))
                };
            if !blocks.is_empty() {
                let ranges = delimited(self.image_token, IMAGE_START, IMAGE_END)?;
                if ranges.len() != blocks.len() {
                    anyhow::bail!(
                        "{MODEL}: {} image blocks in the prompt where {} were written",
                        ranges.len(),
                        blocks.len()
                    );
                }
                let of = |kind| {
                    ranges
                        .iter()
                        .zip(&blocks)
                        .filter(|(_, block)| **block == kind)
                        .map(|(range, _)| *range)
                        .collect::<Vec<_>>()
                };
                if !image_counts.is_empty() {
                    let hashes = seq.image_hashes().ok_or_else(|| {
                        anyhow::anyhow!("{MODEL} images are missing content hashes")
                    })?;
                    layout.extend_ranges(
                        &of(MultimodalKind::Image),
                        hashes,
                        MultimodalKind::Image,
                        MultimodalAttentionPolicy::Causal,
                    );
                }
                if !videos.is_empty() {
                    let hashes = seq
                        .video_hashes()
                        .ok_or_else(|| {
                            anyhow::anyhow!("{MODEL} videos are missing content hashes")
                        })?
                        .iter()
                        .zip(&videos)
                        .flat_map(|(&hash, (labels, _))| {
                            (0..labels.len()).map(move |tubelet| tubelet_hash(hash, tubelet))
                        })
                        .collect::<Vec<_>>();
                    layout.extend_ranges(
                        &of(MultimodalKind::Video),
                        &hashes,
                        MultimodalKind::Video,
                        MultimodalAttentionPolicy::Causal,
                    );
                }
            }
            if let Some(sound) = self.sound.as_ref().filter(|_| !clips.is_empty()) {
                let hashes = seq
                    .audio_hashes()
                    .ok_or_else(|| anyhow::anyhow!("{MODEL} clips are missing content hashes"))?;
                let ranges = delimited(sound.token, SOUND_START, SOUND_END)?;
                layout.extend_ranges(
                    &ranges,
                    hashes,
                    MultimodalKind::Audio,
                    MultimodalAttentionPolicy::Causal,
                );
            }
            seq.set_mm_features(layout.into_features());
        }
        seq.set_toks_and_reallocate(ids, paged_attn_metadata);
        seq.multimodal_mut().has_changed_prompt = true;
        Ok(())
    }

    // the tiles of the images in `range`, each its own patches
    fn select(
        processed: &Preprocessed,
        range: std::ops::Range<usize>,
    ) -> Result<Vec<ImagePatches>> {
        let (Some(first), Some(last)) = (
            processed.images.get(range.start),
            range
                .end
                .checked_sub(1)
                .and_then(|i| processed.images.get(i)),
        ) else {
            return Ok(Vec::new());
        };
        let tiles = first.start..last.end;
        let mut offset = processed.tiles[..tiles.start]
            .iter()
            .map(|(r, c)| r * c)
            .sum::<usize>();
        processed.tiles[tiles]
            .iter()
            .map(|&(rows, cols)| {
                let patches = processed.patches.narrow(0, offset, rows * cols)?;
                offset += rows * cols;
                Ok(ImagePatches {
                    patches,
                    rows,
                    cols,
                    video: false,
                })
            })
            .collect()
    }

    // the sequence's `item`th tubelet, counting through its videos
    fn tubelet(&self, processed: &Preprocessed, mut item: usize) -> anyhow::Result<ImagePatches> {
        let temporal = self.video.map_or(1, |video| video.temporal);
        for video in &processed.videos {
            if item < video.count {
                let (rows, cols) = video.grid;
                let frames = rows * cols * temporal;
                let patches = processed
                    .patches
                    .narrow(0, video.start + item * frames, frames)?
                    .reshape((rows * cols, ()))?;
                return Ok(ImagePatches {
                    patches,
                    rows,
                    cols,
                    video: true,
                });
            }
            item -= video.count;
        }
        anyhow::bail!("{MODEL}: no tubelet {item} among the videos")
    }

    // the image tiles and video tubelets this step's placeholders hold, in prompt order
    fn visuals(
        &self,
        seq: &dyn MediaSequence,
        processed: &Preprocessed,
    ) -> anyhow::Result<Vec<ImagePatches>> {
        let tubelets = processed
            .videos
            .iter()
            .map(|video| video.count)
            .sum::<usize>();
        let active = |kind, count| -> anyhow::Result<std::ops::Range<usize>> {
            if count == 0 {
                Ok(0..0)
            } else if seq.is_chunked_prefill_view() {
                seq.active_local_multimodal_item_range(kind, count)
                    .ok_or_else(|| {
                        anyhow::anyhow!("{MODEL} chunk is missing its active {kind:?} range")
                    })
            } else {
                Ok(seq.count_prefix_cached_mm_items_by_kind(kind)..count)
            }
        };
        let images = active(MultimodalKind::Image, processed.images.len())?;
        let videos = active(MultimodalKind::Video, tubelets)?;
        let mut out = Vec::new();
        for feature in seq.mm_features() {
            let item = feature.item_range.start;
            match feature.kind {
                MultimodalKind::Image if images.contains(&item) => {
                    out.extend(Self::select(processed, item..item + 1)?);
                }
                MultimodalKind::Video if videos.contains(&item) => {
                    out.push(self.tubelet(processed, item)?);
                }
                _ => {}
            }
        }
        Ok(out)
    }
}

/// `prompt` with its `i`th `token` (of `tags`, the token and its opening and closing tags) as `counts[i]` tokens between
/// the tags.
fn expand(
    prompt: &str,
    (token, start, end): (&str, &str, &str),
    counts: &[usize],
    items: &str,
) -> anyhow::Result<String> {
    let placeholders = prompt.matches(token).count();
    if placeholders != counts.len() {
        return Err(InputsProcessorValidationError(format!(
            "The number of `{token}` tokens ({placeholders}) must match the number of {items} ({})",
            counts.len()
        ))
        .into());
    }
    let mut parts = prompt.split(token);
    let mut expanded = parts.next().unwrap_or_default().to_string();
    for (&count, part) in counts.iter().zip(parts) {
        expanded.push_str(start);
        expanded.push_str(&token.repeat(count));
        expanded.push_str(end);
        expanded.push_str(part);
    }
    Ok(expanded)
}

fn image_block(tokens: usize) -> String {
    format!("{IMAGE_START}{}{IMAGE_END}", IMAGE_TOKEN.repeat(tokens))
}

/// What a `<video>` becomes: each tubelet's label and `tokens`-token `<img>` block, the tubelets joined by newlines.
pub fn video_replacement(labels: &[String], tokens: usize) -> String {
    labels
        .iter()
        .map(|label| format!("{label}{}", image_block(tokens)))
        .collect::<Vec<_>>()
        .join("\n")
}

/// `prompt` with its `<image>`s and `<video>`s expanded in the order they appear: the `i`th image as one `<img>` block
/// of `images[i]` tokens, the `i`th video as its tubelets' blocks, each after its label, joined by newlines; and the
/// kind of every `<img>` block written, in order.
fn expand_visual(
    prompt: &str,
    images: &[usize],
    videos: &[(Vec<String>, usize)],
) -> anyhow::Result<(String, Vec<MultimodalKind>)> {
    for (token, expected, items) in [
        (IMAGE_TOKEN, images.len(), "images"),
        (VIDEO_TOKEN, videos.len(), "videos"),
    ] {
        let placeholders = prompt.matches(token).count();
        if placeholders != expected {
            return Err(InputsProcessorValidationError(format!(
                "The number of `{token}` tokens ({placeholders}) must match the number of {items} ({expected})"
            ))
            .into());
        }
    }
    let block = |out: &mut String, tokens: usize| out.push_str(&image_block(tokens));
    let (mut images, mut videos) = (images.iter(), videos.iter());
    let mut out = String::with_capacity(prompt.len());
    let mut kinds = Vec::new();
    let mut rest = prompt;
    loop {
        let next_image = rest.find(IMAGE_TOKEN);
        let next_video = rest.find(VIDEO_TOKEN);
        let (at, video) = match (next_image, next_video) {
            (Some(i), Some(v)) => (i.min(v), v < i),
            (Some(i), None) => (i, false),
            (None, Some(v)) => (v, true),
            (None, None) => break,
        };
        out.push_str(&rest[..at]);
        if video {
            let (labels, tokens) = videos.next().expect("counted above");
            out.push_str(&video_replacement(labels, *tokens));
            kinds.extend(labels.iter().map(|_| MultimodalKind::Video));
            rest = &rest[at + VIDEO_TOKEN.len()..];
        } else {
            block(&mut out, *images.next().expect("counted above"));
            kinds.push(MultimodalKind::Image);
            rest = &rest[at + IMAGE_TOKEN.len()..];
        }
    }
    out.push_str(rest);
    Ok((out, kinds))
}

impl MultimodalInputsProcessor for NemotronOmniInputsProcessor {
    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut dyn MediaSequence],
        _device: &Device,
        _other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("{MODEL} inputs processor requires a tokenizer");
        };
        for seq in input_seqs {
            if !(seq.has_images() || seq.has_videos() || seq.has_audios())
                || seq.multimodal().has_changed_prompt
            {
                continue;
            }
            let processed = if seq.has_images() || seq.has_videos() {
                Some(self.cached_or_preprocess(&mut **seq)?)
            } else {
                None
            };
            let clips = self.clips(&**seq)?;
            self.maybe_expand_prompt(
                &tokenizer,
                &mut **seq,
                processed.as_ref(),
                &clips,
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
        _other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> anyhow::Result<InputProcessorOutput> {
        if no_kv_cache {
            anyhow::bail!("Vision model must have kv cache.");
        }
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("{MODEL} inputs processor requires a tokenizer");
        };
        let mut images = Vec::new();
        let mut sounds = Vec::new();
        if is_prompt {
            for seq in input_seqs
                .iter_mut()
                .filter(|seq| seq.has_images() || seq.has_videos() || seq.has_audios())
            {
                let processed = if seq.has_images() || seq.has_videos() {
                    Some(self.cached_or_preprocess(&mut **seq)?)
                } else {
                    None
                };
                let clips = self.clips(&**seq)?;
                self.maybe_expand_prompt(
                    &tokenizer,
                    &mut **seq,
                    processed.as_ref(),
                    &clips,
                    paged_attn_metadata.as_mut(),
                )?;
                if let Some(sound) = self.sound.as_ref().filter(|_| !clips.is_empty()) {
                    for clip in &clips[active_clip_range(&**seq, clips.len(), MODEL)?] {
                        sounds.push(sound.features(clip)?);
                    }
                }
                let Some(processed) = processed else {
                    continue;
                };
                images.extend(self.visuals(&**seq, &processed)?);
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
        } = host.step_inputs(toks, input_seqs, text, is_prompt, no_kv_cache)?;

        Ok(InputProcessorOutput {
            inputs: Box::new(ModelInputs {
                input_ids: input,
                seqlen_offsets: positions,
                context_lens,
                position_ids,
                pixel_values: None,
                model_specific_args: Box::new(NemotronOmniSpecificArgs { images, sounds }),
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

#[cfg(test)]
mod tests {
    use super::{
        IMAGE_TOKEN, MultimodalKind, VideoInput, expand_visual, target_grid, tile_grid,
        tubelet_labels, video_grid,
    };

    // transformers' `_compute_target_patches` for the same frames: 352 x 240 at 1024 patches rounds 26 x 39 up to
    // 26 x 40, over budget, so down to 26 x 38 (the parity run's grid)
    #[test]
    fn a_video_frame_takes_the_references_grid() {
        assert_eq!(video_grid((352, 240), 1024, 2, true), (26, 38));
        assert_eq!(video_grid((64, 48), 16, 2, true), (2, 4));
        assert_eq!(video_grid((640, 360), 1024, 2, false), (32, 32));
    }

    // vLLM's labels: frame times at int(1000 / fps) ms a frame from the sampled indices, the odd frame alone
    #[test]
    fn tubelets_are_labelled_with_their_frames_times() {
        let frames = (0..3)
            .map(|_| image::DynamicImage::new_rgb8(4, 4))
            .collect();
        let video = VideoInput::from_frames(frames, 30.0, Some(vec![0, 15, 30]));
        assert_eq!(
            tubelet_labels(&video, 2),
            [
                "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.49 seconds: ",
                "Frame 3 sampled at 0.99 seconds: "
            ]
        );
    }

    // placeholders expand in prompt order, a video as one block per tubelet after its label
    #[test]
    fn images_and_videos_expand_where_they_stand() -> anyhow::Result<()> {
        let labels = vec!["A: ".to_string(), "B: ".to_string()];
        let (text, kinds) = expand_visual("x<video>\ny<image>z", &[1], &[(labels, 2)])?;
        let block = |n: usize| format!("<img>{}</img>", IMAGE_TOKEN.repeat(n));
        assert_eq!(
            text,
            format!("xA: {}\nB: {}\ny{}z", block(2), block(2), block(1))
        );
        assert_eq!(
            kinds,
            [
                MultimodalKind::Video,
                MultimodalKind::Video,
                MultimodalKind::Image
            ]
        );
        assert!(expand_visual("<video>", &[], &[]).is_err());
        Ok(())
    }

    // `(cols, rows)` NVIDIA's `calculate_targets` gives for the same sizes at 512 with 12 tiles
    #[test]
    fn tiles_follow_internvls_closest_aspect_ratio() {
        for (size, want) in [
            ((470, 152), (3, 1)),
            ((640, 480), (4, 3)),
            ((512, 512), (1, 1)),
            ((1920, 1080), (4, 2)),
            ((100, 5000), (1, 12)),
            // a tie with 1 x 1, taken by the larger grid as the image covers more than half of it
            ((1000, 1000), (2, 2)),
            ((300, 200), (3, 2)),
            ((1024, 768), (4, 3)),
            ((2000, 500), (4, 1)),
        ] {
            assert_eq!(tile_grid(size, 512, 12), want, "{size:?}");
        }
    }

    const BUDGET: (usize, usize) = (1024, 13312);

    // grids NVIDIA's `_compute_target_patches` gives for the same sizes
    #[test]
    fn the_tile_follows_the_reference_patch_budget() {
        assert_eq!(target_grid((500, 375), 16, 2, BUDGET), (28, 38));
        assert_eq!(target_grid((640, 480), 16, 2, BUDGET), (30, 40));
        assert_eq!(target_grid((4000, 3000), 16, 2, BUDGET), (100, 132));
        // 31 patches exactly: Python's round(31.5) is 32
        assert_eq!(target_grid((496, 1024), 16, 2, BUDGET), (64, 32));
        assert_eq!(target_grid((496, 496), 16, 2, BUDGET), (32, 32));
        assert_eq!(target_grid((100, 5000), 16, 2, BUDGET), (314, 8));
        assert_eq!(target_grid((30, 30), 16, 2, BUDGET), (32, 32));
    }
}

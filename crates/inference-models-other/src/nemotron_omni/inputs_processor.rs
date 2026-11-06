//! Nemotron-H Omni's inputs: one tile per image on a patch grid sized to a patch budget (aspect kept, sides even),
//! antialiased bicubic in float, `<image>` expanded to `<img>`, one `<image>` per token, `</img>`; each clip's
//! normalized log-mel, `<so_embedding>` expanded to `<so_start>`, one `<so_embedding>` per encoder frame, `<so_end>`.

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
};
use crate::paged_attention::{
    PagedAttentionMeta,
    block_hash::{MultimodalAttentionPolicy, MultimodalKind},
};

use super::{Config, ImagePatches, MODEL, NemotronOmniSpecificArgs, sound::SoundConfig};

pub const IMAGE_TOKEN: &str = "<image>";
pub const IMAGE_START: &str = "<img>";
pub const IMAGE_END: &str = "</img>";
pub const SOUND_TOKEN: &str = "<so_embedding>";
pub const SOUND_START: &str = "<so_start>";
pub const SOUND_END: &str = "<so_end>";
const CHANNELS: usize = 3;
const PIXEL_SCALE: f32 = 255.0;

pub struct NemotronOmniInputsProcessor {
    patch_size: usize,
    merge: usize,
    min_patches: usize,
    max_patches: usize,
    mean: Vec<f32>,
    std: Vec<f32>,
    image_token: u32,
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

// the sequence's images, patches concatenated in prompt order, and each one's patch grid
struct Preprocessed {
    patches: Tensor,
    grids: Vec<(usize, usize)>,
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
    pub fn new(cfg: &Config) -> Self {
        Self {
            patch_size: cfg.patch_size,
            merge: cfg.merge(),
            min_patches: cfg.vision_config.min_num_patches,
            max_patches: cfg.vision_config.max_num_patches,
            mean: cfg.norm_mean.clone(),
            std: cfg.norm_std.clone(),
            image_token: cfg.img_context_token_id,
            sound: cfg
                .sound_config
                .as_ref()
                .zip(cfg.sound_context_token_id)
                .map(|(config, token)| SoundInputs {
                    mel: NemoMel::new(config.mel()),
                    config: config.clone(),
                    token,
                }),
        }
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

    /// One image's normalized `(rows * cols, 3 * patch * patch)` patches and its grid.
    pub fn patches(&self, image: &DynamicImage) -> (Vec<f32>, (usize, usize)) {
        let rgb = image.to_rgb8();
        let (width, height) = (rgb.width() as usize, rgb.height() as usize);
        let (rows, cols) = target_grid(
            (width, height),
            self.patch_size,
            self.merge,
            (self.min_patches, self.max_patches),
        );
        let p = self.patch_size;
        let (out_h, out_w) = (rows * p, cols * p);
        let mut planes = vec![0f32; CHANNELS * height * width];
        for (i, pixel) in rgb.pixels().enumerate() {
            for c in 0..CHANNELS {
                planes[c * height * width + i] = f32::from(pixel[c]);
            }
        }
        if (out_h, out_w) != (height, width) {
            planes = inference_vision::aa_bicubic_resize_planar(
                &planes,
                CHANNELS,
                (height, width),
                (out_h, out_w),
            );
        }
        let mut patches = Vec::with_capacity(planes.len());
        for row in 0..rows {
            for col in 0..cols {
                for c in 0..CHANNELS {
                    let plane = &planes[c * out_h * out_w..][..out_h * out_w];
                    for py in 0..p {
                        let line = &plane[(row * p + py) * out_w + col * p..][..p];
                        patches.extend(
                            line.iter()
                                .map(|&v| (v / PIXEL_SCALE - self.mean[c]) / self.std[c]),
                        );
                    }
                }
            }
        }
        (patches, (rows, cols))
    }

    fn tokens(&self, (rows, cols): (usize, usize)) -> usize {
        rows * cols / (self.merge * self.merge)
    }

    fn cached_or_preprocess(&self, seq: &mut dyn MediaSequence) -> Result<Preprocessed> {
        let mm = seq.multimodal();
        if let (Some(patches), Some(grids)) = (&mm.cached_pixel_values, &mm.cached_img_thw) {
            let grids = grids
                .to_vec2::<u32>()?
                .into_iter()
                .map(|g| (g[1] as usize, g[2] as usize))
                .collect();
            return Ok(Preprocessed {
                patches: patches.clone(),
                grids,
            });
        }
        let images = seq.clone_images().expect("Need images by this point.");
        let mut all = Vec::new();
        let mut grids = Vec::with_capacity(images.len());
        for image in &images {
            let (patches, grid) = self.patches(image);
            all.extend(patches);
            grids.push(grid);
        }
        let width = CHANNELS * self.patch_size * self.patch_size;
        let patches = Tensor::from_vec(all, ((), width), &Device::Cpu)?;
        let rows = grids
            .iter()
            .flat_map(|&(r, c)| [1, r as u32, c as u32])
            .collect::<Vec<_>>();
        let mm = seq.multimodal_mut();
        mm.cached_pixel_values = Some(patches.clone());
        mm.cached_img_thw = Some(Tensor::from_vec(rows, (grids.len(), 3), &Device::Cpu)?);
        Ok(Preprocessed { patches, grids })
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
        let grids = images.map_or(&[][..], |images| &images.grids);
        let mut expanded = seq.get_initial_prompt().to_string();
        let image_counts = grids
            .iter()
            .map(|&grid| self.tokens(grid))
            .collect::<Vec<_>>();
        expanded = expand(
            &expanded,
            (IMAGE_TOKEN, IMAGE_START, IMAGE_END),
            &image_counts,
            "images",
        )?;
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
            if !grids.is_empty() {
                let hashes = seq
                    .image_hashes()
                    .ok_or_else(|| anyhow::anyhow!("{MODEL} images are missing content hashes"))?;
                let ranges = delimited(self.image_token, IMAGE_START, IMAGE_END)?;
                layout.extend_ranges(
                    &ranges,
                    hashes,
                    MultimodalKind::Image,
                    MultimodalAttentionPolicy::Causal,
                );
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

    // the images in `range`, each its own patches
    fn select(
        processed: &Preprocessed,
        range: std::ops::Range<usize>,
    ) -> Result<Vec<ImagePatches>> {
        let mut offset = processed.grids[..range.start]
            .iter()
            .map(|(r, c)| r * c)
            .sum::<usize>();
        processed.grids[range]
            .iter()
            .map(|&(rows, cols)| {
                let patches = processed.patches.narrow(0, offset, rows * cols)?;
                offset += rows * cols;
                Ok(ImagePatches {
                    patches,
                    rows,
                    cols,
                })
            })
            .collect()
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
            if !(seq.has_images() || seq.has_audios()) || seq.multimodal().has_changed_prompt {
                continue;
            }
            let processed = if seq.has_images() {
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
                .filter(|seq| seq.has_images() || seq.has_audios())
            {
                let processed = if seq.has_images() {
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
                let count = processed.grids.len();
                let range = if seq.is_chunked_prefill_view() {
                    seq.active_local_multimodal_item_range(MultimodalKind::Image, count)
                        .ok_or_else(|| {
                            anyhow::anyhow!("{MODEL} image chunk is missing its active image range")
                        })?
                } else {
                    seq.count_prefix_cached_mm_items_by_kind(MultimodalKind::Image)..count
                };
                images.extend(Self::select(&processed, range)?);
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
    use super::target_grid;

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

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, ops::Range, sync::Arc};

use image::{DynamicImage, RgbImage, imageops::FilterType};
use inference_tensor::{Device, Result, Tensor};
use tokenizers::Tokenizer;

use crate::device_map::DeviceMapper;
use crate::inkling2_vision::VisionConfig;
use crate::media_inputs::{
    media::{build_mm_features_from_ranges, find_image_placeholder_ranges},
    processor::{
        InnerInputProcessorOutput, InputMetadata, InputProcessorOutput, InputsHost,
        InputsProcessorValidationError, MediaSequence, ModelInputs, MultimodalInputsProcessor,
        TextInputs,
    },
};
use crate::model::recurrent_batch_kind_for_input;
use crate::paged_attention::{PagedAttentionMeta, block_hash::MultimodalKind};

use super::{Config, IMAGE_PAD_TOKEN, Inkling2VlArgs};

// the reference processor's transparency backdrop: 8 px squares of these grays
const CHECKER_SQUARE: u32 = 8;
const CHECKER_LIGHT: u32 = 204;
const CHECKER_DARK: u32 = 153;
const OPAQUE: u32 = 255;

/// Every picture's patches, `[patches, 1200]` each, and its native grid.
struct Preprocessed {
    patches: Vec<Tensor>,
    grids: Vec<(usize, usize)>,
}

pub struct Inkling2VlInputsProcessor {
    vision: VisionConfig,
    image_token_id: u32,
    max_cells: usize,
}

// alpha over the checkerboard, rounded to nearest (vLLM `_composite_on_checkerboard`)
fn to_rgb(image: &DynamicImage) -> RgbImage {
    if !image.color().has_alpha() {
        return image.to_rgb8();
    }
    let rgba = image.to_rgba8();
    RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let p = rgba.get_pixel(x, y).0;
        let a = u32::from(p[3]);
        let light = (x / CHECKER_SQUARE + y / CHECKER_SQUARE).is_multiple_of(2);
        let bg = if light { CHECKER_LIGHT } else { CHECKER_DARK };
        let blend = |c: u8| ((u32::from(c) * a + bg * (OPAQUE - a) + OPAQUE / 2) / OPAQUE) as u8;
        image::Rgb([blend(p[0]), blend(p[1]), blend(p[2])])
    })
}

impl Inkling2VlInputsProcessor {
    pub fn new(config: &Config, max_cells: usize) -> Self {
        Self {
            vision: config.vision.clone(),
            image_token_id: config.image_token_id,
            max_cells,
        }
    }

    // read at its own size unless its cells pass the cap: then scaled, aspect kept, to the most whole cells that fit
    fn fit(&self, image: RgbImage) -> RgbImage {
        let (w, h) = (image.width() as usize, image.height() as usize);
        let (cols, rows) = self.vision.cells(w, h);
        if cols * rows <= self.max_cells {
            return image;
        }
        let cell = self.vision.cell_size();
        let scale = (self.max_cells as f64 / (cols * rows) as f64).sqrt();
        let mut cols = ((w as f64 * scale / cell as f64).floor() as usize).max(1);
        let mut rows = ((h as f64 * scale / cell as f64).floor() as usize).max(1);
        while cols * rows > self.max_cells {
            if cols >= rows { cols -= 1 } else { rows -= 1 }
        }
        image::imageops::resize(
            &image,
            (cols * cell) as u32,
            (rows * cell) as u32,
            FilterType::CatmullRom,
        )
    }

    fn preprocess(&self, images: &[DynamicImage], device: &Device) -> Result<Preprocessed> {
        let mut patches = Vec::with_capacity(images.len());
        let mut grids = Vec::with_capacity(images.len());
        for image in images {
            let rgb = self.fit(to_rgb(image));
            let (w, h) = (rgb.width() as usize, rgb.height() as usize);
            let p = self.vision.patches(rgb.as_raw(), w, h);
            let rows = p.gh * p.gw;
            patches.push(Tensor::from_vec(
                p.values,
                (rows, self.vision.patch_features()),
                device,
            )?);
            grids.push((p.gh, p.gw));
        }
        Ok(Preprocessed { patches, grids })
    }

    fn cells(&self, (gh, gw): (usize, usize)) -> usize {
        let m = self.vision.adapter_merge_size;
        gh * gw / (m * m)
    }

    fn cached_or_preprocess(
        &self,
        seq: &mut dyn MediaSequence,
        device: &Device,
    ) -> Result<Arc<Preprocessed>> {
        if let Some(cached) = seq
            .multimodal()
            .cached_media
            .clone()
            .and_then(|cached| cached.downcast::<Preprocessed>().ok())
        {
            return Ok(cached);
        }
        let images = seq
            .take_images()
            .expect("Need to have images by this point.");
        let processed = Arc::new(self.preprocess(&images, device)?);
        seq.multimodal_mut().cached_media = Some(processed.clone());
        Ok(processed)
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
        let placeholders = prompt.matches(IMAGE_PAD_TOKEN).count();
        if placeholders != processed.grids.len() {
            return Err(InputsProcessorValidationError(format!(
                "The number of `{IMAGE_PAD_TOKEN}` tokens ({placeholders}) must match the number of images ({})",
                processed.grids.len()
            ))
            .into());
        }
        let counts = processed
            .grids
            .iter()
            .map(|&grid| self.cells(grid))
            .collect::<Vec<_>>();
        let mut parts = prompt.split(IMAGE_PAD_TOKEN);
        let mut expanded = parts.next().unwrap_or_default().to_string();
        for (count, part) in counts.iter().zip(parts) {
            expanded.push_str(&IMAGE_PAD_TOKEN.repeat(*count));
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
                .ok_or_else(|| anyhow::Error::msg("Inkling 2 images are missing content hashes"))?
                .to_vec();
            // adjacent pictures would share one run of pads, split by each one's count
            let mut ranges = Vec::with_capacity(counts.len());
            let mut runs = find_image_placeholder_ranges(&ids, self.image_token_id).into_iter();
            let mut run = runs.next();
            for &count in &counts {
                let Some((start, len)) = run else {
                    anyhow::bail!("Inkling 2 prompt has fewer image pads than its images");
                };
                if len < count {
                    anyhow::bail!("Inkling 2 prompt cuts an image's {count} pads short");
                }
                ranges.push((start, count));
                run = if len == count {
                    runs.next()
                } else {
                    Some((start + count, len - count))
                };
            }
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

    fn select(
        processed: &Preprocessed,
        range: Range<usize>,
    ) -> Result<Vec<(Tensor, (usize, usize))>> {
        if range.start > range.end || range.end > processed.grids.len() {
            inference_tensor::bail!(
                "Inkling 2 image range {range:?} exceeds {} preprocessed images",
                processed.grids.len()
            );
        }
        Ok(range
            .map(|i| (processed.patches[i].clone(), processed.grids[i]))
            .collect())
    }
}

impl MultimodalInputsProcessor for Inkling2VlInputsProcessor {
    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut dyn MediaSequence],
        device: &Device,
        _other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("Inkling 2 image processor requires a tokenizer");
        };
        for seq in input_seqs {
            if !seq.has_images() || seq.multimodal().has_changed_prompt {
                continue;
            }
            let processed = self.cached_or_preprocess(&mut **seq, device)?;
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
        _other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> anyhow::Result<InputProcessorOutput> {
        if no_kv_cache {
            anyhow::bail!("Vision model must have kv cache.");
        }
        let mut patches = Vec::new();
        let mut args = Inkling2VlArgs::default();
        if is_prompt {
            let Some(tokenizer) = tokenizer else {
                anyhow::bail!("Inkling 2 image processor requires a tokenizer");
            };
            for seq in input_seqs.iter_mut().filter(|seq| seq.has_images()) {
                let processed = self.cached_or_preprocess(&mut **seq, device)?;
                self.maybe_expand_prompt(
                    &tokenizer,
                    &mut **seq,
                    &processed,
                    paged_attn_metadata.as_mut(),
                )?;
                let count = processed.grids.len();
                let range = if seq.is_chunked_prefill_view() {
                    seq.active_local_multimodal_item_range(MultimodalKind::Image, count)
                        .ok_or_else(|| {
                            inference_tensor::Error::msg(
                                "Inkling 2 image chunk is missing its active image range",
                            )
                        })?
                } else {
                    seq.count_prefix_cached_mm_items_by_kind(MultimodalKind::Image)..count
                };
                for (values, grid) in Self::select(&processed, range)? {
                    patches.push(values);
                    args.grids.push(grid);
                }
            }
        }
        // a verify step stays one, for the text model's rollback of its conv states
        let recurrent_batch_kind = recurrent_batch_kind_for_input(
            is_prompt,
            host.staged_batch_width(input_seqs).is_some(),
        );
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
        } = host.step_inputs(toks, input_seqs, text, is_prompt, no_kv_cache)?;
        Ok(InputProcessorOutput {
            inputs: Box::new(ModelInputs {
                input_ids: input,
                seqlen_offsets: positions,
                context_lens,
                position_ids,
                pixel_values: if patches.is_empty() {
                    None
                } else {
                    Some(Tensor::cat(&patches, 0)?)
                },
                model_specific_args: Box::new(args),
                paged_attn_meta,
                flash_meta,
                recurrent_batch_kind,
            }),
            seq_indices,
        })
    }
}

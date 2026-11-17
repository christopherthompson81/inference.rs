#![allow(clippy::cast_possible_truncation)]

use std::{any::Any, ops::Range, sync::Arc};

use image::{DynamicImage, RgbImage};
use inference_tensor::{Device, Result, Tensor};
use tokenizers::Tokenizer;

use crate::device_map::DeviceMapper;
use crate::media_inputs::{
    media::{build_mm_features_from_ranges, find_image_placeholder_ranges},
    preprocessor_config::PreProcessorConfig,
    processor::{
        InnerInputProcessorOutput, InputMetadata, InputProcessorOutput, InputsHost,
        InputsProcessorValidationError, MediaSequence, ModelInputs, MultimodalInputsProcessor,
        TextInputs,
    },
};
use crate::paged_attention::{PagedAttentionMeta, block_hash::MultimodalKind};

use super::preprocess::image_views;
use super::{Config, DeepseekOcrSpecificArgs};

pub const IMAGE_TOKEN: &str = "<image>";
// `BasicImageTransform`'s normalization
const MEAN: [f64; 3] = [0.5, 0.5, 0.5];
const STD: [f64; 3] = [0.5, 0.5, 0.5];
const PIXEL_SCALE: f64 = 255.0;

/// Every image's views as tensors: the stacked global views, each image's tiles and its grid.
struct Preprocessed {
    globals: Tensor,
    tiles: Vec<Option<Tensor>>,
    grids: Vec<(usize, usize)>,
}

pub struct DeepseekOcrInputsProcessor {
    config: Config,
}

// (3, H, W) pixels normalized as `ToTensor` then `Normalize`
fn to_tensor(image: &RgbImage, mean: [f64; 3], std: [f64; 3], device: &Device) -> Result<Tensor> {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let mut planes = vec![0f32; 3 * h * w];
    for (i, pixel) in image.pixels().enumerate() {
        for c in 0..3 {
            planes[c * h * w + i] =
                ((f64::from(pixel.0[c]) / PIXEL_SCALE - mean[c]) / std[c]) as f32;
        }
    }
    Tensor::from_vec(planes, (3, h, w), device)
}

impl DeepseekOcrInputsProcessor {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
        device: &Device,
    ) -> Result<Preprocessed> {
        let mean = config.image_mean.unwrap_or(MEAN);
        let std = config.image_std.unwrap_or(STD);
        let mut globals = Vec::with_capacity(images.len());
        let mut tiles = Vec::with_capacity(images.len());
        let mut grids = Vec::with_capacity(images.len());
        for image in images {
            let views = image_views(
                image,
                self.config.vision_config.image_size,
                self.config.max_tiles(),
            );
            globals.push(to_tensor(&views.global, mean, std, device)?);
            tiles.push(if views.tiles.is_empty() {
                None
            } else {
                let tiles = views
                    .tiles
                    .iter()
                    .map(|tile| to_tensor(tile, mean, std, device))
                    .collect::<Result<Vec<_>>>()?;
                Some(Tensor::stack(&tiles, 0)?)
            });
            grids.push(views.grid);
        }
        Ok(Preprocessed {
            globals: Tensor::stack(&globals, 0)?,
            tiles,
            grids,
        })
    }

    fn cached_or_preprocess(
        &self,
        seq: &mut dyn MediaSequence,
        config: &PreProcessorConfig,
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
        let processed = Arc::new(self.preprocess(&images, config, device)?);
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
        let placeholders = prompt.matches(IMAGE_TOKEN).count();
        if placeholders != processed.grids.len() {
            return Err(InputsProcessorValidationError(format!(
                "The number of `{IMAGE_TOKEN}` tokens ({placeholders}) must match the number of images ({})",
                processed.grids.len()
            ))
            .into());
        }
        let counts = processed
            .grids
            .iter()
            .map(|&grid| self.config.image_token_count(grid))
            .collect::<Vec<_>>();
        let mut parts = prompt.split(IMAGE_TOKEN);
        let mut expanded = parts.next().unwrap_or_default().to_string();
        for (count, part) in counts.iter().zip(parts) {
            expanded.push_str(&IMAGE_TOKEN.repeat(*count));
            expanded.push_str(part);
        }
        seq.set_initial_prompt(expanded.clone());
        let mut ids = tokenizer
            .encode_fast(expanded, false)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?
            .get_ids()
            .to_vec();
        // the reference prepends BOS whatever the template wrote (Unlimited-OCR's writes none)
        if let Some(bos) = self.config.language_config.bos_token_id
            && ids.first() != Some(&bos)
        {
            ids.insert(0, bos);
        }
        if seq.mm_features().is_empty() {
            let hashes = seq
                .image_hashes()
                .ok_or_else(|| {
                    anyhow::Error::msg("DeepSeek-OCR images are missing content hashes")
                })?
                .to_vec();
            // adjacent images share one run of the token, split by each image's count
            let mut ranges = Vec::with_capacity(counts.len());
            let mut runs =
                find_image_placeholder_ranges(&ids, self.config.image_token_id).into_iter();
            let mut run = runs.next();
            for &count in &counts {
                let Some((start, len)) = run else {
                    anyhow::bail!("DeepSeek-OCR prompt has fewer image tokens than its images");
                };
                if len < count {
                    anyhow::bail!("DeepSeek-OCR prompt cuts an image's {count} tokens short");
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
        hashes: &[u64],
    ) -> Result<Option<(Tensor, DeepseekOcrSpecificArgs)>> {
        if range.start > range.end || range.end > processed.grids.len() {
            inference_tensor::bail!(
                "DeepSeek-OCR image range {range:?} exceeds {} preprocessed images",
                processed.grids.len()
            );
        }
        if range.is_empty() {
            return Ok(None);
        }
        Ok(Some((
            processed.globals.narrow(0, range.start, range.len())?,
            DeepseekOcrSpecificArgs {
                tiles: processed.tiles[range.clone()].to_vec(),
                grids: processed.grids[range.clone()].to_vec(),
                image_hashes: hashes.get(range).map(<[u64]>::to_vec).unwrap_or_default(),
            },
        )))
    }
}

impl MultimodalInputsProcessor for DeepseekOcrInputsProcessor {
    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut dyn MediaSequence],
        device: &Device,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("DeepSeek-OCR image processor requires a tokenizer");
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
            anyhow::bail!("DeepSeek-OCR image processor requires a tokenizer");
        };
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let mut globals = Vec::new();
        let mut args = DeepseekOcrSpecificArgs::default();
        if is_prompt {
            for seq in input_seqs.iter_mut().filter(|seq| seq.has_images()) {
                let processed = self.cached_or_preprocess(&mut **seq, config, device)?;
                self.maybe_expand_prompt(
                    &tokenizer,
                    &mut **seq,
                    &processed,
                    paged_attn_metadata.as_mut(),
                )?;
                let image_count = processed.grids.len();
                let range = if seq.is_chunked_prefill_view() {
                    seq.active_local_multimodal_item_range(MultimodalKind::Image, image_count)
                        .ok_or_else(|| {
                            inference_tensor::Error::msg(
                                "DeepSeek-OCR image chunk is missing its active image range",
                            )
                        })?
                } else {
                    seq.count_prefix_cached_mm_items_by_kind(MultimodalKind::Image)..image_count
                };
                let hashes = seq.image_hashes().map(<[u64]>::to_vec).unwrap_or_default();
                if let Some((views, selected)) = Self::select(&processed, range, &hashes)? {
                    globals.push(views);
                    args.tiles.extend(selected.tiles);
                    args.grids.extend(selected.grids);
                    args.image_hashes.extend(selected.image_hashes);
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
        // hashes key the encoder cache only when every image in the pass has one
        if args.image_hashes.len() != args.grids.len() {
            args.image_hashes.clear();
        }

        Ok(InputProcessorOutput {
            inputs: Box::new(ModelInputs {
                input_ids: input,
                seqlen_offsets: positions,
                context_lens,
                position_ids,
                pixel_values: if globals.is_empty() {
                    None
                } else {
                    Some(Tensor::cat(&globals, 0)?)
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

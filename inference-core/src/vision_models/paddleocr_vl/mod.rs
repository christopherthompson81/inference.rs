//! PaddleOCR-VL: SigLIP/NaViT tower + `mlp_AR` connector + ERNIE-4.5-0.3B LM.
//! Apache-2.0 PaddleOCR-VL (https://github.com/PaddlePaddle/PaddleOCR); spec: transformers `modeling_paddleocr_vl`.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub mod config;
pub mod connector;
pub mod inputs_processor;
pub mod merge;
pub mod preprocess;
pub mod rope_index;
pub mod text;
pub mod vision;

use std::any::Any;

use candle_core::{Device, Result, Tensor};
use inference_quant::ShardedVarBuilder;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::amoe::AnyMoeBaseModelMixin;
use crate::layers::CausalMasker;
use crate::layers_masker::{CausalMaskConfig, PastKvLenCache};
use crate::paged_attention::encoder_cache::{CacheModality, EncoderCacheManager};
use crate::paged_attention::{AttentionImplementation, KvCacheLayout, ModelConfigMetadata};
use crate::pipeline::{
    EitherCache, IsqModel, ModelForwardContext, MultimodalModel, NormalCache, NormalLoadingMetadata,
};

// One OCR page is a few hundred KB of connector embeds.
const ENCODER_CACHE_ENTRIES: usize = 32;

use config::Config;
use connector::Connector;
use merge::Merger;
use rope_index::get_rope_index_batched;
use text::ErnieTextModel;
use vision::VisionModel;

pub struct PaddleOcrVlModel {
    vision: VisionModel,
    connector: Connector,
    merger: Merger,
    text: ErnieTextModel,
    cfg: Config,
    device: Device,
    max_seq_len: usize,
    config_meta: ModelConfigMetadata,
    cache: EitherCache,
    // Preempted seqs re-prefill the whole prompt; keyed by image hash so the tower isn't re-run.
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
    encoder_cache_hits: Arc<AtomicUsize>,
    encoder_cache_misses: Arc<AtomicUsize>,
}

impl PaddleOcrVlModel {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let tcfg = cfg.text_config();
        let vcfg = cfg.vision_config();
        // Non-ISQ parts go on the real device, not ISQ's cpu staging device, else activations hit cuda on cpu.
        let real_dev = normal_loading_metadata.real_device.clone();
        let vision = VisionModel::load(
            vb.pp("visual")
                .pp("vision_model")
                .set_device(real_dev.clone()),
            &vcfg,
        )?;
        let connector = Connector::load(
            vb.pp("mlp_AR").set_device(real_dev.clone()),
            vcfg.hidden_size,
            vcfg.spatial_merge_size,
            tcfg.hidden_size,
        )?;
        let merger = Merger::load(
            vb.pp("model").set_device(real_dev.clone()),
            tcfg.vocab_size,
            tcfg.hidden_size,
            cfg.image_token_id as i64,
        )?;
        let device = real_dev;
        let text = ErnieTextModel::load(
            vb,
            &tcfg,
            normal_loading_metadata.mapper,
            device.clone(),
            normal_loading_metadata.loading_isq,
            attention_mechanism,
        )?;
        // No tensor-parallel sharding, so head counts are unsharded.
        let config_meta = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: tcfg.num_hidden_layers,
            hidden_size: tcfg.hidden_size,
            num_attn_heads: tcfg.num_attention_heads,
            num_kv_heads: tcfg.num_key_value_heads,
            sliding_window: None,
            k_head_dim: tcfg.head_dim,
            v_head_dim: tcfg.head_dim,
            kv_cache_layout: KvCacheLayout::Standard,
        };
        Ok(Self {
            vision,
            connector,
            merger,
            text,
            cfg: cfg.clone(),
            device,
            max_seq_len: cfg.max_position_embeddings,
            config_meta,
            cache: EitherCache::Normal(NormalCache::new(
                tcfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(ENCODER_CACHE_ENTRIES))),
            encoder_cache_hits: Arc::new(AtomicUsize::new(0)),
            encoder_cache_misses: Arc::new(AtomicUsize::new(0)),
        })
    }
}

impl AnyMoeBaseModelMixin for PaddleOcrVlModel {}
impl crate::speculative::SpeculativeTargetMixin for PaddleOcrVlModel {}
impl crate::block_diffusion::BlockDiffusionMixin for PaddleOcrVlModel {}

impl IsqModel for PaddleOcrVlModel {
    // Only the ERNIE LM projections + lm_head are ISQ targets; everything else is residual.
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let mut tensors = self.text.residual_tensors();
        tensors.extend(self.merger.residual_tensors());
        tensors.extend(self.connector.residual_tensors());
        tensors.extend(self.vision.residual_tensors());
        tensors
    }
}

pub(crate) struct PaddleOcrVlVisionSpecificArgs {
    // Whole prompt per row, so mrope positions/deltas recompute statelessly every step (like qwen3_vl).
    pub input_ids_full: Tensor,
    // Per-row image grids, needed on every pass for mrope; empty only when no row has an image.
    pub image_grid_thw: Vec<Vec<(usize, usize, usize)>>,
    pub image_hashes: Vec<Vec<u64>>,
    // Rows whose patches are in `pixel_values` this pass, in concat order; empty on decode.
    pub vision_rows: Vec<usize>,
}

// Images whose placeholder runs this pass embeds: a prefix hit or a later prefill chunk can start past the first image.
fn window_images(
    full_ids: &[u32],
    offset: usize,
    window_image_tokens: usize,
    image_token_id: u32,
) -> Result<std::ops::Range<usize>> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < full_ids.len() {
        if full_ids[i] != image_token_id {
            i += 1;
            continue;
        }
        let start = i;
        while i < full_ids.len() && full_ids[i] == image_token_id {
            i += 1;
        }
        runs.push((start, i - start));
    }
    let first = runs
        .iter()
        .position(|&(start, _)| start >= offset)
        .unwrap_or(runs.len());
    let (mut tokens, mut end) = (0, first);
    while tokens < window_image_tokens && end < runs.len() {
        tokens += runs[end].1;
        end += 1;
    }
    if tokens != window_image_tokens {
        candle_core::bail!(
            "{window_image_tokens} image tokens in this pass do not line up with whole images after position {offset}"
        );
    }
    Ok(first..end)
}

impl MultimodalModel for PaddleOcrVlModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let PaddleOcrVlVisionSpecificArgs {
            input_ids_full,
            image_grid_thw,
            image_hashes,
            vision_rows,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `PaddleOcrVlVisionSpecificArgs`");

        let dev = &self.device;
        let merge = self.cfg.vision_config().spatial_merge_size;
        let image_token_id = self.cfg.image_token_id as i64;
        let seqlen_offsets = ctx.seqlen_offsets();

        let (batch, _full_len) = input_ids_full.dims2()?;
        let grids: Vec<Vec<(usize, usize, usize)>> = if image_grid_thw.is_empty() {
            vec![Vec::new(); batch]
        } else {
            image_grid_thw.clone()
        };
        let (full_pos, deltas) =
            get_rope_index_batched(&input_ids_full, &grids, image_token_id, merge, dev)?;
        let position_ids = crate::vision_models::mrope_position_ids_for_input(
            &full_pos,
            &deltas,
            input_ids,
            seqlen_offsets,
        )?;

        // `pixel_values` is every vision row's images concatenated on dim 0, split back by t*h*w.
        let embeds = if vision_rows.is_empty() {
            self.merger.embed_tokens(input_ids)?
        } else {
            let pv = pixel_values.expect("vision rows without pixel values");
            let text = self.merger.embed_tokens(input_ids)?;
            let mut rows = (0..batch)
                .map(|b| text.narrow(0, b, 1)?.squeeze(0))
                .collect::<Result<Vec<_>>>()?;
            let mut row_start = 0;
            for &b in &vision_rows {
                let row_grids = &image_grid_thw[b];
                let row_ids = input_ids.narrow(0, b, 1)?.flatten_all()?;
                let window_image_tokens = row_ids
                    .to_vec1::<u32>()?
                    .iter()
                    .filter(|&&id| id as i64 == image_token_id)
                    .count();
                let full_ids = input_ids_full
                    .narrow(0, b, 1)?
                    .flatten_all()?
                    .to_vec1::<u32>()?;
                let offset_in_prompt = seqlen_offsets.get(b).copied().unwrap_or(0);
                let active = window_images(
                    &full_ids,
                    offset_in_prompt,
                    window_image_tokens,
                    image_token_id as u32,
                )?;
                let patches_of = |&(t, h, w): &(usize, usize, usize)| t * h * w;
                let mut offset = row_start
                    + row_grids[..active.start]
                        .iter()
                        .map(patches_of)
                        .sum::<usize>();
                row_start += row_grids.iter().map(patches_of).sum::<usize>();
                if active.is_empty() {
                    continue;
                }
                let mut embeds_per_image = Vec::with_capacity(active.len());
                for (i, &(t, h, w)) in row_grids
                    .iter()
                    .enumerate()
                    .skip(active.start)
                    .take(active.len())
                {
                    let key = image_hashes.get(b).and_then(|hs| hs.get(i)).copied();
                    let hit = key.and_then(|k| {
                        let mut guard = self.encoder_cache.lock().expect("encoder cache poisoned");
                        guard.get(CacheModality::Image, k).map(|out| out[0].clone())
                    });
                    let patches = t * h * w;
                    if let Some(embeds) = hit {
                        self.encoder_cache_hits.fetch_add(1, Ordering::Relaxed);
                        offset += patches;
                        embeds_per_image.push(embeds);
                        continue;
                    }
                    self.encoder_cache_misses.fetch_add(1, Ordering::Relaxed);
                    let post_ln = self
                        .vision
                        .forward(&pv.narrow(0, offset, patches)?, t, h, w)?;
                    offset += patches;
                    let embeds = self.connector.forward(&post_ln, t, h, w)?;
                    if let Some(k) = key {
                        self.encoder_cache
                            .lock()
                            .expect("encoder cache poisoned")
                            .insert(CacheModality::Image, k, vec![embeds.clone()]);
                    }
                    embeds_per_image.push(embeds);
                }
                let image_embeds = Tensor::cat(&embeds_per_image, 0)?;
                rows[b] = self.merger.forward(&row_ids, &image_embeds)?;
            }
            Tensor::stack(&rows, 0)?
        };

        let mask = CausalMasker.make_causal_mask(
            input_ids,
            &seqlen_offsets as &dyn PastKvLenCache,
            embeds.dtype(),
            &CausalMaskConfig::default(),
        )?;
        // Keep the mask on later prompt chunks: paged prefix gather reads causality from it, else attends non-causally.

        let mut guard = self.cache.normal();
        let paged = ctx.paged_metadata();
        let paged_ref = paged.as_ref().map(|(kv, meta)| (kv.as_slice(), *meta));
        let logits = self
            .text
            .forward(
                &embeds,
                &position_ids,
                &mut guard.0,
                &mask,
                paged_ref,
                Some(ctx.flash_params()),
            )?
            .logits;
        ctx.logits(&logits)
    }

    fn device(&self) -> &Device {
        &self.device
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.config_meta
    }
    fn encoder_cache_counters(&self) -> Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)> {
        Some((
            self.encoder_cache_hits.clone(),
            self.encoder_cache_misses.clone(),
        ))
    }
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(PaddleOcrVlVisionSpecificArgs {
            input_ids_full: input_ids.clone(),
            image_grid_thw: Vec::new(),
            image_hashes: Vec::new(),
            vision_rows: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::window_images;

    const IMG: u32 = 9;
    // text, image 0 (3 tokens), text, text, image 1 (2 tokens), text
    const FULL: &[u32] = &[5, IMG, IMG, IMG, 6, 5, IMG, IMG, 6];

    #[test]
    fn window_images_follow_the_window_offset() -> candle_core::Result<()> {
        assert_eq!(window_images(FULL, 0, 5, IMG)?, 0..2);
        // prefix hit or later chunk starting after image 0 must embed image 1, not image 0
        assert_eq!(window_images(FULL, 5, 2, IMG)?, 1..2);
        assert!(window_images(FULL, 5, 0, IMG)?.is_empty());
        assert!(window_images(FULL, 0, 4, IMG).is_err());
        Ok(())
    }
}

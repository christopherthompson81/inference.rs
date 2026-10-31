#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::attention::AttentionMask;
use crate::layers::masker::CausalMaskConfig;
use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use crate::qwen2vl::Qwen2VLVisionSpecificArgs;
use inference_quant::ShardedVarBuilder;
use inference_tensor::{Device, Result, Tensor};
pub use text::Qwen3_5TextModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    kv_cache::EitherCache,
    layers::CausalMasker,
    layers::masker::PastKvLenCache,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{
        AttentionImplementation, ModelConfigLike, ModelConfigMetadata,
        encoder_cache::{CacheModality, EncoderCacheManager},
    },
    qwen3_vl::{VisionTokens, encode_media, merge_visual_deepstack, splice_media},
    vision::qwen3_vl::Qwen3VLVisionModel,
};

pub mod config;
mod feed_forward;
mod hyper;
pub mod joint_head;
pub mod mtp;
pub mod packed_gdn;
pub mod packed_visual;
mod ple;
mod qsa;
pub mod speculative;
pub mod text;

pub use config::{Config, TextConfig};
use packed_visual::{PackedVisualEncoder, PackedVisualInput};

pub struct Qwen3_5Model {
    pub(super) text: Qwen3_5TextModel,
    vision: Qwen3VLVisionModel,
    tokens: VisionTokens,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Qwen3_5Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        // Support both original HuggingFace naming (model.visual.*) and MLX naming (vision_tower.*)
        let vision_vb = if vb.contains_tensor("vision_tower.patch_embed.proj.weight") {
            vb.pp("vision_tower")
        } else {
            vb.pp("model").pp("visual")
        }
        .without_lora_registry();
        let vision = Qwen3VLVisionModel::new(
            &cfg.vision_config,
            vision_vb.set_device(normal_loading_metadata.real_device.clone()),
        )?;
        // Use top-level quantization_config if present, otherwise fall back to text_config's
        let mut text_config = cfg.text_config.clone();
        if cfg.quantization_config.is_some() {
            text_config.quantization_config = cfg.quantization_config.clone();
        }
        let text = Qwen3_5TextModel::new(
            &text_config,
            vb.clone(),
            cfg.tie_word_embeddings,
            cfg.mtp,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text,
            vision,
            tokens: VisionTokens {
                spatial_merge_size: cfg.vision_config.spatial_merge_size,
                image_token_id: cfg.image_token_id,
                video_token_id: cfg.video_token_id,
                vision_start_token_id: cfg.vision_start_token_id,
                vision_end_token_id: cfg.vision_end_token_id,
            },
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        args: Qwen2VLVisionSpecificArgs,
        ctx: &ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let Qwen2VLVisionSpecificArgs {
            input_ids_full,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            rope_img_grid_thw,
            rope_vid_grid_thw,
            seqlens,
            continuous_img_pad,
            continuous_vid_pad,
            image_hashes,
            video_hashes,
            packed_layout,
            prompt_position_ids,
        } = args;
        let (input_ids_full, image_hashes, video_hashes) = (
            &input_ids_full,
            image_hashes.as_slice(),
            video_hashes.as_slice(),
        );
        let (packed_layout, prompt_position_ids) =
            (packed_layout.as_ref(), prompt_position_ids.as_ref());
        let seqlen_offsets = ctx.seqlen_offsets();
        // Later chunks and decode rows attend through the paged cache, so only the first chunk needs a mask
        let attention_mask = if ctx.is_first_prompt_chunk() {
            CausalMasker.make_causal_mask(
                input_ids,
                &seqlen_offsets as &dyn PastKvLenCache,
                self.text.dtype,
                &CausalMaskConfig {
                    sliding_window: self.text.cfg.sliding_window,
                    ..Default::default()
                },
            )?
        } else {
            AttentionMask::None
        };

        let input_embeds = self.text.embed_tokens(input_ids)?;
        if let Some(layout) = packed_layout {
            let position_ids = prompt_position_ids.ok_or_else(|| {
                inference_tensor::Error::msg(
                    "packed Qwen3.5 prefill is missing prompt position IDs",
                )
            })?;
            let visual = PackedVisualEncoder::new(
                &self.vision,
                &self.encoder_cache,
                self.tokens.spatial_merge_size,
            )
            .prepare(PackedVisualInput {
                input_embeds,
                pixel_values: pixel_values.as_ref(),
                pixel_values_videos: pixel_values_videos.as_ref(),
                image_grid_thw: image_grid_thw.as_ref(),
                video_grid_thw: video_grid_thw.as_ref(),
                image_hashes,
                video_hashes,
                layout,
            })?;
            return self.text.forward_embeds(
                visual.input_embeds,
                &attention_mask,
                position_ids,
                seqlen_offsets,
                ctx,
                visual.visual_pos_mask.as_ref(),
                visual.deepstack_visual_embeds.as_deref(),
            );
        }
        let mut input_embeds = input_embeds;
        let device = input_embeds.device().clone();
        let encode = |pixel_values: &Tensor, grid_thw: &Tensor, hashes: &[u64], modality| {
            encode_media(
                &self.vision,
                &self.encoder_cache,
                self.tokens.spatial_merge_size,
                pixel_values,
                grid_thw,
                hashes,
                modality,
                (&device, self.text.dtype),
            )
        };

        let mut image = None;
        if let Some(pixel_values) = &pixel_values {
            let Some(grid_thw) = image_grid_thw.as_ref() else {
                inference_tensor::bail!("pixel_values require image_grid_thw");
            };
            let media = encode(pixel_values, grid_thw, image_hashes, CacheModality::Image)?;
            let (embeds, mask) =
                splice_media(input_embeds, &media.embeds, &continuous_img_pad, "Image")?;
            input_embeds = embeds;
            image = Some((mask, media.deepstack));
        }

        let mut video = None;
        if let Some(pixel_values_videos) = &pixel_values_videos {
            let Some(grid_thw) = video_grid_thw.as_ref() else {
                inference_tensor::bail!("pixel_values_videos require video_grid_thw");
            };
            let media = encode(
                pixel_values_videos,
                grid_thw,
                video_hashes,
                CacheModality::Video,
            )?;
            let (embeds, mask) =
                splice_media(input_embeds, &media.embeds, &continuous_vid_pad, "Video")?;
            input_embeds = embeds;
            video = Some((mask, media.deepstack));
        }
        let (visual_pos_masks, deepstack_visual_embeds) = merge_visual_deepstack(image, video)?;

        let position_ids = if rope_img_grid_thw.is_none() && rope_vid_grid_thw.is_none() {
            match crate::vision::mrope::text_decode_position_ids_from_context(input_ids, ctx)? {
                Some(position_ids) => position_ids,
                None => crate::vision::mrope::text_position_ids(input_ids, seqlen_offsets)?,
            }
        } else {
            self.tokens.prompt_position_ids(
                input_ids,
                input_ids_full,
                rope_img_grid_thw.as_ref(),
                rope_vid_grid_thw.as_ref(),
                &seqlens,
                seqlen_offsets,
            )?
        };

        let out = self.text.forward_embeds(
            input_embeds,
            &attention_mask,
            &position_ids,
            seqlen_offsets,
            ctx,
            visual_pos_masks.as_ref(),
            deepstack_visual_embeds.as_deref(),
        )?;
        Ok(out)
    }
}

impl crate::model::BlockDiffusionMixin for Qwen3_5Model {}

impl MultimodalModel for Qwen3_5Model {
    fn supports_packed_prefill(&self) -> bool {
        NormalModel::supports_packed_prefill(&self.text)
    }

    fn bidirectional_prompt(&self) -> bool {
        NormalModel::bidirectional_prompt(&self.text)
    }

    fn joint_decision(
        &self,
        hidden: &Tensor,
        layout: &inference_nn::decision::JointLayout,
    ) -> Result<Vec<f32>> {
        NormalModel::joint_decision(&self.text, hidden, layout)
    }

    fn token_window(&self) -> Option<usize> {
        NormalModel::token_window(&self.text)
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>,
        ctx: &mut crate::model::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut args: Qwen2VLVisionSpecificArgs = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Qwen2VLVisionSpecificArgs`");
        let pixel_values = args.split_pixels(pixel_values);
        self.forward(input_ids, pixel_values, args, ctx)
    }
    fn cache(&self) -> &EitherCache {
        &self.text.cache
    }
    fn device(&self) -> &Device {
        &self.text.device
    }
    fn max_seq_len(&self) -> usize {
        self.text.max_seq_len
    }
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        NormalModel::supports_cuda_decode_graphs(&self.text)
    }
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs_for_args(&self, model_specific_args: &dyn Any) -> bool {
        self.supports_cuda_decode_graphs()
            && model_specific_args
                .downcast_ref::<Qwen2VLVisionSpecificArgs>()
                .is_some()
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.text.cfg
    }
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        self.text.paged_kv_config()
    }
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any> {
        let (batch_size, seq_len) = input_ids.dims2().expect("input ids must be rank 2");
        Box::new(Qwen2VLVisionSpecificArgs {
            input_ids_full: input_ids.clone(),
            pixel_values_videos: None,
            image_grid_thw: None,
            video_grid_thw: None,
            rope_img_grid_thw: None,
            rope_vid_grid_thw: None,
            seqlens: vec![seq_len; batch_size],
            continuous_img_pad: vec![],
            continuous_vid_pad: vec![],
            image_hashes: vec![],
            video_hashes: vec![],
            packed_layout: None,
            prompt_position_ids: None,
        })
    }
    fn encoder_cache(&self) -> Option<&Mutex<EncoderCacheManager>> {
        Some(&self.encoder_cache)
    }
    fn encoder_cache_counters(
        &self,
    ) -> Option<(
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<std::sync::atomic::AtomicUsize>,
    )> {
        Some(
            self.encoder_cache
                .lock()
                .expect("encoder cache poisoned")
                .counters(),
        )
    }
}

impl IsqModel for Qwen3_5Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let mut tensors = self.text.residual_tensors();
        tensors.extend(self.vision.residual_tensors());
        tensors
    }
}

impl AnyMoeBaseModelMixin for Qwen3_5Model {}

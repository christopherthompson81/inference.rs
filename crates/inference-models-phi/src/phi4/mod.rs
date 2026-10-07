#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use inference_quant::ShardedVarBuilder;
use inference_tensor::{Device, Result, Tensor};
use mm_embedding::{InputMode, Phi4MMImageAudioEmbedding, Phi4MMPackedInputs};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::AttentionMask,
    decoder::CausalLm,
    kv_cache::EitherCache,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{
        AttentionImplementation, ModelConfigMetadata, encoder_cache::EncoderCacheManager,
    },
    utils::unvarbuilder::UnVarBuilder,
    vision::multimodal_layout::PackedMultimodalLayout,
};

pub mod audio_embedding;
pub mod config;
pub mod image_embedding;
pub mod inputs_processor;
pub mod mm_embedding;

pub use config::Phi4MMConfig;
pub use image_embedding::PHI4_MM_VISION_CFG;

const ENCODER_CACHE_ENTRIES: usize = 32;

pub struct Phi4MMModel {
    text: CausalLm,
    embed_tokens_extend: Phi4MMImageAudioEmbedding,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Phi4MMModel {
    pub fn new(
        cfg: &Phi4MMConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let text = CausalLm::new(
            &cfg.decoder_spec(),
            vb,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        let embed_tokens_extend = Phi4MMImageAudioEmbedding::new(
            cfg,
            text.embed_tokens().clone(),
            vb_m.dtype(),
            text.stack_mapper()
                .set_nm_device(vb_m.pp("embed_tokens_extend"), false),
        )?;
        Ok(Self {
            text,
            embed_tokens_extend,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(ENCODER_CACHE_ENTRIES))),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_image_embeds: Option<Tensor>,
        image_attention_mask: Option<Tensor>,
        image_sizes: Option<Vec<(u32, u32)>>,
        input_audio_embeds: Option<Tensor>,
        audio_embed_sizes: Option<Vec<usize>>,
        audio_feature_lens: Option<Vec<usize>>,
        audio_vision_modes: Option<Vec<bool>>,
        audio_attention_mask: Option<Tensor>,
        ctx: &mut ModelForwardContext<'_>,
        image_hashes: &[u64],
        audio_hashes: &[u64],
        packed_layout: Option<&PackedMultimodalLayout>,
    ) -> Result<Tensor> {
        let xs = if let Some(packed_layout) = packed_layout {
            self.embed_tokens_extend.forward_packed(
                input_ids,
                Phi4MMPackedInputs {
                    image_embeds: input_image_embeds.as_ref(),
                    image_attention_mask: image_attention_mask.as_ref(),
                    image_sizes: image_sizes.as_deref(),
                    image_hashes,
                    audio_embeds: input_audio_embeds.as_ref(),
                    audio_feature_lens: audio_feature_lens.as_deref(),
                    audio_embed_sizes: audio_embed_sizes.as_deref(),
                    audio_hashes,
                    layout: packed_layout,
                },
                &self.encoder_cache,
            )?
        } else if input_image_embeds.is_some() || input_audio_embeds.is_some() {
            let projection_mode = match (&input_image_embeds, &input_audio_embeds) {
                (Some(_), Some(_)) | (Some(_), None) => InputMode::Vision,
                (None, Some(_)) => InputMode::Speech,
                _ => unreachable!("already know either are some"),
            };

            self.embed_tokens_extend.forward(
                input_ids,
                input_image_embeds.as_ref(),
                &match image_attention_mask.as_ref() {
                    Some(t) => AttentionMask::Custom((*t).clone()),
                    None => AttentionMask::None,
                },
                image_sizes,
                input_audio_embeds.as_ref(),
                audio_embed_sizes,
                audio_vision_modes.as_deref(),
                &match audio_attention_mask.as_ref() {
                    Some(t) => AttentionMask::Custom((*t).clone()),
                    None => AttentionMask::None,
                },
                projection_mode,
                image_hashes,
                &self.encoder_cache,
            )?
        } else {
            return self.text.forward(input_ids, ctx);
        };
        self.text.forward_embeds(input_ids, xs, ctx)
    }
}

#[derive(Default)]
pub struct Phi4MMVisionSpecificArgs {
    pub image_sizes: Option<Vec<(u32, u32)>>,
    pub input_image_embeds: Option<Tensor>,
    pub image_attention_mask: Option<Tensor>,
    pub input_audio_embeds: Option<Tensor>,
    pub audio_embed_sizes: Option<Vec<usize>>,
    pub audio_feature_lens: Option<Vec<usize>>,
    pub audio_vision_modes: Option<Vec<bool>>,
    pub audio_attention_mask: Option<Tensor>,
    pub image_hashes: Vec<u64>,
    pub audio_hashes: Vec<u64>,
    pub packed_layout: Option<PackedMultimodalLayout>,
}

impl crate::speculative::SpeculativeTargetMixin for Phi4MMModel {}

impl crate::model::BlockDiffusionMixin for Phi4MMModel {}

impl MultimodalModel for Phi4MMModel {
    fn supports_packed_prefill(&self) -> bool {
        true
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let Phi4MMVisionSpecificArgs {
            input_image_embeds,
            image_attention_mask,
            image_sizes,
            input_audio_embeds,
            audio_attention_mask,
            audio_embed_sizes,
            audio_feature_lens,
            audio_vision_modes,
            image_hashes,
            audio_hashes,
            packed_layout,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Phi4MMVisionSpecificArgs`");

        self.forward(
            input_ids,
            input_image_embeds,
            image_attention_mask,
            image_sizes,
            input_audio_embeds,
            audio_embed_sizes,
            audio_feature_lens,
            audio_vision_modes,
            audio_attention_mask,
            ctx,
            &image_hashes,
            &audio_hashes,
            packed_layout.as_ref(),
        )
    }
    fn cache(&self) -> &EitherCache {
        NormalModel::cache(&self.text)
    }
    fn device(&self) -> &Device {
        NormalModel::device(&self.text)
    }
    fn max_seq_len(&self) -> usize {
        NormalModel::max_seq_len(&self.text)
    }
    fn config(&self) -> &ModelConfigMetadata {
        NormalModel::config(&self.text)
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(Phi4MMVisionSpecificArgs::default())
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

impl IsqModel for Phi4MMModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("model");
        uvb_m
            .pp("embed_tokens_extend")
            .extend(self.embed_tokens_extend.residual_tensors());
        self.text.residual_tensors_m(uvb_m)
    }
}

impl AnyMoeBaseModelMixin for Phi4MMModel {}

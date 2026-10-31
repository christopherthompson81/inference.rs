#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub mod config;
pub mod inputs_processor;
pub mod vision;

use std::{
    any::Any,
    sync::{Arc, Mutex},
};

pub use config::Idefics3Config;
use inference_nn::vision::siglip::SiglipVisionTransformer;
use inference_quant::{NonZeroOp, ShardedVarBuilder};
use inference_tensor::{D, DType, Device, Result, Tensor};
use vision::Idefics3Connector;

use crate::attention::AttentionMask;
use crate::{
    amoe::{AnyMoeBaseModelMixin, MlpLayer},
    amoe::{AnyMoeConfig, AnyMoeExpertType},
    kv_cache::EitherCache,
    llama::Llama,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{
        AttentionImplementation, ModelConfigMetadata, encoder_cache::EncoderCacheManager,
    },
    utils::unvarbuilder::UnVarBuilder,
    vision::multimodal_layout::PackedMultimodalLayout,
};

pub struct Idefics3SpecificArgs {
    pub pixel_attention_mask: Option<Tensor>,
    pub image_hashes: Vec<u64>,
    pub subimage_counts: Vec<usize>,
    pub packed_layout: Option<PackedMultimodalLayout>,
}

pub struct Idefics3Model {
    text_model: Llama,
    connector: Idefics3Connector,
    vision: SiglipVisionTransformer,
    config: Idefics3Config,
    dtype: DType,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

struct Idefics3ForwardInput<'a> {
    input_ids: &'a Tensor,
    pixel_values: Option<Tensor>,
    pixel_attention_mask: Option<Tensor>,
    image_hashes: &'a [u64],
    subimage_counts: &'a [usize],
    packed_layout: Option<&'a PackedMultimodalLayout>,
}

impl Idefics3Model {
    pub fn new(
        cfg: &Idefics3Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let non_text_vb = vb_m.clone().without_lora_registry();
        let connector = Idefics3Connector::new(
            cfg,
            non_text_vb
                .pp("connector")
                .set_device(normal_loading_metadata.real_device.clone()),
        )?;
        let vision = SiglipVisionTransformer::new(
            &cfg.vision_config.siglip(),
            non_text_vb
                .pp("vision_model")
                .set_device(normal_loading_metadata.real_device.clone()),
        )?;
        let text_model = Llama::new_inner(
            &cfg.text_config.decoder_spec(),
            vb_m.pp("text_model"),
            vb.pp("lm_head"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text_model,
            connector,
            vision,
            config: cfg.clone(),
            dtype: vb.dtype(),
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    fn inputs_merger(
        &self,
        indices: &Tensor,
        input_embeds: &Tensor,
        image_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let mut x_flat = input_embeds.flatten_all()?;
        let src_flat = image_hidden_states.flatten_all()?;

        let current_vals = x_flat.gather(indices, 0)?;
        let diff = (src_flat - current_vals)?;
        x_flat = x_flat.scatter_add(indices, &diff, 0)?;

        x_flat.reshape(input_embeds.shape())
    }

    fn forward_inner(
        &self,
        input: Idefics3ForwardInput<'_>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let Idefics3ForwardInput {
            input_ids,
            pixel_values,
            pixel_attention_mask,
            image_hashes,
            subimage_counts,
            packed_layout,
        } = input;
        let input_embeds = if let Some(pixel_values) = pixel_values {
            let input_embeds = self.text_model.get_input_embeddings(input_ids)?;
            let special_image_mask = input_ids
                .eq(self.config.image_token_id as f64)?
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape())?
                .to_dtype(DType::U32)?;

            let mask_flat = special_image_mask.flatten_all()?;
            // Nonzero before vision model to allow async processing all the way through logits.
            let indices = mask_flat.nonzero()?.squeeze(1)?;

            // == START VISUAL INPUTS INTEGRATION ==
            let (pixel_values, pixel_attention_mask) =
                crate::idefics_common::drop_padding_images(&pixel_values, pixel_attention_mask)?;
            let patch_size = self.config.vision_config.patch_size;
            let patch_attention_mask =
                crate::idefics_common::patch_valid_pixels(&pixel_attention_mask, patch_size)?
                    .gt(0.0)?
                    .to_dtype(DType::U8)?;

            let pixel_values = pixel_values.to_dtype(self.dtype)?;

            // Get seq from vision encoder + connector, with per-image caching
            let (image_hidden_states, encoder_outputs) = if !image_hashes.is_empty()
                && image_hashes.len() == subimage_counts.len()
            {
                let batch = crate::idefics_common::SubimageBatch {
                    label: "Idefics3",
                    image_hashes,
                    subimage_counts,
                    pixel_values: &pixel_values,
                    patch_attention_mask: &patch_attention_mask,
                    rows: None,
                };
                let (hidden, outputs) = crate::idefics_common::encode_subimages_cached(
                    &batch,
                    &self.encoder_cache,
                    |pv, mask| {
                        let hidden =
                            self.vision
                                .forward(pv, &AttentionMask::Custom(mask.clone()), None)?;
                        self.connector.forward(&hidden)
                    },
                )?;
                (hidden, Some(outputs))
            } else {
                let image_hidden_states = self.vision.forward(
                    &pixel_values,
                    &AttentionMask::Custom(patch_attention_mask.clone()),
                    None,
                )?;
                (self.connector.forward(&image_hidden_states)?, None)
            };

            if let Some(layout) = packed_layout {
                let encoder_outputs = encoder_outputs.ok_or_else(|| {
                    inference_tensor::Error::msg(
                        "packed Idefics3 input requires per-image encoder outputs",
                    )
                })?;
                layout
                    .splice_embeddings(&input_embeds, &encoder_outputs)?
                    .to_dtype(self.dtype)?
            } else {
                self.inputs_merger(&indices, &input_embeds, &image_hidden_states)?
                    .to_dtype(self.dtype)?
            }
        } else {
            self.text_model.get_input_embeddings(input_ids)?
        };

        self.text_model.forward_embeds(input_ids, input_embeds, ctx)
    }
}

impl IsqModel for Idefics3Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m
            .pp("connector")
            .pp("modality_projection")
            .pp("proj")
            .add(&self.connector.modality_projection.proj);
        uvb.extend(self.text_model.residual_tensors_m(uvb_m.pp("text_model")));
        uvb_m
            .pp("vision_model")
            .extend(self.vision.residual_tensors());

        uvb.to_safetensors()
    }
}

// AnyMoE is forwarded to the base model
impl AnyMoeBaseModelMixin for Idefics3Model {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        self.text_model.get_mlps()
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.text_model.get_mlps_mut()
    }
    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<ShardedVarBuilder>,
        config: AnyMoeConfig,
        (prefix, mlp): (String, String),
        layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        self.text_model.create_anymoe_layers(
            additional_vbs,
            config,
            (prefix, mlp),
            layers,
            expert_type,
            gate_vb,
        )
    }
    fn amoe_supported(&self) -> bool {
        true
    }
}

impl crate::speculative::SpeculativeTargetMixin for Idefics3Model {}

impl crate::model::BlockDiffusionMixin for Idefics3Model {}

impl MultimodalModel for Idefics3Model {
    fn supports_packed_prefill(&self) -> bool {
        self.text_model.supports_packed_prefill()
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
    ) -> inference_tensor::Result<Tensor> {
        let Idefics3SpecificArgs {
            pixel_attention_mask,
            image_hashes,
            subimage_counts,
            packed_layout,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Idefics3SpecificArgs`");
        self.forward_inner(
            Idefics3ForwardInput {
                input_ids,
                pixel_values,
                pixel_attention_mask,
                image_hashes: &image_hashes,
                subimage_counts: &subimage_counts,
                packed_layout: packed_layout.as_ref(),
            },
            ctx,
        )
    }
    fn cache(&self) -> &EitherCache {
        self.text_model.cache()
    }
    fn device(&self) -> &Device {
        self.text_model.device()
    }
    fn max_seq_len(&self) -> usize {
        self.text_model.max_seq_len()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.text_model.config()
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(Idefics3SpecificArgs {
            pixel_attention_mask: None,
            image_hashes: vec![],
            subimage_counts: vec![],
            packed_layout: None,
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

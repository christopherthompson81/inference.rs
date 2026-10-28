#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub mod config;
pub mod inputs_processor;
pub mod vision;

use inference_quant::{NonZeroOp, ShardedVarBuilder};
use inference_tensor::nn::{LayerNorm, Linear, Module};
use inference_tensor::{D, DType, Device, Result, Tensor};
use vision::VisionModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    kv_cache::EitherCache,
    layers::{layer_norm, linear_b},
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    utils::unvarbuilder::UnVarBuilder,
};

pub use config::Config;

/// The projector's sizes: the pixel unshuffle factor, an optional input LayerNorm, the MLP's widths and activation.
pub struct ProjectorShape {
    pub downsample_factor: usize,
    pub use_layernorm: bool,
    pub hidden_size: usize,
    pub out_size: usize,
    pub bias: bool,
    pub activation: crate::layers::Activation,
}

pub struct MultiModalProjector {
    factor: usize,
    layer_norm: Option<LayerNorm>,
    linear_1: Linear,
    linear_2: Linear,
    activation: crate::layers::Activation,
}

impl MultiModalProjector {
    fn new(config: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let shape = ProjectorShape {
            downsample_factor: config.downsample_factor,
            use_layernorm: config.projector_use_layernorm,
            hidden_size: config.projector_hidden_size,
            out_size: config.text_config.hidden_size,
            bias: config.projector_bias,
            activation: config.projector_hidden_act,
        };
        Self::with_shape(config.vision_config.hidden_size, &shape, vb)
    }

    pub fn with_shape(
        vision_hidden: usize,
        shape: &ProjectorShape,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let in_channels = vision_hidden * shape.downsample_factor.pow(2);
        Ok(Self {
            factor: shape.downsample_factor,
            layer_norm: if shape.use_layernorm {
                Some(layer_norm(in_channels, 1e-5, vb.pp("layer_norm"))?)
            } else {
                None
            },
            linear_1: linear_b(
                in_channels,
                shape.hidden_size,
                shape.bias,
                vb.pp("linear_1"),
            )?,
            linear_2: linear_b(
                shape.hidden_size,
                shape.out_size,
                shape.bias,
                vb.pp("linear_2"),
            )?,
            activation: shape.activation,
        })
    }

    fn pixel_unshuffle(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, height, width, channels) = xs.dims4()?;
        if height % self.factor != 0 || width % self.factor != 0 {
            inference_tensor::bail!(
                "LFM2-VL projector expected image features divisible by {}, got ({height}, {width})",
                self.factor
            );
        }
        xs.reshape((batch, height, width / self.factor, channels * self.factor))?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((
                batch,
                width / self.factor,
                height / self.factor,
                channels * self.factor * self.factor,
            ))?
            .permute((0, 2, 1, 3))?
            .contiguous()
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.pixel_unshuffle(xs)?;
        if let Some(layer_norm) = &self.layer_norm {
            xs = layer_norm.forward(&xs)?;
        }
        let xs = self.linear_1.forward(&xs)?;
        self.linear_2.forward(&self.activation.forward(&xs)?)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        if let Some(layer_norm) = &self.layer_norm {
            uvb.pp("layer_norm").add(layer_norm);
        }
        uvb.pp("linear_1").add(&self.linear_1);
        uvb.pp("linear_2").add(&self.linear_2);
        uvb.to_safetensors()
    }
}

/// Each crop's tower states, cut to its patches, through the projector: `[image tokens, hidden]`, crops in order.
pub fn image_features(
    vision_tower: &VisionModel,
    projector: &MultiModalProjector,
    pixel_values: &Tensor,
    pixel_attention_mask: &Tensor,
    spatial_shapes: &Tensor,
    hidden: usize,
) -> Result<Tensor> {
    let last_hidden_state =
        vision_tower.forward(pixel_values, pixel_attention_mask, spatial_shapes)?;
    let mask_lengths = pixel_attention_mask
        .sum(1)?
        .to_dtype(DType::U32)?
        .to_device(&Device::Cpu)?
        .to_vec1::<u32>()?;
    let shapes = spatial_shapes
        .to_dtype(DType::U32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<u32>()?;
    let mut features = Vec::with_capacity(last_hidden_state.dim(0)?);
    for image_idx in 0..last_hidden_state.dim(0)? {
        let feature_len = mask_lengths[image_idx] as usize;
        let height = shapes[image_idx][0] as usize;
        let width = shapes[image_idx][1] as usize;
        let feature = last_hidden_state
            .get(image_idx)?
            .narrow(0, 0, feature_len)?
            .reshape((1, height, width, ()))?;
        features.push(projector.forward(&feature)?.reshape(((), hidden))?);
    }
    Tensor::cat(&features, 0)
}

pub struct Lfm2VlModel {
    language_model: crate::lfm2::Model,
    vision_tower: VisionModel,
    multi_modal_projector: MultiModalProjector,
    image_token_id: usize,
}

impl Lfm2VlModel {
    pub fn new(
        config: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let device = normal_loading_metadata.real_device.clone();
        let non_text_vb = vb.clone().without_lora_registry();
        let vision_tower = VisionModel::new(
            &config.vision_config,
            non_text_vb
                .pp("model")
                .pp("vision_tower")
                .pp("vision_model")
                .set_device(device.clone()),
        )?;
        let multi_modal_projector = MultiModalProjector::new(
            config,
            non_text_vb
                .pp("model")
                .pp("multi_modal_projector")
                .set_device(device.clone()),
        )?;
        let language_model = crate::lfm2::Model::new_inner(
            &config.text_config,
            vb.pp("model").pp("language_model"),
            vb.pp("lm_head"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;

        Ok(Self {
            language_model,
            vision_tower,
            multi_modal_projector,
            image_token_id: config.image_token_id,
        })
    }

    fn get_image_features(
        &self,
        pixel_values: &Tensor,
        pixel_attention_mask: &Tensor,
        spatial_shapes: &Tensor,
    ) -> Result<Tensor> {
        image_features(
            &self.vision_tower,
            &self.multi_modal_projector,
            pixel_values,
            pixel_attention_mask,
            spatial_shapes,
            self.language_model.config().hidden_size,
        )
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        pixel_attention_mask: Option<Tensor>,
        spatial_shapes: Option<Tensor>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut input_embeds = self.language_model.embed(input_ids)?;
        if let Some(pixel_values) = pixel_values {
            let pixel_attention_mask = pixel_attention_mask.ok_or_else(|| {
                inference_tensor::Error::msg(
                    "LFM2-VL requires pixel_attention_mask with pixel_values",
                )
            })?;
            let spatial_shapes = spatial_shapes.ok_or_else(|| {
                inference_tensor::Error::msg("LFM2-VL requires spatial_shapes with pixel_values")
            })?;
            let image_features = self
                .get_image_features(&pixel_values, &pixel_attention_mask, &spatial_shapes)?
                .to_device(input_embeds.device())?
                .to_dtype(input_embeds.dtype())?;

            let special_image_mask = input_ids
                .eq(self.image_token_id as f64)?
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape().clone())?
                .to_dtype(DType::U32)?;
            let indices = special_image_mask.flatten_all()?.nonzero()?.squeeze(1)?;
            if indices.dim(0)? != image_features.elem_count() {
                inference_tensor::bail!(
                    "LFM2-VL image features and image tokens do not match, tokens: {}, features: {}",
                    indices.dim(0)? / input_embeds.dim(D::Minus1)?,
                    image_features.dim(0)?
                );
            }

            let mut x_flat = input_embeds.flatten_all()?;
            let src_flat = image_features.flatten_all()?;
            let current_vals = x_flat.gather(&indices, 0)?;
            let diff = (src_flat - current_vals)?;
            x_flat = x_flat.scatter_add(&indices, &diff, 0)?;
            input_embeds = x_flat.reshape(input_embeds.shape())?;
        }
        self.language_model
            .forward_embeds(input_ids, input_embeds, ctx)
    }
}

impl IsqModel for Lfm2VlModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.extend(
            self.language_model
                .residual_tensors_m(uvb.pp("model").pp("language_model")),
        );
        uvb.pp("model")
            .pp("vision_tower")
            .pp("vision_model")
            .extend(self.vision_tower.residual_tensors());
        uvb.pp("model")
            .pp("multi_modal_projector")
            .extend(self.multi_modal_projector.residual_tensors());
        uvb.to_safetensors()
    }
}

pub struct Lfm2VlSpecificArgs {
    pub pixel_attention_mask: Option<Tensor>,
    pub spatial_shapes: Option<Tensor>,
}

impl crate::speculative::SpeculativeTargetMixin for Lfm2VlModel {}
impl crate::model::BlockDiffusionMixin for Lfm2VlModel {}
impl AnyMoeBaseModelMixin for Lfm2VlModel {}

impl MultimodalModel for Lfm2VlModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let Lfm2VlSpecificArgs {
            pixel_attention_mask,
            spatial_shapes,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Lfm2VlSpecificArgs`");
        self.forward_inner(
            input_ids,
            pixel_values,
            pixel_attention_mask,
            spatial_shapes,
            ctx,
        )
    }

    fn device(&self) -> &Device {
        self.language_model.device()
    }

    fn cache(&self) -> &EitherCache {
        self.language_model.cache()
    }

    fn max_seq_len(&self) -> usize {
        self.language_model.max_seq_len()
    }

    fn config(&self) -> &ModelConfigMetadata {
        self.language_model.config()
    }

    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(Lfm2VlSpecificArgs {
            pixel_attention_mask: None,
            spatial_shapes: None,
        })
    }

    fn supports_packed_prefill(&self) -> bool {
        self.language_model.supports_packed_prefill()
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
}

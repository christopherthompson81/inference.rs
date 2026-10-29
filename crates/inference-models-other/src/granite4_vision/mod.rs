//! IBM Granite 4 Vision: a SigLIP tower whose layers feed Window Q-Former projectors, each projector's features added
//! at the image positions before its own Granite text layer (deepstack), over LLaVA-Next tiling.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub mod config;
pub mod inputs_processor;
pub mod qformer;

use inference_nn::vision::llava_tiling::select_best_resolution;
use inference_nn::vision::siglip::SiglipVisionTransformer;
use inference_quant::ShardedVarBuilder;
use inference_tensor::{D, DType, Device, IndexOp, Result, Tensor};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    granite::GraniteMoeHybrid,
    kv_cache::EitherCache,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    utils::unvarbuilder::UnVarBuilder,
};
pub use config::Config;
use qformer::WindowQFormerDownsampler;

// transformers' unpad rounds to 7 decimals before truncating
const UNPAD_ROUNDING: f64 = 1e7;
const SPATIAL_OFFSETS: usize = 4;

/// Python's `int(round(x, 7))`.
fn round7_trunc(x: f64) -> usize {
    ((x * UNPAD_ROUNDING).round() / UNPAD_ROUNDING).trunc() as usize
}

/// The feature grid's `(rows, cols)` once LLaVA-Next unpads the tiles' `(rows, cols)` to an `(height, width)` image.
pub fn unpadded_grid(grid: (usize, usize), image: (usize, usize)) -> (usize, usize) {
    let (rows, cols) = grid;
    let (height, width) = (image.0 as f64, image.1 as f64);
    if width / height > cols as f64 / rows as f64 {
        let new_rows = round7_trunc(height * (cols as f64 / width));
        let padding = (rows - new_rows) / 2;
        (rows - 2 * padding, cols)
    } else {
        let new_cols = round7_trunc(width * (rows as f64 / height));
        let padding = (cols - new_cols) / 2;
        (rows, cols - 2 * padding)
    }
}

/// The tile grid `(rows, cols)` LLaVA-Next picks for an `(height, width)` image; the pinpoints are `[height, width]`.
pub fn tile_grid(config: &Config, image: (usize, usize)) -> (usize, usize) {
    let (best_height, best_width) = select_best_resolution(
        (image.0 as u32, image.1 as u32),
        &config.image_grid_pinpoints,
    );
    let tile = config.vision_config.image_size as u32;
    ((best_height / tile) as usize, (best_width / tile) as usize)
}

/// Image tokens for an `(height, width)` image: the base tile, then the unpadded tiles with a newline per row.
pub fn image_token_count(config: &Config, image: (usize, usize)) -> Result<usize> {
    let side = config.features_per_side()?;
    let (rows, cols) = tile_grid(config, image);
    let (rows, cols) = unpadded_grid((rows * side, cols * side), image);
    Ok(side * side + rows * (cols + usize::from(config.use_image_newline_parameter)))
}

pub struct Granite4VisionSpecificArgs {
    /// `(height, width)` of each image whose crops `pixel_values` holds, in order.
    pub image_sizes: Vec<(usize, usize)>,
    /// Crops per image: the base image, then its tiles.
    pub num_crops: Vec<usize>,
}

struct Projection {
    vision_state: usize,
    text_layer: usize,
    projector: WindowQFormerDownsampler,
}

pub struct Granite4VisionModel {
    language_model: GraniteMoeHybrid,
    vision_tower: SiglipVisionTransformer,
    image_newline: Option<Tensor>,
    layerwise: Vec<Projection>,
    spatial: Vec<Projection>,
    config: Config,
    tower_at_root: bool,
}

impl Granite4VisionModel {
    pub fn new(
        config: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let device = normal_loading_metadata.real_device.clone();
        let vision_vb = vb
            .clone()
            .without_lora_registry()
            .pp("model")
            .set_device(device.clone());
        // granite-4.0-3b-vision keeps its tower at the top level, as older LLaVA-Next checkpoints do
        let tower_at_root = !crate::layers::contains_tensor_or_weight_source(
            &vision_vb
                .pp("vision_tower")
                .pp("vision_model")
                .pp("embeddings")
                .pp("patch_embedding"),
            "weight",
        );
        let tower_vb = match tower_at_root {
            true => vb
                .clone()
                .without_lora_registry()
                .set_device(device.clone()),
            false => vision_vb.clone(),
        };
        let vision_tower = SiglipVisionTransformer::new(
            &config.vision_config,
            tower_vb.pp("vision_tower").pp("vision_model"),
        )?;
        let image_newline = config
            .use_image_newline_parameter
            .then(|| vision_vb.get(config.text_config.hidden_size, "image_newline"))
            .transpose()?;
        let layerwise = config
            .deepstack_layer_map
            .iter()
            .enumerate()
            .map(|(i, &(state, text_layer))| {
                Ok(Projection {
                    vision_state: config.hidden_state_index(state)?,
                    text_layer,
                    projector: WindowQFormerDownsampler::new(
                        config,
                        None,
                        vision_vb.pp("layerwise_projectors").pp(i),
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let spatial_state = config.hidden_state_index(config.spatial_vision_layer)?;
        let spatial = if config.use_spatial_sampling {
            config
                .spatial_target_layers
                .iter()
                .take(SPATIAL_OFFSETS)
                .enumerate()
                .map(|(offset, &text_layer)| {
                    Ok(Projection {
                        vision_state: spatial_state,
                        text_layer,
                        projector: WindowQFormerDownsampler::new(
                            config,
                            Some(offset),
                            vision_vb.pp("spatial_projectors").pp(offset),
                        )?,
                    })
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let lm_head = if config.text_config.tie_word_embeddings {
            vb.pp("model").pp("language_model").pp("embed_tokens")
        } else {
            vb.pp("lm_head")
        };
        let language_model = GraniteMoeHybrid::new_inner(
            &config.text_config,
            vb.pp("model").pp("language_model"),
            lm_head,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            language_model,
            vision_tower,
            image_newline,
            layerwise,
            spatial,
            config: config.clone(),
            tower_at_root,
        })
    }

    /// One image's projected crops `(crops, side*side, hidden)` laid out as transformers' `pack_image_features`.
    fn pack(&self, crops: &Tensor, image: (usize, usize)) -> Result<Tensor> {
        let side = self.config.features_per_side()?;
        let base = crops.get(0)?;
        let hidden = base.dim(D::Minus1)?;
        if crops.dim(0)? == 1 {
            return match &self.image_newline {
                Some(newline) => {
                    Tensor::cat(&[base, newline.unsqueeze(0)?.to_dtype(crops.dtype())?], 0)
                }
                None => Ok(base),
            };
        }
        let (rows, cols) = tile_grid(&self.config, image);
        let grid = crops
            .i(1..)?
            .reshape((rows, cols, side, side, hidden))?
            .permute((4, 0, 2, 1, 3))?
            .reshape((hidden, rows * side, cols * side))?;
        let (kept_rows, kept_cols) = unpadded_grid((rows * side, cols * side), image);
        let grid = grid
            .narrow(1, (rows * side - kept_rows) / 2, kept_rows)?
            .narrow(2, (cols * side - kept_cols) / 2, kept_cols)?;
        let grid = match &self.image_newline {
            Some(newline) => {
                let newline = newline
                    .to_dtype(grid.dtype())?
                    .reshape((hidden, 1, 1))?
                    .broadcast_as((hidden, kept_rows, 1))?;
                Tensor::cat(&[grid, newline.contiguous()?], 2)?
            }
            None => grid,
        };
        let tiles = grid.flatten_from(1)?.t()?;
        Tensor::cat(&[base, tiles], 0)
    }

    /// Each projector's packed features for every image: `(text layer, (image tokens, hidden))`.
    fn image_features(
        &self,
        pixel_values: &Tensor,
        args: &Granite4VisionSpecificArgs,
    ) -> Result<Vec<(usize, Tensor)>> {
        if args.image_sizes.len() != args.num_crops.len()
            || args.num_crops.iter().sum::<usize>() != pixel_values.dim(0)?
        {
            inference_tensor::bail!(
                "Granite 4 Vision received {} crops for crop counts {:?}",
                pixel_values.dim(0)?,
                args.num_crops
            );
        }
        let states = self
            .vision_tower
            .forward_all_hidden_states(&pixel_values.to_dtype(self.vision_tower.dtype())?)?;
        // `default` drops a class token the SigLIP tower has none of, so both strategies read every patch
        self.layerwise
            .iter()
            .chain(&self.spatial)
            .map(|projection| {
                let projected = projection
                    .projector
                    .forward(&states[projection.vision_state])?;
                let mut offset = 0;
                let packed = args
                    .num_crops
                    .iter()
                    .zip(&args.image_sizes)
                    .map(|(&crops, &image)| {
                        let packed = self.pack(&projected.narrow(0, offset, crops)?, image);
                        offset += crops;
                        packed
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok((projection.text_layer, Tensor::cat(&packed, 0)?))
            })
            .collect()
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        args: Granite4VisionSpecificArgs,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let embeds = self.language_model.embed(input_ids)?;
        let Some(pixel_values) = pixel_values else {
            return self
                .language_model
                .forward_embeds(input_ids, embeds, ctx, &|_, xs| Ok(xs));
        };
        let features = self.image_features(&pixel_values, &args)?;
        let image_mask = input_ids.eq(f64::from(self.config.image_token_index))?;
        let image_tokens = image_mask
            .to_dtype(DType::U32)?
            .sum_all()?
            .to_scalar::<u32>()? as usize;
        if let Some((_, first)) = features.first()
            && first.dim(0)? != image_tokens
        {
            inference_tensor::bail!(
                "Granite 4 Vision image features and image tokens do not match, tokens: {image_tokens}, features: {}",
                first.dim(0)?
            );
        }
        // the image positions start from zero, and every projector's features are added before its layer
        let keep = image_mask
            .to_dtype(embeds.dtype())?
            .affine(-1.0, 1.0)?
            .unsqueeze(D::Minus1)?;
        let embeds = embeds.broadcast_mul(&keep)?;
        let add_features = |layer: usize, mut xs: Tensor| -> Result<Tensor> {
            for (_, layer_features) in features.iter().filter(|(target, _)| *target == layer) {
                xs = crate::decoder::qwen3_vl::deepstack_process(xs, &image_mask, layer_features)?;
            }
            Ok(xs)
        };
        self.language_model
            .forward_embeds(input_ids, embeds, ctx, &add_features)
    }
}

impl IsqModel for Granite4VisionModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let model = uvb.pp("model");
        model.extend(
            self.language_model
                .residual_tensors_m(model.pp("language_model")),
        );
        let tower = if self.tower_at_root { &uvb } else { &model };
        tower
            .pp("vision_tower")
            .pp("vision_model")
            .extend(self.vision_tower.residual_tensors());
        if let Some(newline) = &self.image_newline {
            model.add_tensor("image_newline", newline.clone());
        }
        for (i, projection) in self.layerwise.iter().enumerate() {
            model
                .pp("layerwise_projectors")
                .pp(i)
                .extend(projection.projector.residual_tensors());
        }
        for (i, projection) in self.spatial.iter().enumerate() {
            model
                .pp("spatial_projectors")
                .pp(i)
                .extend(projection.projector.residual_tensors());
        }
        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for Granite4VisionModel {}
impl crate::model::BlockDiffusionMixin for Granite4VisionModel {}
impl AnyMoeBaseModelMixin for Granite4VisionModel {}

impl MultimodalModel for Granite4VisionModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let args = *model_specific_args
            .downcast::<Granite4VisionSpecificArgs>()
            .expect("Cannot downcast into `Granite4VisionSpecificArgs`");
        self.forward_inner(input_ids, pixel_values, args, ctx)
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
        Box::new(Granite4VisionSpecificArgs {
            image_sizes: Vec::new(),
            num_crops: Vec::new(),
        })
    }

    fn supports_packed_prefill(&self) -> bool {
        false
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
}

//! Nemotron-H Omni (`NemotronH_Omni_Reasoning_V3`, e.g. Nemotron 3.5 Bolt): a Nemotron-H decoder whose `<image>`
//! positions take C-RADIO's features of each image, pixel-shuffled and projected.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub mod inputs_processor;
pub mod radio;
pub mod sound;

use inference_quant::ShardedVarBuilder;
use inference_tensor::{Device, Result, Tensor};
use serde::Serialize;

use crate::granite::GraniteMoeHybrid;
use crate::{
    amoe::AnyMoeBaseModelMixin,
    kv_cache::EitherCache,
    media_inputs::media::place_token_features,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
};

pub const MODEL: &str = "Nemotron-H Omni";
// InternVL's tile count, NVIDIA's `max_num_tiles` default
pub const DEFAULT_MAX_TILES: usize = 12;
pub const LANGUAGE_MODEL: &str = "language_model";

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub layer_norm_eps: f64,
    pub mlp_ratio: f64,
    pub num_cls_tokens: usize,
    pub num_registers: usize,
    /// The CPE grid's side in pixels: `max_img_size / patch_size` positions a side.
    pub max_img_size: usize,
    /// The patch budget a dynamic-resolution image is sized to; without one, images are cut into tiles.
    #[serde(default)]
    pub min_num_patches: Option<usize>,
    #[serde(default)]
    pub max_num_patches: Option<usize>,
}

impl VisionConfig {
    pub fn intermediate_size(&self) -> usize {
        (self.hidden_size as f64 * self.mlp_ratio) as usize
    }
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Config {
    pub llm_config: crate::nemotron_h::Config,
    pub vision_config: VisionConfig,
    pub downsample_ratio: f64,
    pub img_context_token_id: u32,
    pub norm_mean: Vec<f32>,
    pub norm_std: Vec<f32>,
    pub patch_size: usize,
    pub projector_hidden_size: usize,
    #[serde(default)]
    pub vision_final_layernorm: Option<bool>,
    #[serde(default)]
    pub sound_config: Option<sound::SoundConfig>,
    #[serde(default)]
    pub sound_context_token_id: Option<u32>,
    /// A tiled checkpoint's tile side in pixels (Nano VL's 512).
    #[serde(default)]
    pub force_image_size: Option<usize>,
    #[serde(default)]
    pub use_thumbnail: Option<bool>,
}

/// How an image becomes the tower's input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageLayout {
    /// One tile on a patch grid sized to `min..=max` patches, aspect kept (Bolt, Nano Omni).
    Budget { min: usize, max: usize },
    /// InternVL's tiles: up to `max_tiles` squares of `size` pixels on the closest aspect ratio, and a thumbnail of
    /// the whole image after them when there is more than one (Nano VL).
    Tiles {
        size: usize,
        max_tiles: usize,
        thumbnail: bool,
    },
}

impl Config {
    /// Bolt's config, or Nano Omni's (`NemotronH_Nano_Omni_Reasoning_V3`), whose `vision_config` is RADIO's own.
    pub fn parse(json: &str) -> anyhow::Result<Self> {
        let mut value: serde_json::Value = serde_json::from_str(json)?;
        if value["vision_config"].get("hidden_size").is_none() {
            inference_nn::vision::radio::fill_radio_shape(&mut value["vision_config"])?;
            // its reference has no final LayerNorm before the projector
            if value.get("vision_final_layernorm").is_none() {
                value["vision_final_layernorm"] = serde_json::Value::Bool(false);
            }
        }
        Ok(Self::from_json(&value.to_string())?)
    }

    pub fn image_layout(&self, max_tiles: Option<usize>) -> Result<ImageLayout> {
        let v = &self.vision_config;
        match (v.min_num_patches, v.max_num_patches, self.force_image_size) {
            (Some(min), Some(max), _) => Ok(ImageLayout::Budget { min, max }),
            (_, _, Some(size)) => Ok(ImageLayout::Tiles {
                size,
                max_tiles: max_tiles.unwrap_or(DEFAULT_MAX_TILES),
                thumbnail: self.use_thumbnail.unwrap_or(true),
            }),
            _ => inference_tensor::bail!(
                "{MODEL}: the config has neither a patch budget nor `force_image_size`"
            ),
        }
    }

    /// The most patches one image runs through the tower: its budget, or every tile and the thumbnail.
    pub fn max_image_patches(&self) -> Result<usize> {
        Ok(match self.image_layout(None)? {
            ImageLayout::Budget { max, .. } => max,
            ImageLayout::Tiles {
                size, max_tiles, ..
            } => (max_tiles + 1) * (size / self.patch_size).pow(2),
        })
    }

    /// The pixel shuffle's factor on each side.
    pub fn merge(&self) -> usize {
        (1.0 / self.downsample_ratio).round() as usize
    }
}

/// One image as the processor cut it: `(rows * cols, 3 * patch * patch)` normalized patches, row-major.
pub struct ImagePatches {
    pub patches: Tensor,
    pub rows: usize,
    pub cols: usize,
}

pub struct NemotronOmniSpecificArgs {
    /// The images whose placeholders this step holds, in prompt order.
    pub images: Vec<ImagePatches>,
    /// The clips whose placeholders this step holds, in prompt order: each one's `(frames, mels)` normalized log-mel.
    pub sounds: Vec<Tensor>,
}

pub struct NemotronOmniModel {
    text: GraniteMoeHybrid,
    vision: radio::RadioTower,
    sound: Option<(sound::SoundTower, u32)>,
    image_token: u32,
    text_root: &'static str,
}

impl NemotronOmniModel {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        // unset, the reference infers the LayerNorm from the decoder's MTP layers, which this config does not read
        if cfg.vision_final_layernorm != Some(false) {
            inference_tensor::bail!(
                "{MODEL}: only checkpoints with `vision_final_layernorm: false` are supported"
            );
        }
        let vision = radio::RadioTower::new(cfg, vb.clone(), cfg.llm_config.hidden_size)?;
        let sound = match (&cfg.sound_config, cfg.sound_context_token_id) {
            (Some(sound), Some(token)) => Some((
                sound::SoundTower::new(sound, vb.clone(), cfg.llm_config.hidden_size)?,
                token,
            )),
            (Some(_), None) => {
                inference_tensor::bail!("{MODEL}: `sound_config` needs `sound_context_token_id`")
            }
            _ => None,
        };
        let llm = vb.pp(LANGUAGE_MODEL);
        let (root, lm_head) = crate::loaders::nemotron_h_prefixes(&llm);
        let text = GraniteMoeHybrid::new_with_layout(
            &cfg.llm_config.granite()?,
            cfg.llm_config.layout()?,
            llm.pp(root),
            llm.pp(lm_head),
            true,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text,
            vision,
            sound,
            image_token: cfg.img_context_token_id,
            text_root: root,
        })
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        args: NemotronOmniSpecificArgs,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut embeds = self.text.token_embeddings(input_ids)?;
        if !args.images.is_empty() {
            let projected = args
                .images
                .iter()
                .map(|image| self.vision.forward(&image.patches, image.rows, image.cols))
                .collect::<Result<Vec<_>>>()?;
            let projected = Tensor::cat(&projected, 0)?.to_device(embeds.device())?;
            embeds = place_token_features(&embeds, input_ids, self.image_token, &projected, MODEL)?;
        }
        if !args.sounds.is_empty() {
            let Some((tower, token)) = &self.sound else {
                inference_tensor::bail!("{MODEL}: this checkpoint has no sound encoder");
            };
            let projected = args
                .sounds
                .iter()
                .map(|clip| tower.forward(&clip.to_device(embeds.device())?))
                .collect::<Result<Vec<_>>>()?;
            let projected = Tensor::cat(&projected, 0)?;
            embeds = place_token_features(&embeds, input_ids, *token, &projected, MODEL)?;
        }
        self.text
            .forward_embeds(input_ids, embeds, ctx, &|_, x| Ok(x))
    }
}

impl IsqModel for NemotronOmniModel {
    // the vision tower stays as the checkpoint has it
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = crate::utils::unvarbuilder::UnVarBuilder::new();
        self.text
            .residual_tensors_m(uvb.pp(LANGUAGE_MODEL).pp(self.text_root))
    }
}

impl crate::speculative::SpeculativeTargetMixin for NemotronOmniModel {}
impl crate::model::BlockDiffusionMixin for NemotronOmniModel {}
impl AnyMoeBaseModelMixin for NemotronOmniModel {}

impl MultimodalModel for NemotronOmniModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let args = *model_specific_args
            .downcast::<NemotronOmniSpecificArgs>()
            .expect("Cannot downcast into `NemotronOmniSpecificArgs`");
        self.forward_inner(input_ids, args, ctx)
    }

    fn device(&self) -> &Device {
        NormalModel::device(&self.text)
    }

    fn cache(&self) -> &EitherCache {
        NormalModel::cache(&self.text)
    }

    fn max_seq_len(&self) -> usize {
        NormalModel::max_seq_len(&self.text)
    }

    fn config(&self) -> &ModelConfigMetadata {
        NormalModel::config(&self.text)
    }

    fn model_config(
        &self,
    ) -> std::sync::Arc<dyn inference_nn::paged_attention::ModelConfigLike + Send + Sync> {
        self.text.paged_kv_config()
    }

    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        NormalModel::supports_cuda_decode_graphs(&self.text)
    }

    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(NemotronOmniSpecificArgs {
            images: Vec::new(),
            sounds: Vec::new(),
        })
    }

    fn supports_packed_prefill(&self) -> bool {
        false
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
}

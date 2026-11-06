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
pub const LANGUAGE_MODEL: &str = "language_model";
// timm ViTs a RADIO training config names in `args.model`: (name, hidden, heads, layers, mlp ratio)
const RADIO_VITS: [(&str, usize, usize, usize, f64); 1] =
    [("vit_huge_patch16_224", 1280, 16, 32, 4.0)];
// timm's ViT LayerNorm
const RADIO_LAYER_NORM_EPS: f64 = 1e-6;

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
    pub min_num_patches: usize,
    pub max_num_patches: usize,
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
}

/// Fills a RADIO training config's tower shape (Nano Omni's `vision_config`) in as Bolt's config states it.
fn radio_shape(vision: &mut serde_json::Value) -> anyhow::Result<()> {
    let args = &vision["args"];
    let model = args["model"].as_str().unwrap_or_default();
    let Some(&(_, hidden, heads, layers, mlp_ratio)) =
        RADIO_VITS.iter().find(|(name, ..)| *name == model)
    else {
        anyhow::bail!("{MODEL}: RADIO ViT `{model}` is not supported");
    };
    // one cls token per teacher, registers padding the prefix to `register_multiple`
    let cls = if args["cls_token_per_teacher"].as_bool() == Some(true) {
        args["teachers"].as_array().map_or(1, Vec::len)
    } else {
        1
    };
    let multiple = args["register_multiple"].as_u64().unwrap_or(0) as usize;
    let registers = if multiple > 0 {
        multiple - cls % multiple
    } else {
        0
    };
    let shape = serde_json::json!({
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_hidden_layers": layers,
        "mlp_ratio": mlp_ratio,
        "layer_norm_eps": RADIO_LAYER_NORM_EPS,
        "num_cls_tokens": cls,
        "num_registers": registers,
        "max_img_size": args["cpe_max_size"],
    });
    let object = vision
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("{MODEL}: `vision_config` is not an object"))?;
    for (key, value) in shape.as_object().into_iter().flatten() {
        object.entry(key.clone()).or_insert_with(|| value.clone());
    }
    Ok(())
}

impl Config {
    /// Bolt's config, or Nano Omni's (`NemotronH_Nano_Omni_Reasoning_V3`), whose `vision_config` is RADIO's own.
    pub fn parse(json: &str) -> anyhow::Result<Self> {
        let mut value: serde_json::Value = serde_json::from_str(json)?;
        if value["vision_config"].get("hidden_size").is_none() {
            radio_shape(&mut value["vision_config"])?;
            // its reference has no final LayerNorm before the projector
            if value.get("vision_final_layernorm").is_none() {
                value["vision_final_layernorm"] = serde_json::Value::Bool(false);
            }
        }
        Ok(Self::from_json(&value.to_string())?)
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

#[cfg(test)]
mod tests {
    use super::radio_shape;

    // Nano Omni's `vision_config`: RADIO's training args where Bolt's config states the tower's shape
    #[test]
    fn a_radio_training_config_gives_bolts_shape() {
        let mut vision = serde_json::json!({
            "args": {
                "model": "vit_huge_patch16_224",
                "cpe_max_size": 2048,
                "register_multiple": 10,
                "cls_token_per_teacher": true,
                "teachers": [{"name": "clip"}, {"name": "siglip"}, {"name": "dino_v2"}, {"name": "sam"}],
            },
            "min_num_patches": 1024,
            "max_num_patches": 13312,
        });
        radio_shape(&mut vision).unwrap();
        for (key, want) in [
            ("hidden_size", 1280.0),
            ("num_attention_heads", 16.0),
            ("num_hidden_layers", 32.0),
            ("mlp_ratio", 4.0),
            ("num_cls_tokens", 4.0),
            ("num_registers", 6.0),
            ("max_img_size", 2048.0),
        ] {
            assert_eq!(vision[key].as_f64(), Some(want), "{key}");
        }
        // the mmproj of the same checkpoint keeps cls and registers as one table of 10
        let prefix =
            vision["num_cls_tokens"].as_u64().unwrap() + vision["num_registers"].as_u64().unwrap();
        assert_eq!(prefix, 10);
        assert_eq!(vision["min_num_patches"], 1024);
    }

    #[test]
    fn an_unknown_radio_vit_is_refused() {
        let mut vision = serde_json::json!({"args": {"model": "vit_tiny_patch16_224"}});
        let error = radio_shape(&mut vision).unwrap_err();
        assert!(
            error.to_string().contains("vit_tiny_patch16_224"),
            "{error}"
        );
    }
}

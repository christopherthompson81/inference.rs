use serde::Deserialize;

use crate::serde_default_fn;
use inference_nn::vision::siglip::SiglipVisionConfig;

serde_default_fn!(String, default_select_strategy, "default".to_string());
serde_default_fn!(i64, default_spatial_vision_layer, -1);
serde_default_fn!(bool, default_true, true);
serde_default_fn!(usize, default_qformer_layers, 1);
serde_default_fn!(usize, default_qformer_intermediate, 3072);
serde_default_fn!(f64, default_qformer_eps, 1e-12);
// transformers derives the Q-Former's heads from the vision width at 64 per head
const QFORMER_HEAD_DIM: usize = 64;

/// The Window Q-Former projectors' BLIP-2 Q-Former, as `Granite4VisionConfig` defaults it.
#[derive(Debug, Clone, Deserialize)]
pub struct QFormerConfig {
    pub hidden_size: Option<usize>,
    pub num_attention_heads: Option<usize>,
    #[serde(default = "default_qformer_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_qformer_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_qformer_eps")]
    pub layer_norm_eps: f64,
    pub encoder_hidden_size: Option<usize>,
}

impl Default for QFormerConfig {
    fn default() -> Self {
        Self {
            hidden_size: None,
            num_attention_heads: None,
            num_hidden_layers: default_qformer_layers(),
            intermediate_size: default_qformer_intermediate(),
            layer_norm_eps: default_qformer_eps(),
            encoder_hidden_size: None,
        }
    }
}

impl QFormerConfig {
    pub fn hidden(&self, vision_hidden: usize) -> usize {
        self.hidden_size.unwrap_or(vision_hidden)
    }

    pub fn heads(&self, vision_hidden: usize) -> usize {
        self.num_attention_heads
            .unwrap_or(self.hidden(vision_hidden) / QFORMER_HEAD_DIM)
    }

    pub fn encoder_hidden(&self, vision_hidden: usize) -> usize {
        self.encoder_hidden_size.unwrap_or(vision_hidden)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: crate::granite::Config,
    pub vision_config: SiglipVisionConfig,
    #[serde(alias = "image_token_id")]
    pub image_token_index: u32,
    pub image_grid_pinpoints: Vec<(u32, u32)>,
    /// `"query side/window side"`: each window of the tile's patch grid is read by query-side^2 queries.
    pub downsample_rate: String,
    /// `(vision hidden state, text layer)` pairs; a negative state counts from the last.
    pub deepstack_layer_map: Vec<(i64, usize)>,
    #[serde(default)]
    pub spatial_target_layers: Vec<usize>,
    #[serde(default = "default_spatial_vision_layer")]
    pub spatial_vision_layer: i64,
    #[serde(default = "default_true")]
    pub use_spatial_sampling: bool,
    #[serde(default = "default_true")]
    pub use_image_newline_parameter: bool,
    #[serde(default = "default_select_strategy")]
    pub vision_feature_select_strategy: String,
    #[serde(default)]
    pub qformer_config: QFormerConfig,
}

impl Config {
    /// The projector's `(query side, window side)`.
    pub fn downsample_sides(&self) -> inference_tensor::Result<(usize, usize)> {
        let parse = |side: &str| side.trim().parse::<usize>().ok().filter(|side| *side > 0);
        match self.downsample_rate.split_once('/') {
            Some((query, window)) if parse(query).is_some() && parse(window).is_some() => {
                Ok((parse(query).unwrap(), parse(window).unwrap()))
            }
            _ => inference_tensor::bail!(
                "Granite 4 Vision downsample_rate `{}` is not `query/window`",
                self.downsample_rate
            ),
        }
    }

    pub fn patches_per_side(&self) -> usize {
        self.vision_config.image_size / self.vision_config.patch_size
    }

    /// Feature rows (and columns) per tile once the projector has downsampled it.
    pub fn features_per_side(&self) -> inference_tensor::Result<usize> {
        let (query, window) = self.downsample_sides()?;
        Ok(self.patches_per_side() * query / window)
    }

    /// A vision hidden-state index as transformers reads `hidden_states[index]`: 0 is the embeddings.
    pub fn hidden_state_index(&self, index: i64) -> inference_tensor::Result<usize> {
        let states = self.vision_config.num_hidden_layers + 1;
        let resolved = if index < 0 {
            usize::try_from(index.unsigned_abs())
                .ok()
                .and_then(|back| states.checked_sub(back))
        } else {
            usize::try_from(index).ok().filter(|index| *index < states)
        };
        resolved.ok_or_else(|| {
            inference_tensor::Error::msg(format!(
                "Granite 4 Vision names vision hidden state {index} of {states}"
            ))
        })
    }
}

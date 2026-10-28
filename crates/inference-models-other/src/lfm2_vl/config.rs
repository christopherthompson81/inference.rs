use serde::{Deserialize, Serialize};

use crate::{layers::Activation, serde_default_fn};

serde_default_fn!(usize, default_image_token_id, 396);
serde_default_fn!(usize, default_projector_hidden_size, 2560);
serde_default_fn!(usize, default_downsample_factor, 2);
serde_default_fn!(bool, default_true, true);
serde_default_fn!(bool, default_projector_bias, true);
serde_default_fn!(bool, default_projector_use_layernorm, true);
serde_default_fn!(Activation, default_projector_hidden_act, Activation::Gelu);
serde_default_fn!(usize, default_hidden_size, 768);
serde_default_fn!(usize, default_intermediate_size, 3072);
serde_default_fn!(usize, default_num_hidden_layers, 12);
serde_default_fn!(usize, default_num_attention_heads, 12);
serde_default_fn!(usize, default_num_channels, 3);
serde_default_fn!(usize, default_num_patches, 256);
serde_default_fn!(usize, default_patch_size, 16);
serde_default_fn!(
    Activation,
    default_vision_hidden_act,
    Activation::GeluPytorchTanh
);
serde_default_fn!(f64, default_layer_norm_eps, 1e-6);
serde_default_fn!(usize, default_min_tiles, 2);
serde_default_fn!(usize, default_max_tiles, 10);
serde_default_fn!(usize, default_min_image_tokens, 64);
serde_default_fn!(usize, default_max_image_tokens, 256);
serde_default_fn!(usize, default_encoder_patch_size, 16);
serde_default_fn!(usize, default_tile_size, 512);
serde_default_fn!(f64, default_max_pixels_tolerance, 2.0);

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VisionConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_num_patches")]
    pub num_patches: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub text_config: crate::lfm2::Config,
    pub vision_config: VisionConfig,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: usize,
    #[serde(default = "default_projector_hidden_act")]
    pub projector_hidden_act: Activation,
    #[serde(default = "default_projector_hidden_size")]
    pub projector_hidden_size: usize,
    #[serde(default = "default_projector_bias")]
    pub projector_bias: bool,
    #[serde(default = "default_projector_use_layernorm")]
    pub projector_use_layernorm: bool,
    #[serde(default = "default_downsample_factor")]
    pub downsample_factor: usize,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_true")]
    pub do_image_splitting: bool,
    #[serde(default = "default_min_tiles")]
    pub min_tiles: usize,
    #[serde(default = "default_max_tiles")]
    pub max_tiles: usize,
    #[serde(default = "default_true")]
    pub use_thumbnail: bool,
    #[serde(default = "default_min_image_tokens")]
    pub min_image_tokens: usize,
    #[serde(default = "default_max_image_tokens")]
    pub max_image_tokens: usize,
    #[serde(default = "default_encoder_patch_size")]
    pub encoder_patch_size: usize,
    #[serde(default = "default_tile_size")]
    pub tile_size: usize,
    #[serde(default = "default_max_pixels_tolerance")]
    pub max_pixels_tolerance: f64,
}

/// How images are tiled and sized for the vision tower: the config's, or LFM2-VL's defaults.
#[derive(Clone, Debug)]
pub struct Tiling {
    pub downsample_factor: usize,
    pub do_image_splitting: bool,
    pub min_tiles: usize,
    pub max_tiles: usize,
    pub use_thumbnail: bool,
    pub min_image_tokens: usize,
    pub max_image_tokens: usize,
    pub encoder_patch_size: usize,
    pub tile_size: usize,
    pub max_pixels_tolerance: f64,
}

impl Default for Tiling {
    fn default() -> Self {
        Self {
            downsample_factor: default_downsample_factor(),
            do_image_splitting: default_true(),
            min_tiles: default_min_tiles(),
            max_tiles: default_max_tiles(),
            use_thumbnail: default_true(),
            min_image_tokens: default_min_image_tokens(),
            max_image_tokens: default_max_image_tokens(),
            encoder_patch_size: default_encoder_patch_size(),
            tile_size: default_tile_size(),
            max_pixels_tolerance: default_max_pixels_tolerance(),
        }
    }
}

impl Config {
    pub fn tiling(&self) -> Tiling {
        Tiling {
            downsample_factor: self.downsample_factor,
            do_image_splitting: self.do_image_splitting,
            min_tiles: self.min_tiles,
            max_tiles: self.max_tiles,
            use_thumbnail: self.use_thumbnail,
            min_image_tokens: self.min_image_tokens,
            max_image_tokens: self.max_image_tokens,
            encoder_patch_size: self.encoder_patch_size,
            tile_size: self.tile_size,
            max_pixels_tolerance: self.max_pixels_tolerance,
        }
    }
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_channels: default_num_channels(),
            num_patches: default_num_patches(),
            patch_size: default_patch_size(),
            hidden_act: default_vision_hidden_act(),
            layer_norm_eps: default_layer_norm_eps(),
        }
    }
}

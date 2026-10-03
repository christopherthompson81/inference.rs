use serde::Deserialize;

use crate::layers::Activation;

#[derive(Debug, Clone, Deserialize)]
pub struct Idefics3VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_act: Activation,
    pub layer_norm_eps: f64,
}

impl Idefics3VisionConfig {
    pub fn siglip(&self) -> inference_nn::vision::siglip::SiglipVisionConfig {
        inference_nn::vision::siglip::SiglipVisionConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_channels: self.num_channels,
            image_size: self.image_size,
            patch_size: self.patch_size,
            hidden_act: self.hidden_act,
            layer_norm_eps: self.layer_norm_eps,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Idefics3Config {
    pub image_token_id: usize,
    pub vision_config: Idefics3VisionConfig,
    pub text_config: crate::llama::Config,
    pub scale_factor: usize,
}

use crate::serde_default_fn;

serde_default_fn!(usize, default_window_size, 15);
serde_default_fn!(usize, default_downsample_rate, 5);
serde_default_fn!(f64, default_qformer_eps, 1e-12);

/// The CTC Conformer encoder's sizes (`GraniteSpeechEncoderConfig`).
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct EncoderConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub dim_head: usize,
    /// Frames per attention block; attention never crosses a block.
    pub context_size: usize,
    pub max_pos_emb: usize,
    pub conv_kernel_size: usize,
    pub conv_expansion_factor: usize,
    pub feedforward_mult: usize,
    /// The mid-stack CTC head's vocabulary.
    pub output_dim: usize,
    /// Granite Speech Plus: layer outputs (0 the input projection) concatenated with the last before the projector.
    #[serde(default)]
    pub cat_hidden_layers: Option<Vec<usize>>,
}

/// The projector's BLIP-2 Q-Former.
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct ProjectorConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_qformer_eps")]
    pub layer_norm_eps: f64,
    pub encoder_hidden_size: usize,
}

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct Config {
    pub text_config: crate::granite::Config,
    pub encoder_config: EncoderConfig,
    pub projector_config: ProjectorConfig,
    #[serde(alias = "audio_token_id")]
    pub audio_token_index: u32,
    /// Encoder frames per projector window; each window becomes `window_size / downsample_rate` tokens.
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    #[serde(default = "default_downsample_rate")]
    pub downsample_rate: usize,
    #[serde(default)]
    pub has_lora_adapter: bool,
}

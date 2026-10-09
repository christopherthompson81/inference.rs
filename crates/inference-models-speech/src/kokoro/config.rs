use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct IstftNetConfig {
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_rates: Vec<usize>,
    pub gen_istft_hop_size: usize,
    pub gen_istft_n_fft: usize,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlBertConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
}

/// The release's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct KokoroConfig {
    pub istftnet: IstftNetConfig,
    pub hidden_dim: usize,
    pub max_dur: usize,
    pub n_layer: usize,
    pub n_token: usize,
    pub style_dim: usize,
    pub text_encoder_kernel_size: usize,
    pub plbert: PlBertConfig,
    pub vocab: HashMap<String, u32>,
}

use candle_nn::Activation;
use serde::Deserialize;

/// Paddle `inference.yml` label order; HF's `id2label` collapses several of these (e.g. both formula kinds).
pub const LABELS: [&str; 25] = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
];

#[derive(Debug, Clone, Deserialize)]
pub struct BackboneConfig {
    pub arch: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PPDocLayoutV3Config {
    pub backbone_config: BackboneConfig,
    pub d_model: usize,
    pub encoder_hidden_dim: usize,
    pub encoder_in_channels: Vec<usize>,
    #[serde(alias = "feature_strides")]
    pub feat_strides: Vec<usize>,
    pub encoder_layers: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_attention_heads: usize,
    pub encode_proj_layers: Vec<usize>,
    pub positional_encoding_temperature: f64,
    pub encoder_activation_function: Activation,
    pub activation_function: Activation,
    pub hidden_expansion: f64,
    pub decoder_layers: usize,
    pub decoder_ffn_dim: usize,
    pub decoder_attention_heads: usize,
    pub decoder_n_points: usize,
    pub decoder_activation_function: Activation,
    pub decoder_in_channels: Vec<usize>,
    pub num_feature_levels: usize,
    pub num_queries: usize,
    pub layer_norm_eps: f64,
    pub batch_norm_eps: f64,
    #[serde(default = "default_num_prototypes")]
    pub num_prototypes: usize,
    pub mask_feature_channels: Vec<usize>,
    pub x4_feat_dim: usize,
    pub global_pointer_head_size: usize,
    pub id2label: std::collections::HashMap<String, String>,
    #[serde(default = "default_true")]
    pub mask_enhanced: bool,
    #[serde(default)]
    pub learn_initial_query: bool,
    #[serde(default)]
    pub normalize_before: bool,
    #[serde(default)]
    pub anchor_image_size: Option<Vec<usize>>,
    #[serde(default)]
    pub eval_size: Option<Vec<usize>>,
}

fn default_true() -> bool {
    true
}

fn default_num_prototypes() -> usize {
    32
}

impl PPDocLayoutV3Config {
    pub fn num_labels(&self) -> usize {
        self.id2label.len()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Size {
    pub height: usize,
    pub width: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PPDocLayoutV3PreprocessorConfig {
    pub size: Size,
    pub rescale_factor: f64,
    pub image_mean: Vec<f64>,
    pub image_std: Vec<f64>,
}

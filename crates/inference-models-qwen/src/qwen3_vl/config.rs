use inference_quant::QuantizedConfig;

pub use crate::vision::qwen3_vl::VisionConfig;
pub use inference_nn::decoder::qwen3_vl::{MRopeScaling, TextConfig};

#[derive(Debug, Clone, inference_json::FromJson)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub tie_word_embeddings: bool,
    /// Top-level quantization_config takes precedence
    pub quantization_config: Option<QuantizedConfig>,
}

use inference_models_gemma::diffusion_gemma::config::DiffusionGemmaConfig;
use inference_models_gemma::gemma4::config::{
    Gemma4BidirectionalAttention, Gemma4Config, Gemma4VisionConfig,
};
use inference_models_gemma::gemma4::inputs_processor::{
    AUDIO_TOKEN, BOA_TOKEN, BOI_TOKEN, EOA_TOKEN, EOI_TOKEN, Gemma4ImageProcessor, IMAGE_TOKEN,
    VIDEO_TOKEN,
};
use inference_models_gemma::loaders::{DiffusionGemmaLoader, Gemma4Loader};

use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;
use crate::vision_models::processor_config::ProcessorConfig;

// The patch size a text-only checkpoint reports
const NO_VISION_PATCH_SIZE: usize = 16;
const DEFAULT_AUDIO_SEQ_LEN: usize = 750;
const TEMPLATE_ACTION: MessagesAction = MessagesAction::KeepWithAudioAfterText;
const DEFAULT_VIDEO_MAX_SOFT_TOKENS: usize = 70;
const SPECIAL_TOKENS: &[&str] = &[
    IMAGE_TOKEN,
    BOI_TOKEN,
    EOI_TOKEN,
    AUDIO_TOKEN,
    BOA_TOKEN,
    EOA_TOKEN,
    VIDEO_TOKEN,
];

pub struct Gemma4ProcessorSettings {
    pub processor_config: ProcessorConfig,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub default_output_length: usize,
    pub supports_images: bool,
    pub supports_audio: bool,
    pub raw_audio_frame_size: Option<usize>,
    pub is_unified: bool,
    /// Tokens fed per decode step; block-diffusion models set this to the canvas length.
    pub decode_window: Option<usize>,
    pub bidirectional_attention: Gemma4BidirectionalAttention,
    pub vision_attention_on_full_layers: bool,
}

impl Gemma4ProcessorSettings {
    fn image_processor(self) -> Gemma4ImageProcessor {
        let pooled = self.pooling_kernel_size * self.pooling_kernel_size;
        let video_max_soft_tokens = self
            .processor_config
            .video_max_soft_tokens
            .unwrap_or(DEFAULT_VIDEO_MAX_SOFT_TOKENS);
        Gemma4ImageProcessor {
            patch_size: self.patch_size,
            pooling_kernel_size: self.pooling_kernel_size,
            default_output_length: self.default_output_length,
            max_patches: self.default_output_length * pooled,
            audio_seq_length: self
                .processor_config
                .audio_seq_length
                .unwrap_or(DEFAULT_AUDIO_SEQ_LEN),
            raw_audio_frame_size: self.raw_audio_frame_size,
            video_max_patches: video_max_soft_tokens * pooled,
            is_unified: self.is_unified,
            supports_images: self.supports_images,
            supports_audio: self.supports_audio,
            decode_window: self.decode_window,
            bidirectional_attention: self.bidirectional_attention,
            vision_attention_on_full_layers: self.vision_attention_on_full_layers,
        }
    }
}

/// Patch size, pooling kernel, default output length and whether there is a vision tower.
fn vision_geometry(vision: Option<&Gemma4VisionConfig>) -> (usize, usize, usize, bool) {
    vision.map_or((NO_VISION_PATCH_SIZE, 1, 0, false), |v| {
        (
            v.patch_size,
            v.pooling_kernel_size,
            v.default_output_length,
            true,
        )
    })
}

processor_factory!(Gemma4Loader => |model_config, processor_config, _, _| {
    let cfg = Gemma4Config::from_json(model_config).expect("Failed to parse Gemma4Config");
    let (patch_size, pooling_kernel_size, default_output_length, supports_images) =
        vision_geometry(cfg.vision_config.as_ref());
    let raw_audio_frame_size = cfg
        .audio_config
        .as_ref()
        .and_then(|audio_cfg| cfg.is_unified().then_some(audio_cfg.input_feat_size()));
    let settings = Gemma4ProcessorSettings {
        processor_config: processor_config.unwrap_or_default(),
        patch_size,
        pooling_kernel_size,
        default_output_length,
        supports_images,
        supports_audio: cfg.audio_config.is_some(),
        raw_audio_frame_size,
        is_unified: cfg.is_unified(),
        decode_window: None,
        bidirectional_attention: cfg.text_config.bidirectional_attention(),
        vision_attention_on_full_layers: false,
    };
    media_processor(settings.image_processor(), SPECIAL_TOKENS, TEMPLATE_ACTION)
});

processor_factory!(DiffusionGemmaLoader => |model_config, processor_config, _, _| {
    let cfg = DiffusionGemmaConfig::from_json(model_config)
        .expect("Failed to parse DiffusionGemmaConfig");
    let (patch_size, pooling_kernel_size, default_output_length, supports_images) =
        vision_geometry(cfg.vision_config.as_ref());
    let settings = Gemma4ProcessorSettings {
        processor_config: processor_config.unwrap_or_default(),
        patch_size,
        pooling_kernel_size,
        default_output_length,
        supports_images,
        supports_audio: false,
        raw_audio_frame_size: None,
        is_unified: false,
        decode_window: Some(cfg.canvas_length),
        bidirectional_attention: cfg.text_config.bidirectional_attention(),
        vision_attention_on_full_layers: true,
    };
    media_processor(settings.image_processor(), SPECIAL_TOKENS, TEMPLATE_ACTION)
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_audio_seq_length_to_reference_cap() {
        let processor = Gemma4ProcessorSettings {
            processor_config: ProcessorConfig::default(),
            patch_size: 16,
            pooling_kernel_size: 3,
            default_output_length: 280,
            supports_images: true,
            supports_audio: true,
            raw_audio_frame_size: None,
            is_unified: false,
            decode_window: None,
            bidirectional_attention: Gemma4BidirectionalAttention::Vision,
            vision_attention_on_full_layers: false,
        }
        .image_processor();
        assert_eq!(processor.audio_seq_length, 750);
    }
}

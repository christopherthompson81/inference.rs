use std::sync::Arc;

use inference_models_gemma::diffusion_gemma::config::DiffusionGemmaConfig;
use inference_models_gemma::gemma4::config::{Gemma4BidirectionalAttention, Gemma4Config};
use inference_models_gemma::gemma4::inputs_processor::{
    AUDIO_TOKEN, BOA_TOKEN, BOI_TOKEN, EOA_TOKEN, EOI_TOKEN, Gemma4ImageProcessor, IMAGE_TOKEN,
    VIDEO_TOKEN,
};
use inference_models_gemma::loaders::{DiffusionGemmaLoader, Gemma4Loader};

use crate::pipeline::{InputsProcessor, MessagesAction, MultimodalProcessorFactory, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;

pub struct Gemma4Processor {
    patch_size: usize,
    pooling_kernel_size: usize,
    default_output_length: usize,
    max_patches: usize,
    audio_seq_length: usize,
    raw_audio_frame_size: Option<usize>,
    video_max_soft_tokens: usize,
    is_unified: bool,
    supports_images: bool,
    supports_audio: bool,
    decode_window: Option<usize>,
    bidirectional_attention: Gemma4BidirectionalAttention,
    vision_attention_on_full_layers: bool,
}

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

impl Gemma4Processor {
    pub fn new(settings: Gemma4ProcessorSettings) -> Self {
        let Gemma4ProcessorSettings {
            processor_config,
            patch_size,
            pooling_kernel_size,
            default_output_length,
            supports_images,
            supports_audio,
            raw_audio_frame_size,
            is_unified,
            decode_window,
            bidirectional_attention,
            vision_attention_on_full_layers,
        } = settings;
        let max_patches = default_output_length * pooling_kernel_size * pooling_kernel_size;
        let audio_seq_length = processor_config.audio_seq_length.unwrap_or(750);
        let video_max_soft_tokens = processor_config.video_max_soft_tokens.unwrap_or(70);

        Self {
            patch_size,
            pooling_kernel_size,
            default_output_length,
            max_patches,
            audio_seq_length,
            raw_audio_frame_size,
            video_max_soft_tokens,
            is_unified,
            supports_images,
            supports_audio,
            decode_window,
            bidirectional_attention,
            vision_attention_on_full_layers,
        }
    }
}

impl Processor for Gemma4Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        let video_max_patches =
            self.video_max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size;
        Arc::new(MediaInputsProcessor(Arc::new(Gemma4ImageProcessor {
            patch_size: self.patch_size,
            pooling_kernel_size: self.pooling_kernel_size,
            default_output_length: self.default_output_length,
            max_patches: self.max_patches,
            audio_seq_length: self.audio_seq_length,
            raw_audio_frame_size: self.raw_audio_frame_size,
            video_max_patches,
            is_unified: self.is_unified,
            supports_images: self.supports_images,
            supports_audio: self.supports_audio,
            decode_window: self.decode_window,
            bidirectional_attention: self.bidirectional_attention,
            vision_attention_on_full_layers: self.vision_attention_on_full_layers,
        })))
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[
            IMAGE_TOKEN,
            BOI_TOKEN,
            EOI_TOKEN,
            AUDIO_TOKEN,
            BOA_TOKEN,
            EOA_TOKEN,
            VIDEO_TOKEN,
        ]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::KeepWithAudioAfterText
    }
}

impl MultimodalProcessorFactory for Gemma4Loader {
    fn get_processor(
        &self,
        config: &str,
        processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        let cfg = Gemma4Config::from_json(config).expect("Failed to parse Gemma4Config");
        let (patch_size, pooling_kernel_size, default_output_length, supports_images) = cfg
            .vision_config
            .as_ref()
            .map_or((16, 1, 0, false), |vision_cfg| {
                (
                    vision_cfg.patch_size,
                    vision_cfg.pooling_kernel_size,
                    vision_cfg.default_output_length,
                    true,
                )
            });
        let raw_audio_frame_size = cfg
            .audio_config
            .as_ref()
            .and_then(|audio_cfg| cfg.is_unified().then_some(audio_cfg.input_feat_size()));
        Arc::new(Gemma4Processor::new(Gemma4ProcessorSettings {
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
        }))
    }
}

impl MultimodalProcessorFactory for DiffusionGemmaLoader {
    fn get_processor(
        &self,
        config: &str,
        processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        let cfg =
            DiffusionGemmaConfig::from_json(config).expect("Failed to parse DiffusionGemmaConfig");
        let (patch_size, pooling_kernel_size, default_output_length, supports_images) = cfg
            .vision_config
            .as_ref()
            .map_or((16, 1, 0, false), |vision_cfg| {
                (
                    vision_cfg.patch_size,
                    vision_cfg.pooling_kernel_size,
                    vision_cfg.default_output_length,
                    true,
                )
            });
        Arc::new(Gemma4Processor::new(Gemma4ProcessorSettings {
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
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_audio_seq_length_to_reference_cap() {
        let processor = Gemma4Processor::new(Gemma4ProcessorSettings {
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
        });
        assert_eq!(processor.audio_seq_length, 750);
    }
}

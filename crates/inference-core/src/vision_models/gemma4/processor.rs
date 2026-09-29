use std::sync::Arc;

use inference_models_gemma::gemma4::config::Gemma4BidirectionalAttention;
use inference_models_gemma::gemma4::inputs_processor::{
    Gemma4ImageProcessor, AUDIO_TOKEN, BOA_TOKEN, BOI_TOKEN, EOA_TOKEN, EOI_TOKEN, IMAGE_TOKEN,
    VIDEO_TOKEN,
};

use crate::pipeline::{InputsProcessor, MessagesAction, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
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

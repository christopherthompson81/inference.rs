use std::sync::Arc;

use inference_models_gemma::gemma3n::inputs_processor::{
    AUDIO_TOKEN, BOA_TOKEN, BOI_TOKEN, EOA_TOKEN, EOI_TOKEN, Gemma3nImageProcessor, IMAGE_TOKEN,
};
use inference_models_gemma::loaders::Gemma3nLoader;

use crate::pipeline::{InputsProcessor, MessagesAction, MultimodalProcessorFactory, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;

pub struct Gemma3nProcessor {
    vision_soft_tokens_per_image: usize,
    audio_seq_length: usize,
    supports_images: bool,
    supports_audio: bool,
}

impl Gemma3nProcessor {
    pub fn new(processor_config: ProcessorConfig, supports_images: bool) -> Self {
        // Default to 256 soft tokens per image if not specified
        let vision_soft_tokens_per_image = processor_config.image_seq_len.unwrap_or(256);
        // Default to 188 audio tokens as per transformers implementation
        let audio_seq_length = processor_config.audio_seq_length.unwrap_or(188);

        Self {
            vision_soft_tokens_per_image,
            audio_seq_length,
            supports_images,
            supports_audio: true, // Enable audio support
        }
    }

    fn create_full_image_sequence(&self) -> String {
        // Create the full image token sequence: "\n\n<boi>{repeated image tokens}<eoi>\n\n"
        let image_tokens_expanded =
            vec![IMAGE_TOKEN.to_string(); self.vision_soft_tokens_per_image].join("");
        format!("\n\n{BOI_TOKEN}{image_tokens_expanded}{EOI_TOKEN}\n\n")
    }
}

impl Processor for Gemma3nProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(Arc::new(Gemma3nImageProcessor::new(
            self.supports_images,
            self.supports_audio,
            self.create_full_image_sequence(),
            self.audio_seq_length,
        ))))
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[
            IMAGE_TOKEN,
            BOI_TOKEN,
            EOI_TOKEN,
            AUDIO_TOKEN,
            BOA_TOKEN,
            EOA_TOKEN,
        ]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl MultimodalProcessorFactory for Gemma3nLoader {
    fn get_processor(
        &self,
        _config: &str,
        processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Gemma3nProcessor::new(
            processor_config.unwrap_or_default(),
            true,
        ))
    }
}

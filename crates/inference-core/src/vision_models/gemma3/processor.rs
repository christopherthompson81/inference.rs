use std::sync::Arc;

use inference_models_gemma::gemma3::inputs_processor::{
    BOI_TOKEN, EOI_TOKEN, Gemma3ImageProcessor, IMAGE_TOKEN,
};

use crate::pipeline::{InputsProcessor, MessagesAction, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::processor_config::ProcessorConfig;

pub struct Gemma3Processor {
    full_image_sequence: String,
    supports_images: bool,
}

impl Gemma3Processor {
    pub fn new(processor_config: ProcessorConfig, supports_images: bool) -> Self {
        let image_tokens_expanded =
            vec![IMAGE_TOKEN.to_string(); processor_config.image_seq_len.unwrap_or(256)].join("");
        let full_image_sequence = format!("\n\n{BOI_TOKEN}{image_tokens_expanded}{EOI_TOKEN}\n\n");

        Self {
            full_image_sequence,
            supports_images,
        }
    }
}

impl Processor for Gemma3Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(Arc::new(Gemma3ImageProcessor::new(
            self.full_image_sequence.clone(),
            self.supports_images,
        ))))
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[BOI_TOKEN, EOI_TOKEN, IMAGE_TOKEN]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

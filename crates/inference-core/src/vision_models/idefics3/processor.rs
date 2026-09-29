use std::sync::Arc;

use inference_models_llama::idefics3::inputs_processor::Idefics3ImageProcessor;

use crate::pipeline::{InputsProcessor, MessagesAction, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;

pub struct Idefics3Processor {
    config: ProcessorConfig,
    max_edge: Option<u32>,
}

impl Idefics3Processor {
    pub fn new(
        config: ProcessorConfig,
        _preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Self {
        Self { config, max_edge }
    }
}

impl Processor for Idefics3Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        // Default image_seq_len is 169.
        Arc::new(MediaInputsProcessor(Arc::new(Idefics3ImageProcessor::new(
            self.max_edge,
            self.config.image_seq_len.unwrap_or(169),
        ))))
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &["<fake_token_around_image>", "<image>", "<end_of_utterance>"]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

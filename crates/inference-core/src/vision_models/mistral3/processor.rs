use std::sync::Arc;

use inference_models_llama::loaders::Mistral3Loader;
use inference_models_llama::mistral3::inputs_processor::Mistral3ImageProcessor;

use crate::pipeline::{InputsProcessor, MessagesAction, MultimodalProcessorFactory, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;

pub struct Mistral3Processor {
    image_break_token: String,
    image_end_token: String,
    image_token: String,
    patch_size: usize,
    spatial_merge_size: usize,
}

impl Mistral3Processor {
    pub fn new(processor_config: ProcessorConfig) -> Self {
        Self {
            image_break_token: processor_config.image_break_token.unwrap().clone(),
            image_end_token: processor_config.image_end_token.unwrap().clone(),
            image_token: processor_config.image_token.unwrap().clone(),
            patch_size: processor_config.patch_size.unwrap(),
            spatial_merge_size: processor_config.spatial_merge_size.unwrap(),
        }
    }
}

impl Processor for Mistral3Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(Arc::new(Mistral3ImageProcessor::new(
            self.image_break_token.clone(),
            self.image_end_token.clone(),
            self.image_token.clone(),
            self.patch_size,
            self.spatial_merge_size,
        ))))
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl MultimodalProcessorFactory for Mistral3Loader {
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Mistral3Processor::new(processor_config.unwrap_or_default()))
    }
}

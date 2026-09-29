use std::sync::Arc;

use inference_models_qwen::loaders::{Qwen2VLLoader, Qwen2_5VLLoader};
use inference_models_qwen::qwen2vl::inputs_processor::{
    Qwen2VLImageProcessor, IMAGE_PAD, PLACEHOLDER, VIDEO_PAD,
};

use crate::pipeline::{InputsProcessor, MessagesAction, MultimodalProcessorFactory, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;

pub struct Qwen2VLProcessor(Arc<Qwen2VLImageProcessor>);

impl Qwen2VLProcessor {
    pub fn new(max_edge: Option<u32>) -> Self {
        Self(Arc::new(Qwen2VLImageProcessor::new(max_edge)))
    }
}

impl Processor for Qwen2VLProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(self.0.clone()))
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[IMAGE_PAD, VIDEO_PAD, PLACEHOLDER]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

impl MultimodalProcessorFactory for Qwen2VLLoader {
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Qwen2VLProcessor::new(max_edge))
    }
}

impl MultimodalProcessorFactory for Qwen2_5VLLoader {
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Qwen2VLProcessor::new(max_edge))
    }
}

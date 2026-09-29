use std::sync::Arc;

use inference_models_qwen::loaders::{
    Qwen3VLLoader, Qwen3VLMoELoader, Qwen3_5Loader, Qwen3_5MoeLoader,
};
use inference_models_qwen::qwen2vl::inputs_processor::{IMAGE_PAD, PLACEHOLDER, VIDEO_PAD};
use inference_models_qwen::qwen3_vl::inputs_processor::Qwen3VLImageProcessor;

use crate::pipeline::{InputsProcessor, MessagesAction, MultimodalProcessorFactory, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;

pub struct Qwen3VLProcessor(Arc<Qwen3VLImageProcessor>);

impl Qwen3VLProcessor {
    pub fn new(max_edge: Option<u32>) -> Self {
        Self(Arc::new(Qwen3VLImageProcessor::new(max_edge)))
    }
}

impl Processor for Qwen3VLProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(self.0.clone()))
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[IMAGE_PAD, VIDEO_PAD, PLACEHOLDER]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl MultimodalProcessorFactory for Qwen3VLLoader {
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Qwen3VLProcessor::new(max_edge))
    }
}

impl MultimodalProcessorFactory for Qwen3VLMoELoader {
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Qwen3VLProcessor::new(max_edge))
    }
}

impl MultimodalProcessorFactory for Qwen3_5Loader {
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Qwen3VLProcessor::new(max_edge))
    }
}

impl MultimodalProcessorFactory for Qwen3_5MoeLoader {
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Qwen3VLProcessor::new(max_edge))
    }
}

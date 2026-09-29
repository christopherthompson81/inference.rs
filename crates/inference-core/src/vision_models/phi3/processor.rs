use std::sync::Arc;

use inference_models_phi::loaders::Phi3VLoader;
use inference_models_phi::phi3_vision::inputs_processor::Phi3InputsProcessor;

use crate::pipeline::{InputsProcessor, MessagesAction, MultimodalProcessorFactory, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;

pub struct Phi3Processor {
    inputs_processor: Arc<Phi3InputsProcessor>,
}

impl Phi3Processor {
    pub(crate) fn new_processor(
        _: Option<ProcessorConfig>,
        _: PreProcessorConfig,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Self {
            inputs_processor: Arc::new(Phi3InputsProcessor::default()),
        })
    }
}

impl Processor for Phi3Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(self.inputs_processor.clone()))
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

impl MultimodalProcessorFactory for Phi3VLoader {
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Phi3Processor::new_processor(processor_config, preprocessor_config)
    }
}

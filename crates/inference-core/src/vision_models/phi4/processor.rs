use std::sync::Arc;

use inference_models_phi::loaders::Phi4MMLoader;
use inference_models_phi::phi4::inputs_processor::Phi4MMInputsProcessor;

use crate::pipeline::{InputsProcessor, MessagesAction, MultimodalProcessorFactory, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;

pub struct Phi4MMProcessor {
    inputs_processor: Arc<Phi4MMInputsProcessor>,
}

impl Phi4MMProcessor {
    pub(crate) fn new_processor(
        _: Option<ProcessorConfig>,
        pre_processor_config: PreProcessorConfig,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Self {
            inputs_processor: Arc::new(Phi4MMInputsProcessor::new(&pre_processor_config)),
        })
    }
}

impl Processor for Phi4MMProcessor {
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

impl MultimodalProcessorFactory for Phi4MMLoader {
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Phi4MMProcessor::new_processor(processor_config, preprocessor_config)
    }
}

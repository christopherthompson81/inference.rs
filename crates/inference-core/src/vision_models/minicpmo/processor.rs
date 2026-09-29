use std::sync::Arc;

use inference_models_qwen::loaders::MiniCpmOLoader;
use inference_models_qwen::minicpmo::inputs_processor::{
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_SLICE_END_TOKEN,
    DEFAULT_SLICE_START_TOKEN, DEFAULT_UNK_TOKEN, MiniCpmOImageProcessor,
};

use crate::pipeline::{InputsProcessor, MessagesAction, MultimodalProcessorFactory, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;

pub struct MiniCpmOProcessor(Arc<MiniCpmOImageProcessor>);

impl MiniCpmOProcessor {
    pub fn new(
        _config: ProcessorConfig,
        preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Self {
        Self(Arc::new(MiniCpmOImageProcessor::new(preprocessor_config)))
    }
}

impl Processor for MiniCpmOProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(self.0.clone()))
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_SLICE_START_TOKEN,
            DEFAULT_SLICE_END_TOKEN,
            DEFAULT_UNK_TOKEN,
        ]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

impl MultimodalProcessorFactory for MiniCpmOLoader {
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(MiniCpmOProcessor::new(
            processor_config.unwrap_or_default(),
            preprocessor_config,
            max_edge,
        ))
    }
}

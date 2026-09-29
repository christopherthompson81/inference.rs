use std::sync::Arc;

use inference_models_qwen::minicpmo::inputs_processor::{
    MiniCpmOImageProcessor, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_SLICE_END_TOKEN,
    DEFAULT_SLICE_START_TOKEN, DEFAULT_UNK_TOKEN,
};

use crate::pipeline::{InputsProcessor, MessagesAction, Processor};
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

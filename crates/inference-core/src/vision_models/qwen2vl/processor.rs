use std::sync::Arc;

use inference_models_qwen::qwen2vl::inputs_processor::{
    Qwen2VLImageProcessor, IMAGE_PAD, PLACEHOLDER, VIDEO_PAD,
};

use crate::pipeline::{InputsProcessor, MessagesAction, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;

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

use std::sync::Arc;

use inference_models_qwen::qwen2vl::inputs_processor::{IMAGE_PAD, PLACEHOLDER, VIDEO_PAD};
use inference_models_qwen::qwen3_vl::inputs_processor::Qwen3VLImageProcessor;

use crate::pipeline::{InputsProcessor, MessagesAction, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;

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

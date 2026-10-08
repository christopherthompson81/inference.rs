use std::sync::Arc;

use inference_models_llama::loaders::VLlamaLoader;
use inference_models_llama::mllama::inputs_processor::{IMAGE_TOKEN, MLlamaImageProcessor};

use crate::pipeline::{InputsProcessor, MessagesAction, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;

pub struct MLlamaProcessor;

impl MLlamaProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl Processor for MLlamaProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(Arc::new(
            MLlamaImageProcessor::default(),
        )))
    }

    fn retain_prefix_cached_images(&self) -> bool {
        true
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[IMAGE_TOKEN, "<|python_tag|>"]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

processor_factory!(VLlamaLoader => |_, _, _, _| Arc::new(MLlamaProcessor::new()));

#[cfg(test)]
mod tests {
    use super::MLlamaProcessor;
    use crate::pipeline::Processor;

    #[test]
    fn normal_prefix_cache_retains_cross_attention_images() {
        assert!(MLlamaProcessor.retain_prefix_cached_images());
    }
}

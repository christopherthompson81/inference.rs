use std::sync::Arc;

use inference_models_llama::loaders::VLlamaLoader;
use inference_models_llama::mllama::inputs_processor::{IMAGE_TOKEN, MLlamaImageProcessor};

use crate::pipeline::{InputsProcessor, MessagesAction, MultimodalProcessorFactory, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;

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

impl MultimodalProcessorFactory for VLlamaLoader {
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(MLlamaProcessor::new())
    }
}

#[cfg(test)]
mod tests {
    use super::MLlamaProcessor;
    use crate::pipeline::Processor;

    #[test]
    fn normal_prefix_cache_retains_cross_attention_images() {
        assert!(MLlamaProcessor.retain_prefix_cached_images());
    }
}

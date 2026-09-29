use std::sync::Arc;

use either::Either;
use indexmap::IndexMap;
use inference_models_other::loaders::PaddleOcrVlLoader;
use inference_models_other::paddleocr_vl::inputs_processor::{
    IMAGE_END, IMAGE_PLACEHOLDER, IMAGE_START, PaddleOcrVlImageProcessor,
};
use serde_json::Value;

use crate::{
    MessageContent, Tool,
    pipeline::{
        InputsProcessor, MessagesAction, MultimodalProcessorFactory, Processor,
        processing::default_process,
    },
    request::ReasoningEffort,
    vision_models::{
        media_host::MediaInputsProcessor, preprocessor_config::PreProcessorConfig,
        processor_config::ProcessorConfig,
    },
};

pub struct PaddleOcrVlProcessor;

impl Processor for PaddleOcrVlProcessor {
    // The template iterates content as typed parts; a bare string would be iterated per char and dropped.
    fn process(
        &self,
        pipeline: &dyn crate::pipeline::Pipeline,
        messages: Vec<IndexMap<String, MessageContent>>,
        add_generation_prompt: bool,
        add_special_tokens: bool,
        enable_thinking: Option<bool>,
        reasoning_effort: Option<ReasoningEffort>,
        tools: Vec<Tool>,
    ) -> anyhow::Result<(Vec<u32>, String)> {
        let messages = messages
            .into_iter()
            .map(|message| {
                message
                    .into_iter()
                    .map(|(key, value)| match (key.as_str(), value) {
                        ("content", Either::Left(text)) => (
                            key,
                            Either::Right(vec![IndexMap::from([
                                ("type".to_string(), Value::String("text".to_string())),
                                ("text".to_string(), Value::String(text)),
                            ])]),
                        ),
                        (_, value) => (key, value),
                    })
                    .collect()
            })
            .collect();
        default_process(
            pipeline,
            messages,
            add_generation_prompt,
            add_special_tokens,
            enable_thinking,
            reasoning_effort,
            self.template_action(),
            tools,
        )
    }
    // The model takes every image's patches and skips the cached ones itself, so a prefix hit must not drop them.
    fn retain_prefix_cached_images(&self) -> bool {
        true
    }

    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(Arc::new(PaddleOcrVlImageProcessor)))
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[IMAGE_START, IMAGE_PLACEHOLDER, IMAGE_END]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl MultimodalProcessorFactory for PaddleOcrVlLoader {
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(PaddleOcrVlProcessor)
    }
}

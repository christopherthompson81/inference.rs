use std::sync::Arc;

use inference_models_qwen::muse_glimmer::inputs_processor::{
    MuseGlimmerImageProcessor, IMAGE_END, IMAGE_START, IMAGE_TOKEN, VIDEO_END, VIDEO_SEPARATOR,
    VIDEO_START, VIDEO_TOKEN,
};

use crate::pipeline::{InputsProcessor, MessagesAction, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;

pub struct MuseGlimmerProcessor(Arc<MuseGlimmerImageProcessor>);

impl MuseGlimmerProcessor {
    pub fn new(
        config: &PreProcessorConfig,
        max_edge: Option<u32>,
        gguf_collapsed_temporal: bool,
    ) -> anyhow::Result<Self> {
        Ok(Self(Arc::new(MuseGlimmerImageProcessor::new(
            config,
            max_edge,
            gguf_collapsed_temporal,
        )?)))
    }
}

impl Processor for MuseGlimmerProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(self.0.clone()))
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[
            IMAGE_TOKEN,
            IMAGE_START,
            IMAGE_END,
            VIDEO_TOKEN,
            VIDEO_START,
            VIDEO_END,
            VIDEO_SEPARATOR,
        ]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

use std::sync::Arc;

use inference_models_qwen::loaders::MuseGlimmerLoader;
use inference_models_qwen::muse_glimmer::Config as MuseGlimmerConfig;
use inference_models_qwen::muse_glimmer::inputs_processor::{
    IMAGE_END, IMAGE_START, IMAGE_TOKEN, MuseGlimmerImageProcessor, VIDEO_END, VIDEO_SEPARATOR,
    VIDEO_START, VIDEO_TOKEN,
};

use crate::pipeline::{InputsProcessor, MessagesAction, MultimodalProcessorFactory, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;

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

impl MultimodalProcessorFactory for MuseGlimmerLoader {
    fn get_processor(
        &self,
        model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        let cfg = MuseGlimmerConfig::from_json(model_config)
            .expect("Failed to parse Muse-Glimmer config");
        Arc::new(
            MuseGlimmerProcessor::new(&preprocessor_config, max_edge, cfg.gguf_collapsed_temporal)
                .expect("Failed to create Muse-Glimmer processor"),
        )
    }
}

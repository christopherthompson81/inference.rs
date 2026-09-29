use std::sync::Arc;

use inference_models_other::lfm2_vl::inputs_processor::{
    IMAGE_END, IMAGE_START, IMAGE_THUMBNAIL, IMAGE_TOKEN, Lfm2VlImageProcessor,
};
use inference_models_other::loaders::Lfm2VlLoader;

use super::config::Config;
use crate::pipeline::{InputsProcessor, MessagesAction, MultimodalProcessorFactory, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;

pub struct Lfm2VlProcessor(Arc<Lfm2VlImageProcessor>);

impl Lfm2VlProcessor {
    pub fn new(config: &Config, preprocessor_config: &PreProcessorConfig) -> Self {
        Self(Arc::new(Lfm2VlImageProcessor::new(
            config,
            preprocessor_config,
        )))
    }
}

impl Processor for Lfm2VlProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(self.0.clone()))
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[IMAGE_TOKEN, IMAGE_START, IMAGE_END, IMAGE_THUMBNAIL]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl MultimodalProcessorFactory for Lfm2VlLoader {
    fn get_processor(
        &self,
        model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        let cfg = Config::from_json(model_config).expect("Failed to parse LFM2-VL config");
        Arc::new(Lfm2VlProcessor::new(&cfg, &preprocessor_config))
    }
}

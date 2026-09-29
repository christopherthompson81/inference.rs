use std::sync::Arc;

use inference_models_other::lfm2_vl::inputs_processor::{
    Lfm2VlImageProcessor, IMAGE_END, IMAGE_START, IMAGE_THUMBNAIL, IMAGE_TOKEN,
};

use super::config::Config;
use crate::pipeline::{InputsProcessor, MessagesAction, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;

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

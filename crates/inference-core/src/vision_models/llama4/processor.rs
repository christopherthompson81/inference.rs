use std::sync::Arc;

use inference_models_llama::llama4::inputs_processor::{
    IMAGE_END, IMAGE_START, IMAGE_TOKEN, Llama4ImageProcessor, PATCH, TILE_X_SEP, TILE_Y_SEP,
};
use inference_models_llama::loaders::VLlama4Loader;

use crate::pipeline::{InputsProcessor, MessagesAction, MultimodalProcessorFactory, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;

pub struct Llama4Processor {
    patch_size: usize,
    downsample_ratio: usize,
}

impl Llama4Processor {
    pub fn new(cfg: &ProcessorConfig) -> Self {
        Self {
            patch_size: cfg.patch_size.unwrap_or(14),
            downsample_ratio: (1. / cfg.pixel_shuffle_ratio.unwrap_or(0.5).powi(2)).round()
                as usize,
        }
    }
}

impl Processor for Llama4Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(Arc::new(Llama4ImageProcessor {
            patch_size: self.patch_size,
            downsample_ratio: self.downsample_ratio,
        })))
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[
            IMAGE_START,
            IMAGE_END,
            PATCH,
            TILE_X_SEP,
            TILE_Y_SEP,
            IMAGE_TOKEN,
        ]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

impl MultimodalProcessorFactory for VLlama4Loader {
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Llama4Processor::new(&processor_config.unwrap_or_default()))
    }
}

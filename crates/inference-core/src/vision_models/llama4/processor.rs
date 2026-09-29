use std::sync::Arc;

use inference_models_llama::llama4::inputs_processor::{
    Llama4ImageProcessor, IMAGE_END, IMAGE_START, IMAGE_TOKEN, PATCH, TILE_X_SEP, TILE_Y_SEP,
};

use crate::pipeline::{InputsProcessor, MessagesAction, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;
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

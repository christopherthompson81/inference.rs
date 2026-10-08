use inference_models_llama::llama4::inputs_processor::{
    IMAGE_END, IMAGE_START, IMAGE_TOKEN, Llama4ImageProcessor, PATCH, TILE_X_SEP, TILE_Y_SEP,
};
use inference_models_llama::loaders::VLlama4Loader;

use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

const DEFAULT_PATCH_SIZE: usize = 14;
const DEFAULT_PIXEL_SHUFFLE_RATIO: f32 = 0.5;
const SPECIAL_TOKENS: &[&str] = &[
    IMAGE_START,
    IMAGE_END,
    PATCH,
    TILE_X_SEP,
    TILE_Y_SEP,
    IMAGE_TOKEN,
];

processor_factory!(VLlama4Loader => |_, processor_config, _, _| {
    let cfg = processor_config.unwrap_or_default();
    let ratio = cfg.pixel_shuffle_ratio.unwrap_or(DEFAULT_PIXEL_SHUFFLE_RATIO);
    let inputs = Llama4ImageProcessor {
        patch_size: cfg.patch_size.unwrap_or(DEFAULT_PATCH_SIZE),
        downsample_ratio: (1. / ratio.powi(2)).round() as usize,
    };
    media_processor(inputs, SPECIAL_TOKENS, MessagesAction::FlattenOnlyText)
});

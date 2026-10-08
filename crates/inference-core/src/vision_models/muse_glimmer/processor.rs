use inference_models_qwen::loaders::MuseGlimmerLoader;
use inference_models_qwen::muse_glimmer::Config as MuseGlimmerConfig;
use inference_models_qwen::muse_glimmer::inputs_processor::{
    IMAGE_END, IMAGE_START, IMAGE_TOKEN, MuseGlimmerImageProcessor, VIDEO_END, VIDEO_SEPARATOR,
    VIDEO_START, VIDEO_TOKEN,
};

use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

const SPECIAL_TOKENS: &[&str] = &[
    IMAGE_TOKEN,
    IMAGE_START,
    IMAGE_END,
    VIDEO_TOKEN,
    VIDEO_START,
    VIDEO_END,
    VIDEO_SEPARATOR,
];

processor_factory!(MuseGlimmerLoader => |model_config, _, preprocessor_config, max_edge| {
    let cfg =
        MuseGlimmerConfig::from_json(model_config).expect("Failed to parse Muse-Glimmer config");
    let inputs =
        MuseGlimmerImageProcessor::new(&preprocessor_config, max_edge, cfg.gguf_collapsed_temporal)
            .expect("Failed to create Muse-Glimmer processor");
    media_processor(inputs, SPECIAL_TOKENS, MessagesAction::Keep)
});

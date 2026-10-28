use inference_models_other::lfm2_vl::inputs_processor::{
    IMAGE_END, IMAGE_START, IMAGE_THUMBNAIL, IMAGE_TOKEN, Lfm2VlImageProcessor,
};
use inference_models_other::loaders::Lfm2VlLoader;

use super::config::Config;
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor_with_placeholder;

const SPECIAL_TOKENS: &[&str] = &[IMAGE_TOKEN, IMAGE_START, IMAGE_END, IMAGE_THUMBNAIL];

processor_factory!(Lfm2VlLoader => |model_config, _, preprocessor_config, _| {
    let cfg = Config::from_json(model_config).expect("Failed to parse LFM2-VL config");
    let inputs = Lfm2VlImageProcessor::new(&cfg, &preprocessor_config);
    // one image as its chat template writes it, which the inputs processor expands to its tiles and thumbnail
    let placeholder = Some(IMAGE_TOKEN.to_string());
    media_processor_with_placeholder(inputs, SPECIAL_TOKENS, MessagesAction::Keep, placeholder)
});

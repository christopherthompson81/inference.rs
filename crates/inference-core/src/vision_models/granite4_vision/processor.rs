use inference_models_other::granite4_vision::inputs_processor::{
    Granite4VisionInputsProcessor, IMAGE_TOKEN,
};
use inference_models_other::loaders::Granite4VisionLoader;

use super::config::Config;
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor_with_placeholder;

processor_factory!(Granite4VisionLoader => |model_config, _, _, _| {
    let cfg = Config::from_json(model_config).expect("Failed to parse Granite 4 Vision config");
    // the chat template writes one `<image>` per image, which the inputs processor expands to its tokens
    media_processor_with_placeholder(
        Granite4VisionInputsProcessor::new(cfg),
        &[IMAGE_TOKEN],
        MessagesAction::Keep,
        Some(IMAGE_TOKEN.to_string()),
    )
});

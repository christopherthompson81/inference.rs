use inference_models_qwen::loaders::{
    Qwen3_5Loader, Qwen3_5MoeLoader, Qwen3VLLoader, Qwen3VLMoELoader,
};
use inference_models_qwen::qwen2vl::inputs_processor::{
    IMAGE_PAD, PLACEHOLDER, VIDEO_PAD, VISION_END, VISION_START,
};
use inference_models_qwen::qwen3_vl::inputs_processor::Qwen3VLImageProcessor;

use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor_with_placeholder;

const SPECIAL_TOKENS: &[&str] = &[IMAGE_PAD, VIDEO_PAD, PLACEHOLDER];

processor_factory!(
    Qwen3VLLoader, Qwen3VLMoELoader, Qwen3_5Loader, Qwen3_5MoeLoader => |_, _, _, max_edge| {
        let inputs = Qwen3VLImageProcessor::new(max_edge);
        // one image as its chat template writes it, which the inputs processor expands to the image's pads
        let placeholder = format!("{VISION_START}{IMAGE_PAD}{VISION_END}");
        media_processor_with_placeholder(inputs, SPECIAL_TOKENS, MessagesAction::Keep, Some(placeholder))
    }
);

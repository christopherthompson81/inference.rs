use inference_models_qwen::loaders::{Qwen2_5VLLoader, Qwen2VLLoader};
use inference_models_qwen::qwen2vl::inputs_processor::{
    IMAGE_PAD, PLACEHOLDER, Qwen2VLImageProcessor, VIDEO_PAD,
};

use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

const SPECIAL_TOKENS: &[&str] = &[IMAGE_PAD, VIDEO_PAD, PLACEHOLDER];

processor_factory!(Qwen2VLLoader, Qwen2_5VLLoader => |_, _, _, max_edge| {
    let inputs = Qwen2VLImageProcessor::new(max_edge);
    media_processor(inputs, SPECIAL_TOKENS, MessagesAction::FlattenOnlyText)
});

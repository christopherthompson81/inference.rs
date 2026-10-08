use inference_models_qwen::loaders::{
    Qwen3_5Loader, Qwen3_5MoeLoader, Qwen3VLLoader, Qwen3VLMoELoader,
};
use inference_models_qwen::qwen2vl::inputs_processor::{IMAGE_PAD, PLACEHOLDER, VIDEO_PAD};
use inference_models_qwen::qwen3_vl::inputs_processor::Qwen3VLImageProcessor;

use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

const SPECIAL_TOKENS: &[&str] = &[IMAGE_PAD, VIDEO_PAD, PLACEHOLDER];

processor_factory!(
    Qwen3VLLoader, Qwen3VLMoELoader, Qwen3_5Loader, Qwen3_5MoeLoader => |_, _, _, max_edge| {
        let inputs = Qwen3VLImageProcessor::new(max_edge);
        media_processor(inputs, SPECIAL_TOKENS, MessagesAction::Keep)
    }
);

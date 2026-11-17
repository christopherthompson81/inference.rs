use inference_models_other::deepseek_ocr::inputs_processor::{
    DeepseekOcrInputsProcessor, IMAGE_TOKEN,
};
use inference_models_other::loaders::DeepseekOcrLoader;

use super::config::Config;
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor_with_placeholder;

processor_factory!(DeepseekOcrLoader => |model_config, _, _, _| {
    let cfg = Config::from_json(model_config).expect("Failed to parse DeepSeek-OCR config");
    // the chat template writes one `<image>` per image, which the inputs processor expands to its tokens
    media_processor_with_placeholder(
        DeepseekOcrInputsProcessor::new(cfg),
        &[IMAGE_TOKEN],
        MessagesAction::Keep,
        Some(IMAGE_TOKEN.to_string()),
    )
});

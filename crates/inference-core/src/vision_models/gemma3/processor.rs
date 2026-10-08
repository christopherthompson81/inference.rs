use inference_models_gemma::gemma3::config::Gemma3Config;
use inference_models_gemma::gemma3::inputs_processor::{
    BOI_TOKEN, EOI_TOKEN, Gemma3ImageProcessor, IMAGE_TOKEN,
};
use inference_models_gemma::loaders::Gemma3Loader;

use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

const DEFAULT_IMAGE_SEQ_LEN: usize = 256;

processor_factory!(Gemma3Loader => |model_config, processor_config, _, _| {
    // Gemma 3 1b has no vision tower
    let supports_images =
        matches!(Gemma3Config::from_json(model_config).unwrap(), Gemma3Config::WithVision { .. });
    let image_seq_len = processor_config.unwrap_or_default().image_seq_len;
    let image_tokens = IMAGE_TOKEN.repeat(image_seq_len.unwrap_or(DEFAULT_IMAGE_SEQ_LEN));
    let full_image_sequence = format!("\n\n{BOI_TOKEN}{image_tokens}{EOI_TOKEN}\n\n");
    let inputs = Gemma3ImageProcessor::new(full_image_sequence, supports_images);
    media_processor(inputs, &[BOI_TOKEN, EOI_TOKEN, IMAGE_TOKEN], MessagesAction::Keep)
});

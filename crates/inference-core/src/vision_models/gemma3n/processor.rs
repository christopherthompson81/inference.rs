use inference_models_gemma::gemma3n::inputs_processor::{
    AUDIO_TOKEN, BOA_TOKEN, BOI_TOKEN, EOA_TOKEN, EOI_TOKEN, Gemma3nImageProcessor, IMAGE_TOKEN,
};
use inference_models_gemma::loaders::Gemma3nLoader;

use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

const DEFAULT_IMAGE_SEQ_LEN: usize = 256;
// transformers' default audio token count
const DEFAULT_AUDIO_SEQ_LEN: usize = 188;
const SPECIAL_TOKENS: &[&str] = &[
    IMAGE_TOKEN,
    BOI_TOKEN,
    EOI_TOKEN,
    AUDIO_TOKEN,
    BOA_TOKEN,
    EOA_TOKEN,
];

processor_factory!(Gemma3nLoader => |_, processor_config, _, _| {
    let config = processor_config.unwrap_or_default();
    let image_tokens = IMAGE_TOKEN.repeat(config.image_seq_len.unwrap_or(DEFAULT_IMAGE_SEQ_LEN));
    let full_image_sequence = format!("\n\n{BOI_TOKEN}{image_tokens}{EOI_TOKEN}\n\n");
    let audio_seq_length = config.audio_seq_length.unwrap_or(DEFAULT_AUDIO_SEQ_LEN);
    let inputs = Gemma3nImageProcessor::new(true, true, full_image_sequence, audio_seq_length);
    media_processor(inputs, SPECIAL_TOKENS, MessagesAction::Keep)
});

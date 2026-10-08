use inference_models_llama::idefics3::inputs_processor::Idefics3ImageProcessor;
use inference_models_llama::loaders::Idefics3Loader;

use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

const DEFAULT_IMAGE_SEQ_LEN: usize = 169;
const SPECIAL_TOKENS: &[&str] = &["<fake_token_around_image>", "<image>", "<end_of_utterance>"];

processor_factory!(Idefics3Loader => |_, processor_config, _, max_edge| {
    let image_seq_len = processor_config.unwrap_or_default().image_seq_len;
    let inputs =
        Idefics3ImageProcessor::new(max_edge, image_seq_len.unwrap_or(DEFAULT_IMAGE_SEQ_LEN));
    media_processor(inputs, SPECIAL_TOKENS, MessagesAction::Keep)
});

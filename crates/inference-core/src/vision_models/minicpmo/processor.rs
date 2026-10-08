use inference_models_qwen::loaders::MiniCpmOLoader;
use inference_models_qwen::minicpmo::inputs_processor::{
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_SLICE_END_TOKEN,
    DEFAULT_SLICE_START_TOKEN, DEFAULT_UNK_TOKEN, MiniCpmOImageProcessor,
};

use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

const SPECIAL_TOKENS: &[&str] = &[
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_SLICE_START_TOKEN,
    DEFAULT_SLICE_END_TOKEN,
    DEFAULT_UNK_TOKEN,
];

processor_factory!(MiniCpmOLoader => |_, _, preprocessor_config, _| {
    let inputs = MiniCpmOImageProcessor::new(preprocessor_config);
    media_processor(inputs, SPECIAL_TOKENS, MessagesAction::FlattenOnlyText)
});

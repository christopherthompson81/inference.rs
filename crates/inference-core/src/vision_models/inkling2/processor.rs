use inference_models_other::inkling2_vl::inputs_processor::Inkling2VlInputsProcessor;
use inference_models_other::loaders::{Inkling2VlLoader, MAX_IMAGE_CELLS};

use super::{Config, IMAGE_PAD_TOKEN};
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor_with_placeholder;

const SPECIAL_TOKENS: &[&str] = &[IMAGE_PAD_TOKEN];

processor_factory!(Inkling2VlLoader => |model_config, _, _, _| {
    let cfg = Config::parse(model_config).expect("Failed to parse Inkling 2 config");
    let inputs = Inkling2VlInputsProcessor::new(&cfg, MAX_IMAGE_CELLS);
    // the chat template writes one pad a picture, which the inputs processor expands to one a cell
    let placeholder = Some(IMAGE_PAD_TOKEN.to_string());
    media_processor_with_placeholder(inputs, SPECIAL_TOKENS, MessagesAction::Keep, placeholder)
});

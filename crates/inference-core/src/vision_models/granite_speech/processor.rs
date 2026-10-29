use inference_models_other::granite_speech::inputs_processor::{
    AUDIO_TOKEN, GraniteSpeechInputsProcessor,
};
use inference_models_other::loaders::GraniteSpeechLoader;

use super::config::Config;
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

processor_factory!(GraniteSpeechLoader => |model_config, _, _, _| {
    let cfg = Config::from_json(model_config).expect("Failed to parse Granite Speech config");
    // its chat template renders a message's text only
    media_processor(
        GraniteSpeechInputsProcessor::new(cfg),
        &[AUDIO_TOKEN],
        MessagesAction::FlattenOnlyText,
    )
});

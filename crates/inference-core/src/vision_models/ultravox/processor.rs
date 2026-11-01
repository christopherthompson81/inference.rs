use inference_models_llama::loaders::UltravoxLoader;
use inference_models_llama::ultravox::inputs_processor::{AUDIO_TOKEN, UltravoxInputsProcessor};

use super::UltravoxConfig;
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

processor_factory!(UltravoxLoader => |model_config, _, _, _| {
    let cfg = UltravoxConfig::from_json(model_config).expect("Failed to parse Ultravox config");
    media_processor(
        UltravoxInputsProcessor::new(cfg),
        &[AUDIO_TOKEN],
        MessagesAction::FlattenOnlyText,
    )
});

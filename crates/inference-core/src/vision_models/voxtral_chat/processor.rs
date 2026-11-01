use inference_models_llama::loaders::VoxtralChatLoader;
use inference_models_llama::voxtral_chat::inputs_processor::{
    AUDIO_TOKEN, VoxtralChatInputsProcessor,
};

use super::VoxtralChatConfig;
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

processor_factory!(VoxtralChatLoader => |model_config, _, _, _| {
    let cfg = VoxtralChatConfig::from_json(model_config).expect("Failed to parse Voxtral config");
    media_processor(
        VoxtralChatInputsProcessor::new(cfg),
        &[AUDIO_TOKEN],
        MessagesAction::FlattenOnlyText,
    )
});

use inference_models_qwen::loaders::Qwen2AudioLoader;
use inference_models_qwen::qwen2_audio::inputs_processor::{
    AUDIO_TOKEN, Qwen2AudioInputsProcessor,
};

use super::Qwen2AudioConfig;
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

processor_factory!(Qwen2AudioLoader => |model_config, _, _, _| {
    let cfg = Qwen2AudioConfig::from_json(model_config).expect("Failed to parse Qwen2-Audio config");
    media_processor(
        Qwen2AudioInputsProcessor::new(cfg),
        &[AUDIO_TOKEN],
        MessagesAction::FlattenOnlyText,
    )
});

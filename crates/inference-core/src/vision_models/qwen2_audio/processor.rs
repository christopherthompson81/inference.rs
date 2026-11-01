use inference_models_qwen::loaders::{AudioFlamingo3Loader, Qwen2AudioLoader};
use inference_models_qwen::qwen2_audio::inputs_processor::{
    AUDIO_TOKEN, Qwen2AudioInputsProcessor, SOUND_TOKEN,
};

use super::{AudioFlamingo3Config, Qwen2AudioConfig};
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

processor_factory!(Qwen2AudioLoader => |model_config, _, _, _| {
    let cfg = Qwen2AudioConfig::from_json(model_config).expect("Failed to parse Qwen2-Audio config");
    media_processor(
        Qwen2AudioInputsProcessor::new(cfg.spec(), AUDIO_TOKEN),
        &[AUDIO_TOKEN],
        MessagesAction::FlattenOnlyText,
    )
});

processor_factory!(AudioFlamingo3Loader => |model_config, _, _, _| {
    let cfg = AudioFlamingo3Config::from_json(model_config)
        .expect("Failed to parse Audio Flamingo 3 config");
    // the repo's template writes `<sound>` for each audio part
    media_processor(
        Qwen2AudioInputsProcessor::new(cfg.spec(), SOUND_TOKEN),
        &[SOUND_TOKEN],
        MessagesAction::KeepIfTemplateWrites(SOUND_TOKEN),
    )
});

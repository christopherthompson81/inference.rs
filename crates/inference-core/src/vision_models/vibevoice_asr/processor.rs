use inference_models_qwen::loaders::VibeVoiceAsrLoader;
use inference_models_qwen::vibevoice_asr::inputs_processor::{
    AUDIO_TOKEN, VibeVoiceAsrInputsProcessor,
};

use super::VibeVoiceAsrConfig;
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

processor_factory!(VibeVoiceAsrLoader => |model_config, _, _, _| {
    let cfg = VibeVoiceAsrConfig::from_json(model_config).expect("Failed to parse VibeVoice-ASR config");
    // the repo's template writes the clip's placeholder and its duration for each audio part
    media_processor(
        VibeVoiceAsrInputsProcessor::new(cfg.audio_token_id, cfg.hop_length()),
        &[AUDIO_TOKEN],
        MessagesAction::KeepIfTemplateWrites(AUDIO_TOKEN),
    )
});

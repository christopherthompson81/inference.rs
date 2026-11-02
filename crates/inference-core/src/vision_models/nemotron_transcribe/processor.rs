use inference_models_other::loaders::NemotronTranscribeLoader;
use inference_models_other::nemotron_transcribe::inputs_processor::{
    AUDIO_TOKEN, NemotronTranscribeInputsProcessor,
};

use super::Config;
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

processor_factory!(NemotronTranscribeLoader => |model_config, _, _, _| {
    let cfg = Config::from_json(model_config).expect("Failed to parse Nemotron 3.5 Transcribe config");
    media_processor(
        NemotronTranscribeInputsProcessor::new(&cfg),
        &[AUDIO_TOKEN],
        MessagesAction::KeepIfTemplateWrites(AUDIO_TOKEN),
    )
});

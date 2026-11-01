use inference_models_llama::glm_asr::inputs_processor::{AUDIO_TOKEN, GlmAsrInputsProcessor};
use inference_models_llama::loaders::GlmAsrLoader;

use super::GlmAsrConfig;
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

processor_factory!(GlmAsrLoader => |model_config, _, _, _| {
    let cfg = GlmAsrConfig::from_json(model_config).expect("Failed to parse GLM-ASR config");
    // the repo's template writes each audio part's placeholders itself
    media_processor(
        GlmAsrInputsProcessor::new(cfg),
        &[AUDIO_TOKEN],
        MessagesAction::KeepIfTemplateWrites(AUDIO_TOKEN),
    )
});

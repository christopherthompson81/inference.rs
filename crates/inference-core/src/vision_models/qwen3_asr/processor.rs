use inference_models_qwen::loaders::Qwen3AsrLoader;
use inference_models_qwen::qwen3_asr::inputs_processor::{AUDIO_TOKEN, Qwen3AsrInputsProcessor};

use super::config::Config;
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

processor_factory!(Qwen3AsrLoader => |model_config, _, _, _| {
    let cfg = Config::from_json(model_config).expect("Failed to parse Qwen3-ASR config");
    // the `-hf` repos' template writes each audio part's placeholders and drops the text; ChatML takes text only
    media_processor(
        Qwen3AsrInputsProcessor::new(cfg),
        &[AUDIO_TOKEN],
        MessagesAction::KeepIfTemplateWrites(AUDIO_TOKEN),
    )
});

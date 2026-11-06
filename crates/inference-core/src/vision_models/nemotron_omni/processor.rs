use inference_models_other::loaders::NemotronOmniLoader;
use inference_models_other::nemotron_omni::inputs_processor::{
    IMAGE_END, IMAGE_START, IMAGE_TOKEN, NemotronOmniInputsProcessor,
};

use super::Config;
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

processor_factory!(NemotronOmniLoader => |model_config, _, _, _| {
    let cfg = Config::parse(model_config).expect("Failed to parse Nemotron-H Omni config");
    media_processor(
        NemotronOmniInputsProcessor::new(&cfg),
        &[IMAGE_TOKEN, IMAGE_START, IMAGE_END],
        MessagesAction::KeepIfTemplateWrites(IMAGE_TOKEN),
    )
});

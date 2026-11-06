use inference_models_other::loaders::NemotronOmniLoader;
use inference_models_other::nemotron_omni::inputs_processor::{
    IMAGE_END, IMAGE_START, IMAGE_TOKEN, NemotronOmniInputsProcessor,
};

use super::Config;
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

processor_factory!(NemotronOmniLoader => |model_config, _, preprocessor_config, _| {
    let cfg = Config::parse(model_config).expect("Failed to parse Nemotron-H Omni config");
    let inputs = NemotronOmniInputsProcessor::new(&cfg, preprocessor_config.max_num_tiles)
        .expect("Nemotron-H Omni images need a patch budget or a tile size");
    media_processor(
        inputs,
        &[IMAGE_TOKEN, IMAGE_START, IMAGE_END],
        MessagesAction::KeepIfTemplateWrites(IMAGE_TOKEN),
    )
});

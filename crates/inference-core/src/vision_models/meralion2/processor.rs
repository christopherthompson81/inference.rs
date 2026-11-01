use inference_models_gemma::loaders::MERaLiON2Loader;
use inference_models_gemma::meralion2::inputs_processor::{AUDIO_TOKEN, MERaLiON2InputsProcessor};

use super::MERaLiON2Config;
use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

processor_factory!(MERaLiON2Loader => |model_config, _, _, _| {
    let cfg = MERaLiON2Config::from_json(model_config).expect("Failed to parse MERaLiON-2 config");
    media_processor(
        MERaLiON2InputsProcessor::new(cfg),
        &[AUDIO_TOKEN],
        MessagesAction::FlattenOnlyText,
    )
});

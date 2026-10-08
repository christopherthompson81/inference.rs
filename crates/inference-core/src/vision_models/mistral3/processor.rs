use inference_models_llama::loaders::Mistral3Loader;
use inference_models_llama::mistral3::inputs_processor::Mistral3ImageProcessor;

use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

processor_factory!(Mistral3Loader => |_, processor_config, _, _| {
    let config = processor_config.unwrap_or_default();
    let inputs = Mistral3ImageProcessor::new(
        config.image_break_token.unwrap(),
        config.image_end_token.unwrap(),
        config.image_token.unwrap(),
        config.patch_size.unwrap(),
        config.spatial_merge_size.unwrap(),
    );
    media_processor(inputs, &[], MessagesAction::Keep)
});

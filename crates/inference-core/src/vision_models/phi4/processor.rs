use inference_models_phi::loaders::Phi4MMLoader;
use inference_models_phi::phi4::inputs_processor::Phi4MMInputsProcessor;

use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

processor_factory!(Phi4MMLoader => |_, _, preprocessor_config, _| {
    let inputs = Phi4MMInputsProcessor::new(&preprocessor_config);
    media_processor(inputs, &[], MessagesAction::FlattenOnlyText)
});

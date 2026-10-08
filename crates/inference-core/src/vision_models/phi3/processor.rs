use inference_models_phi::loaders::Phi3VLoader;
use inference_models_phi::phi3_vision::inputs_processor::Phi3InputsProcessor;

use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

processor_factory!(Phi3VLoader => |_, _, _, _| {
    media_processor(Phi3InputsProcessor::default(), &[], MessagesAction::FlattenOnlyText)
});

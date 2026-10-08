use inference_models_llama::llava::config::Config as LLaVAConfig;
use inference_models_llama::llava::llava_inputs_processor::LLaVAInputProcessor;
use inference_models_llama::llava::llava_next_inputs_processor::LLaVANextInputProcessor;
use inference_models_llama::loaders::{LLaVALoader, LLaVANextLoader};

use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

fn config(model_config: &str) -> LLaVAConfig {
    LLaVAConfig::from_json(model_config).expect("Failed to parse model config.")
}

processor_factory!(LLaVALoader => |model_config, _, _, _| {
    let inputs = LLaVAInputProcessor::new(config(model_config));
    media_processor(inputs, &[], MessagesAction::FlattenOnlyText)
});

processor_factory!(LLaVANextLoader => |model_config, _, _, _| {
    let inputs = LLaVANextInputProcessor::new(config(model_config));
    media_processor(inputs, &[], MessagesAction::FlattenOnlyText)
});

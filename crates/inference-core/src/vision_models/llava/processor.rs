use std::sync::Arc;

use inference_models_llama::llava::config::Config as LLaVAConfig;
use inference_models_llama::llava::llava_inputs_processor::LLaVAInputProcessor;
use inference_models_llama::llava::llava_next_inputs_processor::LLaVANextInputProcessor;

use crate::pipeline::{InputsProcessor, MessagesAction, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;

pub struct LLaVAProcessor {
    inputs_processor: Arc<LLaVAInputProcessor>,
}

impl Processor for LLaVAProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(self.inputs_processor.clone()))
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

impl LLaVAProcessor {
    pub fn new(config: &str) -> Self {
        let model_config = LLaVAConfig::from_json(config).expect("Failed to parse model config.");
        let inputs_processor = Arc::new(LLaVAInputProcessor::new(model_config));
        Self { inputs_processor }
    }
}

pub struct LLaVANextProcessor {
    inputs_processor: Arc<LLaVANextInputProcessor>,
}

impl Processor for LLaVANextProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(self.inputs_processor.clone()))
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

impl LLaVANextProcessor {
    pub fn new(config: &str) -> Self {
        let model_config = LLaVAConfig::from_json(config).expect("Failed to parse model config.");
        let inputs_processor = Arc::new(LLaVANextInputProcessor::new(model_config));
        Self { inputs_processor }
    }
}

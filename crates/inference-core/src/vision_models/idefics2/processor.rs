use std::sync::Arc;

use indexmap::IndexMap;
use inference_models_llama::idefics2::inputs_processor::Idefics2ImageProcessor;
use inference_models_llama::loaders::Idefics2Loader;

use crate::{
    MessageContent, Pipeline, Tool,
    pipeline::{
        InputsProcessor, MessagesAction, MultimodalProcessorFactory, Processor, apply_chat_template,
    },
    request::ReasoningEffort,
    vision_models::{
        media_host::MediaInputsProcessor, preprocessor_config::PreProcessorConfig,
        processor_config::ProcessorConfig,
    },
};

pub struct Idefics2Processor {
    config: ProcessorConfig,
    preprocessor_config: PreProcessorConfig,
    fake_image_token: &'static str,
    image_token: &'static str,
    max_edge: Option<u32>,
}

impl Idefics2Processor {
    pub fn new(
        config: ProcessorConfig,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Self {
        Self {
            config,
            preprocessor_config,
            fake_image_token: "<fake_token_around_image>",
            image_token: "<image>",
            max_edge,
        }
    }
}

impl Processor for Idefics2Processor {
    fn process(
        &self,
        pipeline: &dyn Pipeline,
        messages: Vec<IndexMap<String, MessageContent>>,
        add_generation_prompt: bool,
        add_special_tokens: bool,
        enable_thinking: Option<bool>,
        reasoning_effort: Option<ReasoningEffort>,
        tools: Vec<Tool>,
    ) -> anyhow::Result<(Vec<u32>, String)> {
        let mut prompt = apply_chat_template(
            pipeline,
            messages,
            add_generation_prompt,
            enable_thinking,
            reasoning_effort,
            self.template_action(),
            tools,
        )?;

        let mut image_str = format!(
            "{}{}{}",
            self.fake_image_token,
            self.image_token.repeat(
                self.config
                    .image_seq_len
                    .expect("Idefics 2 model needs `image_seq_len`")
            ),
            self.fake_image_token
        );
        if self
            .preprocessor_config
            .do_image_splitting
            .is_some_and(|x| x)
        {
            // 4 patches + 1 original
            image_str = image_str.repeat(5);
        }

        prompt = prompt.replace(self.image_token, &image_str);
        // Deal with any adjacent images.
        prompt = prompt.replace(
            &format!("{}{}", self.fake_image_token, self.fake_image_token),
            self.fake_image_token,
        );

        let Some(tokenizer) = &pipeline.tokenizer() else {
            anyhow::bail!("Idefics2InputProcessor requires a specified tokenizer.",);
        };
        let encoding = tokenizer
            .encode_fast(prompt.clone(), add_special_tokens)
            .map_err(anyhow::Error::msg)?;
        Ok((encoding.get_ids().to_vec(), prompt))
    }

    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(Arc::new(Idefics2ImageProcessor::new(
            self.max_edge,
            self.config
                .image_seq_len
                .expect("Idefics 2 model needs `image_seq_len`"),
        ))))
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &["<fake_token_around_image>", "<image>", "<end_of_utterance>"]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl MultimodalProcessorFactory for Idefics2Loader {
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Idefics2Processor::new(
            processor_config.unwrap(),
            preprocessor_config,
            max_edge,
        ))
    }
}

//! A voice activity detection model.

use inference_api::engine::ModelSelected;

use crate::{Model, error::Result, load::LoadOptions};

/// Loads a Silero VAD GGUF; detect speech with [`Model::voice_activity`](inference_api::Engine::voice_activity).
pub struct VoiceActivityModelBuilder {
    pub(crate) model_id: String,
    pub(crate) options: LoadOptions,
}

impl VoiceActivityModelBuilder {
    /// A GGUF file, or a local directory or Hugging Face repo holding one.
    pub fn new(model_id: impl ToString) -> Self {
        Self {
            model_id: model_id.to_string(),
            options: LoadOptions::new(),
        }
    }

    load_options_methods!();

    pub(crate) fn model_selected(&self) -> ModelSelected {
        ModelSelected::VoiceActivity {
            model_id: self.model_id.clone(),
        }
    }

    pub fn into_spec(
        self,
    ) -> (
        inference_api::EngineSpec,
        inference_api::engine::EngineCallbacks,
    ) {
        let model = self.model_selected();
        self.options.spec(model)
    }

    pub async fn build(self) -> Result<Model> {
        let model = self.model_selected();
        self.options.load(model).await
    }
}

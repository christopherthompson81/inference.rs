//! A spoken language identification model.

use inference_api::engine::ModelSelected;

use crate::{Model, error::Result, load::LoadOptions};

/// Loads a SpeechBrain ECAPA-TDNN language identifier (VoxLingua107); identify with
/// [`Model::language_id`](inference_api::Engine::language_id).
pub struct LanguageIdModelBuilder {
    pub(crate) model_id: String,
    pub(crate) options: LoadOptions,
}

impl LanguageIdModelBuilder {
    /// A local directory or Hugging Face repo holding the SpeechBrain checkpoint.
    pub fn new(model_id: impl ToString) -> Self {
        Self {
            model_id: model_id.to_string(),
            options: LoadOptions::new(),
        }
    }

    load_options_methods!();

    pub(crate) fn model_selected(&self) -> ModelSelected {
        ModelSelected::LanguageId {
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

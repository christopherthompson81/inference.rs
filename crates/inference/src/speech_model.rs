//! A speech synthesis model.

use inference_api::engine::{ModelDType, ModelSelected, SpeechGenerationSpec, SpeechLoaderType};

use crate::{Model, error::Result, load::LoadOptions};

/// Loads a speech model of the given architecture.
pub struct SpeechModelBuilder {
    pub(crate) model_id: String,
    pub(crate) dac_model_id: Option<String>,
    pub(crate) loader_type: SpeechLoaderType,
    pub(crate) dtype: ModelDType,
    pub(crate) generation: Option<SpeechGenerationSpec>,
    pub(crate) options: LoadOptions,
}

impl SpeechModelBuilder {
    pub fn new(model_id: impl ToString, loader_type: SpeechLoaderType) -> Self {
        Self {
            model_id: model_id.to_string(),
            dac_model_id: None,
            loader_type,
            dtype: ModelDType::Auto,
            generation: None,
            options: LoadOptions::new(),
        }
    }

    load_options_methods!();

    /// The audio codec the model decodes through, when not the architecture's default.
    pub fn with_dac_model_id(mut self, dac_model_id: impl ToString) -> Self {
        self.dac_model_id = Some(dac_model_id.to_string());
        self
    }

    pub fn with_dtype(mut self, dtype: ModelDType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Sampling for every generation; unset fields keep the architecture's defaults.
    pub fn with_generation(mut self, generation: SpeechGenerationSpec) -> Self {
        self.generation = Some(generation);
        self
    }

    pub(crate) fn model_selected(&self) -> ModelSelected {
        ModelSelected::Speech {
            model_id: self.model_id.clone(),
            dac_model_id: self.dac_model_id.clone(),
            arch: Some(self.loader_type),
            dtype: self.dtype,
            generation: self.generation.clone(),
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

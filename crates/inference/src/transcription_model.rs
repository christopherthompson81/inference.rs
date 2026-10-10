//! A speech recognition model.

use inference_api::engine::{ModelDType, ModelSelected, TranscriptionLoaderType};

use crate::{Model, error::Result, load::LoadOptions};

/// Loads a speech recognition model; transcribe with [`Model::transcription`](inference_api::Engine::transcription).
pub struct TranscriptionModelBuilder {
    pub(crate) model_id: String,
    pub(crate) loader_type: Option<TranscriptionLoaderType>,
    pub(crate) dtype: ModelDType,
    pub(crate) options: LoadOptions,
}

impl TranscriptionModelBuilder {
    /// A model whose architecture is read from its `config.json`.
    pub fn new(model_id: impl ToString) -> Self {
        Self {
            model_id: model_id.to_string(),
            loader_type: None,
            dtype: ModelDType::Auto,
            options: LoadOptions::new(),
        }
    }

    load_options_methods!();

    pub fn with_loader_type(mut self, loader_type: TranscriptionLoaderType) -> Self {
        self.loader_type = Some(loader_type);
        self
    }

    pub fn with_dtype(mut self, dtype: ModelDType) -> Self {
        self.dtype = dtype;
        self
    }

    pub(crate) fn model_selected(&self) -> ModelSelected {
        ModelSelected::Transcription {
            model_id: self.model_id.clone(),
            arch: self.loader_type,
            dtype: self.dtype,
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

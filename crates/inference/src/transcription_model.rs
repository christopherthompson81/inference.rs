//! A speech recognition model.

use inference_api::engine::{ModelDType, ModelSelected, TranscriptionLoaderType};

use crate::{Model, error::Result, load::LoadOptions};

/// Loads a speech recognition model; transcribe with [`Model::transcription`](inference_api::Engine::transcription).
pub struct TranscriptionModelBuilder {
    pub(crate) model_id: String,
    pub(crate) loader_type: Option<TranscriptionLoaderType>,
    pub(crate) vad_model_id: Option<String>,
    pub(crate) dtype: ModelDType,
    pub(crate) options: LoadOptions,
}

impl TranscriptionModelBuilder {
    /// A model whose architecture is read from its `config.json`.
    pub fn new(model_id: impl ToString) -> Self {
        Self {
            model_id: model_id.to_string(),
            loader_type: None,
            vad_model_id: None,
            dtype: ModelDType::Auto,
            options: LoadOptions::new(),
        }
    }

    load_options_methods!();

    pub fn with_loader_type(mut self, loader_type: TranscriptionLoaderType) -> Self {
        self.loader_type = Some(loader_type);
        self
    }

    /// A Silero VAD GGUF (file, directory or HF repo) to cut long recordings at their silences.
    pub fn with_vad_model_id(mut self, vad_model_id: impl ToString) -> Self {
        self.vad_model_id = Some(vad_model_id.to_string());
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
            vad_model_id: self.vad_model_id.clone(),
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

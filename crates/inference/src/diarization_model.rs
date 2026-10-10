//! A speaker diarization model.

use inference_api::engine::{ModelDType, ModelSelected};

use crate::{Model, error::Result, load::LoadOptions};

/// Loads a speaker diarization model; diarize with [`Model::diarization`](inference_api::Engine::diarization).
pub struct DiarizationModelBuilder {
    pub(crate) model_id: String,
    pub(crate) dtype: ModelDType,
    pub(crate) options: LoadOptions,
}

impl DiarizationModelBuilder {
    /// A Hugging Face repo or local path: Nemotron-3 in the transformers layout, or a Streaming Sortformer `.nemo`.
    pub fn new(model_id: impl ToString) -> Self {
        Self {
            model_id: model_id.to_string(),
            dtype: ModelDType::Auto,
            options: LoadOptions::new(),
        }
    }

    load_options_methods!();

    pub fn with_dtype(mut self, dtype: ModelDType) -> Self {
        self.dtype = dtype;
        self
    }

    pub(crate) fn model_selected(&self) -> ModelSelected {
        ModelSelected::Diarization {
            model_id: self.model_id.clone(),
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

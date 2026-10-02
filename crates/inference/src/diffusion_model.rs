//! An image generation model.

use inference_api::engine::{DiffusionLoaderType, ModelDType, ModelSelected};

use crate::{Model, error::Result, load::LoadOptions};

/// Loads a diffusion model of the given architecture.
pub struct DiffusionModelBuilder {
    pub(crate) model_id: String,
    pub(crate) loader_type: DiffusionLoaderType,
    pub(crate) dtype: ModelDType,
    pub(crate) options: LoadOptions,
}

impl DiffusionModelBuilder {
    pub fn new(model_id: impl ToString, loader_type: DiffusionLoaderType) -> Self {
        Self {
            model_id: model_id.to_string(),
            loader_type,
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
        ModelSelected::DiffusionPlain {
            model_id: self.model_id.clone(),
            arch: self.loader_type.clone(),
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

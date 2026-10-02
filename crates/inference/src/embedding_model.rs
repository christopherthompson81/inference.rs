//! An embedding model from Hugging Face or a local directory.

use std::path::PathBuf;

use inference_api::{
    engine::{ModelDType, ModelSelected, UqffWriteConfig},
    sdk::EmbeddingLoaderType,
};

use crate::{Model, error::Result, load::LoadOptions, text_model::uqff_files};

/// Loads an embedding model; every option has the engine's default until set.
pub struct EmbeddingModelBuilder {
    pub(crate) model_id: String,
    pub(crate) loader_type: Option<EmbeddingLoaderType>,
    pub(crate) dtype: ModelDType,
    pub(crate) tokenizer_json: Option<String>,
    pub(crate) topology: Option<String>,
    pub(crate) write_uqff: Option<UqffWriteConfig>,
    pub(crate) from_uqff: Option<Vec<PathBuf>>,
    pub(crate) imatrix: Option<PathBuf>,
    pub(crate) calibration_file: Option<PathBuf>,
    pub(crate) hf_cache_path: Option<PathBuf>,
    pub(crate) options: LoadOptions,
}

impl EmbeddingModelBuilder {
    pub fn new(model_id: impl ToString) -> Self {
        Self {
            model_id: model_id.to_string(),
            loader_type: None,
            dtype: ModelDType::Auto,
            tokenizer_json: None,
            topology: None,
            write_uqff: None,
            from_uqff: None,
            imatrix: None,
            calibration_file: None,
            hf_cache_path: None,
            options: LoadOptions::new(),
        }
    }

    load_options_methods!();

    pub fn with_loader_type(mut self, loader_type: EmbeddingLoaderType) -> Self {
        self.loader_type = Some(loader_type);
        self
    }

    pub fn with_dtype(mut self, dtype: ModelDType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn with_tokenizer_json(mut self, tokenizer_json: impl ToString) -> Self {
        self.tokenizer_json = Some(tokenizer_json.to_string());
        self
    }

    pub fn with_topology_from_path(mut self, path: impl AsRef<std::path::Path>) -> Self {
        self.topology = Some(path.as_ref().to_string_lossy().into_owned());
        self
    }

    pub fn write_uqff(mut self, config: impl Into<UqffWriteConfig>) -> Self {
        self.write_uqff = Some(config.into());
        self
    }

    pub fn from_uqff(mut self, files: Vec<PathBuf>) -> Self {
        self.from_uqff = Some(files);
        self
    }

    pub fn with_imatrix(mut self, path: PathBuf) -> Self {
        self.imatrix = Some(path);
        self
    }

    pub fn with_calibration_file(mut self, path: PathBuf) -> Self {
        self.calibration_file = Some(path);
        self
    }

    pub fn from_hf_cache_path(mut self, path: PathBuf) -> Self {
        self.hf_cache_path = Some(path);
        self
    }

    pub(crate) fn model_selected(&self) -> ModelSelected {
        ModelSelected::Embedding {
            model_id: self.model_id.clone(),
            quant: None,
            tokenizer_json: self.tokenizer_json.clone(),
            arch: self.loader_type.clone(),
            dtype: self.dtype,
            topology: self.topology.clone(),
            write_uqff: self.write_uqff.clone(),
            from_uqff: uqff_files(self.from_uqff.as_deref()),
            imatrix: self.imatrix.clone(),
            calibration_file: self.calibration_file.clone(),
            hf_cache_path: self.hf_cache_path.clone(),
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

/// An embedding model loaded from UQFF shards.
pub struct UqffEmbeddingModelBuilder(EmbeddingModelBuilder);

impl UqffEmbeddingModelBuilder {
    pub fn new(model_id: impl ToString, files: Vec<PathBuf>) -> Self {
        Self(EmbeddingModelBuilder::new(model_id).from_uqff(files))
    }

    pub fn into_inner(self) -> EmbeddingModelBuilder {
        self.0
    }

    pub async fn build(self) -> Result<Model> {
        self.0.build().await
    }
}

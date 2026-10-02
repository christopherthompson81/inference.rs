//! A text model from Hugging Face or a local directory.

use std::path::PathBuf;

use inference_api::engine::{
    IsqOrganization, ModelDType, ModelSelected, NormalLoaderType, UqffWriteConfig,
};

use crate::{Model, error::Result, load::LoadOptions};

/// Loads a text model; every option has the engine's default until set.
pub struct TextModelBuilder {
    pub(crate) model_id: String,
    pub(crate) loader_type: Option<NormalLoaderType>,
    pub(crate) dtype: ModelDType,
    pub(crate) tokenizer_json: Option<String>,
    pub(crate) topology: Option<String>,
    pub(crate) organization: Option<IsqOrganization>,
    pub(crate) write_uqff: Option<UqffWriteConfig>,
    pub(crate) from_uqff: Option<Vec<PathBuf>>,
    pub(crate) imatrix: Option<PathBuf>,
    pub(crate) calibration_file: Option<PathBuf>,
    pub(crate) hf_cache_path: Option<PathBuf>,
    pub(crate) matformer_config_path: Option<PathBuf>,
    pub(crate) matformer_slice_name: Option<String>,
    pub(crate) options: LoadOptions,
}

impl TextModelBuilder {
    pub fn new(model_id: impl ToString) -> Self {
        Self {
            model_id: model_id.to_string(),
            loader_type: None,
            dtype: ModelDType::Auto,
            tokenizer_json: None,
            topology: None,
            organization: None,
            write_uqff: None,
            from_uqff: None,
            imatrix: None,
            calibration_file: None,
            hf_cache_path: None,
            matformer_config_path: None,
            matformer_slice_name: None,
            options: LoadOptions::new(),
        }
    }

    load_options_methods!();

    pub fn with_loader_type(mut self, loader_type: NormalLoaderType) -> Self {
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

    /// Per-layer ISQ and device placement, from a topology file.
    pub fn with_topology_from_path(mut self, path: impl AsRef<std::path::Path>) -> Self {
        self.topology = Some(path.as_ref().to_string_lossy().into_owned());
        self
    }

    /// Quantizes only the MoE experts.
    pub fn with_mixture_qexperts_isq(mut self) -> Self {
        self.organization = Some(IsqOrganization::MoeExpertsOnly);
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

    pub fn with_matformer_config_path(mut self, path: PathBuf) -> Self {
        self.matformer_config_path = Some(path);
        self
    }

    pub fn with_matformer_slice_name(mut self, name: String) -> Self {
        self.matformer_slice_name = Some(name);
        self
    }

    pub(crate) fn model_selected(&self) -> ModelSelected {
        ModelSelected::Plain {
            model_id: self.model_id.clone(),
            quant: None,
            tokenizer_json: self.tokenizer_json.clone(),
            arch: self.loader_type.clone(),
            dtype: self.dtype,
            topology: self.topology.clone(),
            organization: self.organization,
            write_uqff: self.write_uqff.clone(),
            from_uqff: uqff_files(self.from_uqff.as_deref()),
            imatrix: self.imatrix.clone(),
            calibration_file: self.calibration_file.clone(),
            max_seq_len: self.options.auto_map.max_seq_len,
            max_batch_size: self.options.auto_map.max_batch_size,
            hf_cache_path: self.hf_cache_path.clone(),
            matformer_config_path: self.matformer_config_path.clone(),
            matformer_slice_name: self.matformer_slice_name.clone(),
        }
    }

    /// The engine spec and callbacks this loads, for a caller that loads them itself.
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

// A UQFF checkpoint's shards, as the spec takes them.
pub(crate) fn uqff_files(files: Option<&[PathBuf]>) -> Option<String> {
    files.map(|files| {
        files
            .iter()
            .map(|file| file.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(inference_api::sdk::UQFF_MULTI_FILE_DELIMITER)
    })
}

/// A text model loaded from UQFF shards.
pub struct UqffTextModelBuilder(TextModelBuilder);

impl UqffTextModelBuilder {
    pub fn new(model_id: impl ToString, files: Vec<PathBuf>) -> Self {
        Self(TextModelBuilder::new(model_id).from_uqff(files))
    }

    pub fn into_inner(self) -> TextModelBuilder {
        self.0
    }

    pub async fn build(self) -> Result<Model> {
        self.0.build().await
    }
}

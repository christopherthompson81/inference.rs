//! A vision or audio model from Hugging Face or a local directory.

use std::path::PathBuf;

use inference_api::{
    engine::{AutoDeviceMapParams, IsqOrganization, ModelDType, ModelSelected, UqffWriteConfig},
    sdk::MultimodalLoaderType,
};

use crate::{Model, error::Result, load::LoadOptions, text_model::uqff_files};

/// Loads a multimodal model; every option has the engine's default until set.
pub struct MultimodalModelBuilder {
    pub(crate) model_id: String,
    pub(crate) loader_type: Option<MultimodalLoaderType>,
    pub(crate) dtype: ModelDType,
    pub(crate) tokenizer_json: Option<String>,
    pub(crate) topology: Option<String>,
    pub(crate) organization: IsqOrganization,
    pub(crate) write_uqff: Option<UqffWriteConfig>,
    pub(crate) from_uqff: Option<Vec<PathBuf>>,
    pub(crate) imatrix: Option<PathBuf>,
    pub(crate) calibration_file: Option<PathBuf>,
    pub(crate) max_edge: Option<u32>,
    pub(crate) hf_cache_path: Option<PathBuf>,
    pub(crate) matformer_config_path: Option<PathBuf>,
    pub(crate) matformer_slice_name: Option<String>,
    pub(crate) options: LoadOptions,
}

impl MultimodalModelBuilder {
    pub fn new(model_id: impl ToString) -> Self {
        Self {
            model_id: model_id.to_string(),
            loader_type: None,
            dtype: ModelDType::Auto,
            tokenizer_json: None,
            topology: None,
            organization: IsqOrganization::Default,
            write_uqff: None,
            from_uqff: None,
            imatrix: None,
            calibration_file: None,
            max_edge: None,
            hf_cache_path: None,
            matformer_config_path: None,
            matformer_slice_name: None,
            options: LoadOptions::new(),
        }
    }

    load_options_methods!();

    pub fn with_loader_type(mut self, loader_type: MultimodalLoaderType) -> Self {
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

    pub fn with_mixture_qexperts_isq(mut self) -> Self {
        self.organization = IsqOrganization::MoeExpertsOnly;
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

    /// Resizes images so their longer edge is at most `max_edge` pixels.
    pub fn with_max_edge(mut self, max_edge: u32) -> Self {
        self.max_edge = Some(max_edge);
        self
    }

    /// Bytes of encoder outputs kept for reuse across requests.
    pub fn with_encoder_cache_memory_bytes(mut self, bytes: usize) -> Self {
        self.options.runtime.encoder_cache_memory_bytes = Some(bytes);
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
        ModelSelected::MultimodalPlain {
            model_id: self.model_id.clone(),
            quant: None,
            tokenizer_json: self.tokenizer_json.clone(),
            arch: self.loader_type.clone(),
            dtype: self.dtype,
            topology: self.topology.clone(),
            write_uqff: self.write_uqff.clone(),
            from_uqff: uqff_files(self.from_uqff.as_deref()),
            max_edge: self.max_edge,
            calibration_file: self.calibration_file.clone(),
            imatrix: self.imatrix.clone(),
            max_seq_len: self.options.auto_map.max_seq_len,
            max_batch_size: self.options.auto_map.max_batch_size,
            max_num_images: AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES,
            max_image_length: AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
            hf_cache_path: self.hf_cache_path.clone(),
            matformer_config_path: self.matformer_config_path.clone(),
            matformer_slice_name: self.matformer_slice_name.clone(),
            organization: Some(self.organization),
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

/// A multimodal model loaded from UQFF shards.
pub struct UqffMultimodalModelBuilder(MultimodalModelBuilder);

impl UqffMultimodalModelBuilder {
    pub fn new(model_id: impl ToString, files: Vec<PathBuf>) -> Self {
        Self(MultimodalModelBuilder::new(model_id).from_uqff(files))
    }

    pub fn into_inner(self) -> MultimodalModelBuilder {
        self.0
    }

    pub async fn build(self) -> Result<Model> {
        self.0.build().await
    }
}

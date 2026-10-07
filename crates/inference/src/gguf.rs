//! A GGUF checkpoint, optionally with a multimodal projector and dynamic LoRA adapters.

use std::path::PathBuf;

use inference_api::{
    engine::{
        AutoDeviceMapParams, IsqOrganization, LoraAdapterSpec, LoraRuntimeConfig, MmprojSelection,
        ModelDType, ModelSelected, UqffWriteConfig,
    },
    sdk::GGUF_MULTI_FILE_DELIMITER,
};

use crate::{Model, error::Result, load::LoadOptions};

/// Loads a GGUF model from a repository or local directory; every option has the engine's default until set.
pub struct GgufModelBuilder {
    pub(crate) model_id: String,
    pub(crate) files: Vec<String>,
    pub(crate) mmproj_files: Option<Vec<String>>,
    pub(crate) tok_model_id: Option<String>,
    pub(crate) tokenizer_json: Option<String>,
    pub(crate) dtype: ModelDType,
    pub(crate) topology: Option<String>,
    pub(crate) organization: IsqOrganization,
    pub(crate) write_uqff: Option<UqffWriteConfig>,
    pub(crate) imatrix: Option<PathBuf>,
    pub(crate) calibration_file: Option<PathBuf>,
    pub(crate) max_edge: Option<u32>,
    pub(crate) hf_cache_path: Option<PathBuf>,
    pub(crate) matformer_config_path: Option<PathBuf>,
    pub(crate) matformer_slice_name: Option<String>,
    pub(crate) lora_adapters: Option<Vec<LoraAdapterSpec>>,
    pub(crate) lora_runtime_config: LoraRuntimeConfig,
    pub(crate) options: LoadOptions,
}

impl GgufModelBuilder {
    /// `files` are the checkpoint's shards inside `model_id`, in order.
    pub fn new(model_id: impl ToString, files: Vec<impl ToString>) -> Self {
        Self {
            model_id: model_id.to_string(),
            files: files.into_iter().map(|file| file.to_string()).collect(),
            mmproj_files: None,
            tok_model_id: None,
            tokenizer_json: None,
            dtype: ModelDType::Auto,
            topology: None,
            organization: IsqOrganization::Default,
            write_uqff: None,
            imatrix: None,
            calibration_file: None,
            max_edge: None,
            hf_cache_path: None,
            matformer_config_path: None,
            matformer_slice_name: None,
            lora_adapters: None,
            lora_runtime_config: LoraRuntimeConfig::default(),
            options: LoadOptions::new(),
        }
    }

    load_options_methods!();

    /// Turns on runtime LoRA with no adapters preloaded.
    pub fn with_lora(mut self) -> Self {
        self.lora_adapters.get_or_insert_default();
        self
    }

    pub fn with_lora_adapter(
        mut self,
        alias: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        self.lora_adapters
            .get_or_insert_default()
            .push(LoraAdapterSpec::new(alias, source));
        self
    }

    pub fn with_lora_adapter_revision(
        mut self,
        alias: impl Into<String>,
        source: impl Into<String>,
        revision: impl Into<String>,
    ) -> Self {
        self.lora_adapters
            .get_or_insert_default()
            .push(LoraAdapterSpec::new(alias, source).with_revision(revision));
        self
    }

    pub fn with_lora_adapters(
        mut self,
        adapters: impl IntoIterator<Item = LoraAdapterSpec>,
    ) -> Self {
        self.lora_adapters.get_or_insert_default().extend(adapters);
        self
    }

    /// Admission limits for runtime LoRA; turns it on.
    pub fn with_lora_runtime_config(mut self, runtime_config: LoraRuntimeConfig) -> Self {
        self.lora_adapters.get_or_insert_default();
        self.lora_runtime_config = runtime_config;
        self
    }

    /// The repository the tokenizer and chat template come from, when the GGUF's own are not wanted.
    pub fn with_tok_model_id(mut self, tok_model_id: impl ToString) -> Self {
        self.tok_model_id = Some(tok_model_id.to_string());
        self
    }

    /// The multimodal projector's shards, which make this a multimodal model.
    pub fn with_mmproj_files(mut self, files: Vec<impl ToString>) -> Self {
        self.mmproj_files = Some(files.into_iter().map(|file| file.to_string()).collect());
        self
    }

    pub fn with_tokenizer_json(mut self, tokenizer_json: impl ToString) -> Self {
        self.tokenizer_json = Some(tokenizer_json.to_string());
        self
    }

    pub fn with_dtype(mut self, dtype: ModelDType) -> Self {
        self.dtype = dtype;
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

    pub fn with_imatrix(mut self, path: PathBuf) -> Self {
        self.imatrix = Some(path);
        self
    }

    pub fn with_calibration_file(mut self, path: PathBuf) -> Self {
        self.calibration_file = Some(path);
        self
    }

    pub fn with_max_edge(mut self, max_edge: u32) -> Self {
        self.max_edge = Some(max_edge);
        self
    }

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

    pub(crate) fn quantized_filename(&self) -> String {
        self.files.join(GGUF_MULTI_FILE_DELIMITER)
    }

    pub(crate) fn model_selected(&self) -> ModelSelected {
        let multimodal = self.mmproj_files.is_some();
        ModelSelected::GGUF {
            tok_model_id: self.tok_model_id.clone(),
            quantized_model_id: self.model_id.clone(),
            quantized_filename: self.quantized_filename(),
            quant: None,
            tokenizer_json: self.tokenizer_json.clone(),
            mmproj_filename: self
                .mmproj_files
                .as_ref()
                .map(|files| files.join(GGUF_MULTI_FILE_DELIMITER)),
            mmproj_selection: MmprojSelection::Given,
            lora_adapters: self.lora_adapters.clone().unwrap_or_default(),
            lora_runtime_config: self
                .lora_adapters
                .as_ref()
                .map(|_| self.lora_runtime_config),
            dtype: self.dtype,
            topology: self.topology.clone(),
            organization: Some(self.organization),
            write_uqff: self.write_uqff.clone(),
            imatrix: self.imatrix.clone(),
            calibration_file: self.calibration_file.clone(),
            max_edge: self.max_edge,
            max_seq_len: self.options.auto_map.max_seq_len,
            max_batch_size: self.options.auto_map.max_batch_size,
            max_num_images: multimodal.then_some(AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES),
            max_image_length: multimodal.then_some(AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH),
            hf_cache_path: self.hf_cache_path.clone(),
            matformer_config_path: self.matformer_config_path.clone(),
            matformer_slice_name: self.matformer_slice_name.clone(),
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

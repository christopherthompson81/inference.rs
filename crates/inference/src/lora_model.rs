//! A text model serving LoRA adapters that requests select by alias.

use inference_api::engine::{LoraAdapterSpec, LoraRuntimeConfig, MmprojSelection, ModelSelected};

use crate::{Model, TextModelBuilder, error::Result, text_model::uqff_files};

/// Loads a text model with runtime LoRA, preloading the adapters given here.
pub struct LoraModelBuilder {
    pub(crate) text_model: TextModelBuilder,
    pub(crate) adapters: Vec<LoraAdapterSpec>,
    pub(crate) runtime_config: LoraRuntimeConfig,
}

impl LoraModelBuilder {
    pub fn from_text_model_builder(text_model: TextModelBuilder) -> Self {
        Self {
            text_model,
            adapters: Vec::new(),
            runtime_config: LoraRuntimeConfig::default(),
        }
    }

    pub fn with_adapter(mut self, alias: impl Into<String>, source: impl Into<String>) -> Self {
        self.adapters.push(LoraAdapterSpec::new(alias, source));
        self
    }

    pub fn with_adapter_revision(
        mut self,
        alias: impl Into<String>,
        source: impl Into<String>,
        revision: impl Into<String>,
    ) -> Self {
        self.adapters
            .push(LoraAdapterSpec::new(alias, source).with_revision(revision));
        self
    }

    pub fn with_adapters(mut self, adapters: impl IntoIterator<Item = LoraAdapterSpec>) -> Self {
        self.adapters.extend(adapters);
        self
    }

    /// Admission limits for adapters loaded at runtime.
    pub fn with_runtime_config(mut self, runtime_config: LoraRuntimeConfig) -> Self {
        self.runtime_config = runtime_config;
        self
    }

    pub(crate) fn model_selected(&self) -> ModelSelected {
        let base = &self.text_model;
        ModelSelected::Lora {
            model_id: base.model_id.clone(),
            quant: None,
            tokenizer_json: base.tokenizer_json.clone(),
            adapters: self.adapters.clone(),
            runtime_config: self.runtime_config,
            mmproj_selection: MmprojSelection::Given,
            arch: base.loader_type.clone(),
            dtype: base.dtype,
            topology: base.topology.clone(),
            organization: base.organization,
            write_uqff: base.write_uqff.clone(),
            from_uqff: uqff_files(base.from_uqff.as_deref()),
            imatrix: base.imatrix.clone(),
            calibration_file: base.calibration_file.clone(),
            max_edge: None,
            max_seq_len: base.options.auto_map.max_seq_len,
            max_batch_size: base.options.auto_map.max_batch_size,
            max_num_images: None,
            max_image_length: None,
            hf_cache_path: base.hf_cache_path.clone(),
            matformer_config_path: base.matformer_config_path.clone(),
            matformer_slice_name: base.matformer_slice_name.clone(),
        }
    }

    pub fn into_spec(
        self,
    ) -> (
        inference_api::EngineSpec,
        inference_api::engine::EngineCallbacks,
    ) {
        let model = self.model_selected();
        self.text_model.options.spec(model)
    }

    pub async fn build(self) -> Result<Model> {
        let model = self.model_selected();
        self.text_model.options.load(model).await
    }
}

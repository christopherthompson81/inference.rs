use inference_core::{
    AutoDeviceMapParams, LoraAdapterSpec, LoraRuntimeConfig, ModelSelected,
    UQFF_MULTI_FILE_DELIMITER,
};

use crate::{
    Model, TextModelBuilder,
    model_builder_trait::{build_model_from_pipeline, build_text_pipeline_as, join_path_list},
};

/// Wrapper of [`TextModelBuilder`] for LoRA models.
pub struct LoraModelBuilder {
    text_model: TextModelBuilder,
    adapters: Vec<LoraAdapterSpec>,
    runtime_config: LoraRuntimeConfig,
}

impl LoraModelBuilder {
    /// Create a dynamic LoRA builder from a base text model.
    pub fn from_text_model_builder(text_model: TextModelBuilder) -> Self {
        Self {
            text_model,
            adapters: Vec::new(),
            runtime_config: LoraRuntimeConfig::default(),
        }
    }

    /// Preload an adapter under a request-facing alias.
    pub fn with_adapter(mut self, alias: impl Into<String>, source: impl Into<String>) -> Self {
        self.adapters.push(LoraAdapterSpec::new(alias, source));
        self
    }

    /// Preload an adapter repository at a specific Hugging Face revision.
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

    /// Preload several typed adapter specifications.
    pub fn with_adapters(mut self, adapters: impl IntoIterator<Item = LoraAdapterSpec>) -> Self {
        self.adapters.extend(adapters);
        self
    }

    /// Set adapter residency and rank limits.
    pub fn with_runtime_config(mut self, runtime_config: LoraRuntimeConfig) -> Self {
        self.runtime_config = runtime_config;
        self
    }

    /// Build the base model and its dynamic LoRA runtime.
    pub async fn build(self) -> anyhow::Result<Model> {
        let builder = &self.text_model;
        let model_selected = ModelSelected::Lora {
            model_id: builder.model_id.clone(),
            tokenizer_json: builder.tokenizer_json.clone(),
            adapters: self.adapters,
            runtime_config: self.runtime_config,
            arch: builder.loader_type.clone(),
            dtype: builder.dtype,
            topology: builder.topology_path.clone(),
            organization: Some(builder.organization),
            write_uqff: builder.write_uqff.clone(),
            from_uqff: join_path_list(builder.from_uqff.as_deref(), UQFF_MULTI_FILE_DELIMITER),
            imatrix: builder.imatrix.clone(),
            calibration_file: builder.calibration_file.clone(),
            max_edge: None,
            max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
            max_num_images: None,
            max_image_length: None,
            hf_cache_path: builder.hf_cache_path.clone(),
            matformer_config_path: builder.matformer_config_path.clone(),
            matformer_slice_name: builder.matformer_slice_name.clone(),
        };
        let (pipeline, scheduler_config, add_model_config) =
            build_text_pipeline_as(self.text_model, model_selected, Default::default()).await?;
        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}

use inference_core::{
    AutoDeviceMapParams, LoadOverrides, ModelSelected, Ordering, UQFF_MULTI_FILE_DELIMITER,
};

use crate::{
    model_builder_trait::{build_model_from_pipeline, build_text_pipeline_as, join_path_list},
    Model, TextModelBuilder,
};

/// Wrapper of [`TextModelBuilder`] for X-LoRA models.
pub struct XLoraModelBuilder {
    text_model: TextModelBuilder,
    xlora_model_id: String,
    ordering: Ordering,
    tgt_non_granular_index: Option<usize>,
}

impl XLoraModelBuilder {
    /// Create an X-LoRA builder from a [`TextModelBuilder`], X-LoRA model ID, and ordering.
    pub fn from_text_model_builder(
        text_model: TextModelBuilder,
        xlora_model_id: impl ToString,
        ordering: Ordering,
    ) -> Self {
        Self {
            text_model,
            xlora_model_id: xlora_model_id.to_string(),
            ordering,
            tgt_non_granular_index: None,
        }
    }

    /// Set the target non-granular index for X-LoRA scaling.
    pub fn tgt_non_granular_index(mut self, tgt_non_granular_idx: usize) -> Self {
        self.tgt_non_granular_index = Some(tgt_non_granular_idx);
        self
    }

    /// Load the X-LoRA model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        let builder = &self.text_model;
        let model_selected = ModelSelected::XLora {
            model_id: Some(builder.model_id.clone()),
            tokenizer_json: builder.tokenizer_json.clone(),
            xlora_model_id: self.xlora_model_id,
            order: String::new(), // the inline ordering override is used instead
            tgt_non_granular_index: self.tgt_non_granular_index,
            arch: builder.loader_type.clone(),
            dtype: builder.dtype,
            topology: builder.topology_path.clone(),
            write_uqff: builder.write_uqff.clone(),
            from_uqff: join_path_list(builder.from_uqff.as_deref(), UQFF_MULTI_FILE_DELIMITER),
            max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
            hf_cache_path: builder.hf_cache_path.clone(),
            organization: Some(builder.organization),
        };
        let overrides = LoadOverrides {
            ordering: Some(self.ordering),
            ..Default::default()
        };
        let (pipeline, scheduler_config, add_model_config) =
            build_text_pipeline_as(self.text_model, model_selected, overrides).await?;
        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}

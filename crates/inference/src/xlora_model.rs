//! A text model with an X-LoRA classifier mixing its adapters per token.

use std::path::Path;

use inference_api::engine::ModelSelected;

use crate::{Model, TextModelBuilder, error::Result, text_model::uqff_files};

/// Loads a text model with the X-LoRA adapters and classifier of `xlora_model_id`.
pub struct XLoraModelBuilder {
    pub(crate) text_model: TextModelBuilder,
    pub(crate) xlora_model_id: String,
    pub(crate) order: String,
    pub(crate) tgt_non_granular_index: Option<usize>,
}

impl XLoraModelBuilder {
    /// `order` is the ordering file naming the adapters and the layers they apply to.
    pub fn from_text_model_builder(
        text_model: TextModelBuilder,
        xlora_model_id: impl ToString,
        order: impl AsRef<Path>,
    ) -> Self {
        Self {
            text_model,
            xlora_model_id: xlora_model_id.to_string(),
            order: order.as_ref().to_string_lossy().into_owned(),
            tgt_non_granular_index: None,
        }
    }

    /// Runs the classifier only until this token index, then reuses its scalings.
    pub fn tgt_non_granular_index(mut self, tgt_non_granular_idx: usize) -> Self {
        self.tgt_non_granular_index = Some(tgt_non_granular_idx);
        self
    }

    pub(crate) fn model_selected(&self) -> ModelSelected {
        let base = &self.text_model;
        ModelSelected::XLora {
            model_id: Some(base.model_id.clone()),
            quant: None,
            tokenizer_json: base.tokenizer_json.clone(),
            xlora_model_id: self.xlora_model_id.clone(),
            order: self.order.clone(),
            tgt_non_granular_index: self.tgt_non_granular_index,
            arch: base.loader_type.clone(),
            dtype: base.dtype,
            topology: base.topology.clone(),
            write_uqff: base.write_uqff.clone(),
            from_uqff: uqff_files(base.from_uqff.as_deref()),
            max_seq_len: base.options.auto_map.max_seq_len,
            max_batch_size: base.options.auto_map.max_batch_size,
            hf_cache_path: base.hf_cache_path.clone(),
            organization: base.organization,
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

//! A GGUF model with an X-LoRA classifier mixing its adapters per token.

use std::path::Path;

use inference_api::engine::ModelSelected;

use crate::{GgufModelBuilder, Model, error::Result};

const BUILDER: &str = "GgufXLoraModelBuilder";

/// Loads a GGUF model with the X-LoRA adapters and classifier of `xlora_model_id`.
pub struct GgufXLoraModelBuilder {
    pub(crate) gguf_model: GgufModelBuilder,
    pub(crate) xlora_model_id: String,
    pub(crate) order: String,
    pub(crate) tgt_non_granular_index: Option<usize>,
}

impl GgufXLoraModelBuilder {
    /// `order` is the ordering file naming the adapters and the layers they apply to.
    pub fn from_gguf_model_builder(
        gguf_model: GgufModelBuilder,
        xlora_model_id: impl ToString,
        order: impl AsRef<Path>,
    ) -> Self {
        Self {
            gguf_model,
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

    pub(crate) fn checked_model_selected(&self) -> Result<ModelSelected> {
        self.gguf_model.check_static_adapters(BUILDER)?;
        Ok(self.model_selected())
    }

    fn model_selected(&self) -> ModelSelected {
        let base = &self.gguf_model;
        ModelSelected::XLoraGGUF {
            tok_model_id: base.tok_model_id.clone(),
            quantized_model_id: base.model_id.clone(),
            quantized_filename: base.quantized_filename(),
            quant: None,
            xlora_model_id: self.xlora_model_id.clone(),
            order: self.order.clone(),
            tgt_non_granular_index: self.tgt_non_granular_index,
            dtype: base.dtype,
            topology: base.topology.clone(),
            max_seq_len: base.options.auto_map.max_seq_len,
            max_batch_size: base.options.auto_map.max_batch_size,
            tokenizer_json: base.tokenizer_json.clone(),
            organization: Some(base.organization),
            write_uqff: base.write_uqff.clone(),
            imatrix: base.imatrix.clone(),
            calibration_file: base.calibration_file.clone(),
            hf_cache_path: base.hf_cache_path.clone(),
            matformer_config_path: base.matformer_config_path.clone(),
            matformer_slice_name: base.matformer_slice_name.clone(),
        }
    }

    pub fn into_spec(
        self,
    ) -> Result<(
        inference_api::EngineSpec,
        inference_api::engine::EngineCallbacks,
    )> {
        let model = self.checked_model_selected()?;
        Ok(self.gguf_model.options.spec(model))
    }

    pub async fn build(self) -> Result<Model> {
        let model = self.checked_model_selected()?;
        self.gguf_model.options.load(model).await
    }
}

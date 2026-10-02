//! A GGUF model with static LoRA adapters merged in the order an ordering file gives.

use std::path::Path;

use inference_api::engine::ModelSelected;

use crate::{GgufModelBuilder, Model, error::Result};

const BUILDER: &str = "GgufLoraModelBuilder";

/// Loads a GGUF model with the adapters of `lora_model_id`.
pub struct GgufLoraModelBuilder {
    pub(crate) gguf_model: GgufModelBuilder,
    pub(crate) lora_model_id: String,
    pub(crate) order: String,
}

impl GgufLoraModelBuilder {
    /// `order` is the ordering file naming the adapters and the layers they apply to.
    pub fn from_gguf_model_builder(
        gguf_model: GgufModelBuilder,
        lora_model_id: impl ToString,
        order: impl AsRef<Path>,
    ) -> Self {
        Self {
            gguf_model,
            lora_model_id: lora_model_id.to_string(),
            order: order.as_ref().to_string_lossy().into_owned(),
        }
    }

    pub(crate) fn checked_model_selected(&self) -> Result<ModelSelected> {
        self.gguf_model.check_static_adapters(BUILDER)?;
        Ok(self.model_selected())
    }

    fn model_selected(&self) -> ModelSelected {
        let base = &self.gguf_model;
        ModelSelected::LoraGGUF {
            tok_model_id: base.tok_model_id.clone(),
            quantized_model_id: base.model_id.clone(),
            quantized_filename: base.quantized_filename(),
            quant: None,
            adapters_model_id: self.lora_model_id.clone(),
            order: self.order.clone(),
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

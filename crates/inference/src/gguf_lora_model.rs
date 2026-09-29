use inference_core::{GGUF_MULTI_FILE_DELIMITER, LoadOverrides, ModelSelected, Ordering};

use crate::{
    GgufModelBuilder, Model,
    model_builder_trait::{GgufAutoMapDims, build_gguf_pipeline_as, build_model_from_pipeline},
};

/// Wrapper of [`GgufModelBuilder`] for LoRA models.
pub struct GgufLoraModelBuilder {
    gguf_model: GgufModelBuilder,
    lora_model_id: String,
    ordering: Ordering,
}

impl GgufLoraModelBuilder {
    /// Create a GGUF LoRA builder from a [`GgufModelBuilder`], LoRA model ID, and ordering.
    pub fn from_gguf_model_builder(
        gguf_model: GgufModelBuilder,
        lora_model_id: impl ToString,
        ordering: Ordering,
    ) -> Self {
        Self {
            gguf_model,
            lora_model_id: lora_model_id.to_string(),
            ordering,
        }
    }

    /// Load the GGUF LoRA model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        if self.gguf_model.lora_adapters.is_some() {
            anyhow::bail!(
                "`GgufLoraModelBuilder` cannot combine legacy static LoRA with dynamic LoRA; use \
                 `GgufModelBuilder` directly for dynamic adapters"
            );
        }
        if self.gguf_model.mmproj_files.is_some() {
            anyhow::bail!(
                "`GgufLoraModelBuilder` provides legacy static LoRA, which is not supported for \
                 multimodal GGUF; use `GgufModelBuilder::with_lora_adapter`"
            );
        }
        let lora_model_id = self.lora_model_id;
        let select = |builder: &GgufModelBuilder, dims: GgufAutoMapDims| ModelSelected::LoraGGUF {
            tok_model_id: builder.tok_model_id.clone(),
            quantized_model_id: builder.model_id.clone(),
            quantized_filename: builder.files.join(GGUF_MULTI_FILE_DELIMITER),
            adapters_model_id: lora_model_id.clone(),
            order: String::new(), // the inline ordering override is used instead
            dtype: builder.dtype,
            topology: builder.topology_path.clone(),
            max_seq_len: dims.max_seq_len,
            max_batch_size: dims.max_batch_size,
            tokenizer_json: builder.tokenizer_json.clone(),
            organization: Some(builder.organization),
            write_uqff: builder.write_uqff.clone(),
            imatrix: builder.imatrix.clone(),
            calibration_file: builder.calibration_file.clone(),
            hf_cache_path: builder.hf_cache_path.clone(),
            matformer_config_path: builder.matformer_config_path.clone(),
            matformer_slice_name: builder.matformer_slice_name.clone(),
        };
        let overrides = LoadOverrides {
            ordering: Some(self.ordering),
            ..Default::default()
        };
        let (pipeline, scheduler_config, add_model_config) =
            build_gguf_pipeline_as(self.gguf_model, select, overrides).await?;
        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn legacy_lora_rejects_a_dynamic_gguf_builder() {
        let builder = GgufModelBuilder::new("repo", vec!["model.gguf"]).with_lora();
        let ordering = Ordering {
            adapters: None,
            layers: None,
            base_model_id: "repo".to_string(),
            preload_adapters: None,
        };
        let error = GgufLoraModelBuilder::from_gguf_model_builder(builder, "legacy", ordering)
            .build()
            .await
            .err()
            .expect("mixed adapter modes should fail");

        assert!(error.to_string().contains("cannot combine"));
    }
}

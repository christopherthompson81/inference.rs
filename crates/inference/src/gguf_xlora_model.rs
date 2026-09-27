use inference_core::{LoadOverrides, ModelSelected, Ordering, GGUF_MULTI_FILE_DELIMITER};

use crate::{
    model_builder_trait::{build_gguf_pipeline_as, build_model_from_pipeline, GgufAutoMapDims},
    GgufModelBuilder, Model,
};

/// Wrapper of [`GgufModelBuilder`] for X-LoRA models.
pub struct GgufXLoraModelBuilder {
    gguf_model: GgufModelBuilder,
    xlora_model_id: String,
    ordering: Ordering,
    tgt_non_granular_index: Option<usize>,
}

impl GgufXLoraModelBuilder {
    /// Create a GGUF X-LoRA builder from a [`GgufModelBuilder`], X-LoRA model ID, and ordering.
    pub fn from_gguf_model_builder(
        gguf_model: GgufModelBuilder,
        xlora_model_id: impl ToString,
        ordering: Ordering,
    ) -> Self {
        Self {
            gguf_model,
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

    /// Load the GGUF X-LoRA model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        if self.gguf_model.lora_adapters.is_some() {
            anyhow::bail!(
                "`GgufXLoraModelBuilder` cannot combine X-LoRA with dynamic LoRA; use \
                 `GgufModelBuilder` directly for dynamic adapters"
            );
        }
        if self.gguf_model.mmproj_files.is_some() {
            anyhow::bail!("Multimodal GGUF does not currently support X-LoRA adapters");
        }
        let xlora_model_id = self.xlora_model_id;
        let tgt_non_granular_index = self.tgt_non_granular_index;
        let select = |builder: &GgufModelBuilder, dims: GgufAutoMapDims| ModelSelected::XLoraGGUF {
            tok_model_id: builder.tok_model_id.clone(),
            quantized_model_id: builder.model_id.clone(),
            quantized_filename: builder.files.join(GGUF_MULTI_FILE_DELIMITER),
            xlora_model_id: xlora_model_id.clone(),
            order: String::new(), // the inline ordering override is used instead
            tgt_non_granular_index,
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
    async fn xlora_rejects_a_dynamic_gguf_builder() {
        let builder = GgufModelBuilder::new("repo", vec!["model.gguf"]).with_lora();
        let ordering = Ordering {
            adapters: None,
            layers: None,
            base_model_id: "repo".to_string(),
            preload_adapters: None,
        };
        let error = GgufXLoraModelBuilder::from_gguf_model_builder(builder, "xlora", ordering)
            .build()
            .await
            .err()
            .expect("mixed adapter modes should fail");

        assert!(error.to_string().contains("cannot combine"));
    }
}
